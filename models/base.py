import os
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QuantilePredictor(ABC):
    def __init__(self, quantiles: List[float] = None, name: str = "base"):
        from config import QUANTILES
        self.quantiles = quantiles or QUANTILES
        self.name = name
        self.fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "QuantilePredictor":
        pass

    @abstractmethod
    def _predict_quantile(self, X: pd.DataFrame, quantile: float) -> np.ndarray:
        pass

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        from config import QUANTILE_LABELS
        results = {}
        for q, label in zip(self.quantiles, QUANTILE_LABELS):
            results[label] = self._predict_quantile(X, q)
        self._enforce_monotonicity(results)
        return results

    def _enforce_monotonicity(self, results: Dict[str, np.ndarray]):
        if "P10" in results and "P50" in results and "P90" in results:
            p10, p50, p90 = results["P10"], results["P50"], results["P90"]
            violations = np.sum(p10 > p50) + np.sum(p50 > p90)
            if violations > 0:
                logger.warning(f"  [{self.name}] {violations} quantile violations, enforcing monotonicity")
                results["P50"] = np.maximum(p10, np.minimum(p90, p50))
                results["P10"] = np.minimum(results["P10"], results["P50"])
                results["P90"] = np.maximum(results["P90"], results["P50"])


class GBDTQuantileModel(QuantilePredictor):
    def __init__(self, quantiles=None, name="gbdt", params=None):
        super().__init__(quantiles=quantiles, name=name)
        from config import LIGHTGBM_PARAMS
        self.params = params or LIGHTGBM_PARAMS.copy()
        self.models = {}
        self._col_rename = {}

    @staticmethod
    def _sanitize_columns(df: pd.DataFrame) -> tuple:
        rename_map = {}
        for c in df.columns:
            safe = "".join(ch if ch.isalnum() or ch == "_" else f"U{ord(ch):04X}" for ch in str(c))
            if safe != c:
                rename_map[c] = safe
        if rename_map:
            df = df.rename(columns=rename_map)
        return df, rename_map

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight=None, eval_set=None, **kwargs) -> "GBDTQuantileModel":
        import lightgbm as lgb
        X, self._col_rename = self._sanitize_columns(X)

        for q in self.quantiles:
            params = self.params.copy()
            params["objective"] = "quantile"
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)
            fit_kwargs = {"X": X, "y": y}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            if eval_set is not None:
                X_val, y_val = eval_set
                X_val, _ = self._sanitize_columns(X_val)
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            model.fit(**fit_kwargs)
            self.models[q] = model
            logger.info(f"    [{self.name}] Fitted q={q:.1f}")
        self.fitted = True
        return self

    def _predict_quantile(self, X: pd.DataFrame, quantile: float) -> np.ndarray:
        if quantile not in self.models:
            raise ValueError(f"Quantile {quantile} not fitted")
        X, _ = self._sanitize_columns(X)
        return self.models[quantile].predict(X)

    def save(self, path: str, feature_cols: list = None):
        import pickle
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "quantiles": self.quantiles, "params": self.params,
                         "feature_cols": feature_cols}, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.quantiles = data["quantiles"]
        self.params = data["params"]
        self.feature_cols = data.get("feature_cols")
        self.fitted = True
        return self


class DeepQuantileModel(QuantilePredictor):
    def __init__(self, quantiles=None, name="deep", input_dim=1,
                 hidden_size=128, num_heads=4, dropout=0.1,
                 encoder_steps=672, decoder_steps=96):
        super().__init__(quantiles=quantiles, name=name)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.model = None
        self.device = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        class TFTSingleTarget(nn.Module):
            def __init__(self, input_dim, hidden_size, num_heads, dropout, num_quantiles, enc_steps, dec_steps):
                super().__init__()
                self.enc_steps = enc_steps
                self.dec_steps = dec_steps
                self.input_proj = nn.Linear(input_dim, hidden_size)
                pe = torch.zeros(enc_steps + dec_steps, hidden_size)
                pos = torch.arange(0, enc_steps + dec_steps, dtype=torch.float).unsqueeze(1)
                div = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer("pe", pe.unsqueeze(0))
                enc_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                                       dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
                dec_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads,
                                                       dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
                self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
                self.output_proj = nn.Linear(hidden_size, num_quantiles)

            def forward(self, x):
                h = self.input_proj(x)
                seq_len = h.size(1)
                h = h + self.pe[:, :seq_len, :]
                memory = self.encoder(h)
                dec_input = memory[:, -self.dec_steps:, :]
                dec_len = dec_input.size(1)
                mask = nn.Transformer.generate_square_subsequent_mask(dec_len, device=x.device)
                out = self.decoder(dec_input, memory, tgt_mask=mask)
                return self.output_proj(out)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = TFTSingleTarget(
            self.input_dim, self.hidden_size, self.num_heads, self.dropout,
            len(self.quantiles), self.encoder_steps, self.decoder_steps,
        ).to(device)
        logger.info(f"    [{self.name}] TFT built: {sum(p.numel() for p in self.model.parameters())} params, device={device}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set=None, learning_rate=1e-3, batch_size=32,
            max_epochs=100, patience=10, sample_weight=None, **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.model is None:
            self._build_model()

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = self._pinball_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            avg_loss = epoch_loss / len(loader)
            val_loss = avg_loss
            if eval_set is not None:
                val_loss = self._evaluate(eval_set[0], eval_set[1], batch_size)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"    [{self.name}] Epoch {epoch+1}/{max_epochs}, loss={avg_loss:.6f}, best={best_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"    [{self.name}] Early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.fitted = True
        return self

    def _pinball_loss(self, pred, target):
        import torch
        total = torch.tensor(0.0, device=pred.device)
        for i, q in enumerate(self.quantiles):
            error = target - pred[..., i]
            total = total + torch.max(q * error, (q - 1) * error).mean()
        return total / len(self.quantiles)

    def _evaluate(self, X_val, y_val, batch_size):
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_val).to(self.device)
            y_t = torch.FloatTensor(y_val).to(self.device)
            pred = self.model(X_t)
            loss = self._pinball_loss(pred, y_t)
        return loss.item()

    def _predict_quantile(self, X: np.ndarray, quantile: float) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            pred = self.model(X_t).cpu().numpy()
        return pred[..., self.quantiles.index(quantile)]

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        import torch
        from config import QUANTILE_LABELS
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            pred = self.model(X_t).cpu().numpy()
        results = {label: pred[..., i] for i, label in enumerate(QUANTILE_LABELS)}
        self._enforce_monotonicity(results)
        return results

    def save(self, path: str):
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state": self.model.state_dict() if self.model else None,
                     "config": {"input_dim": self.input_dim, "hidden_size": self.hidden_size,
                                "num_heads": self.num_heads, "dropout": self.dropout,
                                "encoder_steps": self.encoder_steps, "decoder_steps": self.decoder_steps,
                                "quantiles": self.quantiles}}, path)

    def load(self, path: str):
        import torch
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        self.input_dim = cfg["input_dim"]; self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_heads"]; self.dropout = cfg["dropout"]
        self.encoder_steps = cfg["encoder_steps"]; self.decoder_steps = cfg["decoder_steps"]
        self.quantiles = cfg["quantiles"]
        self._build_model()
        if ckpt["model_state"] is not None:
            self.model.load_state_dict(ckpt["model_state"])
        self.fitted = True
        return self

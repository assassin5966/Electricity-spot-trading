import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FeatureMeta:
    name: str
    source_column: str
    category: str
    sub_category: str
    description: str
    is_derived: bool = False
    derivation_formula: str = ""
    is_retained: bool = True
    delete_reason: str = ""


GFS_VARIABLE_CATEGORIES = {
    "wind": {
        "prefixes": ["UGRD", "VGRD"],
        "description": "风场类: U/V分量风速",
    },
    "wind_derived": {
        "prefixes": ["WS", "WD", "GUST"],
        "description": "风场派生: 风速/风向/阵风",
    },
    "temperature": {
        "prefixes": ["TMP", "TMAX", "TMIN", "TSOIL", "DPT"],
        "description": "温度类: 气温/露点/土壤温度",
    },
    "humidity": {
        "prefixes": ["RH", "SPFH"],
        "description": "湿度类: 相对湿度/比湿",
    },
    "radiation": {
        "prefixes": ["DSWRF", "DLWRF", "SUNSD", "USWRF", "ULWRF"],
        "description": "辐射类: 短波/长波辐射/日照",
    },
    "cloud": {
        "prefixes": ["TCDC", "LCDC", "MCDC", "HCDC"],
        "description": "云量类: 总/低/中/高云量",
    },
    "precipitation": {
        "prefixes": ["PRATE", "APCP", "CRAIN", "CSNOW", "CFRZR", "CICEP"],
        "description": "降水类: 降水率/累积/降水类型",
    },
    "convective": {
        "prefixes": ["CAPE", "CIN", "HPBL", "PWAT"],
        "description": "对流类: 对流有效位能/对流抑制/边界层高度/可降水量",
    },
    "pressure": {
        "prefixes": ["PRMSL", "PRES"],
        "description": "气压类: 海平面气压/地面气压",
    },
    "flux": {
        "prefixes": ["LHTFL", "SHTFL", "GFLUX"],
        "description": "通量类: 潜热/感热/地面热通量",
    },
    "surface": {
        "prefixes": ["SOILW", "ALBDO", "WEASD", "SNOD", "VEG"],
        "description": "地表类: 土壤湿度/反照率/雪当量/雪深/植被",
    },
    "ozone": {
        "prefixes": ["O3MR"],
        "description": "臭氧类: 臭氧混合比(强制删除)",
    },
    "microphysics": {
        "prefixes": ["CLWMR", "ICMR", "RWMR", "SNMR", "GRLE"],
        "description": "微物理类: 云水/云冰/雨/雪/霰(强制删除)",
    },
}

FORCE_DELETE_PREFIXES = {"O3MR"}
FORCE_DELETE_MICROPHYSICS = {"CLWMR", "ICMR", "RWMR", "SNMR", "GRLE"}
HIGH_ALTITUDE_LEVELS = {"10mb", "20mb", "30mb", "50mb"}

WIND_CORE_FEATURES = {
    "UGRD_588": ("wind", "surface", "10m U分量风速"),
    "VGRD_589": ("wind", "surface", "10m V分量风速"),
    "UGRD_689": ("wind", "surface", "100m U分量风速"),
    "VGRD_690": ("wind", "surface", "100m V分量风速"),
    "GUST_14": ("wind_derived", "surface", "地面阵风风速"),
}

SOLAR_CORE_FEATURES = {
    "DSWRF_653": ("radiation", "surface", "地面短波辐射(核心)"),
    "DLWRF_654": ("radiation", "surface", "地面长波辐射"),
    "SUNSD_622": ("radiation", "surface", "日照时长"),
    "TCDC_636": ("cloud", "surface", "总云量"),
    "LCDC_630": ("cloud", "low", "低云量"),
    "MCDC_632": ("cloud", "mid", "中云量"),
    "HCDC_634": ("cloud", "high", "高云量"),
    "TMP_581": ("temperature", "surface", "2m气温"),
    "RH_584": ("humidity", "surface", "2m相对湿度"),
}

GENERAL_METEO_FEATURES = {
    "PRMSL_1": ("pressure", "surface", "海平面气压"),
    "PRES_639": ("pressure", "surface", "地面气压"),
    "HPBL_712": ("convective", "surface", "边界层高度"),
    "CAPE_624": ("convective", "surface", "对流有效位能"),
    "CIN_625": ("convective", "surface", "对流抑制"),
    "PWAT_626": ("convective", "surface", "可降水量"),
}

SURFACE_AUX_FEATURES = {
    "SOILW_565": ("surface", "soil", "0-10cm土壤湿度"),
    "TSOIL_564": ("temperature", "soil", "土壤温度"),
    "ALBDO_730": ("surface", "albedo", "地面反照率"),
    "WEASD_577": ("surface", "snow", "雪水当量"),
}

ISOBARIC_KEEP_FEATURES = {
    "TMP_196": ("temperature", "1000mb", "1000mb温度"),
    "RH_197": ("humidity", "1000mb", "1000mb相对湿度"),
    "UGRD_202": ("wind", "1000mb", "1000mb U分量"),
    "VGRD_203": ("wind", "1000mb", "1000mb V分量"),
    "TMP_286": ("temperature", "925mb", "925mb温度"),
    "RH_287": ("humidity", "925mb", "925mb相对湿度"),
    "UGRD_292": ("wind", "925mb", "925mb U分量"),
    "VGRD_293": ("wind", "925mb", "925mb V分量"),
    "TMP_302": ("temperature", "850mb", "850mb温度"),
    "RH_303": ("humidity", "850mb", "850mb相对湿度"),
    "TCDC_304": ("cloud", "850mb", "850mb云量"),
    "UGRD_308": ("wind", "850mb", "850mb U分量"),
    "VGRD_309": ("wind", "850mb", "850mb V分量"),
    "TMP_350": ("temperature", "700mb", "700mb温度"),
    "RH_351": ("humidity", "700mb", "700mb相对湿度"),
    "TCDC_352": ("cloud", "700mb", "700mb云量"),
    "UGRD_356": ("wind", "700mb", "700mb U分量"),
    "VGRD_357": ("wind", "700mb", "700mb V分量"),
    "TMP_446": ("temperature", "500mb", "500mb温度"),
    "RH_447": ("humidity", "500mb", "500mb相对湿度"),
    "UGRD_452": ("wind", "500mb", "500mb U分量"),
    "VGRD_453": ("wind", "500mb", "500mb V分量"),
    "TMP_494": ("temperature", "300mb", "300mb温度"),
    "UGRD_500": ("wind", "300mb", "300mb U分量"),
    "VGRD_501": ("wind", "300mb", "300mb V分量"),
}

DERIVED_FEATURES_SPEC = {
    "ws_10m": {
        "formula": "sqrt(UGRD_588^2 + VGRD_589^2)",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "10m风速(派生)",
        "source_columns": ["UGRD_588", "VGRD_589"],
    },
    "ws_100m": {
        "formula": "sqrt(UGRD_689^2 + VGRD_690^2)",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "100m风速(派生)",
        "source_columns": ["UGRD_689", "VGRD_690"],
    },
    "wd_10m": {
        "formula": "arctan2(VGRD_589, UGRD_588) * 180/pi mod 360",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "10m风向(派生)",
        "source_columns": ["UGRD_588", "VGRD_589"],
    },
    "wd_100m": {
        "formula": "arctan2(VGRD_690, UGRD_689) * 180/pi mod 360",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "100m风向(派生)",
        "source_columns": ["UGRD_689", "VGRD_690"],
    },
    "wind_shear_100m_10m": {
        "formula": "ws_100m - ws_10m",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "风切变100m-10m(派生)",
        "source_columns": ["ws_100m", "ws_10m"],
    },
    "wind_shear_850mb_surface": {
        "formula": "ws_850mb - ws_10m",
        "category": "wind_derived",
        "sub_category": "vertical",
        "description": "风切变850mb-地面(派生)",
        "source_columns": ["UGRD_308", "VGRD_309", "UGRD_588", "VGRD_589"],
    },
    "ws_10m_sq": {
        "formula": "ws_10m^2",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "10m风速平方(派生)",
        "source_columns": ["ws_10m"],
    },
    "ws_10m_cu": {
        "formula": "ws_10m^3",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "10m风速立方(派生)",
        "source_columns": ["ws_10m"],
    },
    "ws_100m_sq": {
        "formula": "ws_100m^2",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "100m风速平方(派生)",
        "source_columns": ["ws_100m"],
    },
    "ws_100m_cu": {
        "formula": "ws_100m^3",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "100m风速立方(派生)",
        "source_columns": ["ws_100m"],
    },
    "temp_dew_spread": {
        "formula": "TMP_581 - DPT_583",
        "category": "temperature",
        "sub_category": "surface",
        "description": "温度露点差(派生)",
        "source_columns": ["TMP_581", "DPT_583"],
    },
    "temp_diurnal_range": {
        "formula": "TMAX_586 - TMIN_587",
        "category": "temperature",
        "sub_category": "surface",
        "description": "日温差(派生)",
        "source_columns": ["TMAX_586", "TMIN_587"],
    },
    "cloud_total_weighted": {
        "formula": "0.3*LCDC_630 + 0.4*MCDC_632 + 0.3*HCDC_634",
        "category": "cloud",
        "sub_category": "surface",
        "description": "加权总云量(派生)",
        "source_columns": ["LCDC_630", "MCDC_632", "HCDC_634"],
    },
    "clear_sky_index": {
        "formula": "1 - TCDC_636/100",
        "category": "cloud",
        "sub_category": "surface",
        "description": "晴空指数(派生)",
        "source_columns": ["TCDC_636"],
    },
    "dswrf_sq": {
        "formula": "DSWRF_653^2",
        "category": "radiation",
        "sub_category": "surface",
        "description": "短波辐射平方(派生)",
        "source_columns": ["DSWRF_653"],
    },
    "cape_surface": {
        "formula": "CAPE_624 (rename)",
        "category": "convective",
        "sub_category": "surface",
        "description": "对流有效位能(重命名)",
        "source_columns": ["CAPE_624"],
    },
    "cin_surface": {
        "formula": "CIN_625 (rename)",
        "category": "convective",
        "sub_category": "surface",
        "description": "对流抑制(重命名)",
        "source_columns": ["CIN_625"],
    },
    "pbl_height": {
        "formula": "HPBL_712 (rename)",
        "category": "convective",
        "sub_category": "surface",
        "description": "边界层高度(重命名)",
        "source_columns": ["HPBL_712"],
    },
    "precip_rate": {
        "formula": "PRATE_593 (rename)",
        "category": "precipitation",
        "sub_category": "surface",
        "description": "降水率(重命名)",
        "source_columns": ["PRATE_593"],
    },
    "precip_accum": {
        "formula": "APCP_596 (rename)",
        "category": "precipitation",
        "sub_category": "surface",
        "description": "累积降水(重命名)",
        "source_columns": ["APCP_596"],
    },
    "wd_10m_change": {
        "formula": "diff(wd_10m) wrapped to [-180, 180]",
        "category": "wind_derived",
        "sub_category": "surface",
        "description": "10m风向变化率(派生)",
        "source_columns": ["wd_10m"],
    },
    "ws_850mb": {
        "formula": "sqrt(UGRD_308^2 + VGRD_309^2)",
        "category": "wind_derived",
        "sub_category": "850mb",
        "description": "850mb风速(派生)",
        "source_columns": ["UGRD_308", "VGRD_309"],
    },
    "ws_700mb": {
        "formula": "sqrt(UGRD_356^2 + VGRD_357^2)",
        "category": "wind_derived",
        "sub_category": "700mb",
        "description": "700mb风速(派生)",
        "source_columns": ["UGRD_356", "VGRD_357"],
    },
    "ws_500mb": {
        "formula": "sqrt(UGRD_452^2 + VGRD_453^2)",
        "category": "wind_derived",
        "sub_category": "500mb",
        "description": "500mb风速(派生)",
        "source_columns": ["UGRD_452", "VGRD_453"],
    },
    "ws_1000mb": {
        "formula": "sqrt(UGRD_202^2 + VGRD_203^2)",
        "category": "wind_derived",
        "sub_category": "1000mb",
        "description": "1000mb风速(派生)",
        "source_columns": ["UGRD_202", "VGRD_203"],
    },
    "ws_925mb": {
        "formula": "sqrt(UGRD_292^2 + VGRD_293^2)",
        "category": "wind_derived",
        "sub_category": "925mb",
        "description": "925mb风速(派生)",
        "source_columns": ["UGRD_292", "VGRD_293"],
    },
    "tmp_1000mb": {
        "formula": "TMP_196 (rename)",
        "category": "temperature",
        "sub_category": "1000mb",
        "description": "1000mb温度(重命名)",
        "source_columns": ["TMP_196"],
    },
    "tmp_925mb": {
        "formula": "TMP_286 (rename)",
        "category": "temperature",
        "sub_category": "925mb",
        "description": "925mb温度(重命名)",
        "source_columns": ["TMP_286"],
    },
    "tmp_850mb": {
        "formula": "TMP_302 (rename)",
        "category": "temperature",
        "sub_category": "850mb",
        "description": "850mb温度(重命名)",
        "source_columns": ["TMP_302"],
    },
    "tmp_700mb": {
        "formula": "TMP_350 (rename)",
        "category": "temperature",
        "sub_category": "700mb",
        "description": "700mb温度(重命名)",
        "source_columns": ["TMP_350"],
    },
    "tmp_500mb": {
        "formula": "TMP_446 (rename)",
        "category": "temperature",
        "sub_category": "500mb",
        "description": "500mb温度(重命名)",
        "source_columns": ["TMP_446"],
    },
    "rh_1000mb": {
        "formula": "RH_197 (rename)",
        "category": "humidity",
        "sub_category": "1000mb",
        "description": "1000mb相对湿度(重命名)",
        "source_columns": ["RH_197"],
    },
    "rh_850mb": {
        "formula": "RH_303 (rename)",
        "category": "humidity",
        "sub_category": "850mb",
        "description": "850mb相对湿度(重命名)",
        "source_columns": ["RH_303"],
    },
    "rh_700mb": {
        "formula": "RH_351 (rename)",
        "category": "humidity",
        "sub_category": "700mb",
        "description": "700mb相对湿度(重命名)",
        "source_columns": ["RH_351"],
    },
    "rh_500mb": {
        "formula": "RH_447 (rename)",
        "category": "humidity",
        "sub_category": "500mb",
        "description": "500mb相对湿度(重命名)",
        "source_columns": ["RH_447"],
    },
    "tmp_2m": {
        "formula": "TMP_581 (rename)",
        "category": "temperature",
        "sub_category": "surface",
        "description": "2m气温(重命名)",
        "source_columns": ["TMP_581"],
    },
    "dpt_2m": {
        "formula": "DPT_583 (rename)",
        "category": "temperature",
        "sub_category": "surface",
        "description": "2m露点温度(重命名)",
        "source_columns": ["DPT_583"],
    },
    "rh_2m": {
        "formula": "RH_584 (rename)",
        "category": "humidity",
        "sub_category": "surface",
        "description": "2m相对湿度(重命名)",
        "source_columns": ["RH_584"],
    },
    "dswrf": {
        "formula": "DSWRF_653 (rename)",
        "category": "radiation",
        "sub_category": "surface",
        "description": "地面短波辐射(重命名)",
        "source_columns": ["DSWRF_653"],
    },
    "dlwrf": {
        "formula": "DLWRF_654 (rename)",
        "category": "radiation",
        "sub_category": "surface",
        "description": "地面长波辐射(重命名)",
        "source_columns": ["DLWRF_654"],
    },
    "sunsd": {
        "formula": "SUNSD_622 (rename)",
        "category": "radiation",
        "sub_category": "surface",
        "description": "日照时长(重命名)",
        "source_columns": ["SUNSD_622"],
    },
    "tcdc_low": {
        "formula": "LCDC_630 (rename)",
        "category": "cloud",
        "sub_category": "low",
        "description": "低云量(重命名)",
        "source_columns": ["LCDC_630"],
    },
    "tcdc_mid": {
        "formula": "MCDC_632 (rename)",
        "category": "cloud",
        "sub_category": "mid",
        "description": "中云量(重命名)",
        "source_columns": ["MCDC_632"],
    },
    "tcdc_high": {
        "formula": "HCDC_634 (rename)",
        "category": "cloud",
        "sub_category": "high",
        "description": "高云量(重命名)",
        "source_columns": ["HCDC_634"],
    },
    "tcdc_total": {
        "formula": "TCDC_636 (rename)",
        "category": "cloud",
        "sub_category": "surface",
        "description": "总云量(重命名)",
        "source_columns": ["TCDC_636"],
    },
    "radiation_clear_ratio": {
        "formula": "DSWRF_653 / max(DSWRF_653_rolling_max, 1)",
        "category": "radiation",
        "sub_category": "surface",
        "description": "辐射晴空比(派生)",
        "source_columns": ["DSWRF_653"],
    },
}


class GFSCoarseSelector:
    def __init__(self):
        self.feature_registry: Dict[str, FeatureMeta] = {}
        self.deleted_features: Dict[str, str] = {}
        self.derived_features: Dict[str, FeatureMeta] = {}

    def run_coarse_selection(self, weather_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=" * 60)
        logger.info("GFS COARSE FEATURE SELECTION")
        logger.info("=" * 60)

        raw_cols = [c for c in weather_df.columns if c not in
                    {"TIME_FCST", "TIME_FORECAST", "CITY_NAME", "CITY_CODE", "LON", "LAT"}]
        logger.info(f"  Raw GFS feature columns: {len(raw_cols)}")

        self._step1_clean_and_deduplicate(weather_df, raw_cols)
        self._step2_categorize()
        self._step3_force_delete()
        self._step4_mark_retained()
        self._step5_build_derived_features(weather_df)

        candidate_pool = self._build_candidate_pool()
        feature_doc = self._build_feature_documentation()

        n_retained = sum(1 for m in self.feature_registry.values() if m.is_retained)
        n_derived = len(self.derived_features)
        logger.info(f"  Retained raw features: {n_retained}")
        logger.info(f"  Derived features: {n_derived}")
        logger.info(f"  Deleted features: {len(self.deleted_features)}")
        logger.info(f"  Total candidate pool: {n_retained + n_derived}")

        return candidate_pool, feature_doc

    def _step1_clean_and_deduplicate(self, df: pd.DataFrame, raw_cols: List[str]):
        logger.info("  Step 1: Cleaning and deduplication ...")
        to_delete = {}

        null_ratio = df[raw_cols].isnull().sum() / len(df)
        high_null = null_ratio[null_ratio > 0.8].index.tolist()
        for c in high_null:
            to_delete[c] = "null_ratio > 80%"
        logger.info(f"    High-null columns: {len(high_null)}")

        for c in raw_cols:
            if c in to_delete:
                continue
            if c in df.columns:
                variance = df[c].var()
                if pd.notna(variance) and variance < 1e-10:
                    to_delete[c] = "variance_near_zero"

        for c in raw_cols:
            if c in to_delete:
                continue
            if c in df.columns:
                n_unique = df[c].nunique()
                if n_unique <= 1:
                    to_delete[c] = "single_value"

        for c, reason in to_delete.items():
            self.deleted_features[c] = reason
            self.feature_registry[c] = FeatureMeta(
                name=c, source_column=c, category="unknown", sub_category="unknown",
                description="", is_retained=False, delete_reason=reason,
            )

        logger.info(f"    Cleaned: {len(to_delete)} columns removed")

    def _step2_categorize(self):
        logger.info("  Step 2: Categorizing features ...")
        for name, meta in self.feature_registry.items():
            if not meta.is_retained and meta.delete_reason:
                continue
            category, sub = self._classify_column(name)
            meta.category = category
            meta.sub_category = sub

    def _classify_column(self, col_name: str) -> Tuple[str, str]:
        prefix = col_name.split("_")[0] if "_" in col_name else col_name

        for cat_name, cat_info in GFS_VARIABLE_CATEGORIES.items():
            if prefix in cat_info["prefixes"]:
                level = self._extract_level(col_name)
                return cat_name, level

        return "other", "unknown"

    def _extract_level(self, col_name: str) -> str:
        level_map = {
            "588": "10m", "589": "10m", "689": "100m", "690": "100m",
            "581": "2m", "583": "2m", "584": "2m", "586": "2m", "587": "2m",
            "1": "surface", "639": "surface", "14": "surface",
            "653": "surface", "654": "surface", "622": "surface",
            "630": "low", "632": "mid", "634": "high", "636": "surface",
            "593": "surface", "596": "surface", "604": "surface", "601": "surface",
            "609": "surface", "610": "surface",
            "624": "surface", "625": "surface", "712": "surface", "626": "surface",
            "196": "1000mb", "197": "1000mb", "202": "1000mb", "203": "1000mb",
            "286": "925mb", "287": "925mb", "292": "925mb", "293": "925mb",
            "302": "850mb", "303": "850mb", "304": "850mb", "308": "850mb", "309": "850mb",
            "350": "700mb", "351": "700mb", "352": "700mb", "356": "700mb", "357": "700mb",
            "446": "500mb", "447": "500mb", "452": "500mb", "453": "500mb",
            "494": "300mb", "500": "300mb", "501": "300mb",
            "564": "soil", "565": "soil",
            "730": "surface", "577": "surface", "578": "surface",
        }

        if "_" in col_name:
            suffix = col_name.split("_")[-1]
            return level_map.get(suffix, "unknown")
        return "unknown"

    def _step3_force_delete(self):
        logger.info("  Step 3: Force deleting unwanted features ...")

        for name in list(self.feature_registry.keys()):
            meta = self.feature_registry[name]
            if not meta.is_retained:
                continue

            prefix = name.split("_")[0] if "_" in name else name

            if prefix in FORCE_DELETE_PREFIXES:
                meta.is_retained = False
                meta.delete_reason = "ozone_class_force_delete"
                self.deleted_features[name] = meta.delete_reason
                continue

            if prefix in FORCE_DELETE_MICROPHYSICS:
                meta.is_retained = False
                meta.delete_reason = "microphysics_force_delete"
                self.deleted_features[name] = meta.delete_reason
                continue

        logger.info(f"    Force deleted: {sum(1 for m in self.feature_registry.values() if not m.is_retained and 'force' in m.delete_reason)}")

    def _step4_mark_retained(self):
        logger.info("  Step 4: Marking retained features ...")

        all_keep = {}
        all_keep.update({k: ("wind_core", v[0], v[1], v[2]) for k, v in WIND_CORE_FEATURES.items()})
        all_keep.update({k: ("solar_core", v[0], v[1], v[2]) for k, v in SOLAR_CORE_FEATURES.items()})
        all_keep.update({k: ("general_meteo", v[0], v[1], v[2]) for k, v in GENERAL_METEO_FEATURES.items()})
        all_keep.update({k: ("surface_aux", v[0], v[1], v[2]) for k, v in SURFACE_AUX_FEATURES.items()})
        all_keep.update({k: ("isobaric", v[0], v[1], v[2]) for k, v in ISOBARIC_KEEP_FEATURES.items()})

        extra_keep = {
            "DPT_583": ("temperature", "surface", "2m露点温度"),
            "TMAX_586": ("temperature", "surface", "日最高温"),
            "TMIN_587": ("temperature", "surface", "日最低温"),
            "PRATE_593": ("precipitation", "surface", "降水率"),
            "APCP_596": ("precipitation", "surface", "累积降水"),
            "CRAIN_604": ("precipitation", "surface", "降水类型(雨)"),
            "CSNOW_601": ("precipitation", "surface", "降水类型(雪)"),
            "LHTFL_609": ("flux", "surface", "潜热通量"),
            "SHTFL_610": ("flux", "surface", "感热通量"),
            "SNOD_578": ("surface", "snow", "雪深"),
        }
        all_keep.update({k: ("extra", v[0], v[1], v[2]) for k, v in extra_keep.items()})

        for col_name, (group, cat, sub, desc) in all_keep.items():
            if col_name in self.feature_registry:
                meta = self.feature_registry[col_name]
                if not meta.is_retained and meta.delete_reason in ("null_ratio > 80%", "variance_near_zero", "single_value"):
                    continue
                meta.is_retained = True
                meta.category = cat
                meta.sub_category = sub
                meta.description = desc
            else:
                self.feature_registry[col_name] = FeatureMeta(
                    name=col_name, source_column=col_name,
                    category=cat, sub_category=sub, description=desc,
                    is_retained=True,
                )

        for name, meta in self.feature_registry.items():
            if meta.is_retained and not meta.description:
                if name not in all_keep:
                    meta.is_retained = False
                    meta.delete_reason = "not_in_retained_list"
                    self.deleted_features[name] = meta.delete_reason

        n_retained = sum(1 for m in self.feature_registry.values() if m.is_retained)
        logger.info(f"    Retained raw features: {n_retained}")

    def _step5_build_derived_features(self, weather_df: pd.DataFrame):
        logger.info("  Step 5: Building derived features ...")

        for feat_name, spec in DERIVED_FEATURES_SPEC.items():
            source_cols = spec["source_columns"]
            all_available = all(c in weather_df.columns for c in source_cols)

            meta = FeatureMeta(
                name=feat_name,
                source_column=",".join(source_cols),
                category=spec["category"],
                sub_category=spec["sub_category"],
                description=spec["description"],
                is_derived=True,
                derivation_formula=spec["formula"],
                is_retained=all_available,
                delete_reason="" if all_available else "source_columns_missing",
            )
            self.derived_features[feat_name] = meta

        logger.info(f"    Derived features defined: {len(self.derived_features)}")
        logger.info(f"    Available: {sum(1 for m in self.derived_features.values() if m.is_retained)}")

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "UGRD_588" in df.columns and "VGRD_589" in df.columns:
            df["ws_10m"] = np.sqrt(df["UGRD_588"]**2 + df["VGRD_589"]**2)
            df["wd_10m"] = np.degrees(np.arctan2(df["VGRD_589"], df["UGRD_588"])) % 360

        if "UGRD_689" in df.columns and "VGRD_690" in df.columns:
            df["ws_100m"] = np.sqrt(df["UGRD_689"]**2 + df["VGRD_690"]**2)
            df["wd_100m"] = np.degrees(np.arctan2(df["VGRD_690"], df["UGRD_689"])) % 360

        if "ws_100m" in df.columns and "ws_10m" in df.columns:
            df["wind_shear_100m_10m"] = df["ws_100m"] - df["ws_10m"]

        if "UGRD_308" in df.columns and "VGRD_309" in df.columns and "ws_10m" in df.columns:
            ws_850 = np.sqrt(df["UGRD_308"]**2 + df["VGRD_309"]**2)
            df["wind_shear_850mb_surface"] = ws_850 - df["ws_10m"]

        if "ws_10m" in df.columns:
            df["ws_10m_sq"] = df["ws_10m"] ** 2
            df["ws_10m_cu"] = df["ws_10m"] ** 3

        if "ws_100m" in df.columns:
            df["ws_100m_sq"] = df["ws_100m"] ** 2
            df["ws_100m_cu"] = df["ws_100m"] ** 3

        if "wd_10m" in df.columns:
            df["wd_10m_change"] = df["wd_10m"].diff()
            df["wd_10m_change"] = ((df["wd_10m_change"] + 180) % 360) - 180

        if "TMP_581" in df.columns and "DPT_583" in df.columns:
            df["temp_dew_spread"] = df["TMP_581"] - df["DPT_583"]

        if "TMAX_586" in df.columns and "TMIN_587" in df.columns:
            df["temp_diurnal_range"] = df["TMAX_586"] - df["TMIN_587"]

        if all(c in df.columns for c in ["LCDC_630", "MCDC_632", "HCDC_634"]):
            df["cloud_total_weighted"] = 0.3 * df["LCDC_630"] + 0.4 * df["MCDC_632"] + 0.3 * df["HCDC_634"]

        if "TCDC_636" in df.columns:
            df["clear_sky_index"] = 1 - df["TCDC_636"] / 100

        if "DSWRF_653" in df.columns:
            df["dswrf_sq"] = df["DSWRF_653"] ** 2

        rename_map = {
            "CAPE_624": "cape_surface",
            "CIN_625": "cin_surface",
            "HPBL_712": "pbl_height",
            "PRATE_593": "precip_rate",
            "APCP_596": "precip_accum",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        return df

    def _build_candidate_pool(self) -> pd.DataFrame:
        rows = []
        for name, meta in self.feature_registry.items():
            if meta.is_retained:
                rows.append({
                    "feature_name": name,
                    "source_column": meta.source_column,
                    "category": meta.category,
                    "sub_category": meta.sub_category,
                    "description": meta.description,
                    "is_derived": False,
                    "derivation_formula": "",
                })

        for name, meta in self.derived_features.items():
            if meta.is_retained:
                rows.append({
                    "feature_name": name,
                    "source_column": meta.source_column,
                    "category": meta.category,
                    "sub_category": meta.sub_category,
                    "description": meta.description,
                    "is_derived": True,
                    "derivation_formula": meta.derivation_formula,
                })

        pool = pd.DataFrame(rows)
        logger.info(f"  Candidate pool: {len(pool)} features")
        return pool

    def _build_feature_documentation(self) -> pd.DataFrame:
        rows = []
        for name, meta in self.feature_registry.items():
            rows.append({
                "feature_name": name,
                "source_column": meta.source_column,
                "category": meta.category,
                "sub_category": meta.sub_category,
                "description": meta.description,
                "is_derived": meta.is_derived,
                "derivation_formula": meta.derivation_formula,
                "is_retained": meta.is_retained,
                "delete_reason": meta.delete_reason,
            })

        for name, meta in self.derived_features.items():
            rows.append({
                "feature_name": name,
                "source_column": meta.source_column,
                "category": meta.category,
                "sub_category": meta.sub_category,
                "description": meta.description,
                "is_derived": True,
                "derivation_formula": meta.derivation_formula,
                "is_retained": meta.is_retained,
                "delete_reason": meta.delete_reason,
            })

        doc = pd.DataFrame(rows)
        return doc

    def get_retained_columns(self) -> List[str]:
        retained = [name for name, meta in self.feature_registry.items() if meta.is_retained]
        derived = [name for name, meta in self.derived_features.items() if meta.is_retained]
        return retained + derived

    def get_wind_features(self) -> List[str]:
        features = []
        for name, meta in self.feature_registry.items():
            if meta.is_retained and meta.category in ("wind", "wind_derived"):
                features.append(name)
        for name, meta in self.derived_features.items():
            if meta.is_retained and meta.category in ("wind", "wind_derived"):
                features.append(name)
        return features

    def get_solar_features(self) -> List[str]:
        features = []
        for name, meta in self.feature_registry.items():
            if meta.is_retained and meta.category in ("radiation", "cloud"):
                features.append(name)
        for name, meta in self.derived_features.items():
            if meta.is_retained and meta.category in ("radiation", "cloud"):
                features.append(name)
        return features

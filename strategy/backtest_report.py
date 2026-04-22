import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from strategy.backtest_runner import BacktestResult, DayBacktestResult
from strategy.best_strategy_selector import SelectionResult, StrategyRanking

logger = logging.getLogger(__name__)


class BacktestReport:
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir

    def generate_full_report(self,
                             backtest_results: Dict[str, BacktestResult],
                             selection: SelectionResult = None) -> str:
        logger.info("  Generating full backtest report ...")

        sections = []
        sections.append(self._header())
        sections.append(self._strategy_comparison_table(backtest_results))
        sections.append(self._detailed_summary(backtest_results))

        if selection is not None:
            sections.append(self._selection_report(selection))
            sections.append(self._scenario_best_report(selection))

        sections.append(self._scenario_analysis(backtest_results))
        sections.append(self._daily_profit_table(backtest_results))
        sections.append(self._worst_days_analysis(backtest_results))

        report = "\n\n".join(sections)

        import os
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, "backtest_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        self._export_csv_reports(backtest_results, selection)

        logger.info(f"  Report saved to {report_path}")
        return report

    def _header(self) -> str:
        return (
            "=" * 80 + "\n"
            "电力现货策略回测报告\n"
            "=" * 80 + "\n"
            f"生成时间: {pd.Timestamp.now()}\n"
        )

    def _strategy_comparison_table(self,
                                   backtest_results: Dict[str, BacktestResult]) -> str:
        lines = ["\n" + "-" * 60, "策略对比总表", "-" * 60]
        header = f"{'策略':<25} {'总收益':>12} {'日均收益':>12} {'Sharpe':>10} {'回收金额':>12} {'超阈值':>8}"
        lines.append(header)
        lines.append("-" * 80)

        sorted_results = sorted(
            backtest_results.items(),
            key=lambda x: x[1].summary.get("total_profit", 0),
            reverse=True,
        )

        for name, result in sorted_results:
            s = result.summary
            lines.append(
                f"{name:<25} {s.get('total_profit', 0):>12.2f} "
                f"{s.get('avg_daily_profit', 0):>12.2f} "
                f"{s.get('profit_sharpe', 0):>10.4f} "
                f"{s.get('total_recovery', 0):>12.2f} "
                f"{s.get('total_over_threshold', 0):>8d}"
            )

        return "\n".join(lines)

    def _detailed_summary(self,
                          backtest_results: Dict[str, BacktestResult]) -> str:
        lines = ["\n" + "-" * 60, "详细统计", "-" * 60]

        for name, result in backtest_results.items():
            s = result.summary
            lines.append(f"\n  [{name}]")
            lines.append(f"    总收益: {s.get('total_profit', 0):.2f}")
            lines.append(f"    日均收益: {s.get('avg_daily_profit', 0):.2f}")
            lines.append(f"    收益标准差: {s.get('std_daily_profit', 0):.2f}")
            lines.append(f"    最差日收益: {s.get('worst_day_profit', 0):.2f}")
            lines.append(f"    最佳日收益: {s.get('best_day_profit', 0):.2f}")
            lines.append(f"    Sharpe比率: {s.get('profit_sharpe', 0):.4f}")
            lines.append(f"    总回收金额: {s.get('total_recovery', 0):.2f}")
            lines.append(f"    平均回收比: {s.get('recovery_ratio_avg', 0):.4f}")
            lines.append(f"    超阈值总次数: {s.get('total_over_threshold', 0)}")
            lines.append(f"    回测天数: {s.get('n_days', 0)}")

        return "\n".join(lines)

    def _selection_report(self, selection: SelectionResult) -> str:
        lines = ["\n" + "-" * 60, "策略选择结果", "-" * 60]
        lines.append(f"  最优策略: {selection.best_strategy}")
        lines.append(f"\n  排名:")

        for r in selection.rankings:
            lines.append(
                f"    #{r.rank} {r.name}: "
                f"综合得分={r.composite_score:.4f}, "
                f"总收益={r.total_profit:.2f}, "
                f"Sharpe={r.sharpe_ratio:.4f}, "
                f"最大回撤={r.max_drawdown:.2f}"
            )

        return "\n".join(lines)

    def _scenario_best_report(self, selection: SelectionResult) -> str:
        lines = ["\n" + "-" * 60, "分场景最优策略", "-" * 60]
        for scenario, best_name in selection.scenario_best.items():
            lines.append(f"  {scenario}: {best_name}")
        return "\n".join(lines)

    def _scenario_analysis(self,
                           backtest_results: Dict[str, BacktestResult]) -> str:
        lines = ["\n" + "-" * 60, "分场景分析", "-" * 60]

        all_scenarios = set()
        for result in backtest_results.values():
            all_scenarios.update(result.scenario_summaries.keys())

        for scenario in sorted(all_scenarios):
            lines.append(f"\n  [{scenario}]")
            header = f"    {'策略':<25} {'平均收益':>12} {'最差日':>12} {'回收':>12} {'超阈值':>8} {'稳定性':>10}"
            lines.append(header)
            lines.append("    " + "-" * 75)

            scenario_data = []
            for name, result in backtest_results.items():
                if scenario in result.scenario_summaries:
                    ss = result.scenario_summaries[scenario]
                    scenario_data.append((name, ss))

            scenario_data.sort(key=lambda x: x[1].get("avg_profit", float("-inf")), reverse=True)

            for name, ss in scenario_data:
                lines.append(
                    f"    {name:<25} {ss.get('avg_profit', 0):>12.2f} "
                    f"{ss.get('worst_day_profit', 0):>12.2f} "
                    f"{ss.get('total_recovery', 0):>12.2f} "
                    f"{ss.get('total_over_threshold', 0):>8d} "
                    f"{ss.get('stability', 0):>10.4f}"
                )

        return "\n".join(lines)

    def _daily_profit_table(self,
                            backtest_results: Dict[str, BacktestResult]) -> str:
        lines = ["\n" + "-" * 60, "每日收益明细", "-" * 60]

        best_name = max(
            backtest_results.keys(),
            key=lambda k: backtest_results[k].summary.get("total_profit", 0),
        )
        best_result = backtest_results[best_name]

        header = f"{'日期':<12} {'收益':>12} {'回收':>12} {'超阈值':>8} {'场景':<15}"
        lines.append(f"  [{best_name}]")
        lines.append(f"  {header}")
        lines.append("  " + "-" * 60)

        for date, day_result in sorted(best_result.daily_results.items()):
            lines.append(
                f"  {date:<12} {day_result.settlement.total_profit:>12.2f} "
                f"{day_result.settlement.recovery_cost:>12.2f} "
                f"{day_result.settlement.deviation_over_threshold_count:>8d} "
                f"{day_result.scenario_type:<15}"
            )

        return "\n".join(lines)

    def _worst_days_analysis(self,
                             backtest_results: Dict[str, BacktestResult]) -> str:
        lines = ["\n" + "-" * 60, "最差日分析", "-" * 60]

        for name, result in backtest_results.items():
            sorted_days = sorted(
                result.daily_results.items(),
                key=lambda x: x[1].settlement.total_profit,
            )

            lines.append(f"\n  [{name}] 最差5天:")
            for date, day_result in sorted_days[:5]:
                lines.append(
                    f"    {date}: 收益={day_result.settlement.total_profit:.2f}, "
                    f"回收={day_result.settlement.recovery_cost:.2f}, "
                    f"超阈值={day_result.settlement.deviation_over_threshold_count}, "
                    f"场景={day_result.scenario_type}"
                )

        return "\n".join(lines)

    def _export_csv_reports(self,
                            backtest_results: Dict[str, BacktestResult],
                            selection: SelectionResult = None):
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        rows = []
        for name, result in backtest_results.items():
            for date, day_result in result.daily_results.items():
                rows.append({
                    "strategy": name,
                    "date": date,
                    "total_profit": day_result.settlement.total_profit,
                    "day_ahead_cost": day_result.settlement.day_ahead_cost,
                    "real_time_cost": day_result.settlement.real_time_cost,
                    "recovery_cost": day_result.settlement.recovery_cost,
                    "over_threshold": day_result.settlement.deviation_over_threshold_count,
                    "recovery_ratio": day_result.settlement.recovery_ratio,
                    "scenario_type": day_result.scenario_type,
                    "is_holiday": day_result.is_holiday,
                    "is_post_holiday": day_result.is_post_holiday,
                    "is_missing_info": day_result.is_missing_info,
                })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(self.output_dir, "daily_results.csv"), index=False)

        summary_rows = []
        for name, result in backtest_results.items():
            s = result.summary
            summary_rows.append({
                "strategy": name,
                **{k: v for k, v in s.items()},
            })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(os.path.join(self.output_dir, "strategy_summary.csv"), index=False)

        if selection is not None and selection.rankings:
            ranking_rows = []
            for r in selection.rankings:
                ranking_rows.append({
                    "rank": r.rank,
                    "strategy": r.name,
                    "total_profit": r.total_profit,
                    "avg_daily_profit": r.avg_daily_profit,
                    "risk_adjusted_profit": r.risk_adjusted_profit,
                    "sharpe_ratio": r.sharpe_ratio,
                    "max_drawdown": r.max_drawdown,
                    "total_recovery": r.total_recovery,
                    "over_threshold_count": r.over_threshold_count,
                    "holiday_profit": r.holiday_profit,
                    "missing_info_profit": r.missing_info_profit,
                    "composite_score": r.composite_score,
                })

            df_ranking = pd.DataFrame(ranking_rows)
            df_ranking.to_csv(os.path.join(self.output_dir, "strategy_rankings.csv"), index=False)

        logger.info(f"  CSV reports exported to {self.output_dir}")

    def generate_visualization(self,
                               backtest_results: Dict[str, BacktestResult],
                               selection: SelectionResult = None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except ImportError:
            logger.warning("  matplotlib not available, skipping visualization")
            return

        import os
        os.makedirs(self.output_dir, exist_ok=True)

        self._plot_cumulative_profit(backtest_results)
        self._plot_daily_profit_comparison(backtest_results)
        self._plot_scenario_heatmap(backtest_results)

        if selection is not None:
            self._plot_ranking(selection)

        logger.info(f"  Visualizations saved to {self.output_dir}")

    def _plot_cumulative_profit(self, backtest_results: Dict[str, BacktestResult]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        for name, result in backtest_results.items():
            sorted_days = sorted(result.daily_results.items())
            profits = [r.settlement.total_profit for _, r in sorted_days]
            dates = [d for d, _ in sorted_days]
            cumsum = np.cumsum(profits)
            ax.plot(range(len(cumsum)), cumsum, label=name, linewidth=1.5)

        ax.set_title("累计收益曲线")
        ax.set_xlabel("交易日")
        ax.set_ylabel("累计收益")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        import os
        fig.savefig(os.path.join(self.output_dir, "cumulative_profit.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_daily_profit_comparison(self, backtest_results: Dict[str, BacktestResult]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        names = list(backtest_results.keys())
        n_strategies = len(names)
        bar_width = 0.8 / max(n_strategies, 1)

        all_dates = set()
        for result in backtest_results.values():
            all_dates.update(result.daily_results.keys())
        all_dates = sorted(all_dates)

        for i, name in enumerate(names):
            result = backtest_results[name]
            profits = []
            for d in all_dates:
                if d in result.daily_results:
                    profits.append(result.daily_results[d].settlement.total_profit)
                else:
                    profits.append(0)

            x = np.arange(len(all_dates)) + i * bar_width
            ax.bar(x, profits, bar_width, label=name, alpha=0.7)

        ax.set_title("每日收益对比")
        ax.set_xlabel("交易日")
        ax.set_ylabel("收益")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        import os
        fig.savefig(os.path.join(self.output_dir, "daily_profit_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_scenario_heatmap(self, backtest_results: Dict[str, BacktestResult]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_scenarios = set()
        for result in backtest_results.values():
            all_scenarios.update(result.scenario_summaries.keys())
        all_scenarios = sorted(all_scenarios)

        names = list(backtest_results.keys())

        if not all_scenarios or not names:
            return

        data = np.zeros((len(names), len(all_scenarios)))
        for i, name in enumerate(names):
            for j, scenario in enumerate(all_scenarios):
                ss = backtest_results[name].scenario_summaries.get(scenario, {})
                data[i, j] = ss.get("avg_profit", 0)

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(all_scenarios)))
        ax.set_xticklabels(all_scenarios, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("分场景平均收益热力图")
        fig.colorbar(im, ax=ax)

        for i in range(len(names)):
            for j in range(len(all_scenarios)):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7)

        import os
        fig.savefig(os.path.join(self.output_dir, "scenario_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_ranking(self, selection: SelectionResult):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r.name for r in selection.rankings]
        scores = [r.composite_score for r in selection.rankings]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        colors = ["#2ecc71" if s == max(scores) else "#3498db" for s in scores]
        ax.barh(range(len(names)), scores, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("综合得分")
        ax.set_title("策略综合排名")
        ax.grid(True, alpha=0.3, axis="x")

        import os
        fig.savefig(os.path.join(self.output_dir, "strategy_ranking.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

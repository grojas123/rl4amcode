from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl4am.reporting.comparison import compare_strategy_paths


DISPLAY_METRICS = (
    ("ann_return", "annualised_return"),
    ("ann_vol", "annualised_volatility"),
    ("ann_sharpe", "annualised_sharpe"),
    ("max_dd", "max_drawdown"),
    ("term_eq", None),
)

DISPLAY_NAMES = {
    "a2c_mode": "a2c_mode",
    "a2c_mean": "a2c_mean",
}


def build_comparison_table(
    left_path: str | Path,
    right_path: str | Path,
    output_dir: str | Path,
    use_net: bool = True,
) -> pd.DataFrame:
    """Build a compact comparison table and write CSV/TeX outputs."""
    comparison = compare_strategy_paths(left_path, right_path, use_net=use_net)
    rows: list[dict[str, float | str]] = []
    for display_key, source_key in DISPLAY_METRICS:
        if source_key is None:
            left_value = comparison.left_terminal_equity
            right_value = comparison.right_terminal_equity
        else:
            left_value = comparison.left_metrics[source_key]
            right_value = comparison.right_metrics[source_key]
        rows.append(
            {
                "metric": display_key,
                _display_name(comparison.left_name): left_value,
                _display_name(comparison.right_name): right_value,
            }
        )
    table = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison_table.csv", index=False)
    (out / "comparison_table.tex").write_text(
        _table_to_tex(table),
        encoding="utf-8",
    )
    return table


def _table_to_tex(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    aligns = "l" + "r" * (len(cols) - 1)
    lines = [
        r"\begin{tabular}{" + aligns + "}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        values = [str(row[cols[0]])]
        for col in cols[1:]:
            values.append(_format_metric(float(row[col]), row[cols[0]]))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _format_metric(value: float, metric_name: str) -> str:
    if metric_name in {"ann_return", "ann_vol", "max_dd"}:
        return f"{value:.4f}"
    if metric_name == "ann_sharpe":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)

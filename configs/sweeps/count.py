from __future__ import annotations

from functools import reduce
from operator import mul
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = []
    for path in sorted(root.glob("*.yml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        grid = spec.get("grid") or {}
        seeds = spec.get("seeds") or [None]
        factors = [len(values) for values in grid.values()]
        varying = [factor for factor in factors if factor > 1]
        combos = reduce(mul, factors, 1)
        runs = combos * len(seeds)
        factor_text = "x".join(str(factor) for factor in varying) or "1"
        rows.append((path.name, combos, len(seeds), runs, len(varying), factor_text))

    name_width = max((len(row[0]) for row in rows), default=4)
    factor_width = max((len(row[5]) for row in rows), default=7)
    print(
        f"{'sweep':<{name_width}}  combos  seeds  runs  varying  "
        f"{'factors':<{factor_width}}"
    )
    print(
        f"{'-' * name_width}  ------  -----  ----  -------  "
        f"{'-' * factor_width}"
    )
    for name, combos, seeds, runs, varying, factors in rows:
        print(
            f"{name:<{name_width}}  {combos:>6}  {seeds:>5}  {runs:>4}  "
            f"{varying:>7}  {factors:<{factor_width}}"
        )


if __name__ == "__main__":
    main()

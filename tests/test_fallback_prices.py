"""Until Member 1's folder is filled, prices load from the current external file."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htfe.config import contract_path, cultivation_monthly_path, member1_ready
from htfe.data.build.cultivation_progress import find_monthly_table


def main() -> None:
    assert member1_ready() is False
    path = contract_path("prices")
    frame = pd.read_csv(path, nrows=5)
    if frame.empty:
        raise SystemExit(f"price file is empty: {path}")
    found = find_monthly_table()
    direct = cultivation_monthly_path()
    if found != direct:
        raise SystemExit(f"cultivation path mismatch: {found} vs {direct}")
    if found is not None and found.name != "cultivation_monthly.csv":
        raise SystemExit(f"cultivation file must be the named contract file, got {found}")
    print(f"fallback prices ok rows_peek={len(frame)} path={path}")
    print(f"cultivation_monthly={found}")


if __name__ == "__main__":
    main()

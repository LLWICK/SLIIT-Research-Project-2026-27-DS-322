"""The as-of rule refuses a future report and keeps a ratio above 1."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htfe.training.run_cultivation_fixture import main


def test_asof_fixture() -> None:
    main()


if __name__ == "__main__":
    test_asof_fixture()
    print("as-of check passed")

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "01_models"
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nci_cho_transfer.formal_randomcv80_end_to_end import (
    FIXED_BEST10_SEEDS,
    run_end_to_end,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed ET pipeline with per-target leave-identical NCI source exclusion."
    )
    parser.add_argument("--root-dir", default=str(ROOT))
    parser.add_argument(
        "--run-id",
        default="formal_fixedbest10_leave_identical_compound_out_20260824",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "leave_identical_compound_out"),
    )
    parser.add_argument(
        "--seed-list",
        default=",".join(str(seed) for seed in FIXED_BEST10_SEEDS),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.seed_list.split(",") if value.strip())
    output = run_end_to_end(
        args.root_dir,
        args.run_id,
        seeds=seeds,
        exclude_identical_source=True,
        output_root=args.output_root,
    )
    print(output)

if __name__ == "__main__":
    main()

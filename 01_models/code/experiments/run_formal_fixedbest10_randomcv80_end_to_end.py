from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nci_cho_transfer.formal_randomcv80_end_to_end import run_end_to_end

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the frozen formal fixedbest10 RandomCV80 ET pipeline without prediction-file inputs."
    )
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight-seed", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seeds = (args.preflight_seed,) if args.preflight_seed is not None else None
    if seeds is None:
        print(f"wrote {run_end_to_end(args.root_dir, args.run_id)}")
    else:
        print(f"wrote {run_end_to_end(args.root_dir, args.run_id, seeds=seeds)}")

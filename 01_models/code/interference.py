from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIXED_SEEDS = (
    20260021,
    20260106,
    20260290,
    20260307,
    20260605,
    20260730,
    20260738,
    20260755,
    20260998,
    20262000,
)
ROLE_OFFSETS = {
    "random_cv_oof": 11,
    "train": 23,
    "test": 37,
    "validation": 53,
    "experimental": 71,
}
ROLES = ("random_cv_oof", "test", "validation", "experimental")
LATEST_ET_S3_RUN = "et_s3"
BASELINE_RUN = "et_s2"
REPEAT_STRIDE = 1_000_003

def _r2(y_true: np.ndarray, prediction: np.ndarray) -> float:
    centered = y_true - y_true.mean()
    return float(1.0 - np.square(y_true - prediction).sum() / np.square(centered).sum())

def _load_aligned(root: Path, seed: int) -> pd.DataFrame:
    latest_dir = root / "data_processed" / "components" / LATEST_ET_S3_RUN
    baseline_dir = root / "data_processed" / "components" / BASELINE_RUN
    latest_summary = json.loads((latest_dir / "summary.json").read_text(encoding="utf-8"))
    if latest_summary.get("run_id") != LATEST_ET_S3_RUN:
        raise RuntimeError("latest ET-S3 summary run_id mismatch")
    expected_s3_r2 = {
        "test": 0.8647177006284427,
        "validation": 0.8008133745306238,
        "experimental": 0.8482372455011927,
    }
    for role, expected in expected_s3_r2.items():
        actual = float(latest_summary["holdout_by_role"][role]["r2"])
        if abs(actual - expected) > 1e-12:
            raise RuntimeError(f"latest ET-S3 {role} R2 reference mismatch")

    oof = pd.read_csv(latest_dir / "formal_random_cv_oof_predictions.csv", low_memory=False)
    holdout = pd.read_csv(latest_dir / "formal_holdout_predictions.csv", low_memory=False)
    latest = pd.concat(
        [oof.assign(eval_role="random_cv_oof"), holdout.assign(eval_role=holdout["split_role"])],
        ignore_index=True,
    )
    latest = latest[latest["split_seed"].astype(int).eq(seed)].copy()
    latest = latest.rename(columns={"blend_y_pred": "latest_s3"})

    baseline = pd.read_csv(baseline_dir / "et_s2_no_transfer_predictions.csv", low_memory=False)
    baseline = baseline[baseline["split_seed"].astype(int).eq(seed)].copy()
    baseline["eval_role"] = baseline["split_role"].where(
        baseline["split_role"].ne("random_cv_oof"), "random_cv_oof"
    )
    baseline = baseline.rename(columns={"s2_no_transfer": "reference_prediction"})

    keys = ["sample_id", "eval_role"]
    left = latest[keys + ["split_seed", "split_manifest_hash", "canonical_smiles", "time_h", "y_true", "latest_s3"]]
    right = baseline[keys + ["reference_prediction"]]
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"latest ET-S3 rows do not align for seed {seed}")
    return merged

def run(root: Path, output_dir: Path) -> Path:
    metric_rows: list[dict[str, object]] = []
    for seed in FIXED_SEEDS:
        aligned = _load_aligned(root, seed)
        aligned["split_seed"] = aligned["split_seed"].astype(int)
        for role, group in aligned.groupby("eval_role", sort=False):
            if role == "train":
                continue
            if role not in ROLE_OFFSETS:
                raise RuntimeError(f"unexpected role: {role}")
            group = group.sort_values(["sample_id", "time_h"], kind="mergesort")
            y_true = group["y_true"].to_numpy(dtype=float)
            reference = group["reference_prediction"].to_numpy(dtype=float)
            latest_s3 = group["latest_s3"].to_numpy(dtype=float)
            increment = latest_s3 - reference
            for repeat in range(1, 11):
                rng = np.random.default_rng(int(seed) + ROLE_OFFSETS[role] + (repeat - 1) * REPEAT_STRIDE)
                permuted = rng.permutation(increment)
                prediction = reference + 0.5 * increment + 0.5 * permuted
                metric_rows.append(
                    {
                        "split_seed": int(seed),
                        "split_role": role,
                        "repeat": repeat,
                        "n": int(len(group)),
                        "interference_r2": _r2(y_true, prediction),
                        "interference_rmse": float(np.sqrt(np.mean(np.square(y_true - prediction)))),
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(["split_role", "split_seed", "repeat"])
    expected_rows = len(FIXED_SEEDS) * 10 * len(ROLES)
    if len(metrics) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, got {len(metrics)}")
    summary_rows = (
        metrics.groupby("split_role", as_index=False)
        .agg(
            seed_count=("split_seed", "nunique"),
            repeat_count=("repeat", "count"),
            r2_mean=("interference_r2", "mean"),
            r2_sd=("interference_r2", "std"),
            r2_min=("interference_r2", "min"),
            r2_max=("interference_r2", "max"),
            rmse_mean=("interference_rmse", "mean"),
            rmse_sd=("interference_rmse", "std"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "interference_metrics.csv", index=False)
    summary = {
        "run_id": output_dir.name,
        "latest_et_s3_run": LATEST_ET_S3_RUN,
        "latest_et_s3_source_run": "formal_fixedbest10_randomcv80_adaptive_segment_blend_cvfirst_v1",
        "latest_et_s3_retrained": False,
        "fixed_seeds": list(FIXED_SEEDS),
        "repeats_per_seed": 10,
        "total_interference_repeats": 100,
        "roles": list(ROLES),
        "reference_stage": "ET-S2 frozen optimized no-transfer",
        "interference_definition": "ET-S2 + 0.5*(latest_ET_S3-ET-S2) + 0.5*permuted_increment",
        "permutation_seed_rule": "seed + role_offset + (repeat-1)*1000003",
        "latest_et_s3_is_full_transfer": True,
        "latest_et_s3_holdout_labels_used_for_selection": False,
        "permutation_uses_labels": False,
        "y_true_unchanged": True,
        "row_keys_preserved": True,
        "summary_by_role": summary_rows.to_dict(orient="records"),
    }
    (output_dir / "interference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 10x10 interference repeats from the frozen latest ET-S3.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(run(Path(args.root_dir).resolve(), Path(args.output_dir).resolve()))

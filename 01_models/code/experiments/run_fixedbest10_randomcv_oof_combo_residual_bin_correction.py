from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_PREDICTIONS = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_oof_combo_prediction_cap_fine_v1/"
    "prediction_cap_best_predictions.csv"
)
DEFAULT_ERROR_DECOMPOSITION = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_error_decomposition_v1/"
    "randomcv_best_predictions_with_errors.csv"
)

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def _bin_ids(raw_train: np.ndarray, raw_valid: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(raw_train.astype(float), np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(raw_train), dtype=int), np.zeros(len(raw_valid), dtype=int)
    cutpoints = edges[1:-1]
    return (
        np.digitize(raw_train, cutpoints, right=False).astype(int),
        np.digitize(raw_valid, cutpoints, right=False).astype(int),
    )

def _keys_for_scope(frame: pd.DataFrame, bins: np.ndarray, scope: str) -> np.ndarray:
    if scope == "bin":
        return np.array([f"b={int(bin_id)}" for bin_id in bins], dtype=object)
    if scope == "time":
        return np.array([f"t={float(time_h):g}" for time_h in frame["time_h"].to_numpy(dtype=float)], dtype=object)
    if scope == "time_bin":
        return np.array(
            [
                f"t={float(time_h):g}|b={int(bin_id)}"
                for time_h, bin_id in zip(frame["time_h"].to_numpy(dtype=float), bins)
            ],
            dtype=object,
        )
    if scope == "same_train_count_bin":
        counts = frame["same_canonical_train_count"].fillna(0).to_numpy(dtype=int)
        return np.array([f"c={min(int(count), 3)}|b={int(bin_id)}" for count, bin_id in zip(counts, bins)], dtype=object)
    raise ValueError(f"Unknown scope: {scope}")

def _fold_residual_bin_corrected(
    seed_frame: pd.DataFrame,
    scope: str,
    n_bins: int,
    shrink: float,
    min_segment_rows: int,
) -> np.ndarray:
    raw = seed_frame["y_pred"].to_numpy(dtype=float)
    y = seed_frame["y_true"].to_numpy(dtype=float)
    folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
    pred = raw.copy()
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != int(fold)
        valid_mask = ~train_mask
        train = seed_frame.loc[train_mask].reset_index(drop=True)
        valid = seed_frame.loc[valid_mask].reset_index(drop=True)
        train_bins, valid_bins = _bin_ids(raw[train_mask], raw[valid_mask], int(n_bins))
        train_keys = _keys_for_scope(train, train_bins, scope)
        valid_keys = _keys_for_scope(valid, valid_bins, scope)
        residual = y[train_mask] - raw[train_mask]
        global_bias = float(np.mean(residual))
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for key, value in zip(train_keys, residual):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
        bias = np.array(
            [
                sums[key] / float(counts[key]) if counts.get(key, 0) >= int(min_segment_rows) else global_bias
                for key in valid_keys
            ],
            dtype=float,
        )
        pred[valid_mask] = raw[valid_mask] + float(shrink) * bias
    return pred

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(_resolve(root, args.predictions))
    keys = pd.read_csv(
        _resolve(root, args.error_decomposition_predictions),
        usecols=[
            "split_seed",
            "sample_id",
            "random_cv_fold",
            "y_true",
            "time_h",
            "same_canonical_train_count",
        ],
    )
    frame = predictions.merge(keys, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    if len(frame) != len(predictions):
        raise ValueError(f"Prediction/key alignment failed: expected {len(predictions)}, got {len(frame)}")

    rows = []
    prediction_rows = []
    for scope in args.scopes:
        for n_bins in args.n_bins:
            for shrink in args.shrinks:
                for min_segment_rows in args.min_segment_rows:
                    for seed, seed_frame in frame.groupby("split_seed", sort=True):
                        seed_frame = seed_frame.reset_index(drop=True).copy()
                        pred = _fold_residual_bin_corrected(
                            seed_frame,
                            str(scope),
                            int(n_bins),
                            float(shrink),
                            int(min_segment_rows),
                        )
                        y = seed_frame["y_true"].to_numpy(dtype=float)
                        rows.append(
                            {
                                "scope": str(scope),
                                "n_bins": int(n_bins),
                                "shrink": float(shrink),
                                "min_segment_rows": int(min_segment_rows),
                                "split_seed": int(seed),
                                "r2": float(r2_score(y, pred)),
                                "rmse": float(mean_squared_error(y, pred) ** 0.5),
                            }
                        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "residual_bin_correction_metrics_by_seed.csv", index=False)
    summary = (
        metrics.groupby(["scope", "n_bins", "shrink", "min_segment_rows"], sort=True)
        .agg(
            seed_count=("split_seed", "nunique"),
            random_cv_r2_mean=("r2", "mean"),
            random_cv_r2_min=("r2", "min"),
            random_cv_r2_max=("r2", "max"),
            random_cv_rmse_mean=("rmse", "mean"),
        )
        .reset_index()
        .sort_values(["random_cv_r2_mean", "random_cv_rmse_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary.to_csv(out_dir / "residual_bin_correction_summary.csv", index=False)
    best = summary.iloc[0].to_dict()

    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        pred = _fold_residual_bin_corrected(
            seed_frame,
            str(best["scope"]),
            int(best["n_bins"]),
            float(best["shrink"]),
            int(best["min_segment_rows"]),
        )
        pred_frame = seed_frame[["split_seed", "sample_id", "random_cv_fold", "y_true"]].copy()
        pred_frame["target_mode"] = "residual_bin_corrected_best"
        pred_frame["y_pred"] = pred
        prediction_rows.append(pred_frame)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out_dir / "residual_bin_corrected_best_predictions.csv", index=False)

    payload = {
        "run_id": args.run_id,
        "diagnostic_only": True,
        "fold_train_only": True,
        "uses_holdout_labels": False,
        "source_predictions": args.predictions,
        "source_error_decomposition_predictions": args.error_decomposition_predictions,
        "evaluation_count": int(len(summary)),
        "best_row": best,
        "passes_random_cv80_target": bool(float(best["random_cv_r2_mean"]) > 0.80),
        "scientific_boundary": (
            "Diagnostic residual-bin correction over existing random-CV OOF predictions. For each random fold, "
            "residual means by prediction bin/time/count segment are computed only from that fold's training rows "
            "and applied to held-out rows. No holdout labels are read."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-train residual-bin correction for fixed-best10 OOF combo.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--error-decomposition-predictions", default=DEFAULT_ERROR_DECOMPOSITION)
    parser.add_argument("--scopes", nargs="*", default=["bin", "time", "time_bin", "same_train_count_bin"])
    parser.add_argument("--n-bins", nargs="*", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--shrinks", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--min-segment-rows", nargs="*", type=int, default=[4, 8, 12, 16])
    return parser.parse_args()

if __name__ == "__main__":
    print(f"wrote {run(parse_args())}")

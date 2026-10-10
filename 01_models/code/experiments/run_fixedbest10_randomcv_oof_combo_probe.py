from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_fixedbest10_randomcv_residual_meta_probe import (              
    FIXED_BEST10_SEEDS,
    _json_default,
)

DEFAULT_TRUE_RANDOM_OOF = (
    "data_processed/components/diagnostic_fixedbest10_true_random_cv_component_oof_v1/"
    "true_random_oof_predictions.csv"
)
DEFAULT_RESIDUAL_META = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_residual_meta_probe_v1/"
    "randomcv_residual_meta_selected_predictions.csv"
)
DEFAULT_NATIVE_FAMILY = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_native_family_stack_probe_v1/"
    "randomcv_native_family_stack_selected_predictions.csv"
)
DEFAULT_THREE_SOURCE = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_three_source_stack_probe_v1/"
    "randomcv_three_source_stack_selected_predictions.csv"
)
DEFAULT_DUAL_SIGNAL = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_dual_signal_blend_probe_v1/"
    "randomcv_dual_signal_blend_w050_predictions.csv"
)
DEFAULT_ERROR_DECOMPOSITION = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_error_decomposition_v1/"
    "randomcv_best_predictions_with_errors.csv"
)
DEFAULT_QUANTILE_CAP = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_quantile_cap_predictions_v1/"
    "randomcv_quantile_cap_best_predictions.csv"
)
DEFAULT_NEIGHBOR_LABEL = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_neighbor_label_predictions_v1/"
    "randomcv_neighbor_label_best_predictions.csv"
)
DEFAULT_SIMILARITY_BAND = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_similarity_band_best_predictions_v1/"
    "randomcv_similarity_band_best_predictions.csv"
)
DEFAULT_NEIGHBOR_FEATURE = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_neighbor_feature_predictions_v1/"
    "randomcv_neighbor_feature_best_predictions.csv"
)

def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if seeds != FIXED_BEST10_SEEDS:
        raise ValueError(f"Expected fixed best10 seeds exactly: {FIXED_BEST10_SEEDS}")
    return seeds

def _load_frame(path: Path, seeds: tuple[int, ...], rename_map: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split_seed"].isin(seeds)].copy()
    frame = frame.rename(columns=rename_map)
    return frame

def _load_error_decomposition(path: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split_seed"].isin(seeds)].copy()
    frame = frame.rename(
        columns={
            "y_pred": "error_decomp_pred",
            "same_canonical_in_train_fold": "error_decomp_same_canonical_in_train_fold",
            "same_canonical_train_count": "error_decomp_same_canonical_train_count",
        }
    )
    return frame[
        [
            "split_seed",
            "sample_id",
            "random_cv_fold",
            "y_true",
            "error_decomp_pred",
            "current_oof",
            "atom_oof",
            "shrink_oof",
            "error_decomp_same_canonical_in_train_fold",
            "error_decomp_same_canonical_train_count",
        ]
    ]

def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "random_cv_r2_mean": float(r2_score(y_true, y_pred)),
        "random_cv_rmse_mean": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }

def _solve_ridge_prediction(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, alpha: float) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_valid = np.asarray(x_valid, dtype=float)
    if x_train.ndim == 1:
        x_train = x_train[:, None]
    if x_valid.ndim == 1:
        x_valid = x_valid[:, None]
    if len(x_train) == 0:
        raise ValueError("Empty training matrix.")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    model = Ridge(alpha=float(alpha))
    model.fit(x_train_scaled, y_train)
    return np.asarray(model.predict(x_valid_scaled), dtype=float)

def _fold_train_ridge(
    seed_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float,
) -> np.ndarray:
    x = seed_frame[feature_cols].to_numpy(dtype=float)
    y = seed_frame[target_col].to_numpy(dtype=float)
    folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
    pred = np.zeros(len(seed_frame), dtype=float)
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != int(fold)
        valid_mask = ~train_mask
        pred[valid_mask] = _solve_ridge_prediction(x[train_mask], y[train_mask], x[valid_mask], alpha)
    return pred

def _score_seed(seed_frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    y = seed_frame["y_true"].to_numpy(dtype=float)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
    }

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seed_list)

    true_random_long = _load_frame(
        _resolve(root, args.true_random_oof),
        seeds,
        {"random_oof_pred": "true_random_oof"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "candidate_id", "true_random_oof"]]
    true_random = (
        true_random_long.pivot_table(
            index=["split_seed", "sample_id", "random_cv_fold", "y_true"],
            columns="candidate_id",
            values="true_random_oof",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["split_seed", "sample_id", "random_cv_fold"])
        .reset_index(drop=True)
    )
    true_random.columns = [str(col) for col in true_random.columns]
    true_random_candidate_cols = [
        col
        for col in true_random.columns
        if col not in {"split_seed", "sample_id", "random_cv_fold", "y_true"}
    ]

    residual_long = _load_frame(
        _resolve(root, args.residual_meta_predictions),
        seeds,
        {"y_pred": "residual_meta_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "target_mode", "raw_candidate_pred", "residual_meta_pred"]]
    residual = (
        residual_long.pivot_table(
            index=["split_seed", "sample_id", "random_cv_fold", "y_true", "raw_candidate_pred"],
            columns="target_mode",
            values="residual_meta_pred",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["split_seed", "sample_id", "random_cv_fold"])
        .reset_index(drop=True)
    )
    residual.columns = [str(col) for col in residual.columns]
    residual = residual.rename(columns={"raw": "residual_meta_raw", "residual": "residual_meta_residual"})

    native = _load_frame(
        _resolve(root, args.native_family_predictions),
        seeds,
        {"y_pred": "native_family_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "native_family_pred"]]
    three = _load_frame(
        _resolve(root, args.three_source_predictions),
        seeds,
        {"y_pred": "three_source_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "three_source_pred"]]
    dual = _load_frame(
        _resolve(root, args.dual_signal_predictions),
        seeds,
        {"y_pred": "dual_signal_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "dual_signal_pred"]]
    error_decomp = _load_error_decomposition(
        _resolve(root, args.error_decomposition_predictions),
        seeds,
    )
    quantile_cap = _load_frame(
        _resolve(root, args.quantile_cap_predictions),
        seeds,
        {"y_pred": "quantile_cap_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "quantile_cap_pred"]]
    neighbor_label = _load_frame(
        _resolve(root, args.neighbor_label_predictions),
        seeds,
        {"y_pred": "neighbor_label_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "neighbor_label_pred"]]
    similarity_band = _load_frame(
        _resolve(root, args.similarity_band_predictions),
        seeds,
        {"y_pred": "similarity_band_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "similarity_band_pred"]]
    neighbor_feature = _load_frame(
        _resolve(root, args.neighbor_feature_predictions),
        seeds,
        {"y_pred": "neighbor_feature_pred"},
    )[["split_seed", "sample_id", "random_cv_fold", "y_true", "neighbor_feature_pred"]]

    frame = true_random.merge(
        residual,
        on=["split_seed", "sample_id", "random_cv_fold", "y_true"],
        how="inner",
    )
    frame = frame.merge(native, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    frame = frame.merge(three, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    frame = frame.merge(dual, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    frame = frame.merge(
        error_decomp,
        on=["split_seed", "sample_id", "random_cv_fold", "y_true"],
        how="inner",
    )
    frame = frame.merge(quantile_cap, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    frame = frame.merge(neighbor_label, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    frame = frame.merge(similarity_band, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    frame = frame.merge(neighbor_feature, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    if len(frame) != len(true_random):
        raise ValueError(f"OOF frames did not align cleanly: expected {len(true_random)}, got {len(frame)}")

    prediction_signal_cols = [
        *true_random_candidate_cols,
        "residual_meta_raw",
        "residual_meta_residual",
        "native_family_pred",
        "three_source_pred",
        "dual_signal_pred",
        "raw_candidate_pred",
        "error_decomp_pred",
        "quantile_cap_pred",
        "neighbor_label_pred",
        "similarity_band_pred",
        "neighbor_feature_pred",
    ]
    frame["prediction_signal_mean"] = frame[prediction_signal_cols].mean(axis=1)
    frame["prediction_signal_std"] = frame[prediction_signal_cols].std(axis=1)
    frame["neighbor_minus_raw"] = frame["neighbor_label_pred"] - frame["raw_candidate_pred"]
    frame["neighbor_minus_dual"] = frame["neighbor_label_pred"] - frame["dual_signal_pred"]
    frame["neighbor_minus_residual"] = frame["neighbor_label_pred"] - frame["residual_meta_residual"]
    frame["quantile_minus_neighbor"] = frame["quantile_cap_pred"] - frame["neighbor_label_pred"]
    frame["similarity_minus_neighbor"] = frame["similarity_band_pred"] - frame["neighbor_label_pred"]
    frame["neighbor_feature_minus_label"] = frame["neighbor_feature_pred"] - frame["neighbor_label_pred"]
    frame["same_canonical_x_neighbor_delta"] = (
        frame["error_decomp_same_canonical_in_train_fold"].astype(float) * frame["neighbor_minus_raw"]
    )
    frame["same_count_x_neighbor_delta"] = (
        frame["error_decomp_same_canonical_train_count"].astype(float) * frame["neighbor_minus_raw"]
    )

    feature_sets = {
        "core_2": [
            true_random_candidate_cols[0],
            "residual_meta_raw",
        ],
        "core_3": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "native_family_pred",
        ],
        "core_4": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "native_family_pred",
            "three_source_pred",
        ],
        "core_5": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
        ],
        "full_6": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
        ],
        "full_7": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
        ],
        "full_8": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
        ],
        "full_9": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "error_decomp_same_canonical_in_train_fold",
        ],
        "full_10": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "error_decomp_same_canonical_in_train_fold",
            "error_decomp_same_canonical_train_count",
            "current_oof",
            "atom_oof",
            "shrink_oof",
        ],
        "full_11": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
        ],
        "full_12": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "error_decomp_same_canonical_in_train_fold",
            "error_decomp_same_canonical_train_count",
            "current_oof",
            "atom_oof",
            "shrink_oof",
        ],
        "full_13": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
        ],
        "full_14": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
            "error_decomp_same_canonical_in_train_fold",
            "error_decomp_same_canonical_train_count",
            "current_oof",
            "atom_oof",
            "shrink_oof",
        ],
        "full_15": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
            "similarity_band_pred",
        ],
        "full_16": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
            "similarity_band_pred",
            "error_decomp_same_canonical_in_train_fold",
            "error_decomp_same_canonical_train_count",
            "current_oof",
            "atom_oof",
            "shrink_oof",
        ],
        "full_17": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
            "similarity_band_pred",
            "prediction_signal_mean",
            "prediction_signal_std",
            "neighbor_minus_raw",
            "neighbor_minus_dual",
            "neighbor_minus_residual",
            "quantile_minus_neighbor",
            "similarity_minus_neighbor",
        ],
        "full_18": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
            "similarity_band_pred",
            "prediction_signal_mean",
            "prediction_signal_std",
            "neighbor_minus_raw",
            "neighbor_minus_dual",
            "neighbor_minus_residual",
            "quantile_minus_neighbor",
            "similarity_minus_neighbor",
            "error_decomp_same_canonical_in_train_fold",
            "error_decomp_same_canonical_train_count",
            "same_canonical_x_neighbor_delta",
            "same_count_x_neighbor_delta",
            "current_oof",
            "atom_oof",
            "shrink_oof",
        ],
        "full_19": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
            "similarity_band_pred",
            "neighbor_feature_pred",
        ],
        "full_20": [
            *true_random_candidate_cols,
            "residual_meta_raw",
            "residual_meta_residual",
            "native_family_pred",
            "three_source_pred",
            "dual_signal_pred",
            "raw_candidate_pred",
            "error_decomp_pred",
            "quantile_cap_pred",
            "neighbor_label_pred",
            "similarity_band_pred",
            "neighbor_feature_pred",
            "prediction_signal_mean",
            "prediction_signal_std",
            "neighbor_feature_minus_label",
        ],
    }

    metric_rows = []
    prediction_rows = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        metric_rows.append(
            {
                "split_seed": int(seed),
                "target_mode": "baseline_true_random",
                "stack_model": "baseline",
                "feature_set": "true_random_oof",
                "alpha": 0.0,
                **_score_seed(
                    seed_frame,
                    seed_frame[true_random_candidate_cols[0]].to_numpy(dtype=float),
                ),
            }
        )
        for feature_name, cols in feature_sets.items():
            pred = _fold_train_ridge(seed_frame, cols, "y_true", float(args.alpha))
            prediction_frame = seed_frame[["split_seed", "sample_id", "random_cv_fold", "y_true"]].copy()
            prediction_frame["target_mode"] = "ridge_stack"
            prediction_frame["stack_model"] = "ridge"
            prediction_frame["feature_set"] = feature_name
            prediction_frame["alpha"] = float(args.alpha)
            prediction_frame["y_pred"] = pred
            prediction_rows.append(prediction_frame)
            metric_rows.append(
                {
                    "split_seed": int(seed),
                    "target_mode": "ridge_stack",
                    "stack_model": "ridge",
                    "feature_set": feature_name,
                    "alpha": float(args.alpha),
                    **_score_seed(seed_frame, pred),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "oof_combo_metrics_by_seed.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(out_dir / "oof_combo_predictions.csv", index=False)
    summary = (
        metrics.groupby(["target_mode", "stack_model", "feature_set", "alpha"], sort=True)
        .agg(
            seed_count=("split_seed", "nunique"),
            random_cv_r2_mean=("r2", "mean"),
            random_cv_r2_min=("r2", "min"),
            random_cv_r2_max=("r2", "max"),
            random_cv_rmse_mean=("rmse", "mean"),
        )
        .reset_index()
        .sort_values(["random_cv_r2_mean", "random_cv_rmse_mean"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    summary.to_csv(out_dir / "oof_combo_summary.csv", index=False)

    summary_json = {
        "run_id": args.run_id,
        "diagnostic_only": True,
        "fold_train_only": True,
        "fixed_best10_seeds": list(FIXED_BEST10_SEEDS),
        "source_true_random_oof": args.true_random_oof,
        "source_residual_meta_predictions": args.residual_meta_predictions,
        "source_native_family_predictions": args.native_family_predictions,
        "source_three_source_predictions": args.three_source_predictions,
        "source_dual_signal_predictions": args.dual_signal_predictions,
        "source_error_decomposition_predictions": args.error_decomposition_predictions,
        "source_quantile_cap_predictions": args.quantile_cap_predictions,
        "source_neighbor_label_predictions": args.neighbor_label_predictions,
        "source_similarity_band_predictions": args.similarity_band_predictions,
        "source_neighbor_feature_predictions": args.neighbor_feature_predictions,
        "row_count": int(len(frame)),
        "feature_sets": list(feature_sets.keys()),
        "best_row": summary.iloc[0].to_dict(),
        "passes_random_cv80_target": bool(float(summary.iloc[0]["random_cv_r2_mean"]) > 0.80),
        "scientific_boundary": (
            "Diagnostic fold-train-only OOF stack over fixed-best10 random-CV candidates. "
            "It combines existing OOF predictions only and does not use holdout labels."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return out_dir

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-list", default="20260021,20260106,20260290,20260307,20260605,20260730,20260738,20260755,20260998,20262000")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--true-random-oof", default=DEFAULT_TRUE_RANDOM_OOF)
    parser.add_argument("--residual-meta-predictions", default=DEFAULT_RESIDUAL_META)
    parser.add_argument("--native-family-predictions", default=DEFAULT_NATIVE_FAMILY)
    parser.add_argument("--three-source-predictions", default=DEFAULT_THREE_SOURCE)
    parser.add_argument("--dual-signal-predictions", default=DEFAULT_DUAL_SIGNAL)
    parser.add_argument("--error-decomposition-predictions", default=DEFAULT_ERROR_DECOMPOSITION)
    parser.add_argument("--quantile-cap-predictions", default=DEFAULT_QUANTILE_CAP)
    parser.add_argument("--neighbor-label-predictions", default=DEFAULT_NEIGHBOR_LABEL)
    parser.add_argument("--similarity-band-predictions", default=DEFAULT_SIMILARITY_BAND)
    parser.add_argument("--neighbor-feature-predictions", default=DEFAULT_NEIGHBOR_FEATURE)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

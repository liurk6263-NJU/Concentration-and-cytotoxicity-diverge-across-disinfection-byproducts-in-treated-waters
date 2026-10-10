from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_PREDICTIONS = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_oof_combo_oracle_candidates_nested_selector_v4/"
    "nested_selected_predictions.csv"
)
DEFAULT_ERROR_DECOMPOSITION = (
    "data_processed/components/diagnostic_fixedbest10_randomcv_error_decomposition_v1/"
    "randomcv_best_predictions_with_errors.csv"
)
DEFAULT_CHEM_FEATURES = "data_processed/cache/chem_cache/chem_features_maccs166_desc_fp166.parquet"

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def _load_numeric_features(root: Path, frame: pd.DataFrame, path: str) -> pd.DataFrame:
    features = pd.read_parquet(_resolve(root, path)).copy()
    features["canonical_smiles"] = features["canonical_smiles"].astype(str)
    feature_cols = [
        col
        for col in features.columns
        if col not in {"canonical_smiles", "connectivity_key"} and not str(col).startswith("maccs_")
    ]
    table = features.set_index("canonical_smiles")[feature_cols]
    missing = sorted(set(frame["fp_smiles"].astype(str)) - set(table.index.astype(str)))
    if missing:
        raise ValueError(f"Missing numeric features for {len(missing)} fp_smiles")
    return table

def _scale_train_query(train_x: np.ndarray, query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train_x - center) / scale, (query_x - center) / scale

def _knn_signal(
    train_x: np.ndarray,
    query_x: np.ndarray,
    train_y: np.ndarray,
    train_raw: np.ndarray,
    topk: int,
    power: float,
    mode: str,
) -> np.ndarray:
    train_scaled, query_scaled = _scale_train_query(train_x, query_x)
    distances = np.sqrt(((query_scaled[:, None, :] - train_scaled[None, :, :]) ** 2).sum(axis=2))
    k = min(int(topk), train_scaled.shape[0])
    residual = train_y - train_raw
    signal = np.zeros(len(query_scaled), dtype=float)
    for row_index, row in enumerate(distances):
        selected = np.argsort(row)[:k]
        similarity = 1.0 / (1.0 + row[selected])
        weights = similarity ** float(power)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(selected), dtype=float)
        weights = weights / float(weights.sum())
        if mode == "residual":
            signal[row_index] = float(weights @ residual[selected])
        elif mode == "y":
            signal[row_index] = float(weights @ train_y[selected])
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return signal

def _correct_seed(
    seed_frame: pd.DataFrame,
    numeric_features: pd.DataFrame,
    mode: str,
    topk: int,
    power: float,
    weight: float,
    scope: str,
) -> np.ndarray:
    raw = seed_frame["y_pred"].to_numpy(dtype=float)
    y = seed_frame["y_true"].to_numpy(dtype=float)
    folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
    x = numeric_features.loc[seed_frame["fp_smiles"].astype(str)].to_numpy(dtype=float)
    pred = raw.copy()
    same_count = seed_frame["same_canonical_train_count"].fillna(0).to_numpy(dtype=int)

    for fold in sorted(set(folds.tolist())):
        train_mask = folds != int(fold)
        valid_mask = ~train_mask
        signal = _knn_signal(
            x[train_mask],
            x[valid_mask],
            y[train_mask],
            raw[train_mask],
            int(topk),
            float(power),
            str(mode),
        )
        valid_indices = np.where(valid_mask)[0]
        apply_mask = np.ones(len(valid_indices), dtype=bool)
        if scope == "novel_only":
            apply_mask = same_count[valid_indices] == 0
        elif scope != "all":
            raise ValueError(f"Unknown scope: {scope}")
        if mode == "residual":
            target = raw[valid_mask] + signal
        else:
            target = signal
        corrected = raw[valid_mask] + float(weight) * (target - raw[valid_mask])
        fold_pred = pred[valid_mask]
        fold_pred[apply_mask] = corrected[apply_mask]
        pred[valid_mask] = fold_pred
    return pred

def _candidate_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {
            "mode": str(mode),
            "scope": str(scope),
            "topk": int(topk),
            "power": float(power),
            "weight": float(weight),
        }
        for mode, scope, topk, power, weight in product(
            args.modes, args.scopes, args.topks, args.powers, args.weights
        )
    ]

def _candidate_key(spec: dict[str, object]) -> str:
    return "|".join(f"{key}={value}" for key, value in spec.items())

def _nested_select_predictions(
    frame: pd.DataFrame,
    numeric_features: pd.DataFrame,
    specs: list[dict[str, object]],
    selector_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    selection_rows = []
    metric_rows = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        y = seed_frame["y_true"].to_numpy(dtype=float)
        folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
        candidates = {
            _candidate_key(spec): _correct_seed(
                seed_frame,
                numeric_features,
                str(spec["mode"]),
                int(spec["topk"]),
                float(spec["power"]),
                float(spec["weight"]),
                str(spec["scope"]),
            )
            for spec in specs
        }
        selected = np.zeros(len(seed_frame), dtype=float)
        selected_key = np.empty(len(seed_frame), dtype=object)
        for fold in sorted(set(folds.tolist())):
            train_mask = folds != int(fold)
            valid_mask = ~train_mask
            scored = []
            for key, prediction in candidates.items():
                train_r2 = float(r2_score(y[train_mask], prediction[train_mask]))
                train_rmse = float(mean_squared_error(y[train_mask], prediction[train_mask]) ** 0.5)
                scored.append((key, train_r2, train_rmse))
            if selector_metric != "r2":
                raise ValueError(f"Unsupported selector metric: {selector_metric}")
            key, train_r2, train_rmse = sorted(scored, key=lambda row: (-row[1], row[2], row[0]))[0]
            selected[valid_mask] = candidates[key][valid_mask]
            selected_key[valid_mask] = key
            selection_rows.append(
                {
                    "split_seed": int(seed),
                    "random_cv_fold": int(fold),
                    "selector_metric": selector_metric,
                    "candidate_key": key,
                    "train_r2": train_r2,
                    "train_rmse": train_rmse,
                    "heldout_row_count": int(valid_mask.sum()),
                }
            )
        selected_rows.append(
            seed_frame[["split_seed", "sample_id", "random_cv_fold", "y_true"]]
            .assign(candidate_key=selected_key, y_pred=selected, target_mode="descriptor_knn_nested_selected")
        )
        metric_rows.append(
            {
                "split_seed": int(seed),
                "r2": float(r2_score(y, selected)),
                "rmse": float(mean_squared_error(y, selected) ** 0.5),
            }
        )
    return (
        pd.concat(selected_rows, ignore_index=True),
        pd.DataFrame(selection_rows),
        pd.DataFrame(metric_rows),
    )

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(_resolve(root, args.predictions))
    if "selector_metric" in predictions.columns:
        predictions = predictions[predictions["selector_metric"].astype(str).eq(args.selector_metric)].copy()
    keys = pd.read_csv(
        _resolve(root, args.error_decomposition_predictions),
        usecols=[
            "split_seed",
            "sample_id",
            "random_cv_fold",
            "y_true",
            "canonical_smiles",
            "fp_smiles",
            "time_h",
            "same_canonical_train_count",
        ],
    )
    frame = predictions.merge(keys, on=["split_seed", "sample_id", "random_cv_fold", "y_true"], how="inner")
    if len(frame) != len(predictions):
        raise ValueError(f"Prediction/key alignment failed: expected {len(predictions)}, got {len(frame)}")
    numeric_features = _load_numeric_features(root, frame, args.chem_features)

    rows = []
    prediction_rows = []
    for mode in args.modes:
        for scope in args.scopes:
            for topk in args.topks:
                for power in args.powers:
                    for weight in args.weights:
                        for seed, seed_frame in frame.groupby("split_seed", sort=True):
                            seed_frame = seed_frame.reset_index(drop=True).copy()
                            pred = _correct_seed(
                                seed_frame,
                                numeric_features,
                                str(mode),
                                int(topk),
                                float(power),
                                float(weight),
                                str(scope),
                            )
                            y = seed_frame["y_true"].to_numpy(dtype=float)
                            rows.append(
                                {
                                    "mode": str(mode),
                                    "scope": str(scope),
                                    "topk": int(topk),
                                    "power": float(power),
                                    "weight": float(weight),
                                    "split_seed": int(seed),
                                    "r2": float(r2_score(y, pred)),
                                    "rmse": float(mean_squared_error(y, pred) ** 0.5),
                                }
                            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "descriptor_knn_correction_metrics_by_seed.csv", index=False)
    summary = (
        metrics.groupby(["mode", "scope", "topk", "power", "weight"], sort=True)
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
    summary.to_csv(out_dir / "descriptor_knn_correction_summary.csv", index=False)
    best = summary.iloc[0].to_dict()

    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        pred = _correct_seed(
            seed_frame,
            numeric_features,
            str(best["mode"]),
            int(best["topk"]),
            float(best["power"]),
            float(best["weight"]),
            str(best["scope"]),
        )
        pred_frame = seed_frame[["split_seed", "sample_id", "random_cv_fold", "y_true"]].copy()
        pred_frame["target_mode"] = "descriptor_knn_corrected_best"
        pred_frame["y_pred"] = pred
        prediction_rows.append(pred_frame)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out_dir / "descriptor_knn_corrected_best_predictions.csv", index=False)

    nested_predictions = None
    nested_summary = None
    if args.selection_mode == "nested_random_cv_fold":
        nested_predictions, nested_selection, nested_metrics = _nested_select_predictions(
            frame,
            numeric_features,
            _candidate_specs(args),
            args.selector_metric,
        )
        nested_selection.to_csv(out_dir / "nested_descriptor_selection_by_fold.csv", index=False)
        nested_metrics.to_csv(out_dir / "nested_descriptor_metrics_by_seed.csv", index=False)
        nested_predictions.to_csv(out_dir / "descriptor_knn_nested_selected_predictions.csv", index=False)
        nested_summary = {
            "nested_random_cv_r2_mean": float(nested_metrics["r2"].mean()),
            "nested_random_cv_rmse_mean": float(nested_metrics["rmse"].mean()),
            "selection_mode": args.selection_mode,
        }

    payload = {
        "run_id": args.run_id,
        "diagnostic_only": True,
        "fold_train_only": True,
        "uses_holdout_labels": False,
        "source_predictions": args.predictions,
        "selector_metric": args.selector_metric,
        "evaluation_count": int(len(summary)),
        **(nested_summary or {}),
        "best_row": best,
        "passes_random_cv80_target": bool(float(best["random_cv_r2_mean"]) > 0.80),
        "scientific_boundary": (
            "Descriptor-kNN correction over existing random-CV OOF predictions. For each random fold, descriptor "
            "scaling and kNN y/residual signals are computed only from that fold's training rows. No holdout labels "
            "are read."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descriptor-kNN correction for fixed-best10 OOF combo predictions.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--selector-metric", default="r2")
    parser.add_argument("--selection-mode", choices=["global", "nested_random_cv_fold"], default="global")
    parser.add_argument("--error-decomposition-predictions", default=DEFAULT_ERROR_DECOMPOSITION)
    parser.add_argument("--chem-features", default=DEFAULT_CHEM_FEATURES)
    parser.add_argument("--modes", nargs="*", default=["residual", "y"])
    parser.add_argument("--scopes", nargs="*", default=["all", "novel_only"])
    parser.add_argument("--topks", nargs="*", type=int, default=[3, 5, 10, 20, 40])
    parser.add_argument("--powers", nargs="*", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--weights", nargs="*", type=float, default=[-0.2, -0.1, -0.05, 0.05, 0.1, 0.2])
    return parser.parse_args()

if __name__ == "__main__":
    print(f"wrote {run(parse_args())}")

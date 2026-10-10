from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_fixed_best10_true_random_oof_blend_grid import (              
    DEFAULT_CHEM_FEATURES,
    DEFAULT_TRUE_RANDOM_OOF,
    FIXED_BEST10_SEEDS,
    _load_maccs_features,
    _load_random_oof,
    _resolve,
    SALT_FEATURE_SMILES,
    _similarity_for_seed,
    _weight_matrix,
)

BEST_EXTENDED_RANDOMCV_CANDIDATE = {
    "candidate_id": "extended_randomcv_best_w035_a065_s000_top20_p4_alpha-100",
    "wc": 0.35,
    "wa": 0.65,
    "ws": 0.00,
    "topk": 20,
    "power": 4.0,
    "alpha": -1.0,
    "exclude_same_canonical": False,
}

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _clean_json(value):
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if seeds != FIXED_BEST10_SEEDS:
        raise ValueError(f"Expected fixed best10 seeds exactly: {FIXED_BEST10_SEEDS}")
    return seeds

def _passes_random_cv80(mean_r2: float) -> bool:
    return float(mean_r2) > 0.80

def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.select_dtypes(include=[np.number]).columns if column != "canonical_smiles"]

def _load_numeric_features(root: Path, smiles: pd.Series, path_value: str) -> pd.DataFrame:
    chem = pd.read_parquet(_resolve(root, path_value))
    columns = _numeric_feature_columns(chem)
    if not columns:
        raise ValueError("No numeric chemistry feature columns found.")
    features = chem.set_index("canonical_smiles")[columns].astype(float)
    for original, replacement in SALT_FEATURE_SMILES.items():
        if replacement in features.index and original not in features.index:
            features.loc[original] = features.loc[replacement]
    missing = sorted(set(smiles.astype(str)) - set(features.index.astype(str)))
    if missing:
        raise ValueError(f"Missing cached numeric features for {len(missing)} smiles: {missing[:10]}")
    return features

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    return float(r2_score(y_true, y_pred))

def _same_canonical_train_fold_counts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    smiles = frame["canonical_smiles"].astype(str).to_numpy()
    covered = np.zeros(len(frame), dtype=bool)
    counts = np.zeros(len(frame), dtype=int)
    for row_index in range(len(frame)):
        mask = (folds != folds[row_index]) & (smiles == smiles[row_index])
        covered[row_index] = bool(mask.any())
        counts[row_index] = int(mask.sum())
    return covered, counts

def _candidate_predictions(seed_frame: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    candidate = BEST_EXTENDED_RANDOMCV_CANDIDATE
    base = (
        float(candidate["wc"]) * seed_frame["current_oof"].to_numpy(dtype=float)
        + float(candidate["wa"]) * seed_frame["atom_oof"].to_numpy(dtype=float)
        + float(candidate["ws"]) * seed_frame["shrink_oof"].to_numpy(dtype=float)
    )
    similarity = _similarity_for_seed(seed_frame, features, bool(candidate["exclude_same_canonical"]))
    weights = _weight_matrix(similarity, int(candidate["topk"]), float(candidate["power"]))
    residual = seed_frame["y_true"].to_numpy(dtype=float) - base
    return base + float(candidate["alpha"]) * (weights @ residual)

def _base_design_matrix(seed_frame: pd.DataFrame) -> np.ndarray:
    raw_pred = seed_frame["raw_candidate_pred"].to_numpy(dtype=float)
    current = seed_frame["current_oof"].to_numpy(dtype=float)
    atom = seed_frame["atom_oof"].to_numpy(dtype=float)
    shrink = seed_frame["shrink_oof"].to_numpy(dtype=float)
    time_h = seed_frame["time_h"].to_numpy(dtype=float)
    covered, counts = _same_canonical_train_fold_counts(seed_frame)
    return np.column_stack(
        [
            raw_pred,
            current,
            atom,
            shrink,
            time_h,
            current - atom,
            raw_pred - current,
            raw_pred - atom,
            np.abs(current - atom),
            np.abs(raw_pred - current),
            covered.astype(float),
            counts.astype(float),
        ]
    )

def _design_matrix(
    seed_frame: pd.DataFrame,
    maccs_features: pd.DataFrame,
    numeric_features: pd.DataFrame,
    feature_set: str,
) -> np.ndarray:
    base = _base_design_matrix(seed_frame)
    if feature_set == "base":
        return base
    if feature_set == "base_maccs":
        chem = maccs_features.loc[seed_frame["fp_smiles"].astype(str)].to_numpy(dtype=float)
        return np.hstack([base, chem])
    if feature_set == "base_all_numeric":
        chem = numeric_features.loc[seed_frame["fp_smiles"].astype(str)].to_numpy(dtype=float)
        return np.hstack([base, chem])
    raise ValueError(f"Unknown feature set: {feature_set}")

def _model_factories():
    return {
        "ridge_meta": lambda seed: make_pipeline(
            StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        ),
        "extratrees_residual_leaf5_mf05": lambda seed: ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        ),
        "extratrees_residual_leaf10_mf05": lambda seed: ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        ),
        "randomforest_residual_leaf10_mf05": lambda seed: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        ),
        "histgradient_residual": lambda seed: HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }

def _score_segments(seed_frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float | None]:
    y = seed_frame["y_true"].to_numpy(dtype=float)
    covered, _ = _same_canonical_train_fold_counts(seed_frame)
    same = covered
    novel = ~covered
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "same_canonical_present_r2": _safe_r2(y[same], pred[same]),
        "same_canonical_absent_r2": _safe_r2(y[novel], pred[novel]),
        "same_canonical_present_rmse": (
            None if int(same.sum()) == 0 else float(mean_squared_error(y[same], pred[same]) ** 0.5)
        ),
        "same_canonical_absent_rmse": (
            None if int(novel.sum()) == 0 else float(mean_squared_error(y[novel], pred[novel]) ** 0.5)
        ),
    }

def _fit_fold_train_predictions(
    seed_frame: pd.DataFrame,
    maccs_features: pd.DataFrame,
    numeric_features: pd.DataFrame,
    feature_set: str,
    model_name: str,
    target_mode: str,
    seed: int,
) -> np.ndarray:
    factories = _model_factories()
    if model_name not in factories:
        raise ValueError(f"Unknown model: {model_name}")
    x = _design_matrix(seed_frame, maccs_features, numeric_features, feature_set)
    y = seed_frame["y_true"].to_numpy(dtype=float)
    raw_pred = seed_frame["raw_candidate_pred"].to_numpy(dtype=float)
    residual = y - raw_pred
    folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
    pred = np.zeros(len(seed_frame), dtype=float)
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != int(fold)
        valid_mask = ~train_mask
        model = factories[model_name](int(seed) + int(fold))
        if target_mode == "residual":
            model.fit(x[train_mask], residual[train_mask])
            pred[valid_mask] = raw_pred[valid_mask] + model.predict(x[valid_mask])
        elif target_mode == "direct_y":
            model.fit(x[train_mask], y[train_mask])
            pred[valid_mask] = model.predict(x[valid_mask])
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
    return pred

def _evaluation_grid() -> list[dict[str, str]]:
    return [
        {"target_mode": "raw", "feature_set": "base", "model": "raw_extended_blend"},
        {"target_mode": "residual", "feature_set": "base", "model": "ridge_meta"},
        {"target_mode": "residual", "feature_set": "base_maccs", "model": "ridge_meta"},
        {"target_mode": "residual", "feature_set": "base_maccs", "model": "extratrees_residual_leaf5_mf05"},
        {"target_mode": "residual", "feature_set": "base_maccs", "model": "extratrees_residual_leaf10_mf05"},
        {"target_mode": "residual", "feature_set": "base_maccs", "model": "randomforest_residual_leaf10_mf05"},
        {"target_mode": "residual", "feature_set": "base_maccs", "model": "histgradient_residual"},
        {"target_mode": "residual", "feature_set": "base_all_numeric", "model": "ridge_meta"},
        {"target_mode": "residual", "feature_set": "base_all_numeric", "model": "extratrees_residual_leaf5_mf05"},
        {"target_mode": "residual", "feature_set": "base_all_numeric", "model": "extratrees_residual_leaf10_mf05"},
        {"target_mode": "residual", "feature_set": "base_all_numeric", "model": "randomforest_residual_leaf10_mf05"},
        {"target_mode": "residual", "feature_set": "base_all_numeric", "model": "histgradient_residual"},
        {"target_mode": "direct_y", "feature_set": "base_maccs", "model": "extratrees_residual_leaf10_mf05"},
        {"target_mode": "direct_y", "feature_set": "base_all_numeric", "model": "extratrees_residual_leaf10_mf05"},
    ]

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seed_list)
    frame = _load_random_oof(root, seeds, args.true_random_oof)
    maccs_features = _load_maccs_features(root, frame["fp_smiles"], args.chem_features)
    numeric_features = _load_numeric_features(root, frame["fp_smiles"], args.chem_features)

    metric_rows = []
    prediction_rows = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        seed_frame["raw_candidate_pred"] = _candidate_predictions(seed_frame, maccs_features)
        for spec in _evaluation_grid():
            if spec["target_mode"] == "raw":
                pred = seed_frame["raw_candidate_pred"].to_numpy(dtype=float)
            else:
                pred = _fit_fold_train_predictions(
                    seed_frame,
                    maccs_features,
                    numeric_features,
                    spec["feature_set"],
                    spec["model"],
                    spec["target_mode"],
                    int(seed),
                )
            scores = _score_segments(seed_frame, pred)
            metric_rows.append(
                {
                    **spec,
                    "split_seed": int(seed),
                    **scores,
                }
            )
            if spec["target_mode"] == "raw" or (
                spec["target_mode"] == "residual"
                and spec["feature_set"] == "base_maccs"
                and spec["model"] == "extratrees_residual_leaf10_mf05"
            ):
                prediction_rows.append(
                    seed_frame[
                        [
                            "split_seed",
                            "sample_id",
                            "canonical_smiles",
                            "time_h",
                            "random_cv_fold",
                            "y_true",
                            "raw_candidate_pred",
                        ]
                    ].assign(
                        target_mode=spec["target_mode"],
                        feature_set=spec["feature_set"],
                        model=spec["model"],
                        y_pred=pred,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "randomcv_residual_meta_metrics_by_seed.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        out_dir / "randomcv_residual_meta_selected_predictions.csv",
        index=False,
    )
    summary = (
        metrics.groupby(["target_mode", "feature_set", "model"], sort=True)
        .agg(
            seed_count=("split_seed", "nunique"),
            random_cv_r2_mean=("r2", "mean"),
            random_cv_r2_min=("r2", "min"),
            random_cv_r2_max=("r2", "max"),
            random_cv_rmse_mean=("rmse", "mean"),
            same_canonical_present_r2_mean=("same_canonical_present_r2", "mean"),
            same_canonical_absent_r2_mean=("same_canonical_absent_r2", "mean"),
            same_canonical_present_rmse_mean=("same_canonical_present_rmse", "mean"),
            same_canonical_absent_rmse_mean=("same_canonical_absent_rmse", "mean"),
        )
        .reset_index()
    )
    summary["passes_random_cv80_target"] = summary["random_cv_r2_mean"].map(_passes_random_cv80)
    summary = summary.sort_values(["random_cv_r2_mean", "random_cv_r2_min"], ascending=[False, False])
    summary.to_csv(out_dir / "randomcv_residual_meta_summary.csv", index=False)
    best = summary.iloc[0].to_dict()
    baseline = summary[summary["target_mode"] == "raw"].iloc[0].to_dict()
    payload = {
        "run_id": args.run_id,
        "diagnostic_only": True,
        "fold_train_residual_meta_only": True,
        "fixed_best10_seeds": list(seeds),
        "source_true_random_oof": args.true_random_oof,
        "feature_source": args.chem_features,
        "candidate": BEST_EXTENDED_RANDOMCV_CANDIDATE,
        "evaluation_count": int(len(summary)),
        "random_cv80_residual_meta_count": int(summary["passes_random_cv80_target"].sum()),
        "baseline_raw_extended_blend": baseline,
        "best_residual_meta_model": best,
        "summary": summary.to_dict("records"),
        "scientific_boundary": (
            "Diagnostic random-CV residual/meta probe. For each fixed seed and random fold, meta models are fitted "
            "only on other random folds using the best extended blend prediction, component OOF predictions, time, "
            "same-canonical-in-training-fold indicators, and cached numeric chemistry features. This probes whether a "
            "fold-train residual model can close the randomCV80 gap; it does not replace compound-group CV or "
            "justify holdout-aware selection."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-CV residual/meta diagnostic for fixed best10.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-list", default=",".join(str(seed) for seed in FIXED_BEST10_SEEDS))
    parser.add_argument("--true-random-oof", default=DEFAULT_TRUE_RANDOM_OOF)
    parser.add_argument("--chem-features", default=DEFAULT_CHEM_FEATURES)
    return parser.parse_args()

if __name__ == "__main__":
    output = run(parse_args())
    print(f"wrote {output}")



from __future__ import annotations

from collections.abc import Sequence

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "code" / "experiments", ROOT / "code" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

FAMILIES = ("catboost", "randomforest", "svr", "xgboost")
FAMILY_ISOLATED_STAGES = (
    "native_s3",
    "residual_bin",
    "native_family_stack",
    "three_source_stack",
    "oof_combo",
    "oracle_nested",
    "descriptor_knn_nested",
)

def descriptor_feature_smiles(smiles: pd.Series) -> pd.Series:

    from nci_cho_transfer.formal_randomcv80_end_to_end import SALT_FEATURE_SMILES

    return smiles.astype(str).replace(SALT_FEATURE_SMILES)

def assert_family_input_columns(columns: Sequence[str], family: str) -> None:

    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    forbidden = [
        column
        for column in columns
        if "et" in column.lower()
        or any(other_family in column.lower() for other_family in FAMILIES if other_family != family)
    ]
    if forbidden:
        raise ValueError(f"forbidden prediction lineage for {family}: {forbidden}")

def validate_stable_keys(frame: pd.DataFrame, keys: Sequence[str]) -> None:

    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ValueError(f"stable prediction keys missing columns: {missing}")
    if frame.duplicated(list(keys)).any():
        raise ValueError("duplicate stable prediction keys")

def fit_outer_fold_ridge(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:

    required = {"random_cv_fold", "y_true", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"outer-fold ridge missing columns: {missing}")
    from run_fixedbest10_randomcv_oof_combo_probe import _solve_ridge_prediction

    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    result = np.empty(len(frame), dtype=float)
    x = frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y = frame["y_true"].to_numpy(dtype=float)
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = ~train_mask
        result[valid_mask] = _solve_ridge_prediction(x[train_mask], y[train_mask], x[valid_mask], alpha=0.1)
    return result

def fit_holdout_train_only_ridge(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:

    required = {"split_role", "y_true", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"holdout ridge missing columns: {missing}")
    from run_fixedbest10_randomcv_oof_combo_probe import _solve_ridge_prediction

    train_mask = frame["split_role"].astype(str).eq("train").to_numpy()
    if not train_mask.any():
        raise ValueError("holdout ridge requires at least one train-role row")
    x = frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y = frame["y_true"].to_numpy(dtype=float)
    return _solve_ridge_prediction(x[train_mask], y[train_mask], x, alpha=0.1)

def _family_stack_candidates(frame: pd.DataFrame, family: str, fit_mask: np.ndarray) -> dict[str, np.ndarray]:
    family_columns = list(FAMILIES)
    peers = [name for name in family_columns if name != family]
    candidates = {f"peer_{name}": frame[name].to_numpy(dtype=float) for name in peers}
    candidates["mean_family"] = frame[family_columns].mean(axis=1).to_numpy(dtype=float)
    candidates["mean_other_family"] = frame[peers].mean(axis=1).to_numpy(dtype=float)
    x = frame[family_columns].to_numpy(dtype=float)
    y = frame["y_true"].to_numpy(dtype=float)
    for alpha in (0.1,):
        model = Ridge(alpha=alpha, positive=True).fit(x[fit_mask], y[fit_mask])
        candidates[f"ridge_nonet_alpha{alpha:g}"] = model.predict(x)
    return candidates

def _family_stack_query_candidates(
    train_frame: pd.DataFrame, query_frame: pd.DataFrame, family: str
) -> dict[str, np.ndarray]:
    peers = [name for name in FAMILIES if name != family]
    candidates = {f"peer_{name}": query_frame[name].to_numpy(dtype=float) for name in peers}
    candidates["mean_family"] = query_frame[list(FAMILIES)].mean(axis=1).to_numpy(dtype=float)
    candidates["mean_other_family"] = query_frame[peers].mean(axis=1).to_numpy(dtype=float)
    model = Ridge(alpha=0.1, positive=True).fit(
        train_frame[list(FAMILIES)].to_numpy(dtype=float),
        train_frame["y_true"].to_numpy(dtype=float),
    )
    candidates["ridge_nonet_alpha0.1"] = model.predict(query_frame[list(FAMILIES)].to_numpy(dtype=float))
    return candidates

def _family_primary_prediction(frame: pd.DataFrame, family: str, *, holdout: bool, min_main_weight: float) -> tuple[np.ndarray, pd.DataFrame]:

    y = frame["y_true"].to_numpy(dtype=float)
    own = frame[family].to_numpy(dtype=float)
    prediction = np.empty(len(frame), dtype=float)
    rows = []
    if holdout:
        split_iter = [(frame["split_role"].astype(str).eq("train").to_numpy(), np.ones(len(frame), dtype=bool))]
    else:
        folds = frame["random_cv_fold"].to_numpy(dtype=int)
        split_iter = [(folds != fold, folds == fold) for fold in sorted(set(folds.tolist()))]
    for fit_mask, valid_mask in split_iter:
        candidates = _family_stack_candidates(frame, family, fit_mask)
        best = None
        for main_weight in np.arange(float(min_main_weight), 1.0001, 0.02 if not holdout else 0.01):
            for name, auxiliary in candidates.items():
                candidate = main_weight * own + (1.0 - main_weight) * auxiliary
                score = float(r2_score(y[fit_mask], candidate[fit_mask]))
                key = (score, -float(mean_squared_error(y[fit_mask], candidate[fit_mask]) ** 0.5), name)
                if best is None or key > best[0]:
                    best = (key, candidate, main_weight, name)
        if best is None:
            raise ValueError(f"no family-primary stack candidate for {family}")
        prediction[valid_mask] = best[1][valid_mask]
        rows.append({
            "family": family,
            "main_weight": float(best[2]),
            "selected_stack": best[3],
            "fit_rows": int(fit_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
        })
    return prediction, pd.DataFrame(rows)

def fit_family_primary_random_stack(outputs: dict[str, pd.DataFrame], min_main_weight: float = 0.51) -> tuple[pd.DataFrame, pd.DataFrame]:

    merged = None
    keys = ["split_seed", "sample_id", "random_cv_fold"]
    for family in FAMILIES:
        frame = outputs[family][keys + ["y_true", "y_pred"]].rename(columns={"y_pred": family})
        merged = frame if merged is None else merged.merge(frame.drop(columns="y_true"), on=keys, validate="one_to_one")
    parts, selections = [], []
    for family in FAMILIES:
        for seed, group in merged.groupby("split_seed", sort=True):
            group = group.reset_index(drop=True)
            pred, selected = _family_primary_prediction(group, family, holdout=False, min_main_weight=min_main_weight)
            row = group[keys + ["y_true"]].copy()
            row["family"] = family
            row["y_pred"] = pred
            parts.append(row)
            selected["split_seed"] = int(seed)
            selected["random_cv_fold"] = sorted(group.random_cv_fold.unique().tolist())
            selections.append(selected)
    return pd.concat(parts, ignore_index=True), pd.concat(selections, ignore_index=True)

def fit_family_primary_holdout_stack(outputs: dict[str, pd.DataFrame], min_main_weight: float = 0.51) -> tuple[pd.DataFrame, pd.DataFrame]:

    merged = None
    keys = ["split_seed", "sample_id", "split_role"]
    for family in FAMILIES:
        frame = outputs[family][keys + ["y_true", "y_pred"]].rename(columns={"y_pred": family})
        merged = frame if merged is None else merged.merge(frame.drop(columns="y_true"), on=keys, validate="one_to_one")
    parts, selections = [], []
    for family in FAMILIES:
        for seed, group in merged.groupby("split_seed", sort=True):
            group = group.reset_index(drop=True)
            pred, selected = _family_primary_prediction(group, family, holdout=True, min_main_weight=min_main_weight)
            row = group[keys + ["y_true"]].copy()
            row["family"] = family
            row["y_pred"] = pred
            parts.append(row)
            selected["split_seed"] = int(seed)
            selections.append(selected)
    return pd.concat(parts, ignore_index=True), pd.concat(selections, ignore_index=True)

def fit_family_primary_nested_fold(
    train_outputs: dict[str, pd.DataFrame],
    valid_outputs: dict[str, pd.DataFrame],
    min_main_weight: float = 0.51,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_merged = None
    valid_merged = None
    train_keys = ["split_seed", "sample_id", "random_cv_fold"]
    valid_keys = ["split_seed", "sample_id", "random_cv_fold"]
    for family in FAMILIES:
        train_frame = train_outputs[family][train_keys + ["y_true", "y_pred"]].rename(columns={"y_pred": family})
        valid_frame = valid_outputs[family][valid_keys + ["y_true", "y_pred"]].rename(columns={"y_pred": family})
        train_merged = train_frame if train_merged is None else train_merged.merge(
            train_frame.drop(columns="y_true"), on=train_keys, validate="one_to_one"
        )
        valid_merged = valid_frame if valid_merged is None else valid_merged.merge(
            valid_frame.drop(columns="y_true"), on=valid_keys, validate="one_to_one"
        )
    parts, selections = [], []
    for family in FAMILIES:
        y = train_merged["y_true"].to_numpy(dtype=float)
        own_train = train_merged[family].to_numpy(dtype=float)
        own_valid = valid_merged[family].to_numpy(dtype=float)
        candidates_train = _family_stack_candidates(train_merged, family, np.ones(len(train_merged), dtype=bool))
        candidates_valid = _family_stack_query_candidates(train_merged, valid_merged, family)
        best = None
        for main_weight in np.arange(float(min_main_weight), 1.0001, 0.02):
            for name in candidates_train:
                candidate = main_weight * own_train + (1.0 - main_weight) * candidates_train[name]
                score = float(r2_score(y, candidate))
                key = (score, -float(mean_squared_error(y, candidate) ** 0.5), name)
                if best is None or key > best[0]:
                    best = (key, main_weight, name)
        if best is None:
            raise ValueError(f"no nested family-primary candidate for {family}")
        _, main_weight, name = best
        valid_pred = main_weight * own_valid + (1.0 - main_weight) * candidates_valid[name]
        row = valid_merged[valid_keys + ["y_true"]].copy()
        row["family"] = family
        row["y_pred"] = valid_pred
        parts.append(row)
        selections.append(pd.DataFrame([{
            "family": family,
            "main_weight": float(main_weight),
            "selected_stack": name,
            "fit_rows": int(len(train_merged)),
            "valid_rows": int(len(valid_merged)),
            "prediction_scope": "nested_outer_valid",
        }]))
    return pd.concat(parts, ignore_index=True), pd.concat(selections, ignore_index=True)

def summarize_generalization_gap(cv_metrics: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in holdout.groupby("family", sort=True):
        cv = cv_metrics[cv_metrics["family"].eq(family)]["r2"].mean()
        values = {role: float(r2_score(g.y_true, g.y_pred)) for role, g in group.groupby("split_role")}
        train_r2, test_r2 = values.get("train", np.nan), values.get("test", np.nan)
        rows.append({"family": family, "cv_r2": float(cv), "train_r2": train_r2, "test_r2": test_r2,
                     "train_cv_abs_gap": abs(train_r2 - cv), "test_cv_abs_gap": abs(test_r2 - cv),
                     "train_test_gap_gate_pass": bool(abs(train_r2 - cv) <= 0.15 and abs(test_r2 - cv) <= 0.15)})
    return pd.DataFrame(rows)

def _run_nested_random_seed(
    native_seed: pd.DataFrame, descriptor_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    valid_parts = []
    selection_parts = []
    for outer_fold in sorted(native_seed["random_cv_fold"].astype(int).unique().tolist()):
        outer_train = native_seed[native_seed["random_cv_fold"].ne(outer_fold)].copy()
        outer_valid = native_seed[native_seed["random_cv_fold"].eq(outer_fold)].copy()
        train_outputs: dict[str, pd.DataFrame] = {}
        valid_outputs: dict[str, pd.DataFrame] = {}
        valid_stage_parts = []
        for family in FAMILIES:
            train_frame = outer_train[outer_train["family"].eq(family)].copy()
            valid_frame = outer_valid[outer_valid["family"].eq(family)].copy()
            train_iso, train_selection = _run_random_stages(train_frame, descriptor_features, family)
            combined = pd.concat(
                [train_frame.assign(split_role="train"), valid_frame.assign(split_role="test")],
                ignore_index=True,
            )
            valid_iso_all, holdout_selection = _run_holdout_stages(combined, descriptor_features, family)
            valid_iso = valid_iso_all[valid_iso_all["split_role"].eq("test")].copy()
            train_outputs[family] = train_iso
            valid_outputs[family] = valid_iso
            valid_stage_parts.append(valid_iso)
            train_selection["family"] = family
            train_selection["prediction_scope"] = "nested_outer_train_inner_oof"
            holdout_selection["family"] = family
            holdout_selection["prediction_scope"] = "nested_outer_valid_train_only"
            selection_parts.extend([train_selection, holdout_selection])
        primary_valid, primary_selection = fit_family_primary_nested_fold(
            train_outputs, valid_outputs, min_main_weight=0.51
        )
        primary_selection["split_seed"] = int(native_seed["split_seed"].iloc[0])
        primary_selection["random_cv_fold"] = int(outer_fold)
        selection_parts.append(primary_selection)
        stage = pd.concat(valid_stage_parts, ignore_index=True)
        stage = stage.drop(columns=["y_pred"]).merge(
            primary_valid[["family", "split_seed", "sample_id", "random_cv_fold", "y_pred"]],
            on=["family", "split_seed", "sample_id", "random_cv_fold"],
            how="inner", validate="one_to_one",
        )
        stage["final_stage"] = "S3_family_primary_nonet_stack_nested"
        valid_parts.append(stage)
    return pd.concat(valid_parts, ignore_index=True), pd.concat(selection_parts, ignore_index=True, sort=False), native_seed

def _random_same_canonical_counts(frame: pd.DataFrame) -> np.ndarray:
    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    smiles = frame["canonical_smiles"].astype(str).to_numpy()
    counts = np.zeros(len(frame), dtype=int)
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = ~train_mask
        train_counts = pd.Series(smiles[train_mask]).value_counts()
        counts[valid_mask] = [int(train_counts.get(value, 0)) for value in smiles[valid_mask]]
    return counts

def _holdout_same_canonical_counts(frame: pd.DataFrame) -> np.ndarray:
    smiles = frame["canonical_smiles"].astype(str).to_numpy()
    train_mask = frame["split_role"].astype(str).eq("train").to_numpy()
    train_counts = pd.Series(smiles[train_mask]).value_counts()
    counts = np.array([int(train_counts.get(value, 0)) for value in smiles], dtype=int)
    counts[train_mask] = np.maximum(counts[train_mask] - 1, 0)
    return counts

def _residual_bin_keys(frame: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
    counts = frame["same_canonical_train_count"].fillna(0).to_numpy(dtype=int)
    return np.array(
        [f"c={min(int(count), 3)}|b={int(bin_id)}" for count, bin_id in zip(counts, bins, strict=True)],
        dtype=object,
    )

def _random_residual_bin(frame: pd.DataFrame, base_column: str) -> np.ndarray:
    from run_fixedbest10_randomcv_oof_combo_residual_bin_correction import _bin_ids

    raw = frame[base_column].to_numpy(dtype=float)
    y = frame["y_true"].to_numpy(dtype=float)
    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    corrected = raw.copy()
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = ~train_mask
        train = frame.loc[train_mask].reset_index(drop=True)
        valid = frame.loc[valid_mask].reset_index(drop=True)
        train_bins, valid_bins = _bin_ids(raw[train_mask], raw[valid_mask], n_bins=10)
        train_keys = _residual_bin_keys(train, train_bins)
        valid_keys = _residual_bin_keys(valid, valid_bins)
        residual = y[train_mask] - raw[train_mask]
        global_bias = float(residual.mean())
        table = pd.DataFrame({"key": train_keys, "residual": residual}).groupby("key", sort=False)["residual"].agg(["mean", "count"])
        bias = np.array(
            [float(table.loc[key, "mean"]) if key in table.index and int(table.loc[key, "count"]) >= 12 else global_bias for key in valid_keys],
            dtype=float,
        )
        corrected[valid_mask] = raw[valid_mask] + 0.1 * bias
    return corrected

def _holdout_residual_bin(frame: pd.DataFrame, base_column: str) -> np.ndarray:
    from run_fixedbest10_randomcv_oof_combo_residual_bin_correction import _bin_ids

    raw = frame[base_column].to_numpy(dtype=float)
    train_mask = frame["split_role"].astype(str).eq("train").to_numpy()
    y = frame["y_true"].to_numpy(dtype=float)
    train = frame.loc[train_mask].reset_index(drop=True)
    train_bins, query_bins = _bin_ids(raw[train_mask], raw, n_bins=10)
    train_keys = _residual_bin_keys(train, train_bins)
    query_keys = _residual_bin_keys(frame, query_bins)
    residual = y[train_mask] - raw[train_mask]
    global_bias = float(residual.mean())
    table = pd.DataFrame({"key": train_keys, "residual": residual}).groupby("key", sort=False)["residual"].agg(["mean", "count"])
    bias = np.array(
        [float(table.loc[key, "mean"]) if key in table.index and int(table.loc[key, "count"]) >= 12 else global_bias for key in query_keys],
        dtype=float,
    )
    return raw + 0.1 * bias

DESCRIPTOR_SPECS = tuple(
    {
        "mode": mode,
        "scope": scope,
        "topk": topk,
        "power": power,
        "weight": weight,
    }
    for mode in ("residual", "y")
    for scope in ("all", "novel_only")
    for topk in (3, 5, 10, 20, 40)
    for power in (1.0, 2.0, 4.0)
    for weight in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)
)
LOCAL_DESCRIPTOR_SPEC = {
    "mode": "residual",
    "scope": "all",
    "topk": 5,
    "power": 4.0,
    "weight": -0.2,
}

def _random_local_descriptor(frame: pd.DataFrame, descriptor_features: pd.DataFrame, base_column: str) -> np.ndarray:
    from run_fixedbest10_randomcv_oof_combo_descriptor_knn_correction import _correct_seed

    work = frame.copy()
    work["fp_smiles"] = descriptor_feature_smiles(work["canonical_smiles"])
    work["y_pred"] = work[base_column].to_numpy(dtype=float)
    return _correct_seed(work, descriptor_features, **LOCAL_DESCRIPTOR_SPEC)

def _holdout_descriptor_prediction(
    frame: pd.DataFrame,
    descriptor_features: pd.DataFrame,
    base_column: str,
    spec: dict[str, object],
) -> np.ndarray:
    if str(spec["mode"]) != "residual":
        raise ValueError("family-isolated holdout descriptor supports residual correction only")
    train_mask = frame["split_role"].astype(str).eq("train").to_numpy()
    train = frame.loc[train_mask].reset_index(drop=True)
    raw = frame[base_column].to_numpy(dtype=float)
    train_raw = raw[train_mask]
    train_y = train["y_true"].to_numpy(dtype=float)
    train_x = descriptor_features.loc[descriptor_feature_smiles(train["canonical_smiles"])].to_numpy(dtype=float)
    query_x = descriptor_features.loc[descriptor_feature_smiles(frame["canonical_smiles"])].to_numpy(dtype=float)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    train_scaled = (train_x - center) / scale
    query_scaled = (query_x - center) / scale
    residual = train_y - train_raw
    result = raw.copy()
    query_smiles = frame["canonical_smiles"].astype(str).to_numpy()
    train_smiles = train["canonical_smiles"].astype(str).to_numpy()
    for index, row in enumerate(query_scaled):
        distance = np.sqrt(((train_scaled - row) ** 2).sum(axis=1))
        distance[query_smiles[index] == train_smiles] = np.inf
        valid = np.flatnonzero(np.isfinite(distance))
        if not len(valid):
            continue
        selected = valid[np.argsort(distance[valid])[: min(int(spec["topk"]), len(valid))]]
        weights = (1.0 / (1.0 + distance[selected])) ** float(spec["power"])
        weights = weights / float(weights.sum())
        signal = float(weights @ residual[selected])
        result[index] = raw[index] + float(spec["weight"]) * signal
    return result

def _select_random_candidate(frame: pd.DataFrame, candidate_columns: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    y = frame["y_true"].to_numpy(dtype=float)
    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    selected = np.empty(len(frame), dtype=float)
    rows = []
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = ~train_mask
        scored = []
        for column in candidate_columns:
            pred = frame[column].to_numpy(dtype=float)
            scored.append(
                (column, float(r2_score(y[train_mask], pred[train_mask])), float(mean_squared_error(y[train_mask], pred[train_mask]) ** 0.5))
            )
        candidate, train_r2, train_rmse = sorted(scored, key=lambda value: (-value[1], value[2], value[0]))[0]
        selected[valid_mask] = frame.loc[valid_mask, candidate].to_numpy(dtype=float)
        rows.append(
            {
                "split_seed": int(frame["split_seed"].iloc[0]),
                "stage": "oracle_nested",
                "random_cv_fold": int(fold),
                "candidate_key": candidate,
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "heldout_row_count": int(valid_mask.sum()),
            }
        )
    return selected, pd.DataFrame(rows)

def _select_holdout_candidate(frame: pd.DataFrame, candidate_columns: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    train = frame[frame["split_role"].astype(str).eq("train")]
    scored = []
    for column in candidate_columns:
        scored.append(
            (
                column,
                float(r2_score(train["y_true"], train[column])),
                float(mean_squared_error(train["y_true"], train[column]) ** 0.5),
            )
        )
    candidate, train_r2, train_rmse = sorted(scored, key=lambda value: (-value[1], value[2], value[0]))[0]
    row = pd.DataFrame(
        [
            {
                "split_seed": int(frame["split_seed"].iloc[0]),
                "stage": "oracle_nested",
                "split_role": "train",
                "candidate_key": candidate,
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "heldout_row_count": int(len(frame)),
            }
        ]
    )
    return frame[candidate].to_numpy(dtype=float), row

def _run_random_stages(
    frame: pd.DataFrame, descriptor_features: pd.DataFrame, family: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.reset_index(drop=True).copy()
    result["same_canonical_train_count"] = _random_same_canonical_counts(result)
    assert_family_input_columns(["native_raw"], family)
    result["residual_bin"] = _random_residual_bin(result, "native_raw")
    assert_family_input_columns(["native_raw", "residual_bin"], family)
    result["native_family_stack"] = fit_outer_fold_ridge(result, ["native_raw", "residual_bin"])
    result["local_chem_residual"] = _random_local_descriptor(result, descriptor_features, "native_raw")
    assert_family_input_columns(["native_raw", "residual_bin", "local_chem_residual"], family)
    result["three_source_stack"] = fit_outer_fold_ridge(
        result, ["native_raw", "residual_bin", "local_chem_residual"]
    )
    assert_family_input_columns(["native_family_stack", "three_source_stack", "local_chem_residual"], family)
    result["oof_combo"] = fit_outer_fold_ridge(
        result, ["native_family_stack", "three_source_stack", "local_chem_residual"]
    )
    candidates = [
        "native_raw",
        "residual_bin",
        "native_family_stack",
        "three_source_stack",
        "oof_combo",
        "local_chem_residual",
    ]
    result["oracle_nested"], oracle_selection = _select_random_candidate(result, candidates)
    from run_fixedbest10_randomcv_oof_combo_descriptor_knn_correction import _nested_select_predictions

    descriptor_input = result.copy()
    descriptor_input["fp_smiles"] = descriptor_feature_smiles(descriptor_input["canonical_smiles"])
    descriptor_input["y_pred"] = descriptor_input["oracle_nested"].to_numpy(dtype=float)
    selected, descriptor_selection, _metrics = _nested_select_predictions(
        descriptor_input, descriptor_features, list(DESCRIPTOR_SPECS), "r2"
    )
    keys = ["split_seed", "sample_id", "random_cv_fold", "y_true"]
    result = result.merge(
        selected[keys + ["candidate_key", "y_pred"]], on=keys, how="inner", validate="one_to_one"
    )
    result = result.rename(columns={"candidate_key": "descriptor_candidate_key"})
    descriptor_selection["stage"] = "descriptor_knn_nested"
    return result, pd.concat([oracle_selection, descriptor_selection], ignore_index=True, sort=False)

def _run_holdout_stages(
    frame: pd.DataFrame, descriptor_features: pd.DataFrame, family: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.reset_index(drop=True).copy()
    result["same_canonical_train_count"] = _holdout_same_canonical_counts(result)
    assert_family_input_columns(["native_raw"], family)
    result["residual_bin"] = _holdout_residual_bin(result, "native_raw")
    assert_family_input_columns(["native_raw", "residual_bin"], family)
    result["native_family_stack"] = fit_holdout_train_only_ridge(result, ["native_raw", "residual_bin"])
    result["local_chem_residual"] = _holdout_descriptor_prediction(
        result, descriptor_features, "native_raw", LOCAL_DESCRIPTOR_SPEC
    )
    assert_family_input_columns(["native_raw", "residual_bin", "local_chem_residual"], family)
    result["three_source_stack"] = fit_holdout_train_only_ridge(
        result, ["native_raw", "residual_bin", "local_chem_residual"]
    )
    assert_family_input_columns(["native_family_stack", "three_source_stack", "local_chem_residual"], family)
    result["oof_combo"] = fit_holdout_train_only_ridge(
        result, ["native_family_stack", "three_source_stack", "local_chem_residual"]
    )
    candidates = [
        "native_raw",
        "residual_bin",
        "native_family_stack",
        "three_source_stack",
        "oof_combo",
        "local_chem_residual",
    ]
    result["oracle_nested"], oracle_selection = _select_holdout_candidate(result, candidates)
    descriptor_candidates = []
    train = result[result["split_role"].astype(str).eq("train")]
    for spec in DESCRIPTOR_SPECS:
        if str(spec["mode"]) != "residual":
            continue
        key = "|".join(f"{name}={value}" for name, value in spec.items())
        result[key] = _holdout_descriptor_prediction(result, descriptor_features, "oracle_nested", spec)
        descriptor_candidates.append(key)
    result["y_pred"], descriptor_selection = _select_holdout_candidate(result, descriptor_candidates)
    descriptor_selection["stage"] = "descriptor_knn_nested"
    return result, pd.concat([oracle_selection, descriptor_selection], ignore_index=True, sort=False)

FIXED_BEST10_SEEDS = (
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

def _parse_seeds(value: str, allow_subset: bool) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in FIXED_BEST10_SEEDS for seed in seeds):
        raise ValueError("seed list must be a non-empty unique subset of the frozen best10 seeds")
    expected_order = tuple(seed for seed in FIXED_BEST10_SEEDS if seed in set(seeds))
    if seeds != expected_order:
        raise ValueError("seed list must retain the frozen best10 order")
    if not allow_subset and seeds != FIXED_BEST10_SEEDS:
        raise ValueError("the formal run requires all frozen best10 seeds in order")
    return seeds

def _score(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(list(groups), sort=True):
        values = values if isinstance(values, tuple) else (values,)
        y = group["y_true"].to_numpy(dtype=float)
        pred = group["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(groups, values, strict=True)),
                "n": int(len(group)),
                "r2": float(r2_score(y, pred)),
                "rmse": float(mean_squared_error(y, pred) ** 0.5),
            }
        )
    return pd.DataFrame(rows)

def _lineage_rows() -> pd.DataFrame:
    inputs = {
        "native_s3": "raw_cho_nci_native_candidate_or_blend",
        "residual_bin": "native_raw",
        "native_family_stack": "native_raw,residual_bin",
        "three_source_stack": "native_raw,residual_bin,local_chem_residual",
        "oof_combo": "native_family_stack,three_source_stack,local_chem_residual",
        "oracle_nested": "native_raw,residual_bin,native_family_stack,three_source_stack,oof_combo,local_chem_residual",
        "descriptor_knn_nested": "oracle_nested,raw_numeric_descriptors",
    }
    rows = []
    for family in FAMILIES:
        for stage in FAMILY_ISOLATED_STAGES:
            rows.append(
                {
                    "family": family,
                    "stage": stage,
                    "input_lineage": inputs[stage],
                    "et_prediction_input_used": False,
                    "cross_family_prediction_input_used": False,
                    "historical_prediction_csv_used_as_module_input": False,
                    "holdout_labels_used_for_selection": False,
                }
            )
    return pd.DataFrame(rows)

def run(args: argparse.Namespace) -> Path:
    from run_current_raw_teacher2_full_chain import _native_predictions
    from run_fixedbest10_randomcv_oof_combo_descriptor_knn_correction import _load_numeric_features

    root = Path(args.root_dir).resolve()
    seeds = _parse_seeds(args.seed_list, args.allow_seed_subset)
    out_dir = root / "data_processed" / "components" / args.run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run directory: {out_dir}")

    native_random, native_holdout, raw_audit = _native_predictions(
        root,
        seeds,
        args.gpu,
        args.cv_folds,
        scientific_regularized=bool(args.scientific_regularized),
    )
    descriptor_input = pd.DataFrame(
        {
            "fp_smiles": pd.concat(
                [native_random["canonical_smiles"], native_holdout["canonical_smiles"]], ignore_index=True
            ).pipe(descriptor_feature_smiles)
        }
    )
    descriptor_features = _load_numeric_features(
        root,
        descriptor_input,
        "data_processed/cache/chem_cache/chem_features_maccs166_desc_fp166.parquet",
    )

    random_parts = []
    random_nested_selection_parts = []
    holdout_parts = []
    selection_parts = []
    for family in FAMILIES:
        family_random = native_random[native_random["family"].eq(family)].copy()
        family_holdout = native_holdout[native_holdout["family"].eq(family)].copy()
        for seed in seeds:
            random_seed = family_random[family_random["split_seed"].eq(seed)].copy()
            holdout_seed = family_holdout[family_holdout["split_seed"].eq(seed)].copy()
            if random_seed.empty or holdout_seed.empty:
                raise ValueError(f"missing raw native predictions for {family} seed {seed}")
            if family == FAMILIES[0]:
                nested_seed = native_random[native_random["split_seed"].eq(seed)].copy()
                random_nested, nested_selection, _ = _run_nested_random_seed(nested_seed, descriptor_features)
                random_parts.append(random_nested)
                random_nested_selection_parts.append(nested_selection)

            train_oof, train_oof_selection = _run_random_stages(
                random_seed, descriptor_features, family
            )
            holdout_final, holdout_selection = _run_holdout_stages(holdout_seed, descriptor_features, family)
            train_pred = train_oof.set_index("sample_id")["y_pred"]
            train_mask = holdout_final["split_role"].astype(str).eq("train")
            train_ids = holdout_final.loc[train_mask, "sample_id"]
            if not train_ids.isin(train_pred.index).all():
                raise ValueError(f"holdout train OOF alignment failed for {family} seed {seed}")
            holdout_final.loc[train_mask, "y_pred"] = train_ids.map(train_pred).to_numpy(dtype=float)
            holdout_final["final_stage"] = "S3_family_isolated_seven_module_transfer"
            holdout_selection["family"] = family
            holdout_selection["prediction_scope"] = "holdout"
            train_oof_selection["family"] = family
            train_oof_selection["prediction_scope"] = "holdout_train_inner_oof"
            holdout_parts.append(holdout_final)
            selection_parts.extend([train_oof_selection, holdout_selection])

    random_isolated = pd.concat(random_parts, ignore_index=True)
    holdout_isolated = pd.concat(holdout_parts, ignore_index=True)
    random_primary_selection = pd.concat(random_nested_selection_parts, ignore_index=True, sort=False)
    holdout_outputs = {
        family: holdout_isolated[holdout_isolated["family"].eq(family)].copy()
        for family in FAMILIES
    }
    holdout_primary, holdout_primary_selection = fit_family_primary_holdout_stack(holdout_outputs, min_main_weight=0.51)
    random = random_isolated
    holdout = holdout_isolated.drop(columns=["y_pred"]).merge(
        holdout_primary[["family", "split_seed", "split_role", "sample_id", "y_pred"]],
        on=["family", "split_seed", "split_role", "sample_id"], how="inner", validate="one_to_one",
    )
    random["final_stage"] = "S3_family_primary_nonet_stack_nested"
    holdout["final_stage"] = "S3_family_primary_nonet_stack"
    random_keys = ["family", "split_seed", "sample_id", "random_cv_fold"]
    holdout_keys = ["family", "split_seed", "split_role", "sample_id"]
    validate_stable_keys(random, random_keys)
    validate_stable_keys(holdout, holdout_keys)
    if not np.isfinite(random["y_pred"].to_numpy(dtype=float)).all() or not np.isfinite(
        holdout["y_pred"].to_numpy(dtype=float)
    ).all():
        raise ValueError("family-isolated chain produced non-finite final predictions")

    out_dir.mkdir(parents=True, exist_ok=False)
    random.to_csv(out_dir / "family_isolated_randomcv_predictions.csv", index=False)
    holdout.to_csv(out_dir / "family_isolated_holdout_predictions.csv", index=False)
    pd.concat(selection_parts + [random_primary_selection], ignore_index=True, sort=False).to_csv(
        out_dir / "family_isolated_nested_selection.csv", index=False
    )
    pd.concat(
        [random_primary_selection.assign(prediction_scope="random_cv"),
         holdout_primary_selection.assign(prediction_scope="holdout")],
        ignore_index=True, sort=False,
    ).to_csv(out_dir / "family_primary_stack_selection.csv", index=False)
    pd.DataFrame(
        [{"family": family, "stage": "historical_family_primary_stack",
          "input_lineage": "current_raw_family_isolated_s3_predictions + other_non_et_family_predictions",
          "et_prediction_input_used": False, "cross_family_prediction_input_used": True,
          "cross_family_prediction_scope": "non_et_only_fold_local",
          "historical_prediction_csv_used_as_module_input": False,
          "holdout_labels_used_for_selection": False} for family in FAMILIES]
    ).to_csv(out_dir / "family_primary_stack_lineage.csv", index=False)
    _lineage_rows().to_csv(out_dir / "family_isolated_lineage.csv", index=False)
    raw_audit.to_csv(out_dir / "raw_native_leakage_audit_by_seed.csv", index=False)
    random_metrics = _score(random, ["family", "split_seed"])
    holdout_metrics = _score(holdout, ["family", "split_seed", "split_role"])
    random_metrics.to_csv(out_dir / "randomcv_metrics_by_family_seed.csv", index=False)
    holdout_metrics.to_csv(out_dir / "holdout_metrics_by_family_seed_role.csv", index=False)
    generalization_gap = summarize_generalization_gap(random_metrics, holdout)
    generalization_gap.to_csv(out_dir / "generalization_gap_by_family.csv", index=False)
    summary = {
        "run_id": args.run_id,
        "fixed_seeds": list(seeds),
        "preflight_only": len(seeds) != len(FIXED_BEST10_SEEDS),
        "families": list(FAMILIES),
        "stages": list(FAMILY_ISOLATED_STAGES),
        "historical_s3_stage": "family_primary_nonet_stack",
        "historical_s3_stack_pool": "current_raw_family_isolated_s3_outputs_all_four_non_et_families",
        "historical_s3_min_main_weight": 0.51,
        "raw_native_retrained": True,
        "raw_native_leakage_hard_failure_count": int(raw_audit["hard_failure_count"].sum()),
        "et_prediction_input_used": False,
        "cross_family_prediction_input_used": False,
        "isolated_stage_cross_family_prediction_input_used": False,
        "final_stack_cross_family_non_et_input_used": True,
        "historical_prediction_csv_used_as_module_input": False,
        "uses_test_validation_experimental_labels_for_selection": False,
        "random_stage_fit_scope": "outer_fold_inner_oof_then_outer_valid_train_only",
        "holdout_stage_fit_scope": "same_seed_train_role_only",
        "generalization_gap_gate": "abs(train_r2-cv_r2)<=0.15 and abs(test_r2-cv_r2)<=0.15; report-only",
        "generalization_gap_summary": generalization_gap.to_dict("records"),
        "randomcv_summary": _score(random, ["family"]).to_dict("records"),
        "holdout_summary": _score(holdout, ["family", "split_role"]).to_dict("records"),
        "scientific_boundary": (
            "All final predictions are regenerated from raw CHO/NCI inputs. The seven upstream stages are isolated "
            "from ET and other family predictions; the historical final family-primary stack uses only the four "
            "non-ET family outputs and is fit fold-locally. Random-CV fits exclude the held-out outer fold; holdout "
            "fits use only the same-seed train role labels. Test, validation, and experimental labels are report-only."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-list", default=",".join(str(seed) for seed in FIXED_BEST10_SEEDS))
    parser.add_argument("--allow-seed-subset", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--scientific-regularized",
        action="store_true",
        help="Use the pre-specified regularized native baselines for the scientific stress run.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    print(f"wrote {run(parse_args())}")

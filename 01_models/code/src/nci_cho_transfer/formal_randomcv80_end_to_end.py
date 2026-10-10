from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FORMAL_WEIGHT_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
FIXED_BEST10_SEEDS = (20260021, 20260106, 20260290, 20260307, 20260605, 20260730, 20260738, 20260755, 20260998, 20262000)
CURRENT_ID = "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_no_counterpart_pair"
ATOM_ID = "extratrees_700_domain_rdkit_tt_f65_l1_atom_pair_a30_k3_p2_time48_rdk_a50_k3_p4_time72_rdk_a25_k1"
SHRINK_ID = CURRENT_ID + "__calnone_shrink035"
HOLDOUT80_CANDIDATE = {"w_current": 0.25, "w_atom": 0.50, "w_shrink": 0.25, "topk": 20, "power": 2.0, "alpha": -0.8}
RANDOMCV80_DESCRIPTOR_PARAMS = {"topk": 5, "power": 2.0, "weight": -0.1}
NESTED_CANDIDATE_WC = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
NESTED_CANDIDATE_WA = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
NESTED_CANDIDATE_TOPK = (10, 20, 40)
NESTED_CANDIDATE_POWER = (1.0, 2.0, 4.0)
NESTED_CANDIDATE_ALPHA = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0)
RESIDUAL_META_CANDIDATE = {
    "wc": 0.35,
    "wa": 0.65,
    "ws": 0.0,
    "topk": 20,
    "power": 4.0,
    "alpha": -1.0,
}
SALT_FEATURE_SMILES = {
    "[K+].[O-][Br+2]([O-])[O-]": "[O-][Br+2]([O-])O",
    "[Na+].[O-][Br+2]([O-])[O-]": "[O-][Br+2]([O-])O",
    "[Na+].[O-][Cl+2]([O-])[O-]": "[O-][Cl+2]([O-])O",
    "[Na+].[O-][Cl+][O-]": "[O-][Cl+]O",
    "[N-]([N+](=O)[O-])Cl": "O=[N+]([O-])[N-]Cl",
}
_REQUIRED_COLUMNS = {
    "split_seed",
    "sample_id",
    "y_true",
    "randomcv80_y_pred",
    "holdout80_y_pred",
}

def assert_current_run_branch_bundle(branches: Mapping[str, object]) -> None:
    for name, value in branches.items():
        if isinstance(value, (str, bytes)):
            raise ValueError(f"legacy prediction path is not allowed for branch {name}")
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"branch {name} must be a DataFrame generated in the current run")

def audit_prediction_isolation(rows: pd.DataFrame) -> dict[str, int]:
    required = {"split_seed", "split_role", "sample_id", "canonical_time_key"}
    _validate_rows(rows, required)
    for key in ("sample_id", "canonical_time_key"):
        cross_role = (
            rows.groupby(["split_seed", key], dropna=False)["split_role"]
            .nunique()
            .gt(1)
        )
        count = int(cross_role.sum())
        if count:
            raise ValueError(f"{key} crosses split roles for {count} seed-local keys")
    return {
        "checked_rows": int(len(rows)),
        "cross_role_sample_id_count": 0,
        "cross_role_canonical_time_key_count": 0,
    }

def build_run_manifest(
    run_id: str,
    input_hashes: Mapping[str, str],
    fixed_candidate: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "run_id": str(run_id),
        "formal_weight_grid": list(FORMAL_WEIGHT_GRID),
        "segment_mode": "global",
        "segment_min_rows": 4,
        "fixed_candidate": dict(fixed_candidate),
        "input_hashes": dict(sorted(input_hashes.items())),
        "runtime_prediction_inputs": "current_run_memory_only",
        "random_candidate_selection": "outer_fold_other_folds_only",
        "random_candidate_calibration": "outer_fold_other_folds_only",
        "random_component_prediction": "outer_train_full_fit_predictions_and_outer_valid_fit_only",
        "random_cv_weight_selection": "other_folds_only",
        "holdout_weight_selection": "train_role_only",
        "holdout_descriptor_correction_excludes_same_canonical": True,
    }

def summarize_prediction_metrics(rows: pd.DataFrame, prediction_column: str, groups: list[str]) -> pd.DataFrame:
    _validate_rows(rows, {"y_true", prediction_column, *groups})
    output = []
    for keys, group in rows.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group[prediction_column].to_numpy(dtype=float)
        output.append(
            {
                **dict(zip(groups, keys, strict=True)),
                "n": int(len(group)),
                "r2": float(r2_score(y_true, y_pred)),
                "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }
        )
    return pd.DataFrame(output)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _maccs_features(root_dir: Path, smiles: pd.Series) -> pd.DataFrame:
    path = root_dir / "data_processed" / "cache" / "chem_cache" / "chem_features_maccs166_desc_fp166.parquet"
    table = pd.read_parquet(path)
    columns = [column for column in table.columns if str(column).startswith("maccs_")]
    features = table.set_index("canonical_smiles")[columns].astype(np.float32)
    requested = sorted(set(smiles.astype(str)))
    missing = sorted(
        original
        for original in requested
        if SALT_FEATURE_SMILES.get(original, original) not in set(features.index.astype(str))
    )
    if missing:
        fallback_rows = {}
        for canonical in missing:
            lookup = SALT_FEATURE_SMILES.get(canonical, canonical)
            molecule = Chem.MolFromSmiles(str(lookup))
            if molecule is None:
                raise ValueError(f"Cannot generate MACCS features for canonical SMILES: {canonical}")
            bit_vector = MACCSkeys.GenMACCSKeys(molecule)
            fallback_rows[canonical] = {
                column: float(bit_vector[int(str(column).split("_")[-1])]) for column in columns
            }
        fallback = pd.DataFrame.from_dict(fallback_rows, orient="index").astype(np.float32)
        fallback.index.name = features.index.name
        features = pd.concat([features, fallback], axis=0)
    aliases = {}
    for original in requested:
        lookup = SALT_FEATURE_SMILES.get(original, original)
        if original not in features.index and lookup in features.index:
            aliases[original] = features.loc[lookup].to_numpy(dtype=np.float32)
    if aliases:
        features = pd.concat([features, pd.DataFrame.from_dict(aliases, orient="index", columns=columns)], axis=0)
    return features

def _tanimoto(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query_bits = (query > 0).astype(np.float32)
    reference_bits = (reference > 0).astype(np.float32)
    intersection = query_bits @ reference_bits.T
    denominator = query_bits.sum(axis=1)[:, None] + reference_bits.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)

def _topk_similarity_weights(similarity: np.ndarray, topk: int, power: float) -> np.ndarray:
    weights = np.zeros_like(similarity, dtype=float)
    for index, row in enumerate(similarity):
        valid = np.flatnonzero(np.isfinite(row))
        if not len(valid):
            continue
        selected = valid[np.argsort(row[valid])[-min(int(topk), len(valid)):]]
        raw = np.maximum(row[selected], 0.0) ** float(power)
        if float(raw.sum()) > 0.0:
            weights[index, selected] = raw / float(raw.sum())
    return weights

def _base_prediction(frame: pd.DataFrame) -> np.ndarray:
    return (
        HOLDOUT80_CANDIDATE["w_current"] * frame["current_pred"].to_numpy(dtype=float)
        + HOLDOUT80_CANDIDATE["w_atom"] * frame["atom_pred"].to_numpy(dtype=float)
        + HOLDOUT80_CANDIDATE["w_shrink"] * frame["shrink_pred"].to_numpy(dtype=float)
    )

def _residual_local_random_oof(frame: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    base = _base_prediction(frame)
    y = frame["y_true"].to_numpy(dtype=float)
    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    x = features.loc[frame["canonical_smiles"].astype(str)].to_numpy(dtype=float)
    result = np.zeros(len(frame), dtype=float)
    for fold in sorted(set(folds.tolist())):
        fit = folds != fold
        valid = ~fit
        weights = _topk_similarity_weights(
            _tanimoto(x[valid], x[fit]),
            int(HOLDOUT80_CANDIDATE["topk"]),
            float(HOLDOUT80_CANDIDATE["power"]),
        )
        result[valid] = base[valid] + float(HOLDOUT80_CANDIDATE["alpha"]) * (weights @ (y[fit] - base[fit]))
    return result

def _fold_train_linear_calibration(frame: pd.DataFrame, prediction_column: str) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    y = frame["y_true"].to_numpy(dtype=float)
    raw = frame[prediction_column].to_numpy(dtype=float)
    for fold in sorted(set(folds.tolist())):
        fit = folds != fold
        valid = ~fit
        model = LinearRegression().fit(raw[fit, None], y[fit])
        result[valid] = model.predict(raw[valid, None])
    return result

def _descriptor_knn_random_oof(frame: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    raw = frame["holdout80_y_pred"].to_numpy(dtype=float)
    y = frame["y_true"].to_numpy(dtype=float)
    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    x = features.loc[frame["canonical_smiles"].astype(str)].to_numpy(dtype=float)
    result = raw.copy()
    for fold in sorted(set(folds.tolist())):
        fit = folds != fold
        valid = ~fit
        center = x[fit].mean(axis=0)
        scale = x[fit].std(axis=0)
        scale[scale == 0.0] = 1.0
        differences = ((x[valid] - center) / scale)[:, None, :] - ((x[fit] - center) / scale)[None, :, :]
        distances = np.sqrt((differences**2).sum(axis=2))
        signal = np.zeros(int(valid.sum()), dtype=float)
        residual = y[fit] - raw[fit]
        for row_index, row in enumerate(distances):
            selected = np.argsort(row)[: min(int(RANDOMCV80_DESCRIPTOR_PARAMS["topk"]), len(row))]
            similarity = 1.0 / (1.0 + row[selected])
            weight = similarity ** float(RANDOMCV80_DESCRIPTOR_PARAMS["power"])
            signal[row_index] = float((weight / weight.sum()) @ residual[selected])
        result[valid] = raw[valid] + float(RANDOMCV80_DESCRIPTOR_PARAMS["weight"]) * signal
    return result

def _nested_candidate_grid() -> list[dict[str, float | int | str]]:
    candidates = []
    for wc in NESTED_CANDIDATE_WC:
        for wa in NESTED_CANDIDATE_WA:
            ws = round(1.0 - wc - wa, 10)
            if ws < -1e-12:
                continue
            for topk in NESTED_CANDIDATE_TOPK:
                for power in NESTED_CANDIDATE_POWER:
                    for alpha in NESTED_CANDIDATE_ALPHA:
                        candidate = {
                            "wc": float(wc),
                            "wa": float(wa),
                            "ws": float(max(ws, 0.0)),
                            "topk": int(topk),
                            "power": float(power),
                            "alpha": float(alpha),
                        }
                        candidate["candidate_key"] = "|".join(
                            f"{name}={candidate[name]}" for name in ("wc", "wa", "ws", "topk", "power", "alpha")
                        )
                        candidates.append(candidate)
    if len(candidates) != 3888:
        raise AssertionError(f"unexpected nested candidate count: {len(candidates)}")
    return candidates

def _nested_candidate_random_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    candidates: list[dict[str, float | int | str]] | None = None,
    component_contexts: Mapping[tuple[int, int], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    _validate_rows(
        frame,
        {"split_seed", "sample_id", "random_cv_fold", "canonical_smiles", "y_true", "current_pred", "atom_pred", "shrink_pred"},
    )
    candidates = list(candidates) if candidates is not None else _nested_candidate_grid()
    candidates = [
        {
            **candidate,
            "candidate_key": str(candidate.get("candidate_key", candidate.get("candidate_id", index))),
        }
        for index, candidate in enumerate(candidates)
    ]
    output = []

    def _linear_fit_predict(train_raw: np.ndarray, train_y: np.ndarray, query_raw: np.ndarray) -> np.ndarray:
        center = float(train_raw.mean())
        variance = float(np.dot(train_raw - center, train_raw - center))
        if variance <= 0.0:
            return np.full(len(query_raw), float(train_y.mean()), dtype=float)
        slope = float(np.dot(train_raw - center, train_y - float(train_y.mean())) / variance)
        intercept = float(train_y.mean()) - slope * center
        return intercept + slope * query_raw

    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
        y = seed_frame["y_true"].to_numpy(dtype=float)
        x = features.loc[seed_frame["canonical_smiles"].astype(str)].to_numpy(dtype=float)
        selected_rows = []
        for outer_fold in sorted(set(folds.tolist())):
            if component_contexts is None:
                base_components = np.column_stack(
                    [seed_frame[column].to_numpy(dtype=float) for column in ("current_pred", "atom_pred", "shrink_pred")]
                )
            else:
                context_key = (int(seed), int(outer_fold))
                if context_key not in component_contexts:
                    raise ValueError(f"missing nested component context for seed/fold {context_key}")
                context = component_contexts[context_key]
                _validate_rows(context, {"sample_id", "current_pred", "atom_pred", "shrink_pred"})
                if context["sample_id"].duplicated().any():
                    raise ValueError(f"nested component context has duplicate sample_id for {context_key}")
                context = context.set_index("sample_id")
                sample_ids = seed_frame["sample_id"].astype(str)
                if not sample_ids.isin(context.index.astype(str)).all():
                    raise ValueError(f"nested component context is missing rows for {context_key}")
                aligned = context.reindex(sample_ids.tolist())
                base_components = aligned[["current_pred", "atom_pred", "shrink_pred"]].to_numpy(dtype=float)
            outer_train = folds != outer_fold
            inner_folds = folds[outer_train]
            inner_indices = np.flatnonzero(outer_train)
            inner_weights: dict[tuple[int, float], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
            for topk in sorted({int(candidate["topk"]) for candidate in candidates}):
                for power in sorted({float(candidate["power"]) for candidate in candidates}):
                    cached = []
                    for inner_fold in sorted(set(inner_folds.tolist())):
                        inner_valid = inner_folds == inner_fold
                        inner_fit = ~inner_valid
                        similarity = _tanimoto(x[inner_indices[inner_valid]], x[inner_indices[inner_fit]])
                        weights = _topk_similarity_weights(similarity, topk, power)
                        cached.append((inner_valid, inner_fit, weights))
                    inner_weights[(topk, power)] = cached
            candidate_scores = []
            calibrated_inner_by_candidate = []
            for candidate in candidates:
                base = base_components @ np.array([candidate["wc"], candidate["wa"], candidate["ws"]], dtype=float)
                inner_raw = np.zeros(int(outer_train.sum()), dtype=float)
                for inner_valid, inner_fit, weights in inner_weights[(int(candidate["topk"]), float(candidate["power"]))]:
                    inner_raw[inner_valid] = base[inner_indices[inner_valid]] + float(candidate["alpha"]) * (
                        weights @ (y[inner_indices[inner_fit]] - base[inner_indices[inner_fit]])
                    )
                inner_calibrated = np.zeros_like(inner_raw)
                if len(set(inner_folds.tolist())) < 2:
                    inner_calibrated[:] = inner_raw
                else:
                    for inner_fold in sorted(set(inner_folds.tolist())):
                        valid = inner_folds == inner_fold
                        fit = ~valid
                        inner_calibrated[valid] = _linear_fit_predict(
                            inner_raw[fit], y[inner_indices[fit]], inner_raw[valid]
                        )
                residual = y[inner_indices] - inner_calibrated
                denominator = float(np.dot(y[inner_indices] - y[inner_indices].mean(), y[inner_indices] - y[inner_indices].mean()))
                score = 1.0 - float(np.dot(residual, residual)) / denominator
                candidate_scores.append((score, float(np.sqrt(np.mean(residual**2))), str(candidate["candidate_key"])))
                calibrated_inner_by_candidate.append(inner_raw)
            candidate_scores.sort(key=lambda row: (-row[0], row[1], row[2]))
            selected_key = candidate_scores[0][2]
            selected = next(candidate for candidate in candidates if str(candidate["candidate_key"]) == selected_key)
            base = base_components @ np.array([selected["wc"], selected["wa"], selected["ws"]], dtype=float)
            valid_indices = np.flatnonzero(~outer_train)
            similarity = _tanimoto(x[valid_indices], x[inner_indices])
            weights = _topk_similarity_weights(similarity, int(selected["topk"]), float(selected["power"]))
            raw_valid = base[valid_indices] + float(selected["alpha"]) * (
                weights @ (y[inner_indices] - base[inner_indices])
            )
            selected_index = next(index for index, candidate in enumerate(candidates) if str(candidate["candidate_key"]) == selected_key)
            inner_raw = calibrated_inner_by_candidate[selected_index]
            if len(set(inner_folds.tolist())) < 2:
                y_pred = raw_valid
            else:
                y_pred = _linear_fit_predict(inner_raw, y[inner_indices], raw_valid)
            result = seed_frame.loc[valid_indices, ["split_seed", "sample_id", "random_cv_fold", "y_true"]].copy()
            result["candidate_key"] = selected_key
            result["raw_candidate_pred"] = raw_valid
            result["y_pred"] = y_pred
            selected_rows.append(result)
        output.extend(selected_rows)
    result = pd.concat(output, ignore_index=True)
    if result.duplicated(["split_seed", "random_cv_fold", "sample_id"]).any():
        raise ValueError("Nested candidate selection produced duplicate OOF rows")
    return result

def _same_canonical_counts(query: pd.Series, reference: pd.Series) -> np.ndarray:
    query_values = query.astype(str).to_numpy()
    reference_values = reference.astype(str).to_numpy()
    return np.asarray([(reference_values == value).sum() for value in query_values], dtype=float)

def _residual_meta_design(
    frame: pd.DataFrame,
    indices: np.ndarray,
    raw: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    current = frame.iloc[indices]["current_pred"].to_numpy(dtype=float)
    atom = frame.iloc[indices]["atom_pred"].to_numpy(dtype=float)
    shrink = frame.iloc[indices]["shrink_pred"].to_numpy(dtype=float)
    time_h = frame.iloc[indices]["time_h"].to_numpy(dtype=float)
    reference = frame.iloc[reference_indices]["canonical_smiles"]
    counts = _same_canonical_counts(frame.iloc[indices]["canonical_smiles"], reference)
    covered = (counts > 0.0).astype(float)
    return np.column_stack(
        [
            raw,
            current,
            atom,
            shrink,
            time_h,
            current - atom,
            raw - current,
            raw - atom,
            np.abs(current - atom),
            np.abs(raw - current),
            covered,
            counts,
        ]
    )

def _raw_residual_meta_candidate(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
) -> np.ndarray:
    candidate = RESIDUAL_META_CANDIDATE
    components = frame[["current_pred", "atom_pred", "shrink_pred"]].to_numpy(dtype=float)
    base = components @ np.array([candidate["wc"], candidate["wa"], candidate["ws"]], dtype=float)
    x = features.loc[frame["canonical_smiles"].astype(str)].to_numpy(dtype=float)
    weights = _topk_similarity_weights(
        _tanimoto(x[query_indices], x[fit_indices]),
        int(candidate["topk"]),
        float(candidate["power"]),
    )
    y = frame["y_true"].to_numpy(dtype=float)
    return base[query_indices] + float(candidate["alpha"]) * (
        weights @ (y[fit_indices] - base[fit_indices])
    )

def _nested_residual_meta_random_oof(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    _validate_rows(
        frame,
        {
            "split_seed",
            "sample_id",
            "random_cv_fold",
            "canonical_smiles",
            "time_h",
            "y_true",
            "current_pred",
            "atom_pred",
            "shrink_pred",
        },
    )
    outputs = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
        y = seed_frame["y_true"].to_numpy(dtype=float)
        for outer_fold in sorted(set(folds.tolist())):
            outer_train = folds != outer_fold
            outer_valid = ~outer_train
            outer_train_indices = np.flatnonzero(outer_train)
            outer_valid_indices = np.flatnonzero(outer_valid)
            raw_train = np.zeros(int(outer_train.sum()), dtype=float)
            train_counts = np.zeros(int(outer_train.sum()), dtype=float)
            inner_folds = folds[outer_train]
            for inner_fold in sorted(set(inner_folds.tolist())):
                inner_valid = inner_folds == inner_fold
                inner_fit = ~inner_valid
                query_indices = outer_train_indices[inner_valid]
                fit_indices = outer_train_indices[inner_fit]
                raw_train[inner_valid] = _raw_residual_meta_candidate(
                    seed_frame, features, fit_indices, query_indices
                )
                train_counts[inner_valid] = _same_canonical_counts(
                    seed_frame.iloc[query_indices]["canonical_smiles"],
                    seed_frame.iloc[fit_indices]["canonical_smiles"],
                )
            raw_valid = _raw_residual_meta_candidate(
                seed_frame, features, outer_train_indices, outer_valid_indices
            )
            x_train = _residual_meta_design(
                seed_frame,
                outer_train_indices,
                raw_train,
                outer_train_indices,
            )
            x_train[:, -2:] = np.column_stack([(train_counts > 0.0).astype(float), train_counts])
            x_valid = _residual_meta_design(
                seed_frame,
                outer_valid_indices,
                raw_valid,
                outer_train_indices,
            )
            model = ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=10,
                max_features=0.5,
                random_state=int(seed) + int(outer_fold),
                n_jobs=-1,
            )
            model.fit(x_train, y[outer_train_indices] - raw_train)
            pred_valid = raw_valid + model.predict(x_valid)
            result = seed_frame.loc[outer_valid_indices, ["split_seed", "sample_id", "random_cv_fold", "y_true"]].copy()
            result["candidate_key"] = "residual_meta_inner_oof_extratrees_leaf10_mf05"
            result["raw_candidate_pred"] = raw_valid
            result["y_pred"] = pred_valid
            outputs.append(result)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["split_seed", "random_cv_fold", "sample_id"]).any():
        raise ValueError("Nested residual-meta selection produced duplicate OOF rows")
    return result

def _randomcv80_branch_predictions(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    component_contexts: Mapping[tuple[int, int], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()
        nested = _nested_candidate_random_oof(seed_frame, features, component_contexts=component_contexts)
        seed_frame = seed_frame.drop(columns=["y_true"], errors="ignore").merge(
            nested[["split_seed", "sample_id", "random_cv_fold", "y_true", "candidate_key", "raw_candidate_pred", "y_pred"]],
            on=["split_seed", "sample_id", "random_cv_fold"],
            how="left",
            validate="one_to_one",
        )
        seed_frame["holdout80_y_pred"] = seed_frame["y_pred"]
        seed_frame["randomcv80_y_pred"] = seed_frame["y_pred"]
        rows.append(seed_frame)
    return pd.concat(rows, ignore_index=True)

def _holdout_branch_predictions(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    holdout_rows: list[pd.DataFrame] = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True).copy()

        reference = seed_frame[seed_frame["split_role"].astype(str).eq("train")].reset_index(drop=True)
        reference_base = _base_prediction(reference)
        ref_x = features.loc[reference["canonical_smiles"].astype(str)].to_numpy(dtype=float)
        output_roles: list[pd.DataFrame] = []
        for role, role_frame in seed_frame.groupby("split_role", sort=True):
            role_frame = role_frame.reset_index(drop=True).copy()
            role_x = features.loc[role_frame["canonical_smiles"].astype(str)].to_numpy(dtype=float)
            similarity = _tanimoto(role_x, ref_x)
            same = role_frame["canonical_smiles"].astype(str).to_numpy()[:, None] == reference["canonical_smiles"].astype(str).to_numpy()[None, :]
            similarity[same] = -np.inf
            residual = reference["y_true"].to_numpy(dtype=float) - reference_base
            local = _base_prediction(role_frame) + float(HOLDOUT80_CANDIDATE["alpha"]) * (
                _topk_similarity_weights(similarity, int(HOLDOUT80_CANDIDATE["topk"]), float(HOLDOUT80_CANDIDATE["power"])) @ residual
            )
            role_frame["raw_holdout80_y_pred"] = local
            output_roles.append(role_frame)
        seed_holdout = pd.concat(output_roles, ignore_index=True)
        train = seed_holdout[seed_holdout["split_role"].astype(str).eq("train")]
        calibrator = LinearRegression().fit(
            train[["raw_holdout80_y_pred"]].to_numpy(dtype=float), train["y_true"].to_numpy(dtype=float)
        )
        seed_holdout["holdout80_y_pred"] = calibrator.predict(seed_holdout[["raw_holdout80_y_pred"]].to_numpy(dtype=float))

        x = features.loc[seed_holdout["canonical_smiles"].astype(str)].to_numpy(dtype=float)
        center = ref_x.mean(axis=0)
        scale = ref_x.std(axis=0)
        scale[scale == 0.0] = 1.0
        ref_scaled = (ref_x - center) / scale
        raw = seed_holdout["holdout80_y_pred"].to_numpy(dtype=float)
        train_raw = train["raw_holdout80_y_pred"].to_numpy(dtype=float)
        train_residual = train["y_true"].to_numpy(dtype=float) - train_raw
        reference_canonical = reference["canonical_smiles"].astype(str).to_numpy()
        query_canonical = seed_holdout["canonical_smiles"].astype(str).to_numpy()
        descriptor = np.zeros(len(seed_holdout), dtype=float)
        for index, row in enumerate((x - center) / scale):
            distances = np.sqrt(((ref_scaled - row) ** 2).sum(axis=1))
            if str(seed_holdout.iloc[index]["split_role"]) == "train":
                distances[reference_canonical == query_canonical[index]] = np.inf
            valid = np.flatnonzero(np.isfinite(distances))
            if not len(valid):
                continue
            selected = valid[np.argsort(distances[valid])[: min(int(RANDOMCV80_DESCRIPTOR_PARAMS["topk"]), len(valid))]]
            similarity = 1.0 / (1.0 + distances[selected])
            weight = similarity ** float(RANDOMCV80_DESCRIPTOR_PARAMS["power"])
            descriptor[index] = float((weight / weight.sum()) @ train_residual[selected])
        seed_holdout["randomcv80_y_pred"] = raw + float(RANDOMCV80_DESCRIPTOR_PARAMS["weight"]) * descriptor
        holdout_rows.append(seed_holdout)
    return pd.concat(holdout_rows, ignore_index=True)

def _nested_component_predictions_for_outer_fold(
    matrix: object,
    train: pd.DataFrame,
    spec: object,
    cv_folds: int,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    from .models import (
        _feature_matrix_for_spec,
        _fit_predict_raw_candidate,
        _oof_raw_predictions,
        calibrate_predictions,
        feature_columns_for_spec,
        residual_baseline_for_spec,
    )

    if getattr(spec, "family", None) in {
        "transfer_blend",
        "calibrated_transfer_blend",
        "gated_transfer",
        "a90_gated_transfer",
        "a90_delta_gated_transfer",
        "risk_adjusted_gated_transfer",
        "disagreement_gated_transfer",
        "risk_routed_transfer",
    }:
        raise ValueError(f"Nested component helper does not support composite spec: {spec.candidate_id}")

    feature_columns = feature_columns_for_spec(matrix, spec)
    x = _feature_matrix_for_spec(train, feature_columns, spec)
    y = train["y"].to_numpy(dtype=float)
    baseline = residual_baseline_for_spec(train, spec)
    def fit_and_predict(model_fit_idx: np.ndarray, query_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_pred, train_pred = _fit_predict_raw_candidate(
            train,
            x,
            y,
            baseline,
            spec,
            model_fit_idx,
            query_idx,
            return_fit_predictions=True,
        )
        if str(spec.calibration_method) == "none":
            return np.asarray(train_pred, dtype=float), np.asarray(raw_pred, dtype=float)
        calibration_frame = train.iloc[model_fit_idx].reset_index(drop=True)
        calibration_train_pred = _oof_raw_predictions(
            calibration_frame,
            x[model_fit_idx],
            y[model_fit_idx],
            baseline[model_fit_idx],
            spec,
            cv_folds=min(int(cv_folds), int(calibration_frame["canonical_smiles"].nunique())),
        )
        calibrated_query = calibrate_predictions(
            calibration_train_pred,
            y[model_fit_idx],
            raw_pred,
            method=spec.calibration_method,
            train_frame=calibration_frame,
            query_frame=train.iloc[query_idx].reset_index(drop=True),
        )
        calibrated_train = calibrate_predictions(
            calibration_train_pred,
            y[model_fit_idx],
            train_pred,
            method=spec.calibration_method,
            train_frame=calibration_frame,
            query_frame=calibration_frame,
        )
        return np.asarray(calibrated_train, dtype=float), np.asarray(calibrated_query, dtype=float)

    return fit_and_predict(np.asarray(fit_idx, dtype=int), np.asarray(valid_idx, dtype=int))

def _matrix_for_split(base_matrix, split):
    from .features import ModelMatrix

    role_frames = [split.train, split.test, split.validation, split.experimental]
    ordered_ids = pd.concat(
        [frame[["sample_id"]] for frame in role_frames],
        ignore_index=True,
    )["sample_id"].astype(str)
    if ordered_ids.duplicated().any() or base_matrix.frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("model-matrix reuse requires unique sample_id values")
    indexed = base_matrix.frame.copy()
    indexed["sample_id"] = indexed["sample_id"].astype(str)
    indexed = indexed.set_index("sample_id", drop=False)
    missing = sorted(set(ordered_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"model-matrix reuse is missing sample ids: {missing[:5]}")
    frame = indexed.loc[ordered_ids].reset_index(drop=True)
    frame["split_role"] = np.repeat(
        ["train", "test", "validation", "experimental"],
        [len(role_frame) for role_frame in role_frames],
    )
    return ModelMatrix(frame=frame, feature_columns=list(base_matrix.feature_columns))

def _component_predictions(
    root_dir: Path,
    seeds: tuple[int, ...],
    exclude_identical_source: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[int, int], pd.DataFrame]]:
    from .audit import audit_split_and_features
    from .data_contract import load_data_contract
    from .features import build_model_matrix
    from .models import build_candidates, candidate_by_id, fit_selected_and_evaluate
    from .splits import build_frozen_split
    import sys

    experiments_dir = root_dir / "code" / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    from export_fixedbest10_true_random_cv_component_oof import _random_splits

    bundle = load_data_contract(root_dir)
    target_smiles = set(bundle.cho_literature["canonical_smiles"].astype(str)) | set(
        bundle.experimental_validation["canonical_smiles"].astype(str)
    )
    source_smiles = set(bundle.nci_source_rows["canonical_smiles"].astype(str))
    identical_structures = target_smiles & source_smiles
    candidates = build_candidates("quick", None, 0)
    current = candidate_by_id(candidates, CURRENT_ID)
    specs = [current, candidate_by_id(candidates, ATOM_ID), replace(current, candidate_id=SHRINK_ID, residual_shrinkage=0.35, calibration_method="none")]
    random_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    nested_contexts: dict[tuple[int, int], pd.DataFrame] = {}
    reusable_matrix = None
    for seed in seeds:
        split = build_frozen_split(bundle, split_seed=int(seed))
        if exclude_identical_source and reusable_matrix is not None:
            matrix = _matrix_for_split(reusable_matrix, split)
        else:
            matrix = build_model_matrix(
                bundle,
                split,
                use_cache=True,
                exclude_identical_source=exclude_identical_source,
            )
            if exclude_identical_source:
                reusable_matrix = matrix
        audit = audit_split_and_features(split, matrix)
        audit_rows.append(
            {
                "split_seed": int(seed),
                "split_manifest_hash": split.manifest_hash,
                "leave_identical_compound_out": bool(exclude_identical_source),
                "source_target_identical_structure_count": int(len(identical_structures)),
                **audit,
            }
        )
        if int(audit["hard_failure_count"]):
            raise RuntimeError(f"matrix leakage audit failed for seed {seed}")
        train = matrix.role("train").reset_index(drop=True)
        component_predictions: dict[str, np.ndarray] = {}
        outer_splits = _random_splits(len(train), int(seed), 5)
        outer_fold_ids = np.full(len(train), -1, dtype=int)
        for outer_fold, (_fit_idx, valid_idx) in enumerate(outer_splits):
            outer_fold_ids[np.asarray(valid_idx, dtype=int)] = int(outer_fold)
        for outer_fold, (fit_idx, valid_idx) in enumerate(outer_splits):
            context = train[["sample_id"]].copy()
            for spec in specs:
                fit_pred, valid_pred = _nested_component_predictions_for_outer_fold(
                    matrix,
                    train,
                    spec,
                    5,
                    np.asarray(fit_idx, dtype=int),
                    np.asarray(valid_idx, dtype=int),
                )
                values = np.zeros(len(train), dtype=float)
                values[np.asarray(fit_idx, dtype=int)] = fit_pred
                values[np.asarray(valid_idx, dtype=int)] = valid_pred
                context[{
                    CURRENT_ID: "current_pred",
                    ATOM_ID: "atom_pred",
                    SHRINK_ID: "shrink_pred",
                }[spec.candidate_id]] = values
                component_predictions.setdefault(spec.candidate_id, np.zeros(len(train), dtype=float))
                component_predictions[spec.candidate_id][np.asarray(valid_idx, dtype=int)] = valid_pred
            nested_contexts[(int(seed), int(outer_fold))] = context

        for spec in specs:
            valid_rows = train[["sample_id", "canonical_smiles", "time_h", "y"]].copy()
            valid_rows = valid_rows.rename(columns={"y": "y_true"})
            valid_rows["candidate_id"] = spec.candidate_id
            valid_rows["split_seed"] = int(seed)
            valid_rows["random_cv_fold"] = outer_fold_ids
            valid_rows["random_oof_pred"] = component_predictions[spec.candidate_id]
            random_parts.append(valid_rows.assign(split_manifest_hash=split.manifest_hash))
            _, holdout = fit_selected_and_evaluate(matrix, spec)
            holdout_parts.append(holdout.assign(split_seed=int(seed), split_manifest_hash=split.manifest_hash))
    return (
        pd.concat(random_parts, ignore_index=True),
        pd.concat(holdout_parts, ignore_index=True),
        pd.DataFrame(audit_rows),
        nested_contexts,
    )

def _merge_component_predictions(random_components: pd.DataFrame, holdout_components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_columns = {CURRENT_ID: "current_pred", ATOM_ID: "atom_pred", SHRINK_ID: "shrink_pred"}

    def merge(frame: pd.DataFrame, prediction_column: str, role_columns: list[str]) -> pd.DataFrame:
        pieces = []
        for candidate_id, column in component_columns.items():
            part = frame[frame["candidate_id"].astype(str).eq(candidate_id)].copy()
            keep = ["split_seed", "split_manifest_hash", *role_columns, "sample_id", "canonical_smiles", "time_h", "y_true", prediction_column]
            pieces.append(part[keep].rename(columns={prediction_column: column}))
        keys = ["split_seed", "split_manifest_hash", *role_columns, "sample_id", "canonical_smiles", "time_h", "y_true"]
        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.merge(piece, on=keys, how="inner", validate="one_to_one")
        return merged

    random = merge(random_components, "random_oof_pred", ["random_cv_fold"])
    holdout = merge(holdout_components, "y_pred", ["split_role"])
    for frame in (random, holdout):
        frame["canonical_time_key"] = frame["canonical_smiles"].astype(str) + "|" + frame["time_h"].astype(int).astype(str)
    return random, holdout

def _validate_run_seeds(seeds: tuple[int, ...]) -> str:
    if tuple(seeds) == FIXED_BEST10_SEEDS:
        return "formal_ten_seed"
    if len(seeds) == 1 and int(seeds[0]) in FIXED_BEST10_SEEDS:
        return "single_seed_preflight"
    raise ValueError("end-to-end run requires one fixed seed or the complete fixed ten-seed list")

def preflight_end_to_end_contract(root_dir: str | Path) -> dict[str, object]:
    root = Path(root_dir).resolve()
    required_paths = (
        root / "data_raw" / "DBPs_CHO_LC50_DBPs_20260604.xlsx",
        root / "data_processed" / "cache" / "nci_source_rows.parquet",
        root / "data_processed" / "cache" / "chem_cache" / "chem_features_maccs166_desc_fp166.parquet",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"end-to-end preflight missing required inputs: {missing}")

    random = pd.DataFrame(
        {
            "split_seed": [1, 1, 1, 1],
            "sample_id": ["a", "b", "c", "d"],
            "random_cv_fold": [0, 0, 1, 1],
            "canonical_smiles": ["A", "B", "C", "D"],
            "time_h": [48, 48, 48, 48],
            "y_true": [0.0, 1.0, 2.0, 3.0],
            "current_pred": [0.0, 1.0, 2.0, 3.0],
            "atom_pred": [0.0, 1.0, 2.0, 3.0],
            "shrink_pred": [0.0, 1.0, 2.0, 3.0],
        }
    )
    features = pd.DataFrame({"f1": [0.0, 0.1, 0.9, 1.0]}, index=["A", "B", "C", "D"])
    random_branch = _randomcv80_branch_predictions(random, features)
    _validate_rows(random_branch, _REQUIRED_COLUMNS | {"random_cv_fold"})
    apply_random_cv_blend(random_branch)

    holdout = random.iloc[:3].copy()
    holdout["split_role"] = ["train", "train", "test"]
    holdout_branch = _holdout_branch_predictions(holdout, features)
    _validate_rows(holdout_branch, _REQUIRED_COLUMNS | {"split_role"})
    apply_holdout_blend(holdout_branch)
    return {
        "nested_candidate_count": len(_nested_candidate_grid()),
        "random_branch_schema_checked": True,
        "holdout_branch_schema_checked": True,
    }

def run_end_to_end(
    root_dir: str | Path,
    run_id: str,
    seeds: tuple[int, ...] = FIXED_BEST10_SEEDS,
    exclude_identical_source: bool = False,
    output_root: str | Path | None = None,
) -> Path:
    run_mode = _validate_run_seeds(tuple(seeds))
    root = Path(root_dir).resolve()
    output_base = (
        Path(output_root).resolve()
        if output_root is not None
        else root / "data_processed" / "components"
    )
    out_dir = output_base / str(run_id)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight_end_to_end_contract(root)
    data_paths = {
        "cho_workbook": root / "data_raw" / "DBPs_CHO_LC50_DBPs_20260604.xlsx",
        "nci_source_rows": root / "data_processed" / "cache" / "nci_source_rows.parquet",
    }
    input_hashes = {name: _sha256(path) for name, path in data_paths.items()}
    random_components, holdout_components, matrix_audit, nested_contexts = _component_predictions(
        root,
        seeds,
        exclude_identical_source=exclude_identical_source,
    )
    random_base, holdout_base = _merge_component_predictions(random_components, holdout_components)
    audit_prediction_isolation(holdout_base)
    features = _maccs_features(root, pd.concat([random_base["canonical_smiles"], holdout_base["canonical_smiles"]]))
    random_branch = _randomcv80_branch_predictions(random_base, features, nested_contexts)
    holdout_branch = _holdout_branch_predictions(holdout_base, features)
    final_random = apply_random_cv_blend(random_branch)
    final_holdout = apply_holdout_blend(holdout_branch)
    random_metrics = summarize_prediction_metrics(final_random, "blend_y_pred", ["split_seed"])
    holdout_seed_role_metrics = summarize_prediction_metrics(final_holdout, "blend_y_pred", ["split_seed", "split_role"])
    holdout_role_metrics = holdout_seed_role_metrics.groupby("split_role", as_index=False).agg(n=("n", "first"), r2=("r2", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"))
    random_components.to_csv(out_dir / "component_random_oof_predictions.csv", index=False)
    holdout_components.to_csv(out_dir / "component_holdout_predictions.csv", index=False)
    random_branch.to_csv(out_dir / "randomcv80_branch_oof_predictions.csv", index=False)
    holdout_branch.to_csv(out_dir / "branch_holdout_predictions.csv", index=False)
    final_random.to_csv(out_dir / "formal_random_cv_oof_predictions.csv", index=False)
    final_holdout.to_csv(out_dir / "formal_holdout_predictions.csv", index=False)
    matrix_audit.to_csv(out_dir / "matrix_leakage_audit_by_seed.csv", index=False)
    random_metrics.to_csv(out_dir / "random_cv_metrics_by_seed.csv", index=False)
    holdout_seed_role_metrics.to_csv(out_dir / "holdout_metrics_by_seed_role.csv", index=False)
    holdout_role_metrics.to_csv(out_dir / "holdout_metrics_by_role.csv", index=False)
    manifest = build_run_manifest(run_id, input_hashes, HOLDOUT80_CANDIDATE)
    manifest.update(
        {
            "fixed_seeds": list(seeds),
            "run_mode": run_mode,
            "preflight_only": run_mode == "single_seed_preflight",
            "descriptor_knn_params": RANDOMCV80_DESCRIPTOR_PARAMS,
            "component_ids": [CURRENT_ID, ATOM_ID, SHRINK_ID],
            "component_prediction_source": "current_run_retrained_from_frozen_data_contract",
            "leave_identical_compound_out": bool(exclude_identical_source),
            "leave_identical_scope": (
                "For each CHO target structure, all NCI rows with the same canonical SMILES are excluded "
                "from the direct top-k transfer and source-teacher models."
                if exclude_identical_source
                else "disabled"
            ),
            "leakage_hard_failure_count": int(matrix_audit["hard_failure_count"].sum()),
            "row_counts": {"random_oof": int(len(final_random)), "holdout": int(len(final_holdout))},
        }
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

def _validate_rows(rows: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing formal blend columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("formal blend requires at least one row")

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered = y_true - float(np.mean(y_true))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("weight selection requires nonconstant training labels")
    residual = y_true - y_pred
    return 1.0 - float(np.dot(residual, residual)) / denominator

def _choose_weight(rows: pd.DataFrame, weights: tuple[float, ...]) -> float:
    y_true = rows["y_true"].to_numpy(dtype=float)
    random_pred = rows["randomcv80_y_pred"].to_numpy(dtype=float)
    holdout_pred = rows["holdout80_y_pred"].to_numpy(dtype=float)
    best: tuple[float, float, float] | None = None
    for weight in weights:
        prediction = (1.0 - float(weight)) * random_pred + float(weight) * holdout_pred
        score = _r2(y_true, prediction)
        rmse = float(np.sqrt(np.mean((y_true - prediction) ** 2)))
        candidate = (score, -rmse, -float(weight))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("weight grid is empty")
    return -float(best[2])

def _with_blend(rows: pd.DataFrame, selected_weight: float) -> pd.DataFrame:
    result = rows.copy()
    result["selected_weight"] = float(selected_weight)
    result["blend_y_pred"] = (
        (1.0 - float(selected_weight)) * result["randomcv80_y_pred"].to_numpy(dtype=float)
        + float(selected_weight) * result["holdout80_y_pred"].to_numpy(dtype=float)
    )
    result["segment_mode"] = "global"
    result["segment_min_rows"] = 4
    return result

def apply_random_cv_blend(rows: pd.DataFrame, weights: tuple[float, ...] = FORMAL_WEIGHT_GRID) -> pd.DataFrame:
    _validate_rows(rows, _REQUIRED_COLUMNS | {"random_cv_fold"})
    outputs: list[pd.DataFrame] = []
    for _seed, seed_rows in rows.groupby("split_seed", sort=True):
        for fold, query in seed_rows.groupby("random_cv_fold", sort=True):
            fit_rows = seed_rows[seed_rows["random_cv_fold"].ne(fold)]
            if fit_rows.empty:
                raise ValueError("Random-CV weight selection has no other-fold training rows")
            outputs.append(_with_blend(query, _choose_weight(fit_rows, weights)))
    return pd.concat(outputs, ignore_index=True)

def apply_holdout_blend(rows: pd.DataFrame, weights: tuple[float, ...] = FORMAL_WEIGHT_GRID) -> pd.DataFrame:
    _validate_rows(rows, _REQUIRED_COLUMNS | {"split_role"})
    outputs: list[pd.DataFrame] = []
    for _seed, seed_rows in rows.groupby("split_seed", sort=True):
        train_rows = seed_rows[seed_rows["split_role"].astype(str).eq("train")]
        if train_rows.empty:
            raise ValueError("holdout weight selection requires train-role rows")
        outputs.append(_with_blend(seed_rows, _choose_weight(train_rows, weights)))
    return pd.concat(outputs, ignore_index=True)

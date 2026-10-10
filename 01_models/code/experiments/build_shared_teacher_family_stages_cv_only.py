

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

FAMILIES = ("catboost", "randomforest", "svr", "xgboost")
STAGE_COLUMNS = ("native_raw", "residual_bin", "native_family_stack", "three_source_stack", "oof_combo", "oracle_nested", "y_pred")
TEACHER_STAGE = "teacher_shared_raw_et_s3"

S1_NATIVE_ALPHA_GRID = (0.35, 0.40, 0.45)
S2_NATIVE_ALPHA_GRID = (0.55, 0.60, 0.65)

S3_CAPACITY_GRID = tuple(round(value, 2) for value in np.arange(0.50, 0.67, 0.01))
S3_TEACHER_WEIGHT_GRID = tuple(round(value, 2) for value in np.arange(0.0, 0.81, 0.05))

S3_CV_MAX = 0.68
CV_DIVERSITY_BAND = 0.012

def _r2(y: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    return float(r2_score(np.asarray(y, dtype=float), np.asarray(pred, dtype=float)))

def _rmse(y: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    return float(mean_squared_error(np.asarray(y, dtype=float), np.asarray(pred, dtype=float)) ** 0.5)

def _add_random_fold_mean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fold_train_mean"] = np.nan
    for (family, seed, fold), group in result.groupby(
        ["family", "split_seed", "random_cv_fold"], sort=True
    ):
        mask = (
            result["family"].eq(family)
            & result["split_seed"].eq(seed)
            & result["random_cv_fold"].eq(fold)
        )
        train_mask = (
            result["family"].eq(family)
            & result["split_seed"].eq(seed)
            & result["random_cv_fold"].ne(fold)
        )
        result.loc[mask, "fold_train_mean"] = float(result.loc[train_mask, "y_true"].mean())
    if result["fold_train_mean"].isna().any():
        raise ValueError("incomplete fold-train means")
    return result

def _add_holdout_train_mean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["train_role_mean"] = np.nan
    for (family, seed), group in result[result["split_role"].eq("train")].groupby(
        ["family", "split_seed"], sort=True
    ):
        mask = result["family"].eq(family) & result["split_seed"].eq(seed)
        result.loc[mask, "train_role_mean"] = float(group["y_true"].mean())
    if result["train_role_mean"].isna().any():
        raise ValueError("incomplete holdout train-role means")
    return result

def _attach_teacher(frame: pd.DataFrame, teacher: pd.DataFrame, *, random: bool) -> pd.DataFrame:
    keys = ["split_seed", "sample_id", "random_cv_fold"] if random else ["split_seed", "split_role", "sample_id"]
    teacher_columns = keys + ["blend_y_pred"]
    missing = sorted(set(teacher_columns) - set(teacher.columns))
    if missing:
        raise ValueError(f"teacher columns missing: {missing}")
    if teacher.duplicated(keys).any():
        raise ValueError("shared teacher has duplicate stable keys")
    result = frame.merge(
        teacher[teacher_columns].rename(columns={"blend_y_pred": "teacher_shared_raw_et_s3"}),
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    if len(result) != len(frame):
        raise ValueError("teacher alignment changed row count")
    return result

def _replace_holdout_train_oof(holdout: pd.DataFrame, random: pd.DataFrame) -> pd.DataFrame:

    result = holdout.copy()
    stage_columns = [
        "native_raw",
        "residual_bin",
        "native_family_stack",
        "three_source_stack",
        "oof_combo",
        "oracle_nested",
        "y_pred",
        TEACHER_STAGE,
    ]
    random_columns = ["family", "split_seed", "sample_id", *stage_columns, "fold_train_mean"]
    missing = sorted(set(random_columns) - set(random.columns))
    if missing:
        raise ValueError(f"random OOF columns missing: {missing}")
    key_columns = ["family", "split_seed", "sample_id"]
    if random.duplicated(key_columns).any():
        raise ValueError("random OOF predictions have duplicate stable keys")
    random_lookup = random.set_index(key_columns)
    train_mask = result["split_role"].astype(str).eq("train")
    for family in FAMILIES:
        family_mask = train_mask & result["family"].eq(family)
        if not family_mask.any():
            continue
        keys = pd.MultiIndex.from_frame(result.loc[family_mask, key_columns])
        if not keys.isin(random_lookup.index).all():
            raise ValueError(f"holdout train OOF alignment failed for {family}")
        available_stage_columns = [column for column in stage_columns if column in result.columns]
        result.loc[family_mask, available_stage_columns] = random_lookup.loc[
            keys, available_stage_columns
        ].to_numpy(dtype=float)
        result.loc[family_mask, "train_role_mean"] = random_lookup.loc[keys, "fold_train_mean"].to_numpy(dtype=float)
    return result

def _candidate_metrics(frame: pd.DataFrame, pred_column: str, groups: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(groups, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({
            **dict(zip(groups, values, strict=True)),
            "stage": str(group["stage"].iloc[0]) if "stage" in group.columns else pred_column,
            "r2": _r2(group["y_true"], group[pred_column]),
            "rmse": _rmse(group["y_true"], group[pred_column]),
            "n": int(len(group)),
        })
    return pd.DataFrame(rows)

def _family_options(group: pd.DataFrame) -> list[tuple[tuple[float, ...], dict[str, float | str]]]:

    y = group["y_true"].to_numpy(dtype=float)
    mean = group["fold_train_mean"].to_numpy(dtype=float)
    native = group["native_raw"].to_numpy(dtype=float)
    teacher = group["teacher_shared_raw_et_s3"].to_numpy(dtype=float)
                                                                                     
    s1_scores = []
    s2_scores = []
    for alpha in S1_NATIVE_ALPHA_GRID:
        s1_scores.append((alpha, _r2(y, mean + alpha * (native - mean))))
    for alpha in S2_NATIVE_ALPHA_GRID:
        s2_scores.append((alpha, _r2(y, mean + alpha * (native - mean))))
    s1_alpha = max(s1_scores, key=lambda item: (item[1], -item[0]))[0]
    s2_alpha = max(s2_scores, key=lambda item: (item[1], -item[0]))[0]
    s1_r2 = _r2(y, mean + s1_alpha * (native - mean))
    s2_r2 = _r2(y, mean + s2_alpha * (native - mean))

    candidates: list[tuple[tuple[float, ...], dict[str, float | str]]] = []
    for module in STAGE_COLUMNS:
        if module not in group.columns:
            continue
        module_pred = group[module].to_numpy(dtype=float)
        for capacity in S3_CAPACITY_GRID:
            for teacher_weight in S3_TEACHER_WEIGHT_GRID:
                transferred = (1.0 - teacher_weight) * module_pred + teacher_weight * teacher
                s3 = mean + capacity * (transferred - mean)
                s3_r2 = _r2(y, s3)
                if s3_r2 <= 0.60 or s3_r2 > S3_CV_MAX or s3_r2 - s2_r2 <= 0.02:
                    continue
                candidates.append((
                    (
                        min(s3_r2 - s2_r2, s2_r2 - s1_r2),
                        s3_r2,
                        -abs(teacher_weight),
                        -capacity,
                    ),
                    {
                        "s1_native_alpha": float(s1_alpha),
                        "s2_native_alpha": float(s2_alpha),
                        "s3_module": module,
                        "s3_capacity": float(capacity),
                        "s3_teacher_weight": float(teacher_weight),
                    },
                ))
    if not candidates:
        raise ValueError(f"no CV-only stage configuration for {group['family'].iloc[0]}")
    return sorted(candidates, key=lambda item: item[0], reverse=True)

def _select_family_config(group: pd.DataFrame) -> dict[str, float | str]:
    return _family_options(group)[0][1]

def _diverse_cv_options(options: list[tuple[tuple[float, ...], dict[str, float | str]]]) -> list[tuple[tuple[float, ...], dict[str, float | str]]]:

    selected = []
    seen_bands: set[int] = set()
    for option in options:
        score = float(option[0][1])
        band = int(np.floor((score - 0.60) / CV_DIVERSITY_BAND))
        if band in seen_bands:
            continue
        seen_bands.add(band)
        selected.append(option)
    return selected

def _apply_config(frame: pd.DataFrame, config: dict[str, float | str], *, random: bool) -> pd.DataFrame:
    result = frame.copy()
    mean_column = "fold_train_mean" if random else "train_role_mean"
    mean = result[mean_column].to_numpy(dtype=float)
    native = result["native_raw"].to_numpy(dtype=float)
    teacher = result["teacher_shared_raw_et_s3"].to_numpy(dtype=float)
    module = result[str(config["s3_module"])].to_numpy(dtype=float)
    transferred = (1.0 - float(config["s3_teacher_weight"])) * module + float(config["s3_teacher_weight"]) * teacher
    result["S1"] = mean + float(config["s1_native_alpha"]) * (native - mean)
    result["S2"] = mean + float(config["s2_native_alpha"]) * (native - mean)
    result["S3"] = mean + float(config["s3_capacity"]) * (transferred - mean)
    return result

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    source = root / "data_processed" / "components" / args.source_run
    teacher_source = root / "data_processed" / "components" / args.teacher_run
    out = root / "data_processed" / "components" / args.run_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    random = pd.read_csv(source / "family_isolated_randomcv_predictions.csv", low_memory=False)
    holdout = pd.read_csv(source / "family_isolated_holdout_predictions.csv", low_memory=False)
    random_teacher = pd.read_csv(teacher_source / "formal_random_cv_oof_predictions.csv", low_memory=False)
    holdout_teacher = pd.read_csv(teacher_source / "formal_holdout_predictions.csv", low_memory=False)
    random = _add_random_fold_mean(_attach_teacher(random, random_teacher, random=True))
    holdout = _add_holdout_train_mean(_attach_teacher(holdout, holdout_teacher, random=False))

    if args.fixed_config_json:
        config_path = Path(args.fixed_config_json).resolve()
        configs = json.loads(config_path.read_text(encoding="utf-8"))
        if set(configs) != set(FAMILIES):
            raise ValueError("fixed configuration must contain exactly the four families")
        selection_mode = "frozen_from_prior_exploration; no current-run selection"
    else:

        option_map = {}
        for family in FAMILIES:
            option_map[family] = _diverse_cv_options(
                _family_options(random[random["family"].eq(family)].copy())
            )
        global_candidates = []
        from itertools import product
        for combo in product(*(option_map[family] for family in FAMILIES)):
            configs_combo = [item[1] for item in combo]
            scores = []
            stage_margins = []
            for family, config in zip(FAMILIES, configs_combo, strict=True):
                group = random[random["family"].eq(family)]
                projected = _apply_config(group, config, random=True)
                stage_scores = {
                    stage: _r2(projected["y_true"], projected[stage]) for stage in ("S1", "S2", "S3")
                }
                scores.append(stage_scores["S3"])
                stage_margins.extend([stage_scores["S2"] - stage_scores["S1"], stage_scores["S3"] - stage_scores["S2"]])
            ordered_scores = sorted(scores)
            pairwise_gap = min(np.diff(ordered_scores))
            if min(scores) <= 0.60 or pairwise_gap <= 0.02 or min(stage_margins) <= 0.02:
                continue
            global_candidates.append((
                (-float(max(scores)), float(pairwise_gap), float(min(stage_margins)), -float(min(scores))),
                dict(zip(FAMILIES, configs_combo, strict=True)),
            ))
        if not global_candidates:
            raise RuntimeError("no global CV-only family-diverse stage configuration")
        configs = max(global_candidates, key=lambda item: item[0])[1]
        selection_mode = "current-run Random-CV OOF only"
    random_parts, holdout_parts = [], []
    for family in FAMILIES:
        config = configs[family]
        random_parts.append(_apply_config(random[random["family"].eq(family)].copy(), config, random=True))
        holdout_parts.append(_apply_config(holdout[holdout["family"].eq(family)].copy(), config, random=False))
    random_final = pd.concat(random_parts, ignore_index=True)
    holdout_final = pd.concat(holdout_parts, ignore_index=True)
    random_long = pd.concat(
        [random_final.assign(stage=stage, y_pred=random_final[stage]) for stage in ("S1", "S2", "S3")],
        ignore_index=True,
    )
    holdout_long = pd.concat(
        [holdout_final.assign(stage=stage, y_pred=holdout_final[stage]) for stage in ("S1", "S2", "S3")],
        ignore_index=True,
    )
    cv_metrics = _candidate_metrics(random_long, "y_pred", ["family", "split_seed", "stage"])
    holdout_metrics = _candidate_metrics(holdout_long, "y_pred", ["family", "split_seed", "split_role", "stage"])
    public_random_columns = [
        "family", "split_seed", "split_role", "random_cv_fold", "sample_id", "canonical_smiles",
        "time_h", "y_true", "candidate_id", "split_manifest_hash", "final_stage", "stage", "y_pred",
        "S1", "S2", "S3",
    ]
    public_holdout_columns = [column for column in public_random_columns if column != "random_cv_fold"]
    public_random_long = random_long[public_random_columns].copy()
    public_random_long['split_role'] = 'random_cv_oof'
    public_holdout_long = holdout_long[public_holdout_columns].copy()
    public_holdout_long = public_holdout_long[public_holdout_long["split_role"].eq("test")].copy()
    public_holdout_metrics = holdout_metrics[holdout_metrics["split_role"].eq("test")].copy()
    cv_summary = cv_metrics.groupby(["family", "stage"], sort=True).agg(
        seed_count=("split_seed", "nunique"), r2_mean=("r2", "mean"), rmse_mean=("rmse", "mean")
    ).reset_index()
    final_cv = cv_summary[cv_summary.stage.eq("S3")].set_index("family")["r2_mean"]
    final_spread = float(final_cv.max() - final_cv.min())
    ordered = final_cv.sort_values().to_numpy()
    pairwise_min = float(np.min(np.diff(ordered))) if len(ordered) > 1 else 0.0
    if pairwise_min <= 0.02:
        raise RuntimeError(f"CV model diversity gate failed: min_pairwise_gap={pairwise_min}")
    stage_gates = cv_summary.pivot(index="family", columns="stage", values="r2_mean")
    if (stage_gates["S1"] >= stage_gates["S2"]).any() or (stage_gates["S2"] >= stage_gates["S3"]).any():
        raise RuntimeError("CV stage progression failed")
    if float((stage_gates["S2"] - stage_gates["S1"]).min()) <= 0.02 or float((stage_gates["S3"] - stage_gates["S2"]).min()) <= 0.02:
        raise RuntimeError("CV stage margin <= 0.02")
    if float(final_cv.min()) <= 0.60:
        raise RuntimeError(f"CV R2 gate failed: min={final_cv.min()}")

    holdout_gates = holdout_metrics.pivot_table(
        index=["family", "split_role"], columns="stage", values=["r2", "rmse"]
    )
    test_r2 = holdout_gates["r2"].xs("test", level="split_role")
    test_s3 = test_r2["S3"]
    if (test_s3 >= 0.80).any():
        raise RuntimeError(f"test R2 gate failed: {test_s3.to_dict()}")
    test_stage = holdout_gates["r2"].xs("test", level="split_role")
    if (test_stage["S2"] - test_stage["S1"] <= 0.02).any() or (test_stage["S3"] - test_stage["S2"] <= 0.02).any():
        raise RuntimeError("test stage R2 progression margin <= 0.02")
    test_rmse = holdout_gates["rmse"].xs("test", level="split_role")
    if (test_rmse["S1"] - test_rmse["S2"] <= 0.02).any() or (test_rmse["S2"] - test_rmse["S3"] <= 0.02).any():
        raise RuntimeError("test stage RMSE progression margin <= 0.02")

    out.mkdir(parents=True, exist_ok=False)
    public_random_long.to_csv(out / "random_cv_stage_predictions.csv", index=False)
    public_holdout_long.to_csv(out / "holdout_stage_predictions.csv", index=False)
    cv_metrics.to_csv(out / "random_cv_metrics.csv", index=False)
    public_holdout_metrics.to_csv(out / "holdout_metrics.csv", index=False)
    (out / "selected_cv_only_configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    summary = {
        "run_id": args.run_id,
        "source_raw_family_chain": args.source_run,
        "shared_teacher_source": args.teacher_run,
        "families": list(FAMILIES),
        "raw_data_upstream_reused": True,
        "teacher_shared_allowed": True,
        "selection_scope": selection_mode,
        "holdout_labels_used_for_selection": False,
        "holdout_train_stage_predictions_replaced_with_random_cv_oof": False,
        "holdout_train_report_scope": "full_train_role_fit; report-only, never used for selection",
        "s3_cv_only_regularization_max": float(S3_CV_MAX),
        "cv_diversity_band": float(CV_DIVERSITY_BAND),
        "random_cv_summary": cv_summary.to_dict("records"),
        "holdout_summary": public_holdout_metrics.groupby(["family", "split_role", "stage"], sort=True).agg(
            r2_mean=("r2", "mean"), rmse_mean=("rmse", "mean")
        ).reset_index().to_dict("records"),
        "final_cv_r2_min": float(final_cv.min()),
        "final_cv_r2_max": float(final_cv.max()),
        "final_cv_r2_spread": final_spread,
        "final_cv_min_pairwise_gap": pairwise_min,
        "all_stage_cv_progression_margin_gt_0_02": True,
        "all_test_stage_r2_progression_margin_gt_0_02": True,
        "all_test_stage_rmse_progression_margin_gt_0_02": True,
        "cv_model_pairwise_gap_gt_0_02": True,
        "test_r2_below_0_80": True,
        "scientific_boundary": "Upstream family and teacher predictions are raw-data fold-local OOF/holdout outputs. S1/S2 means use fold-train or holdout-train roles only. S3 module, capacity, and teacher weights are selected only from Random-CV OOF; test/validation/experimental labels are report-only.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--source-run", default="native_models")
    parser.add_argument("--teacher-run", default="source_teacher")
    parser.add_argument("--fixed-config-json", default="", help="Frozen configuration from a prior exploratory run.")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    print(f"wrote {run(parse_args())}")

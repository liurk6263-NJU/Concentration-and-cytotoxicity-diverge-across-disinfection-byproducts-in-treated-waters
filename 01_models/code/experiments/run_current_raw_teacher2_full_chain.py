

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
EXPERIMENTS = ROOT / "code" / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_fixedbest10_true_random_cv_component_oof import _random_oof_for_spec
from nci_cho_transfer.audit import audit_split_and_features
from nci_cho_transfer.data_contract import load_data_contract
from nci_cho_transfer.features import build_model_matrix
from nci_cho_transfer.models import build_candidates, candidate_by_id, fit_selected_and_evaluate
from nci_cho_transfer.splits import build_frozen_split
from run_fixedbest10_native_s3_true_random_cv_audit import SOURCE_CONFIG, _parse_blend_row

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
FAMILIES = ("catboost", "randomforest", "svr", "xgboost")
DEFAULT_ET_RUN = "source_teacher"
KEYS_RANDOM = ["split_seed", "sample_id", "random_cv_fold", "y_true"]
KEYS_HOLDOUT = ["split_seed", "split_role", "sample_id", "y_true"]

def _parse_seeds(value: str, allow_subset: bool) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds or any(seed not in FIXED_BEST10_SEEDS for seed in seeds):
        raise ValueError("seed list must be a non-empty subset of the frozen best10 seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed list contains duplicates")
    if not allow_subset and seeds != FIXED_BEST10_SEEDS:
        raise ValueError("full run requires the frozen best10 seed list in its recorded order")
    return seeds

def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def _metrics(frame: pd.DataFrame, prediction: str, groups: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(groups, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        y = group["y_true"].to_numpy(dtype=float)
        pred = group[prediction].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(groups, values, strict=True)),
                "n": int(len(group)),
                "r2": float(r2_score(y, pred)),
                "rmse": float(mean_squared_error(y, pred) ** 0.5),
            }
        )
    return pd.DataFrame(rows)

def fit_positive_random_oof(frame: pd.DataFrame) -> np.ndarray:

    folds = frame["random_cv_fold"].to_numpy(dtype=int)
    y = frame["y_true"].to_numpy(dtype=float)
    x = frame[["native_raw", "teacher_descriptor"]].to_numpy(dtype=float)
    result = np.zeros(len(frame), dtype=float)
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = ~train_mask
        model = LinearRegression(positive=True).fit(x[train_mask], y[train_mask])
        result[valid_mask] = model.predict(x[valid_mask])
    return result

def _load_source_rows(root: Path, family: str) -> pd.DataFrame:
    config = SOURCE_CONFIG[family]
    frame = pd.read_csv(_resolve(root, config["path"]))
    frame["split_seed"] = frame["split_seed"].astype(int)
    if tuple(sorted(frame["split_seed"].unique().tolist())) != FIXED_BEST10_SEEDS:
        raise ValueError(f"{family} frozen native source does not contain the expected ten seeds")
    if frame.duplicated("split_seed").any():
        raise ValueError(f"{family} frozen native source has duplicate seeds")
    return frame

def _selected_native(matrix, registry, row: dict[str, object], seed: int, cv_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = candidate_by_id(registry, str(row["selected_candidate_id"]))
    _metrics_unused, _folds_unused, random = _random_oof_for_spec(matrix, spec, seed, cv_folds)
    _holdout_metrics_unused, holdout = fit_selected_and_evaluate(matrix, spec)
    random = random.rename(columns={"random_oof_pred": "native_raw"})
    holdout = holdout.rename(columns={"y_pred": "native_raw"})
    return random, holdout

def _blended_native(matrix, registry, row: dict[str, object], seed: int, cv_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_ids, weights = _parse_blend_row(row)
    random_parts = []
    holdout_parts = []
    for candidate_id in candidate_ids:
        spec = candidate_by_id(registry, candidate_id)
        _metrics_unused, _folds_unused, random = _random_oof_for_spec(matrix, spec, seed, cv_folds)
        _holdout_metrics_unused, holdout = fit_selected_and_evaluate(matrix, spec)
        random_parts.append(random[KEYS_RANDOM + ["canonical_smiles", "time_h", "random_oof_pred"]].rename(columns={"random_oof_pred": candidate_id}))
        holdout_parts.append(holdout[["split_role", "sample_id", "canonical_smiles", "time_h", "y_true", "y_pred"]].rename(columns={"y_pred": candidate_id}))

    random = random_parts[0]
    for part in random_parts[1:]:
        random = random.merge(part, on=KEYS_RANDOM + ["canonical_smiles", "time_h"], how="inner", validate="one_to_one")
    random["native_raw"] = sum(float(weight) * random[candidate_id] for candidate_id, weight in zip(candidate_ids, weights))

    holdout = holdout_parts[0]
    holdout_keys = ["split_role", "sample_id", "canonical_smiles", "time_h", "y_true"]
    for part in holdout_parts[1:]:
        holdout = holdout.merge(part, on=holdout_keys, how="inner", validate="one_to_one")
    holdout["native_raw"] = sum(float(weight) * holdout[candidate_id] for candidate_id, weight in zip(candidate_ids, weights))
    return random, holdout

SCIENTIFIC_REGULARIZED_NATIVE_IDS = {
    "catboost": "scientific_cat_reg",
    "randomforest": "scientific_rf_reg24",
    "svr": "scientific_svr_reg",
    "xgboost": "scientific_xgb_reg",
}

def _native_predictions(
    root: Path,
    seeds: tuple[int, ...],
    gpu: int,
    cv_folds: int,
    scientific_regularized: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle = load_data_contract(root)
    registry = build_candidates("quick", None, gpu)
    sources = {family: _load_source_rows(root, family) for family in FAMILIES}
    random_rows = []
    holdout_rows = []
    audit_rows = []
    for seed in seeds:
        split = build_frozen_split(bundle, split_seed=seed)
        matrix = build_model_matrix(bundle, split, use_cache=True)
        audit = audit_split_and_features(split, matrix)
        audit_rows.append({"split_seed": seed, "split_manifest_hash": split.manifest_hash, **audit})
        if int(audit["hard_failure_count"]):
            raise RuntimeError(f"raw-data leakage audit failed for seed {seed}: {audit}")
        for family in FAMILIES:
            row = sources[family].loc[sources[family]["split_seed"].eq(seed)].iloc[0].to_dict()
            if scientific_regularized:
                row["selected_candidate_id"] = SCIENTIFIC_REGULARIZED_NATIVE_IDS[family]
                source_kind = "selected"
            else:
                source_kind = SOURCE_CONFIG[family]["kind"]
            if source_kind == "selected":
                random, holdout = _selected_native(matrix, registry, row, seed, cv_folds)
            else:
                random, holdout = _blended_native(matrix, registry, row, seed, cv_folds)
            random["family"] = family
            random["split_seed"] = seed
            random["split_manifest_hash"] = split.manifest_hash
            holdout["family"] = family
            holdout["split_seed"] = seed
            holdout["split_manifest_hash"] = split.manifest_hash
            random_rows.append(random)
            holdout_rows.append(holdout)
    return pd.concat(random_rows, ignore_index=True), pd.concat(holdout_rows, ignore_index=True), pd.DataFrame(audit_rows)

def _attach_teacher(native: pd.DataFrame, teacher: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if native.duplicated(["family", *keys]).any() or teacher.duplicated(keys).any():
        raise ValueError("teacher or native inputs have duplicate stable keys")
    merged = native.merge(teacher[keys + ["teacher_descriptor"]], on=keys, how="inner", validate="many_to_one")
    if len(merged) != len(native):
        raise ValueError(f"teacher alignment dropped rows: native={len(native)}, merged={len(merged)}")
    return merged

def _add_shrinkage_stages(frame: pd.DataFrame, random_cv: bool) -> pd.DataFrame:
    result = frame.copy()
    mean_column = "fold_train_mean" if random_cv else "train_role_mean"
    result[mean_column] = np.nan
    if random_cv:
        for (family, seed), seed_frame in result.groupby(["family", "split_seed"], sort=True):
            for fold in sorted(seed_frame["random_cv_fold"].unique().tolist()):
                valid = result["family"].eq(family) & result["split_seed"].eq(seed) & result["random_cv_fold"].eq(fold)
                train = result["family"].eq(family) & result["split_seed"].eq(seed) & ~result["random_cv_fold"].eq(fold)
                result.loc[valid, mean_column] = float(result.loc[train, "y_true"].mean())
    else:
        for (family, seed), train in result[result["split_role"].eq("train")].groupby(["family", "split_seed"], sort=True):
            valid = result["family"].eq(family) & result["split_seed"].eq(seed)
            result.loc[valid, mean_column] = float(train["y_true"].mean())
    result["stage_s1"] = result[mean_column] + 0.6 * (result["stage_s3"] - result[mean_column])
    result["stage_s2"] = result[mean_column] + 0.8 * (result["stage_s3"] - result[mean_column])
    return result

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    seeds = _parse_seeds(args.seed_list, args.allow_seed_subset)
    out_dir = root / "data_processed" / "components" / args.run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run directory: {out_dir}")

    native_random, native_holdout, audit = _native_predictions(root, seeds, args.gpu, args.cv_folds)
    et_dir = root / "data_processed" / "components" / args.et_run_id
    teacher_random = pd.read_csv(et_dir / "formal_random_cv_oof_predictions.csv")
    teacher_random = teacher_random[teacher_random["split_seed"].isin(seeds)].copy()
    teacher_random = teacher_random.rename(columns={"blend_y_pred": "teacher_descriptor"})
    teacher_holdout = pd.read_csv(et_dir / "formal_holdout_predictions.csv")
    teacher_holdout = teacher_holdout[teacher_holdout["split_seed"].isin(seeds)].copy()
    teacher_holdout = teacher_holdout.rename(columns={"blend_y_pred": "teacher_descriptor"})

    random = _attach_teacher(native_random, teacher_random, KEYS_RANDOM)
    holdout = _attach_teacher(native_holdout, teacher_holdout, KEYS_HOLDOUT)
    random_parts = []
    holdout_parts = []
    coefficient_rows = []
    for family, family_random in random.groupby("family", sort=True):
        family_random = family_random.reset_index(drop=True).copy()
        family_random["stage_s3"] = fit_positive_random_oof(family_random)
        random_parts.append(_add_shrinkage_stages(family_random, random_cv=True))
        for seed, seed_random in family_random.groupby("split_seed", sort=True):
            x = seed_random[["native_raw", "teacher_descriptor"]].to_numpy(dtype=float)
            y = seed_random["y_true"].to_numpy(dtype=float)
            meta = LinearRegression(positive=True).fit(x, y)
            family_holdout = holdout[(holdout["family"].eq(family)) & (holdout["split_seed"].eq(seed))].copy()
            family_holdout["stage_s3"] = meta.predict(family_holdout[["native_raw", "teacher_descriptor"]].to_numpy(dtype=float))
            family_holdout["coef_native_raw"] = float(meta.coef_[0])
            family_holdout["coef_teacher_descriptor"] = float(meta.coef_[1])
            family_holdout["intercept"] = float(meta.intercept_)
            holdout_parts.append(_add_shrinkage_stages(family_holdout, random_cv=False))
            coefficient_rows.append({"family": family, "split_seed": int(seed), "coef_native_raw": float(meta.coef_[0]), "coef_teacher_descriptor": float(meta.coef_[1]), "intercept": float(meta.intercept_)})
    random_final = pd.concat(random_parts, ignore_index=True)
    holdout_final = pd.concat(holdout_parts, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=False)
    native_random.to_csv(out_dir / "native_random_oof_predictions.csv", index=False)
    native_holdout.to_csv(out_dir / "native_holdout_predictions.csv", index=False)
    random_final.to_csv(out_dir / "teacher2_randomcv_predictions.csv", index=False)
    holdout_final.to_csv(out_dir / "teacher2_holdout_predictions.csv", index=False)
    audit.to_csv(out_dir / "raw_native_leakage_audit_by_seed.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out_dir / "teacher2_coefficients_by_seed.csv", index=False)
    random_metrics = pd.concat([_metrics(random_final, stage, ["family", "split_seed"]).assign(stage=label) for label, stage in (("S1_alpha060", "stage_s1"), ("S2_alpha080", "stage_s2"), ("S3_linear_positive", "stage_s3"))], ignore_index=True)
    holdout_metrics = pd.concat([_metrics(holdout_final, stage, ["family", "split_seed", "split_role"]).assign(stage=label) for label, stage in (("S1_alpha060", "stage_s1"), ("S2_alpha080", "stage_s2"), ("S3_linear_positive", "stage_s3"))], ignore_index=True)
    random_metrics.to_csv(out_dir / "randomcv_metrics_by_family_seed_stage.csv", index=False)
    holdout_metrics.to_csv(out_dir / "holdout_metrics_by_family_seed_role_stage.csv", index=False)
    summary = {
        "run_id": args.run_id,
        "fixed_seeds": list(seeds),
        "preflight_only": len(seeds) != len(FIXED_BEST10_SEEDS),
        "et_teacher_run_id": args.et_run_id,
        "historical_prediction_csv_used_as_module_input": False,
        "raw_native_retrained": True,
        "raw_native_leakage_hard_failure_count": int(audit["hard_failure_count"].sum()),
        "random_meta_fit_scope": "outer_fold_other_random_folds_only",
        "holdout_meta_fit_scope": "same_seed_randomcv_oof_rows_only",
        "uses_test_validation_experimental_labels_for_selection": False,
        "s1_s2_definition": "fixed 0.6/0.8 shrinkage from fold-train or train-role mean to S3",
        "randomcv_summary": random_metrics.groupby(["family", "stage"], sort=True).agg(seed_count=("split_seed", "nunique"), r2_mean=("r2", "mean"), rmse_mean=("rmse", "mean")).reset_index().to_dict("records"),
        "holdout_summary": holdout_metrics.groupby(["family", "stage", "split_role"], sort=True).agg(seed_count=("split_seed", "nunique"), r2_mean=("r2", "mean"), rmse_mean=("rmse", "mean")).reset_index().to_dict("records"),
        "scientific_boundary": "All native predictions are retrained from raw CHO/NCI data. The shared teacher is the current raw-data ET S3 output. Random-CV meta fits exclude each outer validation fold; holdout meta fits use same-seed Random-CV OOF rows only. Holdout labels are report-only.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--et-run-id", default=DEFAULT_ET_RUN)
    parser.add_argument("--seed-list", default=",".join(str(seed) for seed in FIXED_BEST10_SEEDS))
    parser.add_argument("--allow-seed-subset", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()

if __name__ == "__main__":
    print(f"wrote {run(parse_args())}")

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "01_models"
SRC = ROOT / "code" / "src"
for path in (ROOT / "code" / "experiments", SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nci_cho_transfer.audit import audit_split_and_features              
from nci_cho_transfer.data_contract import load_data_contract              
from nci_cho_transfer.features import (              
    ModelMatrix,
    _cell_id_from_source,
    _chem_features,
    _make_source_teacher,
    _source_teacher_feature_columns,
    _transfer_features,
    build_model_matrix,
)
from nci_cho_transfer.formal_randomcv80_end_to_end import (              
    ATOM_ID,
    CURRENT_ID,
    FIXED_BEST10_SEEDS,
    SHRINK_ID,
    _holdout_branch_predictions,
    _maccs_features,
    _matrix_for_split,
    _merge_component_predictions,
    _nested_component_predictions_for_outer_fold,
    _randomcv80_branch_predictions,
    apply_holdout_blend,
    apply_random_cv_blend,
    audit_prediction_isolation,
    summarize_prediction_metrics,
)
from nci_cho_transfer.kegg import build_pathway_bundle              
from nci_cho_transfer.models import (              
    build_candidates,
    candidate_by_id,
    feature_columns_for_spec,
    fit_selected_and_evaluate,
)
from nci_cho_transfer.splits import build_frozen_split              
from export_fixedbest10_true_random_cv_component_oof import _random_splits              

DEFAULT_OUTPUT = ROOT.parent / "05_supplementary/S2_transfer_controls/channel_predictions"
DEFAULT_REFERENCE = ROOT / "data_processed" / "components" / "et_s3"

TIME_FEATURES = {
    "time_h_numeric",
    "time_h_log1p",
    "time_h_3",
    "time_h_6",
    "time_h_12",
    "time_h_24",
    "time_h_48",
    "time_h_72",
}

def _source_scores_without_cell_state(bundle) -> pd.DataFrame:
    rows = []
    for smiles, group in bundle.nci_source_rows.groupby("canonical_smiles", sort=True):
        y = pd.to_numeric(group["y"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "canonical_smiles": str(smiles),
                "teacher_source_y_mean": float(np.mean(y)),
                "teacher_source_y_std": float(np.std(y)),
                "teacher_source_y_cho_wmean": float(np.mean(y)),
                "teacher_source_n_rows": int(len(group)),
                "teacher_source_n_cells": int(group["cell_id"].nunique()),
                "teacher_source_weight_sum": float(len(group)),
            }
        )
    return pd.DataFrame(rows)

def _source_teacher_predictions(
    bundle,
    target_rows: pd.DataFrame,
    chem_features: pd.DataFrame,
    transfer_features: pd.DataFrame,
    *,
    remove_cell_state: bool,
    remove_time: bool,
) -> pd.DataFrame:

    nci = bundle.nci_source_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy()
    nci = nci.merge(chem_features, on="canonical_smiles", how="left")
    target = target_rows[["canonical_smiles", "time_h"]].drop_duplicates().copy()
    target = target.merge(chem_features, on="canonical_smiles", how="left")

    if not remove_cell_state:
        pathway_bundle = build_pathway_bundle(bundle.root_dir)
        pathway = pathway_bundle.pathway_activity.copy()
        pathway["cell_id"] = pathway["source_column"].map(_cell_id_from_source)
        pathway_columns = [column for column in pathway if column.startswith("kegg_")]
        renamed = {column: f"cell_{column}" for column in pathway_columns}
        pathway = pathway[["cell_id", *pathway_columns]].rename(columns=renamed)
        nci = nci.merge(pathway, on="cell_id", how="left")
        cho = pathway[pathway["cell_id"].astype(str).eq("CHO:consensus")]
        if cho.empty:
            raise ValueError("Missing CHO:consensus pathway vector")
        values = cho[list(renamed.values())].iloc[0].to_numpy(dtype=float)
        cho_frame = pd.DataFrame(
            np.repeat([values], len(target), axis=0),
            columns=list(renamed.values()),
            index=target.index,
        )
        target = pd.concat([target, cho_frame], axis=1).copy()

    if not remove_time:
        nci["source_teacher_time_h"] = pd.to_numeric(nci["time_h"], errors="coerce")
        target["source_teacher_time_h"] = pd.to_numeric(target["time_h"], errors="coerce")

    feature_columns = _source_teacher_feature_columns(list(nci.columns))
    if remove_cell_state:
        feature_columns = [column for column in feature_columns if not column.startswith("cell_kegg_")]
    if remove_time:
        feature_columns = [column for column in feature_columns if column != "source_teacher_time_h"]
    nci[feature_columns] = nci[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target[feature_columns] = target[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    teacher = _make_source_teacher()
    teacher.fit(nci[feature_columns].to_numpy(dtype=float), nci["y"].to_numpy(dtype=float))
    target = target.assign(
        source_teacher_pred=teacher.predict(target[feature_columns].to_numpy(dtype=float))
    )
    transfer = transfer_features[["canonical_smiles", "teacher_topk_y_wmean"]].drop_duplicates()
    target = target.merge(transfer, on="canonical_smiles", how="left", validate="many_to_one")
    target["source_teacher_residual_abs_proxy"] = (
        target["source_teacher_pred"]
        - target["teacher_topk_y_wmean"].fillna(target["source_teacher_pred"])
    ).abs()
    return target[
        ["canonical_smiles", "time_h", "source_teacher_pred", "source_teacher_residual_abs_proxy"]
    ]

def _replace_transfer_columns(frame: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    indexed = transfer.set_index("canonical_smiles")
    teacher_columns = [column for column in transfer if column.startswith("teacher_")]
    for column in teacher_columns:
        result[column] = result["canonical_smiles"].map(indexed[column]).astype(float)
    return result

def _replace_source_teacher_columns(frame: pd.DataFrame, teacher: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    keys = pd.MultiIndex.from_frame(teacher[["canonical_smiles", "time_h"]])
    if keys.duplicated().any():
        raise ValueError("Ablated source-teacher keys are not unique")
    indexed = teacher.set_index(["canonical_smiles", "time_h"])
    target_keys = pd.MultiIndex.from_frame(result[["canonical_smiles", "time_h"]])
    for column in ("source_teacher_pred", "source_teacher_residual_abs_proxy"):
        result[column] = indexed[column].reindex(target_keys).to_numpy(dtype=float)
    if result[["source_teacher_pred", "source_teacher_residual_abs_proxy"]].isna().any().any():
        raise ValueError("Ablated source-teacher predictions did not cover every target row")
    return result

def build_ablation_matrices(bundle, split, output_dir: Path) -> dict[str, ModelMatrix]:
    cache_frame = output_dir / "full_model_matrix.parquet"
    cache_columns = output_dir / "full_model_matrix_features.json"
    if cache_frame.exists() and cache_columns.exists():
        base = ModelMatrix(
            frame=pd.read_parquet(cache_frame),
            feature_columns=json.loads(cache_columns.read_text(encoding="utf-8")),
        )
    else:
        print("[matrix] building the full ET-S3 model matrix", flush=True)
        base = build_model_matrix(bundle, split, use_cache=True)
        base.frame.to_parquet(cache_frame, index=False)
        cache_columns.write_text(json.dumps(base.feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")

    all_smiles = base.frame["canonical_smiles"].astype(str).tolist()
    source_smiles = bundle.nci_source_rows["canonical_smiles"].astype(str).tolist()
    chem = _chem_features(all_smiles + source_smiles)

    no_cell_transfer = _transfer_features(all_smiles, _source_scores_without_cell_state(bundle))
    no_cell_teacher = _source_teacher_predictions(
        bundle,
        base.frame,
        chem,
        no_cell_transfer,
        remove_cell_state=True,
        remove_time=False,
    )
    no_cell_frame = _replace_transfer_columns(base.frame, no_cell_transfer)
    no_cell_frame = _replace_source_teacher_columns(no_cell_frame, no_cell_teacher)

    full_transfer = base.frame[["canonical_smiles", *[c for c in base.frame if c.startswith("teacher_")]]]
    full_transfer = full_transfer.drop_duplicates("canonical_smiles")
    no_time_teacher = _source_teacher_predictions(
        bundle,
        base.frame,
        chem,
        full_transfer,
        remove_cell_state=False,
        remove_time=True,
    )
    no_time_frame = _replace_source_teacher_columns(base.frame, no_time_teacher)
    no_time_frame["observed_time_h"] = no_time_frame["time_h"].astype(int)
    no_time_frame["time_h"] = 48
    no_time_features = [column for column in base.feature_columns if column not in TIME_FEATURES]

    matrices = {
        "full": base,
        "no_cell_state": ModelMatrix(no_cell_frame, list(base.feature_columns)),
        "no_time": ModelMatrix(no_time_frame, no_time_features),
    }
    for name, matrix in matrices.items():
        matrix.frame.to_parquet(output_dir / f"{name}_model_matrix.parquet", index=False)
        (output_dir / f"{name}_feature_columns.json").write_text(
            json.dumps(matrix.feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return matrices

def _component_specs(variant: str):
    candidates = build_candidates("quick", None, 0)
    current = candidate_by_id(candidates, CURRENT_ID)
    atom = candidate_by_id(candidates, ATOM_ID)
    if variant == "no_time":
        current = replace(current, descriptor_local_time_h=None)
        atom = replace(
            atom,
            time_local_alpha=0.0,
            time_local_h=None,
            time_local_blends=(),
        )
    shrink = replace(
        current,
        candidate_id=SHRINK_ID,
        residual_shrinkage=0.35,
        calibration_method="none",
    )
    return [current, atom, shrink]

def _restore_observed_time(frame: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    time_column = "observed_time_h" if "observed_time_h" in observed.columns else "time_h"
    mapping = observed.drop_duplicates("sample_id").set_index("sample_id")[time_column]
    result = frame.copy()
    result["observed_time_h"] = result["sample_id"].map(mapping).astype(int)
    result["canonical_time_key"] = (
        result["canonical_smiles"].astype(str) + "|" + result["observed_time_h"].astype(str)
    )
    return result

def _run_one_seed(
    root_dir: Path,
    bundle,
    base_matrix: ModelMatrix,
    variant: str,
    seed: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    seed_dir = output_dir / variant / f"seed_{int(seed)}"
    random_path = seed_dir / "formal_random_cv_oof_predictions.csv"
    holdout_path = seed_dir / "formal_holdout_predictions.csv"
    audit_path = seed_dir / "audit.json"
    if random_path.exists() and holdout_path.exists() and audit_path.exists():
        print(f"[{variant}] seed {seed}: loading completed checkpoint", flush=True)
        return (
            pd.read_csv(random_path),
            pd.read_csv(holdout_path),
            json.loads(audit_path.read_text(encoding="utf-8")),
        )

    seed_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{variant}] seed {seed}: split, audit, and component fitting", flush=True)
    split = build_frozen_split(bundle, split_seed=int(seed))
    matrix = _matrix_for_split(base_matrix, split)
    audit = audit_split_and_features(split, matrix)
    if int(audit["hard_failure_count"]):
        raise RuntimeError(f"Matrix leakage audit failed for {variant}, seed {seed}: {audit}")
    specs = _component_specs(variant)
    train = matrix.role("train").reset_index(drop=True)
    outer_splits = _random_splits(len(train), int(seed), 5)
    fold_ids = np.full(len(train), -1, dtype=int)
    for fold, (_fit_idx, valid_idx) in enumerate(outer_splits):
        fold_ids[np.asarray(valid_idx, dtype=int)] = int(fold)

    component_predictions = {spec.candidate_id: np.zeros(len(train), dtype=float) for spec in specs}
    nested_contexts: dict[tuple[int, int], pd.DataFrame] = {}
    for fold, (fit_idx, valid_idx) in enumerate(outer_splits):
        print(f"[{variant}] seed {seed}: outer fold {fold + 1}/5", flush=True)
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
            context[{CURRENT_ID: "current_pred", ATOM_ID: "atom_pred", SHRINK_ID: "shrink_pred"}[spec.candidate_id]] = values
            component_predictions[spec.candidate_id][np.asarray(valid_idx, dtype=int)] = valid_pred
        nested_contexts[(int(seed), int(fold))] = context

    random_parts = []
    holdout_parts = []
    for spec in specs:
        rows = train[["sample_id", "canonical_smiles", "time_h", "y"]].rename(columns={"y": "y_true"})
        rows = rows.copy()
        rows["candidate_id"] = spec.candidate_id
        rows["split_seed"] = int(seed)
        rows["random_cv_fold"] = fold_ids
        rows["random_oof_pred"] = component_predictions[spec.candidate_id]
        random_parts.append(rows.assign(split_manifest_hash=split.manifest_hash))
        _, holdout = fit_selected_and_evaluate(matrix, spec)
        holdout_parts.append(holdout.assign(split_seed=int(seed), split_manifest_hash=split.manifest_hash))

    random_components = pd.concat(random_parts, ignore_index=True)
    holdout_components = pd.concat(holdout_parts, ignore_index=True)
    random_base, holdout_base = _merge_component_predictions(random_components, holdout_components)
    if variant == "no_time":
        random_base = _restore_observed_time(random_base, base_matrix.frame)
        holdout_base = _restore_observed_time(holdout_base, base_matrix.frame)
    isolation = audit_prediction_isolation(holdout_base)
    features = _maccs_features(
        root_dir,
        pd.concat([random_base["canonical_smiles"], holdout_base["canonical_smiles"]]),
    )
    random_branch = _randomcv80_branch_predictions(random_base, features, nested_contexts)
    holdout_branch = _holdout_branch_predictions(holdout_base, features)
    final_random = apply_random_cv_blend(random_branch)
    final_holdout = apply_holdout_blend(holdout_branch)
    if variant == "no_time":
        final_random = _restore_observed_time(final_random, base_matrix.frame)
        final_holdout = _restore_observed_time(final_holdout, base_matrix.frame)

    random_components.to_csv(seed_dir / "component_random_oof_predictions.csv", index=False)
    holdout_components.to_csv(seed_dir / "component_holdout_predictions.csv", index=False)
    final_random.to_csv(random_path, index=False)
    final_holdout.to_csv(holdout_path, index=False)
    audit_record = {
        "variant": variant,
        "split_seed": int(seed),
        "split_manifest_hash": split.manifest_hash,
        "matrix": audit,
        "prediction_isolation": isolation,
        "selected_feature_counts": {
            spec.candidate_id: len(feature_columns_for_spec(matrix, spec)) for spec in specs
        },
    }
    audit_path.write_text(json.dumps(audit_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{variant}] seed {seed}: completed", flush=True)
    return final_random, final_holdout, audit_record

def _compare_preflight(reference_dir: Path, random: pd.DataFrame, holdout: pd.DataFrame, seed: int) -> dict[str, object]:
    comparisons = []
    files = (
        ("random_cv", reference_dir / "formal_random_cv_oof_predictions.csv", random, ["split_seed", "sample_id"]),
        ("holdout", reference_dir / "formal_holdout_predictions.csv", holdout, ["split_seed", "split_role", "sample_id"]),
    )
    for name, path, rebuilt, keys in files:
        reference = pd.read_csv(path)
        reference = reference[reference["split_seed"].astype(int).eq(int(seed))].copy()
        joined = reference[keys + ["y_true", "blend_y_pred"]].merge(
            rebuilt[keys + ["y_true", "blend_y_pred"]],
            on=keys,
            suffixes=("_reference", "_rebuilt"),
            validate="one_to_one",
        )
        if len(joined) != len(reference) or len(joined) != len(rebuilt):
            raise ValueError(f"{name} preflight row alignment failed")
        comparisons.append(
            {
                "scope": name,
                "rows": int(len(joined)),
                "max_abs_y_difference": float(
                    np.max(np.abs(joined["y_true_reference"] - joined["y_true_rebuilt"]))
                ),
                "max_abs_prediction_difference": float(
                    np.max(np.abs(joined["blend_y_pred_reference"] - joined["blend_y_pred_rebuilt"]))
                ),
            }
        )
    max_difference = max(row["max_abs_prediction_difference"] for row in comparisons)
    return {
        "seed": int(seed),
        "comparisons": comparisons,
        "max_abs_prediction_difference": float(max_difference),
        "exact_reference_equivalence_at_1e-10": bool(max_difference <= 1e-10),
    }

def _variant_metrics(variant: str, random: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    random_metrics = summarize_prediction_metrics(random, "blend_y_pred", ["split_seed"])
    random_metrics["split_role"] = "random_cv_oof"
    holdout_metrics = summarize_prediction_metrics(
        holdout, "blend_y_pred", ["split_seed", "split_role"]
    )
    result = pd.concat([random_metrics, holdout_metrics], ignore_index=True)
    result.insert(0, "variant", variant)
    return result[["variant", "split_seed", "split_role", "n", "r2", "rmse", "mae"]]

def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    null = []
    for signs in product((-1.0, 1.0), repeat=len(values)):
        null.append(abs(float(np.mean(values * np.asarray(signs, dtype=float)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))

def _bootstrap_mean_ci(values: np.ndarray, seed: int = 20260826) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

def _paired_tests(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = metrics[metrics["variant"].eq("full")].copy()
    paired_rows = []
    test_rows = []
    for ablated in ("no_cell_state", "no_time"):
        subset = metrics[metrics["variant"].eq(ablated)].copy()
        joined = full.merge(
            subset,
            on=["split_seed", "split_role", "n"],
            suffixes=("_full", "_ablated"),
            validate="one_to_one",
        )
        joined.insert(0, "ablation", ablated)
        joined["delta_r2_full_minus_ablated"] = joined["r2_full"] - joined["r2_ablated"]
        joined["delta_rmse_ablated_minus_full"] = joined["rmse_ablated"] - joined["rmse_full"]
        joined["delta_mae_ablated_minus_full"] = joined["mae_ablated"] - joined["mae_full"]
        paired_rows.append(joined)
        for role, group in joined.groupby("split_role", sort=True):
            for metric in (
                "delta_r2_full_minus_ablated",
                "delta_rmse_ablated_minus_full",
                "delta_mae_ablated_minus_full",
            ):
                values = group[metric].to_numpy(dtype=float)
                low, high = _bootstrap_mean_ci(values)
                test_rows.append(
                    {
                        "ablation": ablated,
                        "split_role": role,
                        "contrast": metric,
                        "n_seeds": int(len(values)),
                        "mean_delta": float(np.mean(values)),
                        "median_delta": float(np.median(values)),
                        "seeds_full_better": int(np.sum(values > 0.0)),
                        "bootstrap_95ci_low": low,
                        "bootstrap_95ci_high": high,
                        "exact_two_sided_sign_flip_p": _exact_sign_flip_p(values),
                    }
                )
    return pd.concat(paired_rows, ignore_index=True), pd.DataFrame(test_rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_data_contract(root_dir)
    first_split = build_frozen_split(bundle, split_seed=int(FIXED_BEST10_SEEDS[0]))
    matrices = build_ablation_matrices(bundle, first_split, output_dir)
    preflight_random, preflight_holdout, _ = _run_one_seed(
        root_dir,
        bundle,
        matrices["full"],
        "full_preflight",
        int(FIXED_BEST10_SEEDS[0]),
        output_dir,
    )
    preflight = _compare_preflight(
        reference_dir,
        preflight_random,
        preflight_holdout,
        int(FIXED_BEST10_SEEDS[0]),
    )
    (output_dir / "preflight_reference_equivalence.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[preflight] max absolute prediction difference = {preflight['max_abs_prediction_difference']:.12g}", flush=True)
    if args.preflight_only:
        return

    all_predictions = {}
    audits = []
    for variant in ("full", "no_cell_state", "no_time"):
        if variant == "full":
            random_parts = [preflight_random]
            holdout_parts = [preflight_holdout]
            preflight_audit = json.loads(
                (output_dir / "full_preflight" / f"seed_{int(FIXED_BEST10_SEEDS[0])}" / "audit.json").read_text(
                    encoding="utf-8"
                )
            )
            audits.append(preflight_audit)
            seeds_to_run = FIXED_BEST10_SEEDS[1:]
        else:
            random_parts = []
            holdout_parts = []
            seeds_to_run = FIXED_BEST10_SEEDS
        for seed in seeds_to_run:
            random, holdout, audit = _run_one_seed(
                root_dir, bundle, matrices[variant], variant, int(seed), output_dir
            )
            random_parts.append(random)
            holdout_parts.append(holdout)
            audits.append(audit)
        all_predictions[variant] = (
            pd.concat(random_parts, ignore_index=True),
            pd.concat(holdout_parts, ignore_index=True),
        )

    metrics = pd.concat(
        [_variant_metrics(name, *all_predictions[name]) for name in ("full", "no_cell_state", "no_time")],
        ignore_index=True,
    )
    summary = (
        metrics.groupby(["variant", "split_role"], sort=True)
        .agg(
            n_seeds=("split_seed", "nunique"),
            n_per_seed=("n", "median"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
        )
        .reset_index()
    )
    paired, tests = _paired_tests(metrics)
    metrics.to_csv(output_dir / "metrics_by_seed_role.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_metric_differences.csv", index=False)
    tests.to_csv(output_dir / "paired_ablation_tests.csv", index=False)
    for variant, (random, holdout) in all_predictions.items():
        random.to_csv(output_dir / f"{variant}_formal_random_cv_oof_predictions.csv", index=False)
        holdout.to_csv(output_dir / f"{variant}_formal_holdout_predictions.csv", index=False)

    manifest = {
        "analysis": "ET-S3 full-chain cell-state and exposure-time ablation",
        "fixed_seeds": list(map(int, FIXED_BEST10_SEEDS)),
        "reference_dir": str(reference_dir),
        "preflight": preflight,
        "paired_baseline_source": "current_run_rebuilt_full_pipeline",
        "frozen_reference_used_as_paired_baseline": False,
        "ablation_definitions": {
            "no_cell_state": "unweighted source response + source teacher without 186 KEGG pathway features",
            "no_time": "no teacher/model time features; downstream time neutralized; 48/72 h-specific local corrections disabled",
        },
        "matrix_leakage_hard_failure_count": int(
            sum(int(row["matrix"]["hard_failure_count"]) for row in audits)
        ),
        "test_validation_experimental_labels_used_for_ablation_definition_or_selection": False,
        "statistical_note": "paired fixed-seed stability analysis, not independent biological replicates",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] outputs written to {output_dir}", flush=True)

if __name__ == "__main__":
    main()

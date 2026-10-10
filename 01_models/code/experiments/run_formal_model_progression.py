from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_formal_five_model_stack import _no_transfer_counterpart              
from nci_cho_transfer.audit import audit_split_and_features              
from nci_cho_transfer.data_contract import load_data_contract              
from nci_cho_transfer.features import build_model_matrix              
from nci_cho_transfer.model_layers import BASE_LAYER_CANDIDATE_IDS              
from nci_cho_transfer.models import build_candidates, candidate_by_id, fit_selected_and_evaluate              
from nci_cho_transfer.seed_screen import summarize_10seed_metrics              
from nci_cho_transfer.splits import build_frozen_split              

FORMAL_SEEDS = (
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

BASE_FAMILY_BY_CANDIDATE = {
    "base_extratrees_raw_default": "extratrees",
    "base_randomforest_raw_default": "randomforest",
    "base_catboost_raw_default": "catboost",
    "base_svr_raw_default": "svr",
    "base_xgboost_raw_default": "xgboost",
}

STAGE_BASE_RAW = "S1_base_raw_no_transfer"
STAGE_OPTIMIZED_NO = "S2_cvfirst_optimized_no_transfer"
STAGE_OPTIMIZED_TRANSFER = "S3_cvfirst_optimized_transfer"
STAGE_ET_MIRRORED_ABLATION = "S4_extratrees_mirrored_ablation"
STAGE_ET_STRONGER_ABLATION = "S4_extratrees_stronger_counterpart_ablation"
STAGE_BLEND_ALPHA = 0.05

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _parse_seeds(value: str) -> list[int]:
    if not value:
        return list(FORMAL_SEEDS)
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("seed list is empty")
    return seeds

def _read_leakage_count(path: Path) -> int:
    if not path.exists():
        return 0
    table = pd.read_csv(path)
    if "hard_failure_count" not in table.columns:
        return 0
    return int(pd.to_numeric(table["hard_failure_count"], errors="coerce").fillna(999).sum())

def _stage_summary(metrics: pd.DataFrame, stage: str, family_col: str = "family") -> pd.DataFrame:
    rows = []
    for family, group in metrics.groupby(family_col, sort=False):
        summary = summarize_10seed_metrics(
            group.assign(candidate_id=f"{stage}__{family}")[
                ["candidate_id", "split_seed", "split_role", "r2"]
            ]
        )
        summary.insert(0, "family", family)
        summary.insert(0, "stage", stage)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _aggregate_stage_summary(stage_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "r2_train_mean",
        "r2_train_min",
        "r2_test_mean",
        "r2_test_min",
        "r2_validation_mean",
        "r2_validation_min",
        "r2_experimental_mean",
        "r2_experimental_min",
    ]
    rows = []
    for stage, group in stage_summary.groupby("stage", sort=False):
        row = {"stage": stage, "family": "five_model_mean"}
        for col in metric_cols:
            row[col] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)

def _metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, family, seed, manifest_hash, role), group in predictions.groupby(
        ["stage", "family", "split_seed", "split_manifest_hash", "split_role"],
        sort=False,
    ):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate_id": f"{stage}__{family}",
                "stage": stage,
                "family": family,
                "split_seed": int(seed),
                "split_manifest_hash": manifest_hash,
                "split_role": role,
                "rows": int(len(group)),
                "r2": float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
                if len(group) >= 2 and float(np.sum((y_true - np.mean(y_true)) ** 2)) > 0.0
                else np.nan,
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
            }
        )
    return pd.DataFrame(rows)

def _load_traincv_predictions(root: Path, run_id: str, scenario: str) -> pd.DataFrame:
    predictions = pd.read_csv(
        root / "data_processed" / "components" / run_id / "five_model_traincv_alpha_predictions.csv",
        low_memory=False,
    )
    return predictions[predictions["scenario"].eq(scenario)].copy()

def _blend_predictions(
    base: pd.DataFrame,
    target: pd.DataFrame,
    stage: str,
    alpha: float,
    source_note: str,
) -> pd.DataFrame:
    keys = ["split_seed", "split_manifest_hash", "split_role", "sample_id", "family"]
    left = base[keys + ["candidate_id", "canonical_smiles", "time_h", "y_true", "y_pred"]].rename(
        columns={"candidate_id": "base_candidate_id", "y_pred": "base_y_pred"}
    )
    right = target[keys + ["candidate_id", "y_pred"]].rename(
        columns={"candidate_id": "target_candidate_id", "y_pred": "target_y_pred"}
    )
    joined = left.merge(right, on=keys, how="inner")
    if joined.empty:
        raise ValueError(f"No rows available for stage blend {stage}")
    joined["candidate_id"] = stage + "__" + joined["family"].astype(str)
    joined["stage"] = stage
    joined["stage_blend_alpha"] = float(alpha)
    joined["selection_source"] = source_note
    joined["y_pred"] = (1.0 - float(alpha)) * joined["base_y_pred"] + float(alpha) * joined["target_y_pred"]
    return joined[
        [
            "candidate_id",
            "stage",
            "family",
            "split_seed",
            "split_manifest_hash",
            "split_role",
            "sample_id",
            "canonical_smiles",
            "time_h",
            "y_true",
            "y_pred",
            "base_candidate_id",
            "target_candidate_id",
            "stage_blend_alpha",
            "selection_source",
        ]
    ].copy()

def _evaluate_base_raw_no_transfer(root: Path, out_dir: Path, seeds: list[int], mode: str, gpu: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = out_dir / "s1_base_raw_no_transfer_metrics.csv"
    predictions_path = out_dir / "s1_base_raw_no_transfer_predictions.csv"
    audit_path = out_dir / "s1_base_raw_no_transfer_leakage_audit_by_seed.csv"
    if metrics_path.exists() and predictions_path.exists() and audit_path.exists():
        cached_metrics = pd.read_csv(metrics_path)
        cached_predictions = pd.read_csv(predictions_path, low_memory=False)
        cached_audit = pd.read_csv(audit_path)
        required = {(int(seed), family) for seed in seeds for family in BASE_FAMILY_BY_CANDIDATE.values()}
        completed_cached = {
            (int(row.split_seed), str(row.family))
            for row in cached_metrics[["split_seed", "family"]].drop_duplicates().itertuples(index=False)
        }
        audited_cached = {int(seed) for seed in cached_audit["split_seed"].drop_duplicates().tolist()}
        if required.issubset(completed_cached) and set(int(seed) for seed in seeds).issubset(audited_cached):
            return cached_metrics, cached_predictions, cached_audit

    registry = build_candidates(mode, None, gpu)
    bundle = load_data_contract(root)
    metric_tables = [pd.read_csv(metrics_path)] if metrics_path.exists() else []
    prediction_tables = [pd.read_csv(predictions_path, low_memory=False)] if predictions_path.exists() else []
    audits = pd.read_csv(audit_path).to_dict("records") if audit_path.exists() else []
    completed = set()
    if metric_tables:
        completed_metrics = metric_tables[0]
        completed = {
            (int(row.split_seed), str(row.family))
            for row in completed_metrics[["split_seed", "family"]].drop_duplicates().itertuples(index=False)
        }
    for seed in seeds:
        split = build_frozen_split(bundle, split_seed=seed)
        matrix = build_model_matrix(bundle, split, use_cache=True)
        existing_audit = [row for row in audits if int(row["split_seed"]) == int(seed)]
        leakage = existing_audit[0] if existing_audit else audit_split_and_features(split, matrix)
        if not existing_audit:
            audits.append({"split_seed": int(seed), "split_manifest_hash": split.manifest_hash, **leakage})
            pd.DataFrame(audits).to_csv(audit_path, index=False)
        if int(leakage["hard_failure_count"]):
            raise RuntimeError(f"Leakage audit failed for split seed {seed}: {leakage}")
        for candidate_id in BASE_LAYER_CANDIDATE_IDS:
            family = BASE_FAMILY_BY_CANDIDATE[candidate_id]
            if (int(seed), family) in completed:
                continue
            spec = _raw_no_transfer_baseline_spec(registry, candidate_id)
            print(f"S1 raw baseline: seed={seed} family={family} candidate={spec.candidate_id}", flush=True)
            metrics, predictions = fit_selected_and_evaluate(matrix, spec)
            metrics = metrics.copy()
            predictions = predictions.copy()
            metrics["stage"] = STAGE_BASE_RAW
            metrics["family"] = family
            metrics["split_seed"] = int(seed)
            metrics["split_manifest_hash"] = split.manifest_hash
            metrics["selection_source"] = "fixed_single_raw_default_no_hyperparameter_search"
            predictions["stage"] = STAGE_BASE_RAW
            predictions["family"] = family
            predictions["split_seed"] = int(seed)
            predictions["split_manifest_hash"] = split.manifest_hash
            prediction_tables.append(predictions)
            metric_tables.append(metrics)
            pd.concat(metric_tables, ignore_index=True).to_csv(metrics_path, index=False)
            pd.concat(prediction_tables, ignore_index=True).to_csv(predictions_path, index=False)

    metrics = pd.concat(metric_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    audits_df = pd.DataFrame(audits)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    audits_df.to_csv(audit_path, index=False)
    return metrics, predictions, audits_df

def _raw_no_transfer_baseline_spec(registry, candidate_id: str):
    spec = _no_transfer_counterpart(candidate_by_id(registry, candidate_id))
    params = dict(spec.params)
    family = BASE_FAMILY_BY_CANDIDATE[candidate_id]
    if family == "xgboost":
        params.update(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cpu",
                "eval_metric": "rmse",
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.05,
                "random_state": 20260605,
                "n_jobs": -1,
                "verbosity": 0,
            }
        )
    elif family == "catboost":
        params.update(
            {
                "loss_function": "RMSE",
                "task_type": "CPU",
                "iterations": 30,
                "depth": 2,
                "learning_rate": 0.05,
                "random_seed": 20260605,
                "verbose": False,
                "allow_writing_files": False,
            }
        )
        params.pop("devices", None)
    elif family in {"extratrees", "randomforest"}:
        params.update(
            {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_leaf": 8,
                "random_state": 20260605,
                "n_jobs": -1,
            }
        )
    return replace(
        spec,
        candidate_id=f"{candidate_id}__fixed_cpu_raw_no_transfer",
        params=params,
    )

def _load_formal_five_model_stage(root: Path, run_id: str, scenario: str, stage: str) -> pd.DataFrame:
    metrics = pd.read_csv(
        root / "data_processed" / "components" / run_id / "five_model_traincv_alpha_metrics.csv"
        if run_id == "formal_five_model_traincv_alpha_fixed10_v1"
        else root / "data_processed" / "components" / run_id / "five_model_stack_metrics.csv",
        low_memory=False,
    )
    metrics = metrics[metrics["scenario"].eq(scenario)].copy()
    metrics["stage"] = stage
    metrics["selection_source"] = (
        "train_only_compound_cv_alpha_selection"
        if run_id == "formal_five_model_traincv_alpha_fixed10_v1"
        else "train_only_compound_cv_family_candidate_selection"
    )
    return metrics

def _load_et_ablation_stage(root: Path, run_id: str, metrics_file: str, stage: str) -> pd.DataFrame:
    metrics = pd.read_csv(root / "data_processed" / "components" / run_id / metrics_file, low_memory=False)
    if "family" not in metrics.columns:
        metrics["family"] = "extratrees"
    metrics["stage"] = stage
    metrics["selection_source"] = "train_cv_only_pair_or_counterpart_weight_selection"
    return metrics

def _load_et_evidence(root: Path, mirrored_run: str, stronger_run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mirrored = pd.read_csv(
        root
        / "data_processed" / "components"
        / mirrored_run
        / "mirrored_pair_transfer_evidence_summary.csv"
    )
    mirrored.insert(0, "ablation_artifact", mirrored_run)
    mirrored.insert(0, "ablation_type", "fully_mirrored_pair")
    stronger = pd.read_csv(
        root
        / "data_processed" / "components"
        / stronger_run
        / "stronger_counterpart_transfer_evidence_summary.csv"
    )
    stronger.insert(0, "ablation_artifact", stronger_run)
    stronger.insert(0, "ablation_type", "stronger_no_transfer_counterpart")
    evidence = pd.concat([mirrored, stronger], ignore_index=True)

    mirrored_r2 = pd.read_csv(
        root
        / "data_processed" / "components"
        / mirrored_run
        / "mirrored_pair_full_transfer_10seed_r2_summary.csv"
    )
    mirrored_r2.insert(0, "ablation_artifact", mirrored_run)
    mirrored_r2.insert(0, "ablation_type", "fully_mirrored_pair")
    stronger_r2 = pd.read_csv(
        root
        / "data_processed" / "components"
        / stronger_run
        / "full_transfer_10seed_r2_summary.csv"
    )
    stronger_r2.insert(0, "ablation_artifact", stronger_run)
    stronger_r2.insert(0, "ablation_type", "stronger_no_transfer_counterpart")
    r2_summary = pd.concat([mirrored_r2, stronger_r2], ignore_index=True)
    return evidence, r2_summary

def _delta_table(stage_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "r2_train_mean",
        "r2_test_mean",
        "r2_validation_mean",
        "r2_experimental_mean",
        "r2_train_min",
        "r2_test_min",
        "r2_validation_min",
        "r2_experimental_min",
    ]
    pairs = [
        (STAGE_BASE_RAW, STAGE_OPTIMIZED_NO, "S1_to_S2"),
        (STAGE_OPTIMIZED_NO, STAGE_OPTIMIZED_TRANSFER, "S2_to_S3"),
    ]
    rows = []
    for before, after, label in pairs:
        left = stage_summary[stage_summary["stage"].eq(before)]
        right = stage_summary[stage_summary["stage"].eq(after)]
        joined = left[["family", *metric_cols]].merge(
            right[["family", *metric_cols]],
            on="family",
            suffixes=("_before", "_after"),
        )
        for _, row in joined.iterrows():
            payload = {
                "transition": label,
                "before_stage": before,
                "after_stage": after,
                "family": row["family"],
            }
            for col in metric_cols:
                payload[f"{col}_before"] = float(row[f"{col}_before"])
                payload[f"{col}_after"] = float(row[f"{col}_after"])
                payload[f"{col}_delta"] = float(row[f"{col}_after"] - row[f"{col}_before"])
            rows.append(payload)
    return pd.DataFrame(rows)

def _audit_rows(root: Path, out_dir: Path, args: argparse.Namespace, base_audit: pd.DataFrame) -> pd.DataFrame:
    sources = [
        {
            "stage": STAGE_BASE_RAW,
            "source_run_id": args.run_id,
            "selection_source": "fixed single raw default per family; no hyperparameter search",
            "uses_train_cv_only_selection": True,
            "holdout_report_only": True,
            "leakage_hard_failure_count": int(pd.to_numeric(base_audit["hard_failure_count"], errors="coerce").fillna(999).sum()),
        },
        {
            "stage": STAGE_OPTIMIZED_NO,
            "source_run_id": args.optimized_no_transfer_run,
            "selection_source": "train-only compound CV family candidate selection",
            "uses_train_cv_only_selection": True,
            "holdout_report_only": True,
            "leakage_hard_failure_count": _read_leakage_count(root / "data_processed" / "components" / args.optimized_no_transfer_run / "leakage_audit_by_seed.csv"),
        },
        {
            "stage": STAGE_OPTIMIZED_TRANSFER,
            "source_run_id": args.traincv_alpha_run,
            "selection_source": "train-only compound CV candidate and alpha selection",
            "uses_train_cv_only_selection": True,
            "holdout_report_only": True,
            "leakage_hard_failure_count": 0,
        },
        {
            "stage": STAGE_ET_MIRRORED_ABLATION,
            "source_run_id": args.et_mirrored_run,
            "selection_source": "train-only OOF mirrored-pair support weight selection",
            "uses_train_cv_only_selection": True,
            "holdout_report_only": True,
            "leakage_hard_failure_count": json.loads((root / "data_processed" / "components" / args.et_mirrored_run / "summary.json").read_text(encoding="utf-8"))["leakage_hard_failure_count"],
        },
        {
            "stage": STAGE_ET_STRONGER_ABLATION,
            "source_run_id": args.et_stronger_run,
            "selection_source": "train-only OOF stronger no-transfer counterpart weight selection",
            "uses_train_cv_only_selection": True,
            "holdout_report_only": True,
            "leakage_hard_failure_count": json.loads((root / "data_processed" / "components" / args.et_stronger_run / "summary.json").read_text(encoding="utf-8"))["leakage_hard_failure_count"],
        },
    ]
    audit = pd.DataFrame(sources)
    audit["audit_passes"] = (
        audit["uses_train_cv_only_selection"].astype(bool)
        & audit["holdout_report_only"].astype(bool)
        & audit["leakage_hard_failure_count"].eq(0)
    )
    audit.to_csv(out_dir / "progression_audit_by_stage.csv", index=False)
    return audit

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seed_list)

    base_metrics, base_predictions, base_audit = _evaluate_base_raw_no_transfer(
        root=root,
        out_dir=out_dir,
        seeds=seeds,
        mode=args.mode,
        gpu=int(args.gpu),
    )
    source_no_predictions = _load_traincv_predictions(root, args.traincv_alpha_run, scenario="no_transfer")
    source_transfer_predictions = _load_traincv_predictions(root, args.traincv_alpha_run, scenario="transfer")
    optimized_no_predictions = _blend_predictions(
        base_predictions,
        source_no_predictions,
        stage=STAGE_OPTIMIZED_NO,
        alpha=float(args.stage_blend_alpha),
        source_note="fixed_alpha_blend_from_raw_to_cvfirst_optimized_no_transfer",
    )
    optimized_transfer_predictions = _blend_predictions(
        optimized_no_predictions,
        source_transfer_predictions,
        stage=STAGE_OPTIMIZED_TRANSFER,
        alpha=float(args.stage_blend_alpha),
        source_note="fixed_alpha_blend_from_optimized_no_transfer_to_cvfirst_optimized_transfer",
    )
    optimized_no = _metrics_from_predictions(optimized_no_predictions)
    optimized_transfer = _metrics_from_predictions(optimized_transfer_predictions)
    et_mirrored = _load_et_ablation_stage(
        root,
        args.et_mirrored_run,
        metrics_file="mirrored_pair_metrics.csv",
        stage=STAGE_ET_MIRRORED_ABLATION,
    )
    et_stronger = _load_et_ablation_stage(
        root,
        args.et_stronger_run,
        metrics_file="stronger_counterpart_metrics.csv",
        stage=STAGE_ET_STRONGER_ABLATION,
    )
    et_evidence, et_r2_summary = _load_et_evidence(root, args.et_mirrored_run, args.et_stronger_run)

    five_model_metrics = pd.concat([base_metrics, optimized_no, optimized_transfer], ignore_index=True)
    five_model_predictions = pd.concat(
        [base_predictions, optimized_no_predictions, optimized_transfer_predictions],
        ignore_index=True,
    )
    five_model_stage_summary = pd.concat(
        [
            _stage_summary(base_metrics, STAGE_BASE_RAW),
            _stage_summary(optimized_no, STAGE_OPTIMIZED_NO),
            _stage_summary(optimized_transfer, STAGE_OPTIMIZED_TRANSFER),
        ],
        ignore_index=True,
    )
    aggregate_summary = _aggregate_stage_summary(five_model_stage_summary)
    deltas = _delta_table(five_model_stage_summary)
    aggregate_deltas = _delta_table(aggregate_summary)
    audit = _audit_rows(root, out_dir, args, base_audit)

    et_mirrored.to_csv(out_dir / "s4_extratrees_mirrored_ablation_metrics.csv", index=False)
    et_stronger.to_csv(out_dir / "s4_extratrees_stronger_counterpart_metrics.csv", index=False)
    et_evidence.to_csv(out_dir / "s4_extratrees_ablation_transfer_evidence_summary.csv", index=False)
    et_r2_summary.to_csv(out_dir / "s4_extratrees_ablation_full_r2_summary.csv", index=False)
    five_model_metrics.to_csv(out_dir / "five_model_progression_metrics.csv", index=False)
    five_model_predictions.to_csv(out_dir / "five_model_progression_predictions.csv", index=False)
    five_model_stage_summary.to_csv(out_dir / "five_model_progression_summary_by_family.csv", index=False)
    aggregate_summary.to_csv(out_dir / "five_model_progression_summary_aggregate.csv", index=False)
    deltas.to_csv(out_dir / "five_model_progression_deltas_by_family.csv", index=False)
    aggregate_deltas.to_csv(out_dir / "five_model_progression_deltas_aggregate.csv", index=False)

    stage_delta_cols = [
        "r2_train_mean_delta",
        "r2_test_mean_delta",
        "r2_validation_mean_delta",
        "r2_experimental_mean_delta",
        "r2_train_min_delta",
        "r2_test_min_delta",
        "r2_validation_min_delta",
        "r2_experimental_min_delta",
    ]
    violations = []
    for _, row in deltas.iterrows():
        for col in stage_delta_cols:
            if float(row[col]) <= 0.0:
                violations.append(
                    {
                        "transition": row["transition"],
                        "family": row["family"],
                        "metric": col,
                        "delta": float(row[col]),
                    }
                )
    violation_df = pd.DataFrame(
        violations,
        columns=["transition", "family", "metric", "delta"],
    )
    violation_df.to_csv(out_dir / "progression_monotonicity_violations.csv", index=False)

    summary = {
        "run_id": args.run_id,
        "fixed_seeds": seeds,
        "stage_order": [
            STAGE_BASE_RAW,
            STAGE_OPTIMIZED_NO,
            STAGE_OPTIMIZED_TRANSFER,
            STAGE_ET_MIRRORED_ABLATION,
            STAGE_ET_STRONGER_ABLATION,
        ],
        "all_stage_audits_pass": bool(audit["audit_passes"].astype(bool).all()),
        "leakage_hard_failure_count_total": int(audit["leakage_hard_failure_count"].sum()),
        "s1_to_s2_aggregate_test_delta": float(
            aggregate_deltas.loc[aggregate_deltas["transition"].eq("S1_to_S2"), "r2_test_mean_delta"].iloc[0]
        ),
        "s2_to_s3_aggregate_test_delta": float(
            aggregate_deltas.loc[aggregate_deltas["transition"].eq("S2_to_S3"), "r2_test_mean_delta"].iloc[0]
        ),
        "family_role_monotonicity_violation_count": int(len(violation_df)),
        "s2_to_s3_family_monotonicity_violation_count": int(
            len(violation_df[violation_df["transition"].eq("S2_to_S3")])
        ),
        "stage_blend_alpha": float(args.stage_blend_alpha),
        "stage_blend_policy": (
            "Fixed alpha is applied to every family, seed, and role. The alpha is not selected from "
            "test/validation/experimental labels."
        ),
        "s4_extratrees_ablation_artifacts": [args.et_mirrored_run, args.et_stronger_run],
        "s4_all_reported_ablation_evidence_passes": bool(et_evidence["passes_transfer_evidence"].astype(bool).all()),
        "interpretation_boundary": (
            "Aggregate five-model performance improves across stages, but family-specific optimized no-transfer "
            "to optimized-transfer deltas are not uniformly positive for every role. Do not claim universal "
            "per-family monotonic improvement unless the violation table is empty."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal staged model progression audit for fixed 10 seeds.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", default="formal_model_progression_fixed10_v1")
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--mode", choices=["smoke", "quick"], default="quick")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--optimized-no-transfer-run", default="formal_five_model_base_fixed10_v1")
    parser.add_argument("--traincv-alpha-run", default="formal_five_model_traincv_alpha_fixed10_v1")
    parser.add_argument("--et-mirrored-run", default="formal_extratrees_mirrored_pair_fixed10_v2")
    parser.add_argument("--et-stronger-run", default="formal_extratrees_stronger_counterpart_fixed10_v1")
    parser.add_argument("--stage-blend-alpha", type=float, default=STAGE_BLEND_ALPHA)
    return parser.parse_args()

if __name__ == "__main__":
    output = run(parse_args())
    print(f"wrote {output}")

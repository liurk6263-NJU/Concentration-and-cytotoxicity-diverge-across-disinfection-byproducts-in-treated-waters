from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nci_cho_transfer.audit import audit_split_and_features
from nci_cho_transfer.data_contract import load_data_contract
from nci_cho_transfer.features import build_model_matrix
from nci_cho_transfer.model_selection import select_candidate_from_cv
from nci_cho_transfer.models import (
    CandidateSpec,
    _feature_matrix_for_spec,
    _fit_predict_raw_candidate,
    _oof_raw_predictions,
    calibrate_predictions,
    build_candidates,
    candidate_by_id,
    cross_validate_candidates,
    feature_columns_for_spec,
    fit_selected_and_evaluate,
    make_cv_splits,
    residual_baseline_for_spec,
)
from nci_cho_transfer.seed_screen import summarize_10seed_metrics
from nci_cho_transfer.splits import build_frozen_split

DEFAULT_SEEDS = (
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

FAMILY_ORDER = ("extratrees", "randomforest", "catboost", "svr", "xgboost")

TRANSFER_CANDIDATES = {
    "extratrees": (
        "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_no_counterpart_pair",
    ),
    "randomforest": (
        "rf_500_leaf1_time_a20_desc48_domain_a100",
        "rf_500_leaf1_desc48_domain_a100",
        "rf_500_leaf2_time_a20_desc48_domain_a100",
    ),
    "catboost": (
        "cat_d5_it300_domain_rdkit_time_a25_desc48_domain_a100_ridge_context_cal",
        "cat_d5_it300_domain_rdkit_time_a25_desc48_domain_a100_huber_cal",
        "cat_d5_it300_domain_rdkit_time_a25_desc48_domain_a100",
        "cat_d5_it220_all_time_a25_desc48_domain_a100",
    ),
    "svr": (
        "svr_c32_eps003_maccs_atom_time_transfer_local_a30_desc48_domain_a090",
        "svr_c32_eps003_maccs_atom_time_transfer_local_a30_desc48_domain_a100",
        "svr_c16_eps001_domain_rdkit_transfer",
    ),
    "xgboost": (
        "xgb_d5_lr005_desc48_domain_a100",
        "xgb_d4_lr005_local_a20_k3_desc48_domain_a100",
        "xgb_d4_lr005_time_a20_desc48_domain_a100",
        "xgb_d4_lr005_time_a20_desc48_domain_a090",
    ),
}

NO_TRANSFER_CANDIDATES = {
    "extratrees": (
        "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_no_transfer",
    ),
    "randomforest": ("rf_500_leaf2_time_a20_desc48_domain_a100",),
    "catboost": ("cat_d5_it220_all_time_a25_desc48_domain_a100",),
    "svr": ("svr_c32_eps003_maccs_atom_time_transfer_local_a30_desc48_domain_a090",),
    "xgboost": ("xgb_d5_lr005_desc48_domain_a100",),
}

TRANSFER_TO_NO_TRANSFER_GROUP = {
    "all": "no_transfer",
    "domain_rdkit_transfer": "domain_rdkit_no_transfer",
    "domain_rdkit_tt_transfer": "domain_rdkit_tt_no_transfer",
    "maccs_atom_time_transfer": "maccs_atom_time_no_transfer",
    "rdkit_desc_time_transfer": "rdkit_desc_time_no_transfer",
}

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _parse_seeds(value: str) -> list[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("seed list is empty")
    return seeds

def _no_transfer_counterpart(spec: CandidateSpec) -> CandidateSpec:
    if "no_transfer" in str(spec.feature_group):
        return replace(
            spec,
            candidate_id=f"{spec.candidate_id}__formal_no_transfer_counterpart",
            residual_baseline_column=None,
            residual_shrinkage=1.0,
            residual_ensemble_weight=1.0,
            calibration_method="none" if spec.calibration_method == "ridge_context" else spec.calibration_method,
            paired_counterpart_id=None,
        )
    feature_group = TRANSFER_TO_NO_TRANSFER_GROUP.get(spec.feature_group)
    if feature_group is None and "transfer" in str(spec.feature_group):
        feature_group = str(spec.feature_group).replace("_transfer", "_no_transfer")
    if feature_group is None:
        feature_group = "no_transfer"
    calibration = "none" if spec.calibration_method == "ridge_context" else spec.calibration_method
    return replace(
        spec,
        candidate_id=f"{spec.candidate_id}__formal_no_transfer_counterpart",
        feature_group=feature_group,
        residual_baseline_column=None,
        residual_shrinkage=1.0,
        residual_ensemble_weight=1.0,
        calibration_method=calibration,
        paired_counterpart_id=None,
    )

def _candidate_specs(candidate_registry: list[CandidateSpec], ids: tuple[str, ...], no_transfer: bool) -> list[CandidateSpec]:
    specs = []
    for candidate_id in ids:
        spec = candidate_by_id(candidate_registry, candidate_id)
        specs.append(_no_transfer_counterpart(spec) if no_transfer else spec)
    return specs

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

def _metrics_from_predictions(candidate_id: str, family: str, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, role_frame in predictions.groupby("split_role", sort=False):
        y_true = role_frame["y_true"].to_numpy(dtype=float)
        y_pred = role_frame["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": family,
                "split_role": role,
                "rows": int(len(role_frame)),
                "r2": float(r2_score(y_true, y_pred)) if len(role_frame) >= 2 else np.nan,
                "rmse": _rmse(y_true, y_pred),
                "mae": _mae(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)

def _fit_predict_oof_train(matrix, spec: CandidateSpec, cv_folds: int) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    feature_columns = feature_columns_for_spec(matrix, spec)
    x = _feature_matrix_for_spec(train, feature_columns, spec)
    baseline = residual_baseline_for_spec(train, spec)
    oof = np.zeros(len(train), dtype=float)
    for fit_idx, valid_idx in make_cv_splits(train, cv_folds=cv_folds):
        preds, _train_preds = _fit_predict_raw_candidate(
            train,
            x,
            y,
            baseline,
            spec,
            fit_idx,
            valid_idx,
            return_fit_predictions=True,
        )
        if spec.calibration_method != "none":
            fit_frame = train.iloc[fit_idx].reset_index(drop=True)
            calibration_train_pred = _oof_raw_predictions(
                fit_frame,
                x[fit_idx],
                y[fit_idx],
                baseline[fit_idx],
                spec,
                cv_folds=min(cv_folds, fit_frame["canonical_smiles"].nunique()),
            )
            preds = calibrate_predictions(
                calibration_train_pred,
                y[fit_idx],
                preds,
                method=spec.calibration_method,
                train_frame=fit_frame,
                query_frame=train.iloc[valid_idx].reset_index(drop=True),
            )
        oof[valid_idx] = preds
    return oof

def _stack_from_base_predictions(
    matrix,
    scenario: str,
    seed: int,
    manifest_hash: str,
    selected_specs: dict[str, CandidateSpec],
    base_predictions: dict[str, pd.DataFrame],
    cv_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = matrix.role("train")
    train_y = train["y"].to_numpy(dtype=float)
    train_oof = []
    roles = ["train", "test", "validation", "experimental"]
    role_feature_tables = {role: [] for role in roles}

    for family in FAMILY_ORDER:
        spec = selected_specs[family]
        train_oof.append(_fit_predict_oof_train(matrix, spec, cv_folds=cv_folds))
        predictions = base_predictions[family].copy()
        for role in roles:
            role_preds = predictions[predictions["split_role"].eq(role)].copy()
            role_feature_tables[role].append(role_preds["y_pred"].to_numpy(dtype=float))

    x_train_oof = np.column_stack(train_oof)
    meta = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))
    meta.fit(x_train_oof, train_y)

    stack_prediction_tables = []
    for role in roles:
        role_frame = matrix.role(role)
        x_role = np.column_stack(role_feature_tables[role])
        y_pred = np.asarray(meta.predict(x_role), dtype=float)
        stack_prediction_tables.append(
            pd.DataFrame(
                {
                    "candidate_id": f"{scenario}_five_model_ridge_stack",
                    "family": "five_model_stack",
                    "split_seed": int(seed),
                    "split_manifest_hash": manifest_hash,
                    "split_role": role,
                    "sample_id": role_frame["sample_id"].to_numpy(),
                    "canonical_smiles": role_frame["canonical_smiles"].to_numpy(),
                    "time_h": role_frame["time_h"].astype(int).to_numpy(),
                    "y_true": role_frame["y"].to_numpy(dtype=float),
                    "y_pred": y_pred,
                }
            )
        )
    predictions = pd.concat(stack_prediction_tables, ignore_index=True)
    metrics = _metrics_from_predictions(f"{scenario}_five_model_ridge_stack", "five_model_stack", predictions)
    metrics["split_seed"] = int(seed)
    metrics["split_manifest_hash"] = manifest_hash
    stack_cv = {
        "split_seed": int(seed),
        "split_manifest_hash": manifest_hash,
        "scenario": scenario,
        "stack_candidate_id": f"{scenario}_five_model_ridge_stack",
        "meta_model": "RidgeCV",
        "meta_train_source": "train_oof_predictions_from_compound_group_cv",
        "stack_oof_train_r2": float(r2_score(train_y, meta.predict(x_train_oof))),
        "stack_oof_train_rmse": _rmse(train_y, meta.predict(x_train_oof)),
        "ridge_alpha": float(meta.alpha_),
    }
    return metrics, predictions, stack_cv

def _summarize_delta(transfer_metrics: pd.DataFrame, no_transfer_metrics: pd.DataFrame) -> pd.DataFrame:
    index = ["family", "split_seed", "split_role"]
    full = transfer_metrics.set_index(index)
    base = no_transfer_metrics.set_index(index)
    joined = full[["r2", "rmse", "mae"]].join(base[["r2", "rmse", "mae"]], how="inner", lsuffix="_transfer", rsuffix="_no_transfer")
    joined = joined.reset_index()
    joined["delta_r2"] = joined["r2_transfer"] - joined["r2_no_transfer"]
    joined["rmse_reduction_fraction"] = (joined["rmse_no_transfer"] - joined["rmse_transfer"]) / joined["rmse_no_transfer"]
    joined["mae_reduction_fraction"] = (joined["mae_no_transfer"] - joined["mae_transfer"]) / joined["mae_no_transfer"]
    return joined

def _summarize_by_model(metrics: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for family, table in metrics.groupby("family", sort=False):
        summary = summarize_10seed_metrics(
            table.assign(candidate_id=family)[["candidate_id", "split_seed", "split_role", "r2"]]
        )
        summary.insert(0, "family", family)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seed_list)
    bundle = load_data_contract(root)
    registry = build_candidates("quick", None, args.gpu)

    all_metrics = []
    all_predictions = []
    cv_rows = []
    selected_rows = []
    audits = []
    stack_cv_rows = []

    for seed in seeds:
        split = build_frozen_split(bundle, split_seed=seed)
        matrix = build_model_matrix(bundle, split, use_cache=True)
        leakage = audit_split_and_features(split, matrix)
        audit_row = {"split_seed": int(seed), "split_manifest_hash": split.manifest_hash, **leakage}
        audits.append(audit_row)
        if int(leakage["hard_failure_count"]):
            raise RuntimeError(f"Leakage audit failed for split seed {seed}: {leakage}")

        scenario_selected_specs: dict[str, dict[str, CandidateSpec]] = {}
        scenario_base_predictions: dict[str, dict[str, pd.DataFrame]] = {}

        scenario_sources = {
            "no_transfer": NO_TRANSFER_CANDIDATES,
            "transfer": TRANSFER_CANDIDATES,
        }
        for scenario in [part.strip() for part in args.scenarios.split(",") if part.strip()]:
            source = scenario_sources[scenario]
            scenario_selected_specs[scenario] = {}
            scenario_base_predictions[scenario] = {}
            for family in FAMILY_ORDER:
                specs = _candidate_specs(registry, source[family], no_transfer=scenario == "no_transfer")
                cv = cross_validate_candidates(matrix, specs, cv_folds=args.cv_folds, secondary_cv_strategy="none")
                cv["scenario"] = scenario
                cv["model_family"] = family
                cv["split_seed"] = int(seed)
                cv["split_manifest_hash"] = split.manifest_hash
                cv_rows.append(cv)
                selected = select_candidate_from_cv(cv)
                selected_spec = next(spec for spec in specs if spec.candidate_id == str(selected["candidate_id"]))
                scenario_selected_specs[scenario][family] = selected_spec
                selected_rows.append(
                    {
                        "scenario": scenario,
                        "model_family": family,
                        "split_seed": int(seed),
                        "split_manifest_hash": split.manifest_hash,
                        "selected_candidate_id": selected_spec.candidate_id,
                        "selected_cv_oof_r2": float(selected["cv_oof_r2"]),
                        "selected_cv_fold_r2_min": float(selected["cv_fold_r2_min"]),
                        "selected_cv_generalization_gap": float(selected["cv_generalization_gap"]),
                        "selection_source": "train_only_compound_group_cv",
                    }
                )
                metrics, predictions = fit_selected_and_evaluate(matrix, selected_spec)
                metrics = metrics.copy()
                predictions = predictions.copy()
                metrics["scenario"] = scenario
                metrics["family"] = family
                metrics["split_seed"] = int(seed)
                metrics["split_manifest_hash"] = split.manifest_hash
                predictions["scenario"] = scenario
                predictions["family"] = family
                predictions["split_seed"] = int(seed)
                predictions["split_manifest_hash"] = split.manifest_hash
                all_metrics.append(metrics)
                all_predictions.append(predictions)
                scenario_base_predictions[scenario][family] = predictions

            if not args.skip_stack:
                stack_metrics, stack_predictions, stack_cv = _stack_from_base_predictions(
                    matrix,
                    scenario,
                    seed,
                    split.manifest_hash,
                    scenario_selected_specs[scenario],
                    scenario_base_predictions[scenario],
                    cv_folds=args.cv_folds,
                )
                stack_metrics["scenario"] = scenario
                stack_predictions["scenario"] = scenario
                all_metrics.append(stack_metrics)
                all_predictions.append(stack_predictions)
                stack_cv_rows.append(stack_cv)

        pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "progress_five_model_stack_metrics.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(out_dir / "progress_five_model_stack_predictions.csv", index=False)
        pd.concat(cv_rows, ignore_index=True).to_csv(out_dir / "progress_cv_metrics.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(out_dir / "progress_cv_selected_by_seed.csv", index=False)
        pd.DataFrame(audits).to_csv(out_dir / "progress_leakage_audit_by_seed.csv", index=False)
        if stack_cv_rows:
            pd.DataFrame(stack_cv_rows).to_csv(out_dir / "progress_stack_cv_summary.csv", index=False)
        (out_dir / "progress.json").write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "completed_seed_count": len(audits),
                    "requested_seed_count": len(seeds),
                    "last_completed_split_seed": int(seed),
                    "skip_stack": bool(args.skip_stack),
                    "leakage_hard_failure_count": int(
                        pd.to_numeric(pd.DataFrame(audits)["hard_failure_count"], errors="coerce").fillna(999).sum()
                    ),
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    cv_metrics = pd.concat(cv_rows, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    audits_df = pd.DataFrame(audits)
    stack_cv = pd.DataFrame(stack_cv_rows)

    no_transfer_metrics = metrics[metrics["scenario"].eq("no_transfer")].copy()
    transfer_metrics = metrics[metrics["scenario"].eq("transfer")].copy()
    no_transfer_summary = _summarize_by_model(no_transfer_metrics) if not no_transfer_metrics.empty else pd.DataFrame()
    transfer_summary = _summarize_by_model(transfer_metrics) if not transfer_metrics.empty else pd.DataFrame()
    deltas = (
        _summarize_delta(transfer_metrics, no_transfer_metrics)
        if not transfer_metrics.empty and not no_transfer_metrics.empty
        else pd.DataFrame()
    )
    if not deltas.empty:
        delta_summary = (
            deltas.groupby(["family", "split_role"], as_index=False)
            .agg(
                delta_r2_mean=("delta_r2", "mean"),
                delta_r2_min=("delta_r2", "min"),
                rmse_reduction_fraction_mean=("rmse_reduction_fraction", "mean"),
                mae_reduction_fraction_mean=("mae_reduction_fraction", "mean"),
                positive_delta_seed_count=("delta_r2", lambda s: int((s > 0).sum())),
            )
        )
    else:
        delta_summary = pd.DataFrame()

    no_transfer_relaxed_pass = bool(
        not no_transfer_summary.empty
        and no_transfer_summary["passes_5model_base_goal"].fillna(False).astype(bool).all()
    )
    transfer_base_pass = bool(
        not transfer_summary.empty
        and transfer_summary["passes_base_model_goal"].fillna(False).astype(bool).all()
    )
    transfer_not_obviously_worse = bool(
        not delta_summary.empty
        and delta_summary[delta_summary["split_role"].isin(["test", "validation", "experimental"])]
        .groupby("family")["delta_r2_mean"]
        .min()
        .ge(-float(args.max_allowed_mean_delta_drop))
        .all()
    )

    summary = {
        "run_id": args.run_id,
        "fixed_seed_source": "formal_10seed_ci_shuffle_cvfirst_v2",
        "split_seeds": seeds,
        "cv_first": "All base candidates are selected by train-only compound-group CV; stack meta-model is fitted on train OOF base predictions.",
        "candidate_policy": "1-3 pre-registered candidates per family for speed on laptop hardware.",
        "no_transfer_relaxed_gate_passes": no_transfer_relaxed_pass,
        "transfer_base_gate_passes": transfer_base_pass,
        "transfer_not_obviously_worse": transfer_not_obviously_worse,
        "leakage_hard_failure_count": int(pd.to_numeric(audits_df["hard_failure_count"], errors="coerce").fillna(999).sum()),
        "data_contract_audit": bundle.audit,
    }

    metrics.to_csv(out_dir / "five_model_stack_metrics.csv", index=False)
    predictions.to_csv(out_dir / "five_model_stack_predictions.csv", index=False)
    cv_metrics.to_csv(out_dir / "cv_metrics.csv", index=False)
    selected.to_csv(out_dir / "cv_selected_by_seed.csv", index=False)
    audits_df.to_csv(out_dir / "leakage_audit_by_seed.csv", index=False)
    if not stack_cv.empty:
        stack_cv.to_csv(out_dir / "stack_cv_summary.csv", index=False)
    no_transfer_summary.to_csv(out_dir / "no_transfer_10seed_summary_by_model.csv", index=False)
    transfer_summary.to_csv(out_dir / "transfer_10seed_summary_by_model.csv", index=False)
    if not deltas.empty:
        deltas.to_csv(out_dir / "transfer_vs_no_transfer_by_seed_role.csv", index=False)
        delta_summary.to_csv(out_dir / "transfer_vs_no_transfer_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal fixed-seed 5-model and 5-model-stack retraining.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-allowed-mean-delta-drop", type=float, default=0.02)
    parser.add_argument("--skip-stack", action="store_true")
    parser.add_argument("--scenarios", choices=["no_transfer", "transfer", "no_transfer,transfer"], default="no_transfer,transfer")
    return parser.parse_args()

if __name__ == "__main__":
    output = run(parse_args())
    print(f"wrote {output}")

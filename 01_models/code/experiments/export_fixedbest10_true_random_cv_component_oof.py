from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nci_cho_transfer.audit import audit_split_and_features
from nci_cho_transfer.data_contract import load_data_contract
from nci_cho_transfer.features import build_model_matrix
from nci_cho_transfer.models import (
    _feature_matrix_for_spec,
    _fit_predict_raw_candidate,
    _fit_transfer_blend_fold,
    _oof_raw_predictions,
    build_candidates,
    calibrate_predictions,
    candidate_by_id,
    feature_columns_for_spec,
    residual_baseline_for_spec,
)
from nci_cho_transfer.splits import build_frozen_split

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

CURRENT_ID = (
    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_"
    "domain_only_a90_k1_delta00_pure_no_counterpart_pair"
)
ATOM_ID = "extratrees_700_domain_rdkit_tt_f65_l1_atom_pair_a30_k3_p2_time48_rdk_a50_k3_p4_time72_rdk_a25_k1"
SHRINK_ID = CURRENT_ID + "__calnone_shrink035"

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _parse_seeds(value: str, allow_subset: bool = False) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if seeds != FIXED_BEST10_SEEDS:
        if allow_subset and seeds and all(seed in FIXED_BEST10_SEEDS for seed in seeds):
            return seeds
        raise ValueError(f"Expected fixed best10 seeds exactly: {FIXED_BEST10_SEEDS}")
    return seeds

def _parse_candidate_ids(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())

def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)

def _completed_predictions(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["candidate_id", "split_seed"])
    return {
        (str(row["candidate_id"]), int(row["split_seed"]))
        for row in frame[["candidate_id", "split_seed"]].drop_duplicates().to_dict("records")
    }

def _random_splits(n: int, seed: int, cv_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))
    placeholder = np.zeros(n, dtype=float)
    return [(fit_idx, valid_idx) for fit_idx, valid_idx in splitter.split(placeholder)]

def _random_oof_for_spec(matrix, spec, seed: int, cv_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = matrix.role("train").reset_index(drop=True)
    y = train["y"].to_numpy(dtype=float)
    is_transfer_blend = spec.family in {"transfer_blend", "calibrated_transfer_blend"}
    if is_transfer_blend:
        x = np.empty((len(train), 0), dtype=float)
        baseline = np.zeros(len(train), dtype=float)
    else:
        feature_columns = feature_columns_for_spec(matrix, spec)
        x = _feature_matrix_for_spec(train, feature_columns, spec)
        baseline = residual_baseline_for_spec(train, spec)

    oof = np.zeros(len(train), dtype=float)
    fold_ids = np.full(len(train), -1, dtype=int)
    fold_rows = []
    for fold_id, (fit_idx, valid_idx) in enumerate(_random_splits(len(train), seed, cv_folds)):
        if is_transfer_blend:
            preds, train_preds = _fit_transfer_blend_fold(
                train,
                matrix,
                y,
                spec,
                fit_idx,
                valid_idx,
                return_fit_predictions=True,
            )
        else:
            preds, train_preds = _fit_predict_raw_candidate(
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
            if is_transfer_blend:
                raise ValueError(
                    "Top-level calibration_method is not supported for transfer_blend OOF export; "
                    "use calibrated_transfer_blend candidates instead."
                )
            fit_frame = train.iloc[fit_idx].reset_index(drop=True)
            fit_x = x[fit_idx]
            fit_y = y[fit_idx]
            fit_baseline = baseline[fit_idx]
            calibration_train_pred = _oof_raw_predictions(
                fit_frame,
                fit_x,
                fit_y,
                fit_baseline,
                spec,
                cv_folds=min(cv_folds, fit_frame["canonical_smiles"].nunique()),
            )
            preds = calibrate_predictions(
                calibration_train_pred,
                fit_y,
                preds,
                method=spec.calibration_method,
                train_frame=fit_frame,
                query_frame=train.iloc[valid_idx].reset_index(drop=True),
            )
            train_preds = calibrate_predictions(
                calibration_train_pred,
                fit_y,
                train_preds,
                method=spec.calibration_method,
                train_frame=fit_frame,
                query_frame=fit_frame,
            )
        oof[valid_idx] = preds
        fold_ids[valid_idx] = int(fold_id)
        fold_rows.append(
            {
                "candidate_id": spec.candidate_id,
                "split_seed": int(seed),
                "fold_id": int(fold_id),
                "fold_r2": float(r2_score(y[valid_idx], preds)),
                "fold_rmse": float(mean_squared_error(y[valid_idx], preds) ** 0.5),
                "fold_train_r2": float(r2_score(y[fit_idx], train_preds)),
                "n_valid": int(len(valid_idx)),
            }
        )

    predictions = train[["sample_id", "canonical_smiles", "time_h", "y"]].copy()
    predictions = predictions.rename(columns={"y": "y_true"})
    predictions["candidate_id"] = spec.candidate_id
    predictions["split_seed"] = int(seed)
    predictions["random_cv_fold"] = fold_ids
    predictions["random_oof_pred"] = oof

    metrics = pd.DataFrame(
        [
            {
                "candidate_id": spec.candidate_id,
                "split_seed": int(seed),
                "random_cv_oof_r2": float(r2_score(y, oof)),
                "random_cv_oof_rmse": float(mean_squared_error(y, oof) ** 0.5),
                "cv_folds": int(cv_folds),
            }
        ]
    )
    return metrics, pd.DataFrame(fold_rows), predictions

def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in metrics.groupby("candidate_id", sort=True):
        r2 = pd.to_numeric(group["random_cv_oof_r2"], errors="coerce")
        rmse = pd.to_numeric(group["random_cv_oof_rmse"], errors="coerce")
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "seed_count": int(group["split_seed"].nunique()),
                "random_cv_oof_r2_mean": float(r2.mean()),
                "random_cv_oof_r2_min": float(r2.min()),
                "random_cv_oof_r2_max": float(r2.max()),
                "random_cv_oof_rmse_mean": float(rmse.mean()),
                "passes_random_cv80_target": bool(r2.mean() > 0.80),
            }
        )
    return pd.DataFrame(rows)

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seed_list, allow_subset=bool(args.allow_seed_subset))

    predictions_path = out_dir / "true_random_oof_predictions.csv"
    metrics_path = out_dir / "true_random_oof_metrics.csv"
    folds_path = out_dir / "true_random_oof_fold_metrics.csv"
    audit_path = out_dir / "leakage_audit_by_seed.csv"
    completed = _completed_predictions(predictions_path)

    bundle = load_data_contract(root)
    candidates = build_candidates(args.mode, None, args.gpu)
    candidate_ids = _parse_candidate_ids(args.candidate_ids)
    if candidate_ids:
        specs = [candidate_by_id(candidates, candidate_id) for candidate_id in candidate_ids]
    else:
        current = candidate_by_id(candidates, CURRENT_ID)
        atom = candidate_by_id(candidates, ATOM_ID)
        shrink = replace(current, candidate_id=SHRINK_ID, residual_shrinkage=0.35, calibration_method="none")
        specs = [current, atom, shrink]

    for seed in seeds:
        split = build_frozen_split(bundle, split_seed=seed)
        matrix = build_model_matrix(bundle, split, use_cache=True)
        audit = audit_split_and_features(split, matrix)
        _append_csv(audit_path, pd.DataFrame([{"split_seed": int(seed), "split_manifest_hash": split.manifest_hash, **audit}]))
        if int(audit["hard_failure_count"]):
            raise RuntimeError(f"Leakage audit failed for split seed {seed}: {audit}")
        for spec in specs:
            key = (str(spec.candidate_id), int(seed))
            if key in completed:
                continue
            metrics, fold_metrics, predictions = _random_oof_for_spec(matrix, spec, int(seed), args.cv_folds)
            metrics["split_manifest_hash"] = split.manifest_hash
            fold_metrics["split_manifest_hash"] = split.manifest_hash
            predictions["split_manifest_hash"] = split.manifest_hash
            _append_csv(metrics_path, metrics)
            _append_csv(folds_path, fold_metrics)
            _append_csv(predictions_path, predictions)
            completed.add(key)

        metrics_all = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        summary_table = _summarize(metrics_all) if not metrics_all.empty else pd.DataFrame()
        summary_table.to_csv(out_dir / "true_random_oof_summary.csv", index=False)
        payload = {
            "run_id": args.run_id,
            "diagnostic_only": True,
            "fixed_best10_seeds": list(seeds),
            "formal_fixed_best10_seed_set": bool(seeds == FIXED_BEST10_SEEDS),
            "allow_seed_subset": bool(args.allow_seed_subset),
            "candidate_ids": [spec.candidate_id for spec in specs],
            "completed_seed_candidate_pairs": int(len(completed)),
            "random_cv_target_r2": 0.80,
            "any_component_passes_random_cv80": bool(
                not summary_table.empty and summary_table["passes_random_cv80_target"].fillna(False).any()
            ),
            "summary": summary_table.to_dict("records"),
            "scientific_boundary": (
                "Diagnostic true random KFold OOF predictions for fixed-best10 component candidates. "
                "Random KFold is a control only and does not replace compound-group formal CV."
            ),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export true random KFold OOF predictions for fixed-best10 components.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "quick"], default="quick")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed-list", default=",".join(str(seed) for seed in FIXED_BEST10_SEEDS))
    parser.add_argument("--allow-seed-subset", action="store_true")
    parser.add_argument(
        "--candidate-ids",
        default="",
        help="Optional comma-separated candidate ids. Defaults to the original three ET component candidates.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    output = run(parse_args())
    print(f"wrote {output}")

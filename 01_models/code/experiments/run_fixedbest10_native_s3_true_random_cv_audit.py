from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "code" / "experiments"
SRC = ROOT / "code" / "src"
for path in (EXPERIMENTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_fixedbest10_true_random_cv_component_oof import _random_oof_for_spec
from nci_cho_transfer.audit import audit_split_and_features
from nci_cho_transfer.data_contract import load_data_contract
from nci_cho_transfer.features import build_model_matrix
from nci_cho_transfer.models import build_candidates, candidate_by_id
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

SOURCE_CONFIG = {
    "randomforest": {
        "kind": "selected",
        "path": "data_processed/components/rf_configuration/cv_selected_by_seed.csv",
    },
    "xgboost": {
        "kind": "selected",
        "path": "data_processed/components/xgb_configuration/cv_selected_by_seed.csv",
    },
    "catboost": {
        "kind": "blend",
        "path": "data_processed/components/catboost_configuration/traincv_selected_weights.csv",
    },
    "svr": {
        "kind": "blend",
        "path": "data_processed/components/svr_configuration/traincv_selected_weights.csv",
    },
}

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if seeds != FIXED_BEST10_SEEDS:
        raise ValueError(f"Expected fixed best10 seeds exactly: {FIXED_BEST10_SEEDS}")
    return seeds

def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if path.exists():

        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [column for column in header if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV append schema mismatch: missing={missing}")
        frame = frame.loc[:, header]
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)

def _completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["family", "split_seed"])
    return {(str(row["family"]), int(row["split_seed"])) for row in frame.drop_duplicates().to_dict("records")}

def _families(value: str) -> list[str]:
    requested = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(SOURCE_CONFIG))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")
    return requested

def _load_source_rows(root: Path, family: str, seeds: tuple[int, ...]) -> pd.DataFrame:
    config = SOURCE_CONFIG[family]
    frame = pd.read_csv(_resolve(root, config["path"]))
    frame["split_seed"] = frame["split_seed"].astype(int)
    observed = tuple(sorted(frame["split_seed"].unique().tolist()))
    if observed != seeds:
        raise ValueError(f"{family} source seeds do not match fixed best10 seeds: observed={observed}")
    if int(frame.duplicated("split_seed").sum()):
        raise ValueError(f"{family} source has duplicated split_seed rows.")
    return frame.sort_values("split_seed").reset_index(drop=True)

def _parse_blend_row(row: dict) -> tuple[list[str], list[float]]:
    ids = [part.strip() for part in str(row["candidate_ids"]).split(",") if part.strip()]
    weights = [float(part.strip()) for part in str(row["selected_weights"]).split(",") if part.strip()]
    if len(ids) != len(weights):
        raise ValueError(f"Blend candidate/weight length mismatch: {row}")
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"Blend weights do not sum to 1: {row}")
    return ids, weights

def _selected_seed_predictions(matrix, candidates, row: dict, seed: int, cv_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_id = str(row["selected_candidate_id"])
    spec = candidate_by_id(candidates, candidate_id)
    metrics, _, predictions = _random_oof_for_spec(matrix, spec, seed, cv_folds)
    predictions = predictions.rename(columns={"random_oof_pred": "y_pred"})
    predictions["component_candidate_id"] = candidate_id
    predictions["component_weight"] = 1.0
    return metrics, predictions

def _blend_seed_predictions(matrix, candidates, row: dict, seed: int, cv_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_ids, weights = _parse_blend_row(row)
    component_metrics = []
    merged = None
    for candidate_id, weight in zip(candidate_ids, weights):
        spec = candidate_by_id(candidates, candidate_id)
        metrics, _, predictions = _random_oof_for_spec(matrix, spec, seed, cv_folds)
        metrics["component_weight"] = float(weight)
        component_metrics.append(metrics)
        part = predictions[
            ["sample_id", "canonical_smiles", "time_h", "y_true", "split_seed", "random_cv_fold", "random_oof_pred"]
        ].rename(columns={"random_oof_pred": candidate_id})
        merged = part if merged is None else merged.merge(
            part,
            on=["sample_id", "canonical_smiles", "time_h", "y_true", "split_seed", "random_cv_fold"],
            how="inner",
        )
    assert merged is not None
    y_pred = np.zeros(len(merged), dtype=float)
    for candidate_id, weight in zip(candidate_ids, weights):
        y_pred += float(weight) * merged[candidate_id].to_numpy(dtype=float)
    predictions = merged[["sample_id", "canonical_smiles", "time_h", "y_true", "split_seed", "random_cv_fold"]].copy()
    predictions["candidate_id"] = f"{row.get('family', 'native')}_native_s3_blend"
    predictions["y_pred"] = y_pred
    predictions["component_candidate_id"] = ",".join(candidate_ids)
    predictions["component_weight"] = ",".join(f"{weight:.6f}" for weight in weights)
    return pd.concat(component_metrics, ignore_index=True), predictions

def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame(rows)
    for family, group in metrics.groupby("family", sort=True):
        r2 = pd.to_numeric(group["random_cv_oof_r2"], errors="coerce")
        rmse = pd.to_numeric(group["random_cv_oof_rmse"], errors="coerce")
        rows.append(
            {
                "family": str(family),
                "seed_count": int(group["split_seed"].nunique()),
                "random_cv_oof_r2_mean": float(r2.mean()),
                "random_cv_oof_r2_min": float(r2.min()),
                "random_cv_oof_r2_max": float(r2.max()),
                "random_cv_oof_rmse_mean": float(rmse.mean()),
                "passes_random_cv80_target": bool(r2.mean() > 0.80),
            }
        )
    return pd.DataFrame(rows)

def _source_config_for_completed(families: list[str], metrics: pd.DataFrame) -> dict:
    completed_families = set(families)
    if not metrics.empty and "family" in metrics.columns:
        completed_families |= set(metrics["family"].astype(str).unique().tolist())
    return {family: SOURCE_CONFIG[family] for family in sorted(completed_families)}

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seed_list)
    families = _families(args.families)

    metrics_path = out_dir / "native_s3_true_random_cv_metrics.csv"
    component_metrics_path = out_dir / "native_s3_true_random_cv_component_metrics.csv"
    predictions_path = out_dir / "native_s3_true_random_cv_predictions.csv"
    audit_path = out_dir / "leakage_audit_by_seed.csv"
    completed = _completed(metrics_path)

    bundle = load_data_contract(root)
    candidates = build_candidates(args.mode, None, args.gpu)
    sources = {family: _load_source_rows(root, family, seeds) for family in families}

    for seed in seeds:
        split = build_frozen_split(bundle, split_seed=seed)
        matrix = build_model_matrix(bundle, split, use_cache=True)
        audit = audit_split_and_features(split, matrix)
        if not audit_path.exists() or seed not in set(pd.read_csv(audit_path)["split_seed"].astype(int)):
            _append_csv(
                audit_path,
                pd.DataFrame([{"split_seed": seed, "split_manifest_hash": split.manifest_hash, **audit}]),
            )
        if int(audit["hard_failure_count"]):
            raise RuntimeError(f"Leakage audit failed for split seed {seed}: {audit}")

        for family in families:
            key = (family, int(seed))
            if key in completed:
                continue
            row = sources[family][sources[family]["split_seed"].astype(int) == int(seed)].iloc[0].to_dict()
            if SOURCE_CONFIG[family]["kind"] == "selected":
                component_metrics, predictions = _selected_seed_predictions(matrix, candidates, row, seed, args.cv_folds)
                source_candidate = str(row["selected_candidate_id"])
            else:
                component_metrics, predictions = _blend_seed_predictions(matrix, candidates, row, seed, args.cv_folds)
                source_candidate = str(row["candidate_ids"])
                predictions["candidate_id"] = f"{family}_native_s3_blend"

            y_true = predictions["y_true"].to_numpy(dtype=float)
            y_pred = predictions["y_pred"].to_numpy(dtype=float)
            metric = pd.DataFrame(
                [
                    {
                        "family": family,
                        "split_seed": int(seed),
                        "split_manifest_hash": split.manifest_hash,
                        "source_candidate_id": source_candidate,
                        "selection_source": SOURCE_CONFIG[family]["path"],
                        "random_cv_oof_r2": float(r2_score(y_true, y_pred)),
                        "random_cv_oof_rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
                        "n": int(len(predictions)),
                    }
                ]
            )
            component_metrics["family"] = family
            component_metrics["split_manifest_hash"] = split.manifest_hash
            predictions["family"] = family
            predictions["split_manifest_hash"] = split.manifest_hash
            predictions["source_candidate_id"] = source_candidate
            _append_csv(metrics_path, metric)
            _append_csv(component_metrics_path, component_metrics)
            _append_csv(predictions_path, predictions)
            completed.add(key)

        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        summary_table = _summarize(metrics)
        summary_table.to_csv(out_dir / "native_s3_true_random_cv_summary.csv", index=False)
        payload = {
            "run_id": args.run_id,
            "diagnostic_only": True,
            "fixed_best10_seeds": list(seeds),
            "families_requested_this_run": families,
            "families_completed_in_artifact": sorted(metrics["family"].astype(str).unique().tolist())
            if not metrics.empty and "family" in metrics.columns
            else [],
            "completed_family_seed_pairs": int(len(completed)),
            "random_cv_target_r2": float(args.random_cv_target),
            "any_family_passes_random_cv80": bool(
                not summary_table.empty and summary_table["passes_random_cv80_target"].fillna(False).any()
            ),
            "leakage_hard_failure_count": int(
                pd.to_numeric(pd.read_csv(audit_path)["hard_failure_count"], errors="coerce").sum()
            )
            if audit_path.exists()
            else 0,
            "summary": summary_table.to_dict("records"),
            "source_config": _source_config_for_completed(families, metrics),
            "scientific_boundary": (
                "Diagnostic true random KFold audit for fixedbest10 native S3 sources. Candidate identities and "
                "blend weights are taken from train-only compound-group CV artifacts; random KFold is report-only "
                "and does not replace formal compound-group CV."
            ),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="True random CV audit for fixedbest10 native S3 sources.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--families", default="randomforest,svr")
    parser.add_argument("--mode", choices=["smoke", "quick"], default="quick")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-cv-target", type=float, default=0.80)
    parser.add_argument("--seed-list", default=",".join(str(seed) for seed in FIXED_BEST10_SEEDS))
    return parser.parse_args()

if __name__ == "__main__":
    output = run(parse_args())
    print(f"wrote {output}")

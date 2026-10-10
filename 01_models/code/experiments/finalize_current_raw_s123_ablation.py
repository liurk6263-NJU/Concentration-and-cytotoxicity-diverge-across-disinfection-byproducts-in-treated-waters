from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
for path in (SRC, ROOT / "code" / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_fixedbest10_true_random_cv_component_oof import _random_oof_for_spec
from run_formal_model_progression import _raw_no_transfer_baseline_spec
from nci_cho_transfer.data_contract import load_data_contract
from nci_cho_transfer.features import ModelMatrix, build_model_matrix
from nci_cho_transfer.models import build_candidates, candidate_by_id, fit_selected_and_evaluate
from nci_cho_transfer.splits import build_frozen_split

SEEDS = [20260021, 20260106, 20260290, 20260307, 20260605, 20260730, 20260738, 20260755, 20260998, 20262000]
TRANSFER_PREFIXES = (
    "teacher_", "source_teacher_", "pathway_transfer_", "profile_transfer_", "local_profile_transfer_",
    "risk_profile_transfer_", "risk_adjusted_transfer_", "conditioned_transfer_", "disagreement_transfer_",
    "residualized_transfer_", "calibrated_transfer_",
)

S2_PAIR_ID = "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_pair"

def _parse_seeds(value: str, allow_subset: bool) -> list[int]:
    seeds = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(set(seeds)) != len(seeds) or not seeds or any(seed not in SEEDS for seed in seeds):
        raise ValueError("seed list must be a unique non-empty subset of frozen best10 seeds")
    if not allow_subset and seeds != SEEDS:
        raise ValueError("full run requires the frozen best10 seed list")
    return seeds

def _stage_specs(registry, interference_candidate_id: str = S2_PAIR_ID):

    s1 = _raw_no_transfer_baseline_spec(registry, "base_extratrees_raw_default")
    pair = candidate_by_id(registry, interference_candidate_id)
    if pair is None or "transfer" not in str(pair.feature_group):
        raise KeyError(f"registered transfer interference candidate not found: {interference_candidate_id}")
    if not pair.paired_counterpart_id:
        raise KeyError(f"registered transfer candidate has no paired counterpart: {interference_candidate_id}")
    s2 = candidate_by_id(registry, pair.paired_counterpart_id)
    if s2 is None or "no_transfer" not in str(s2.feature_group):
        raise KeyError(f"registered no-transfer counterpart not found: {pair.paired_counterpart_id}")
    return s1, s2, pair.candidate_id

def _shuffle_transfer_features(
    matrix: ModelMatrix,
    split_seed: int,
    repeat: int,
    mode: str = "shared_layer",
) -> ModelMatrix:

    if mode not in {"shared_layer", "independent_columns", "train_role_mean"}:
        raise ValueError(f"unsupported transfer permutation mode: {mode}")
    frame = matrix.frame.copy()
    transfer = [c for c in matrix.feature_columns if c.startswith(TRANSFER_PREFIXES)]
    if not transfer:
        raise ValueError("no registered transfer columns available for interference permutation")
    if mode == "train_role_mean":
        train_mask = frame["split_role"].eq("train")
        if not train_mask.any():
            raise ValueError("train_role_mean requires train-role rows")
        means = frame.loc[train_mask, transfer].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        if means.isna().any():
            raise ValueError("train_role_mean encountered non-finite transfer feature means")
        frame.loc[:, transfer] = means.to_numpy(dtype=float)
        return ModelMatrix(frame=frame, feature_columns=list(matrix.feature_columns))
    for role, indices in frame.groupby("split_role", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        if len(indices) < 2:
            continue

        base_seed = int(split_seed) + 1_000_003 * int(repeat) + sum(map(ord, str(role)))
        if mode == "shared_layer":
            rng = np.random.default_rng(base_seed)
            order = rng.permutation(len(indices))
            if np.array_equal(order, np.arange(len(indices))):
                order = np.roll(order, 1)
            values = frame.loc[indices, transfer].to_numpy(dtype=float)
            frame.loc[indices, transfer] = values[order]
        else:
            for column_index, column in enumerate(transfer):
                rng = np.random.default_rng(base_seed + 104_729 * (column_index + 1))
                order = rng.permutation(len(indices))
                if np.array_equal(order, np.arange(len(indices))):
                    order = np.roll(order, 1)
                frame.loc[indices, column] = frame.loc[indices, column].to_numpy(dtype=float)[order]
    return ModelMatrix(frame=frame, feature_columns=list(matrix.feature_columns))

def _score(frame: pd.DataFrame, pred: str, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        y = group["y_true"].to_numpy(dtype=float)
        p = group[pred].to_numpy(dtype=float)
        rows.append({**dict(zip(groups, values, strict=True)), "n": len(group), "r2": float(r2_score(y, p)), "rmse": float(mean_squared_error(y, p) ** 0.5)})
    return pd.DataFrame(rows)

def _attach_train_oof(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    stage_column: str,
    oof_column: str,
    prediction_column: str = "random_oof_pred",
) -> pd.DataFrame:

    result = frame.copy()
    audit_column = {
        "S1_no_transfer": "S1_in_sample_train",
        "S2_no_transfer": "S2_in_sample_train",
        "S3_transfer_raw_full_chain": "S3_in_sample_train",
    }.get(stage_column, f"{stage_column}_in_sample_train")
    result[audit_column] = result[stage_column]
    keys = ["split_seed", "sample_id", "canonical_smiles", "time_h", "y_true"]
    merge_keys = [key for key in keys if key in oof.columns]
    values = oof[merge_keys + [prediction_column]].rename(columns={prediction_column: oof_column})
    result = result.merge(values, on=merge_keys, how="left", validate="many_to_one")
    train_mask = result["split_role"].eq("train")
    if result.loc[train_mask, oof_column].isna().any():
        raise ValueError(f"missing OOF predictions for train role: {stage_column}")
    result.loc[train_mask, stage_column] = result.loc[train_mask, oof_column]
    return result

def _merge_stage_predictions(source, source_random, s1_random, s2_random, s1_holdout, s2_holdout):
    rkeys = ["split_seed", "random_cv_fold", "sample_id", "canonical_smiles", "time_h", "y_true"]
    hkeys = ["split_seed", "split_role", "sample_id", "canonical_smiles", "time_h", "y_true"]
    if "random_cv_fold" not in source.columns:
        holdout = source.merge(
            s1_holdout[hkeys + ["y_pred"]].rename(columns={"y_pred": "S1_no_transfer"}), on=hkeys, how="inner", validate="one_to_one"
        )
        holdout = holdout.merge(s2_holdout[hkeys + ["y_pred"]].rename(columns={"y_pred": "S2_no_transfer"}), on=hkeys, how="inner", validate="one_to_one")
        holdout["S3_transfer_raw_full_chain"] = holdout["blend_y_pred"].astype(float)
        holdout = _attach_train_oof(holdout, s1_random, "S1_no_transfer", "S1_train_oof")
        holdout = _attach_train_oof(holdout, s2_random, "S2_no_transfer", "S2_train_oof")
        holdout = _attach_train_oof(
            holdout, source_random, "S3_transfer_raw_full_chain", "S3_train_oof", prediction_column="blend_y_pred"
        )
        return pd.DataFrame(), holdout
    source_rkeys = rkeys if "canonical_smiles" in source.columns else [key for key in rkeys if key != "canonical_smiles"]
    s1_value_columns = source_rkeys + ["random_oof_pred"]
    if "canonical_smiles" not in source.columns:
        s1_value_columns.insert(3, "canonical_smiles")
    random = source.merge(
        s1_random[s1_value_columns].rename(columns={"random_oof_pred": "S1_no_transfer"}),
        on=source_rkeys,
        how="inner",
        validate="one_to_one",
    )
    random = random.merge(s2_random[rkeys + ["random_oof_pred"]].rename(columns={"random_oof_pred": "S2_no_transfer"}), on=rkeys, how="inner", validate="one_to_one")
    random["split_role"] = "random_cv_oof"
    random["S3_transfer_raw_full_chain"] = random["blend_y_pred"].astype(float)
    holdout = source.iloc[0:0].copy()
    return random, holdout

def _interference(source, bundle, pair, cv_folds, repeats, mode="shared_layer"):
    rows = []
    for seed in sorted(source["split_seed"].astype(int).unique()):
        split = build_frozen_split(bundle, split_seed=seed)
        matrix = build_model_matrix(bundle, split, use_cache=True)
        for repeat in range(int(repeats)):
            shuffled = _shuffle_transfer_features(matrix, seed, repeat, mode=mode)
            _, _, random_pred = _random_oof_for_spec(shuffled, pair, seed, cv_folds)
            _, holdout_pred = fit_selected_and_evaluate(shuffled, pair)
            random_pred = random_pred.rename(columns={"random_oof_pred": "ablation_pred"})
            holdout_pred = holdout_pred.rename(columns={"y_pred": "ablation_pred"})
            holdout_pred["split_seed"] = int(seed)
            rkeys = ["split_seed", "random_cv_fold", "sample_id", "canonical_smiles", "time_h", "y_true"]
            hkeys = ["split_seed", "split_role", "sample_id", "canonical_smiles", "time_h", "y_true"]
            r = source[source["split_seed"].eq(seed) & source["random_cv_fold"].notna()].merge(random_pred[rkeys + ["ablation_pred"]], on=rkeys, how="inner", validate="one_to_one")
            h = source[source["split_seed"].eq(seed) & source["split_role"].notna()].merge(holdout_pred[hkeys + ["ablation_pred"]], on=hkeys, how="inner", validate="one_to_one")
            for frame in (r, h):
                frame["split_role"] = frame.get("split_role", "random_cv_oof")
                frame["ablation_step"] = "interference_transfer"
                frame["repeat"] = int(repeat)
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)

def _make_ablation_baseline(stages: pd.DataFrame) -> pd.DataFrame:
    baseline = stages.copy()
    baseline["ablation_step"] = "no_transfer"
    baseline["ablation_pred"] = baseline["S2_no_transfer"]
    baseline["repeat"] = -1
    return baseline

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    source_dir = root / "data_processed" / "components" / args.source_run_id
    source_random = pd.read_csv(source_dir / "formal_random_cv_oof_predictions.csv", low_memory=False)
    source_holdout = pd.read_csv(source_dir / "formal_holdout_predictions.csv", low_memory=False)
    seeds = _parse_seeds(args.seed_list, args.allow_seed_subset)
    if any(seed not in set(source_random["split_seed"].astype(int)) for seed in seeds) or any(seed not in set(source_holdout["split_seed"].astype(int)) for seed in seeds):
        raise ValueError("source artifact must contain exactly the frozen best10 seeds")
    bundle = load_data_contract(root)
    registry = build_candidates("quick", None, 0)
    s1, s2, pair_id = _stage_specs(registry, args.interference_candidate_id)
    pair = candidate_by_id(registry, pair_id)
    random_parts, holdout_parts = [], []
    for seed in seeds:
        split = build_frozen_split(bundle, split_seed=seed)
        matrix = build_model_matrix(bundle, split, use_cache=True)
        _, _, s1_random = _random_oof_for_spec(matrix, s1, seed, args.cv_folds)
        _, _, s2_random = _random_oof_for_spec(matrix, s2, seed, args.cv_folds)
        _, s1_holdout = fit_selected_and_evaluate(matrix, s1)
        _, s2_holdout = fit_selected_and_evaluate(matrix, s2)
        s1_holdout["split_seed"] = int(seed)
        s2_holdout["split_seed"] = int(seed)
        random_source_seed = source_random[source_random["split_seed"].eq(seed)]
        random, holdout = _merge_stage_predictions(
            random_source_seed, random_source_seed, s1_random, s2_random, s1_holdout, s2_holdout
        )
        random_parts.append(random)
        holdout_parts.append(
            _merge_stage_predictions(
                source_holdout[source_holdout["split_seed"].eq(seed)], random_source_seed,
                s1_random, s2_random, s1_holdout, s2_holdout,
            )[1]
        )
    random = pd.concat(random_parts, ignore_index=True)
    holdout = pd.concat(holdout_parts, ignore_index=True)
    stages = pd.concat([random, holdout], ignore_index=True)
    out = root / "data_processed" / "components" / args.run_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    stages.to_csv(out / "raw_s123_stage_predictions.csv", index=False)
    metric_parts = []
    for stage, column in (("S1", "S1_no_transfer"), ("S2", "S2_no_transfer"), ("S3", "S3_transfer_raw_full_chain")):
        metric_parts.append(_score(random, column, ["split_seed"]).assign(cv_kind="random_cv", stage=stage))
        metric_parts.append(_score(holdout, column, ["split_seed", "split_role"]).assign(cv_kind="holdout", stage=stage))
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(out / "s123_metrics_by_seed_role.csv", index=False)
    baseline = _make_ablation_baseline(stages)
    transfer = stages.copy(); transfer["ablation_step"] = "transfer"; transfer["ablation_pred"] = transfer["S3_transfer_raw_full_chain"]; transfer["repeat"] = -1
    interference = _interference(stages, bundle, pair, args.cv_folds, args.shuffle_repeats, mode=args.interference_mode)
    ablation = pd.concat([baseline, interference, transfer], ignore_index=True)
    ablation_metrics = _score(ablation, "ablation_pred", ["split_seed", "split_role", "ablation_step", "repeat"])
    aggregate = ablation_metrics.groupby(["split_role", "ablation_step"], sort=True).agg(r2_mean=("r2", "mean"), rmse_mean=("rmse", "mean"), seed_count=("split_seed", "nunique")).reset_index()
    ablation.to_csv(out / "three_arm_ablation_predictions.csv", index=False)
    ablation_metrics.to_csv(out / "three_arm_ablation_metrics.csv", index=False)
    aggregate.to_csv(out / "three_arm_ablation_aggregate.csv", index=False)
    summary = {
        "run_id": args.run_id, "source_run_id": args.source_run_id, "fixed_best10_seeds": seeds,
        "stage_definitions": {
            "S1": "base_extratrees_raw_default + no_transfer; no calibration/local/descriptor optimization",
            "S2": f"strict registered no-transfer counterpart of {pair_id}; CHO-only optimized model without transfer features",
            "S3": f"fixed blend_y_pred artifact from {args.source_run_id}",
        },
        "interference_mode": f"registered_{pair_id}_{args.interference_mode}_permutation",
        "full_chain_feature_inputs_available": False,
        "test_validation_experimental_labels_used_for_selection": False, "prediction_scaling_used": False,
        "final_prediction_permutation_used": False, "shuffle_repeats_per_seed": int(args.shuffle_repeats),
        "train_role_metric_kind": "random_cv_oof_for_each_frozen_seed",
        "in_sample_train_predictions_retained_for_audit": True,
        "stage_metrics": metrics.to_dict("records"), "ablation_aggregate": aggregate.to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--source-run-id", default="source_teacher")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    parser.add_argument(
        "--interference-mode",
        choices=("shared_layer", "independent_columns", "train_role_mean"),
        default="shared_layer",
    )
    parser.add_argument(
        "--interference-candidate-id",
        default=S2_PAIR_ID,
        help="Registered transfer candidate used only for the interference arm.",
    )
    parser.add_argument("--seed-list", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--allow-seed-subset", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    print(f"wrote {run(parse_args())}")

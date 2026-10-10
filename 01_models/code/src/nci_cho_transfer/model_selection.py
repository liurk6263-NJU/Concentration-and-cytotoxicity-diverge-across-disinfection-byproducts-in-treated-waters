from __future__ import annotations

import pandas as pd

HOLDOUT_METRIC_COLUMNS = {
    "r2_test",
    "r2_validation",
    "r2_experimental",
    "test_r2",
    "validation_r2",
    "experimental_r2",
    "R2_test",
    "R2_validation",
    "R2_experimental",
}
CV_OOF_EQUIVALENCE_BAND = 0.003
PAIRED_CV_DELTA_R2_MIN = 0.005
PAIRED_CV_ERROR_REDUCTION_MIN = 0.0

def select_candidate_from_cv(cv_metrics: pd.DataFrame) -> dict:
    leaked = HOLDOUT_METRIC_COLUMNS & set(cv_metrics.columns)
    if leaked:
        raise ValueError(f"Candidate selection received holdout metrics: {sorted(leaked)}")
    required = {"candidate_id", "cv_oof_r2"}
    missing = required - set(cv_metrics.columns)
    if missing:
        raise ValueError(f"CV metrics missing required columns: {sorted(missing)}")
    sortable = cv_metrics.copy()
    if "cv_fold_r2_min" in sortable.columns:
        best_oof = float(sortable["cv_oof_r2"].max())
        sortable = sortable[sortable["cv_oof_r2"] >= best_oof - CV_OOF_EQUIVALENCE_BAND].copy()
        if "cv_generalization_gap" in sortable.columns:
            ordered = sortable.sort_values(
                ["cv_generalization_gap", "cv_fold_r2_min", "cv_oof_r2", "candidate_id"],
                ascending=[True, False, False, True],
            )
        else:
            ordered = sortable.sort_values(
                ["cv_fold_r2_min", "cv_oof_r2", "candidate_id"],
                ascending=[False, False, True],
            )
    else:
        ordered = sortable.sort_values(["cv_oof_r2", "candidate_id"], ascending=[False, True])
    return ordered.iloc[0].to_dict()

def select_paired_transfer_positive_from_cv(cv_metrics: pd.DataFrame) -> dict:
    leaked = HOLDOUT_METRIC_COLUMNS & set(cv_metrics.columns)
    if leaked:
        raise ValueError(f"Candidate selection received holdout metrics: {sorted(leaked)}")
    required = {"candidate_id", "paired_counterpart_id", "cv_oof_r2", "cv_oof_rmse"}
    missing = required - set(cv_metrics.columns)
    if missing:
        raise ValueError(f"CV metrics missing required columns: {sorted(missing)}")

    by_id = cv_metrics.set_index("candidate_id", drop=False)
    pair_rows = []
    for row in cv_metrics[cv_metrics["paired_counterpart_id"].astype(str).ne("")].to_dict("records"):
        counterpart_id = str(row.get("paired_counterpart_id", ""))
        if counterpart_id not in by_id.index:
            continue
        counterpart = by_id.loc[counterpart_id]
        if isinstance(counterpart, pd.DataFrame):
            counterpart = counterpart.iloc[0]
        full_r2 = float(row["cv_oof_r2"])
        baseline_r2 = float(counterpart["cv_oof_r2"])
        full_rmse = float(row["cv_oof_rmse"])
        baseline_rmse = float(counterpart["cv_oof_rmse"])
        rmse_reduction = (baseline_rmse - full_rmse) / baseline_rmse if baseline_rmse else float("nan")
        if full_r2 - baseline_r2 < PAIRED_CV_DELTA_R2_MIN:
            continue
        if rmse_reduction < PAIRED_CV_ERROR_REDUCTION_MIN:
            continue
        enriched = dict(row)
        enriched["cv_paired_counterpart_r2"] = baseline_r2
        enriched["cv_paired_counterpart_rmse"] = baseline_rmse
        enriched["cv_paired_delta_r2"] = full_r2 - baseline_r2
        enriched["cv_paired_rmse_reduction_fraction"] = rmse_reduction
        pair_rows.append(enriched)

    if pair_rows:
        paired = pd.DataFrame(pair_rows)
        best_oof = float(paired["cv_oof_r2"].max())
        paired = paired[paired["cv_oof_r2"] >= best_oof - CV_OOF_EQUIVALENCE_BAND].copy()
        sort_columns = ["cv_paired_delta_r2", "cv_paired_rmse_reduction_fraction"]
        ascending = [False, False]
        if "cv_generalization_gap" in paired.columns:
            sort_columns.append("cv_generalization_gap")
            ascending.append(True)
        if "cv_fold_r2_min" in paired.columns:
            sort_columns.append("cv_fold_r2_min")
            ascending.append(False)
        sort_columns.extend(["cv_oof_r2", "candidate_id"])
        ascending.extend([False, True])
        return paired.sort_values(sort_columns, ascending=ascending).iloc[0].to_dict()

    return select_candidate_from_cv(cv_metrics)

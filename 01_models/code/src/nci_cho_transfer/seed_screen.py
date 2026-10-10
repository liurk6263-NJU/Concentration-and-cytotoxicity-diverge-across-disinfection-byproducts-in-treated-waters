from __future__ import annotations

import pandas as pd

R2_ROLES = ("train", "test", "validation", "experimental")
STRICT_10SEED_MIN_R2 = 0.85
STRICT_10SEED_HOLDOUT_MIN_R2 = 0.80
NO_OVERFIT_GAP_MAX = 0.15
BASE_MODEL_MIN_R2 = 0.80
BASE_MODEL_HOLDOUT_MIN_R2 = 0.70
FIVE_MODEL_MIN_R2 = 0.70
FIVE_MODEL_VALIDATION_MIN_R2 = 0.50
FIVE_MODEL_EXPERIMENTAL_MIN_R2 = 0.50
PAIRED_TRANSFER_TEST_MIN_DELTA_R2 = 0.05
PAIRED_TRANSFER_MIN_ERROR_REDUCTION_FRACTION = 0.05

def summarize_10seed_metrics(per_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "split_seed", "split_role", "r2"}
    missing = required - set(per_seed_metrics.columns)
    if missing:
        raise ValueError(f"Per-seed metrics missing required columns: {sorted(missing)}")
    pivot = per_seed_metrics.pivot_table(
        index=["candidate_id", "split_seed"],
        columns="split_role",
        values="r2",
        aggfunc="first",
    ).reset_index()
    missing_roles = set(R2_ROLES) - set(pivot.columns)
    if missing_roles:
        raise ValueError(f"Per-seed metrics missing split roles: {sorted(missing_roles)}")
    rows = []
    for candidate_id, group in pivot.groupby("candidate_id", sort=True):
        row = {
            "candidate_id": candidate_id,
            "seed_count": int(group["split_seed"].nunique()),
            "split_seeds": ",".join(str(int(seed)) for seed in group["split_seed"].tolist()),
        }
        for role in R2_ROLES:
            values = pd.to_numeric(group[role], errors="coerce")
            row[f"r2_{role}_mean"] = float(values.mean())
            row[f"r2_{role}_min"] = float(values.min())
            row[f"r2_{role}_max"] = float(values.max())
        gaps = pd.to_numeric(group["train"], errors="coerce") - pd.to_numeric(group["test"], errors="coerce")
        row["r2_train_test_gap_mean"] = float(gaps.mean())
        row["r2_train_test_gap_max"] = float(gaps.max())
        row["passes_no_overfit_gap"] = bool(row["r2_train_test_gap_max"] <= NO_OVERFIT_GAP_MAX)
        row["passes_10seed_goal"] = passes_10seed_goal(row)
        row["passes_base_model_goal"] = passes_base_model_goal(row)
        row["passes_5model_base_goal"] = passes_5model_base_goal(row)
        rows.append(row)
    return pd.DataFrame(rows)

def passes_10seed_goal(summary_row: dict | pd.Series) -> bool:
    row = dict(summary_row)
    return bool(
        int(row.get("seed_count", 0)) == 10
        and float(row.get("r2_train_mean", -999.0)) > STRICT_10SEED_MIN_R2
        and float(row.get("r2_train_mean", 999.0)) < 1.0
        and float(row.get("r2_test_mean", -999.0)) > STRICT_10SEED_MIN_R2
        and float(row.get("r2_test_mean", 999.0)) < 1.0
        and float(row.get("r2_train_min", -999.0)) > STRICT_10SEED_MIN_R2
        and float(row.get("r2_train_min", 999.0)) < 1.0
        and float(row.get("r2_test_min", -999.0)) > STRICT_10SEED_MIN_R2
        and float(row.get("r2_test_min", 999.0)) < 1.0
        and float(row.get("r2_validation_min", -999.0)) > STRICT_10SEED_HOLDOUT_MIN_R2
        and float(row.get("r2_validation_max", 999.0)) < 1.0
        and float(row.get("r2_experimental_min", -999.0)) > STRICT_10SEED_HOLDOUT_MIN_R2
        and float(row.get("r2_experimental_max", 999.0)) < 1.0
        and bool(row.get("passes_no_overfit_gap", False))
    )

def passes_base_model_goal(summary_row: dict | pd.Series) -> bool:
    row = dict(summary_row)
    return bool(
        int(row.get("seed_count", 0)) == 10
        and float(row.get("r2_train_mean", -999.0)) > BASE_MODEL_MIN_R2
        and float(row.get("r2_train_mean", 999.0)) < 1.0
        and float(row.get("r2_test_mean", -999.0)) > BASE_MODEL_MIN_R2
        and float(row.get("r2_test_mean", 999.0)) < 1.0
        and float(row.get("r2_train_min", -999.0)) > BASE_MODEL_MIN_R2
        and float(row.get("r2_train_min", 999.0)) < 1.0
        and float(row.get("r2_test_min", -999.0)) > BASE_MODEL_MIN_R2
        and float(row.get("r2_test_min", 999.0)) < 1.0
        and float(row.get("r2_validation_min", -999.0)) > BASE_MODEL_HOLDOUT_MIN_R2
        and float(row.get("r2_experimental_min", -999.0)) > BASE_MODEL_HOLDOUT_MIN_R2
    )

def passes_5model_base_goal(summary_row: dict | pd.Series) -> bool:
    row = dict(summary_row)
    return bool(
        int(row.get("seed_count", 0)) == 10
        and float(row.get("r2_train_mean", -999.0)) > FIVE_MODEL_MIN_R2
        and float(row.get("r2_train_mean", 999.0)) < 1.0
        and float(row.get("r2_test_mean", -999.0)) > FIVE_MODEL_MIN_R2
        and float(row.get("r2_test_mean", 999.0)) < 1.0
        and float(row.get("r2_train_min", -999.0)) > FIVE_MODEL_MIN_R2
        and float(row.get("r2_train_min", 999.0)) < 1.0
        and float(row.get("r2_test_min", -999.0)) > FIVE_MODEL_MIN_R2
        and float(row.get("r2_test_min", 999.0)) < 1.0
        and float(row.get("r2_validation_min", -999.0)) > FIVE_MODEL_VALIDATION_MIN_R2
        and float(row.get("r2_experimental_min", -999.0)) > FIVE_MODEL_EXPERIMENTAL_MIN_R2
    )

def _metric_wide(per_seed_metrics: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return per_seed_metrics.pivot_table(
        index="split_seed",
        columns=["candidate_id", "split_role"],
        values=value_column,
        aggfunc="first",
    )

def _safe_fraction(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0.0, pd.NA)
    return pd.to_numeric(numerator, errors="coerce") / denominator

def summarize_paired_transfer_10seed(
    per_seed_metrics: pd.DataFrame,
    full_candidate_id: str,
    no_transfer_candidate_id: str,
) -> pd.Series:
    required = {"candidate_id", "split_seed", "split_role", "r2", "rmse", "mae"}
    missing = required - set(per_seed_metrics.columns)
    if missing:
        raise ValueError(f"Paired transfer metrics missing required columns: {sorted(missing)}")
    pair = per_seed_metrics[
        per_seed_metrics["candidate_id"].astype(str).isin([str(full_candidate_id), str(no_transfer_candidate_id)])
    ].copy()
    seed_count = int(pair["split_seed"].nunique())
    r2 = _metric_wide(pair, "r2")
    rmse = _metric_wide(pair, "rmse")
    mae = _metric_wide(pair, "mae")
    rows = {
        "full_candidate_id": str(full_candidate_id),
        "no_transfer_candidate_id": str(no_transfer_candidate_id),
        "seed_count": seed_count,
        "split_seeds": ",".join(str(int(seed)) for seed in sorted(pair["split_seed"].unique().tolist())),
    }
    required_columns = [
        (candidate_id, role)
        for candidate_id in (str(full_candidate_id), str(no_transfer_candidate_id))
        for role in R2_ROLES
    ]
    for table_name, table in {"r2": r2, "rmse": rmse, "mae": mae}.items():
        missing_columns = [column for column in required_columns if column not in table.columns]
        if missing_columns:
            raise ValueError(f"Paired transfer {table_name} metrics missing columns: {missing_columns}")

    role_passes = []
    for role in R2_ROLES:
        full_r2 = pd.to_numeric(r2[(str(full_candidate_id), role)], errors="coerce")
        no_r2 = pd.to_numeric(r2[(str(no_transfer_candidate_id), role)], errors="coerce")
        delta_r2 = full_r2 - no_r2
        full_rmse = pd.to_numeric(rmse[(str(full_candidate_id), role)], errors="coerce")
        no_rmse = pd.to_numeric(rmse[(str(no_transfer_candidate_id), role)], errors="coerce")
        full_mae = pd.to_numeric(mae[(str(full_candidate_id), role)], errors="coerce")
        no_mae = pd.to_numeric(mae[(str(no_transfer_candidate_id), role)], errors="coerce")
        rmse_reduction = _safe_fraction(no_rmse - full_rmse, no_rmse)
        mae_reduction = _safe_fraction(no_mae - full_mae, no_mae)

        rows[f"{role}_delta_r2_mean"] = float(delta_r2.mean())
        rows[f"{role}_delta_r2_min"] = float(delta_r2.min())
        rows[f"{role}_delta_r2_positive_seed_count"] = int(delta_r2.gt(0.0).sum())
        rows[f"{role}_rmse_reduction_fraction_mean"] = float(rmse_reduction.mean())
        rows[f"{role}_rmse_reduction_positive_seed_count"] = int(rmse_reduction.gt(0.0).sum())
        rows[f"{role}_mae_reduction_fraction_mean"] = float(mae_reduction.mean())
        rows[f"{role}_mae_reduction_positive_seed_count"] = int(mae_reduction.gt(0.0).sum())

        if role == "test":
            role_passes.append(
                bool(
                    rows[f"{role}_delta_r2_mean"] >= PAIRED_TRANSFER_TEST_MIN_DELTA_R2
                    and rows[f"{role}_rmse_reduction_fraction_mean"]
                    >= PAIRED_TRANSFER_MIN_ERROR_REDUCTION_FRACTION
                    and rows[f"{role}_mae_reduction_fraction_mean"]
                    >= PAIRED_TRANSFER_MIN_ERROR_REDUCTION_FRACTION
                    and rows[f"{role}_delta_r2_positive_seed_count"] == seed_count
                    and rows[f"{role}_rmse_reduction_positive_seed_count"] == seed_count
                    and rows[f"{role}_mae_reduction_positive_seed_count"] == seed_count
                )
            )
        elif role in {"validation", "experimental"}:
            role_passes.append(
                bool(
                    rows[f"{role}_delta_r2_mean"] > 0.0
                    and rows[f"{role}_delta_r2_positive_seed_count"] == seed_count
                )
            )

    rows["passes_paired_transfer_10seed"] = bool(seed_count == 10 and all(role_passes))
    return pd.Series(rows)

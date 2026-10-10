from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DELTA_R2 = 0.05
MIN_ERROR_REDUCTION_FRACTION = 0.05

def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, random_state: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1 or int(n_bootstrap) <= 0:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(int(random_state))
    sample_means = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(values, size=len(values), replace=True)
        sample_means.append(float(np.mean(sample)))
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return float(low), float(high)

def _metric_by_step(metrics: pd.DataFrame, split_role: str, value_column: str) -> pd.DataFrame:
    role_metrics = metrics[metrics["split_role"].astype(str).eq(str(split_role))].copy()
    return role_metrics.pivot_table(
        index="split_seed",
        columns="ablation_step",
        values=value_column,
        aggfunc="mean",
    )

def _safe_reduction_fraction(baseline: pd.Series, full: pd.Series) -> pd.Series:
    denominator = baseline.replace(0.0, np.nan)
    return (baseline - full) / denominator

def shuffle_transfer_matrix(frame: pd.DataFrame, feature_columns: list[str], split_seed: int) -> pd.DataFrame:
    shuffled = frame.copy()
    rng = np.random.default_rng(int(split_seed) + 20260616)
    transfer_columns = [
        column
        for column in feature_columns
        if any(
            str(column).startswith(prefix)
            for prefix in (
                "teacher_",
                "source_teacher_",
                "pathway_transfer_",
                "profile_transfer_",
                "local_profile_transfer_",
                "risk_profile_transfer_",
                "risk_adjusted_transfer_",
                "conditioned_transfer_",
                "disagreement_transfer_",
                "residualized_transfer_",
                "calibrated_transfer_",
            )
        )
    ]
    for column in transfer_columns:
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    return shuffled

def shuffle_transfer_matrix_repeated(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split_seed: int,
    repeats: int,
) -> list[pd.DataFrame]:
    return [
        shuffle_transfer_matrix(frame=frame, feature_columns=feature_columns, split_seed=int(split_seed) + offset)
        for offset in range(max(1, int(repeats)))
    ]

def summarize_transfer_evidence(
    metrics: pd.DataFrame,
    baseline_step: str,
    full_transfer_step: str,
    shuffled_step: str,
    external_roles: tuple[str, ...] = ("validation", "experimental"),
    formal_external_roles: tuple[str, ...] = ("experimental",),
    n_bootstrap: int = 2000,
    random_state: int = 20260616,
    uses_compound_group_split: bool = True,
) -> pd.DataFrame:
    required = {"split_seed", "ablation_step", "split_role", "r2", "rmse", "mae"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Transfer evidence metrics missing columns: {sorted(missing)}")
    rows = []
    for role_index, role in enumerate(external_roles):
        r2 = _metric_by_step(metrics, role, "r2")
        rmse = _metric_by_step(metrics, role, "rmse")
        mae = _metric_by_step(metrics, role, "mae")
        missing_steps = [
            step
            for step in (baseline_step, full_transfer_step, shuffled_step)
            if step not in r2.columns or step not in rmse.columns or step not in mae.columns
        ]
        if missing_steps:
            raise ValueError(f"Transfer evidence role {role} missing ablation steps: {sorted(set(missing_steps))}")

        delta_r2 = r2[full_transfer_step] - r2[baseline_step]
        rmse_reduction = _safe_reduction_fraction(rmse[baseline_step], rmse[full_transfer_step])
        mae_reduction = _safe_reduction_fraction(mae[baseline_step], mae[full_transfer_step])
        shuffled = r2[shuffled_step]
        shuffled_by_seed = shuffled.groupby(level=0).mean() if isinstance(shuffled.index, pd.MultiIndex) else shuffled
        full_vs_shuffled = r2[full_transfer_step] - shuffled_by_seed
        ci_low, ci_high = _bootstrap_ci(
            delta_r2.to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            random_state=int(random_state) + int(role_index),
        )

        passes_delta = float(delta_r2.mean()) >= MIN_DELTA_R2
        passes_rmse = float(rmse_reduction.mean()) >= MIN_ERROR_REDUCTION_FRACTION
        passes_mae = float(mae_reduction.mean()) >= MIN_ERROR_REDUCTION_FRACTION
        passes_ci = bool(np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0.0 or ci_high < 0.0))
        external_improved = float(delta_r2.mean()) > 0.0
        shuffled_decreased = float(full_vs_shuffled.mean()) > 0.0
        passes = bool(
            passes_delta
            and passes_rmse
            and passes_mae
            and passes_ci
            and bool(uses_compound_group_split)
            and external_improved
            and shuffled_decreased
        )

        rows.append(
            {
                "split_role": role,
                "baseline_step": baseline_step,
                "full_transfer_step": full_transfer_step,
                "shuffled_step": shuffled_step,
                "seed_count": int(delta_r2.dropna().shape[0]),
                "baseline_r2_mean": float(r2[baseline_step].mean()),
                "full_transfer_r2_mean": float(r2[full_transfer_step].mean()),
                "delta_r2_mean": float(delta_r2.mean()),
                "delta_r2_min": float(delta_r2.min()),
                "delta_r2_ci_low": ci_low,
                "delta_r2_ci_high": ci_high,
                "rmse_reduction_fraction_mean": float(rmse_reduction.mean()),
                "mae_reduction_fraction_mean": float(mae_reduction.mean()),
                "shuffled_r2_mean": float(shuffled_by_seed.mean()),
                "shuffled_r2_min": float(shuffled_by_seed.min()),
                "full_vs_shuffled_delta_r2_mean": float(full_vs_shuffled.mean()),
                "full_vs_shuffled_delta_r2_min": float(full_vs_shuffled.min()),
                "full_vs_shuffled_positive_seeds": int((full_vs_shuffled > 0.0).sum()),
                "uses_compound_group_split": bool(uses_compound_group_split),
                "is_formal_transfer_evidence_role": bool(role in formal_external_roles),
                "passes_delta_r2": bool(passes_delta),
                "passes_rmse_reduction": bool(passes_rmse),
                "passes_mae_reduction": bool(passes_mae),
                "passes_delta_r2_ci": bool(passes_ci),
                "passes_external_validation_improved": bool(external_improved),
                "passes_shuffled_transfer_negative_control": bool(shuffled_decreased),
                "passes_transfer_evidence": passes,
            }
        )
    summary = pd.DataFrame(rows)
    summary.attrs["formal_transfer_evidence_passes"] = bool(
        not summary.empty
        and summary[summary["is_formal_transfer_evidence_role"]]["passes_transfer_evidence"].all()
    )
    return summary

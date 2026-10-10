from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

BASE_LAYER_CANDIDATE_IDS = (
    "base_xgboost_raw_default",
    "base_catboost_raw_default",
    "base_extratrees_raw_default",
    "base_randomforest_raw_default",
    "base_svr_raw_default",
)

OPTIMIZED_EXTRATREES_CANDIDATE_IDS = (
    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00",
    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a100_k1_delta00",
    "extratrees_500_domain_rdkit_f65_l2_atom_pair_a30_k3_p2_time48_rdk_a35_k3_p4_seed7_desc_time48_domain_only_a70_k1_delta00",
)
OPTIMIZED_RANDOMFOREST_CANDIDATE_IDS = (
    "rf_500_leaf1_desc48_domain_a100",
    "rf_500_leaf1_time_a20_desc48_domain_a100",
    "rf_500_leaf2_time_a20_desc48_domain_a100",
)
OPTIMIZED_CATBOOST_CANDIDATE_IDS = (
    "cat_d5_it220_all_time_a25_desc48_domain_a100",
    "cat_d5_it300_domain_rdkit_time_a25_desc48_domain_a100_huber_cal",
    "cat_d5_it300_domain_rdkit_time_a25_desc48_domain_a100_ridge_context_cal",
    "cat_d5_it300_rdkit_desc_time_a25_desc48_domain_a100_huber_cal",
)
OPTIMIZED_SVR_CANDIDATE_IDS = (
    "svr_c32_eps003_maccs_atom_time_transfer_local_a30_desc48_domain_a090",
    "svr_c32_eps003_maccs_atom_time_transfer_local_a30_desc48_domain_a100",
)
OPTIMIZED_XGBOOST_CANDIDATE_IDS = (
    "xgb_d4_lr005_time_a20_desc48_domain_a090",
    "xgb_d4_lr005_time_a20_desc48_domain_a100",
    "xgb_d4_lr005_time_a40_desc48_domain_a100",
    "xgb_d4_lr005_local_a20_k1_desc48_domain_a100",
    "xgb_d4_lr005_local_a20_k3_desc48_domain_a100",
)

OPTIMIZED_LAYER_CANDIDATE_IDS = (
    *OPTIMIZED_EXTRATREES_CANDIDATE_IDS,
    *OPTIMIZED_RANDOMFOREST_CANDIDATE_IDS,
    *OPTIMIZED_CATBOOST_CANDIDATE_IDS,
    *OPTIMIZED_SVR_CANDIDATE_IDS,
    *OPTIMIZED_XGBOOST_CANDIDATE_IDS,
)

STACKING_LAYER_CANDIDATE_IDS = (
    "stack_et_svr_ridge",
    "stack_et_svr_ridge_time_a20_k3",
)

@dataclass(frozen=True)
class LayerReport:
    candidate_count: int
    violations: list[str]

def model_layer_pools() -> dict[str, tuple[str, ...]]:
    return {
        "base": BASE_LAYER_CANDIDATE_IDS,
        "optimized": OPTIMIZED_LAYER_CANDIDATE_IDS,
        "best": (*OPTIMIZED_LAYER_CANDIDATE_IDS, *STACKING_LAYER_CANDIDATE_IDS),
    }

def fixed_base_model_pools() -> dict[str, tuple[str, ...]]:
    return {
        "Base__xgboost_raw": ("base_xgboost_raw_default",),
        "Base__catboost_raw": ("base_catboost_raw_default",),
        "Base__extratrees_raw": ("base_extratrees_raw_default",),
        "Base__randomforest_raw": ("base_randomforest_raw_default",),
        "Base__svr_raw": ("base_svr_raw_default",),
    }

def cvfirst_report_pools() -> dict[str, tuple[str, ...]]:
    pools = dict(fixed_base_model_pools())
    pools.update(
        {
            "Base__cv_selected_raw_5family": BASE_LAYER_CANDIDATE_IDS,
            "Optimized__extratrees_transfer": OPTIMIZED_EXTRATREES_CANDIDATE_IDS,
            "Optimized__randomforest_transfer": OPTIMIZED_RANDOMFOREST_CANDIDATE_IDS,
            "Optimized__catboost_transfer": OPTIMIZED_CATBOOST_CANDIDATE_IDS,
            "Optimized__svr_transfer": OPTIMIZED_SVR_CANDIDATE_IDS,
            "Optimized__xgboost_transfer": OPTIMIZED_XGBOOST_CANDIDATE_IDS,
            "Optimized__cv_selected_nonstack": OPTIMIZED_LAYER_CANDIDATE_IDS,
            "Best__cv_selected_with_stack": (*OPTIMIZED_LAYER_CANDIDATE_IDS, *STACKING_LAYER_CANDIDATE_IDS),
        }
    )
    return pools

def _violations_for_pool(candidates: Iterable[object], pool: tuple[str, ...]) -> list[str]:
    by_id = {getattr(candidate, "candidate_id"): candidate for candidate in candidates}
    violations: list[str] = []
    for candidate_id in pool:
        spec = by_id.get(candidate_id)
        if spec is None:
            violations.append(f"missing:{candidate_id}")
            continue
        if candidate_id in STACKING_LAYER_CANDIDATE_IDS:
            continue
        if getattr(spec, "family", None) == "stacking":
            violations.append(f"stacking_in_nonstack_pool:{candidate_id}")
        if candidate_id in BASE_LAYER_CANDIDATE_IDS and any(
            [
                getattr(spec, "local_alpha", 0.0) > 0.0,
                getattr(spec, "time_local_alpha", 0.0) > 0.0,
                getattr(spec, "descriptor_local_alpha", 0.0) > 0.0,
                getattr(spec, "residual_baseline_column", None) is not None,
                getattr(spec, "calibration_method", "none") != "none",
            ]
        ):
            violations.append(f"not_raw_base:{candidate_id}")
    return violations

def validate_model_layer_pools(candidates: Iterable[object]) -> dict[str, dict[str, object]]:
    pools = model_layer_pools()
    return {
        name: {
            "candidate_count": len(pool),
            "violations": _violations_for_pool(candidates, pool),
        }
        for name, pool in pools.items()
    }

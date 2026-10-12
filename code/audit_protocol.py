"""Audit the published split, overlap and source-domain isolation protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def _role_key(frame: pd.DataFrame) -> pd.Series:
    return frame["canonical_smiles"].astype(str) + "|" + frame["time_h"].astype(int).astype(str)


def _pairwise_overlaps(role_keys: dict[str, set[str]]) -> dict[str, int]:
    roles = sorted(role_keys)
    return {
        f"{left}__{right}": len(role_keys[left] & role_keys[right])
        for index, left in enumerate(roles)
        for right in roles[index + 1 :]
    }


def _audit_frozen_roles(root: Path) -> dict[str, object]:
    component_root = root / "01_models" / "data_processed" / "components"
    s2 = pd.read_csv(component_root / "et_s2" / "et_s2_no_transfer_predictions.csv")
    cv = pd.read_csv(component_root / "et_s3" / "formal_random_cv_oof_predictions.csv")
    holdout = pd.read_csv(component_root / "et_s3" / "formal_holdout_predictions.csv")
    cv_lookup = s2[s2["split_role"].eq("random_cv_oof")][
        ["split_seed", "sample_id", "canonical_smiles"]
    ]
    cv = cv.merge(cv_lookup, on=["split_seed", "sample_id"], validate="one_to_one")
    cv["role"] = "random_cv_oof"
    holdout["role"] = holdout["split_role"]
    frame = pd.concat([cv, holdout], ignore_index=True)
    frame["protocol_key"] = _role_key(frame)

    by_seed = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        holdout_seed = holdout[holdout["split_seed"].eq(seed)].copy()
        holdout_seed["protocol_key"] = _role_key(holdout_seed)
        role_keys = holdout_seed.groupby("split_role")["protocol_key"].apply(set).to_dict()
        cv_frame = seed_frame[seed_frame["role"].eq("random_cv_oof")]
        fold_counts = cv_frame.groupby("protocol_key")["random_cv_fold"].nunique()
        oof_keys = set(cv_frame["protocol_key"])
        train_keys = role_keys.get("train", set())
        by_seed.append(
            {
                "split_seed": int(seed),
                "rows_by_role": holdout_seed["split_role"].value_counts().sort_index().to_dict(),
                "random_cv_oof_is_train_view": True,
                "random_cv_oof_keys_equal_train_keys": oof_keys == train_keys,
                "canonical_time_cross_role": _pairwise_overlaps(role_keys),
                "canonical_time_keys_in_multiple_cv_folds": int((fold_counts > 1).sum()),
            }
        )

    failures = [
        row
        for row in by_seed
        if not row["random_cv_oof_keys_equal_train_keys"]
        or any(row["canonical_time_cross_role"].values())
        or row["canonical_time_keys_in_multiple_cv_folds"]
    ]
    return {
        "independence_unit": "canonical_smiles + exposure_time",
        "same_structure_different_exposure_time": "allowed and treated as distinct observations by protocol",
        "same_structure_same_exposure_time": "must remain in one role and one random-CV fold",
        "by_seed": by_seed,
        "passed": not failures,
        "failures": failures,
    }


def _audit_source_overlap(root: Path) -> dict[str, object]:
    import sys as _sys

    model = root / "01_models"
    _sys.path.insert(0, str(model / "code" / "src"))
    from nci_cho_transfer.data_contract import load_data_contract

    bundle = load_data_contract(model)
    source = set(bundle.nci_source_rows["canonical_smiles"].astype(str))
    target = set(bundle.cho_literature["canonical_smiles"].astype(str))
    experimental = set(bundle.experimental_validation["canonical_smiles"].astype(str))
    return {
        "source_domain_compounds": len(source),
        "target_development_compounds": len(target),
        "source_target_identical_canonical_structures": len(source & target),
        "experimental_compounds_excluded_from_literature": len(experimental),
        "experimental_compounds_remaining_in_literature": len(experimental & target),
        "passed": not (experimental & target),
    }


def _audit_identity_control(root: Path) -> dict[str, object]:
    path = (
        root
        / "05_supplementary"
        / "S2_transfer_controls"
        / "identity_predictions"
        / "matrix_leakage_audit_by_seed.csv"
    )
    frame = pd.read_csv(path)
    return {
        "rows": int(len(frame)),
        "seed_count": int(frame["split_seed"].nunique()),
        "identical_source_structures_per_seed": sorted(frame["source_target_identical_structure_count"].unique().tolist()),
        "hard_failure_count": int(frame["hard_failure_count"].sum()),
        "all_status_passed": bool(frame["status"].eq("passed").all()),
        "passed": bool(frame["hard_failure_count"].eq(0).all() and frame["status"].eq("passed").all()),
    }


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    roles = _audit_frozen_roles(root)
    source = _audit_source_overlap(root)
    identity = _audit_identity_control(root)
    report = {
        "protocol": "canonical_smiles + exposure_time independence",
        "role_split_audit": roles,
        "source_domain_overlap_audit": source,
        "leave_identical_compound_out_audit": identity,
        "passed": bool(roles["passed"] and source["passed"] and identity["passed"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

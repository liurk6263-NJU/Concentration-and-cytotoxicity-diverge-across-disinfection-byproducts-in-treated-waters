from __future__ import annotations

from .features import FORBIDDEN_FEATURE_COLUMNS, ModelMatrix
from .splits import FrozenSplit

def audit_split_and_features(split: FrozenSplit, matrix: ModelMatrix) -> dict:
    role_frames = {
        "train": split.train,
        "test": split.test,
        "validation": split.validation,
        "experimental": split.experimental,
    }
    failures = []
    roles = list(role_frames)
    for i, left in enumerate(roles):
        for right in roles[i + 1 :]:
            left_keys = set(role_frames[left]["canonical_time_key"])
            right_keys = set(role_frames[right]["canonical_time_key"])
            overlap = sorted(left_keys & right_keys)
            if overlap:
                failures.append({"type": "canonical_time_overlap", "left": left, "right": right, "count": len(overlap)})
            if "sample_id" in role_frames[left].columns and "sample_id" in role_frames[right].columns:
                sample_overlap = sorted(set(role_frames[left]["sample_id"]) & set(role_frames[right]["sample_id"]))
                if sample_overlap:
                    failures.append({"type": "sample_id_overlap", "left": left, "right": right, "count": len(sample_overlap)})

    validation_dois = set(split.validation["doi"])
    if validation_dois & set(split.train["doi"]):
        failures.append({"type": "validation_doi_in_train", "count": len(validation_dois & set(split.train["doi"]))})
    if validation_dois & set(split.test["doi"]):
        failures.append({"type": "validation_doi_in_test", "count": len(validation_dois & set(split.test["doi"]))})

    leaked_features = sorted(FORBIDDEN_FEATURE_COLUMNS & set(matrix.feature_columns))
    if leaked_features:
        failures.append({"type": "label_equivalent_feature", "columns": leaked_features})

    return {
        "status": "passed" if not failures else "failed",
        "hard_failure_count": len(failures),
        "failures": failures,
        "feature_count": len(matrix.feature_columns),
    }

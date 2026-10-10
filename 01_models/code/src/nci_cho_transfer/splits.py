from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_contract import DataContractBundle

VALIDATION_SEED = 20260705
SPLIT_SEED = 20260605
VALIDATION_ROWS = 37
TEST_ROWS = 37

@dataclass(frozen=True)
class FrozenSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame
    experimental: pd.DataFrame
    manifest: dict
    manifest_hash: str

def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _time_vector(frame: pd.DataFrame, times: list[int]) -> np.ndarray:
    counts = frame["time_h"].value_counts().reindex(times, fill_value=0)
    return counts.to_numpy(dtype=float)

def _select_validation_dois(frame: pd.DataFrame) -> list[str]:
    times = sorted(int(v) for v in frame["time_h"].unique())
    target = _time_vector(frame, times) / len(frame) * VALIDATION_ROWS
    groups = []
    for doi, group in frame.groupby("doi", sort=True):
        if len(group) <= VALIDATION_ROWS:
            groups.append((str(doi), int(len(group)), _time_vector(group, times).astype(int)))

    states: dict[int, list[tuple[tuple[str, ...], np.ndarray]]] = {0: [(tuple(), np.zeros(len(times), dtype=int))]}
    for doi, rows, vector in sorted(groups, key=lambda item: _stable_hash(item[0])):
        updated = {total: list(items) for total, items in states.items()}
        for total, items in states.items():
            new_total = total + rows
            if new_total > VALIDATION_ROWS + 1:
                continue
            for dois, old_vector in items:
                new_dois = tuple(sorted(dois + (doi,)))
                new_vector = old_vector + vector
                updated.setdefault(new_total, []).append((new_dois, new_vector))
        for total, items in updated.items():
            updated[total] = sorted(
                items,
                key=lambda item: (
                    abs(total - VALIDATION_ROWS),
                    float(np.abs(item[1] - target).sum()),
                    len(item[0]),
                    _stable_hash("||".join(item[0])),
                ),
            )[:512]
        states = updated

    candidates = []
    for total in (VALIDATION_ROWS, VALIDATION_ROWS - 1, VALIDATION_ROWS + 1):
        for dois, vector in states.get(total, []):
            candidates.append(
                (
                    abs(total - VALIDATION_ROWS),
                    float(np.abs(vector - target).sum()),
                    len(dois),
                    _stable_hash("||".join(dois)),
                    list(dois),
                )
            )
    if not candidates:
        raise ValueError("No DOI combination can produce a validation holdout near 10%")
    return sorted(candidates)[0][4]

def _manifest_hash(manifest: dict) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _train_test_stratify_labels(frame: pd.DataFrame) -> pd.Series | None:
    if "y" in frame.columns:
        labels = pd.qcut(pd.to_numeric(frame["y"], errors="coerce"), q=5, duplicates="drop").astype(str)
        if labels.nunique() >= 2 and labels.value_counts().min() >= 2:
            return labels
    labels = frame["time_h"].astype(int).astype(str)
    if labels.nunique() >= 2 and labels.value_counts().min() >= 2:
        return labels
    return None

def build_frozen_split(bundle: DataContractBundle, split_seed: int = SPLIT_SEED) -> FrozenSplit:
    frame = bundle.cho_literature.copy()
    validation_dois = _select_validation_dois(frame)
    validation = frame[frame["doi"].isin(validation_dois)].copy()
    train_test_pool = frame[~frame["doi"].isin(validation_dois)].copy()

    stratify = _train_test_stratify_labels(train_test_pool)
    train, test = train_test_split(
        train_test_pool,
        train_size=len(train_test_pool) - TEST_ROWS,
        test_size=TEST_ROWS,
        random_state=int(split_seed),
        stratify=stratify,
    )
    role_frames = {
        "train": train.copy(),
        "test": test.copy(),
        "validation": validation.copy(),
        "experimental": bundle.experimental_validation.copy(),
    }
    for role, role_frame in role_frames.items():
        role_frame["split_role"] = role
        role_frames[role] = role_frame.sort_values("canonical_time_key").reset_index(drop=True)

    manifest = {
        "protocol": "20260604_doi_holdout_8_1_1_v1",
        "validation_seed": VALIDATION_SEED,
        "split_seed": int(split_seed),
        "validation_dois": validation_dois,
        "counts": {role: int(len(role_frame)) for role, role_frame in role_frames.items()},
        "time_counts": {
            role: {str(k): int(v) for k, v in role_frame["time_h"].value_counts().sort_index().items()}
            for role, role_frame in role_frames.items()
        },
        "validation_sample_ids": role_frames["validation"]["sample_id"].tolist(),
        "test_sample_ids": role_frames["test"]["sample_id"].tolist(),
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    return FrozenSplit(
        train=role_frames["train"],
        test=role_frames["test"],
        validation=role_frames["validation"],
        experimental=role_frames["experimental"],
        manifest=manifest,
        manifest_hash=manifest["manifest_hash"],
    )

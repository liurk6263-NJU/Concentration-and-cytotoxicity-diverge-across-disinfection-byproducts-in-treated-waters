import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
TANIMOTO_CUTOFF = 0.333

def _fingerprints(smiles: pd.Series, generator, cache: dict[str, object]) -> list[object]:
    out = []
    for value in smiles.astype(str):
        if value not in cache:
            mol = Chem.MolFromSmiles(value)
            cache[value] = generator.GetFingerprint(mol) if mol is not None else None
        out.append(cache[value])
    return out

def _max_tanimoto(smiles: pd.Series, references: pd.Series, generator, cache: dict[str, object]) -> np.ndarray:
    query_fps = _fingerprints(smiles, generator, cache)
    reference_fps = [fp for fp in _fingerprints(references, generator, cache) if fp is not None]
    if not reference_fps:
        return np.full(len(smiles), np.nan)
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)) if fp is not None else np.nan for fp in query_fps],
        dtype=float,
    )

def _oos_rows_with_similarity(raw: pd.DataFrame, generator, cache: dict[str, object]) -> pd.DataFrame:
    selected = raw[raw["split_role"].isin(["random_cv_oof", "test", "validation", "experimental"])].copy()
    parts = []
    for split_seed, seed_rows in selected.groupby("split_seed", sort=True):
        train_smiles = raw.loc[
            raw["split_seed"].eq(split_seed) & raw["split_role"].eq("train"), "canonical_smiles"
        ].drop_duplicates()
        for role, role_rows in seed_rows.groupby("split_role", sort=False):
            if role != "random_cv_oof":
                part = role_rows.copy()
                part["max_tanimoto"] = _max_tanimoto(part["canonical_smiles"], train_smiles, generator, cache)
                parts.append(part)
                continue
            for fold, query in role_rows.groupby("random_cv_fold", sort=True):
                fit_smiles = role_rows.loc[~role_rows["random_cv_fold"].eq(fold), "canonical_smiles"].drop_duplicates()
                part = query.copy()
                part["max_tanimoto"] = _max_tanimoto(part["canonical_smiles"], fit_smiles, generator, cache)
                parts.append(part)
    return pd.concat(parts, ignore_index=True)

def _aggregate_labeled_oos(oos: pd.DataFrame) -> pd.DataFrame:
    frame = (
        oos.groupby("sample_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            canonical_smiles=("canonical_smiles", "first"),
            pLC50_mean=("blend_y_pred", "mean"),
            pLC50_sd=("blend_y_pred", lambda x: x.std(ddof=1)),
            S2_mean=("S2_no_transfer", "mean"),
            max_tanimoto=("max_tanimoto", "median"),
            max_tanimoto_min=("max_tanimoto", "min"),
            evaluation_roles=("split_role", lambda x: ",".join(sorted(set(x)))),
            seed_n=("blend_y_pred", "size"),
        )
    )
    frame["evaluation_set"] = np.where(
        frame["evaluation_roles"].eq("validation"),
        "validation",
        np.where(frame["evaluation_roles"].eq("experimental"), "experimental", "development"),
    )
    frame["abs_error"] = (frame["y_true"] - frame["pLC50_mean"]).abs()
    frame["S3_minus_S2_abs_error_gain"] = (
        (frame["y_true"] - frame["S2_mean"]).abs() - frame["abs_error"]
    )
    if not frame["seed_n"].eq(10).all():
        raise RuntimeError("Every labeled sample must have exactly 10 out-of-sample predictions")
    return frame

def _tanimoto_band(value: float) -> str:
    if value >= 0.50:
        return "high_ge_0.50"
    if value >= TANIMOTO_CUTOFF:
        return "within_0.333_0.50"
    if value >= TANIMOTO_CUTOFF - 0.10:
        return "borderline_0.233_0.333"
    return "outside_lt_0.233"

def _sd_band(value: float, median: float, q75: float) -> str:
    if value <= median:
        return "low"
    if value <= q75:
        return "moderate"
    return "high"
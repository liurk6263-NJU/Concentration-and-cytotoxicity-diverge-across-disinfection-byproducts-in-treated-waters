from __future__ import annotations

import numpy as np
import pandas as pd

CELL_ID_ALIASES = {"MDA-MB-231/ATCC": "MDA-MB-231"}

def normalize_cell_id(value: object) -> str:
    cell_id = str(value)
    return CELL_ID_ALIASES.get(cell_id, cell_id)

def development_sd_cutoff(
    rows: pd.DataFrame,
    set_column: str = "evaluation_set",
    sd_column: str = "pLC50_sd",
) -> float:
    development = pd.to_numeric(
        rows.loc[rows[set_column].astype(str).eq("development"), sd_column],
        errors="coerce",
    ).dropna()
    if development.empty:
        raise ValueError("No development-set SD values are available")
    return float(development.median())

def composite_ad_mask(
    rows: pd.DataFrame,
    tanimoto_cutoff: float,
    sd_cutoff: float,
    tanimoto_column: str = "max_tanimoto",
    sd_column: str = "pLC50_sd",
) -> pd.Series:
    tanimoto = pd.to_numeric(rows[tanimoto_column], errors="coerce")
    sd = pd.to_numeric(rows[sd_column], errors="coerce")
    return tanimoto.ge(float(tanimoto_cutoff)) & sd.le(float(sd_cutoff))

def build_expected_error_lookup(calibration: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    required = {"tanimoto_band", "sd_band", "abs_error"}
    missing = required - set(calibration.columns)
    if missing:
        raise ValueError(f"Expected-error calibration is missing columns: {sorted(missing)}")
    grouped = calibration.groupby(["tanimoto_band", "sd_band"], dropna=False)["abs_error"]
    lookup = grouped.agg(n="size", expected_mae="mean", rmse=lambda x: float(np.sqrt(np.mean(np.square(x))))).reset_index()
    lookup["error_q90"] = grouped.quantile(0.90).to_numpy(dtype=float)
    lookup["supported_for_ad"] = lookup["n"].ge(int(min_rows))
    return lookup

def attach_cell_similarity(
    source_rows: pd.DataFrame,
    similarity: pd.DataFrame,
    similarity_column: str = "cell_similarity",
) -> pd.DataFrame:
    left = source_rows.copy()
    right = similarity[["cell_id", similarity_column]].copy()
    left["cell_id_normalized"] = left["cell_id"].map(normalize_cell_id)
    right["cell_id_normalized"] = right["cell_id"].map(normalize_cell_id)
    right = right.drop(columns="cell_id").drop_duplicates("cell_id_normalized")
    merged = left.merge(right, on="cell_id_normalized", how="left")
    if merged[similarity_column].isna().any():
        missing = sorted(merged.loc[merged[similarity_column].isna(), "cell_id"].astype(str).unique())
        raise ValueError(f"Cell-state similarity is missing for source cells: {missing}")
    return merged

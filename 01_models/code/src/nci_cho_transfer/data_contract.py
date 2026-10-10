from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHO_WORKBOOK = "DBPs_CHO_LC50_DBPs_20260604.xlsx"
NCI_SOURCE_CACHE = Path("data_processed/cache/nci_source_rows.parquet")

@dataclass(frozen=True)
class DataContractBundle:
    root_dir: Path
    cho_literature: pd.DataFrame
    experimental_validation: pd.DataFrame
    nci_source_rows: pd.DataFrame
    audit: dict

def _canonical_time_key(frame: pd.DataFrame) -> pd.Series:
    return frame["canonical_smiles"].astype(str) + "|" + frame["time_h"].astype(int).astype(str)

def _standardize_literature(raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(raw.columns)
    frame = pd.DataFrame(
        {
            "source_row_number": np.arange(1, len(raw) + 1),
            "name": raw[cols[1]].astype(str),
            "cas": raw[cols[2]].astype(str),
            "raw_smiles": raw[cols[3]].astype(str),
            "canonical_smiles": raw[cols[4]].astype(str),
            "inchi_key": raw[cols[5]].astype(str),
            "y": pd.to_numeric(raw[cols[6]], errors="coerce"),
            "actual_lc50_m": pd.to_numeric(raw[cols[7]], errors="coerce"),
            "cell_line": raw[cols[8]].astype(str),
            "time_h": pd.to_numeric(raw[cols[9]], errors="coerce").astype(int),
            "doi": raw[cols[10]].astype(str),
            "title_en": raw[cols[11]].astype(str),
            "title_cn": raw[cols[12]].astype(str),
        }
    )
    frame["source_domain"] = "cho_k1_literature"
    frame["canonical_time_key"] = _canonical_time_key(frame)
    if frame["y"].isna().any() or frame["time_h"].isna().any():
        raise ValueError("CHO literature contains invalid target or time values")
    return frame

def _standardize_experimental(raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(raw.columns)
    frame = pd.DataFrame(
        {
            "source_row_number": np.arange(1, len(raw) + 1),
            "name": raw[cols[0]].astype(str),
            "cas": raw[cols[1]].astype(str),
            "raw_smiles": raw[cols[2]].astype(str),
            "canonical_smiles": raw[cols[3]].astype(str),
            "inchi_key": raw[cols[4]].astype(str),
            "y": pd.to_numeric(raw[cols[5]], errors="coerce"),
            "actual_lc50_m": pd.to_numeric(raw[cols[6]], errors="coerce"),
            "time_h": pd.to_numeric(raw[cols[8]], errors="coerce").astype(int),
        }
    )
    frame["source_domain"] = "cho_k1_experimental"
    frame["doi"] = "EXPERIMENTAL_VALIDATION"
    frame["title_en"] = "experimental validation"
    frame["title_cn"] = "experimental validation"
    frame["canonical_time_key"] = _canonical_time_key(frame)
    frame["sample_id"] = [f"exp_{i:04d}" for i in range(1, len(frame) + 1)]
    if set(frame["time_h"]) != {48}:
        raise ValueError("Experimental validation rows must all use 48 h")
    return frame

def _aggregate_canonical_time(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = frame.sort_values(["canonical_time_key", "doi", "source_row_number"])
    for idx, (key, group) in enumerate(ordered.groupby("canonical_time_key", sort=True), start=1):
        first = group.iloc[0]
        rows.append(
            {
                "sample_id": f"cho_{idx:05d}",
                "source_domain": "cho_k1_literature",
                "compound_id": first["canonical_smiles"],
                "raw_smiles": first["raw_smiles"],
                "canonical_smiles": first["canonical_smiles"],
                "canonical_time_key": key,
                "name": first["name"],
                "cas": first["cas"],
                "time_h": int(first["time_h"]),
                "y": float(group["y"].mean()),
                "actual_lc50_m": float(group["actual_lc50_m"].mean()),
                "n_obs": int(len(group)),
                "raw_source_rows": ",".join(map(str, group["source_row_number"].tolist())),
                "doi": first["doi"],
                "title_en": first["title_en"],
                "title_cn": first["title_cn"],
            }
        )
    return pd.DataFrame(rows)

def _load_nci_source_rows(root_dir: Path) -> pd.DataFrame:
    cache_path = root_dir / NCI_SOURCE_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(f"NCI source cache not found: {cache_path}")
    frame = pd.read_parquet(cache_path).copy()
    required = {"canonical_smiles", "cell_id", "time_h", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"NCI source cache missing columns: {sorted(missing)}")
    return frame

def load_data_contract(root_dir: str | Path = ".") -> DataContractBundle:
    root = Path(root_dir).resolve()
    workbook = root / "data_raw" / CHO_WORKBOOK
    if not workbook.exists():
        raise FileNotFoundError(f"Required CHO workbook not found: {workbook}")

    literature_raw = pd.read_excel(workbook, sheet_name=0)
    experimental_raw = pd.read_excel(workbook, sheet_name=1)
    literature = _standardize_literature(literature_raw)
    experimental = _standardize_experimental(experimental_raw)

    experimental_compounds = set(experimental["canonical_smiles"])
    excluded = literature[literature["canonical_smiles"].isin(experimental_compounds)].copy()
    literature_pool = literature[~literature["canonical_smiles"].isin(experimental_compounds)].copy()
    literature_aggregated = _aggregate_canonical_time(literature_pool)

    audit = {
        "cho_workbook": CHO_WORKBOOK,
        "cho_literature_source_rows": int(len(literature)),
        "experimental_validation_rows": int(len(experimental)),
        "excluded_literature_rows_by_experimental_compound": int(len(excluded)),
        "cho_literature_rows_after_compound_exclusion": int(len(literature_pool)),
        "cho_literature_canonical_time_rows": int(len(literature_aggregated)),
        "experimental_compounds": sorted(experimental_compounds),
        "excluded_literature_source_rows": excluded["source_row_number"].astype(int).tolist(),
    }
    if len(literature_aggregated) != 365 or len(experimental) != 4:
        raise ValueError(f"Unexpected 20260604 contract counts: {audit}")

    return DataContractBundle(
        root_dir=root,
        cho_literature=literature_aggregated,
        experimental_validation=experimental,
        nci_source_rows=_load_nci_source_rows(root),
        audit=audit,
    )

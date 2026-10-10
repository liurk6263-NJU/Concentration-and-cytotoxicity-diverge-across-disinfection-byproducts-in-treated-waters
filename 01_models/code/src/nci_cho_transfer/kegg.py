from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RNA_WORKBOOK = "harmonized_gene_matrix_log2TPM1.xlsx"
KEGG_GMT = "c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt"
MIN_GENES_PER_PATHWAY = 5

@dataclass(frozen=True)
class PathwayBundle:
    pathway_activity: pd.DataFrame
    cho_weights: pd.DataFrame
    metadata: dict

def parse_gmt(path: Path) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = "kegg_" + parts[0]
            genes = [gene.upper() for gene in parts[2:] if gene]
            pathways[name] = genes
    return pathways

def _cell_id_from_source(source_column: str) -> str:
    if source_column.startswith("CHO:"):
        return source_column
    if ":" in source_column:
        return source_column.split(":", 1)[1]
    return source_column

def build_pathway_bundle(root_dir: str | Path = ".") -> PathwayBundle:
    root = Path(root_dir).resolve()
    rna = pd.read_excel(root / "data_raw" / RNA_WORKBOOK, sheet_name=0)
    pathways = parse_gmt(root / "data_raw" / KEGG_GMT)

    gene_symbols = rna["human_gene_symbol"].astype(str).str.upper()
    expr_cols = [
        col
        for col in rna.columns
        if col not in {"human_entrez_id", "human_gene_symbol", "hamster_geneid", "hamster_gene_symbol"}
    ]
    expr = rna[expr_cols].apply(pd.to_numeric, errors="coerce")

    activity_columns: dict[str, object] = {"source_column": expr_cols}
    pathway_gene_counts: dict[str, int] = {}
    for pathway, genes in sorted(pathways.items()):
        mask = gene_symbols.isin(set(genes))
        present = int(mask.sum())
        if present < MIN_GENES_PER_PATHWAY:
            continue
        pathway_gene_counts[pathway] = present
        activity_columns[pathway] = expr.loc[mask, :].mean(axis=0).to_numpy()
    activity = pd.DataFrame(activity_columns)

    nci = activity[~activity["source_column"].astype(str).str.startswith("CHO:")].copy()
    cho = activity[activity["source_column"].eq("CHO:consensus")]
    if cho.empty:
        raise ValueError("RNA matrix is missing CHO:consensus")

    feature_cols = [c for c in activity.columns if c != "source_column"]
    nci_matrix = nci[feature_cols].to_numpy(dtype=float)
    cho_vector = cho[feature_cols].iloc[0].to_numpy(dtype=float)
    nci_centered = nci_matrix - np.nanmean(nci_matrix, axis=1, keepdims=True)
    cho_centered = cho_vector - np.nanmean(cho_vector)
    denom = np.linalg.norm(nci_centered, axis=1) * np.linalg.norm(cho_centered)
    sims = np.divide(nci_centered @ cho_centered, denom, out=np.zeros(len(nci)), where=denom > 0)
    positive = np.clip(sims, 0, None)
    if float(positive.sum()) == 0.0:
        positive = np.ones_like(positive)
    weights = positive / positive.sum()
    cho_weights = pd.DataFrame(
        {
            "cell_id": [_cell_id_from_source(str(v)) for v in nci["source_column"]],
            "weight": weights,
        }
    )

    metadata = {
        "n_cells_total": int(len(activity)),
        "n_nci_cells": int(len(nci)),
        "n_cho_cells": int(activity["source_column"].astype(str).str.startswith("CHO:").sum()),
        "n_pathways": int(len(feature_cols)),
        "min_genes_per_pathway": MIN_GENES_PER_PATHWAY,
        "pathway_gene_counts": pathway_gene_counts,
        "cho_source": "CHO:consensus",
    }
    return PathwayBundle(pathway_activity=activity, cho_weights=cho_weights, metadata=metadata)

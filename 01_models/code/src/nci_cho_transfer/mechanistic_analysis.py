from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr

FACTOR_ORDER = ("struct", "cell", "time")
FORBIDDEN_LABEL_COLUMNS = {"y", "target_plc50", "actual_lc50_m", "pLC50", "LC50(M)"}
FACTOR_FEATURE_COLUMNS = [
    "factor_transfer_y_wmean",
    "factor_transfer_y_wstd",
    "factor_transfer_weight_max",
    "factor_transfer_effective_n",
    "factor_transfer_weight_entropy",
    "factor_transfer_top1_struct_sim",
]
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MAIN_STRUCTURE_CHANGE_TAGS = (
    "heavy_halogen_gain",
    "heavy_halogen_z_increase",
    "halogen_count_gain",
    "halogen_count_loss",
    "carboxyl_loss",
    "carboxyl_gain",
)
PREFERRED_INTERACTION_PATHWAYS = (
    "kegg_KEGG_APOPTOSIS",
    "kegg_KEGG_P53_SIGNALING_PATHWAY",
    "kegg_KEGG_BASE_EXCISION_REPAIR",
    "kegg_KEGG_OXIDATIVE_PHOSPHORYLATION",
    "kegg_KEGG_GLUTATHIONE_METABOLISM",
    "kegg_KEGG_CELL_CYCLE",
    "kegg_KEGG_DNA_REPLICATION",
    "kegg_KEGG_MAPK_SIGNALING_PATHWAY",
    "kegg_KEGG_METABOLISM_OF_XENOBIOTICS_BY_CYTOCHROME_P450",
)
STRUCTURE_PATHWAY_GROUP_COLUMNS = {
    "Heavy halogen": [
        "heavy_halogen_count",
        "domain_bromine_iodine_count",
        "domain_cl_br_i_count",
        "domain_halogen_mass",
        "domain_max_halogen_z",
    ],
    "Halogen count": ["halogen_count", "domain_halogen_atom_fraction", "domain_halogen_diversity"],
    "Aromatic halogen": ["domain_aromatic_halogen_count", "aromatic_atom_count"],
    "Carbonyl": ["fg_aldehyde", "fg_quinone", "fg_amide", "rdkit_NumHAcceptors"],
    "Carboxyl": ["fg_carboxylic_acid"],
    "Hydrophobicity": ["rdkit_MolLogP", "domain_logp_per_heavy_atom"],
    "Polarity": ["rdkit_TPSA", "domain_tpsa_per_mw", "rdkit_NumHDonors", "rdkit_NumHAcceptors"],
}
PATHWAY_LABEL_OVERRIDES = {
    "KEGG_APOPTOSIS": "Apoptosis",
    "KEGG_P53_SIGNALING_PATHWAY": "p53 signaling",
    "KEGG_BASE_EXCISION_REPAIR": "Base excision repair",
    "KEGG_OXIDATIVE_PHOSPHORYLATION": "OxPhos",
    "KEGG_GLUTATHIONE_METABOLISM": "Glutathione metabolism",
    "KEGG_CELL_CYCLE": "Cell cycle",
    "KEGG_DNA_REPLICATION": "DNA replication",
    "KEGG_MAPK_SIGNALING_PATHWAY": "MAPK signaling",
    "KEGG_METABOLISM_OF_XENOBIOTICS_BY_CYTOCHROME_P450": "Xenobiotic P450",
}
STRUCTURE_CHANGE_LABELS = {
    "heavy_halogen_gain": "Heavy halogen gain",
    "heavy_halogen_z_increase": "Heavy halogen Z increase",
    "halogen_count_gain": "Halogen count gain",
    "halogen_count_loss": "Halogen count loss",
    "carboxyl_loss": "Carboxyl loss",
    "carboxyl_gain": "Carboxyl gain",
}

def _pathway_label(pathway: str) -> str:
    label = str(pathway)
    label = label.removeprefix("pathway_transfer_")
    label = label.removeprefix("kegg_")
    if label in PATHWAY_LABEL_OVERRIDES:
        return PATHWAY_LABEL_OVERRIDES[label]
    label = label.removeprefix("KEGG_")
    return label.replace("_", " ").title().replace("P53", "p53")

def pathway_display_label(pathway: str) -> str:
    return _pathway_label(pathway)

def build_nci_compound_map(response_matrix: pd.DataFrame) -> pd.DataFrame:
    smiles = response_matrix["canonical_smiles"].astype(str).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame(
        {
            "compound_id": [f"C{idx:02d}" for idx in range(1, len(smiles) + 1)],
            "canonical_smiles": smiles,
        }
    )

def filter_main_structure_change_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    order = {tag: idx for idx, tag in enumerate(MAIN_STRUCTURE_CHANGE_TAGS)}
    filtered = summary[summary["structure_change"].isin(order)].copy()
    filtered["_main_order"] = filtered["structure_change"].map(order)
    return filtered.sort_values("_main_order").drop(columns=["_main_order"]).reset_index(drop=True)

def build_pathway_concordance_table(pathway_activity: pd.DataFrame) -> pd.DataFrame:
    frame = pathway_activity.copy()
    feature_cols = [col for col in frame.columns if col.startswith("kegg_")]
    cho = frame[frame["source_column"].eq("CHO:consensus")]
    if cho.empty:
        raise ValueError("Missing CHO:consensus pathway activity")
    nci = frame[~frame["source_column"].astype(str).str.startswith("CHO:")].copy()
    if nci.empty:
        raise ValueError("Missing NCI pathway activity rows")
    cho_values = cho[feature_cols].iloc[0].to_numpy(dtype=float)
    nci_values = nci[feature_cols].mean(axis=0).to_numpy(dtype=float)
    if len(feature_cols) >= 2 and np.std(cho_values) > 0 and np.std(nci_values) > 0:
        r_value, p_value = pearsonr(cho_values, nci_values)
    else:
        r_value, p_value = np.nan, np.nan
    out = pd.DataFrame(
        {
            "pathway": feature_cols,
            "pathway_label": [_pathway_label(col) for col in feature_cols],
            "cho_activity": cho_values,
            "nci_consensus_activity": nci_values,
            "abs_activity_delta": np.abs(cho_values - nci_values),
            "pearson_r": float(r_value),
            "pearson_p": float(p_value),
        }
    )
    return out.sort_values("pathway").reset_index(drop=True)

def select_interaction_pathways(pathway_activity: pd.DataFrame, max_pathways: int = 9) -> list[str]:
    feature_cols = [col for col in pathway_activity.columns if col.startswith("kegg_")]
    selected = [pathway for pathway in PREFERRED_INTERACTION_PATHWAYS if pathway in feature_cols]
    if len(selected) >= int(max_pathways):
        return selected[: int(max_pathways)]
    variance = pathway_activity[feature_cols].var(numeric_only=True).sort_values(ascending=False)
    for pathway in variance.index:
        if pathway not in selected:
            selected.append(pathway)
        if len(selected) >= int(max_pathways):
            break
    return selected

def build_structure_pathway_feature_groups(feature_columns: Sequence[str]) -> dict[str, list[str]]:
    available = set(feature_columns)
    groups = {}
    for group_name, columns in STRUCTURE_PATHWAY_GROUP_COLUMNS.items():
        present = [column for column in columns if column in available]
        if present:
            groups[group_name] = present
    return groups

def _cell_id_from_source(source_column: str) -> str:
    if source_column.startswith("CHO:"):
        return source_column
    if ":" in source_column:
        return source_column.split(":", 1)[1]
    return source_column

def _mol(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)

def _fingerprint(smiles: str):
    mol = _mol(str(smiles))
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)

def canonicalize_smiles(smiles: str) -> str:
    mol = _mol(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True)

def normalize_nonnegative_weights(values: Sequence[float]) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.ones(len(weights), dtype=float) / max(1, len(weights))
    return weights / total

def three_factor_shapley(
    values: dict[frozenset[str], float],
    factors: Sequence[str] = FACTOR_ORDER,
) -> pd.DataFrame:
    factor_list = tuple(factors)
    all_factors = frozenset(factor_list)
    missing = [combo for r in range(4) for combo in itertools.combinations(factor_list, r) if frozenset(combo) not in values]
    if missing:
        raise ValueError(f"Missing Shapley values for combinations: {missing}")
    factorial_n = math.factorial(len(factor_list))
    rows = []
    for factor in factor_list:
        other_factors = [item for item in factor_list if item != factor]
        shapley = 0.0
        for r in range(len(other_factors) + 1):
            for subset_tuple in itertools.combinations(other_factors, r):
                subset = frozenset(subset_tuple)
                coeff = math.factorial(len(subset)) * math.factorial(len(factor_list) - len(subset) - 1) / factorial_n
                shapley += coeff * (float(values[subset | {factor}]) - float(values[subset]))
        rows.append({"factor": factor, "shapley_value": float(shapley)})
    frame = pd.DataFrame(rows)
    abs_sum = float(frame["shapley_value"].abs().sum())
    signed_sum = float(frame["shapley_value"].sum())
    frame["absolute_ratio"] = frame["shapley_value"].abs() / abs_sum if abs_sum else 0.0
    frame["signed_ratio"] = frame["shapley_value"] / signed_sum if signed_sum else 0.0
    frame["full_minus_empty"] = float(values[all_factors]) - float(values[frozenset()])
    return frame

def build_nci_response_matrix(nci_source_rows: pd.DataFrame) -> pd.DataFrame:
    matrix = nci_source_rows.pivot_table(
        index="canonical_smiles",
        columns="cell_id",
        values="y",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    matrix.index.name = "canonical_smiles"
    return matrix.reset_index()

def build_nci_response_stats(nci_source_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = nci_source_rows.groupby("canonical_smiles", sort=True)["y"]
    return grouped.agg(
        nci_toxicity_mean="mean",
        nci_toxicity_var="var",
        nci_toxicity_min="min",
        nci_toxicity_max="max",
        nci_response_count="count",
    ).reset_index().assign(
        nci_toxicity_range=lambda frame: frame["nci_toxicity_max"] - frame["nci_toxicity_min"]
    )

def load_nci_panel_annotations(root_dir: str | Path) -> pd.DataFrame:
    raw_path = Path(root_dir).resolve() / "data_raw" / "DBPs_NCI60_lc50_base.csv"
    raw = pd.read_csv(raw_path)
    raw = raw.rename(columns={"CELL_NAME": "cell_id", "PANEL_NAME": "panel_name", "SMILES": "raw_smiles"})
    raw["canonical_smiles"] = raw["raw_smiles"].map(canonicalize_smiles)
    panel = (
        raw[["cell_id", "panel_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["cell_id", "panel_name"])
        .groupby("cell_id", as_index=False)
        .first()
    )
    compound_panel = raw[["canonical_smiles", "cell_id", "panel_name"]].drop_duplicates()
    return panel.merge(compound_panel, on=["cell_id", "panel_name"], how="left")

def build_cell_similarity_table(pathway_activity: pd.DataFrame, cho_weights: pd.DataFrame) -> pd.DataFrame:
    frame = pathway_activity.copy()
    frame["cell_id"] = frame["source_column"].astype(str).map(_cell_id_from_source)
    feature_cols = [col for col in frame.columns if col.startswith("kegg_")]
    cho = frame[frame["source_column"].eq("CHO:consensus")]
    if cho.empty:
        raise ValueError("Missing CHO:consensus pathway activity")
    nci = frame[~frame["source_column"].astype(str).str.startswith("CHO:")].copy()
    cho_values = cho[feature_cols].iloc[0].to_numpy(dtype=float)
    cho_centered = cho_values - np.nanmean(cho_values)
    nci_values = nci[feature_cols].to_numpy(dtype=float)
    nci_centered = nci_values - np.nanmean(nci_values, axis=1, keepdims=True)
    denom = np.linalg.norm(nci_centered, axis=1) * np.linalg.norm(cho_centered)
    sims = np.divide(nci_centered @ cho_centered, denom, out=np.zeros(len(nci)), where=denom > 0)
    out = pd.DataFrame({"cell_id": nci["cell_id"].to_numpy(), "cell_similarity": sims})
    out = out.merge(cho_weights.rename(columns={"weight": "cho_pathway_weight"}), on="cell_id", how="left")
    out["cho_pathway_weight"] = pd.to_numeric(out["cho_pathway_weight"], errors="coerce").fillna(0.0)
    return out.sort_values("cell_id").reset_index(drop=True)

def _structure_profile(smiles: str) -> dict[str, float]:
    mol = _mol(smiles)
    if mol is None:
        return {}
    counts = {symbol: 0 for symbol in ["F", "Cl", "Br", "I"]}
    aromatic_halogen = 0
    max_halogen_z = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in counts:
            continue
        counts[symbol] += 1
        max_halogen_z = max(max_halogen_z, int(atom.GetAtomicNum()))
        if any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()):
            aromatic_halogen += 1
    patterns = {
        "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        "phenol": "c[OX2H1]",
        "carboxyl": "C(=O)[OX2H1]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "quinone": "O=C1C=CC(=O)C=C1",
        "amide": "C(=O)N",
        "trihalomethyl": "C([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]",
    }
    profile = {f"count_{symbol}": float(count) for symbol, count in counts.items()}
    profile.update(
        {
            "halogen_count": float(sum(counts.values())),
            "heavy_halogen_count": float(counts["Cl"] + counts["Br"] + counts["I"]),
            "aromatic_halogen_count": float(aromatic_halogen),
            "max_halogen_z": float(max_halogen_z),
            "mol_wt": float(Descriptors.MolWt(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
        }
    )
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        profile[f"fg_{name}"] = float(len(mol.GetSubstructMatches(pattern))) if pattern is not None else 0.0
    return profile

def describe_structure_change(smiles_a: str, smiles_b: str) -> list[str]:
    a = _structure_profile(smiles_a)
    b = _structure_profile(smiles_b)
    if not a or not b:
        return ["invalid_structure"]
    tags: list[str] = []
    if b["halogen_count"] > a["halogen_count"]:
        tags.append("halogen_count_gain")
    if b["halogen_count"] < a["halogen_count"]:
        tags.append("halogen_count_loss")
    if b["heavy_halogen_count"] > a["heavy_halogen_count"]:
        tags.append("heavy_halogen_gain")
    if b["max_halogen_z"] > a["max_halogen_z"]:
        tags.append("heavy_halogen_z_increase")
    if a["count_Cl"] > b["count_Cl"] and (b["count_Br"] > a["count_Br"] or b["count_I"] > a["count_I"]):
        tags.append("Cl_to_Br_or_I")
    if a["count_Br"] > b["count_Br"] and b["count_I"] > a["count_I"]:
        tags.append("Br_to_I")
    if b["aromatic_halogen_count"] != a["aromatic_halogen_count"]:
        tags.append("aromatic_halogen_change")
    for group in ["nitro", "phenol", "carboxyl", "aldehyde", "quinone", "amide", "trihalomethyl"]:
        delta = b[f"fg_{group}"] - a[f"fg_{group}"]
        if delta > 0:
            tags.append(f"{group}_gain")
        elif delta < 0:
            tags.append(f"{group}_loss")
    if not tags:
        tags.append("other_minor_change")
    return sorted(set(tags))

def build_cho_similarity_pairs(
    cho_literature: pd.DataFrame,
    similarity_threshold: float = 0.5,
    max_time_delta_h: int = 24,
) -> pd.DataFrame:
    frame = cho_literature.reset_index(drop=True).copy()
    fps = [_fingerprint(smiles) for smiles in frame["canonical_smiles"].astype(str)]
    rows = []
    for i in range(len(frame)):
        fp_i = fps[i]
        if fp_i is None:
            continue
        for j in range(i + 1, len(frame)):
            if abs(int(frame.loc[i, "time_h"]) - int(frame.loc[j, "time_h"])) > int(max_time_delta_h):
                continue
            fp_j = fps[j]
            if fp_j is None:
                continue
            sim = float(DataStructs.TanimotoSimilarity(fp_i, fp_j))
            if sim < float(similarity_threshold):
                continue
            y_a = float(frame.loc[i, "y"])
            y_b = float(frame.loc[j, "y"])
            rows.append(
                {
                    "sample_id_a": frame.loc[i, "sample_id"],
                    "sample_id_b": frame.loc[j, "sample_id"],
                    "canonical_smiles_a": frame.loc[i, "canonical_smiles"],
                    "canonical_smiles_b": frame.loc[j, "canonical_smiles"],
                    "name_a": frame.loc[i].get("name", ""),
                    "name_b": frame.loc[j].get("name", ""),
                    "time_h_a": int(frame.loc[i, "time_h"]),
                    "time_h_b": int(frame.loc[j, "time_h"]),
                    "pLC50_a": y_a,
                    "pLC50_b": y_b,
                    "tanimoto": sim,
                    "delta_pLC50": y_b - y_a,
                    "abs_delta_pLC50": abs(y_b - y_a),
                    "structure_change_tags": ";".join(
                        describe_structure_change(frame.loc[i, "canonical_smiles"], frame.loc[j, "canonical_smiles"])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["tanimoto", "abs_delta_pLC50"], ascending=[False, True]).reset_index(drop=True)

def summarize_structure_changes(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in pairs.itertuples(index=False):
        for tag in str(row.structure_change_tags).split(";"):
            if tag:
                rows.append({"structure_change": tag, "delta_pLC50": float(row.delta_pLC50)})
    if not rows:
        return pd.DataFrame(columns=["structure_change", "n", "mean_delta", "median_delta", "iqr_delta"])
    long = pd.DataFrame(rows)
    grouped = long.groupby("structure_change", sort=True)["delta_pLC50"]
    summary = grouped.agg(n="count", mean_delta="mean", median_delta="median").reset_index()
    q75 = grouped.quantile(0.75).rename("q75")
    q25 = grouped.quantile(0.25).rename("q25")
    summary = summary.merge(q25, on="structure_change").merge(q75, on="structure_change")
    summary["iqr_delta"] = summary["q75"] - summary["q25"]
    return summary.sort_values(["n", "mean_delta"], ascending=[False, False]).reset_index(drop=True)

def _factor_combo_label(factors: Iterable[str]) -> str:
    selected = tuple(factor for factor in FACTOR_ORDER if factor in set(factors))
    if not selected:
        return "T0_None"
    return "T" + str({
        ("struct",): 1,
        ("cell",): 2,
        ("time",): 3,
        ("struct", "cell"): 4,
        ("struct", "time"): 5,
        ("cell", "time"): 6,
        ("struct", "cell", "time"): 7,
    }[selected]) + "_" + "_".join(part.capitalize() for part in selected)

def factor_combinations() -> list[tuple[str, tuple[str, ...]]]:
    combos = []
    for factors in [
        (),
        ("struct",),
        ("cell",),
        ("time",),
        ("struct", "cell"),
        ("struct", "time"),
        ("cell", "time"),
        ("struct", "cell", "time"),
    ]:
        combos.append((_factor_combo_label(factors), factors))
    return combos

def build_factor_transfer_features(
    target_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    cell_weights: pd.DataFrame,
    factors: Iterable[str],
    time_scale_h: float = 24.0,
    topk: int = 5,
) -> pd.DataFrame:
    factor_set = set(factors)
    unknown = factor_set - set(FACTOR_ORDER)
    if unknown:
        raise ValueError(f"Unknown transfer factors: {sorted(unknown)}")
    source = source_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy().reset_index(drop=True)
    source["cell_weight"] = source["cell_id"].map(
        cell_weights.set_index("cell_id")["weight"].to_dict()
    ).fillna(0.0)
    source_fps = [_fingerprint(smiles) for smiles in source["canonical_smiles"].astype(str)]
    source_y = pd.to_numeric(source["y"], errors="coerce").to_numpy(dtype=float)
    source_time = pd.to_numeric(source["time_h"], errors="coerce").to_numpy(dtype=float)
    source_cell_weight = pd.to_numeric(source["cell_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    rows = []
    for target in target_rows.itertuples(index=False):
        target_fp = _fingerprint(str(target.canonical_smiles))
        if target_fp is None:
            struct_sim = np.zeros(len(source), dtype=float)
        else:
            valid = [(idx, fp) for idx, fp in enumerate(source_fps) if fp is not None]
            struct_sim = np.zeros(len(source), dtype=float)
            if valid:
                valid_indices = [idx for idx, _ in valid]
                struct_sim[valid_indices] = DataStructs.BulkTanimotoSimilarity(target_fp, [fp for _, fp in valid])
        cell_sim = source_cell_weight.copy()
        time_sim = np.exp(-np.abs(source_time - float(target.time_h)) / float(time_scale_h))
        raw_weights = np.ones(len(source), dtype=float)
        if "struct" in factor_set:
            raw_weights *= np.clip(struct_sim, 0.0, None)
        if "cell" in factor_set:
            raw_weights *= np.clip(cell_sim, 0.0, None)
        if "time" in factor_set:
            raw_weights *= np.clip(time_sim, 0.0, None)
        weights = normalize_nonnegative_weights(raw_weights)
        order = np.argsort(-weights)[: max(1, int(topk))]
        y_wmean = float(np.average(source_y, weights=weights))
        y_wstd = float(np.sqrt(np.average((source_y - y_wmean) ** 2, weights=weights)))
        entropy = float(-(weights * np.log(np.clip(weights, 1e-12, None))).sum())
        effective_n = float(1.0 / np.square(weights).sum()) if float(np.square(weights).sum()) else 0.0
        rows.append(
            {
                "sample_id": target.sample_id,
                "factor_transfer_y_wmean": y_wmean,
                "factor_transfer_y_wstd": y_wstd,
                "factor_transfer_weight_max": float(weights.max()) if len(weights) else 0.0,
                "factor_transfer_effective_n": effective_n,
                "factor_transfer_weight_entropy": entropy,
                "factor_transfer_top1_struct_sim": float(struct_sim[order[0]]) if len(order) else 0.0,
            }
        )
    features = pd.DataFrame(rows)
    leaked = FORBIDDEN_LABEL_COLUMNS & set(features.columns)
    if leaked:
        raise ValueError(f"Label-equivalent columns leaked into transfer factors: {sorted(leaked)}")
    return features

def build_pathway_transfer_features(
    target_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    pathway_activity: pd.DataFrame,
    top_pathways: Sequence[str],
    time_scale_h: float = 24.0,
) -> pd.DataFrame:
    pathway_frame = pathway_activity.copy()
    pathway_frame["cell_id"] = pathway_frame["source_column"].astype(str).map(_cell_id_from_source)
    selected_pathways = [pathway for pathway in top_pathways if pathway in pathway_frame.columns]
    if not selected_pathways:
        raise ValueError("No selected KEGG pathways are present in pathway_activity")
    nci_pathway = pathway_frame[~pathway_frame["source_column"].astype(str).str.startswith("CHO:")][
        ["cell_id", *selected_pathways]
    ].copy()
    source = source_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy().reset_index(drop=True)
    source = source.merge(nci_pathway, on="cell_id", how="left")
    source[selected_pathways] = source[selected_pathways].apply(pd.to_numeric, errors="coerce")
    source[selected_pathways] = source[selected_pathways].fillna(source[selected_pathways].median(numeric_only=True)).fillna(0.0)

    source_fps = [_fingerprint(smiles) for smiles in source["canonical_smiles"].astype(str)]
    source_y = pd.to_numeric(source["y"], errors="coerce").to_numpy(dtype=float)
    source_time = pd.to_numeric(source["time_h"], errors="coerce").to_numpy(dtype=float)
    pathway_values = source[selected_pathways].to_numpy(dtype=float)
    pathway_center = np.nanmean(pathway_values, axis=0, keepdims=True)
    pathway_scale = np.nanstd(pathway_values, axis=0, keepdims=True)
    pathway_z = np.divide(
        pathway_values - pathway_center,
        np.where(pathway_scale > 0, pathway_scale, 1.0),
        out=np.zeros_like(pathway_values, dtype=float),
        where=np.isfinite(pathway_values),
    )

    rows = []
    for target in target_rows.itertuples(index=False):
        target_fp = _fingerprint(str(target.canonical_smiles))
        struct_sim = np.zeros(len(source), dtype=float)
        if target_fp is not None:
            valid = [(idx, fp) for idx, fp in enumerate(source_fps) if fp is not None]
            if valid:
                valid_indices = [idx for idx, _ in valid]
                struct_sim[valid_indices] = DataStructs.BulkTanimotoSimilarity(target_fp, [fp for _, fp in valid])
        time_sim = np.exp(-np.abs(source_time - float(target.time_h)) / float(time_scale_h))
        weights = normalize_nonnegative_weights(np.clip(struct_sim, 0.0, None) * np.clip(time_sim, 0.0, None))
        row = {"sample_id": target.sample_id}
        for pathway_idx, pathway_name in enumerate(selected_pathways):
            column = f"pathway_transfer_{pathway_name}"
            row[column] = float(np.average(source_y * pathway_z[:, pathway_idx], weights=weights))
        rows.append(row)
    features = pd.DataFrame(rows)
    leaked = FORBIDDEN_LABEL_COLUMNS & set(features.columns)
    if leaked:
        raise ValueError(f"Label-equivalent columns leaked into pathway transfer features: {sorted(leaked)}")
    return features

def build_cross_cell_transfer_pairs(
    cho_rows: pd.DataFrame,
    nci_rows: pd.DataFrame,
    cell_similarity: pd.DataFrame,
    prediction_errors: pd.DataFrame,
    similarity_threshold: float = 0.5,
    max_sources_per_target: int = 120,
) -> pd.DataFrame:
    nci = nci_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy().reset_index(drop=True)
    nci_fps = [_fingerprint(smiles) for smiles in nci["canonical_smiles"].astype(str)]
    cell_map = cell_similarity.set_index("cell_id")[["cell_similarity", "cho_pathway_weight"]].to_dict("index")
    error_map = prediction_errors.set_index("sample_id")["abs_model_error"].to_dict()
    rows = []
    for target in cho_rows.itertuples(index=False):
        target_fp = _fingerprint(str(target.canonical_smiles))
        if target_fp is None:
            continue
        sims = np.zeros(len(nci), dtype=float)
        valid = [(idx, fp) for idx, fp in enumerate(nci_fps) if fp is not None]
        if valid:
            valid_indices = [idx for idx, _ in valid]
            sims[valid_indices] = DataStructs.BulkTanimotoSimilarity(target_fp, [fp for _, fp in valid])
        exact = nci["canonical_smiles"].astype(str).eq(str(target.canonical_smiles)).to_numpy()
        candidate_idx = np.flatnonzero((sims >= float(similarity_threshold)) | exact)
        candidate_idx = candidate_idx[np.argsort(-sims[candidate_idx])[: int(max_sources_per_target)]]
        for idx in candidate_idx:
            nci_row = nci.iloc[int(idx)]
            cell = cell_map.get(str(nci_row["cell_id"]), {"cell_similarity": 0.0, "cho_pathway_weight": 0.0})
            s_time = float(np.exp(-abs(float(target.time_h) - float(nci_row["time_h"])) / 24.0))
            rows.append(
                {
                    "sample_id": target.sample_id,
                    "canonical_smiles_cho": target.canonical_smiles,
                    "canonical_smiles_nci": nci_row["canonical_smiles"],
                    "cell_id": nci_row["cell_id"],
                    "time_h_cho": int(target.time_h),
                    "time_h_nci": int(nci_row["time_h"]),
                    "y_cho": float(target.y),
                    "y_nci": float(nci_row["y"]),
                    "S_struct": float(sims[int(idx)]),
                    "S_cell": float(cell["cell_similarity"]),
                    "S_cell_weight": float(cell["cho_pathway_weight"]),
                    "S_time": s_time,
                    "is_exact_compound": bool(exact[int(idx)]),
                    "abs_source_target_delta": abs(float(nci_row["y"]) - float(target.y)),
                    "abs_model_error": float(error_map.get(target.sample_id, np.nan)),
                }
            )
    return pd.DataFrame(rows)

def summarize_cross_cell_stats(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x_col, y_col in [
        ("S_cell", "abs_source_target_delta"),
        ("S_cell_weight", "abs_source_target_delta"),
        ("S_cell", "abs_model_error"),
        ("S_cell_weight", "abs_model_error"),
        ("S_struct", "abs_source_target_delta"),
        ("S_time", "abs_source_target_delta"),
    ]:
        sub = pairs[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(sub[x_col], sub[y_col])
        rows.append({"x": x_col, "y": y_col, "n": int(len(sub)), "spearman_rho": float(rho), "p_value": float(p_value)})
    return pd.DataFrame(rows)

def prediction_errors_from_artifact(artifact_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(artifact_dir / "predictions.csv")
    predictions["abs_model_error"] = (predictions["y_true"] - predictions["y_pred"]).abs()
    return (
        predictions.groupby("sample_id", as_index=False)
        .agg(abs_model_error=("abs_model_error", "mean"), prediction_count=("abs_model_error", "size"))
    )

def ablation_base_feature_columns(feature_columns: Sequence[str]) -> list[str]:
    selected = [
        col
        for col in feature_columns
        if col.startswith("rdkit_")
        or col.startswith("maccs_")
        or col.startswith("tt_")
        or col.startswith("atom_")
        or col.startswith("domain_")
        or col.startswith("fg_")
        or col.startswith("time_h_")
        or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
    ]
    return [col for col in selected if not (col.startswith("teacher_") or col.startswith("source_teacher_"))]

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return {"r2": r2, "rmse": rmse, "mae": mae}

def grouped_permutation_importance(
    estimator,
    frame: pd.DataFrame,
    y: np.ndarray,
    feature_columns: Sequence[str],
    groups: dict[str, list[str]],
    repeats: int = 5,
    random_state: int = 20260611,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    x_base = frame[list(feature_columns)].to_numpy(dtype=float)
    base_pred = np.asarray(estimator.predict(x_base), dtype=float)
    base_metrics = regression_metrics(y, base_pred)
    rows = []
    for group_name, columns in groups.items():
        group_cols = [col for col in columns if col in feature_columns]
        if not group_cols:
            continue
        col_indices = [list(feature_columns).index(col) for col in group_cols]
        drops = []
        rmse_increases = []
        for _ in range(int(repeats)):
            x_perm = x_base.copy()
            order = rng.permutation(len(x_perm))
            x_perm[:, col_indices] = x_perm[order][:, col_indices]
            pred = np.asarray(estimator.predict(x_perm), dtype=float)
            metrics = regression_metrics(y, pred)
            drops.append(base_metrics["r2"] - metrics["r2"])
            rmse_increases.append(metrics["rmse"] - base_metrics["rmse"])
        rows.append(
            {
                "feature_group": group_name,
                "n_features": len(group_cols),
                "baseline_r2": base_metrics["r2"],
                "mean_delta_r2": float(np.mean(drops)),
                "std_delta_r2": float(np.std(drops)),
                "mean_delta_rmse": float(np.mean(rmse_increases)),
                "std_delta_rmse": float(np.std(rmse_increases)),
            }
        )
    return pd.DataFrame(rows)

def summarize_model_transfer_ablation(
    metrics: pd.DataFrame,
    baseline_step: str,
    full_transfer_step: str,
) -> pd.DataFrame:
    required = {"split_seed", "ablation_step", "split_role", "r2"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Model transfer ablation metrics missing columns: {sorted(missing)}")
    aggregations = {
        "seed_count": ("split_seed", "nunique"),
        "r2_mean": ("r2", "mean"),
        "r2_min": ("r2", "min"),
        "r2_max": ("r2", "max"),
    }
    if "rmse" in metrics.columns:
        aggregations["rmse_mean"] = ("rmse", "mean")
    if "mae" in metrics.columns:
        aggregations["mae_mean"] = ("mae", "mean")
    grouped = (
        metrics.groupby(["ablation_step", "split_role"], as_index=False)
        .agg(**aggregations)
        .sort_values(["split_role", "ablation_step"])
        .reset_index(drop=True)
    )
    pivot = metrics.pivot_table(index=["split_seed", "split_role"], columns="ablation_step", values="r2")
    delta_rows = []
    for role in sorted(metrics["split_role"].astype(str).unique()):
        if role not in pivot.index.get_level_values("split_role"):
            continue
        role_pivot = pivot.xs(role, level="split_role")
        if baseline_step not in role_pivot.columns:
            continue
        baseline = role_pivot[baseline_step]
        for step in role_pivot.columns:
            delta = role_pivot[step] - baseline
            delta_rows.append(
                {
                    "ablation_step": step,
                    "split_role": role,
                    "delta_vs_baseline_mean": float(delta.mean()),
                    "delta_vs_baseline_min": float(delta.min()),
                    "delta_vs_baseline_max": float(delta.max()),
                    "delta_vs_baseline_positive_seeds": int((delta > 0).sum()),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    summary = grouped.merge(deltas, on=["ablation_step", "split_role"], how="left")
    summary["baseline_step"] = baseline_step
    summary["full_transfer_step"] = full_transfer_step
    summary["is_full_transfer_step"] = summary["ablation_step"].eq(full_transfer_step)
    return summary

def compare_main_model_performance_lock(
    artifact_metrics_wide: pd.DataFrame,
    model_ablation_metrics: pd.DataFrame,
    full_transfer_step: str,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    role_columns = {
        "r2_train": "train",
        "r2_test": "test",
        "r2_validation": "validation",
        "r2_experimental": "experimental",
    }
    missing = {"split_seed", *role_columns} - set(artifact_metrics_wide.columns)
    if missing:
        raise ValueError(f"Artifact metrics missing columns: {sorted(missing)}")
    required_model = {"split_seed", "ablation_step", "split_role", "r2"}
    missing_model = required_model - set(model_ablation_metrics.columns)
    if missing_model:
        raise ValueError(f"Model ablation metrics missing columns: {sorted(missing_model)}")
    artifact_long = artifact_metrics_wide[["split_seed", *role_columns]].melt(
        id_vars="split_seed",
        var_name="role_column",
        value_name="artifact_r2",
    )
    artifact_long["split_role"] = artifact_long["role_column"].map(role_columns)
    full = model_ablation_metrics[model_ablation_metrics["ablation_step"].eq(full_transfer_step)].copy()
    full = full[["split_seed", "split_role", "r2"]].rename(columns={"r2": "recomputed_r2"})
    merged = artifact_long.merge(full, on=["split_seed", "split_role"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows between artifact metrics and full-transfer ablation metrics")
    merged["abs_r2_diff"] = (merged["artifact_r2"] - merged["recomputed_r2"]).abs()
    max_diff = float(merged["abs_r2_diff"].max())
    return {
        "compared_rows": int(len(merged)),
        "compared_seed_count": int(merged["split_seed"].nunique()),
        "max_abs_r2_diff": max_diff,
        "mean_abs_r2_diff": float(merged["abs_r2_diff"].mean()),
        "performance_unchanged": bool(max_diff <= float(tolerance)),
        "tolerance": float(tolerance),
    }

def permutation_structure_pathway_interactions(
    estimator,
    frame: pd.DataFrame,
    y: np.ndarray,
    feature_columns: Sequence[str],
    structure_groups: dict[str, list[str]],
    pathway_groups: dict[str, list[str]],
    repeats: int = 5,
    random_state: int = 20260612,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    feature_list = list(feature_columns)
    x_base = frame[feature_list].to_numpy(dtype=float)
    base_pred = np.asarray(estimator.predict(x_base), dtype=float)
    base_r2 = regression_metrics(y, base_pred)["r2"]

    def mean_drop(columns: Sequence[str]) -> float:
        group_cols = [col for col in columns if col in feature_list]
        if not group_cols:
            return np.nan
        col_indices = [feature_list.index(col) for col in group_cols]
        drops = []
        for _ in range(int(repeats)):
            x_perm = x_base.copy()
            order = rng.permutation(len(x_perm))
            x_perm[:, col_indices] = x_perm[order][:, col_indices]
            pred = np.asarray(estimator.predict(x_perm), dtype=float)
            drops.append(base_r2 - regression_metrics(y, pred)["r2"])
        return float(np.mean(drops))

    rows = []
    for structure_name, structure_cols in structure_groups.items():
        drop_structure = mean_drop(structure_cols)
        if not np.isfinite(drop_structure):
            continue
        for pathway_name, pathway_cols in pathway_groups.items():
            drop_pathway = mean_drop(pathway_cols)
            drop_joint = mean_drop([*structure_cols, *pathway_cols])
            if not np.isfinite(drop_pathway) or not np.isfinite(drop_joint):
                continue
            rows.append(
                {
                    "structure_group": structure_name,
                    "pathway": pathway_name,
                    "baseline_r2": float(base_r2),
                    "drop_r2_structure": float(drop_structure),
                    "drop_r2_pathway": float(drop_pathway),
                    "drop_r2_joint": float(drop_joint),
                    "interaction_delta_r2": float(drop_joint - drop_structure - drop_pathway),
                }
            )
    return pd.DataFrame(rows)

def plot_nci_cross_cell_heatmap(
    response_matrix: pd.DataFrame,
    selected_compounds: Sequence[str],
    out_path: Path,
    compound_map: pd.DataFrame | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    selected = [compound for compound in selected_compounds if compound in set(response_matrix["canonical_smiles"])]
    if not selected:
        selected = response_matrix["canonical_smiles"].head(20).tolist()
    table = response_matrix[response_matrix["canonical_smiles"].isin(selected)].set_index("canonical_smiles")
    values = table.apply(pd.to_numeric, errors="coerce")
    centered = values.sub(values.mean(axis=1), axis=0)
    scale = values.std(axis=1).replace(0, 1.0)
    z = centered.div(scale, axis=0).fillna(0.0)
    if compound_map is not None and not compound_map.empty:
        id_map = compound_map.set_index("canonical_smiles")["compound_id"].to_dict()
        z.index = [id_map.get(smiles, smiles) for smiles in z.index]
    width = max(10, min(22, z.shape[1] * 0.25))
    height = max(5, min(16, z.shape[0] * 0.35))
    plt.figure(figsize=(width, height))
    sns.heatmap(z, cmap="vlag", center=0, cbar_kws={"label": "Row z-score pLC50"})
    plt.xlabel("NCI-60 cell line")
    plt.ylabel("NCI source compound")
    plt.title("Figure 3.3-A NCI-60 cross-cell toxicity response")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_cell_similarity_transfer_scatter(
    pairs: pd.DataFrame,
    stats: pd.DataFrame,
    out_path: Path,
    pathway_concordance: pd.DataFrame | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False)
    if pathway_concordance is not None and not pathway_concordance.empty:
        sub = pathway_concordance[["nci_consensus_activity", "cho_activity", "pearson_r", "pearson_p"]].dropna()
        axes[0].scatter(sub["nci_consensus_activity"], sub["cho_activity"], s=18, alpha=0.7, edgecolor="none")
        if len(sub):
            r_value = float(sub["pearson_r"].iloc[0])
            p_value = float(sub["pearson_p"].iloc[0])
            axes[0].text(0.04, 0.96, f"Pearson R={r_value:.3f}\np={p_value:.2g}", transform=axes[0].transAxes, va="top")
        axes[0].set_xlabel("NCI-60 consensus KEGG activity")
        axes[0].set_ylabel("CHO-K1 KEGG activity")
    else:
        axes[0].axis("off")
    axes[0].grid(alpha=0.25)

    x_col, y_col = "S_cell", "abs_model_error"
    sub = pairs[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].scatter(sub[x_col], sub[y_col], s=14, alpha=0.55, edgecolor="none")
    stat = stats[(stats["x"].eq(x_col)) & (stats["y"].eq(y_col))]
    if not stat.empty:
        rho = float(stat.iloc[0]["spearman_rho"])
        p_value = float(stat.iloc[0]["p_value"])
        axes[1].text(0.04, 0.96, f"Spearman rho={rho:.3f}\np={p_value:.2g}", transform=axes[1].transAxes, va="top")
    axes[1].set_xlabel("CHO-K1 vs NCI cell KEGG similarity")
    axes[1].set_ylabel("Best-model |prediction error|")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Figure 3.3-B Cell-state rationale and boundary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_cho_structure_network(
    cho_literature: pd.DataFrame,
    pairs: pd.DataFrame,
    out_path: Path,
    edge_threshold: float = 0.6,
    max_edges: int = 240,
) -> None:
    import networkx as nx
    import plotly.graph_objects as go

    edges = pairs[pairs["tanimoto"].ge(edge_threshold)].sort_values("tanimoto", ascending=False).head(max_edges)
    graph = nx.Graph()
    node_rows = cho_literature.set_index("sample_id")
    for row in edges.itertuples(index=False):
        graph.add_edge(row.sample_id_a, row.sample_id_b, weight=float(row.tanimoto))
    for node in list(graph.nodes):
        if node in node_rows.index:
            graph.nodes[node]["y"] = float(node_rows.loc[node, "y"])
            graph.nodes[node]["name"] = str(node_rows.loc[node, "name"])
            graph.nodes[node]["time_h"] = int(node_rows.loc[node, "time_h"])
    if graph.number_of_nodes() == 0:
        out_path.write_text("<html><body><p>No CHO similarity network edges passed the threshold.</p></body></html>", encoding="utf-8")
        return
    pos = nx.spring_layout(graph, seed=20260611, weight="weight")
    edge_x, edge_y = [], []
    for source, target in graph.edges:
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_x, node_y, colors, text = [], [], [], []
    for node in graph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        colors.append(graph.nodes[node].get("y", np.nan))
        text.append(f"{node}<br>{graph.nodes[node].get('name', '')}<br>pLC50={graph.nodes[node].get('y', np.nan):.3f}<br>{graph.nodes[node].get('time_h', '')} h")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.8, color="rgba(80,80,80,0.35)"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=8, color=colors, colorscale="Viridis", colorbar=dict(title="pLC50")),
            text=text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Figure 3.3-C CHO-K1 similar-structure toxicity network",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000,
        height=720,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")

def plot_structure_change_delta(pairs: pd.DataFrame, summary: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary = filter_main_structure_change_summary(summary)
    if summary.empty:
        out_path.write_text("")
        return
    top_tags = summary.head(top_n)["structure_change"].tolist()
    rows = []
    for row in pairs.itertuples(index=False):
        for tag in str(row.structure_change_tags).split(";"):
            if tag in top_tags:
                rows.append({"structure_change": tag, "delta_pLC50": float(row.delta_pLC50)})
    long = pd.DataFrame(rows)
    order = [tag for tag in top_tags if tag in set(long["structure_change"])]
    plt.figure(figsize=(10, max(5, len(order) * 0.36)))
    sns.boxplot(data=long, y="structure_change", x="delta_pLC50", order=order, color="#d9e6f2", showfliers=False)
    sns.stripplot(data=long, y="structure_change", x="delta_pLC50", order=order, color="#24476b", size=2.5, alpha=0.45)
    ax = plt.gca()
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([STRUCTURE_CHANGE_LABELS.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Delta pLC50 (B - A)")
    plt.ylabel("Structure change type")
    plt.title("Figure 3.3-C Structure transformation and CHO-K1 toxicity difference")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_structure_pathway_interaction_heatmap(interactions: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if interactions.empty:
        out_path.write_text("")
        return
    plot_frame = interactions.copy()
    pivot = plot_frame.pivot_table(
        index="structure_group",
        columns="pathway",
        values="interaction_delta_r2",
        aggfunc="mean",
    )
    row_order = [row for row in STRUCTURE_PATHWAY_GROUP_COLUMNS if row in pivot.index]
    col_order = [col for col in [_pathway_label(p) for p in PREFERRED_INTERACTION_PATHWAYS] if col in pivot.columns]
    pivot = pivot.reindex(index=row_order or None, columns=col_order or None)
    plt.figure(figsize=(11, max(4.8, 0.55 * len(pivot.index))))
    values = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if pivot.size and np.isfinite(values).any() else 0.0
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-vmax if vmax else None,
        vmax=vmax if vmax else None,
        cbar_kws={"label": "Interaction Delta R2"},
        linewidths=0.4,
        linecolor="white",
    )
    plt.xlabel("KEGG pathway context")
    plt.ylabel("Structure group")
    plt.title("Figure 3.3-D Structure x KEGG pathway interaction")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_transfer_factor_contribution(
    ablation_metrics: pd.DataFrame,
    shapley: pd.DataFrame,
    permutation: pd.DataFrame,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    del ablation_metrics, shapley
    perm = permutation.copy()
    if "analysis_model" in perm.columns and perm["analysis_model"].eq("structure_pathway_t7").any():
        perm = perm[perm["analysis_model"].eq("structure_pathway_t7")]
    perm = perm.groupby("feature_group", as_index=False)["mean_delta_r2"].mean().sort_values("mean_delta_r2")
    label_map = {
        "chemistry_structure": "Chemistry structure",
        "transfer_factor_features": "NCI-60 transfer response",
        "exposure_time": "Exposure time",
        "cell_state_pathway_features": "Cell-state features",
    }
    perm["feature_group_label"] = perm["feature_group"].map(label_map).fillna(perm["feature_group"])
    plt.figure(figsize=(8.5, max(3.6, 0.55 * len(perm))))
    if not perm.empty:
        sns.barplot(data=perm, y="feature_group_label", x="mean_delta_r2", color="#6f9fb5")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Mean R2 drop")
    plt.ylabel("")
    plt.title("Figure 3.3-E Feature-group contribution decomposition")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_model_transfer_ablation(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if summary.empty:
        out_path.write_text("")
        return
    roles = ["test", "validation", "experimental"]
    plot_frame = summary[summary["split_role"].isin(roles)].copy()
    plot_frame = plot_frame[~plot_frame["ablation_step"].str.startswith("A_")].copy()
    label_map = {
        "B_transfer_features_only": "Transfer features",
        "C_nci_response_residual": "NCI response residual",
        "D_full_transfer_selected": "Full transfer",
    }
    plot_frame["ablation_label"] = plot_frame["ablation_step"].map(label_map).fillna(plot_frame["ablation_step"])
    plt.figure(figsize=(9.5, 5.2))
    sns.barplot(
        data=plot_frame,
        y="ablation_label",
        x="delta_vs_baseline_mean",
        hue="split_role",
        hue_order=roles,
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Mean R2 delta vs structure/time baseline")
    plt.ylabel("")
    plt.title("Model-family transfer ablation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

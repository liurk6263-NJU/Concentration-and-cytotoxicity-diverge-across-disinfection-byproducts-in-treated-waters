from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor

from .data_contract import DataContractBundle
from .kegg import build_pathway_bundle
from .splits import FrozenSplit

FORBIDDEN_FEATURE_COLUMNS = {
    "y",
    "target_plc50",
    "actual_lc50_m",
    "pLC50",
    "LC50(M)",
    "LC50(M)_由pLC50换算",
    "LC50(μM)_由pLC50换算",
}

DOMAIN_SMARTS = {
    "fg_carboxylic_acid": "C(=O)[OX2H1]",
    "fg_phenol": "c[OX2H1]",
    "fg_aldehyde": "[CX3H1](=O)[#6]",
    "fg_nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "fg_quinone": "O=C1C=CC(=O)C=C1",
    "fg_alpha_amino_acid": "[NX3;H2,H1;!$(NC=O)][CX4H]C(=O)[OX2H1]",
    "fg_amide": "C(=O)N",
    "fg_trihalomethyl": "C([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]",
}
DOMAIN_PATTERNS = {name: Chem.MolFromSmarts(pattern) for name, pattern in DOMAIN_SMARTS.items()}
DOMAIN_FEATURE_COLUMNS = [
    "domain_halogen_mass",
    "domain_halogen_fraction_mw",
    "domain_halogen_atom_fraction",
    "domain_aromatic_halogen_count",
    "domain_aliphatic_halogen_count",
    "domain_max_halogen_z",
    "domain_halogen_diversity",
    "domain_bromine_iodine_count",
    "domain_cl_br_i_count",
    "domain_tpsa_per_mw",
    "domain_logp_per_heavy_atom",
    *DOMAIN_SMARTS.keys(),
]
TOPOLOGICAL_TORSION_FP_SIZE = 1024
TOPOLOGICAL_TORSION_GENERATOR = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
    fpSize=TOPOLOGICAL_TORSION_FP_SIZE
)
MORGAN_FP_SIZE = 1024
MORGAN_FEATURE_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_FP_SIZE)

@dataclass(frozen=True)
class ModelMatrix:
    frame: pd.DataFrame
    feature_columns: list[str]

    def role(self, split_role: str) -> pd.DataFrame:
        return self.frame[self.frame["split_role"].eq(split_role)].copy()

def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)

def _fingerprint(mol):
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(mol)

def _add_empty_domain_features(row: dict[str, float | str]) -> None:
    for column in DOMAIN_FEATURE_COLUMNS:
        row[column] = 0.0

def _add_empty_topological_torsion_features(row: dict[str, float | str]) -> None:
    for idx in range(TOPOLOGICAL_TORSION_FP_SIZE):
        row[f"tt_{idx:04d}"] = 0.0

def _add_empty_morgan_features(row: dict[str, float | str]) -> None:
    for idx in range(MORGAN_FP_SIZE):
        row[f"morgan_{idx:04d}"] = 0.0

def _add_topological_torsion_features(row: dict[str, float | str], mol) -> None:
    fp = TOPOLOGICAL_TORSION_GENERATOR.GetFingerprint(mol)
    on_bits = set(fp.GetOnBits())
    for idx in range(TOPOLOGICAL_TORSION_FP_SIZE):
        row[f"tt_{idx:04d}"] = float(idx in on_bits)

def _add_morgan_features(row: dict[str, float | str], mol) -> None:
    fp = MORGAN_FEATURE_GENERATOR.GetFingerprint(mol)
    on_bits = set(fp.GetOnBits())
    for idx in range(MORGAN_FP_SIZE):
        row[f"morgan_{idx:04d}"] = float(idx in on_bits)

def _add_domain_features(row: dict[str, float | str], mol) -> None:
    mol_wt = float(Descriptors.MolWt(mol))
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    halogen_mass = 0.0
    halogen_count = 0
    aromatic_halogen_count = 0
    aliphatic_halogen_count = 0
    max_halogen_z = 0
    halogen_symbols = set()
    bromine_iodine_count = 0
    cl_br_i_count = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in {"F", "Cl", "Br", "I"}:
            continue
        halogen_count += 1
        halogen_symbols.add(symbol)
        halogen_mass += float(atom.GetMass())
        max_halogen_z = max(max_halogen_z, int(atom.GetAtomicNum()))
        attached_to_aromatic = any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors())
        if attached_to_aromatic:
            aromatic_halogen_count += 1
        else:
            aliphatic_halogen_count += 1
        if symbol in {"Br", "I"}:
            bromine_iodine_count += 1
        if symbol in {"Cl", "Br", "I"}:
            cl_br_i_count += 1
    row.update(
        {
            "domain_halogen_mass": float(halogen_mass),
            "domain_halogen_fraction_mw": float(halogen_mass / mol_wt) if mol_wt else 0.0,
            "domain_halogen_atom_fraction": float(halogen_count / heavy_atoms) if heavy_atoms else 0.0,
            "domain_aromatic_halogen_count": float(aromatic_halogen_count),
            "domain_aliphatic_halogen_count": float(aliphatic_halogen_count),
            "domain_max_halogen_z": float(max_halogen_z),
            "domain_halogen_diversity": float(len(halogen_symbols)),
            "domain_bromine_iodine_count": float(bromine_iodine_count),
            "domain_cl_br_i_count": float(cl_br_i_count),
            "domain_tpsa_per_mw": float(rdMolDescriptors.CalcTPSA(mol) / mol_wt) if mol_wt else 0.0,
            "domain_logp_per_heavy_atom": float(Descriptors.MolLogP(mol) / heavy_atoms) if heavy_atoms else 0.0,
        }
    )
    for name, pattern in DOMAIN_PATTERNS.items():
        row[name] = float(len(mol.GetSubstructMatches(pattern))) if pattern is not None else 0.0

def _chem_features(smiles_values: list[str]) -> pd.DataFrame:
    descriptor_items = Descriptors.descList
    rows = []
    for smiles in sorted(set(map(str, smiles_values))):
        mol = _mol_from_smiles(smiles)
        row: dict[str, float | str] = {"canonical_smiles": smiles}
        if mol is None:
            for name, _ in descriptor_items:
                row[f"rdkit_{name}"] = np.nan
            for idx in range(167):
                row[f"maccs_{idx:03d}"] = 0.0
            for atom_symbol in ["C", "N", "O", "F", "Cl", "Br", "I"]:
                row[f"atom_{atom_symbol}"] = 0.0
            row["halogen_count"] = 0.0
            row["heavy_halogen_count"] = 0.0
            row["aromatic_atom_count"] = 0.0
            _add_empty_domain_features(row)
            _add_empty_topological_torsion_features(row)
            _add_empty_morgan_features(row)
        else:
            for name, fn in descriptor_items:
                try:
                    value = float(fn(mol))
                except Exception:
                    value = np.nan
                row[f"rdkit_{name}"] = value if np.isfinite(value) else np.nan
            maccs = MACCSkeys.GenMACCSKeys(mol)
            bits = list(maccs.ToBitString())
            for idx, bit in enumerate(bits):
                row[f"maccs_{idx:03d}"] = float(bit)
            atom_counts = {symbol: 0 for symbol in ["C", "N", "O", "F", "Cl", "Br", "I"]}
            aromatic_atoms = 0
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_counts:
                    atom_counts[symbol] += 1
                if atom.GetIsAromatic():
                    aromatic_atoms += 1
            for atom_symbol, count in atom_counts.items():
                row[f"atom_{atom_symbol}"] = float(count)
            row["halogen_count"] = float(atom_counts["F"] + atom_counts["Cl"] + atom_counts["Br"] + atom_counts["I"])
            row["heavy_halogen_count"] = float(atom_counts["Cl"] + atom_counts["Br"] + atom_counts["I"])
            row["aromatic_atom_count"] = float(aromatic_atoms)
            _add_domain_features(row, mol)
            _add_topological_torsion_features(row, mol)
            _add_morgan_features(row, mol)
        rows.append(row)
    frame = pd.DataFrame(rows)
    numeric_cols = [c for c in frame.columns if c != "canonical_smiles"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    frame[numeric_cols] = frame[numeric_cols].fillna(frame[numeric_cols].median(numeric_only=True)).fillna(0.0)
    return frame

def _source_scores(bundle: DataContractBundle) -> pd.DataFrame:
    pathway_bundle = build_pathway_bundle(bundle.root_dir)
    weights = pathway_bundle.cho_weights.rename(columns={"weight": "cho_pathway_weight"})
    nci = bundle.nci_source_rows.merge(weights, on="cell_id", how="left")
    nci["cho_pathway_weight"] = nci["cho_pathway_weight"].fillna(0.0)
    if float(nci["cho_pathway_weight"].sum()) == 0.0:
        nci["cho_pathway_weight"] = 1.0

    rows = []
    for smiles, group in nci.groupby("canonical_smiles", sort=True):
        weights_arr = group["cho_pathway_weight"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        weight_sum = float(weights_arr.sum())
        weighted_mean = float(np.average(y, weights=weights_arr)) if weight_sum > 0 else float(np.mean(y))
        rows.append(
            {
                "canonical_smiles": smiles,
                "teacher_source_y_mean": float(np.mean(y)),
                "teacher_source_y_std": float(np.std(y)),
                "teacher_source_y_cho_wmean": weighted_mean,
                "teacher_source_n_rows": int(len(group)),
                "teacher_source_n_cells": int(group["cell_id"].nunique()),
                "teacher_source_weight_sum": weight_sum,
            }
        )
    return pd.DataFrame(rows)

def _response_profile_features(bundle: DataContractBundle, target_rows: pd.DataFrame) -> pd.DataFrame:
    nci = bundle.nci_source_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy()
    profile_rows = []
    for smiles, group in nci.groupby("canonical_smiles", sort=True):
        values = pd.to_numeric(group["y"], errors="coerce").to_numpy(dtype=float)
        profile_rows.append(
            {
                "canonical_smiles": smiles,
                "profile_nci_y_mean": float(np.mean(values)),
                "profile_nci_y_std": float(np.std(values)),
                "profile_nci_y_min": float(np.min(values)),
                "profile_nci_y_max": float(np.max(values)),
                "profile_nci_y_range": float(np.max(values) - np.min(values)),
                "profile_nci_y_q25": float(np.quantile(values, 0.25)),
                "profile_nci_y_q50": float(np.quantile(values, 0.50)),
                "profile_nci_y_q75": float(np.quantile(values, 0.75)),
                "profile_nci_effective_n": float(len(values)),
            }
        )
    profile = pd.DataFrame(profile_rows)
    profile_target = target_rows[["sample_id", "canonical_smiles"]].drop_duplicates().merge(
        profile,
        on="canonical_smiles",
        how="left",
    )
    profile_target = profile_target.rename(
        columns={
            "profile_nci_y_mean": "profile_transfer_nci_y_mean",
            "profile_nci_y_std": "profile_transfer_nci_y_std",
            "profile_nci_y_min": "profile_transfer_nci_y_min",
            "profile_nci_y_max": "profile_transfer_nci_y_max",
            "profile_nci_y_range": "profile_transfer_nci_y_range",
            "profile_nci_y_q25": "profile_transfer_nci_y_q25",
            "profile_nci_y_q50": "profile_transfer_nci_y_q50",
            "profile_nci_y_q75": "profile_transfer_nci_y_q75",
            "profile_nci_effective_n": "profile_transfer_effective_n",
        }
    )
    numeric_cols = [c for c in profile_target.columns if c not in {"sample_id", "canonical_smiles"}]
    profile_target[numeric_cols] = profile_target[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return profile_target[["sample_id", *numeric_cols]]

def _source_rows_for_target(
    source_rows: pd.DataFrame,
    target_smiles: str,
    exclude_identical_source: bool = False,
) -> pd.DataFrame:
    if not exclude_identical_source:
        return source_rows.copy()
    return source_rows.loc[
        source_rows["canonical_smiles"].astype(str).ne(str(target_smiles))
    ].copy()

def _transfer_features(
    target_smiles: list[str],
    source_scores: pd.DataFrame,
    exclude_identical_source: bool = False,
) -> pd.DataFrame:
    source_smiles = source_scores["canonical_smiles"].astype(str).tolist()
    source_mols = [_mol_from_smiles(smiles) for smiles in source_smiles]
    source_fps = [_fingerprint(mol) for mol in source_mols]
    source_y = source_scores["teacher_source_y_cho_wmean"].to_numpy(dtype=float)

    exact = source_scores.set_index("canonical_smiles")
    rows = []
    for smiles in sorted(set(map(str, target_smiles))):
        eligible = source_scores["canonical_smiles"].astype(str).ne(smiles).to_numpy()
        if not exclude_identical_source:
            eligible = np.ones(len(source_scores), dtype=bool)
        mol = _mol_from_smiles(smiles)
        fp = _fingerprint(mol)
        sims = np.zeros(len(source_fps), dtype=float)
        if fp is not None:
            valid = [(idx, source_fp) for idx, source_fp in enumerate(source_fps) if source_fp is not None]
            if valid:
                valid_indices = [idx for idx, _ in valid]
                valid_sims = DataStructs.BulkTanimotoSimilarity(fp, [source_fp for _, source_fp in valid])
                sims[valid_indices] = valid_sims
        order = np.flatnonzero(eligible)
        order = order[np.argsort(-sims[order])[:5]]
        top_sims = sims[order]
        top_y = source_y[order]
        sim_weights = top_sims / top_sims.sum() if top_sims.sum() > 0 else np.ones(len(top_sims)) / len(top_sims)
        row = {
            "canonical_smiles": smiles,
            "teacher_exact_count": 0.0,
            "teacher_exact_y_mean": np.nan,
            "teacher_exact_y_cho_wmean": np.nan,
            "teacher_top1_sim": float(top_sims[0]) if len(top_sims) else 0.0,
            "teacher_topk_y_wmean": float(np.average(top_y, weights=sim_weights)) if len(top_y) else np.nan,
            "teacher_topk_y_std": float(np.std(top_y)) if len(top_y) else np.nan,
            "teacher_topk_sim_mean": float(np.mean(top_sims)) if len(top_sims) else 0.0,
            "teacher_topk_sim_max": float(np.max(top_sims)) if len(top_sims) else 0.0,
            "teacher_topk_n": float(len(top_sims)),
        }
        if not exclude_identical_source and smiles in exact.index:
            source_row = exact.loc[smiles]
            row["teacher_exact_count"] = float(source_row["teacher_source_n_rows"])
            row["teacher_exact_y_mean"] = float(source_row["teacher_source_y_mean"])
            row["teacher_exact_y_cho_wmean"] = float(source_row["teacher_source_y_cho_wmean"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    numeric_cols = [c for c in frame.columns if c != "canonical_smiles"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame

def _cell_id_from_source(source_column: str) -> str:
    if str(source_column).startswith("CHO:"):
        return str(source_column)
    if ":" in str(source_column):
        return str(source_column).split(":", 1)[1]
    return str(source_column)

def _source_teacher_feature_columns(columns: list[str]) -> list[str]:
    return [
        col
        for col in columns
        if col.startswith("rdkit_")
        or col.startswith("maccs_")
        or col.startswith("atom_")
        or col.startswith("domain_")
        or col.startswith("fg_")
        or col.startswith("cell_kegg_")
        or col in {"halogen_count", "heavy_halogen_count", "aromatic_atom_count", "source_teacher_time_h"}
    ]

def _source_teacher_sample_weights(nci: pd.DataFrame) -> np.ndarray:
    if "cho_pathway_weight" not in nci.columns:
        raise ValueError("NCI source teacher rows must include cho_pathway_weight")
    weights = pd.to_numeric(nci["cho_pathway_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0.0:
        return np.ones(len(nci), dtype=float)

    cell_counts = nci.groupby("cell_id")["cell_id"].transform("size").astype(float)
    per_row = weights / cell_counts
    total = float(per_row.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.ones(len(nci), dtype=float)
    normalized = per_row / (total / max(1, len(per_row)))
    return normalized.to_numpy(dtype=float)

def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = np.clip(weights[mask], 0.0, None)
    if len(values) == 0:
        return 0.0
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(values), dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / float(sorted_weights.sum())
    idx = int(np.searchsorted(cumulative, float(quantile), side="left"))
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])

def _weighted_prediction_distribution(predictions: np.ndarray, weights: np.ndarray, prefix: str) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(predictions) & np.isfinite(weights)
    predictions = predictions[mask]
    weights = np.clip(weights[mask], 0.0, None)
    if len(predictions) == 0:
        predictions = np.zeros(1, dtype=float)
        weights = np.ones(1, dtype=float)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(predictions), dtype=float)
    normalized = weights / float(weights.sum())
    wmean = float(np.average(predictions, weights=normalized))
    wstd = float(np.sqrt(np.average((predictions - wmean) ** 2, weights=normalized)))
    q25 = _weighted_quantile(predictions, normalized, 0.25)
    q50 = _weighted_quantile(predictions, normalized, 0.50)
    q75 = _weighted_quantile(predictions, normalized, 0.75)
    min_value = float(np.min(predictions))
    max_value = float(np.max(predictions))
    return {
        f"{prefix}_wmean": wmean,
        f"{prefix}_wstd": wstd,
        f"{prefix}_min": min_value,
        f"{prefix}_max": max_value,
        f"{prefix}_q25": q25,
        f"{prefix}_q50": q50,
        f"{prefix}_q75": q75,
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(max_value - min_value),
    }

def _local_profile_transfer_features(bundle: DataContractBundle, target_rows: pd.DataFrame) -> pd.DataFrame:
    pathway_bundle = build_pathway_bundle(bundle.root_dir)
    cell_weights = pathway_bundle.cho_weights.rename(columns={"weight": "cho_pathway_weight"})
    nci = bundle.nci_source_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy()
    nci = nci.merge(cell_weights, on="cell_id", how="left")
    nci["cho_pathway_weight"] = pd.to_numeric(nci["cho_pathway_weight"], errors="coerce").fillna(0.0)
    if float(nci["cho_pathway_weight"].sum()) <= 0.0:
        nci["cho_pathway_weight"] = 1.0

    source_smiles = nci["canonical_smiles"].astype(str).tolist()
    source_fps = [_fingerprint(_mol_from_smiles(smiles)) for smiles in source_smiles]
    source_y = pd.to_numeric(nci["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    source_time = pd.to_numeric(nci["time_h"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    source_cell_weight = pd.to_numeric(nci["cho_pathway_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    rows = []
    for row in target_rows[["sample_id", "canonical_smiles", "time_h"]].drop_duplicates().itertuples(index=False):
        query_fp = _fingerprint(_mol_from_smiles(str(row.canonical_smiles)))
        similarities = np.zeros(len(source_fps), dtype=float)
        if query_fp is not None:
            valid = [(idx, fp) for idx, fp in enumerate(source_fps) if fp is not None]
            if valid:
                valid_indices = [idx for idx, _ in valid]
                valid_sims = DataStructs.BulkTanimotoSimilarity(query_fp, [fp for _, fp in valid])
                similarities[valid_indices] = valid_sims
        time_delta = np.abs(source_time - float(row.time_h))
        time_weight = np.exp(-time_delta / 24.0)
        weights = np.power(np.clip(similarities, 0.0, None), 2.0) * time_weight * np.clip(source_cell_weight, 0.0, None)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(source_y), dtype=float)
        distribution = _weighted_prediction_distribution(source_y, weights, "local_profile_transfer")
        normalized = weights / float(weights.sum())
        distribution.update(
            {
                "sample_id": row.sample_id,
                "local_profile_transfer_effective_n": float(1.0 / np.sum(normalized**2)),
                "local_profile_transfer_sim_max": float(np.max(similarities)) if len(similarities) else 0.0,
                "local_profile_transfer_sim_mean": float(np.average(similarities, weights=normalized)),
                "local_profile_transfer_time_weight_mean": float(np.average(time_weight, weights=normalized)),
                "local_profile_transfer_cell_weight_mean": float(np.average(source_cell_weight, weights=normalized)),
            }
        )
        rows.append(distribution)

    frame = pd.DataFrame(rows)
    numeric_cols = [col for col in frame.columns if col != "sample_id"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame

def _risk_profile_transfer_features(bundle: DataContractBundle, target_rows: pd.DataFrame) -> pd.DataFrame:
    pathway_bundle = build_pathway_bundle(bundle.root_dir)
    cell_weights = pathway_bundle.cho_weights.rename(columns={"weight": "cho_pathway_weight"})[["cell_id", "cho_pathway_weight"]]
    nci = bundle.nci_source_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy()
    chem = _chem_features(nci["canonical_smiles"].astype(str).tolist())
    nci = nci.merge(chem, on="canonical_smiles", how="left").merge(cell_weights, on="cell_id", how="left")
    nci["cho_pathway_weight"] = pd.to_numeric(nci["cho_pathway_weight"], errors="coerce").fillna(0.0)
    if float(nci["cho_pathway_weight"].sum()) <= 0.0:
        nci["cho_pathway_weight"] = 1.0

    risk_mask = (
        nci["fg_nitro"].astype(float).gt(0.0)
        | nci["fg_trihalomethyl"].astype(float).gt(0.0)
        | nci["fg_quinone"].astype(float).gt(0.0)
        | nci["domain_halogen_atom_fraction"].astype(float).gt(0.0)
        | nci["domain_bromine_iodine_count"].astype(float).gt(0.0)
        | nci["domain_cl_br_i_count"].astype(float).gt(0.0)
    )
    risk_nci = nci[risk_mask].copy()
    if risk_nci.empty:
        risk_nci = nci.copy()

    source_smiles = risk_nci["canonical_smiles"].astype(str).tolist()
    source_fps = [_fingerprint(_mol_from_smiles(smiles)) for smiles in source_smiles]
    source_y = pd.to_numeric(risk_nci["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    source_time = pd.to_numeric(risk_nci["time_h"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    source_cell_weight = pd.to_numeric(risk_nci["cho_pathway_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    source_risk_strength = (
        risk_nci["fg_nitro"].astype(float).fillna(0.0)
        + risk_nci["fg_trihalomethyl"].astype(float).fillna(0.0)
        + risk_nci["fg_quinone"].astype(float).fillna(0.0)
        + risk_nci["domain_halogen_atom_fraction"].astype(float).fillna(0.0)
        + risk_nci["domain_bromine_iodine_count"].astype(float).fillna(0.0)
        + risk_nci["domain_cl_br_i_count"].astype(float).fillna(0.0)
    ).to_numpy(dtype=float)

    rows = []
    target_frame = target_rows[["sample_id", "canonical_smiles", "time_h"]].drop_duplicates().copy()
    target_chem = _chem_features(target_frame["canonical_smiles"].astype(str).tolist())
    target_frame = target_frame.merge(target_chem, on="canonical_smiles", how="left")
    for row in target_frame.itertuples(index=False):
        query_fp = _fingerprint(_mol_from_smiles(str(row.canonical_smiles)))
        similarities = np.zeros(len(source_fps), dtype=float)
        if query_fp is not None:
            valid = [(idx, fp) for idx, fp in enumerate(source_fps) if fp is not None]
            if valid:
                valid_indices = [idx for idx, _ in valid]
                valid_sims = DataStructs.BulkTanimotoSimilarity(query_fp, [fp for _, fp in valid])
                similarities[valid_indices] = valid_sims
        time_delta = np.abs(source_time - float(row.time_h))
        time_weight = np.exp(-time_delta / 24.0)
        weights = np.power(np.clip(similarities, 0.0, None), 2.0) * time_weight * np.clip(source_cell_weight, 0.0, None)
        weights = weights * np.clip(source_risk_strength, 0.0, None)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(source_y), dtype=float)
        normalized = weights / float(weights.sum())
        distribution = _weighted_prediction_distribution(source_y, weights, "risk_profile_transfer")
        distribution.update(
            {
                "sample_id": row.sample_id,
                "risk_profile_transfer_effective_n": float(1.0 / np.sum(normalized**2)),
                "risk_profile_transfer_same_risk_weight_fraction": float(normalized[source_risk_strength > 0].sum())
                if np.any(source_risk_strength > 0)
                else 0.0,
                "risk_profile_transfer_sim_max": float(np.max(similarities)) if len(similarities) else 0.0,
                "risk_profile_transfer_time_weight_mean": float(np.average(time_weight, weights=normalized)),
            }
        )
        rows.append(distribution)
    frame = pd.DataFrame(rows)
    numeric_cols = [col for col in frame.columns if col != "sample_id"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame

def _risk_adjusted_transfer_features(bundle: DataContractBundle, target_rows: pd.DataFrame) -> pd.DataFrame:
    target = target_rows[["sample_id", "canonical_smiles"]].drop_duplicates().copy()
    local_profile = _local_profile_transfer_features(bundle, target_rows).copy()
    risk_profile = _risk_profile_transfer_features(bundle, target_rows).copy()
    teacher = _transfer_features(target["canonical_smiles"].astype(str).tolist(), _source_scores(bundle)).copy()

    frame = (
        target.merge(local_profile, on="sample_id", how="left")
        .merge(risk_profile, on="sample_id", how="left")
        .merge(
            teacher[
                [
                    "canonical_smiles",
                    "teacher_topk_y_wmean",
                    "teacher_topk_y_std",
                    "teacher_topk_sim_mean",
                    "teacher_topk_sim_max",
                ]
            ],
            on="canonical_smiles",
            how="left",
        )
    )
    local_wmean = pd.to_numeric(frame["local_profile_transfer_wmean"], errors="coerce").fillna(0.0)
    local_effective_n = pd.to_numeric(frame["local_profile_transfer_effective_n"], errors="coerce").fillna(0.0)
    risk_wmean = pd.to_numeric(frame["risk_profile_transfer_wmean"], errors="coerce").fillna(0.0)
    risk_effective_n = pd.to_numeric(frame["risk_profile_transfer_effective_n"], errors="coerce").fillna(0.0)
    teacher_wmean = pd.to_numeric(frame["teacher_topk_y_wmean"], errors="coerce").fillna(local_wmean)
    teacher_std = pd.to_numeric(frame["teacher_topk_y_std"], errors="coerce").fillna(0.0)
    teacher_sim = pd.to_numeric(frame["teacher_topk_sim_mean"], errors="coerce").fillna(0.0)
    risk_fraction = pd.to_numeric(frame["risk_profile_transfer_same_risk_weight_fraction"], errors="coerce").fillna(0.0)
    risk_gate = np.clip(0.55 * risk_fraction + 0.45 * np.tanh(risk_effective_n / 12.0), 0.0, 1.0)
    suppressed_wmean = local_wmean + (1.0 - risk_gate) * (teacher_wmean - local_wmean)
    adjusted_wmean = (1.0 - risk_gate) * suppressed_wmean + risk_gate * risk_wmean
    frame["risk_adjusted_transfer_risk_gate"] = risk_gate
    frame["risk_adjusted_transfer_delta_from_teacher"] = teacher_wmean - local_wmean
    frame["risk_adjusted_transfer_suppressed_wmean"] = suppressed_wmean
    frame["risk_adjusted_transfer_wmean"] = adjusted_wmean
    frame["risk_adjusted_transfer_signed_delta"] = adjusted_wmean - teacher_wmean
    frame["risk_adjusted_transfer_uncertainty"] = 1.0 / (1.0 + 0.5 * (local_effective_n + risk_effective_n))
    frame["risk_adjusted_transfer_teacher_std"] = teacher_std
    frame["risk_adjusted_transfer_teacher_sim_mean"] = teacher_sim
    numeric_cols = [col for col in frame.columns if col not in {"sample_id", "canonical_smiles"}]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame[
        [
            "sample_id",
            "risk_adjusted_transfer_risk_gate",
            "risk_adjusted_transfer_delta_from_teacher",
            "risk_adjusted_transfer_suppressed_wmean",
            "risk_adjusted_transfer_wmean",
            "risk_adjusted_transfer_signed_delta",
            "risk_adjusted_transfer_uncertainty",
            "risk_adjusted_transfer_teacher_std",
            "risk_adjusted_transfer_teacher_sim_mean",
        ]
    ]

def _conditioned_transfer_features(bundle: DataContractBundle, target_rows: pd.DataFrame) -> pd.DataFrame:
    target = target_rows[["sample_id", "canonical_smiles", "time_h"]].drop_duplicates().copy()
    local_profile = _local_profile_transfer_features(bundle, target_rows).copy()
    risk_profile = _risk_profile_transfer_features(bundle, target_rows).copy()
    teacher = _transfer_features(target["canonical_smiles"].astype(str).tolist(), _source_scores(bundle)).copy()
    target_chem = _chem_features(target["canonical_smiles"].astype(str).tolist())

    frame = (
        target.merge(target_chem, on="canonical_smiles", how="left")
        .merge(local_profile, on="sample_id", how="left")
        .merge(risk_profile, on="sample_id", how="left")
        .merge(
            teacher[
                [
                    "canonical_smiles",
                    "teacher_topk_y_wmean",
                    "teacher_topk_y_std",
                    "teacher_topk_sim_mean",
                    "teacher_topk_sim_max",
                ]
            ],
            on="canonical_smiles",
            how="left",
        )
    )
    local_wmean = pd.to_numeric(frame["local_profile_transfer_wmean"], errors="coerce").fillna(0.0)
    local_q25 = pd.to_numeric(frame["local_profile_transfer_q25"], errors="coerce").fillna(0.0)
    local_q50 = pd.to_numeric(frame["local_profile_transfer_q50"], errors="coerce").fillna(local_wmean)
    local_effective_n = pd.to_numeric(frame["local_profile_transfer_effective_n"], errors="coerce").fillna(0.0)
    risk_wmean = pd.to_numeric(frame["risk_profile_transfer_wmean"], errors="coerce").fillna(0.0)
    risk_q25 = pd.to_numeric(frame["risk_profile_transfer_q25"], errors="coerce").fillna(risk_wmean)
    risk_effective_n = pd.to_numeric(frame["risk_profile_transfer_effective_n"], errors="coerce").fillna(0.0)
    teacher_wmean = pd.to_numeric(frame["teacher_topk_y_wmean"], errors="coerce").fillna(local_wmean)
    teacher_std = pd.to_numeric(frame["teacher_topk_y_std"], errors="coerce").fillna(0.0)
    teacher_sim = pd.to_numeric(frame["teacher_topk_sim_mean"], errors="coerce").fillna(0.0)
    risk_fraction = pd.to_numeric(frame["risk_profile_transfer_same_risk_weight_fraction"], errors="coerce").fillna(0.0)
    time72_mask = pd.to_numeric(frame["time_h"], errors="coerce").fillna(0.0).astype(int).eq(72).astype(float)
    risk_indicator = np.clip(
        0.35 * frame["fg_nitro"].astype(float).fillna(0.0)
        + 0.25 * frame["fg_trihalomethyl"].astype(float).fillna(0.0)
        + 0.20 * frame["domain_halogen_atom_fraction"].astype(float).fillna(0.0)
        + 0.10 * frame["domain_bromine_iodine_count"].astype(float).fillna(0.0)
        + 0.10 * time72_mask,
        0.0,
        1.0,
    )
    risk_gate = np.clip(0.40 * risk_fraction + 0.30 * np.tanh(risk_effective_n / 12.0) + 0.20 * risk_indicator + 0.10 * time72_mask, 0.0, 1.0)
    clean_mid = 0.70 * teacher_wmean + 0.30 * local_q50
    risk_low_tail = 0.70 * risk_wmean + 0.30 * risk_q25
    conditioned_wmean = (1.0 - risk_gate) * clean_mid + risk_gate * risk_low_tail
    frame["conditioned_transfer_risk_gate"] = risk_gate
    frame["conditioned_transfer_time72_risk_gate"] = risk_gate * time72_mask
    frame["conditioned_transfer_risk_indicator"] = risk_indicator
    frame["conditioned_transfer_clean_mid"] = clean_mid
    frame["conditioned_transfer_risk_low_tail"] = risk_low_tail
    frame["conditioned_transfer_wmean"] = conditioned_wmean
    frame["conditioned_transfer_delta_from_teacher"] = conditioned_wmean - teacher_wmean
    frame["conditioned_transfer_uncertainty"] = 1.0 / (1.0 + 0.4 * (local_effective_n + risk_effective_n) + 0.3 * teacher_std + 0.2 * teacher_sim)
    numeric_cols = [col for col in frame.columns if col not in {"sample_id", "canonical_smiles"}]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame[
        [
            "sample_id",
            "conditioned_transfer_risk_gate",
            "conditioned_transfer_time72_risk_gate",
            "conditioned_transfer_risk_indicator",
            "conditioned_transfer_clean_mid",
            "conditioned_transfer_risk_low_tail",
            "conditioned_transfer_wmean",
            "conditioned_transfer_delta_from_teacher",
            "conditioned_transfer_uncertainty",
        ]
    ]

def _disagreement_transfer_features(target_rows: pd.DataFrame, transfer: pd.DataFrame, source_teacher: pd.DataFrame) -> pd.DataFrame:
    target = target_rows[["sample_id", "canonical_smiles", "time_h"]].drop_duplicates().copy()
    transfer_subset = transfer[
        [
            "canonical_smiles",
            "teacher_topk_y_wmean",
            "teacher_topk_y_std",
            "teacher_topk_sim_mean",
            "teacher_topk_sim_max",
        ]
    ].copy()
    source_subset = source_teacher[
        [
            "canonical_smiles",
            "time_h",
            "source_teacher_pred",
            "source_teacher_weighted_pred",
            "source_teacher_cell_wmean",
            "source_teacher_weighted_cell_wmean",
            "source_teacher_cell_wstd",
            "source_teacher_weighted_cell_wstd",
        ]
    ].copy()
    frame = target.merge(transfer_subset, on="canonical_smiles", how="left").merge(
        source_subset,
        on=["canonical_smiles", "time_h"],
        how="left",
    )
    teacher = pd.to_numeric(frame["teacher_topk_y_wmean"], errors="coerce").fillna(0.0)
    teacher_std = pd.to_numeric(frame["teacher_topk_y_std"], errors="coerce").fillna(0.0)
    teacher_sim = pd.to_numeric(frame["teacher_topk_sim_mean"], errors="coerce").fillna(0.0)
    source = pd.to_numeric(frame["source_teacher_pred"], errors="coerce").fillna(teacher)
    weighted_source = pd.to_numeric(frame["source_teacher_weighted_pred"], errors="coerce").fillna(source)
    cell = pd.to_numeric(frame["source_teacher_cell_wmean"], errors="coerce").fillna(source)
    weighted_cell = pd.to_numeric(frame["source_teacher_weighted_cell_wmean"], errors="coerce").fillna(weighted_source)
    cell_std = pd.to_numeric(frame["source_teacher_cell_wstd"], errors="coerce").fillna(0.0)
    weighted_cell_std = pd.to_numeric(frame["source_teacher_weighted_cell_wstd"], errors="coerce").fillna(0.0)
    frame["disagreement_transfer_source_minus_teacher"] = source - teacher
    frame["disagreement_transfer_weighted_source_minus_teacher"] = weighted_source - teacher
    frame["disagreement_transfer_cell_minus_teacher"] = cell - teacher
    frame["disagreement_transfer_weighted_cell_minus_teacher"] = weighted_cell - teacher
    frame["disagreement_transfer_source_cell_gap_signed"] = weighted_source - weighted_cell
    frame["disagreement_transfer_abs_source_minus_teacher"] = (source - teacher).abs()
    frame["disagreement_transfer_abs_weighted_source_minus_teacher"] = (weighted_source - teacher).abs()
    frame["disagreement_transfer_uncertainty"] = teacher_std + 0.5 * (cell_std + weighted_cell_std) + (1.0 - teacher_sim)
    numeric_cols = [col for col in frame.columns if col not in {"sample_id", "canonical_smiles", "time_h"}]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame[
        [
            "sample_id",
            "disagreement_transfer_source_minus_teacher",
            "disagreement_transfer_weighted_source_minus_teacher",
            "disagreement_transfer_cell_minus_teacher",
            "disagreement_transfer_weighted_cell_minus_teacher",
            "disagreement_transfer_source_cell_gap_signed",
            "disagreement_transfer_abs_source_minus_teacher",
            "disagreement_transfer_abs_weighted_source_minus_teacher",
            "disagreement_transfer_uncertainty",
        ]
    ]

def _residualized_transfer_features(bundle: DataContractBundle, target_rows: pd.DataFrame) -> pd.DataFrame:
    target = target_rows[["sample_id", "canonical_smiles", "time_h"]].drop_duplicates().copy()
    local_profile = _local_profile_transfer_features(bundle, target_rows).copy()
    risk_profile = _risk_profile_transfer_features(bundle, target_rows).copy()
    profile = _response_profile_features(bundle, target_rows).copy()
    transfer = _transfer_features(target["canonical_smiles"].astype(str).tolist(), _source_scores(bundle)).copy()
    chem = _chem_features(target["canonical_smiles"].astype(str).tolist())
    source_teacher = _source_domain_teacher_features(bundle, target_rows, chem, transfer).copy()
    conditioned = _conditioned_transfer_features(bundle, target_rows).copy()
    risk_adjusted = _risk_adjusted_transfer_features(bundle, target_rows).copy()

    frame = (
        target.merge(local_profile, on="sample_id", how="left")
        .merge(risk_profile, on="sample_id", how="left")
        .merge(profile, on="sample_id", how="left")
        .merge(conditioned, on="sample_id", how="left")
        .merge(risk_adjusted, on="sample_id", how="left")
        .merge(
            source_teacher[
                [
                    "canonical_smiles",
                    "time_h",
                    "source_teacher_weighted_pred",
                    "source_teacher_weighted_cell_wmean",
                    "source_teacher_weighted_cell_wstd",
                ]
            ],
            on=["canonical_smiles", "time_h"],
            how="left",
        )
        .merge(
            transfer[
                [
                    "canonical_smiles",
                    "teacher_topk_y_wmean",
                    "teacher_topk_y_std",
                    "teacher_topk_sim_mean",
                    "teacher_topk_sim_max",
                    "teacher_topk_n",
                ]
            ],
            on="canonical_smiles",
            how="left",
        )
    )

    local_wmean = pd.to_numeric(frame["local_profile_transfer_wmean"], errors="coerce").fillna(0.0)
    local_wstd = pd.to_numeric(frame["local_profile_transfer_wstd"], errors="coerce").fillna(0.0)
    local_eff = pd.to_numeric(frame["local_profile_transfer_effective_n"], errors="coerce").fillna(0.0)
    profile_mean = pd.to_numeric(frame["profile_transfer_nci_y_mean"], errors="coerce").fillna(local_wmean)
    profile_q25 = pd.to_numeric(frame["profile_transfer_nci_y_q25"], errors="coerce").fillna(profile_mean)
    profile_q75 = pd.to_numeric(frame["profile_transfer_nci_y_q75"], errors="coerce").fillna(profile_mean)
    profile_std = pd.to_numeric(frame["profile_transfer_nci_y_std"], errors="coerce").fillna(0.0)
    profile_eff = pd.to_numeric(frame["profile_transfer_effective_n"], errors="coerce").fillna(0.0)
    teacher = pd.to_numeric(frame["teacher_topk_y_wmean"], errors="coerce").fillna(profile_mean)
    teacher_std = pd.to_numeric(frame["teacher_topk_y_std"], errors="coerce").fillna(0.0)
    teacher_sim = pd.to_numeric(frame["teacher_topk_sim_mean"], errors="coerce").fillna(0.0)
    source_pred = pd.to_numeric(frame["source_teacher_weighted_pred"], errors="coerce").fillna(teacher)
    source_cell = pd.to_numeric(frame["source_teacher_weighted_cell_wmean"], errors="coerce").fillna(source_pred)
    source_cell_std = pd.to_numeric(frame["source_teacher_weighted_cell_wstd"], errors="coerce").fillna(0.0)
    risk_fraction = pd.to_numeric(frame["risk_profile_transfer_same_risk_weight_fraction"], errors="coerce").fillna(0.0)
    risk_eff = pd.to_numeric(frame["risk_profile_transfer_effective_n"], errors="coerce").fillna(0.0)

    conditioned_wmean = pd.to_numeric(frame["conditioned_transfer_wmean"], errors="coerce").fillna(local_wmean)
    risk_adjusted_wmean = pd.to_numeric(frame["risk_adjusted_transfer_wmean"], errors="coerce").fillna(local_wmean)
    local_residual = conditioned_wmean - local_wmean
    source_residual = source_pred - teacher
    profile_residual = profile_mean - local_wmean
    risk_residual = risk_adjusted_wmean - local_wmean
    consensus_residual = 0.40 * local_residual + 0.25 * risk_residual + 0.20 * profile_residual + 0.15 * source_residual
    alignment = 1.0 / (
        1.0
        + np.abs(local_residual - risk_residual)
        + 0.5 * np.abs(profile_residual - source_residual)
        + 0.25 * np.abs(teacher - source_cell)
    )
    stability = np.clip(
        0.35 * np.tanh(local_eff / 10.0)
        + 0.30 * np.tanh(risk_eff / 10.0)
        + 0.20 * np.tanh(profile_eff / 10.0)
        + 0.15 * np.clip(teacher_sim, 0.0, 1.0),
        0.0,
        1.0,
    )
    risk_gate = np.clip(0.40 * risk_fraction + 0.35 * stability + 0.25 * alignment, 0.0, 1.0)
    residualized_wmean = local_wmean + risk_gate * consensus_residual
    suppressed_wmean = 0.65 * local_wmean + 0.35 * residualized_wmean
    uncertainty = 1.0 / (1.0 + teacher_std + profile_std + local_wstd + source_cell_std + 0.5 * np.abs(source_residual))

    frame["residualized_transfer_local_residual"] = local_residual
    frame["residualized_transfer_source_residual"] = source_residual
    frame["residualized_transfer_profile_residual"] = profile_residual
    frame["residualized_transfer_risk_residual"] = risk_residual
    frame["residualized_transfer_consensus_residual"] = consensus_residual
    frame["residualized_transfer_risk_gate"] = risk_gate
    frame["residualized_transfer_alignment"] = alignment
    frame["residualized_transfer_suppressed_wmean"] = suppressed_wmean
    frame["residualized_transfer_wmean"] = residualized_wmean
    frame["residualized_transfer_uncertainty"] = uncertainty

    numeric_cols = [col for col in frame.columns if col not in {"sample_id", "canonical_smiles", "time_h"}]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame[
        [
            "sample_id",
            "residualized_transfer_local_residual",
            "residualized_transfer_source_residual",
            "residualized_transfer_profile_residual",
            "residualized_transfer_risk_residual",
            "residualized_transfer_consensus_residual",
            "residualized_transfer_risk_gate",
            "residualized_transfer_alignment",
            "residualized_transfer_suppressed_wmean",
            "residualized_transfer_wmean",
            "residualized_transfer_uncertainty",
        ]
    ]

def _calibrated_transfer_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "canonical_smiles",
        "time_h",
        "teacher_topk_y_wmean",
        "teacher_topk_y_std",
        "teacher_topk_sim_mean",
        "local_profile_transfer_wmean",
        "local_profile_transfer_effective_n",
        "conditioned_transfer_wmean",
        "conditioned_transfer_uncertainty",
        "residualized_transfer_wmean",
        "residualized_transfer_uncertainty",
        "source_teacher_weighted_pred",
        "source_teacher_weighted_residual_abs_proxy",
        "source_teacher_weighted_cell_cho_gap",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Calibrated transfer requires merged feature columns: {sorted(missing)}")

    view = frame[["sample_id"]].drop_duplicates().copy()
    teacher = pd.to_numeric(frame["teacher_topk_y_wmean"], errors="coerce").fillna(0.0)
    teacher_std = pd.to_numeric(frame["teacher_topk_y_std"], errors="coerce").fillna(0.0)
    teacher_sim = pd.to_numeric(frame["teacher_topk_sim_mean"], errors="coerce").fillna(0.0)
    local = pd.to_numeric(frame["local_profile_transfer_wmean"], errors="coerce").fillna(teacher)
    conditioned_wmean = pd.to_numeric(frame["conditioned_transfer_wmean"], errors="coerce").fillna(local)
    residualized_wmean = pd.to_numeric(frame["residualized_transfer_wmean"], errors="coerce").fillna(local)
    source_weighted = pd.to_numeric(frame["source_teacher_weighted_pred"], errors="coerce").fillna(teacher)
    local_eff = pd.to_numeric(frame["local_profile_transfer_effective_n"], errors="coerce").fillna(0.0)
    conditioned_uncertainty = pd.to_numeric(frame["conditioned_transfer_uncertainty"], errors="coerce").fillna(1.0)
    residualized_uncertainty = pd.to_numeric(frame["residualized_transfer_uncertainty"], errors="coerce").fillna(1.0)
    source_residual = pd.to_numeric(frame["source_teacher_weighted_residual_abs_proxy"], errors="coerce").fillna(0.0)
    source_gap = pd.to_numeric(frame["source_teacher_weighted_cell_cho_gap"], errors="coerce").fillna(0.0)

    sources = np.vstack(
        [
            teacher.to_numpy(dtype=float),
            local.to_numpy(dtype=float),
            conditioned_wmean.to_numpy(dtype=float),
            residualized_wmean.to_numpy(dtype=float),
            source_weighted.to_numpy(dtype=float),
        ]
    )
    consensus = np.mean(sources, axis=0)
    disagreement = np.std(sources, axis=0)
    reliability = np.clip(
        0.35 * np.clip(teacher_sim.to_numpy(dtype=float), 0.0, 1.0)
        + 0.25 * np.tanh(local_eff.to_numpy(dtype=float) / 10.0)
        + 0.20 / (1.0 + np.clip(disagreement, 0.0, None))
        + 0.20 / (1.0 + conditioned_uncertainty.to_numpy(dtype=float) + residualized_uncertainty.to_numpy(dtype=float)),
        0.0,
        1.0,
    )
    uncertainty = 1.0 / (
        1.0
        + reliability
        + 1.0 / (1.0 + teacher_std.to_numpy(dtype=float))
        + 1.0 / (1.0 + source_residual.to_numpy(dtype=float) + source_gap.to_numpy(dtype=float))
    )
    centered_wmean = consensus - np.median(consensus)
    wmean = reliability * consensus + (1.0 - reliability) * (0.50 * local.to_numpy(dtype=float) + 0.50 * teacher.to_numpy(dtype=float))

    view["calibrated_transfer_wmean"] = wmean
    view["calibrated_transfer_centered_wmean"] = centered_wmean
    view["calibrated_transfer_consensus"] = consensus
    view["calibrated_transfer_delta_from_teacher"] = wmean - teacher.to_numpy(dtype=float)
    view["calibrated_transfer_disagreement"] = disagreement
    view["calibrated_transfer_reliability"] = reliability
    view["calibrated_transfer_uncertainty"] = uncertainty

    numeric_cols = [col for col in view.columns if col != "sample_id"]
    view[numeric_cols] = view[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return view[
        [
            "sample_id",
            "calibrated_transfer_wmean",
            "calibrated_transfer_centered_wmean",
            "calibrated_transfer_consensus",
            "calibrated_transfer_delta_from_teacher",
            "calibrated_transfer_disagreement",
            "calibrated_transfer_reliability",
            "calibrated_transfer_uncertainty",
        ]
    ]

def _make_source_teacher() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.50,
        random_state=20260605,
        n_jobs=-1,
    )

def _source_domain_teacher_features(
    bundle: DataContractBundle,
    target_rows: pd.DataFrame,
    chem_features: pd.DataFrame,
    transfer_features: pd.DataFrame,
    exclude_identical_source: bool = False,
    _pathway_bundle=None,
) -> pd.DataFrame:
    pathway_bundle = _pathway_bundle or build_pathway_bundle(bundle.root_dir)
    if exclude_identical_source:
        source_smiles = set(bundle.nci_source_rows["canonical_smiles"].astype(str))
        target_smiles = target_rows["canonical_smiles"].astype(str)
        outputs = []
        unmatched = target_rows.loc[~target_smiles.isin(source_smiles)].copy()
        if not unmatched.empty:
            outputs.append(
                _source_domain_teacher_features(
                    bundle,
                    unmatched,
                    chem_features,
                    transfer_features,
                    _pathway_bundle=pathway_bundle,
                )
            )
        for smiles in sorted(set(target_smiles) & source_smiles):
            target_part = target_rows.loc[target_smiles.eq(smiles)].copy()
            source_part = _source_rows_for_target(
                bundle.nci_source_rows,
                smiles,
                exclude_identical_source=True,
            )
            if source_part.empty:
                raise ValueError(f"leave-identical control removed every NCI source row for {smiles}")
            outputs.append(
                _source_domain_teacher_features(
                    replace(bundle, nci_source_rows=source_part),
                    target_part,
                    chem_features,
                    transfer_features.loc[
                        transfer_features["canonical_smiles"].astype(str).eq(smiles)
                    ].copy(),
                    _pathway_bundle=pathway_bundle,
                )
            )
        return pd.concat(outputs, ignore_index=True)

    pathway = pathway_bundle.pathway_activity.copy()
    pathway["cell_id"] = pathway["source_column"].map(_cell_id_from_source)
    pathway_cols = [col for col in pathway.columns if col.startswith("kegg_")]
    renamed_pathway_cols = {col: f"cell_{col}" for col in pathway_cols}
    pathway = pathway[["cell_id", *pathway_cols]].rename(columns=renamed_pathway_cols)
    cell_feature_cols = list(renamed_pathway_cols.values())
    weights = pathway_bundle.cho_weights.rename(columns={"weight": "cho_pathway_weight"})

    nci = bundle.nci_source_rows[["canonical_smiles", "cell_id", "time_h", "y"]].copy()
    nci = (
        nci.merge(weights, on="cell_id", how="left")
        .merge(chem_features, on="canonical_smiles", how="left")
        .merge(pathway, on="cell_id", how="left")
    )
    nci["cho_pathway_weight"] = pd.to_numeric(nci["cho_pathway_weight"], errors="coerce").fillna(0.0)
    cho_pathway = pathway[pathway["cell_id"].eq("CHO:consensus")]
    if cho_pathway.empty:
        raise ValueError("Missing CHO:consensus pathway features for source-domain teacher")
    cho_values = cho_pathway[cell_feature_cols].iloc[0].to_dict()

    target = target_rows[["canonical_smiles", "time_h"]].drop_duplicates().copy()
    target = target.merge(chem_features, on="canonical_smiles", how="left")
    cho_pathway_features = pd.DataFrame(
        np.repeat([[float(cho_values[column]) for column in cell_feature_cols]], len(target), axis=0),
        columns=cell_feature_cols,
        index=target.index,
    )
    target = pd.concat([target, cho_pathway_features], axis=1).copy()

    nci_time = pd.to_numeric(nci["time_h"], errors="coerce").rename("source_teacher_time_h")
    target_time = pd.to_numeric(target["time_h"], errors="coerce").rename("source_teacher_time_h")
    nci = pd.concat([nci.copy(), nci_time], axis=1).copy()
    target = pd.concat([target.copy(), target_time], axis=1).copy()
    teacher_feature_cols = _source_teacher_feature_columns(list(nci.columns))
    nci[teacher_feature_cols] = nci[teacher_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target[teacher_feature_cols] = target[teacher_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    teacher = _make_source_teacher()
    teacher.fit(nci[teacher_feature_cols].to_numpy(dtype=float), nci["y"].to_numpy(dtype=float))
    source_teacher_pred = pd.Series(
        teacher.predict(target[teacher_feature_cols].to_numpy(dtype=float)),
        index=target.index,
        name="source_teacher_pred",
    )

    weighted_teacher = _make_source_teacher()
    weighted_teacher.fit(
        nci[teacher_feature_cols].to_numpy(dtype=float),
        nci["y"].to_numpy(dtype=float),
        sample_weight=_source_teacher_sample_weights(nci),
    )
    source_teacher_weighted_pred = pd.Series(
        weighted_teacher.predict(target[teacher_feature_cols].to_numpy(dtype=float)),
        index=target.index,
        name="source_teacher_weighted_pred",
    )
    target = pd.concat([target, source_teacher_pred, source_teacher_weighted_pred], axis=1).copy()

    nci_cell_context = weights.merge(pathway, on="cell_id", how="inner").sort_values("cell_id").reset_index(drop=True)
    if nci_cell_context.empty:
        raise ValueError("Missing NCI cell pathway contexts for source-domain teacher distribution")
    cell_weights = nci_cell_context["cho_pathway_weight"].to_numpy(dtype=float)
    repeated = target.loc[target.index.repeat(len(nci_cell_context))].reset_index(drop=True)
    repeated_cell_values = np.tile(
        nci_cell_context[cell_feature_cols].to_numpy(dtype=float),
        (len(target), 1),
    )
    repeated[cell_feature_cols] = repeated_cell_values
    repeated[teacher_feature_cols] = repeated[teacher_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cell_pred = teacher.predict(repeated[teacher_feature_cols].to_numpy(dtype=float))
    weighted_cell_pred = weighted_teacher.predict(repeated[teacher_feature_cols].to_numpy(dtype=float))
    tiled_weights = np.tile(cell_weights, len(target))
    distribution_rows = []
    n_cells = len(nci_cell_context)
    for target_idx in range(len(target)):
        start = target_idx * n_cells
        end = start + n_cells
        stats = _weighted_prediction_distribution(cell_pred[start:end], tiled_weights[start:end], "source_teacher_cell")
        stats.update(
            _weighted_prediction_distribution(
                weighted_cell_pred[start:end],
                tiled_weights[start:end],
                "source_teacher_weighted_cell",
            )
        )
        distribution_rows.append(stats)
    distribution = pd.DataFrame(distribution_rows, index=target.index)
    target = pd.concat([target, distribution], axis=1).copy()
    target["source_teacher_cell_cho_gap"] = (
        target["source_teacher_pred"] - target["source_teacher_cell_wmean"]
    ).abs()
    target["source_teacher_weighted_cell_cho_gap"] = (
        target["source_teacher_weighted_pred"] - target["source_teacher_weighted_cell_wmean"]
    ).abs()

    transfer_subset = transfer_features[["canonical_smiles", "teacher_topk_y_wmean"]].copy()
    target = target.merge(transfer_subset, on="canonical_smiles", how="left").copy()
    source_teacher_residual_abs_proxy = (
        target["source_teacher_pred"] - target["teacher_topk_y_wmean"].fillna(target["source_teacher_pred"])
    ).abs().rename("source_teacher_residual_abs_proxy")
    source_teacher_weighted_residual_abs_proxy = (
        target["source_teacher_weighted_pred"]
        - target["teacher_topk_y_wmean"].fillna(target["source_teacher_weighted_pred"])
    ).abs().rename("source_teacher_weighted_residual_abs_proxy")
    target = pd.concat([target, source_teacher_residual_abs_proxy, source_teacher_weighted_residual_abs_proxy], axis=1).copy()
    return target[
        [
            "canonical_smiles",
            "time_h",
            "source_teacher_pred",
            "source_teacher_residual_abs_proxy",
            "source_teacher_weighted_pred",
            "source_teacher_weighted_residual_abs_proxy",
            "source_teacher_cell_wmean",
            "source_teacher_cell_wstd",
            "source_teacher_cell_min",
            "source_teacher_cell_max",
            "source_teacher_cell_q25",
            "source_teacher_cell_q50",
            "source_teacher_cell_q75",
            "source_teacher_cell_iqr",
            "source_teacher_cell_range",
            "source_teacher_cell_cho_gap",
            "source_teacher_weighted_cell_wmean",
            "source_teacher_weighted_cell_wstd",
            "source_teacher_weighted_cell_min",
            "source_teacher_weighted_cell_max",
            "source_teacher_weighted_cell_q25",
            "source_teacher_weighted_cell_q50",
            "source_teacher_weighted_cell_q75",
            "source_teacher_weighted_cell_iqr",
            "source_teacher_weighted_cell_range",
            "source_teacher_weighted_cell_cho_gap",
        ]
    ]

def _pathway_transfer_features(bundle: DataContractBundle, target_rows: pd.DataFrame) -> pd.DataFrame:
    from .mechanistic_analysis import build_pathway_transfer_features, select_interaction_pathways

    pathway_bundle = build_pathway_bundle(bundle.root_dir)
    top_pathways = select_interaction_pathways(pathway_bundle.pathway_activity)
    return build_pathway_transfer_features(
        target_rows,
        bundle.nci_source_rows,
        pathway_bundle.pathway_activity,
        top_pathways=top_pathways,
    )

def build_model_matrix(
    bundle: DataContractBundle,
    split: FrozenSplit,
    use_cache: bool = True,
    exclude_identical_source: bool = False,
) -> ModelMatrix:
    del use_cache
    all_rows = pd.concat(
        [split.train, split.test, split.validation, split.experimental],
        ignore_index=True,
        sort=False,
    )
    smiles_values = all_rows["canonical_smiles"].astype(str).tolist()
    chem = _chem_features(smiles_values)
    nci_smiles = bundle.nci_source_rows["canonical_smiles"].astype(str).tolist()
    chem_for_teacher = _chem_features(smiles_values + nci_smiles)
    transfer = _transfer_features(
        smiles_values,
        _source_scores(bundle),
        exclude_identical_source=exclude_identical_source,
    )
    source_teacher = _source_domain_teacher_features(
        bundle,
        all_rows,
        chem_for_teacher,
        transfer,
        exclude_identical_source=exclude_identical_source,
    )
    pathway_transfer = _pathway_transfer_features(bundle, all_rows)
    profile_transfer = _response_profile_features(bundle, all_rows)
    local_profile_transfer = _local_profile_transfer_features(bundle, all_rows)
    risk_profile_transfer = _risk_profile_transfer_features(bundle, all_rows)
    risk_adjusted_transfer = _risk_adjusted_transfer_features(bundle, all_rows)
    conditioned_transfer = _conditioned_transfer_features(bundle, all_rows)
    disagreement_transfer = _disagreement_transfer_features(all_rows, transfer, source_teacher)
    residualized_transfer = _residualized_transfer_features(bundle, all_rows)
    frame = (
        all_rows.merge(chem, on="canonical_smiles", how="left")
        .merge(transfer, on="canonical_smiles", how="left")
        .merge(source_teacher, on=["canonical_smiles", "time_h"], how="left")
        .merge(pathway_transfer, on="sample_id", how="left")
        .merge(profile_transfer, on="sample_id", how="left")
        .merge(local_profile_transfer, on="sample_id", how="left")
        .merge(risk_profile_transfer, on="sample_id", how="left")
        .merge(risk_adjusted_transfer, on="sample_id", how="left")
        .merge(conditioned_transfer, on="sample_id", how="left")
        .merge(disagreement_transfer, on="sample_id", how="left")
        .merge(residualized_transfer, on="sample_id", how="left")
    )
    frame = frame.copy()
    frame["time_h_numeric"] = pd.to_numeric(frame["time_h"], errors="coerce")
    frame["time_h_log1p"] = np.log1p(frame["time_h_numeric"])
    time_dummies = pd.get_dummies(frame["time_h_numeric"].astype(int), prefix="time_h", dtype=float)
    for expected_time in [3, 6, 12, 24, 48, 72]:
        column = f"time_h_{expected_time}"
        if column not in time_dummies:
            time_dummies[column] = 0.0
    frame = pd.concat([frame, time_dummies], axis=1)
    pathway_transfer_cols = [col for col in frame.columns if col.startswith("pathway_transfer_")]
    if pathway_transfer_cols:
        pathway_transfer_values = frame[pathway_transfer_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        frame["pathway_transfer_summary_mean"] = pathway_transfer_values.mean(axis=1)
        frame["pathway_transfer_summary_std"] = pathway_transfer_values.std(axis=1).fillna(0.0)
        frame["pathway_transfer_summary_max"] = pathway_transfer_values.max(axis=1)
    else:
        frame["pathway_transfer_summary_mean"] = 0.0
        frame["pathway_transfer_summary_std"] = 0.0
        frame["pathway_transfer_summary_max"] = 0.0
    time48_mask = frame["time_h_numeric"].astype(int).eq(48)
    frame["pathway_transfer_time48_mean"] = np.where(time48_mask, frame["pathway_transfer_summary_mean"], 0.0)
    frame["pathway_transfer_time48_std"] = np.where(time48_mask, frame["pathway_transfer_summary_std"], 0.0)
    frame["pathway_transfer_time48_max"] = np.where(time48_mask, frame["pathway_transfer_summary_max"], 0.0)
    profile_transfer_cols = [col for col in frame.columns if col.startswith("profile_transfer_")]
    if profile_transfer_cols:
        profile_transfer_values = frame[profile_transfer_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        frame["profile_transfer_summary_mean"] = profile_transfer_values.mean(axis=1)
        frame["profile_transfer_summary_std"] = profile_transfer_values.std(axis=1).fillna(0.0)
        frame["profile_transfer_summary_max"] = profile_transfer_values.max(axis=1)
    else:
        frame["profile_transfer_summary_mean"] = 0.0
        frame["profile_transfer_summary_std"] = 0.0
        frame["profile_transfer_summary_max"] = 0.0
    risk_profile_transfer_cols = [col for col in frame.columns if col.startswith("risk_profile_transfer_")]
    if risk_profile_transfer_cols:
        risk_profile_transfer_values = frame[risk_profile_transfer_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        frame["risk_profile_transfer_summary_mean"] = risk_profile_transfer_values.mean(axis=1)
        frame["risk_profile_transfer_summary_std"] = risk_profile_transfer_values.std(axis=1).fillna(0.0)
        frame["risk_profile_transfer_summary_max"] = risk_profile_transfer_values.max(axis=1)
    else:
        frame["risk_profile_transfer_summary_mean"] = 0.0
        frame["risk_profile_transfer_summary_std"] = 0.0
        frame["risk_profile_transfer_summary_max"] = 0.0
    calibrated_transfer = _calibrated_transfer_features(frame)
    frame = frame.merge(calibrated_transfer, on="sample_id", how="left")

    candidate_features = [
        col
        for col in frame.columns
        if col.startswith("rdkit_")
        or col.startswith("maccs_")
        or col.startswith("tt_")
        or col.startswith("morgan_")
        or col.startswith("atom_")
        or col.startswith("domain_")
        or col.startswith("fg_")
        or col.startswith("teacher_")
        or col.startswith("source_teacher_")
        or col.startswith("pathway_transfer_")
        or col.startswith("profile_transfer_")
        or col.startswith("local_profile_transfer_")
        or col.startswith("risk_profile_transfer_")
        or col.startswith("risk_adjusted_transfer_")
        or col.startswith("conditioned_transfer_")
        or col.startswith("disagreement_transfer_")
        or col.startswith("residualized_transfer_")
        or col.startswith("calibrated_transfer_")
        or col.startswith("time_h_")
        or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
    ]
    forbidden = FORBIDDEN_FEATURE_COLUMNS & set(candidate_features)
    if forbidden:
        raise ValueError(f"Label-equivalent columns leaked into features: {sorted(forbidden)}")
    frame[candidate_features] = frame[candidate_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return ModelMatrix(frame=frame, feature_columns=candidate_features)

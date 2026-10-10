from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdFingerprintGenerator
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from .features import ModelMatrix

@dataclass(frozen=True)
class TimeLocalBlendSpec:
    time_h: int
    alpha: float
    topk: int = 3
    power: float = 2.0
    fingerprint: str = "morgan_r2"

@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    params: dict
    feature_group: str = "all"
    local_alpha: float = 0.0
    local_topk: int = 3
    local_power: float = 2.0
    local_fingerprint: str = "morgan_r2"
    time_local_alpha: float = 0.0
    time_local_h: int | None = None
    time_local_topk: int = 3
    time_local_power: float = 2.0
    time_local_fingerprint: str = "morgan_r2"
    time_local_blends: tuple[TimeLocalBlendSpec, ...] = ()
    local_time_scale: float | None = None
    local_statistic: str = "weighted_mean"
    local_quantile: float = 0.25
    descriptor_local_alpha: float = 0.0
    descriptor_local_topk: int = 3
    descriptor_local_power: float = 2.0
    descriptor_local_feature_group: str = "domain_teacher"
    descriptor_local_scaler: str = "robust"
    descriptor_local_time_h: int | None = None
    descriptor_local_min_delta: float | None = None
    residual_baseline_column: str | None = None
    residual_shrinkage: float = 1.0
    residual_ensemble_weight: float = 1.0
    calibration_method: str = "none"
    paired_counterpart_id: str | None = None

LOCAL_FINGERPRINT_GENERATORS = {
    "morgan_r1": rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048),
    "morgan_r2": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
    "morgan_r3": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048),
    "atom_pair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
    "topological_torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048),
}

def _fingerprint(smiles: str, kind: str = "morgan_r2"):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    if kind in LOCAL_FINGERPRINT_GENERATORS:
        return LOCAL_FINGERPRINT_GENERATORS[kind].GetFingerprint(mol)
    if kind == "rdk":
        return RDKFingerprint(mol, fpSize=2048)
    if kind == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    raise ValueError(f"Unknown local fingerprint kind: {kind}")

def local_read_across_predict(
    fit_smiles: list[str] | pd.Series,
    fit_y: list[float] | np.ndarray | pd.Series,
    query_smiles: list[str] | pd.Series,
    fit_time_h: list[float] | np.ndarray | pd.Series | None = None,
    query_time_h: list[float] | np.ndarray | pd.Series | None = None,
    topk: int = 3,
    power: float = 2.0,
    exclude_self: bool = False,
    time_scale: float | None = None,
    statistic: str = "weighted_mean",
    quantile: float = 0.25,
    fingerprint_kind: str = "morgan_r2",
) -> np.ndarray:
    fit_smiles_list = [str(value) for value in fit_smiles]
    fit_y_array = np.asarray(fit_y, dtype=float)
    fit_time_array = np.asarray(fit_time_h, dtype=float) if fit_time_h is not None else None
    query_time_array = np.asarray(query_time_h, dtype=float) if query_time_h is not None else None
    fit_fps = [_fingerprint(smiles, kind=fingerprint_kind) for smiles in fit_smiles_list]
    preds = []
    for query_idx, query in enumerate([str(value) for value in query_smiles]):
        query_fp = _fingerprint(query, kind=fingerprint_kind)
        sims = np.zeros(len(fit_smiles_list), dtype=float)
        if query_fp is not None:
            valid = [(idx, fp) for idx, fp in enumerate(fit_fps) if fp is not None]
            if valid:
                valid_indices = [idx for idx, _ in valid]
                valid_sims = DataStructs.BulkTanimotoSimilarity(query_fp, [fp for _, fp in valid])
                sims[valid_indices] = valid_sims
        if time_scale is not None and fit_time_array is not None and query_time_array is not None:
            delta = np.abs(fit_time_array - query_time_array[query_idx])
            sims = sims * np.exp(-delta / float(time_scale))
        if exclude_self:
            excluded = []
            for idx, smiles in enumerate(fit_smiles_list):
                same_smiles = smiles == query
                same_time = (
                    fit_time_array is None
                    or query_time_array is None
                    or float(fit_time_array[idx]) == float(query_time_array[query_idx])
                )
                if same_smiles and same_time:
                    excluded.append(idx)
            sims[excluded] = -1.0
        order = np.argsort(-sims)
        order = [idx for idx in order if sims[idx] >= 0][: max(1, int(topk))]
        if not order:
            preds.append(float(np.mean(fit_y_array)))
            continue
        selected_sims = sims[order]
        selected_y = fit_y_array[order]
        if statistic == "quantile":
            preds.append(float(np.quantile(selected_y, float(quantile))))
            continue
        if statistic == "mean":
            preds.append(float(np.mean(selected_y)))
            continue
        if statistic != "weighted_mean":
            raise ValueError(f"Unknown local read-across statistic: {statistic}")
        weights = np.power(np.clip(selected_sims, 0.0, None), float(power))
        if float(weights.sum()) == 0.0:
            weights = np.ones(len(order), dtype=float)
        preds.append(float(np.average(selected_y, weights=weights)))
    return np.asarray(preds, dtype=float)

def descriptor_local_read_across_predict(
    fit_frame: pd.DataFrame,
    fit_y: list[float] | np.ndarray | pd.Series,
    query_frame: pd.DataFrame,
    feature_columns: list[str],
    topk: int = 3,
    power: float = 2.0,
    exclude_self: bool = False,
    query_indices: np.ndarray | None = None,
    scaler: str = "robust",
) -> np.ndarray:
    if not feature_columns:
        raise ValueError("Descriptor local read-across requires at least one feature column")
    missing = set(feature_columns) - set(fit_frame.columns) | (set(feature_columns) - set(query_frame.columns))
    if missing:
        raise ValueError(f"Descriptor local features missing columns: {sorted(missing)}")
    fit_y_array = np.asarray(fit_y, dtype=float)
    fit_x = fit_frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    query_x = query_frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if scaler == "robust":
        from sklearn.preprocessing import RobustScaler

        transformer = RobustScaler()
    elif scaler == "standard":
        from sklearn.preprocessing import StandardScaler

        transformer = StandardScaler()
    else:
        raise ValueError(f"Unknown descriptor local scaler: {scaler}")
    fit_z = transformer.fit_transform(fit_x)
    query_z = transformer.transform(query_x)
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(query_z, fit_z, metric="euclidean")
    if exclude_self:
        if query_indices is None:
            query_indices = query_frame.index.to_numpy()
        fit_index = pd.Index(fit_frame.index)
        for query_row, query_index in enumerate(query_indices):
            matches = np.flatnonzero(fit_index == query_index)
            distances[query_row, matches] = np.inf
    preds = []
    for distance_row in distances:
        finite_order = [idx for idx in np.argsort(distance_row) if np.isfinite(distance_row[idx])]
        order = finite_order[: max(1, int(topk))]
        if not order:
            preds.append(float(np.mean(fit_y_array)))
            continue
        selected_distances = distance_row[order]
        similarities = 1.0 / (1.0 + np.clip(selected_distances, 0.0, None))
        weights = np.power(similarities, float(power))
        if float(weights.sum()) == 0.0:
            weights = np.ones(len(order), dtype=float)
        preds.append(float(np.average(fit_y_array[order], weights=weights)))
    return np.asarray(preds, dtype=float)

def _descriptor_local_gate(
    base_pred: np.ndarray,
    descriptor_pred: np.ndarray,
    query_frame: pd.DataFrame,
    spec: CandidateSpec,
) -> np.ndarray:
    gate = np.ones(len(query_frame), dtype=bool)
    if spec.descriptor_local_time_h is not None:
        gate &= query_frame["time_h"].astype(int).to_numpy() == int(spec.descriptor_local_time_h)
    if spec.descriptor_local_min_delta is not None:
        gate &= (np.asarray(base_pred, dtype=float) - np.asarray(descriptor_pred, dtype=float)) >= float(
            spec.descriptor_local_min_delta
        )
    return gate

def apply_descriptor_local_blend(
    base_pred: np.ndarray,
    fit_frame: pd.DataFrame,
    fit_y: list[float] | np.ndarray | pd.Series,
    query_frame: pd.DataFrame,
    spec: CandidateSpec,
    exclude_self: bool = False,
    query_indices: np.ndarray | None = None,
) -> np.ndarray:
    blended = np.asarray(base_pred, dtype=float).copy()
    if float(spec.descriptor_local_alpha) <= 0.0:
        return blended
    descriptor_columns = descriptor_local_feature_columns(fit_frame, spec.descriptor_local_feature_group)
    descriptor_pred = descriptor_local_read_across_predict(
        fit_frame=fit_frame,
        fit_y=fit_y,
        query_frame=query_frame,
        feature_columns=descriptor_columns,
        topk=spec.descriptor_local_topk,
        power=spec.descriptor_local_power,
        exclude_self=exclude_self,
        query_indices=query_indices,
        scaler=spec.descriptor_local_scaler,
    )
    gate = _descriptor_local_gate(blended, descriptor_pred, query_frame, spec)
    blended[gate] = (1.0 - spec.descriptor_local_alpha) * blended[gate] + spec.descriptor_local_alpha * descriptor_pred[gate]
    return blended

def apply_time_specific_local_blend(
    base_pred: np.ndarray,
    fit_frame: pd.DataFrame,
    fit_y: np.ndarray,
    query_frame: pd.DataFrame,
    time_h: int,
    alpha: float,
    topk: int,
    power: float = 2.0,
    fingerprint_kind: str = "morgan_r2",
    exclude_self: bool = False,
) -> np.ndarray:
    blended = np.asarray(base_pred, dtype=float).copy()
    if float(alpha) <= 0.0:
        return blended
    fit_mask = fit_frame["time_h"].astype(int).eq(int(time_h)).to_numpy()
    query_mask = query_frame["time_h"].astype(int).eq(int(time_h)).to_numpy()
    if int(fit_mask.sum()) == 0 or int(query_mask.sum()) == 0:
        return blended
    fit_subset = fit_frame.loc[fit_mask].reset_index(drop=True)
    query_subset = query_frame.loc[query_mask].reset_index(drop=True)
    local_pred = local_read_across_predict(
        fit_smiles=fit_subset["canonical_smiles"],
        fit_y=np.asarray(fit_y, dtype=float)[fit_mask],
        query_smiles=query_subset["canonical_smiles"],
        fit_time_h=fit_subset["time_h"],
        query_time_h=query_subset["time_h"],
        topk=min(int(topk), int(fit_mask.sum())),
        power=power,
        exclude_self=exclude_self,
        fingerprint_kind=fingerprint_kind,
    )
    blended[query_mask] = (1.0 - float(alpha)) * blended[query_mask] + float(alpha) * local_pred
    return blended

def apply_time_specific_local_blends(
    base_pred: np.ndarray,
    fit_frame: pd.DataFrame,
    fit_y: np.ndarray,
    query_frame: pd.DataFrame,
    blends: tuple[TimeLocalBlendSpec, ...],
    exclude_self: bool = False,
) -> np.ndarray:
    blended = np.asarray(base_pred, dtype=float).copy()
    for blend in blends:
        blended = apply_time_specific_local_blend(
            base_pred=blended,
            fit_frame=fit_frame,
            fit_y=fit_y,
            query_frame=query_frame,
            time_h=int(blend.time_h),
            alpha=float(blend.alpha),
            topk=int(blend.topk),
            power=float(blend.power),
            fingerprint_kind=str(blend.fingerprint),
            exclude_self=exclude_self,
        )
    return blended

def time_local_blends_for_spec(spec: CandidateSpec) -> tuple[TimeLocalBlendSpec, ...]:
    blends: list[TimeLocalBlendSpec] = []
    if spec.time_local_alpha > 0:
        blends.append(
            TimeLocalBlendSpec(
                time_h=int(spec.time_local_h) if spec.time_local_h is not None else 48,
                alpha=float(spec.time_local_alpha),
                topk=int(spec.time_local_topk),
                power=float(spec.time_local_power),
                fingerprint=str(spec.time_local_fingerprint),
            )
        )
    blends.extend(spec.time_local_blends)
    return tuple(blends)

def format_time_local_blends(spec: CandidateSpec) -> str:
    parts = []
    for blend in time_local_blends_for_spec(spec):
        parts.append(
            f"{int(blend.time_h)}:{blend.fingerprint}:a{float(blend.alpha):.3g}:"
            f"k{int(blend.topk)}:p{float(blend.power):.3g}"
        )
    return ";".join(parts)

def _is_standard_source_teacher_feature(column: str) -> bool:
    return (
        column.startswith("source_teacher_")
        and not column.startswith("source_teacher_weighted_")
        and not column.startswith("source_teacher_cell_")
    )

def _is_weighted_source_teacher_feature(column: str) -> bool:
    return column.startswith("source_teacher_weighted_") and not column.startswith("source_teacher_weighted_cell_")

def _is_cell_source_teacher_feature(column: str) -> bool:
    return column.startswith("source_teacher_cell_")

def _is_weighted_cell_source_teacher_feature(column: str) -> bool:
    return column.startswith("source_teacher_weighted_cell_")

def feature_columns_for_spec(matrix: ModelMatrix, spec: CandidateSpec) -> list[str]:
    columns = list(matrix.feature_columns)
    if spec.feature_group == "all":
        selected = columns
    elif spec.feature_group == "no_source_teacher":
        selected = [col for col in columns if not col.startswith("source_teacher_")]
    elif spec.feature_group == "no_transfer":
        selected = [
            col
            for col in columns
            if not (
                col.startswith("teacher_")
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
            )
        ]
    elif spec.feature_group == "rdkit_desc_time_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("rdkit_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "rdkit_desc_time_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("rdkit_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_profile_summary_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("profile_transfer_summary_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_pathway_summary_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_pathway_summary_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_pathway_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("pathway_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_pathway_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_profile_summary_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_local_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_profile_local_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("profile_transfer_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_profile_local_risk_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("profile_transfer_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("risk_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_atom_local_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_atom_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_risk_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("risk_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_local_risk_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("risk_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_atom_local_risk_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("risk_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group in {
        "domain_rdkit_profile_no_transfer",
        "domain_rdkit_local_profile_no_transfer",
        "domain_rdkit_atom_local_profile_no_transfer",
        "domain_rdkit_atom_profile_no_transfer",
        "domain_rdkit_risk_profile_no_transfer",
        "domain_rdkit_local_risk_profile_no_transfer",
        "domain_rdkit_atom_local_risk_profile_no_transfer",
    }:
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or (
                    spec.feature_group in {
                        "domain_rdkit_atom_local_profile_no_transfer",
                        "domain_rdkit_atom_profile_no_transfer",
                        "domain_rdkit_atom_local_risk_profile_no_transfer",
                    }
                    and (col.startswith("maccs_") or col.startswith("atom_"))
                )
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_calibrated_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("calibrated_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_calibrated_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_pathway_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("pathway_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_pathway_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_pathway_summary_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_pathway_summary_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_pathway_time48_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("pathway_transfer_time48_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_pathway_time48_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_profile_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_profile_summary_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("profile_transfer_summary_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_profile_summary_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_local_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("local_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_local_profile_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_local_profile_clean_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_local_profile_clean_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_risk_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("risk_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_risk_profile_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_risk_adjusted_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("risk_adjusted_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_risk_adjusted_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_conditioned_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("conditioned_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_conditioned_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_disagreement_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("disagreement_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_disagreement_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_residualized_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("residualized_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_residualized_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_calibrated_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("calibrated_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_calibrated_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_weighted_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_weighted_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_weighted_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_cell_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or _is_cell_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_cell_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_weighted_cell_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("teacher_")
            or _is_weighted_source_teacher_feature(col)
            or _is_weighted_cell_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_weighted_cell_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_morgan_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("morgan_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_morgan_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("morgan_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_maccs_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("teacher_")
            or _is_standard_source_teacher_feature(col)
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "domain_rdkit_tt_maccs_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("tt_")
            or col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "maccs_atom_time_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "maccs_atom_time_pathway_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("pathway_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "maccs_atom_time_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "maccs_atom_time_local_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "maccs_atom_time_risk_profile_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("risk_profile_transfer_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "maccs_atom_time_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group == "maccs_atom_time_pathway_no_transfer":
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    elif spec.feature_group in {
        "maccs_atom_time_profile_no_transfer",
        "maccs_atom_time_local_profile_no_transfer",
        "maccs_atom_time_risk_profile_no_transfer",
    }:
        selected = [
            col
            for col in columns
            if col.startswith("maccs_")
            or col.startswith("atom_")
            or col.startswith("time_h_")
            or col in {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
        ]
    else:
        raise ValueError(f"Unknown feature group: {spec.feature_group}")
    if not selected:
        raise ValueError(f"Feature group {spec.feature_group} selected no columns")
    return selected

def _transfer_risk_suppression_mask(frame: pd.DataFrame) -> np.ndarray:
    time_h = pd.to_numeric(frame.get("time_h_numeric", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    nitro = pd.to_numeric(frame.get("fg_nitro", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    trihalo = pd.to_numeric(frame.get("fg_trihalomethyl", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    halogen_fraction = pd.to_numeric(
        frame.get("domain_halogen_atom_fraction", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    bromine_iodine = pd.to_numeric(
        frame.get("domain_bromine_iodine_count", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    return (
        time_h.eq(72.0)
        & (nitro.gt(0.0) | trihalo.gt(0.0) | halogen_fraction.ge(0.30) | bromine_iodine.ge(2.0))
    ).to_numpy(dtype=bool)

def _is_transfer_channel_column(column: str) -> bool:
    return (
        column.startswith("teacher_")
        or column.startswith("source_teacher_")
        or column.startswith("pathway_transfer_")
        or column.startswith("profile_transfer_")
        or column.startswith("local_profile_transfer_")
        or column.startswith("risk_profile_transfer_")
        or column.startswith("risk_adjusted_transfer_")
        or column.startswith("conditioned_transfer_")
        or column.startswith("disagreement_transfer_")
            or column.startswith("residualized_transfer_")
            or column.startswith("calibrated_transfer_")
    )

def _feature_matrix_for_spec(frame: pd.DataFrame, columns: list[str], spec: CandidateSpec) -> np.ndarray:
    x = frame[columns].to_numpy(dtype=float)
    if not bool(spec.params.get("suppress_transfer_72h_risk", False)):
        return x
    transfer_indices = [idx for idx, column in enumerate(columns) if _is_transfer_channel_column(column)]
    if not transfer_indices:
        return x
    mask = _transfer_risk_suppression_mask(frame)
    if not mask.any():
        return x
    transformed = np.asarray(x, dtype=float).copy()
    transformed[np.ix_(mask, transfer_indices)] = 0.0
    return transformed

def residual_baseline_for_spec(frame: pd.DataFrame, spec: CandidateSpec) -> np.ndarray:
    if spec.residual_baseline_column is None:
        return np.zeros(len(frame), dtype=float)
    if spec.residual_baseline_column not in frame.columns:
        raise ValueError(f"Residual baseline column not found: {spec.residual_baseline_column}")
    values = pd.to_numeric(frame[spec.residual_baseline_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if bool(spec.params.get("suppress_transfer_72h_risk", False)) and _is_transfer_channel_column(
        spec.residual_baseline_column
    ):
        values = np.where(_transfer_risk_suppression_mask(frame), 0.0, values)
    return float(spec.residual_shrinkage) * values

def descriptor_local_feature_columns(frame: pd.DataFrame, feature_group: str) -> list[str]:
    columns = list(frame.columns)
    base_counts = {"halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
    time_columns = {"time_h_numeric", "time_h_log1p"}
    if feature_group == "domain_only":
        selected = [
            col
            for col in columns
            if col.startswith("domain_") or col.startswith("fg_") or col in base_counts
        ]
    elif feature_group == "domain_teacher":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col in base_counts
            or col in time_columns
        ]
    elif feature_group == "rdkit_domain_teacher":
        selected = [
            col
            for col in columns
            if col.startswith("rdkit_")
            or col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col in base_counts
            or col in time_columns
        ]
    else:
        raise ValueError(f"Unknown descriptor local feature group: {feature_group}")
    if not selected:
        raise ValueError(f"Descriptor local feature group {feature_group} selected no columns")
    return selected

CALIBRATION_CONTEXT_COLUMNS = [
    "time_h_numeric",
    "time_h_log1p",
    "teacher_top1_sim",
    "teacher_topk_sim_mean",
    "teacher_topk_sim_max",
    "teacher_topk_y_wmean",
    "source_teacher_pred",
    "source_teacher_residual_abs_proxy",
    "source_teacher_weighted_pred",
    "source_teacher_weighted_residual_abs_proxy",
    "calibrated_transfer_wmean",
    "calibrated_transfer_reliability",
    "calibrated_transfer_uncertainty",
    "calibrated_transfer_disagreement",
]

def _calibration_context_matrix(pred: np.ndarray, frame: pd.DataFrame | None) -> np.ndarray:
    if frame is None:
        raise ValueError("Context calibration requires the matching feature frame")
    pred = np.asarray(pred, dtype=float)
    columns: dict[str, np.ndarray] = {
        "pred": pred,
        "pred_squared": pred**2,
    }
    for column in CALIBRATION_CONTEXT_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            values = np.zeros(len(frame), dtype=float)
        columns[column] = values
    pathway_columns = [column for column in frame.columns if str(column).startswith("pathway_transfer_")]
    if pathway_columns:
        pathway_values = frame[pathway_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pathway_mean = pathway_values.mean(axis=1) if pathway_values.shape[1] else np.zeros(len(frame), dtype=float)
        pathway_std = pathway_values.std(axis=1) if pathway_values.shape[1] else np.zeros(len(frame), dtype=float)
        pathway_max = pathway_values.max(axis=1) if pathway_values.shape[1] else np.zeros(len(frame), dtype=float)
    else:
        pathway_mean = np.zeros(len(frame), dtype=float)
        pathway_std = np.zeros(len(frame), dtype=float)
        pathway_max = np.zeros(len(frame), dtype=float)
    columns["pathway_transfer_mean"] = np.asarray(pathway_mean, dtype=float)
    columns["pathway_transfer_std"] = np.asarray(pathway_std, dtype=float)
    columns["pathway_transfer_max"] = np.asarray(pathway_max, dtype=float)
    return pd.DataFrame(columns).to_numpy(dtype=float)

def calibrate_predictions(
    train_pred: np.ndarray,
    train_y: np.ndarray,
    query_pred: np.ndarray,
    method: str = "none",
    train_frame: pd.DataFrame | None = None,
    query_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    method = str(method)
    query_pred = np.asarray(query_pred, dtype=float)
    if method == "none":
        return query_pred
    train_pred = np.asarray(train_pred, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    if method == "ridge_context":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        calibrator = make_pipeline(StandardScaler(), RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0)))
        train_matrix = _calibration_context_matrix(train_pred, train_frame)
        query_matrix = _calibration_context_matrix(query_pred, query_frame)
        calibrator.fit(train_matrix, train_y)
        return np.asarray(calibrator.predict(query_matrix), dtype=float)

    train_matrix = train_pred.reshape(-1, 1)
    query_matrix = query_pred.reshape(-1, 1)
    if method == "linear":
        from sklearn.linear_model import LinearRegression

        calibrator = LinearRegression()
    elif method == "ridge":
        from sklearn.linear_model import RidgeCV

        calibrator = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))
    elif method == "huber":
        from sklearn.linear_model import HuberRegressor

        calibrator = HuberRegressor(alpha=0.01)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    calibrator.fit(train_matrix, train_y)
    return np.asarray(calibrator.predict(query_matrix), dtype=float)

def make_cv_splits(train: pd.DataFrame, cv_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    return make_group_cv_splits(train, group_column="canonical_smiles", cv_folds=cv_folds)

def make_group_cv_splits(
    train: pd.DataFrame,
    group_column: str,
    cv_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if group_column not in train.columns:
        raise ValueError(f"Group CV requires {group_column}")
    if "canonical_smiles" not in train.columns:
        raise ValueError("Compound-group CV requires canonical_smiles")
    groups = train[group_column].astype(str).to_numpy()
    n_groups = len(set(groups))
    if n_groups < int(cv_folds):
        raise ValueError(f"Need at least {cv_folds} unique groups for {group_column} CV, got {n_groups}")
    splitter = GroupKFold(n_splits=int(cv_folds))
    x_placeholder = np.zeros(len(train), dtype=float)
    return [(fit_idx, valid_idx) for fit_idx, valid_idx in splitter.split(x_placeholder, train["y"], groups)]

QUICK_PRIORITY_CANDIDATE_IDS = (
    "a90_conditioned_delta_transfer_etr_s60",
    "a90_conditioned_delta_no_transfer",
    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_risk_suppressed",
    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_risk_suppressed_no_transfer",
    "a90_local_profile_gated_transfer_etr_s100",
    "a90_local_profile_gated_no_transfer",
    "a90_conditioned_ridge_risk_suppressed_transfer_s35_u060_sim025",
    "a90_conditioned_ridge_risk_suppressed_no_transfer",
    "a90_conditioned_ridge_conservative_transfer_s25_b005",
    "a90_conditioned_ridge_conservative_no_transfer",
    "a90_conditioned_ridge_confident_transfer_s35_u060_sim025",
    "a90_conditioned_ridge_confident_no_transfer",
    "a90_calibrated_gated_transfer_etr_s100",
    "a90_calibrated_gated_no_transfer",
    "a90_teacher_topk_risk_routed_transfer",
    "a90_teacher_topk_risk_routed_no_transfer",
    "risk_adjusted_gated_transfer_etr_s100",
    "risk_adjusted_gated_no_transfer",
    "extratrees_400_domain_rdkit_tt_conditioned_clean_mid_s25_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_400_domain_rdkit_tt_conditioned_clean_mid_s25_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_400_domain_rdkit_tt_conditioned_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_400_domain_rdkit_tt_conditioned_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
    "a90_delta_gated_transfer_etr_s100",
    "a90_delta_gated_no_transfer",
    "disagreement_gated_transfer_etr_s100",
    "disagreement_gated_no_transfer",
    "extratrees_400_domain_rdkit_tt_calibrated_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_400_domain_rdkit_tt_calibrated_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
    "a90_calibrated_gated_transfer_etr_s100",
    "a90_calibrated_gated_no_transfer",
    "extratrees_400_domain_rdkit_tt_disagreement_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_400_domain_rdkit_tt_disagreement_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
    "risk_adjusted_gated_transfer_logit_c2_s125",
    "risk_adjusted_gated_no_transfer_logit_c2_s125",
    "extratrees_400_domain_rdkit_tt_risk_adjusted_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_400_domain_rdkit_tt_risk_adjusted_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
    "a90_risk_profile_gated_transfer_etr_s100",
    "a90_risk_profile_gated_no_transfer",
    "a90_risk_profile_gated_transfer_logit_c2_s125",
    "a90_risk_profile_gated_no_transfer_logit_c2_s125",
    "a90_local_profile_gated_transfer_logit_c2_s125",
    "a90_local_profile_gated_no_transfer_logit_c2_s125",
    "extratrees_400_domain_rdkit_tt_local_profile_f85_l2_desc_time48_domain_a90_k1_delta00",
    "extratrees_400_domain_rdkit_tt_local_profile_no_transfer_f85_l2_desc_time48_domain_a90_k1_delta00",
    "a90_risk_gated_transfer_etr_s100",
    "a90_risk_gated_no_transfer",
    "a90_risk_gated_transfer_logit_c2_s125",
    "a90_risk_gated_no_transfer_logit_c2_s125",
    "gated_transfer_all_context_etr_s100",
    "gated_transfer_all_context_no_transfer_etr_s100",
    "transfer_blend_et65_l1_svr16_eps005_w395_b290",
    "transfer_blend_et65_l1_svr16_eps005_no_transfer_w395_b000",
    "residualized_gated_transfer_etr_s100",
    "residualized_gated_no_transfer",
    "residualized_gated_transfer_a90_etr_s100",
    "residualized_gated_no_transfer_a90",
    "residualized_gated_transfer_logit_c2_s125",
    "residualized_gated_no_transfer_logit_c2_s125",
    "extratrees_400_domain_rdkit_tt_residualized_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_400_domain_rdkit_tt_residualized_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
    "extratrees_700_domain_rdkit_tt_profile_plain_f70_l2",
    "extratrees_700_domain_rdkit_tt_profile_plain_no_transfer_f70_l2",
    "extratrees_300_domain_rdkit_tt_profile_plain_f85_l2_desc_time48_domain_a100_k1_delta10",
    "extratrees_300_domain_rdkit_tt_profile_plain_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta10",
    "extratrees_400_domain_rdkit_tt_profile_summary_f85_l2_desc_time48_domain_a100_k1_delta10",
    "extratrees_400_domain_rdkit_tt_profile_summary_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta10",
    "svr_c16_eps001_domain_rdkit_transfer",
    "svr_c16_eps001_domain_rdkit_no_transfer",
    "extratrees_700_domain_rdkit_tt_profile_f70_l2_teacher_topk_resid65_ens100_huber",
    "extratrees_700_domain_rdkit_tt_profile_no_transfer_f70_l2",
    "extratrees_700_domain_rdkit_tt_pathway_time48_f70_l2_teacher_topk_resid65_ens100_huber",
    "extratrees_700_domain_rdkit_tt_pathway_time48_no_transfer_f70_l2",
    "extratrees_700_domain_rdkit_tt_pathway_summary_f70_l2_teacher_topk_resid65_ens100_ridge_context",
    "extratrees_700_domain_rdkit_tt_pathway_summary_no_transfer_f70_l2",
    "extratrees_700_domain_rdkit_tt_pathway_f70_l2_teacher_topk_resid65_ens100_ridge_context",
    "extratrees_700_domain_rdkit_tt_pathway_no_transfer_f70_l2",
    "extratrees_700_maccs_atom_time_pathway_transfer_leaf2_f70",
    "extratrees_700_maccs_atom_time_pathway_no_transfer_leaf2_f70",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a50_k1_delta00",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a50_k1_delta10",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a40_k1_delta00",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a40_k1_delta10",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a30_k1_delta00",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a30_k1_delta10",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a20_k1_delta00",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a20_k1_delta10",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a15_k1_delta00",
    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a10_k1_delta00",
)

def _transfer_blend_base_specs(spec: CandidateSpec) -> tuple[CandidateSpec, CandidateSpec, CandidateSpec, CandidateSpec]:
    et_params = {
        "n_estimators": int(spec.params.get("et_n_estimators", 700)),
        "min_samples_leaf": int(spec.params.get("et_min_samples_leaf", 1)),
        "max_features": float(spec.params.get("et_max_features", 0.65)),
        "random_state": int(spec.params.get("et_random_state", 20260605)),
        "n_jobs": int(spec.params.get("n_jobs", -1)),
    }
    svr_params = {
        "C": float(spec.params.get("svr_C", 16.0)),
        "epsilon": float(spec.params.get("svr_epsilon", 0.005)),
        "gamma": spec.params.get("svr_gamma", "scale"),
    }
    return (
        CandidateSpec(f"{spec.candidate_id}__et_full", "extratrees", et_params, feature_group="domain_rdkit_tt_transfer"),
        CandidateSpec(f"{spec.candidate_id}__et_no", "extratrees", et_params, feature_group="domain_rdkit_tt_no_transfer"),
        CandidateSpec(f"{spec.candidate_id}__svr_full", "svr", svr_params, feature_group="domain_rdkit_transfer"),
        CandidateSpec(f"{spec.candidate_id}__svr_no", "svr", svr_params, feature_group="domain_rdkit_no_transfer"),
    )

def _calibrated_transfer_blend_base_specs(
    spec: CandidateSpec,
) -> tuple[CandidateSpec, CandidateSpec, CandidateSpec, CandidateSpec]:
    et_full, et_no, svr_full, svr_no = _transfer_blend_base_specs(spec)
    calibration_method = str(spec.params.get("calibration_method", "huber"))
    et_full = CandidateSpec(
        et_full.candidate_id,
        et_full.family,
        et_full.params,
        feature_group="domain_rdkit_tt_calibrated_transfer",
        calibration_method=calibration_method,
    )
    et_no = CandidateSpec(
        et_no.candidate_id,
        et_no.family,
        et_no.params,
        feature_group="domain_rdkit_tt_calibrated_no_transfer",
        calibration_method=calibration_method,
    )
    svr_full = CandidateSpec(
        svr_full.candidate_id,
        svr_full.family,
        svr_full.params,
        feature_group="domain_rdkit_calibrated_transfer",
        calibration_method="ridge_context" if calibration_method == "ridge_context" else calibration_method,
    )
    svr_no = CandidateSpec(
        svr_no.candidate_id,
        svr_no.family,
        svr_no.params,
        feature_group="domain_rdkit_calibrated_no_transfer",
        calibration_method="ridge_context" if calibration_method == "ridge_context" else calibration_method,
    )
    return et_full, et_no, svr_full, svr_no

def _transfer_blend_feature_count(matrix: ModelMatrix, spec: CandidateSpec) -> int:
    base_specs = (
        _calibrated_transfer_blend_base_specs(spec)
        if spec.family == "calibrated_transfer_blend"
        else _transfer_blend_base_specs(spec)
    )
    return len(set().union(*(set(feature_columns_for_spec(matrix, base_spec)) for base_spec in base_specs)))

def _a90_gated_feature_count(matrix: ModelMatrix, spec: CandidateSpec) -> int:
    full_spec, no_spec = _a90_gated_base_specs(spec)
    gate_columns = _gated_transfer_feature_columns(matrix.frame, str(spec.params.get("gate_feature_group", "risk_context")))
    return len(
        set(feature_columns_for_spec(matrix, full_spec))
        | set(feature_columns_for_spec(matrix, no_spec))
        | set(gate_columns)
    )

def _a90_delta_gated_feature_count(matrix: ModelMatrix, spec: CandidateSpec) -> int:
    anchor_spec, _, delta_spec, delta_no_spec = _a90_delta_gated_base_specs(spec)
    gate_columns = _gated_transfer_feature_columns(matrix.frame, str(spec.params.get("gate_feature_group", "all_context")))
    return len(
        set(feature_columns_for_spec(matrix, anchor_spec))
        | set().union(
            *(
                set(feature_columns_for_spec(matrix, base_spec))
                for gated_spec in (delta_spec, delta_no_spec)
                for base_spec in _transfer_blend_base_specs(gated_spec)
            )
        )
        | set(gate_columns)
    )

def _risk_adjusted_gated_feature_count(matrix: ModelMatrix, spec: CandidateSpec) -> int:
    full_spec, no_spec = _risk_adjusted_gated_base_specs(spec)
    gate_columns = _gated_transfer_feature_columns(
        matrix.frame,
        str(spec.params.get("gate_feature_group", "risk_adjusted_context")),
    )
    return len(
        set(feature_columns_for_spec(matrix, full_spec))
        | set(feature_columns_for_spec(matrix, no_spec))
        | set(gate_columns)
    )

def _disagreement_gated_feature_count(matrix: ModelMatrix, spec: CandidateSpec) -> int:
    full_spec, no_spec = _disagreement_gated_base_specs(spec)
    gate_columns = _gated_transfer_feature_columns(
        matrix.frame,
        str(spec.params.get("gate_feature_group", "disagreement_context")),
    )
    return len(
        set(feature_columns_for_spec(matrix, full_spec))
        | set(feature_columns_for_spec(matrix, no_spec))
        | set(gate_columns)
    )

def _risk_routed_feature_count(matrix: ModelMatrix, spec: CandidateSpec) -> int:
    full_spec, no_spec = _risk_routed_base_specs(spec)
    return len(set(feature_columns_for_spec(matrix, full_spec)) | set(feature_columns_for_spec(matrix, no_spec)))

def _fit_transfer_blend_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    et_full, et_no, svr_full, svr_no = (
        _calibrated_transfer_blend_base_specs(spec)
        if spec.family == "calibrated_transfer_blend"
        else _transfer_blend_base_specs(spec)
    )
    blend_weight = float(spec.params.get("blend_weight", 0.395))
    residual_weight = float(spec.params.get("blend_residual_weight", 0.0))
    use_transfer = bool(spec.params.get("transfer_blend_use_transfer", True))
    et_primary = et_full if use_transfer else et_no
    svr_primary = svr_full if use_transfer else svr_no

    def raw(base_spec: CandidateSpec) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        columns = feature_columns_for_spec(matrix, base_spec)
        x = _feature_matrix_for_spec(train, columns, base_spec)
        baseline = residual_baseline_for_spec(train, base_spec)
        return _fit_predict_raw_candidate(
            train,
            x,
            y,
            baseline,
            base_spec,
            fit_idx,
            valid_idx,
            return_fit_predictions=return_fit_predictions,
        )

    if return_fit_predictions:
        et_full_valid, et_full_fit = raw(et_primary)
        svr_full_valid, svr_full_fit = raw(svr_primary)
        svr_no_valid, svr_no_fit = raw(svr_no)
        valid_pred = (
            blend_weight * et_full_valid
            + (1.0 - blend_weight) * svr_full_valid
            + residual_weight * (svr_full_valid - svr_no_valid)
        )
        fit_pred = (
            blend_weight * et_full_fit
            + (1.0 - blend_weight) * svr_full_fit
            + residual_weight * (svr_full_fit - svr_no_fit)
        )
        return np.asarray(valid_pred, dtype=float), np.asarray(fit_pred, dtype=float)

    et_full_valid = raw(et_primary)
    svr_full_valid = raw(svr_primary)
    svr_no_valid = raw(svr_no)
    return np.asarray(
        blend_weight * et_full_valid
        + (1.0 - blend_weight) * svr_full_valid
        + residual_weight * (svr_full_valid - svr_no_valid),
        dtype=float,
    )

def _fit_transfer_blend_and_predict_role(
    matrix: ModelMatrix,
    spec: CandidateSpec,
    role: str,
) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    role_frame = matrix.role(role)
    et_full, et_no, svr_full, svr_no = (
        _calibrated_transfer_blend_base_specs(spec)
        if spec.family == "calibrated_transfer_blend"
        else _transfer_blend_base_specs(spec)
    )
    blend_weight = float(spec.params.get("blend_weight", 0.395))
    residual_weight = float(spec.params.get("blend_residual_weight", 0.0))
    use_transfer = bool(spec.params.get("transfer_blend_use_transfer", True))
    et_primary = et_full if use_transfer else et_no
    svr_primary = svr_full if use_transfer else svr_no

    def role_pred(base_spec: CandidateSpec) -> np.ndarray:
        columns = feature_columns_for_spec(matrix, base_spec)
        train_x = _feature_matrix_for_spec(train, columns, base_spec)
        estimator = _make_estimator(base_spec)
        estimator.fit(train_x, y - residual_baseline_for_spec(train, base_spec))
        role_x = _feature_matrix_for_spec(role_frame, columns, base_spec)
        return residual_baseline_for_spec(role_frame, base_spec) + np.asarray(estimator.predict(role_x), dtype=float)

    et_full_pred = role_pred(et_primary)
    svr_full_pred = role_pred(svr_primary)
    svr_no_pred = role_pred(svr_no)
    return np.asarray(
        blend_weight * et_full_pred
        + (1.0 - blend_weight) * svr_full_pred
        + residual_weight * (svr_full_pred - svr_no_pred),
        dtype=float,
    )

def _gated_transfer_base_params(spec: CandidateSpec) -> dict:
    return {
        "et_n_estimators": int(spec.params.get("et_n_estimators", 700)),
        "et_min_samples_leaf": int(spec.params.get("et_min_samples_leaf", 1)),
        "et_max_features": float(spec.params.get("et_max_features", 0.65)),
        "et_random_state": int(spec.params.get("et_random_state", 20260605)),
        "svr_C": float(spec.params.get("svr_C", 16.0)),
        "svr_epsilon": float(spec.params.get("svr_epsilon", 0.005)),
        "svr_gamma": spec.params.get("svr_gamma", "scale"),
        "blend_weight": float(spec.params.get("blend_weight", 0.245)),
        "blend_residual_weight": float(spec.params.get("blend_residual_weight", 0.33)),
    }

def _gated_transfer_specs(spec: CandidateSpec) -> tuple[CandidateSpec, CandidateSpec]:
    params = _gated_transfer_base_params(spec)
    no_params = {
        **params,
        "transfer_blend_use_transfer": False,
        "blend_residual_weight": 0.0,
    }
    raw_params = {
        **params,
        "transfer_blend_use_transfer": True,
    }
    return (
        CandidateSpec(f"{spec.candidate_id}__raw_full", "transfer_blend", raw_params),
        CandidateSpec(f"{spec.candidate_id}__no", "transfer_blend", no_params),
    )

def _gated_transfer_feature_columns(frame: pd.DataFrame, feature_group: str) -> list[str]:
    columns = list(frame.columns)
    base = {"time_h_numeric", "time_h_log1p", "halogen_count", "heavy_halogen_count", "aromatic_atom_count"}
    if feature_group == "all_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("rdkit_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "transfer_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("profile_transfer_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "risk_context":
        risk_columns = {
            "source_teacher_residual_abs_proxy",
            "source_teacher_weighted_residual_abs_proxy",
            "source_teacher_cell_cho_gap",
            "source_teacher_weighted_cell_cho_gap",
            "teacher_top1_sim",
            "teacher_topk_sim_mean",
            "teacher_topk_sim_max",
            "teacher_topk_y_std",
            "teacher_topk_n",
            "domain_halogen_fraction_mw",
            "domain_halogen_atom_fraction",
            "domain_aliphatic_halogen_count",
            "domain_aromatic_halogen_count",
            "domain_max_halogen_z",
            "domain_halogen_diversity",
            "domain_bromine_iodine_count",
            "domain_cl_br_i_count",
            "fg_nitro",
            "fg_trihalomethyl",
            "fg_quinone",
            "fg_phenol",
            *base,
        }
        selected = [
            col
            for col in columns
            if col in risk_columns
            or col.startswith("profile_transfer_summary_")
            or col.startswith("pathway_transfer_summary_")
        ]
    elif feature_group == "risk_profile_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("risk_profile_transfer_")
            or col.startswith("profile_transfer_summary_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "residualized_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("residualized_transfer_")
            or col.startswith("profile_transfer_summary_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "risk_adjusted_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("risk_adjusted_transfer_")
            or col.startswith("profile_transfer_summary_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "conditioned_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("conditioned_transfer_")
            or col.startswith("profile_transfer_summary_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "disagreement_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("disagreement_transfer_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "local_profile_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("local_profile_transfer_")
            or col.startswith("profile_transfer_summary_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in base
        ]
    elif feature_group == "calibrated_context":
        selected = [
            col
            for col in columns
            if col.startswith("domain_")
            or col.startswith("fg_")
            or col.startswith("teacher_")
            or col.startswith("source_teacher_")
            or col.startswith("calibrated_transfer_")
            or col.startswith("profile_transfer_summary_")
            or col.startswith("pathway_transfer_summary_")
            or col.startswith("time_h_")
            or col in base
        ]
    else:
        raise ValueError(f"Unknown gated transfer feature group: {feature_group}")
    if not selected:
        raise ValueError(f"Gated transfer feature group {feature_group} selected no columns")
    return selected

def _make_gate_estimator(spec: CandidateSpec):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gate_model = str(spec.params.get("gate_model", "logit_c2"))
    if gate_model == "logit_c2":
        from sklearn.linear_model import LogisticRegression

        return "classifier", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=2.0))
    if gate_model == "logit":
        from sklearn.linear_model import LogisticRegression

        return "classifier", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))
    if gate_model == "etr_benefit":
        from sklearn.ensemble import ExtraTreesRegressor

        return "regressor", ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.70,
            random_state=7,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown gated transfer gate model: {gate_model}")

def _gate_values_from_signal(
    gate_mode: str,
    gate_estimator,
    fit_features: np.ndarray,
    query_features: np.ndarray,
    target: np.ndarray,
    scale: float,
    min_positive_benefit: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if gate_mode == "classifier":
        gate_target = (np.asarray(target, dtype=float) > 0.0).astype(int)
        if len(set(gate_target.tolist())) < 2:
            fit_gate = np.full(len(fit_features), float(gate_target[0]), dtype=float)
            query_gate = np.full(len(query_features), float(gate_target[0]), dtype=float)
        else:
            gate_estimator.fit(fit_features, gate_target)
            fit_gate = gate_estimator.predict_proba(fit_features)[:, 1]
            query_gate = gate_estimator.predict_proba(query_features)[:, 1]
    elif gate_mode == "regressor":
        gate_target = np.asarray(target, dtype=float)
        gate_estimator.fit(fit_features, gate_target)
        fit_signal = np.asarray(gate_estimator.predict(fit_features), dtype=float)
        query_signal = np.asarray(gate_estimator.predict(query_features), dtype=float)
        denominator = float(np.std(fit_signal) + 1e-9)
        fit_gate = 1.0 / (1.0 + np.exp(-fit_signal / denominator))
        query_gate = 1.0 / (1.0 + np.exp(-query_signal / denominator))
        threshold = float(min_positive_benefit)
        if threshold > 0.0:
            fit_gate = np.where(fit_signal >= threshold, fit_gate, 0.0)
            query_gate = np.where(query_signal >= threshold, query_gate, 0.0)
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    return (
        np.clip(float(scale) * fit_gate, 0.0, 1.0),
        np.clip(float(scale) * query_gate, 0.0, 1.0),
    )

def _apply_a90_delta_confidence_gate(
    gate: np.ndarray,
    frame: pd.DataFrame,
    max_uncertainty: float | None = None,
    min_teacher_similarity: float | None = None,
    suppress_72h_risk: bool = False,
) -> np.ndarray:
    filtered = np.asarray(gate, dtype=float).copy()
    if max_uncertainty is not None and "conditioned_transfer_uncertainty" in frame.columns:
        uncertainty = pd.to_numeric(frame["conditioned_transfer_uncertainty"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)
        filtered = np.where(uncertainty <= float(max_uncertainty), filtered, 0.0)
    if min_teacher_similarity is not None and "teacher_topk_sim_mean" in frame.columns:
        similarity = pd.to_numeric(frame["teacher_topk_sim_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        filtered = np.where(similarity >= float(min_teacher_similarity), filtered, 0.0)
    if bool(suppress_72h_risk):
        time_h = pd.to_numeric(frame.get("time_h_numeric", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
        nitro = pd.to_numeric(frame.get("fg_nitro", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
        trihalo = pd.to_numeric(frame.get("fg_trihalomethyl", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
        halogen_fraction = pd.to_numeric(
            frame.get("domain_halogen_atom_fraction", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        bromine_iodine = pd.to_numeric(
            frame.get("domain_bromine_iodine_count", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        risk_mask = time_h.eq(72.0) & (
            nitro.gt(0.0) | trihalo.gt(0.0) | halogen_fraction.ge(0.30) | bromine_iodine.ge(2.0)
        )
        filtered = np.where(risk_mask.to_numpy(dtype=bool), 0.0, filtered)
    return np.clip(filtered, 0.0, 1.0)

def _fit_gated_transfer_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    raw_spec, no_spec = _gated_transfer_specs(spec)
    raw_valid, raw_fit = _fit_transfer_blend_fold(
        train,
        matrix,
        y,
        raw_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    no_valid, no_fit = _fit_transfer_blend_fold(
        train,
        matrix,
        y,
        no_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    if not bool(spec.params.get("transfer_enabled", True)):
        return (no_valid, no_fit) if return_fit_predictions else no_valid

    benefit = np.square(y[fit_idx] - no_fit) - np.square(y[fit_idx] - raw_fit)
    feature_group = str(spec.params.get("gate_feature_group", "all_context"))
    gate_columns = _gated_transfer_feature_columns(train, feature_group)
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    gate_fit, gate_valid = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train.iloc[fit_idx][gate_columns].to_numpy(dtype=float),
        query_features=train.iloc[valid_idx][gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    preds = no_valid + gate_valid * (raw_valid - no_valid)
    fit_preds = no_fit + gate_fit * (raw_fit - no_fit)
    return (np.asarray(preds, dtype=float), np.asarray(fit_preds, dtype=float)) if return_fit_predictions else np.asarray(preds, dtype=float)

def _fit_gated_transfer_and_predict_role(matrix: ModelMatrix, spec: CandidateSpec, role: str) -> np.ndarray:
    raw_spec, no_spec = _gated_transfer_specs(spec)
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    no_oof = _oof_gated_base_predictions(matrix, no_spec)
    raw_oof = _oof_gated_base_predictions(matrix, raw_spec)
    no_pred = _fit_transfer_blend_and_predict_role(matrix, no_spec, role)
    if not bool(spec.params.get("transfer_enabled", True)):
        return no_pred
    raw_pred = _fit_transfer_blend_and_predict_role(matrix, raw_spec, role)
    benefit = np.square(y - no_oof) - np.square(y - raw_oof)
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "all_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    role_frame = matrix.role(role)
    _, gate = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train[gate_columns].to_numpy(dtype=float),
        query_features=role_frame[gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    return np.asarray(no_pred + gate * (raw_pred - no_pred), dtype=float)

def _oof_gated_base_predictions(matrix: ModelMatrix, spec: CandidateSpec) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    oof = np.zeros(len(train), dtype=float)
    for fit_idx, valid_idx in make_cv_splits(train, cv_folds=5):
        oof[valid_idx] = _fit_transfer_blend_fold(train, matrix, y, spec, fit_idx, valid_idx)
    return oof

def _a90_gated_base_specs(spec: CandidateSpec) -> tuple[CandidateSpec, CandidateSpec]:
    def optional_column(value, default: str) -> str | None:
        if value is None:
            return None
        return str(value if value != "" else default)

    base_params = {
        "n_estimators": int(spec.params.get("n_estimators", 400)),
        "min_samples_leaf": int(spec.params.get("min_samples_leaf", 2)),
        "max_features": float(spec.params.get("max_features", 0.70)),
        "random_state": int(spec.params.get("random_state", 20260605)),
        "n_jobs": int(spec.params.get("n_jobs", -1)),
    }
    common = {
        "calibration_method": str(spec.params.get("calibration_method", "huber")),
        "descriptor_local_alpha": float(spec.params.get("descriptor_local_alpha", 0.90)),
        "descriptor_local_topk": int(spec.params.get("descriptor_local_topk", 1)),
        "descriptor_local_power": float(spec.params.get("descriptor_local_power", 2.0)),
        "descriptor_local_feature_group": str(spec.params.get("descriptor_local_feature_group", "domain_only")),
        "descriptor_local_scaler": str(spec.params.get("descriptor_local_scaler", "robust")),
        "descriptor_local_time_h": int(spec.params.get("descriptor_local_time_h", 48)),
        "descriptor_local_min_delta": float(spec.params.get("descriptor_local_min_delta", 0.0)),
    }
    full = CandidateSpec(
        f"{spec.candidate_id}__a90_full",
        "extratrees",
        base_params,
        feature_group=str(spec.params.get("full_feature_group", "domain_rdkit_tt_transfer")),
        residual_baseline_column=optional_column(spec.params.get("full_residual_baseline_column", "teacher_topk_y_wmean"), "teacher_topk_y_wmean"),
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 0.65)),
        residual_ensemble_weight=1.0,
        **common,
    )
    no_transfer = CandidateSpec(
        f"{spec.candidate_id}__a90_no_transfer",
        "extratrees",
        base_params,
        feature_group=str(spec.params.get("no_transfer_feature_group", "domain_rdkit_tt_no_transfer")),
        residual_baseline_column=(
            optional_column(spec.params.get("no_transfer_residual_baseline_column", "teacher_topk_y_wmean"), "teacher_topk_y_wmean")
            if bool(spec.params.get("no_transfer_uses_residual_baseline", True))
            else None
        ),
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 0.65)),
        residual_ensemble_weight=1.0,
        **common,
    )
    return full, no_transfer

def _a90_delta_anchor_spec(spec: CandidateSpec) -> CandidateSpec:
    base_params = {
        "n_estimators": int(spec.params.get("n_estimators", 400)),
        "min_samples_leaf": int(spec.params.get("min_samples_leaf", 2)),
        "max_features": float(spec.params.get("max_features", 0.70)),
        "random_state": int(spec.params.get("random_state", 20260605)),
        "n_jobs": int(spec.params.get("n_jobs", -1)),
    }
    return CandidateSpec(
        f"{spec.candidate_id}__anchor",
        "extratrees",
        base_params,
        feature_group=str(spec.params.get("anchor_feature_group", "domain_rdkit_tt_transfer")),
        residual_baseline_column=str(spec.params.get("anchor_residual_baseline_column", "teacher_topk_y_wmean"))
        if spec.params.get("anchor_residual_baseline_column", "teacher_topk_y_wmean")
        else None,
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 0.65)),
        residual_ensemble_weight=1.0,
        calibration_method=str(spec.params.get("calibration_method", "huber")),
        descriptor_local_alpha=float(spec.params.get("descriptor_local_alpha", 0.90)),
        descriptor_local_topk=int(spec.params.get("descriptor_local_topk", 1)),
        descriptor_local_power=float(spec.params.get("descriptor_local_power", 2.0)),
        descriptor_local_feature_group=str(spec.params.get("descriptor_local_feature_group", "domain_only")),
        descriptor_local_scaler=str(spec.params.get("descriptor_local_scaler", "robust")),
        descriptor_local_time_h=int(spec.params.get("descriptor_local_time_h", 48)),
        descriptor_local_min_delta=float(spec.params.get("descriptor_local_min_delta", 0.0)),
    )

def _a90_delta_gated_delta_spec(spec: CandidateSpec, use_transfer: bool) -> CandidateSpec:
    delta_family = str(spec.params.get("delta_family", "gated_transfer"))
    if delta_family in {"conditioned_clean_mid", "conditioned_ridge"}:
        if delta_family == "conditioned_ridge":
            base_params = {"alphas": tuple(spec.params.get("delta_ridge_alphas", (0.1, 1.0, 10.0, 100.0)))}
        else:
            base_params = {
                "n_estimators": int(spec.params.get("delta_n_estimators", 400)),
                "min_samples_leaf": int(spec.params.get("delta_min_samples_leaf", 2)),
                "max_features": float(spec.params.get("delta_max_features", 0.85)),
                "random_state": int(spec.params.get("delta_random_state", 20260605)),
                "n_jobs": int(spec.params.get("delta_n_jobs", -1)),
            }
        common = {
            "calibration_method": str(spec.params.get("delta_calibration_method", "huber")),
            "descriptor_local_alpha": float(spec.params.get("delta_descriptor_local_alpha", 1.00)),
            "descriptor_local_topk": int(spec.params.get("delta_descriptor_local_topk", 1)),
            "descriptor_local_power": float(spec.params.get("delta_descriptor_local_power", 2.0)),
            "descriptor_local_feature_group": str(spec.params.get("delta_descriptor_local_feature_group", "domain_only")),
            "descriptor_local_scaler": str(spec.params.get("delta_descriptor_local_scaler", "robust")),
            "descriptor_local_time_h": int(spec.params.get("delta_descriptor_local_time_h", 48)),
            "descriptor_local_min_delta": float(spec.params.get("delta_descriptor_local_min_delta", 0.0)),
        }
        if bool(use_transfer):
            return CandidateSpec(
                f"{spec.candidate_id}__delta_conditioned_full",
                "ridge" if delta_family == "conditioned_ridge" else "extratrees",
                base_params,
                feature_group="domain_rdkit_tt_conditioned_transfer",
                residual_baseline_column="conditioned_transfer_clean_mid",
                residual_shrinkage=float(spec.params.get("delta_residual_shrinkage", 0.25)),
                residual_ensemble_weight=1.0,
                **common,
            )
        return CandidateSpec(
            f"{spec.candidate_id}__delta_conditioned_no",
            "ridge" if delta_family == "conditioned_ridge" else "extratrees",
            base_params,
            feature_group="domain_rdkit_tt_conditioned_no_transfer",
            residual_ensemble_weight=1.0,
            **common,
        )
    if delta_family != "gated_transfer":
        raise ValueError(f"Unknown a90 delta family: {delta_family}")
    return CandidateSpec(
        f"{spec.candidate_id}__delta_{'full' if use_transfer else 'no'}",
        "gated_transfer",
        {
            "et_n_estimators": int(spec.params.get("delta_et_n_estimators", 700)),
            "et_min_samples_leaf": int(spec.params.get("delta_et_min_samples_leaf", 1)),
            "et_max_features": float(spec.params.get("delta_et_max_features", 0.65)),
            "et_random_state": int(spec.params.get("delta_et_random_state", 20260605)),
            "svr_C": float(spec.params.get("delta_svr_C", 16.0)),
            "svr_epsilon": float(spec.params.get("delta_svr_epsilon", 0.005)),
            "svr_gamma": spec.params.get("delta_svr_gamma", "scale"),
            "blend_weight": float(spec.params.get("delta_blend_weight", 0.245)),
            "blend_residual_weight": float(spec.params.get("delta_blend_residual_weight", 0.33)),
            "gate_feature_group": str(spec.params.get("gate_feature_group", "all_context")),
            "gate_model": str(spec.params.get("gate_model", "etr_benefit")),
            "gate_scale": float(spec.params.get("gate_scale", 1.0)),
            "gate_min_positive_benefit": float(spec.params.get("gate_min_positive_benefit", 0.0)),
            "transfer_enabled": bool(use_transfer),
        },
        feature_group="gated_transfer_all_context" if use_transfer else "gated_transfer_all_context_no_transfer",
    )

def _a90_delta_gated_base_specs(
    spec: CandidateSpec,
) -> tuple[CandidateSpec, CandidateSpec, CandidateSpec, CandidateSpec]:
    anchor_spec = _a90_delta_anchor_spec(spec)
    no_params = dict(spec.params)
    no_params["anchor_feature_group"] = "domain_rdkit_tt_no_transfer"
    no_params["anchor_residual_baseline_column"] = (
        "teacher_topk_y_wmean" if bool(spec.params.get("no_transfer_uses_residual_baseline", True)) else None
    )
    no_anchor_spec = _a90_delta_anchor_spec(
        CandidateSpec(f"{spec.candidate_id}__no_anchor_cfg", spec.family, no_params)
    )
    delta_spec = _a90_delta_gated_delta_spec(spec, use_transfer=bool(spec.params.get("transfer_enabled", True)))
    delta_no_spec = _a90_delta_gated_delta_spec(spec, use_transfer=False)
    return anchor_spec, no_anchor_spec, delta_spec, delta_no_spec

def _fit_raw_spec_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    columns = feature_columns_for_spec(matrix, spec)
    return _fit_predict_raw_candidate(
        train,
        _feature_matrix_for_spec(train, columns, spec),
        y,
        residual_baseline_for_spec(train, spec),
        spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )

def _fit_a90_delta_gated_transfer_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    anchor_spec, no_anchor_spec, delta_spec, delta_no_spec = _a90_delta_gated_base_specs(spec)
    active_anchor = anchor_spec if bool(spec.params.get("transfer_enabled", True)) else no_anchor_spec
    anchor_valid, anchor_fit = _fit_raw_spec_fold(train, matrix, y, active_anchor, fit_idx, valid_idx)
    if not bool(spec.params.get("transfer_enabled", True)):
        return (anchor_valid, anchor_fit) if return_fit_predictions else anchor_valid

    if delta_spec.family == "gated_transfer":
        delta_valid, delta_fit = _fit_gated_transfer_fold(
            train,
            matrix,
            y,
            delta_spec,
            fit_idx,
            valid_idx,
            return_fit_predictions=True,
        )
        delta_no_valid, delta_no_fit = _fit_gated_transfer_fold(
            train,
            matrix,
            y,
            delta_no_spec,
            fit_idx,
            valid_idx,
            return_fit_predictions=True,
        )
    else:
        delta_valid, delta_fit = _fit_raw_spec_fold(train, matrix, y, delta_spec, fit_idx, valid_idx)
        delta_no_valid, delta_no_fit = _fit_raw_spec_fold(train, matrix, y, delta_no_spec, fit_idx, valid_idx)
    fit_delta = np.asarray(delta_fit, dtype=float) - np.asarray(delta_no_fit, dtype=float)
    valid_delta = np.asarray(delta_valid, dtype=float) - np.asarray(delta_no_valid, dtype=float)
    benefit = np.square(y[fit_idx] - anchor_fit) - np.square(y[fit_idx] - (anchor_fit + fit_delta))
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "all_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    gate_fit, gate_valid = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train.iloc[fit_idx][gate_columns].to_numpy(dtype=float),
        query_features=train.iloc[valid_idx][gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("anchor_delta_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    gate_fit = _apply_a90_delta_confidence_gate(
        gate_fit,
        train.iloc[fit_idx],
        max_uncertainty=spec.params.get("delta_max_uncertainty"),
        min_teacher_similarity=spec.params.get("delta_min_teacher_similarity"),
        suppress_72h_risk=bool(spec.params.get("suppress_72h_risk", False)),
    )
    gate_valid = _apply_a90_delta_confidence_gate(
        gate_valid,
        train.iloc[valid_idx],
        max_uncertainty=spec.params.get("delta_max_uncertainty"),
        min_teacher_similarity=spec.params.get("delta_min_teacher_similarity"),
        suppress_72h_risk=bool(spec.params.get("suppress_72h_risk", False)),
    )
    valid_pred = anchor_valid + gate_valid * valid_delta
    fit_pred = anchor_fit + gate_fit * fit_delta
    return (np.asarray(valid_pred, dtype=float), np.asarray(fit_pred, dtype=float)) if return_fit_predictions else np.asarray(valid_pred, dtype=float)

def _oof_a90_delta_anchor_predictions(matrix: ModelMatrix, spec: CandidateSpec) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    columns = feature_columns_for_spec(matrix, spec)
    x = _feature_matrix_for_spec(train, columns, spec)
    baseline = residual_baseline_for_spec(train, spec)
    return _oof_raw_predictions(train, x, y, baseline, spec, cv_folds=5)

def _fit_a90_delta_anchor_and_predict_role(matrix: ModelMatrix, spec: CandidateSpec, role: str) -> np.ndarray:
    _, predictions = fit_selected_and_evaluate(matrix, spec)
    return predictions[predictions["split_role"].astype(str).eq(str(role))]["y_pred"].to_numpy(dtype=float)

def _fit_a90_delta_gated_transfer_and_predict_role(matrix: ModelMatrix, spec: CandidateSpec, role: str) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    role_frame = matrix.role(role)
    anchor_spec, no_anchor_spec, delta_spec, delta_no_spec = _a90_delta_gated_base_specs(spec)
    active_anchor = anchor_spec if bool(spec.params.get("transfer_enabled", True)) else no_anchor_spec
    anchor_pred = _fit_a90_delta_anchor_and_predict_role(matrix, active_anchor, role)
    if not bool(spec.params.get("transfer_enabled", True)):
        return anchor_pred

    anchor_oof = _oof_a90_delta_anchor_predictions(matrix, anchor_spec)
    if delta_spec.family == "gated_transfer":
        delta_oof = _oof_gated_base_predictions(matrix, delta_spec)
        delta_no_oof = _oof_gated_base_predictions(matrix, delta_no_spec)
    else:
        delta_oof = _oof_a90_delta_anchor_predictions(matrix, delta_spec)
        delta_no_oof = _oof_a90_delta_anchor_predictions(matrix, delta_no_spec)
    train_delta = delta_oof - delta_no_oof
    benefit = np.square(y - anchor_oof) - np.square(y - (anchor_oof + train_delta))
    if delta_spec.family == "gated_transfer":
        delta_pred = _fit_gated_transfer_and_predict_role(matrix, delta_spec, role)
        delta_no_pred = _fit_gated_transfer_and_predict_role(matrix, delta_no_spec, role)
    else:
        delta_pred = _fit_a90_delta_anchor_and_predict_role(matrix, delta_spec, role)
        delta_no_pred = _fit_a90_delta_anchor_and_predict_role(matrix, delta_no_spec, role)
    role_delta = delta_pred - delta_no_pred
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "all_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    _, gate = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train[gate_columns].to_numpy(dtype=float),
        query_features=role_frame[gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("anchor_delta_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    gate = _apply_a90_delta_confidence_gate(
        gate,
        role_frame,
        max_uncertainty=spec.params.get("delta_max_uncertainty"),
        min_teacher_similarity=spec.params.get("delta_min_teacher_similarity"),
        suppress_72h_risk=bool(spec.params.get("suppress_72h_risk", False)),
    )
    return np.asarray(anchor_pred + gate * role_delta, dtype=float)

def _risk_adjusted_gated_base_specs(spec: CandidateSpec) -> tuple[CandidateSpec, CandidateSpec]:
    base_params = {
        "n_estimators": int(spec.params.get("n_estimators", 400)),
        "min_samples_leaf": int(spec.params.get("min_samples_leaf", 2)),
        "max_features": float(spec.params.get("max_features", 0.85)),
        "random_state": int(spec.params.get("random_state", 20260605)),
        "n_jobs": int(spec.params.get("n_jobs", -1)),
    }
    common = {
        "calibration_method": str(spec.params.get("calibration_method", "huber")),
        "descriptor_local_alpha": float(spec.params.get("descriptor_local_alpha", 1.00)),
        "descriptor_local_topk": int(spec.params.get("descriptor_local_topk", 1)),
        "descriptor_local_power": float(spec.params.get("descriptor_local_power", 2.0)),
        "descriptor_local_feature_group": str(spec.params.get("descriptor_local_feature_group", "domain_only")),
        "descriptor_local_scaler": str(spec.params.get("descriptor_local_scaler", "robust")),
        "descriptor_local_time_h": int(spec.params.get("descriptor_local_time_h", 48)),
        "descriptor_local_min_delta": float(spec.params.get("descriptor_local_min_delta", 0.0)),
    }
    full = CandidateSpec(
        f"{spec.candidate_id}__risk_adjusted_full",
        "extratrees",
        base_params,
        feature_group="domain_rdkit_tt_risk_adjusted_transfer",
        residual_baseline_column="risk_adjusted_transfer_wmean",
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 0.65)),
        residual_ensemble_weight=1.0,
        **common,
    )
    no_transfer = CandidateSpec(
        f"{spec.candidate_id}__risk_adjusted_no_transfer",
        "extratrees",
        base_params,
        feature_group="domain_rdkit_tt_risk_adjusted_no_transfer",
        residual_baseline_column=(
            "risk_adjusted_transfer_wmean" if bool(spec.params.get("no_transfer_uses_residual_baseline", True)) else None
        ),
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 0.65)),
        residual_ensemble_weight=1.0,
        **common,
    )
    return full, no_transfer

def _disagreement_gated_base_specs(spec: CandidateSpec) -> tuple[CandidateSpec, CandidateSpec]:
    base_params = {
        "n_estimators": int(spec.params.get("n_estimators", 400)),
        "min_samples_leaf": int(spec.params.get("min_samples_leaf", 2)),
        "max_features": float(spec.params.get("max_features", 0.85)),
        "random_state": int(spec.params.get("random_state", 20260605)),
        "n_jobs": int(spec.params.get("n_jobs", -1)),
    }
    common = {
        "calibration_method": str(spec.params.get("calibration_method", "huber")),
        "descriptor_local_alpha": float(spec.params.get("descriptor_local_alpha", 1.00)),
        "descriptor_local_topk": int(spec.params.get("descriptor_local_topk", 1)),
        "descriptor_local_power": float(spec.params.get("descriptor_local_power", 2.0)),
        "descriptor_local_feature_group": str(spec.params.get("descriptor_local_feature_group", "domain_only")),
        "descriptor_local_scaler": str(spec.params.get("descriptor_local_scaler", "robust")),
        "descriptor_local_time_h": int(spec.params.get("descriptor_local_time_h", 48)),
        "descriptor_local_min_delta": float(spec.params.get("descriptor_local_min_delta", 0.0)),
    }
    full = CandidateSpec(
        f"{spec.candidate_id}__disagreement_full",
        "extratrees",
        base_params,
        feature_group="domain_rdkit_tt_disagreement_transfer",
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 1.0)),
        residual_ensemble_weight=1.0,
        **common,
    )
    no_transfer = CandidateSpec(
        f"{spec.candidate_id}__disagreement_no_transfer",
        "extratrees",
        base_params,
        feature_group="domain_rdkit_tt_disagreement_no_transfer",
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 1.0)),
        residual_ensemble_weight=1.0,
        **common,
    )
    return full, no_transfer

def _fit_a90_gated_transfer_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    full_spec, no_spec = _a90_gated_base_specs(spec)

    full_columns = feature_columns_for_spec(matrix, full_spec)
    no_columns = feature_columns_for_spec(matrix, no_spec)
    full_valid, full_fit = _fit_predict_raw_candidate(
        train,
        _feature_matrix_for_spec(train, full_columns, full_spec),
        y,
        residual_baseline_for_spec(train, full_spec),
        full_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    no_valid, no_fit = _fit_predict_raw_candidate(
        train,
        _feature_matrix_for_spec(train, no_columns, no_spec),
        y,
        residual_baseline_for_spec(train, no_spec),
        no_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    if not bool(spec.params.get("transfer_enabled", True)):
        return (no_valid, no_fit) if return_fit_predictions else no_valid

    benefit = np.square(y[fit_idx] - no_fit) - np.square(y[fit_idx] - full_fit)
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "risk_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    gate_fit, gate_valid = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train.iloc[fit_idx][gate_columns].to_numpy(dtype=float),
        query_features=train.iloc[valid_idx][gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    valid_pred = no_valid + gate_valid * (full_valid - no_valid)
    fit_pred = no_fit + gate_fit * (full_fit - no_fit)
    return (np.asarray(valid_pred, dtype=float), np.asarray(fit_pred, dtype=float)) if return_fit_predictions else np.asarray(valid_pred, dtype=float)

def _oof_a90_gated_base_predictions(matrix: ModelMatrix, spec: CandidateSpec) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    columns = feature_columns_for_spec(matrix, spec)
    x = _feature_matrix_for_spec(train, columns, spec)
    baseline = residual_baseline_for_spec(train, spec)
    return _oof_raw_predictions(train, x, y, baseline, spec, cv_folds=5)

def _fit_a90_gated_transfer_and_predict_role(matrix: ModelMatrix, spec: CandidateSpec, role: str) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    role_frame = matrix.role(role)
    full_spec, no_spec = _a90_gated_base_specs(spec)

    no_metrics, no_predictions = fit_selected_and_evaluate(matrix, no_spec)
    del no_metrics
    no_pred = (
        no_predictions[no_predictions["split_role"].astype(str).eq(str(role))]
        ["y_pred"]
        .to_numpy(dtype=float)
    )
    if not bool(spec.params.get("transfer_enabled", True)):
        return no_pred

    full_metrics, full_predictions = fit_selected_and_evaluate(matrix, full_spec)
    del full_metrics
    full_pred = (
        full_predictions[full_predictions["split_role"].astype(str).eq(str(role))]
        ["y_pred"]
        .to_numpy(dtype=float)
    )
    if len(full_pred) != len(role_frame) or len(no_pred) != len(role_frame):
        raise ValueError("A90 gated transfer prediction alignment failed")

    no_oof = _oof_a90_gated_base_predictions(matrix, no_spec)
    full_oof = _oof_a90_gated_base_predictions(matrix, full_spec)
    benefit = np.square(y - no_oof) - np.square(y - full_oof)
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "risk_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    _, gate = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train[gate_columns].to_numpy(dtype=float),
        query_features=role_frame[gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    return np.asarray(no_pred + gate * (full_pred - no_pred), dtype=float)

def _fit_risk_adjusted_gated_transfer_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    full_spec, no_spec = _risk_adjusted_gated_base_specs(spec)

    full_columns = feature_columns_for_spec(matrix, full_spec)
    no_columns = feature_columns_for_spec(matrix, no_spec)
    full_valid, full_fit = _fit_predict_raw_candidate(
        train,
        _feature_matrix_for_spec(train, full_columns, full_spec),
        y,
        residual_baseline_for_spec(train, full_spec),
        full_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    no_valid, no_fit = _fit_predict_raw_candidate(
        train,
        _feature_matrix_for_spec(train, no_columns, no_spec),
        y,
        residual_baseline_for_spec(train, no_spec),
        no_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    if not bool(spec.params.get("transfer_enabled", True)):
        return (no_valid, no_fit) if return_fit_predictions else no_valid

    benefit = np.square(y[fit_idx] - no_fit) - np.square(y[fit_idx] - full_fit)
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "risk_adjusted_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    gate_fit, gate_valid = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train.iloc[fit_idx][gate_columns].to_numpy(dtype=float),
        query_features=train.iloc[valid_idx][gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    valid_pred = no_valid + gate_valid * (full_valid - no_valid)
    fit_pred = no_fit + gate_fit * (full_fit - no_fit)
    return (np.asarray(valid_pred, dtype=float), np.asarray(fit_pred, dtype=float)) if return_fit_predictions else np.asarray(valid_pred, dtype=float)

def _fit_risk_adjusted_gated_transfer_and_predict_role(matrix: ModelMatrix, spec: CandidateSpec, role: str) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    role_frame = matrix.role(role)
    full_spec, no_spec = _risk_adjusted_gated_base_specs(spec)

    no_metrics, no_predictions = fit_selected_and_evaluate(matrix, no_spec)
    del no_metrics
    no_pred = no_predictions[no_predictions["split_role"].astype(str).eq(str(role))]["y_pred"].to_numpy(dtype=float)
    if not bool(spec.params.get("transfer_enabled", True)):
        return no_pred

    full_metrics, full_predictions = fit_selected_and_evaluate(matrix, full_spec)
    del full_metrics
    full_pred = full_predictions[full_predictions["split_role"].astype(str).eq(str(role))]["y_pred"].to_numpy(dtype=float)
    if len(full_pred) != len(role_frame) or len(no_pred) != len(role_frame):
        raise ValueError("risk-adjusted gated transfer prediction alignment failed")

    no_oof = _oof_risk_adjusted_gated_base_predictions(matrix, no_spec)
    full_oof = _oof_risk_adjusted_gated_base_predictions(matrix, full_spec)
    benefit = np.square(y - no_oof) - np.square(y - full_oof)
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "risk_adjusted_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    _, gate = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train[gate_columns].to_numpy(dtype=float),
        query_features=role_frame[gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    return np.asarray(no_pred + gate * (full_pred - no_pred), dtype=float)

def _oof_risk_adjusted_gated_base_predictions(matrix: ModelMatrix, spec: CandidateSpec) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    columns = feature_columns_for_spec(matrix, spec)
    x = _feature_matrix_for_spec(train, columns, spec)
    baseline = residual_baseline_for_spec(train, spec)
    return _oof_raw_predictions(train, x, y, baseline, spec, cv_folds=5)

def _fit_disagreement_gated_transfer_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    full_spec, no_spec = _disagreement_gated_base_specs(spec)

    full_columns = feature_columns_for_spec(matrix, full_spec)
    no_columns = feature_columns_for_spec(matrix, no_spec)
    full_valid, full_fit = _fit_predict_raw_candidate(
        train,
        _feature_matrix_for_spec(train, full_columns, full_spec),
        y,
        residual_baseline_for_spec(train, full_spec),
        full_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    no_valid, no_fit = _fit_predict_raw_candidate(
        train,
        _feature_matrix_for_spec(train, no_columns, no_spec),
        y,
        residual_baseline_for_spec(train, no_spec),
        no_spec,
        fit_idx,
        valid_idx,
        return_fit_predictions=True,
    )
    if not bool(spec.params.get("transfer_enabled", True)):
        return (no_valid, no_fit) if return_fit_predictions else no_valid

    benefit = np.square(y[fit_idx] - no_fit) - np.square(y[fit_idx] - full_fit)
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "disagreement_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    gate_fit, gate_valid = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train.iloc[fit_idx][gate_columns].to_numpy(dtype=float),
        query_features=train.iloc[valid_idx][gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    valid_pred = no_valid + gate_valid * (full_valid - no_valid)
    fit_pred = no_fit + gate_fit * (full_fit - no_fit)
    return (np.asarray(valid_pred, dtype=float), np.asarray(fit_pred, dtype=float)) if return_fit_predictions else np.asarray(valid_pred, dtype=float)

def _fit_disagreement_gated_transfer_and_predict_role(matrix: ModelMatrix, spec: CandidateSpec, role: str) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    role_frame = matrix.role(role)
    full_spec, no_spec = _disagreement_gated_base_specs(spec)

    no_metrics, no_predictions = fit_selected_and_evaluate(matrix, no_spec)
    del no_metrics
    no_pred = no_predictions[no_predictions["split_role"].astype(str).eq(str(role))]["y_pred"].to_numpy(dtype=float)
    if not bool(spec.params.get("transfer_enabled", True)):
        return no_pred

    full_metrics, full_predictions = fit_selected_and_evaluate(matrix, full_spec)
    del full_metrics
    full_pred = full_predictions[full_predictions["split_role"].astype(str).eq(str(role))]["y_pred"].to_numpy(dtype=float)
    if len(full_pred) != len(role_frame) or len(no_pred) != len(role_frame):
        raise ValueError("disagreement gated transfer prediction alignment failed")

    no_oof = _oof_disagreement_gated_base_predictions(matrix, no_spec)
    full_oof = _oof_disagreement_gated_base_predictions(matrix, full_spec)
    benefit = np.square(y - no_oof) - np.square(y - full_oof)
    gate_columns = _gated_transfer_feature_columns(train, str(spec.params.get("gate_feature_group", "disagreement_context")))
    gate_mode, gate_estimator = _make_gate_estimator(spec)
    _, gate = _gate_values_from_signal(
        gate_mode=gate_mode,
        gate_estimator=gate_estimator,
        fit_features=train[gate_columns].to_numpy(dtype=float),
        query_features=role_frame[gate_columns].to_numpy(dtype=float),
        target=benefit,
        scale=float(spec.params.get("gate_scale", 1.0)),
        min_positive_benefit=float(spec.params.get("gate_min_positive_benefit", 0.0)),
    )
    return np.asarray(no_pred + gate * (full_pred - no_pred), dtype=float)

def _oof_disagreement_gated_base_predictions(matrix: ModelMatrix, spec: CandidateSpec) -> np.ndarray:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    columns = feature_columns_for_spec(matrix, spec)
    x = _feature_matrix_for_spec(train, columns, spec)
    baseline = residual_baseline_for_spec(train, spec)
    return _oof_raw_predictions(train, x, y, baseline, spec, cv_folds=5)

def _risk_routed_base_specs(spec: CandidateSpec) -> tuple[CandidateSpec, CandidateSpec]:
    base_params = {
        "n_estimators": int(spec.params.get("n_estimators", 400)),
        "min_samples_leaf": int(spec.params.get("min_samples_leaf", 2)),
        "max_features": float(spec.params.get("max_features", 0.70)),
        "random_state": int(spec.params.get("random_state", 20260605)),
        "n_jobs": int(spec.params.get("n_jobs", -1)),
    }
    common = {
        "calibration_method": str(spec.params.get("calibration_method", "huber")),
        "descriptor_local_alpha": float(spec.params.get("descriptor_local_alpha", 0.90)),
        "descriptor_local_topk": int(spec.params.get("descriptor_local_topk", 1)),
        "descriptor_local_power": float(spec.params.get("descriptor_local_power", 2.0)),
        "descriptor_local_feature_group": str(spec.params.get("descriptor_local_feature_group", "domain_only")),
        "descriptor_local_scaler": str(spec.params.get("descriptor_local_scaler", "robust")),
        "descriptor_local_time_h": int(spec.params.get("descriptor_local_time_h", 48)),
        "descriptor_local_min_delta": float(spec.params.get("descriptor_local_min_delta", 0.0)),
    }
    full = CandidateSpec(
        f"{spec.candidate_id}__risk_router_full",
        "extratrees",
        base_params,
        feature_group=str(spec.params.get("full_feature_group", "domain_rdkit_tt_transfer")),
        residual_baseline_column=str(spec.params.get("full_residual_baseline_column", "teacher_topk_y_wmean")),
        residual_shrinkage=float(spec.params.get("residual_shrinkage", 0.65)),
        residual_ensemble_weight=1.0,
        **common,
    )
    no_transfer = CandidateSpec(
        f"{spec.candidate_id}__risk_router_no_transfer",
        "extratrees",
        base_params,
        feature_group=str(spec.params.get("no_transfer_feature_group", "domain_rdkit_tt_no_transfer")),
        residual_shrinkage=1.0,
        residual_ensemble_weight=1.0,
        **common,
    )
    return full, no_transfer

def _route_transfer_predictions_by_risk(
    frame: pd.DataFrame,
    full_pred: np.ndarray,
    no_transfer_pred: np.ndarray,
) -> np.ndarray:
    full = np.asarray(full_pred, dtype=float)
    no_transfer = np.asarray(no_transfer_pred, dtype=float)
    if len(full) != len(no_transfer) or len(full) != len(frame):
        raise ValueError("Risk-routed transfer prediction alignment failed")
    return np.where(_transfer_risk_suppression_mask(frame), no_transfer, full)

def _fit_risk_routed_transfer_fold(
    train: pd.DataFrame,
    matrix: ModelMatrix,
    y: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    full_spec, no_spec = _risk_routed_base_specs(spec)
    no_valid, no_fit = _fit_raw_spec_fold(train, matrix, y, no_spec, fit_idx, valid_idx)
    if not bool(spec.params.get("transfer_enabled", True)):
        return (no_valid, no_fit) if return_fit_predictions else no_valid
    full_valid, full_fit = _fit_raw_spec_fold(train, matrix, y, full_spec, fit_idx, valid_idx)
    valid_pred = _route_transfer_predictions_by_risk(train.iloc[valid_idx], full_valid, no_valid)
    fit_pred = _route_transfer_predictions_by_risk(train.iloc[fit_idx], full_fit, no_fit)
    return (np.asarray(valid_pred, dtype=float), np.asarray(fit_pred, dtype=float)) if return_fit_predictions else np.asarray(valid_pred, dtype=float)

def _fit_risk_routed_transfer_and_predict_role(matrix: ModelMatrix, spec: CandidateSpec, role: str) -> np.ndarray:
    role_frame = matrix.role(role)
    full_spec, no_spec = _risk_routed_base_specs(spec)
    no_metrics, no_predictions = fit_selected_and_evaluate(matrix, no_spec)
    del no_metrics
    no_pred = no_predictions[no_predictions["split_role"].astype(str).eq(str(role))]["y_pred"].to_numpy(dtype=float)
    if not bool(spec.params.get("transfer_enabled", True)):
        return no_pred
    full_metrics, full_predictions = fit_selected_and_evaluate(matrix, full_spec)
    del full_metrics
    full_pred = full_predictions[full_predictions["split_role"].astype(str).eq(str(role))]["y_pred"].to_numpy(dtype=float)
    return _route_transfer_predictions_by_risk(role_frame, full_pred, no_pred)

def _prioritize_candidates(candidates: list[CandidateSpec], candidate_ids: tuple[str, ...]) -> list[CandidateSpec]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    priority_ids = set(candidate_ids)
    prioritized = [by_id[candidate_id] for candidate_id in candidate_ids if candidate_id in by_id]
    remaining = [candidate for candidate in candidates if candidate.candidate_id not in priority_ids]
    return prioritized + remaining

ESTIMATOR_PARAM_METADATA_KEYS = {
    "suppress_transfer_72h_risk",
    "transfer_risk_suppression",
}

def _estimator_params(spec: CandidateSpec) -> dict:
    return {key: value for key, value in spec.params.items() if key not in ESTIMATOR_PARAM_METADATA_KEYS}

def build_candidates(mode: str = "quick", max_candidates: int | None = None, gpu: int = 0) -> list[CandidateSpec]:
    xgb_device = "cuda" if int(gpu) == 0 else f"cuda:{int(gpu)}"
    xgb_base = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": xgb_device,
        "eval_metric": "rmse",
        "random_state": 20260605,
        "n_jobs": 1,
        "verbosity": 0,
    }
    cat_base = {
        "loss_function": "RMSE",
        "task_type": "GPU",
        "devices": str(int(gpu)),
        "random_seed": 20260605,
        "verbose": False,
        "allow_writing_files": False,
    }
    depth_grid = [2, 3, 4] if mode == "smoke" else [2, 3, 4, 5]
    candidates: list[CandidateSpec] = [
        CandidateSpec(
            "base_xgboost_raw_default",
            "xgboost",
            {**xgb_base, "n_estimators": 100},
        ),
        CandidateSpec(
            "base_catboost_raw_default",
            "catboost",
            cat_base,
        ),
        CandidateSpec(
            "base_catboost_tiny_raw_default",
            "catboost",
            {**cat_base, "iterations": 24, "depth": 2, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
        ),
        CandidateSpec(
            "base_extratrees_raw_default",
            "extratrees",
            {"random_state": 20260605, "n_jobs": -1},
        ),
        CandidateSpec(
            "base_randomforest_raw_default",
            "randomforest",
            {"random_state": 20260605, "n_jobs": -1},
        ),
        CandidateSpec(
            "base_svr_raw_default",
            "svr",
            {},
        ),
    ]
    for depth in depth_grid:
        params = dict(xgb_base)
        params.update(
            {
                "n_estimators": 40 if mode == "smoke" else 160,
                "max_depth": depth,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "min_child_weight": 2.0,
            }
        )
        candidates.append(CandidateSpec(f"xgb_d{depth}_lr005", "xgboost", params))
        if mode != "smoke" and depth in {3, 4}:
            for alpha in (0.20, 0.40, 0.60):
                for topk in (1, 3):
                    candidates.append(
                        CandidateSpec(
                            f"xgb_d{depth}_lr005_local_a{int(alpha * 100):02d}_k{topk}",
                            "xgboost",
                            params,
                            local_alpha=alpha,
                            local_topk=topk,
                            local_power=2.0,
                        )
                    )
            for alpha in (0.20, 0.40):
                candidates.append(
                    CandidateSpec(
                        f"xgb_d{depth}_lr005_time_a{int(alpha * 100):02d}_k3",
                        "xgboost",
                        params,
                        local_alpha=alpha,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                    )
                )
            for alpha in (0.20, 0.35):
                candidates.append(
                    CandidateSpec(
                        f"xgb_d{depth}_lr005_q25_a{int(alpha * 100):02d}_k5",
                        "xgboost",
                        params,
                        local_alpha=alpha,
                        local_topk=5,
                        local_power=2.0,
                        local_statistic="quantile",
                        local_quantile=0.25,
                    )
                )
    if mode != "smoke":
        params = dict(xgb_base)
        params.update(
            {
                "n_estimators": 160,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "min_child_weight": 2.0,
            }
        )
        for alpha in (0.90, 1.00):
            candidates.append(
                CandidateSpec(
                    f"xgb_d4_lr005_time_a40_desc48_domain_a{int(alpha * 100):03d}",
                    "xgboost",
                    params,
                    local_alpha=0.40,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=alpha,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                )
            )
        for alpha in (0.90, 1.00):
            candidates.append(
                CandidateSpec(
                    f"xgb_d4_lr005_time_a20_desc48_domain_a{int(alpha * 100):03d}",
                    "xgboost",
                    params,
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=alpha,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                )
            )
        for topk in (1, 3):
            candidates.append(
                CandidateSpec(
                    f"xgb_d4_lr005_local_a20_k{topk}_desc48_domain_a100",
                    "xgboost",
                    params,
                    local_alpha=0.20,
                    local_topk=topk,
                    local_power=2.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                )
            )
        candidates.append(
            CandidateSpec(
                "xgb_d4_lr005_local_a20_k1_time_a20_desc48_domain_a100",
                "xgboost",
                params,
                local_alpha=0.20,
                local_topk=1,
                local_power=2.0,
                local_time_scale=24.0,
                descriptor_local_alpha=1.00,
                descriptor_local_topk=1,
                descriptor_local_power=2.0,
                descriptor_local_feature_group="domain_only",
                descriptor_local_scaler="robust",
                descriptor_local_time_h=48,
                descriptor_local_min_delta=0.0,
            )
        )
        candidates.extend(
            [
                CandidateSpec(
                    "xgb_d4_lr005_domain_rdkit_no_transfer_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="domain_rdkit_no_transfer",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_domain_rdkit_transfer_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="domain_rdkit_transfer",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_domain_rdkit_no_transfer_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_domain_rdkit_profile_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="domain_rdkit_profile_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_domain_rdkit_profile_resid10_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="domain_rdkit_profile_transfer",
                    residual_baseline_column="profile_transfer_nci_y_mean",
                    residual_shrinkage=0.10,
                    residual_ensemble_weight=0.50,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_domain_rdkit_local_profile_resid50_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=0.50,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_domain_rdkit_local_profile_resid75_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n160_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 160},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n160_domain_rdkit_local_profile_resid100_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 160},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n160_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_resid75_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q50_resid75_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a20_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.10,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a05_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.05,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_k1_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.10,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_k7_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.10,
                    local_topk=7,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k3_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k2_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=2,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k6_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=6,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a12_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.12,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a18_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.18,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d3_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 3, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d5_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 5, "n_estimators": 260},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n300_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 300},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_atom_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n300_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 300},
                    feature_group="domain_rdkit_atom_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr003_n450_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.03, "n_estimators": 450},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n420_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 420},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_tt_local_profile_clean_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_tt_local_profile_clean_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a10_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.10,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_nodesc_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                ),
                CandidateSpec(
                    "xgb_d5_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 5, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d3_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a20_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.20,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k3_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_summary_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_summary_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_pathway_summary_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_pathway_summary_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_pathway_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_pathway_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n520_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 520},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n560_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 560},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_local_risk_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_local_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_risk_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_risk_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_local_risk_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_local_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr006_n220_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.06, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_atom_profile_no_transfer_atom_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_atom_local_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a15_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a30_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a20_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a20_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a10_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.10,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a05_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.05,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a05_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a10_k1_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_k1_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a10_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.10,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a10_k7_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.10,
                    local_topk=7,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_k7_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid65_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.85,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a15_k3_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k3_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a15_k2_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=2,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k2_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a15_k6_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=6,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k6_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a12_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.12,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a12_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a18_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.18,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a18_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d3_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 3, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d3_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d5_lr005_n260_domain_rdkit_local_profile_q25_resid75_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 5, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d5_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n300_domain_rdkit_local_profile_q25_resid75_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 300},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n300_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_atom_local_profile_q25_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_atom_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr005_n260_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n300_domain_rdkit_atom_local_profile_q25_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 300},
                    feature_group="domain_rdkit_atom_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr005_n300_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_atom_local_profile_q50_resid75_atom_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_atom_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr005_n220_domain_rdkit_atom_profile_no_transfer_atom_a25_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr003_n450_domain_rdkit_local_profile_q25_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.03, "n_estimators": 450},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr003_n450_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr003_n450_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.03, "n_estimators": 450},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr003_n450_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr003_n450_domain_rdkit_local_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.03, "n_estimators": 450},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr003_n450_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid75_ew75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.75,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid75_ew65_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.65,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n420_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 420},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n420_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_tt_local_profile_clean_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_tt_local_profile_clean_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_tt_local_profile_clean_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_tt_local_profile_clean_q25_resid100_ew100_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_tt_local_profile_clean_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_tt_local_profile_clean_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a10_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.10,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a10_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desck2_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=2,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_nodesc_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_nodesc_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d5_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 5, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d5_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d3_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d3_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_t48rdk_a10_k3_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    time_local_alpha=0.10,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=2.0,
                    time_local_fingerprint="rdk",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_t48rdk_a35_k3_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    time_local_alpha=0.35,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=2.0,
                    time_local_fingerprint="rdk",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a20_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.20,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a20_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k3_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k3_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_summary_mean_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_summary_transfer",
                    residual_baseline_column="profile_transfer_summary_mean",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_profile_summary_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_pathway_summary_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_pathway_summary_transfer",
                    residual_baseline_column="pathway_transfer_summary_mean",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_pathway_summary_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_pathway_summary_direct_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_pathway_summary_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_pathway_summary_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_pathway_direct_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_pathway_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_pathway_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_local_risk_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_atom_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_local_risk_profile_riskq25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    residual_baseline_column="risk_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_atom_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_local_risk_profile_localq50_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_atom_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_atom_local_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_local_profile_q50_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_atom_local_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_profile_transfer",
                    residual_baseline_column="profile_transfer_nci_y_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_atom_profile_q50_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_atom_profile_transfer",
                    residual_baseline_column="profile_transfer_nci_y_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_risk_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_risk_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_local_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_local_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_profile_local_risk_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_profile_local_risk_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_risk_profile_riskq25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_risk_profile_transfer",
                    residual_baseline_column="risk_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n480_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid80_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.80,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid70_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.70,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew80_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.80,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q25_resid75_ew90_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.90,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_domain_rdkit_local_profile_q35_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q35",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n520_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 520},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n520_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n560_domain_rdkit_local_profile_q25_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 560},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n560_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n420_domain_rdkit_local_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 420},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n420_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n420_domain_rdkit_local_profile_q50_resid75_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 420},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n420_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q50_resid75_ew75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.75,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid60_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.60,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid65_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid60_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.60,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid65_ew85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid70_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.70,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q75_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q75",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_local_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n360_domain_rdkit_atom_local_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q50_resid65_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q25_resid85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.85,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_profile_q50_resid85_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.85,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_profile_mean_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_profile_transfer",
                    residual_baseline_column="profile_transfer_nci_y_mean",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_profile_transfer",
                    residual_baseline_column="profile_transfer_nci_y_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n360_domain_rdkit_atom_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_risk_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_risk_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n360_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_risk_profile_riskq50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_risk_profile_transfer",
                    residual_baseline_column="risk_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n360_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_local_risk_profile_riskq25_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_local_risk_profile_transfer",
                    residual_baseline_column="risk_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n360_domain_rdkit_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_local_risk_profile_localq50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n360_domain_rdkit_atom_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n360_domain_rdkit_atom_local_risk_profile_riskq50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 360},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    residual_baseline_column="risk_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "xgb_d4_lr004_n360_domain_rdkit_atom_local_risk_profile_no_transfer_atom_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "xgb_d4_lr006_n220_domain_rdkit_local_profile_q25_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.06, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr006_n220_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr006_n220_domain_rdkit_local_profile_q50_resid75_atom_a15_k4_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.06, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr006_n220_domain_rdkit_profile_no_transfer_atom_a15_k4_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid50_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.50,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n260_domain_rdkit_local_profile_q25_resid100_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 260},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n260_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q50_resid75_atom_a10_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.10,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a10_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q50_resid75_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n300_domain_rdkit_local_profile_q50_resid75_atom_a15_k5_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 300},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=5,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n300_domain_rdkit_profile_no_transfer_atom_a15_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a15_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid75_atom_a30_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a30_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q50_resid75_atom_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q50",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid100_atom_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n220_domain_rdkit_local_profile_q25_resid50_atom_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 220},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.50,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n220_domain_rdkit_profile_no_transfer_atom_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n300_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 300},
                    feature_group="domain_rdkit_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_n300_domain_rdkit_local_profile_resid35_local_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "n_estimators": 300},
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_n300_domain_rdkit_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_no_transfer_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_transfer_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_maccs_atom_time_no_transfer_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_no_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_maccs_atom_time_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_pathway_no_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_pathway_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_pathway_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_pathway_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_maccs_atom_time_pathway_no_transfer_local_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_maccs_atom_time_pathway_no_transfer_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="maccs_atom_time_pathway_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_maccs_atom_time_pathway_summary_resid75_ew85_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="maccs_atom_time_pathway_transfer",
                    residual_baseline_column="pathway_transfer_summary_mean",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_maccs_atom_time_pathway_no_transfer_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_maccs_atom_time_profile_no_transfer_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="maccs_atom_time_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr004_n480_maccs_atom_time_profile_q25_resid75_ew85_a25_desc48_domain_a100",
                    "xgboost",
                    {**params, "learning_rate": 0.04, "n_estimators": 480},
                    feature_group="maccs_atom_time_profile_transfer",
                    residual_baseline_column="profile_transfer_nci_y_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr004_n480_maccs_atom_time_profile_no_transfer_a25_desc48_domain_a100",
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_profile_no_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_profile_no_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "xgb_d4_lr005_maccs_atom_time_profile_transfer_local_a25_desc48_domain_a100",
                    "xgboost",
                    params,
                    feature_group="maccs_atom_time_profile_transfer",
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="xgb_d4_lr005_maccs_atom_time_profile_no_transfer_local_a25_desc48_domain_a100",
                ),
            ]
        )
        for depth in (4, 5):
            params = dict(xgb_base)
            params.update(
                {
                    "n_estimators": 160,
                    "max_depth": depth,
                    "learning_rate": 0.05,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 3.0,
                    "min_child_weight": 2.0,
                }
            )
            candidates.append(
                CandidateSpec(
                    f"xgb_d{depth}_lr005_desc48_domain_a100",
                    "xgboost",
                    params,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                )
            )
    if mode != "smoke" or (max_candidates is None or max_candidates > len(candidates)):
        for depth in [2, 3, 4]:
            params = dict(cat_base)
            params.update(
                {
                    "iterations": 60 if mode == "smoke" else 180,
                    "depth": depth,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 8.0,
                }
            )
            candidates.append(CandidateSpec(f"cat_d{depth}_lr005", "catboost", params))
            if mode != "smoke" and depth in {3, 4}:
                candidates.append(
                    CandidateSpec(
                        f"cat_d{depth}_lr005_local_a40_k3",
                        "catboost",
                        params,
                        local_alpha=0.40,
                        local_topk=3,
                        local_power=2.0,
                    )
                )
                for alpha in (0.20, 0.40):
                    candidates.append(
                        CandidateSpec(
                            f"cat_d{depth}_lr005_time_a{int(alpha * 100):02d}_k3",
                            "catboost",
                            params,
                            local_alpha=alpha,
                            local_topk=3,
                            local_power=2.0,
                            local_time_scale=24.0,
                        )
                    )
        if mode != "smoke":
            params = dict(cat_base)
            params.update(
                {
                    "iterations": 220,
                    "depth": 5,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 8.0,
                }
            )
            for alpha in (0.60, 0.90, 1.00):
                candidates.append(
                    CandidateSpec(
                        f"cat_d5_it220_all_time_a25_desc48_domain_a{int(alpha * 100):02d}",
                        "catboost",
                        params,
                        local_alpha=0.25,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=alpha,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                    )
                )
            for alpha in (0.70, 0.80, 0.90, 1.00):
                for local_alpha in (0.20, 0.40):
                    candidates.append(
                        CandidateSpec(
                            f"cat_d4_lr005_time_a{int(local_alpha * 100):02d}_desc48_domain_a{int(alpha * 100):03d}",
                            "catboost",
                            {
                                **cat_base,
                                "iterations": 180,
                                "depth": 4,
                                "learning_rate": 0.05,
                                "l2_leaf_reg": 8.0,
                            },
                            local_alpha=local_alpha,
                            local_topk=3,
                            local_power=2.0,
                            local_time_scale=24.0,
                            descriptor_local_alpha=alpha,
                            descriptor_local_topk=1,
                            descriptor_local_power=2.0,
                            descriptor_local_feature_group="domain_only",
                            descriptor_local_scaler="robust",
                            descriptor_local_time_h=48,
                            descriptor_local_min_delta=0.0,
                        )
                    )
                candidates.append(
                    CandidateSpec(
                        f"cat_d4_lr005_local_a40_k3_desc48_domain_a{int(alpha * 100):03d}",
                        "catboost",
                        {
                            **cat_base,
                            "iterations": 180,
                            "depth": 4,
                            "learning_rate": 0.05,
                            "l2_leaf_reg": 8.0,
                        },
                        local_alpha=0.40,
                        local_topk=3,
                        local_power=2.0,
                        descriptor_local_alpha=alpha,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                    )
                )
            cat_reduced_params = {
                **cat_base,
                "iterations": 300,
                "depth": 5,
                "learning_rate": 0.04,
                "l2_leaf_reg": 10.0,
            }
            cat_strong_s2_params = {
                **cat_base,
                "iterations": 600,
                "depth": 6,
                "learning_rate": 0.03,
                "l2_leaf_reg": 6.0,
            }
            cat_atom_s2_params = {
                **cat_base,
                "iterations": 600,
                "depth": 5,
                "learning_rate": 0.035,
                "l2_leaf_reg": 8.0,
            }
            cat_atom_strong_model_params = {
                **cat_base,
                "iterations": 900,
                "depth": 6,
                "learning_rate": 0.04,
                "l2_leaf_reg": 3.0,
            }
            for tag, feature_group in (
                ("domain_rdkit", "domain_rdkit_transfer"),
                ("rdkit_desc", "rdkit_desc_time_transfer"),
                ("maccs_atom", "maccs_atom_time_transfer"),
            ):
                for alpha in (0.90, 1.00):
                    candidates.append(
                        CandidateSpec(
                            f"cat_d5_it300_{tag}_time_a25_desc48_domain_a{int(alpha * 100):03d}",
                            "catboost",
                            cat_reduced_params,
                            feature_group=feature_group,
                            local_alpha=0.25,
                            local_topk=3,
                            local_power=2.0,
                            local_time_scale=24.0,
                            descriptor_local_alpha=alpha,
                            descriptor_local_topk=1,
                            descriptor_local_power=2.0,
                            descriptor_local_feature_group="domain_only",
                            descriptor_local_scaler="robust",
                            descriptor_local_time_h=48,
                            descriptor_local_min_delta=0.0,
                        )
                    )
                    if alpha == 1.00 and feature_group in {"domain_rdkit_transfer", "rdkit_desc_time_transfer"}:
                        candidates.append(
                            CandidateSpec(
                                f"cat_d5_it300_{tag}_time_a25_desc48_domain_a100_huber_cal",
                                "catboost",
                                cat_reduced_params,
                                feature_group=feature_group,
                                local_alpha=0.25,
                                local_topk=3,
                                local_power=2.0,
                                local_time_scale=24.0,
                                descriptor_local_alpha=1.00,
                                descriptor_local_topk=1,
                                descriptor_local_power=2.0,
                                descriptor_local_feature_group="domain_only",
                                descriptor_local_scaler="robust",
                                descriptor_local_time_h=48,
                                descriptor_local_min_delta=0.0,
                                calibration_method="huber",
                                paired_counterpart_id=(
                                    "cat_d5_it300_domain_rdkit_no_transfer_desc48_domain_a100_huber_cal"
                                    if feature_group == "domain_rdkit_transfer"
                                    else None
                                ),
                            )
                        )
                        if feature_group == "domain_rdkit_transfer":
                            candidates.append(
                                CandidateSpec(
                                    f"cat_d5_it300_{tag}_time_a25_desc48_domain_a100_ridge_context_cal",
                                    "catboost",
                                    cat_reduced_params,
                                    feature_group=feature_group,
                                    local_alpha=0.25,
                                    local_topk=3,
                                    local_power=2.0,
                                    local_time_scale=24.0,
                                    descriptor_local_alpha=1.00,
                                    descriptor_local_topk=1,
                                    descriptor_local_power=2.0,
                                    descriptor_local_feature_group="domain_only",
                                    descriptor_local_scaler="robust",
                                    descriptor_local_time_h=48,
                                    descriptor_local_min_delta=0.0,
                                    calibration_method="ridge_context",
                                )
                            )
                            candidates.append(
                                CandidateSpec(
                                    "cat_d5_it300_domain_rdkit_profile_time_a25_desc48_domain_a100_huber_cal",
                                    "catboost",
                                    cat_reduced_params,
                                    feature_group="domain_rdkit_profile_transfer",
                                    local_alpha=0.25,
                                    local_topk=3,
                                    local_power=2.0,
                                    local_time_scale=24.0,
                                    descriptor_local_alpha=1.00,
                                    descriptor_local_topk=1,
                                    descriptor_local_power=2.0,
                                    descriptor_local_feature_group="domain_only",
                                    descriptor_local_scaler="robust",
                                    descriptor_local_time_h=48,
                                    descriptor_local_min_delta=0.0,
                                    calibration_method="huber",
                                    paired_counterpart_id=(
                                        "cat_d5_it300_domain_rdkit_profile_no_transfer_desc48_domain_a100_huber_cal"
                                    ),
                                )
                            )
                if feature_group == "maccs_atom_time_transfer" and alpha == 1.00:
                    candidates.append(
                        CandidateSpec(
                            "cat_d5_it300_maccs_atom_time_a25_desc48_domain_a100_huber_cal",
                            "catboost",
                            cat_reduced_params,
                            feature_group=feature_group,
                            local_alpha=0.25,
                            local_topk=3,
                            local_power=2.0,
                            local_time_scale=24.0,
                            descriptor_local_alpha=1.00,
                            descriptor_local_topk=1,
                            descriptor_local_power=2.0,
                            descriptor_local_feature_group="domain_only",
                            descriptor_local_scaler="robust",
                            descriptor_local_time_h=48,
                            descriptor_local_min_delta=0.0,
                            calibration_method="huber",
                            paired_counterpart_id="cat_d5_it300_maccs_atom_time_no_transfer_desc48_domain_a100_huber_cal",
                        )
                    )
                    candidates.append(
                        CandidateSpec(
                            "cat_d5_it300_maccs_atom_time_pathway_a25_desc48_domain_a100_huber_cal",
                            "catboost",
                            cat_reduced_params,
                            feature_group="maccs_atom_time_pathway_transfer",
                            local_alpha=0.25,
                            local_topk=3,
                            local_power=2.0,
                            local_time_scale=24.0,
                            descriptor_local_alpha=1.00,
                            descriptor_local_topk=1,
                            descriptor_local_power=2.0,
                            descriptor_local_feature_group="domain_only",
                            descriptor_local_scaler="robust",
                            descriptor_local_time_h=48,
                            descriptor_local_min_delta=0.0,
                            calibration_method="huber",
                            paired_counterpart_id=(
                                "cat_d5_it300_maccs_atom_time_pathway_no_transfer_desc48_domain_a100_huber_cal"
                            ),
                        )
                    )
                    candidates.append(
                        CandidateSpec(
                            "cat_d5_it300_maccs_atom_time_profile_a25_desc48_domain_a100_huber_cal",
                            "catboost",
                            cat_reduced_params,
                            feature_group="maccs_atom_time_profile_transfer",
                            local_alpha=0.25,
                            local_topk=3,
                            local_power=2.0,
                            local_time_scale=24.0,
                            descriptor_local_alpha=1.00,
                            descriptor_local_topk=1,
                            descriptor_local_power=2.0,
                            descriptor_local_feature_group="domain_only",
                            descriptor_local_scaler="robust",
                            descriptor_local_time_h=48,
                            descriptor_local_min_delta=0.0,
                            calibration_method="huber",
                            paired_counterpart_id=(
                                "cat_d5_it300_maccs_atom_time_profile_no_transfer_desc48_domain_a100_huber_cal"
                            ),
                        )
                    )
            candidates.extend(
                [
                    CandidateSpec(
                        "cat_d5_it300_domain_rdkit_no_transfer_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_reduced_params,
                        feature_group="domain_rdkit_no_transfer",
                        local_alpha=0.25,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d5_it300_domain_rdkit_profile_no_transfer_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_reduced_params,
                        feature_group="domain_rdkit_profile_no_transfer",
                        local_alpha=0.25,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d6_it600_domain_rdkit_local_profile_no_transfer_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_strong_s2_params,
                        feature_group="domain_rdkit_local_profile_no_transfer",
                        local_alpha=0.20,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d6_it600_domain_rdkit_local_profile_time_a20_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_strong_s2_params,
                        feature_group="domain_rdkit_local_profile_transfer",
                        local_alpha=0.20,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d6_it600_domain_rdkit_local_profile_no_transfer_desc48_domain_a100_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a10_k4_desc48_domain_a025_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_no_transfer",
                        local_alpha=0.10,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.25,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_q50_resid25_local_a10_k4_desc48_domain_a025_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        residual_baseline_column="local_profile_transfer_q50",
                        residual_shrinkage=0.25,
                        residual_ensemble_weight=1.00,
                        local_alpha=0.10,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.25,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a10_k4_desc48_domain_a025_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_q50_resid50_local_a10_k4_desc48_domain_a025_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        residual_baseline_column="local_profile_transfer_q50",
                        residual_shrinkage=0.50,
                        residual_ensemble_weight=1.00,
                        local_alpha=0.10,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.25,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a10_k4_desc48_domain_a025_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_q50_resid75_local_a10_k4_desc48_domain_a025_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        residual_baseline_column="local_profile_transfer_q50",
                        residual_shrinkage=0.75,
                        residual_ensemble_weight=1.00,
                        local_alpha=0.10,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.25,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a10_k4_desc48_domain_a025_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_q50_resid100_local_a10_k4_desc48_domain_a025_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        residual_baseline_column="local_profile_transfer_q50",
                        residual_shrinkage=1.00,
                        residual_ensemble_weight=1.00,
                        local_alpha=0.10,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.25,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a10_k4_desc48_domain_a025_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_model_desc48_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_no_transfer",
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_transfer_model_desc48_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_model_desc48_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d6_it900_domain_rdkit_atom_local_profile_no_transfer_model_desc48",
                        "catboost",
                        cat_atom_strong_model_params,
                        feature_group="domain_rdkit_atom_local_profile_no_transfer",
                    ),
                    CandidateSpec(
                        "cat_d6_it900_domain_rdkit_atom_local_profile_transfer_model_desc48",
                        "catboost",
                        cat_atom_strong_model_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        paired_counterpart_id=(
                            "cat_d6_it900_domain_rdkit_atom_local_profile_no_transfer_model_desc48"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a05_k4_desc48_domain_a010_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_no_transfer",
                        local_alpha=0.05,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.10,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_transfer_local_a05_k4_desc48_domain_a010_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        local_alpha=0.05,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.10,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a05_k4_desc48_domain_a010_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_q50_resid50_local_a05_k4_desc48_domain_a010_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        residual_baseline_column="local_profile_transfer_q50",
                        residual_shrinkage=0.50,
                        residual_ensemble_weight=1.00,
                        local_alpha=0.05,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=0.10,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a05_k4_desc48_domain_a010_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a20_k4_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_no_transfer",
                        local_alpha=0.20,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d5_it600_domain_rdkit_atom_local_profile_transfer_local_a20_k4_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_atom_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        local_alpha=0.20,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id=(
                            "cat_d5_it600_domain_rdkit_atom_local_profile_no_transfer_local_a20_k4_desc48_domain_a100_huber_cal"
                        ),
                    ),
                    CandidateSpec(
                        "cat_d6_it600_raw_no_transfer_huber_cal",
                        "catboost",
                        cat_strong_s2_params,
                        local_alpha=0.20,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d6_it600_raw_atom_local_profile_transfer_local_a20_k4_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_strong_s2_params,
                        feature_group="domain_rdkit_atom_local_profile_transfer",
                        local_alpha=0.20,
                        local_topk=4,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                        paired_counterpart_id="cat_d6_it600_raw_no_transfer_huber_cal",
                    ),
                    CandidateSpec(
                        "cat_d5_it300_maccs_atom_time_no_transfer_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_reduced_params,
                        feature_group="maccs_atom_time_no_transfer",
                        local_alpha=0.25,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d5_it300_maccs_atom_time_pathway_no_transfer_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_reduced_params,
                        feature_group="maccs_atom_time_pathway_no_transfer",
                        local_alpha=0.25,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                    CandidateSpec(
                        "cat_d5_it300_maccs_atom_time_profile_no_transfer_desc48_domain_a100_huber_cal",
                        "catboost",
                        cat_reduced_params,
                        feature_group="maccs_atom_time_profile_no_transfer",
                        local_alpha=0.25,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        calibration_method="huber",
                    ),
                ]
            )
            candidates.append(
                CandidateSpec(
                    "cat_d5_it300_domain_rdkit_tt_resid65_teacher_topk_desc48_domain_a100_huber_cal",
                    "catboost",
                    cat_reduced_params,
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.25,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    calibration_method="huber",
                )
            )
    if mode != "smoke":
        adaptive_f70_teacher_topk_specs = []
        for n_estimators in (200, 400, 700):
            adaptive_f70_teacher_topk_specs.extend(
                CandidateSpec(
                    f"extratrees_{n_estimators}_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a{int(alpha * 100):02d}_k1_delta{int(min_delta * 100):02d}",
                    "extratrees",
                    {
                        "n_estimators": n_estimators,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=alpha,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=min_delta,
                    paired_counterpart_id=(
                        "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_no_transfer"
                        if n_estimators == 400 and alpha == 0.90 and min_delta == 0.0
                        else "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a50_k1_delta00_no_transfer"
                        if n_estimators == 400 and alpha == 0.50 and min_delta == 0.0
                        else None
                    ),
                )
                for alpha, min_delta in (
                    (1.00, 0.00),
                    (0.90, 0.00),
                    (0.80, 0.00),
                    (0.80, 0.10),
                    (0.70, 0.00),
                    (0.70, 0.10),
                    (0.60, 0.00),
                    (0.60, 0.10),
                    (0.50, 0.00),
                    (0.50, 0.10),
                    (0.40, 0.00),
                    (0.40, 0.10),
                    (0.30, 0.00),
                    (0.30, 0.10),
                    (0.20, 0.00),
                    (0.20, 0.10),
                    (0.15, 0.00),
                    (0.10, 0.00),
                )
            )
        candidates.extend(
            [
                *adaptive_f70_teacher_topk_specs,
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_no_transfer",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_no_transfer",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_pair",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_pure_no_counterpart_pair",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_120_domain_rdkit_tt_f35_l6_no_transfer_plain",
                ),
                CandidateSpec(
                    "extratrees_120_domain_rdkit_tt_f35_l6_no_transfer_desc_time48_domain_only_a90_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 120,
                        "min_samples_leaf": 6,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_120_domain_rdkit_tt_f35_l6_no_transfer_plain",
                    "extratrees",
                    {
                        "n_estimators": 80,
                        "min_samples_leaf": 8,
                        "max_features": 0.25,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_risk_suppressed_no_transfer",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_risk_suppressed",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "suppress_transfer_72h_risk": True,
                        "transfer_risk_suppression": "72h_nitro_trihalo_high_halogen",
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a90_k1_delta00_risk_suppressed_no_transfer",
                ),
                CandidateSpec(
                    "a90_teacher_topk_risk_routed_no_transfer",
                    "risk_routed_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "full_feature_group": "domain_rdkit_tt_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_no_transfer",
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "transfer_enabled": False,
                    },
                    feature_group="risk_routed_no_transfer",
                ),
                CandidateSpec(
                    "a90_teacher_topk_risk_routed_transfer",
                    "risk_routed_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "full_feature_group": "domain_rdkit_tt_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_no_transfer",
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "transfer_enabled": True,
                    },
                    feature_group="risk_routed_transfer",
                    paired_counterpart_id="a90_teacher_topk_risk_routed_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_local_profile_no_transfer_f85_l2_desc_time48_domain_a90_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_local_profile_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_local_profile_f85_l2_desc_time48_domain_a90_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=0.90,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_local_profile_no_transfer_f85_l2_desc_time48_domain_a90_k1_delta00",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_risk_adjusted_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_risk_adjusted_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_risk_adjusted_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_risk_adjusted_transfer",
                    residual_baseline_column="risk_adjusted_transfer_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_risk_adjusted_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_calibrated_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_calibrated_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_calibrated_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_calibrated_transfer",
                    residual_baseline_column="calibrated_transfer_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_calibrated_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_calibrated_direct_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_calibrated_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_calibrated_direct_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_calibrated_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_calibrated_direct_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_calibrated_context_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_calibrated_no_transfer",
                    calibration_method="ridge_context",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_calibrated_context_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_calibrated_transfer",
                    calibration_method="ridge_context",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_calibrated_context_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                ),
                CandidateSpec(
                    "a90_calibrated_gated_no_transfer",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_calibrated_transfer",
                        "full_residual_baseline_column": "calibrated_transfer_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_calibrated_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "calibrated_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_calibrated_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_calibrated_gated_no_transfer_s150",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_calibrated_transfer",
                        "full_residual_baseline_column": "calibrated_transfer_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_calibrated_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "calibrated_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.50,
                        "gate_min_positive_benefit": 0.0,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_calibrated_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_calibrated_gated_transfer_etr_s100",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_calibrated_transfer",
                        "full_residual_baseline_column": "calibrated_transfer_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_calibrated_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "calibrated_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_calibrated_gated_transfer",
                    paired_counterpart_id="a90_calibrated_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_calibrated_gated_transfer_etr_s150",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_calibrated_transfer",
                        "full_residual_baseline_column": "calibrated_transfer_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_calibrated_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "calibrated_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.50,
                        "gate_min_positive_benefit": 0.0,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_calibrated_gated_transfer",
                    paired_counterpart_id="a90_calibrated_gated_no_transfer_s150",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_conditioned_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_conditioned_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_conditioned_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_conditioned_transfer",
                    residual_baseline_column="conditioned_transfer_wmean",
                    residual_shrinkage=0.55,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_conditioned_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_conditioned_clean_mid_s25_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_conditioned_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_conditioned_clean_mid_s25_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_conditioned_transfer",
                    residual_baseline_column="conditioned_transfer_clean_mid",
                    residual_shrinkage=0.25,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_conditioned_clean_mid_s25_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_disagreement_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_disagreement_no_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_disagreement_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_disagreement_transfer",
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_disagreement_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                ),
                CandidateSpec(
                    "disagreement_gated_no_transfer",
                    "disagreement_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "disagreement_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": False,
                    },
                    feature_group="disagreement_gated_no_transfer",
                ),
                CandidateSpec(
                    "disagreement_gated_transfer_etr_s100",
                    "disagreement_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "disagreement_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": True,
                    },
                    feature_group="disagreement_gated_transfer",
                    paired_counterpart_id="disagreement_gated_no_transfer",
                ),
                CandidateSpec(
                    "risk_adjusted_gated_no_transfer",
                    "risk_adjusted_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_adjusted_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="risk_adjusted_gated_no_transfer",
                ),
                CandidateSpec(
                    "risk_adjusted_gated_transfer_etr_s100",
                    "risk_adjusted_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_adjusted_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="risk_adjusted_gated_transfer",
                    paired_counterpart_id="risk_adjusted_gated_no_transfer",
                ),
                CandidateSpec(
                    "risk_adjusted_gated_no_transfer_logit_c2_s125",
                    "risk_adjusted_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_adjusted_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="risk_adjusted_gated_no_transfer",
                ),
                CandidateSpec(
                    "risk_adjusted_gated_transfer_logit_c2_s125",
                    "risk_adjusted_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_adjusted_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="risk_adjusted_gated_transfer",
                    paired_counterpart_id="risk_adjusted_gated_no_transfer_logit_c2_s125",
                ),
                CandidateSpec(
                    "a90_risk_gated_no_transfer",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_risk_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_risk_gated_transfer_etr_s100",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_risk_gated_transfer",
                    paired_counterpart_id="a90_risk_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_risk_gated_no_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_risk_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_risk_gated_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_risk_gated_transfer",
                    paired_counterpart_id="a90_risk_gated_no_transfer_logit_c2_s125",
                ),
                CandidateSpec(
                    "a90_local_profile_gated_no_transfer",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "local_profile_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_local_profile_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_risk_profile_gated_no_transfer",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_profile_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_risk_profile_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_risk_profile_gated_transfer_etr_s100",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_profile_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_risk_profile_gated_transfer",
                    paired_counterpart_id="a90_risk_profile_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_risk_profile_gated_no_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_profile_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_risk_profile_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_risk_profile_gated_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "risk_profile_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_risk_profile_gated_transfer",
                    paired_counterpart_id="a90_risk_profile_gated_no_transfer_logit_c2_s125",
                ),
                CandidateSpec(
                    "a90_local_profile_gated_transfer_etr_s100",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "local_profile_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_local_profile_gated_transfer",
                    paired_counterpart_id="a90_local_profile_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_local_profile_gated_no_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "local_profile_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_local_profile_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_local_profile_gated_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "gate_feature_group": "local_profile_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "no_transfer_uses_residual_baseline": False,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_local_profile_gated_transfer",
                    paired_counterpart_id="a90_local_profile_gated_no_transfer_logit_c2_s125",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_domain_only_a50_k1_delta00_no_transfer",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=0.50,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_600_leaf1",
                    "extratrees",
                    {
                        "n_estimators": 600,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                ),
                CandidateSpec(
                    "extratrees_700_sqrt_rdkit_transfer",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="rdkit_desc_time_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_sqrt_no_source_teacher",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="no_source_teacher",
                ),
                CandidateSpec(
                    "extratrees_700_sqrt_no_transfer",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="no_transfer",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_no_transfer_w395_b000",
                    "transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.395,
                        "blend_residual_weight": 0.0,
                        "transfer_blend_use_transfer": False,
                    },
                    feature_group="transfer_blend_no_transfer",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_w395_b290",
                    "transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.395,
                        "blend_residual_weight": 0.290,
                        "transfer_blend_use_transfer": True,
                    },
                    feature_group="transfer_blend_transfer",
                    paired_counterpart_id="transfer_blend_et65_l1_svr16_eps005_no_transfer_w395_b000",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_no_transfer_w370_b000",
                    "transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.370,
                        "blend_residual_weight": 0.0,
                        "transfer_blend_use_transfer": False,
                    },
                    feature_group="transfer_blend_no_transfer",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_w370_b330",
                    "transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.370,
                        "blend_residual_weight": 0.330,
                        "transfer_blend_use_transfer": True,
                    },
                    feature_group="transfer_blend_transfer",
                    paired_counterpart_id="transfer_blend_et65_l1_svr16_eps005_no_transfer_w370_b000",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_calibrated_no_transfer_w370_b000",
                    "calibrated_transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.370,
                        "blend_residual_weight": 0.0,
                        "calibration_method": "huber",
                        "transfer_blend_use_transfer": False,
                    },
                    feature_group="transfer_blend_no_transfer",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_calibrated_w370_b330",
                    "calibrated_transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.370,
                        "blend_residual_weight": 0.330,
                        "calibration_method": "huber",
                        "transfer_blend_use_transfer": True,
                    },
                    feature_group="transfer_blend_transfer",
                    paired_counterpart_id="transfer_blend_et65_l1_svr16_eps005_calibrated_no_transfer_w370_b000",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_calibrated_no_transfer_w395_b000",
                    "calibrated_transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.395,
                        "blend_residual_weight": 0.0,
                        "calibration_method": "huber",
                        "transfer_blend_use_transfer": False,
                    },
                    feature_group="transfer_blend_no_transfer",
                ),
                CandidateSpec(
                    "transfer_blend_et65_l1_svr16_eps005_calibrated_w395_b290",
                    "calibrated_transfer_blend",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.395,
                        "blend_residual_weight": 0.290,
                        "calibration_method": "huber",
                        "transfer_blend_use_transfer": True,
                    },
                    feature_group="transfer_blend_transfer",
                    paired_counterpart_id="transfer_blend_et65_l1_svr16_eps005_calibrated_no_transfer_w395_b000",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_residualized_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_residualized_no_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_residualized_f85_l2_desc_time48_domain_a100_k1_delta00",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_residualized_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "extratrees_400_domain_rdkit_tt_residualized_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta00"
                    ),
                ),
                CandidateSpec(
                    "residualized_gated_no_transfer",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": False,
                    },
                    feature_group="residualized_gated_no_transfer",
                ),
                CandidateSpec(
                    "residualized_gated_no_transfer_a90",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.90,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": False,
                    },
                    feature_group="residualized_gated_no_transfer",
                ),
                CandidateSpec(
                    "residualized_gated_transfer_a90_etr_s100",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.90,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": True,
                    },
                    feature_group="residualized_gated_transfer",
                    paired_counterpart_id="residualized_gated_no_transfer_a90",
                ),
                CandidateSpec(
                    "residualized_gated_transfer_etr_s100",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "gate_min_positive_benefit": 0.001,
                        "transfer_enabled": True,
                    },
                    feature_group="residualized_gated_transfer",
                    paired_counterpart_id="residualized_gated_no_transfer",
                ),
                CandidateSpec(
                    "residualized_gated_no_transfer_a90_s35_benefit010",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "gate_min_positive_benefit": 0.010,
                        "transfer_enabled": False,
                    },
                    feature_group="residualized_gated_no_transfer",
                ),
                CandidateSpec(
                    "residualized_gated_transfer_a90_s35_benefit010",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "gate_min_positive_benefit": 0.010,
                        "transfer_enabled": True,
                    },
                    feature_group="residualized_gated_transfer",
                    paired_counterpart_id="residualized_gated_no_transfer_a90_s35_benefit010",
                ),
                CandidateSpec(
                    "residualized_gated_no_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "transfer_enabled": False,
                    },
                    feature_group="residualized_gated_no_transfer",
                ),
                CandidateSpec(
                    "residualized_gated_transfer_logit_c2_s125",
                    "a90_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 1.00,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "full_feature_group": "domain_rdkit_tt_residualized_transfer",
                        "full_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_feature_group": "domain_rdkit_tt_residualized_no_transfer",
                        "no_transfer_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "gate_feature_group": "residualized_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "transfer_enabled": True,
                    },
                    feature_group="residualized_gated_transfer",
                    paired_counterpart_id="residualized_gated_no_transfer_logit_c2_s125",
                ),
                CandidateSpec(
                    "a90_transfer_blend_correction_no_transfer",
                    "a90_delta_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "delta_blend_weight": 0.245,
                        "delta_blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "anchor_delta_scale": 0.35,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_transfer_blend_correction_no_transfer",
                ),
                CandidateSpec(
                    "a90_transfer_blend_correction_transfer_s35",
                    "a90_delta_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_transfer",
                        "anchor_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "delta_blend_weight": 0.245,
                        "delta_blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "anchor_delta_scale": 0.35,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_transfer_blend_correction_transfer",
                    paired_counterpart_id="a90_transfer_blend_correction_no_transfer",
                ),
                CandidateSpec(
                    "gated_transfer_all_context_no_transfer_etr_s100",
                    "gated_transfer",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.245,
                        "blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "transfer_enabled": False,
                    },
                    feature_group="gated_transfer_all_context_no_transfer",
                ),
                CandidateSpec(
                    "gated_transfer_all_context_etr_s100",
                    "gated_transfer",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.245,
                        "blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "transfer_enabled": True,
                    },
                    feature_group="gated_transfer_all_context",
                    paired_counterpart_id="gated_transfer_all_context_no_transfer_etr_s100",
                ),
                CandidateSpec(
                    "a90_delta_gated_no_transfer",
                    "a90_delta_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "delta_blend_weight": 0.245,
                        "delta_blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "anchor_delta_scale": 1.00,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_delta_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_delta_gated_transfer_etr_s100",
                    "a90_delta_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_transfer",
                        "anchor_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": True,
                        "delta_blend_weight": 0.245,
                        "delta_blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 1.00,
                        "anchor_delta_scale": 1.00,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_delta_gated_transfer",
                    paired_counterpart_id="a90_delta_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_delta_no_transfer",
                    "a90_delta_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.60,
                        "anchor_delta_scale": 0.60,
                        "delta_family": "conditioned_clean_mid",
                        "delta_residual_shrinkage": 0.25,
                        "transfer_enabled": False,
                    },
                    feature_group="a90_delta_gated_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_delta_transfer_etr_s60",
                    "a90_delta_gated_transfer",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                        "residual_shrinkage": 0.65,
                        "calibration_method": "huber",
                        "descriptor_local_alpha": 0.90,
                        "descriptor_local_topk": 1,
                        "descriptor_local_power": 2.0,
                        "descriptor_local_feature_group": "domain_only",
                        "descriptor_local_scaler": "robust",
                        "descriptor_local_time_h": 48,
                        "descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_transfer",
                        "anchor_residual_baseline_column": "teacher_topk_y_wmean",
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.60,
                        "anchor_delta_scale": 0.60,
                        "delta_family": "conditioned_clean_mid",
                        "delta_residual_shrinkage": 0.25,
                        "transfer_enabled": True,
                    },
                    feature_group="a90_delta_gated_transfer",
                    paired_counterpart_id="a90_conditioned_delta_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_correction_no_transfer",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.50,
                        "anchor_delta_scale": 0.50,
                        "gate_min_positive_benefit": 0.0,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": False,
                    },
                    feature_group="a90_conditioned_correction_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_correction_transfer_s50",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.50,
                        "anchor_delta_scale": 0.50,
                        "gate_min_positive_benefit": 0.0,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": True,
                    },
                    feature_group="a90_conditioned_ridge_correction_transfer",
                    paired_counterpart_id="a90_conditioned_ridge_correction_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_conservative_no_transfer",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.25,
                        "anchor_delta_scale": 0.25,
                        "gate_min_positive_benefit": 0.005,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": False,
                    },
                    feature_group="a90_conditioned_correction_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_conservative_transfer_s25_b005",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.25,
                        "anchor_delta_scale": 0.25,
                        "gate_min_positive_benefit": 0.005,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": True,
                    },
                    feature_group="a90_conditioned_ridge_conservative_transfer",
                    paired_counterpart_id="a90_conditioned_ridge_conservative_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_confident_no_transfer",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "anchor_delta_scale": 0.35,
                        "gate_min_positive_benefit": 0.0,
                        "delta_max_uncertainty": 0.60,
                        "delta_min_teacher_similarity": 0.25,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": False,
                    },
                    feature_group="a90_conditioned_confident_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_confident_transfer_s35_u060_sim025",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "anchor_delta_scale": 0.35,
                        "gate_min_positive_benefit": 0.0,
                        "delta_max_uncertainty": 0.60,
                        "delta_min_teacher_similarity": 0.25,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": True,
                    },
                    feature_group="a90_conditioned_ridge_confident_transfer",
                    paired_counterpart_id="a90_conditioned_ridge_confident_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_risk_suppressed_no_transfer",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "anchor_delta_scale": 0.35,
                        "gate_min_positive_benefit": 0.0,
                        "delta_max_uncertainty": 0.60,
                        "delta_min_teacher_similarity": 0.25,
                        "suppress_72h_risk": False,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": False,
                    },
                    feature_group="a90_conditioned_risk_suppressed_no_transfer",
                ),
                CandidateSpec(
                    "a90_conditioned_ridge_risk_suppressed_transfer_s35_u060_sim025",
                    "a90_delta_gated_transfer",
                    {
                        "delta_ridge_alphas": (0.1, 1.0, 10.0, 100.0),
                        "delta_calibration_method": "none",
                        "delta_descriptor_local_alpha": 0.90,
                        "delta_descriptor_local_topk": 1,
                        "delta_descriptor_local_power": 2.0,
                        "delta_descriptor_local_feature_group": "domain_only",
                        "delta_descriptor_local_scaler": "robust",
                        "delta_descriptor_local_time_h": 48,
                        "delta_descriptor_local_min_delta": 0.0,
                        "anchor_feature_group": "domain_rdkit_tt_no_transfer",
                        "anchor_residual_baseline_column": None,
                        "no_transfer_uses_residual_baseline": False,
                        "gate_feature_group": "conditioned_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.35,
                        "anchor_delta_scale": 0.35,
                        "gate_min_positive_benefit": 0.0,
                        "delta_max_uncertainty": 0.60,
                        "delta_min_teacher_similarity": 0.25,
                        "suppress_72h_risk": True,
                        "delta_family": "conditioned_ridge",
                        "transfer_enabled": True,
                    },
                    feature_group="a90_conditioned_ridge_risk_suppressed_transfer",
                    paired_counterpart_id="a90_conditioned_ridge_risk_suppressed_no_transfer",
                ),
                CandidateSpec(
                    "gated_transfer_all_context_no_transfer_logit_c2_s125",
                    "gated_transfer",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.245,
                        "blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "transfer_enabled": False,
                    },
                    feature_group="gated_transfer_all_context_no_transfer",
                ),
                CandidateSpec(
                    "gated_transfer_all_context_logit_c2_s125",
                    "gated_transfer",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.245,
                        "blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "logit_c2",
                        "gate_scale": 1.25,
                        "transfer_enabled": True,
                    },
                    feature_group="gated_transfer_all_context",
                    paired_counterpart_id="gated_transfer_all_context_no_transfer_logit_c2_s125",
                ),
                CandidateSpec(
                    "gated_transfer_all_context_no_transfer_etr_s80_benefit050",
                    "gated_transfer",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.245,
                        "blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.80,
                        "gate_min_positive_benefit": 0.50,
                        "transfer_enabled": False,
                    },
                    feature_group="gated_transfer_all_context_no_transfer",
                ),
                CandidateSpec(
                    "gated_transfer_all_context_etr_s80_benefit050",
                    "gated_transfer",
                    {
                        "et_n_estimators": 700,
                        "et_min_samples_leaf": 1,
                        "et_max_features": 0.65,
                        "et_random_state": 20260605,
                        "svr_C": 16.0,
                        "svr_epsilon": 0.005,
                        "svr_gamma": "scale",
                        "blend_weight": 0.245,
                        "blend_residual_weight": 0.33,
                        "gate_feature_group": "all_context",
                        "gate_model": "etr_benefit",
                        "gate_scale": 0.80,
                        "gate_min_positive_benefit": 0.50,
                        "transfer_enabled": True,
                    },
                    feature_group="gated_transfer_all_context",
                    paired_counterpart_id="gated_transfer_all_context_no_transfer_etr_s80_benefit050",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_profile_plain_no_transfer_f70_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_profile_plain_f70_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_transfer",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_profile_plain_no_transfer_f70_l2",
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_profile_summary_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta10",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_summary_no_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.10,
                ),
                CandidateSpec(
                    "extratrees_400_domain_rdkit_tt_profile_summary_f85_l2_desc_time48_domain_a100_k1_delta10",
                    "extratrees",
                    {
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_summary_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.10,
                    paired_counterpart_id="extratrees_400_domain_rdkit_tt_profile_summary_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta10",
                ),
                CandidateSpec(
                    "extratrees_300_domain_rdkit_tt_profile_plain_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta10",
                    "extratrees",
                    {
                        "n_estimators": 300,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_no_transfer",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.10,
                ),
                CandidateSpec(
                    "extratrees_300_domain_rdkit_tt_profile_plain_f85_l2_desc_time48_domain_a100_k1_delta10",
                    "extratrees",
                    {
                        "n_estimators": 300,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_transfer",
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.10,
                    paired_counterpart_id="extratrees_300_domain_rdkit_tt_profile_plain_no_transfer_f85_l2_desc_time48_domain_a100_k1_delta10",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_profile_no_transfer_f70_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_profile_f70_l2_teacher_topk_resid65_ens100_huber",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_profile_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_profile_no_transfer_f70_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_pathway_time48_no_transfer_f70_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_time48_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_pathway_time48_f70_l2_teacher_topk_resid65_ens100_huber",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_time48_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_pathway_time48_no_transfer_f70_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_pathway_summary_no_transfer_f70_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_summary_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_pathway_summary_f70_l2_teacher_topk_resid65_ens100_ridge_context",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_summary_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="ridge_context",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_pathway_summary_no_transfer_f70_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_pathway_no_transfer_f70_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_pathway_f70_l2_teacher_topk_resid65_ens100_ridge_context",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="ridge_context",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_pathway_no_transfer_f70_l2",
                ),
                CandidateSpec(
                    "extratrees_700_maccs_atom_time_pathway_no_transfer_leaf2_f70",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_pathway_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_maccs_atom_time_pathway_transfer_leaf2_f70",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_pathway_transfer",
                    paired_counterpart_id="extratrees_700_maccs_atom_time_pathway_no_transfer_leaf2_f70",
                ),
                CandidateSpec(
                    "extratrees_700_sqrt_maccs_transfer",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f35_l2_d10",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "max_depth": 10,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f50_l2_d10",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.50,
                        "max_depth": 10,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2_atom_pair_a30_k3_p2_time48_rdk_a35_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.35,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                ),
                CandidateSpec(
                    "extratrees_500_domain_rdkit_f65_l2_atom_pair_a30_k3_p2_time48_rdk_a35_k3_p4_seed7",
                    "extratrees",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 7,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.35,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                ),
                CandidateSpec(
                    "extratrees_500_domain_rdkit_f65_l2_atom_pair_a30_k3_p2_time48_rdk_a35_k3_p4_seed7_desc_domain_teacher_a05_k3_p2",
                    "extratrees",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 7,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.35,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                    descriptor_local_alpha=0.05,
                    descriptor_local_topk=3,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_teacher",
                    descriptor_local_scaler="robust",
                ),
                CandidateSpec(
                    "extratrees_500_domain_rdkit_f65_l2_atom_pair_a30_k3_p2_time48_rdk_a35_k3_p4_seed7_desc_domain_only_a05_k1",
                    "extratrees",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 7,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.35,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                    descriptor_local_alpha=0.05,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                ),
                *[
                    CandidateSpec(
                        f"extratrees_500_domain_rdkit_f65_l2_atom_pair_a30_k3_p2_time48_rdk_a35_k3_p4_seed7_desc_time48_domain_only_a{int(alpha * 100):02d}_k1_delta00",
                        "extratrees",
                        {
                            "n_estimators": 500,
                            "min_samples_leaf": 2,
                            "max_features": 0.65,
                            "random_state": 7,
                            "n_jobs": -1,
                        },
                        feature_group="domain_rdkit_transfer",
                        local_alpha=0.30,
                        local_topk=3,
                        local_power=2.0,
                        local_fingerprint="atom_pair",
                        time_local_alpha=0.35,
                        time_local_h=48,
                        time_local_topk=3,
                        time_local_power=4.0,
                        time_local_fingerprint="rdk",
                        descriptor_local_alpha=alpha,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                    )
                    for alpha in (0.40, 0.50, 0.60, 0.70)
                ],
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f65_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_morgan_no_transfer_f65_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_morgan_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_morgan_f65_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_morgan_transfer",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_morgan_no_transfer_f65_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_morgan_no_transfer_f80_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_morgan_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_morgan_f80_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_morgan_transfer",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_morgan_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_morgan_f80_l2_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_morgan_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_morgan_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_maccs_no_transfer_f80_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_maccs_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_maccs_f80_l2_source_resid35_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_maccs_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_maccs_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_maccs_f80_l2_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_maccs_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_maccs_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f50_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.50,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f50_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.50,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f50_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f35_l2_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_local_a15_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_local_a20_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.20,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f35_l2_local_a20_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.20,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_local_a20_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_cell_no_transfer_f35_l2_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_cell_no_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_cell_f35_l2_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_cell_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_cell_no_transfer_f35_l2_local_a15_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_atom_pair_a30_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=4.0,
                    local_fingerprint="atom_pair",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f35_l2_atom_pair_a30_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=4.0,
                    local_fingerprint="atom_pair",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_atom_pair_a30_k3_p4",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_atom_pair_a30_k3_p4_time48_rdk_a50_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=4.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.50,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f35_l2_atom_pair_a30_k3_p4_time48_rdk_a50_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=4.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.50,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_atom_pair_a30_k3_p4_time48_rdk_a50_k3_p4",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f35_l2_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f35_l2_local_a10_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f80_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f65_l1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l1_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_local_a15_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_cell_no_transfer_f65_l1_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_cell_no_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_cell_f65_l1_local_a15_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_cell_transfer",
                    local_alpha=0.15,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_cell_no_transfer_f65_l1_local_a15_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_atom_pair_a30_k3_p2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l1_atom_pair_a30_k3_p2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_atom_pair_a30_k3_p2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_atom_pair_a30_k3_p2_time48_rdk_a50_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.50,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l1_atom_pair_a30_k3_p2_time48_rdk_a50_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.50,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_atom_pair_a30_k3_p2_time48_rdk_a50_k3_p4",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l1_atom_pair_a30_k3_p2_time48_rdk_a50_k3_p4_time72_rdk_a25_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.50,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                    time_local_blends=(
                        TimeLocalBlendSpec(
                            time_h=72,
                            alpha=0.25,
                            topk=1,
                            power=1.0,
                            fingerprint="rdk",
                        ),
                    ),
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l1_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_local_a10_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_local_a20_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_no_transfer",
                    local_alpha=0.20,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l1_local_a20_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.20,
                    local_topk=1,
                    local_power=2.0,
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l1_local_a20_k1",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l2_source_resid50",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.50,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_ens75",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_weighted_no_transfer_f80_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_weighted_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_weighted_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_weighted_transfer",
                    residual_baseline_column="source_teacher_weighted_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_weighted_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_weighted_source_resid45_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_weighted_transfer",
                    residual_baseline_column="source_teacher_weighted_pred",
                    residual_shrinkage=0.45,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_weighted_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_weighted_source_resid35_ens90_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_weighted_transfer",
                    residual_baseline_column="source_teacher_weighted_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.90,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_weighted_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_weighted_cell_no_transfer_f80_l2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_weighted_cell_no_transfer",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_weighted_cell_f80_l2_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_weighted_cell_transfer",
                    residual_baseline_column="source_teacher_weighted_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_weighted_cell_no_transfer_f80_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_teacher_topk_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_teacher_topk_resid45_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.45,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_teacher_topk_resid60_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.60,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_teacher_topk_resid60_ens90_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.60,
                    residual_ensemble_weight=0.90,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid25_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.25,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid30_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.30,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid40_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.40,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid45_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.45,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f85_l2_source_resid45_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.45,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f85_l2_source_resid45_ens75_ridge_context_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.45,
                    residual_ensemble_weight=0.75,
                    calibration_method="ridge_context",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f85_l2_source_resid40_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.40,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f100_l2_source_resid40_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 1.0,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.40,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_ens50_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.50,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_ens90_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.90,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_ens75_ridge_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="ridge",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_ens75_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f80_l2_source_resid35_ens75_huber_cal_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.80,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l2_source_resid35",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l2_source_resid35_ens75",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l2_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                    paired_counterpart_id="extratrees_700_domain_rdkit_tt_no_transfer_f65_l2",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_tt_f65_l2_source_resid35_ens75_ridge_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="ridge",
                ),
                CandidateSpec(
                    "extratrees_900_domain_rdkit_tt_f65_l2_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 900,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2_source_resid50",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.50,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2_source_resid35",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2_source_resid35_ens75",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2_source_resid35_ens75_atom_pair_a20_k3_p2",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2_source_resid35_ens75_atom_pair_a20_k3_p2_time48_rdk_a35_k3_p4",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    time_local_alpha=0.35,
                    time_local_h=48,
                    time_local_topk=3,
                    time_local_power=4.0,
                    time_local_fingerprint="rdk",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f65_l2_source_resid35_ens75_huber_cal",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.35,
                    residual_ensemble_weight=0.75,
                    calibration_method="huber",
                ),
                CandidateSpec(
                    "extratrees_700_domain_rdkit_f50_l2_d10_source_resid50",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.50,
                        "max_depth": 10,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_transfer",
                    residual_baseline_column="source_teacher_pred",
                    residual_shrinkage=0.50,
                ),
                CandidateSpec(
                    "extratrees_700_f035_d10_rdkit_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": 0.35,
                        "max_depth": 10,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="rdkit_desc_time_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_f035_d10_rdkit_leaf2_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 2,
                        "max_features": 0.35,
                        "max_depth": 10,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="rdkit_desc_time_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_700_sqrt_rdkit_local_a10_k1",
                    "extratrees",
                    {
                        "n_estimators": 700,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="rdkit_desc_time_transfer",
                    local_alpha=0.10,
                    local_topk=1,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_600_leaf2_local_a20_k3",
                    "extratrees",
                    {
                        "n_estimators": 600,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "extratrees_600_leaf2_q25_a20_k5",
                    "extratrees",
                    {
                        "n_estimators": 600,
                        "min_samples_leaf": 2,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    local_alpha=0.20,
                    local_topk=5,
                    local_power=2.0,
                    local_statistic="quantile",
                    local_quantile=0.25,
                ),
                CandidateSpec(
                    "rf_500_leaf1",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                ),
                CandidateSpec(
                    "rf_500_leaf1_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_maccs_atom_time_transfer_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.60,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_500_maccs_atom_time_no_transfer_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_500_maccs_atom_time_no_transfer_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.60,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                *[
                    CandidateSpec(
                        f"rf_500_leaf1_f{int(max_features * 100):02d}_desc48_domain_a100",
                        "randomforest",
                        {
                            "n_estimators": 500,
                            "min_samples_leaf": 1,
                            "max_features": max_features,
                            "random_state": 20260605,
                            "n_jobs": -1,
                        },
                        descriptor_local_alpha=1.00,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                    )
                    for max_features in (0.55, 0.85)
                ],
                CandidateSpec(
                    "rf_500_leaf1_f100_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 1.0,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_leaf1_time_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_leaf1_f100_time_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 1.0,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_leaf1_maccs_atom_time_pathway_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_pathway_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_500_leaf1_maccs_atom_time_pathway_no_transfer_a20_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_500_leaf1_maccs_atom_time_pathway_no_transfer_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_pathway_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_leaf1_maccs_atom_time_local_profile_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_local_profile_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_500_leaf1_maccs_atom_time_local_profile_no_transfer_a20_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_500_leaf1_maccs_atom_time_local_profile_no_transfer_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="maccs_atom_time_local_profile_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_leaf1_domain_rdkit_tt_pathway_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_500_leaf1_domain_rdkit_tt_pathway_no_transfer_a20_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_500_leaf1_domain_rdkit_local_profile_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_500_leaf1_domain_rdkit_local_profile_no_transfer_a20_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_500_leaf1_domain_rdkit_local_profile_no_transfer_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_f85_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_f55_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.55,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_f60_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.60,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_f50_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.50,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_f45_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.45,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_f35_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_domain_rdkit_local_profile_no_transfer_a10_k5_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_no_transfer",
                    local_alpha=0.10,
                    local_topk=5,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_650_leaf1_domain_rdkit_local_profile_resid25_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=0.25,
                    residual_ensemble_weight=0.75,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_domain_rdkit_local_profile_resid100_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_f85_domain_rdkit_local_profile_resid100_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.85,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_f85_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_f55_domain_rdkit_local_profile_resid100_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.55,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_f55_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_f60_domain_rdkit_local_profile_resid100_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.60,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_f60_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_f50_domain_rdkit_local_profile_resid100_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.50,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_f50_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_f45_domain_rdkit_local_profile_resid100_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.45,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_f45_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_f35_domain_rdkit_local_profile_resid100_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.35,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_f35_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_domain_rdkit_local_profile_resid100_a10_k5_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_wmean",
                    residual_shrinkage=1.00,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.10,
                    local_topk=5,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_domain_rdkit_local_profile_no_transfer_a10_k5_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_650_leaf1_domain_rdkit_local_profile_q25_resid75_a15_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 650,
                        "min_samples_leaf": 1,
                        "max_features": 0.65,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_local_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=1.00,
                    local_alpha=0.15,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id="rf_650_leaf1_domain_rdkit_local_profile_no_transfer_a15_desc48_domain_a100",
                ),
                CandidateSpec(
                    "rf_500_leaf1_domain_rdkit_tt_pathway_no_transfer_a20_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_pathway_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_leaf1_domain_rdkit_tt_resid65_teacher_topk_desc48_domain_a100",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 1,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    feature_group="domain_rdkit_tt_transfer",
                    residual_baseline_column="teacher_topk_y_wmean",
                    residual_shrinkage=0.65,
                    residual_ensemble_weight=1.00,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "rf_500_leaf2_time_a20_k3",
                    "randomforest",
                    {
                        "n_estimators": 500,
                        "min_samples_leaf": 2,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                ),
                *[
                    CandidateSpec(
                        f"rf_500_leaf2_time_a20_desc48_domain_a{int(alpha * 100):03d}",
                        "randomforest",
                        {
                            "n_estimators": 500,
                            "min_samples_leaf": 2,
                            "max_features": 0.70,
                            "random_state": 20260605,
                            "n_jobs": -1,
                        },
                        local_alpha=0.20,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                        descriptor_local_alpha=alpha,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                    )
                    for alpha in (0.90, 1.00)
                ],
                CandidateSpec(
                    "svr_rbf_c16_eps005",
                    "svr",
                    {"C": 16.0, "epsilon": 0.05, "gamma": "scale"},
                ),
                CandidateSpec(
                    "svr_c16_eps001_domain_rdkit_no_transfer",
                    "svr",
                    {"C": 16.0, "epsilon": 0.001, "gamma": "scale"},
                    feature_group="domain_rdkit_no_transfer",
                ),
                CandidateSpec(
                    "svr_c16_eps001_domain_rdkit_transfer",
                    "svr",
                    {"C": 16.0, "epsilon": 0.001, "gamma": "scale"},
                    feature_group="domain_rdkit_transfer",
                    paired_counterpart_id="svr_c16_eps001_domain_rdkit_no_transfer",
                ),
                CandidateSpec(
                    "svr_c32_eps003_domain_rdkit_risk_profile_no_transfer_local_a30_desc48_domain_a100",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="domain_rdkit_risk_profile_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "svr_c32_eps003_domain_rdkit_risk_profile_transfer_local_a30_desc48_domain_a100",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="domain_rdkit_risk_profile_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "svr_c32_eps003_domain_rdkit_risk_profile_no_transfer_local_a30_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "svr_c48_eps002_domain_rdkit_risk_profile_no_transfer_local_a20_desc48_domain_a100",
                    "svr",
                    {"C": 48.0, "epsilon": 0.02, "gamma": "scale"},
                    feature_group="domain_rdkit_risk_profile_no_transfer",
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "svr_c48_eps002_domain_rdkit_risk_profile_resid15_local_a20_desc48_domain_a100",
                    "svr",
                    {"C": 48.0, "epsilon": 0.02, "gamma": "scale"},
                    feature_group="domain_rdkit_risk_profile_transfer",
                    residual_baseline_column="risk_profile_transfer_wmean",
                    residual_shrinkage=0.15,
                    residual_ensemble_weight=0.50,
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "svr_c48_eps002_domain_rdkit_risk_profile_no_transfer_local_a20_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "svr_c96_eps001_domain_rdkit_atom_local_risk_profile_no_transfer_local_a15_k4_desc48_domain_a100",
                    "svr",
                    {"C": 96.0, "epsilon": 0.01, "gamma": "scale"},
                    feature_group="domain_rdkit_atom_local_risk_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "svr_c96_eps001_domain_rdkit_atom_local_risk_profile_q25_resid75_local_a15_k4_desc48_domain_a100",
                    "svr",
                    {"C": 96.0, "epsilon": 0.01, "gamma": "scale"},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    residual_baseline_column="local_profile_transfer_q25",
                    residual_shrinkage=0.75,
                    residual_ensemble_weight=0.85,
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "svr_c96_eps001_domain_rdkit_atom_local_risk_profile_no_transfer_local_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "svr_c96_eps001_domain_rdkit_atom_local_risk_profile_transfer_local_a15_k4_desc48_domain_a100",
                    "svr",
                    {"C": 96.0, "epsilon": 0.01, "gamma": "scale"},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "svr_c96_eps001_domain_rdkit_atom_local_risk_profile_no_transfer_local_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "svr_c128_eps001_domain_rdkit_atom_local_risk_profile_no_transfer_local_a15_k4_desc48_domain_a100",
                    "svr",
                    {"C": 128.0, "epsilon": 0.01, "gamma": "scale"},
                    feature_group="domain_rdkit_atom_local_risk_profile_no_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "svr_c128_eps001_domain_rdkit_atom_local_risk_profile_transfer_local_a15_k4_desc48_domain_a100",
                    "svr",
                    {"C": 128.0, "epsilon": 0.01, "gamma": "scale"},
                    feature_group="domain_rdkit_atom_local_risk_profile_transfer",
                    local_alpha=0.15,
                    local_topk=4,
                    local_power=2.0,
                    local_fingerprint="atom_pair",
                    local_time_scale=24.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "svr_c128_eps001_domain_rdkit_atom_local_risk_profile_no_transfer_local_a15_k4_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "svr_rbf_c32_eps003_local_a20_k3",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                ),
                *[
                    CandidateSpec(
                        f"svr_rbf_c{int(c_value)}_eps{int(epsilon * 100):03d}_time_a{int(alpha * 100):02d}_k3",
                        "svr",
                        {"C": c_value, "epsilon": epsilon, "gamma": "scale"},
                        local_alpha=alpha,
                        local_topk=3,
                        local_power=2.0,
                        local_time_scale=24.0,
                    )
                    for c_value, epsilon in ((16.0, 0.05), (32.0, 0.03))
                    for alpha in (0.20, 0.40)
                ],
                CandidateSpec(
                    "svr_c32_eps003_maccs_atom_time_transfer_local_a30_k3",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="maccs_atom_time_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "svr_c32_eps003_maccs_atom_time_pathway_transfer_local_a30_k3",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="maccs_atom_time_pathway_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                ),
                CandidateSpec(
                    "svr_c32_eps003_maccs_atom_time_pathway_no_transfer_local_a30_desc48_domain_a100",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="maccs_atom_time_pathway_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "svr_c32_eps003_maccs_atom_time_pathway_transfer_local_a30_desc48_domain_a100",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="maccs_atom_time_pathway_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "svr_c32_eps003_maccs_atom_time_pathway_no_transfer_local_a30_desc48_domain_a100"
                    ),
                ),
                CandidateSpec(
                    "svr_c32_eps003_maccs_atom_time_risk_profile_no_transfer_local_a30_desc48_domain_a100",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="maccs_atom_time_risk_profile_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "svr_c32_eps003_maccs_atom_time_risk_profile_transfer_local_a30_desc48_domain_a100",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="maccs_atom_time_risk_profile_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                    paired_counterpart_id=(
                        "svr_c32_eps003_maccs_atom_time_risk_profile_no_transfer_local_a30_desc48_domain_a100"
                    ),
                ),
                *[
                    CandidateSpec(
                        f"svr_c32_eps003_maccs_atom_time_transfer_local_a30_desc48_domain_a{int(alpha * 100):03d}",
                        "svr",
                        {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                        feature_group="maccs_atom_time_transfer",
                        local_alpha=0.30,
                        local_topk=3,
                        local_power=2.0,
                        descriptor_local_alpha=alpha,
                        descriptor_local_topk=1,
                        descriptor_local_power=2.0,
                        descriptor_local_feature_group="domain_only",
                        descriptor_local_scaler="robust",
                        descriptor_local_time_h=48,
                        descriptor_local_min_delta=0.0,
                        paired_counterpart_id=(
                            "svr_c32_eps003_maccs_atom_time_no_transfer_local_a30_desc48_domain_a100"
                            if alpha == 1.00
                            else None
                        ),
                    )
                    for alpha in (0.90, 1.00)
                ],
                CandidateSpec(
                    "svr_c32_eps003_maccs_atom_time_no_transfer_local_a30_desc48_domain_a100",
                    "svr",
                    {"C": 32.0, "epsilon": 0.03, "gamma": "scale"},
                    feature_group="maccs_atom_time_no_transfer",
                    local_alpha=0.30,
                    local_topk=3,
                    local_power=2.0,
                    descriptor_local_alpha=1.00,
                    descriptor_local_topk=1,
                    descriptor_local_power=2.0,
                    descriptor_local_feature_group="domain_only",
                    descriptor_local_scaler="robust",
                    descriptor_local_time_h=48,
                    descriptor_local_min_delta=0.0,
                ),
                CandidateSpec(
                    "stack_et_svr_ridge",
                    "stacking",
                    {
                        "cv": 5,
                        "random_state": 20260605,
                    },
                ),
                CandidateSpec(
                    "stack_et_svr_ridge_time_a20_k3",
                    "stacking",
                    {
                        "cv": 5,
                        "random_state": 20260605,
                    },
                    local_alpha=0.20,
                    local_topk=3,
                    local_power=2.0,
                    local_time_scale=24.0,
                ),
            ]
        )
    if mode == "quick":

        candidates.extend(
            [
                CandidateSpec(
                    "scientific_rf_reg24",
                    "randomforest",
                    {
                        "n_estimators": 300,
                        "min_samples_leaf": 24,
                        "max_features": 0.70,
                        "random_state": 20260605,
                        "n_jobs": -1,
                    },
                ),
                CandidateSpec(
                    "scientific_xgb_reg",
                    "xgboost",
                    {
                        **xgb_base,
                        "n_estimators": 140,
                        "max_depth": 2,
                        "learning_rate": 0.03,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                        "reg_lambda": 10.0,
                        "min_child_weight": 8.0,
                    },
                ),
                CandidateSpec(
                    "scientific_cat_reg",
                    "catboost",
                    {
                        **cat_base,
                        "iterations": 120,
                        "depth": 2,
                        "learning_rate": 0.03,
                        "l2_leaf_reg": 12.0,
                    },
                ),
                CandidateSpec(
                    "scientific_svr_reg",
                    "svr",
                    {"C": 8.0, "epsilon": 0.05, "gamma": "scale"},
                ),
            ]
        )
        candidates = _prioritize_candidates(candidates, QUICK_PRIORITY_CANDIDATE_IDS)
    if max_candidates is not None:
        return candidates[:max_candidates]
    return candidates

def _make_estimator(spec: CandidateSpec):
    params = _estimator_params(spec)
    if spec.family == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    if spec.family == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**params)
    if spec.family == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(**params)
    if spec.family == "randomforest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)
    if spec.family == "svr":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        return make_pipeline(StandardScaler(), SVR(**params))
    if spec.family == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), RidgeCV(**params))
    if spec.family == "stacking":
        from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        estimators = [
            (
                "et",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=1,
                    max_features=0.65,
                    random_state=20260605,
                    n_jobs=-1,
                ),
            ),
            ("svr", make_pipeline(StandardScaler(), SVR(C=16.0, epsilon=0.05, gamma="scale"))),
            ("ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)))),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)),
            cv=int(spec.params.get("cv", 5)),
            n_jobs=-1,
            passthrough=False,
        )
    raise ValueError(f"Unknown model family: {spec.family}")

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

CV_TIME_SUBGROUPS = (3, 6, 12, 24, 48, 72)

def _time_subgroup_cv_metrics(train: pd.DataFrame, y: np.ndarray, oof: np.ndarray) -> dict[str, float]:
    if "time_h" not in train.columns:
        return {}
    metrics: dict[str, float] = {}
    times = train["time_h"].astype(int).to_numpy()
    for time_h in CV_TIME_SUBGROUPS:
        mask = times == int(time_h)
        metrics[f"time{time_h}_n"] = int(mask.sum())
        metrics[f"time{time_h}_oof_r2"] = float(r2_score(y[mask], oof[mask])) if int(mask.sum()) >= 2 else np.nan
        metrics[f"time{time_h}_rmse"] = _rmse(y[mask], oof[mask]) if int(mask.sum()) >= 1 else np.nan
    return metrics

def _fit_predict_raw_candidate(
    train: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    spec: CandidateSpec,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    return_fit_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    estimator = _make_estimator(spec)
    estimator.fit(x[fit_idx], y[fit_idx] - baseline[fit_idx])
    preds = baseline[valid_idx] + np.asarray(estimator.predict(x[valid_idx]), dtype=float)
    fit_preds = baseline[fit_idx] + np.asarray(estimator.predict(x[fit_idx]), dtype=float)
    residual_weight = float(spec.residual_ensemble_weight)
    if spec.residual_baseline_column is not None and residual_weight < 1.0:
        plain_estimator = _make_estimator(spec)
        plain_estimator.fit(x[fit_idx], y[fit_idx])
        plain_preds = np.asarray(plain_estimator.predict(x[valid_idx]), dtype=float)
        preds = (1.0 - residual_weight) * plain_preds + residual_weight * preds
        plain_fit_preds = np.asarray(plain_estimator.predict(x[fit_idx]), dtype=float)
        fit_preds = (1.0 - residual_weight) * plain_fit_preds + residual_weight * fit_preds
    if spec.local_alpha > 0:
        local_preds = local_read_across_predict(
            fit_smiles=train.iloc[fit_idx]["canonical_smiles"],
            fit_y=y[fit_idx],
            query_smiles=train.iloc[valid_idx]["canonical_smiles"],
            fit_time_h=train.iloc[fit_idx]["time_h"],
            query_time_h=train.iloc[valid_idx]["time_h"],
            topk=spec.local_topk,
            power=spec.local_power,
            time_scale=spec.local_time_scale,
            statistic=spec.local_statistic,
            quantile=spec.local_quantile,
            fingerprint_kind=spec.local_fingerprint,
        )
        preds = (1.0 - spec.local_alpha) * preds + spec.local_alpha * local_preds
        local_fit_preds = local_read_across_predict(
            fit_smiles=train.iloc[fit_idx]["canonical_smiles"],
            fit_y=y[fit_idx],
            query_smiles=train.iloc[fit_idx]["canonical_smiles"],
            fit_time_h=train.iloc[fit_idx]["time_h"],
            query_time_h=train.iloc[fit_idx]["time_h"],
            topk=spec.local_topk,
            power=spec.local_power,
            exclude_self=True,
            time_scale=spec.local_time_scale,
            statistic=spec.local_statistic,
            quantile=spec.local_quantile,
            fingerprint_kind=spec.local_fingerprint,
        )
        fit_preds = (1.0 - spec.local_alpha) * fit_preds + spec.local_alpha * local_fit_preds
    time_blends = time_local_blends_for_spec(spec)
    if time_blends:
        preds = apply_time_specific_local_blends(
            base_pred=preds,
            fit_frame=train.iloc[fit_idx].reset_index(drop=True),
            fit_y=y[fit_idx],
            query_frame=train.iloc[valid_idx].reset_index(drop=True),
            blends=time_blends,
        )
        fit_preds = apply_time_specific_local_blends(
            base_pred=fit_preds,
            fit_frame=train.iloc[fit_idx].reset_index(drop=True),
            fit_y=y[fit_idx],
            query_frame=train.iloc[fit_idx].reset_index(drop=True),
            blends=time_blends,
            exclude_self=True,
        )
    if spec.descriptor_local_alpha > 0:
        preds = apply_descriptor_local_blend(
            base_pred=preds,
            fit_frame=train.iloc[fit_idx],
            fit_y=y[fit_idx],
            query_frame=train.iloc[valid_idx],
            spec=spec,
        )
        fit_preds = apply_descriptor_local_blend(
            base_pred=fit_preds,
            fit_frame=train.iloc[fit_idx],
            fit_y=y[fit_idx],
            query_frame=train.iloc[fit_idx],
            spec=spec,
            exclude_self=True,
            query_indices=train.iloc[fit_idx].index.to_numpy(),
        )
    if return_fit_predictions:
        return np.asarray(preds, dtype=float), np.asarray(fit_preds, dtype=float)
    return np.asarray(preds, dtype=float)

def _oof_raw_predictions(
    train: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    spec: CandidateSpec,
    cv_folds: int,
) -> np.ndarray:
    oof = np.zeros(len(train), dtype=float)
    for fit_idx, valid_idx in make_cv_splits(train, cv_folds=cv_folds):
        oof[valid_idx] = _fit_predict_raw_candidate(train, x, y, baseline, spec, fit_idx, valid_idx)
    return oof

def cross_validate_candidates(
    matrix: ModelMatrix,
    candidates: list[CandidateSpec],
    cv_folds: int = 5,
    secondary_cv_strategy: str | None = None,
) -> pd.DataFrame:
    train = matrix.role("train")
    y = train["y"].to_numpy(dtype=float)
    cv_splits = make_cv_splits(train, cv_folds=cv_folds)

    rows = []
    for spec in candidates:
        if spec.family in {
            "transfer_blend",
            "calibrated_transfer_blend",
            "gated_transfer",
            "a90_gated_transfer",
            "a90_delta_gated_transfer",
            "risk_adjusted_gated_transfer",
            "disagreement_gated_transfer",
            "risk_routed_transfer",
        }:
            feature_columns = []
            x = np.empty((len(train), 0), dtype=float)
            baseline = np.zeros(len(train), dtype=float)
        else:
            feature_columns = feature_columns_for_spec(matrix, spec)
            x = _feature_matrix_for_spec(train, feature_columns, spec)
            baseline = residual_baseline_for_spec(train, spec)

        def run_cv(cv_splits_to_run: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
            oof = np.zeros(len(train), dtype=float)
            fold_r2 = []
            fold_rmse = []
            fold_train_r2 = []
            for fit_idx, valid_idx in cv_splits_to_run:
                if spec.family == "gated_transfer":
                    preds, train_preds = _fit_gated_transfer_fold(
                        train,
                        matrix,
                        y,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                elif spec.family == "a90_gated_transfer":
                    preds, train_preds = _fit_a90_gated_transfer_fold(
                        train,
                        matrix,
                        y,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                elif spec.family == "a90_delta_gated_transfer":
                    preds, train_preds = _fit_a90_delta_gated_transfer_fold(
                        train,
                        matrix,
                        y,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                elif spec.family == "risk_adjusted_gated_transfer":
                    preds, train_preds = _fit_risk_adjusted_gated_transfer_fold(
                        train,
                        matrix,
                        y,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                elif spec.family == "disagreement_gated_transfer":
                    preds, train_preds = _fit_disagreement_gated_transfer_fold(
                        train,
                        matrix,
                        y,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                elif spec.family == "risk_routed_transfer":
                    preds, train_preds = _fit_risk_routed_transfer_fold(
                        train,
                        matrix,
                        y,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                elif spec.family in {"transfer_blend", "calibrated_transfer_blend"}:
                    preds, train_preds = _fit_transfer_blend_fold(
                        train,
                        matrix,
                        y,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                else:
                    preds, train_preds = _fit_predict_raw_candidate(
                        train,
                        x,
                        y,
                        baseline,
                        spec,
                        fit_idx,
                        valid_idx,
                        return_fit_predictions=True,
                    )
                if spec.calibration_method != "none":
                    fit_frame = train.iloc[fit_idx].reset_index(drop=True)
                    fit_x = x[fit_idx]
                    fit_y = y[fit_idx]
                    fit_baseline = baseline[fit_idx]
                    calibration_train_pred = _oof_raw_predictions(
                        fit_frame,
                        fit_x,
                        fit_y,
                        fit_baseline,
                        spec,
                        cv_folds=min(cv_folds, fit_frame["canonical_smiles"].nunique()),
                    )
                    preds = calibrate_predictions(
                        calibration_train_pred,
                        fit_y,
                        preds,
                        method=spec.calibration_method,
                        train_frame=fit_frame,
                        query_frame=train.iloc[valid_idx].reset_index(drop=True),
                    )
                    train_preds = calibrate_predictions(
                        calibration_train_pred,
                        fit_y,
                        train_preds,
                        method=spec.calibration_method,
                        train_frame=fit_frame,
                        query_frame=fit_frame,
                    )
                oof[valid_idx] = preds
                fold_r2.append(float(r2_score(y[valid_idx], preds)))
                fold_rmse.append(_rmse(y[valid_idx], preds))
                fold_train_r2.append(float(r2_score(y[fit_idx], train_preds)))
            fold_r2_mean = float(np.mean(fold_r2))
            fold_train_r2_mean = float(np.mean(fold_train_r2))
            return {
                "oof_r2": float(r2_score(y, oof)),
                "oof_rmse": _rmse(y, oof),
                "fold_r2_mean": fold_r2_mean,
                "fold_r2_min": float(np.min(fold_r2)),
                "fold_rmse_mean": float(np.mean(fold_rmse)),
                "fold_train_r2_mean": fold_train_r2_mean,
                "generalization_gap": max(0.0, fold_train_r2_mean - fold_r2_mean),
                **_time_subgroup_cv_metrics(train, y, oof),
            }

        primary = run_cv(cv_splits)
        row = {
            "candidate_id": spec.candidate_id,
            "paired_counterpart_id": spec.paired_counterpart_id or "",
            "family": spec.family,
            "feature_group": spec.feature_group,
            "feature_count": (
                _transfer_blend_feature_count(matrix, spec)
                if spec.family in {"transfer_blend", "calibrated_transfer_blend", "gated_transfer"}
                else _a90_gated_feature_count(matrix, spec)
                if spec.family == "a90_gated_transfer"
                else _a90_delta_gated_feature_count(matrix, spec)
                if spec.family == "a90_delta_gated_transfer"
                else _risk_adjusted_gated_feature_count(matrix, spec)
                if spec.family == "risk_adjusted_gated_transfer"
                else _disagreement_gated_feature_count(matrix, spec)
                if spec.family == "disagreement_gated_transfer"
                else _risk_routed_feature_count(matrix, spec)
                if spec.family == "risk_routed_transfer"
                else len(feature_columns)
            ),
            "cv_strategy": "compound_group_kfold",
            "residual_baseline_column": spec.residual_baseline_column or "",
            "residual_shrinkage": float(spec.residual_shrinkage),
            "residual_ensemble_weight": float(spec.residual_ensemble_weight),
            "calibration_method": spec.calibration_method,
            "local_fingerprint": spec.local_fingerprint,
            "descriptor_local_alpha": float(spec.descriptor_local_alpha),
            "descriptor_local_topk": int(spec.descriptor_local_topk),
            "descriptor_local_power": float(spec.descriptor_local_power),
            "descriptor_local_feature_group": spec.descriptor_local_feature_group,
            "descriptor_local_scaler": spec.descriptor_local_scaler,
            "descriptor_local_time_h": int(spec.descriptor_local_time_h) if spec.descriptor_local_time_h is not None else "",
            "descriptor_local_min_delta": (
                float(spec.descriptor_local_min_delta) if spec.descriptor_local_min_delta is not None else ""
            ),
            "time_local_alpha": float(spec.time_local_alpha),
            "time_local_h": int(spec.time_local_h) if spec.time_local_h is not None else "",
            "time_local_fingerprint": spec.time_local_fingerprint,
            "time_local_blends": format_time_local_blends(spec),
            "cv_oof_r2": primary["oof_r2"],
            "cv_oof_rmse": primary["oof_rmse"],
            "cv_fold_r2_mean": primary["fold_r2_mean"],
            "cv_fold_r2_min": primary["fold_r2_min"],
            "cv_fold_rmse_mean": primary["fold_rmse_mean"],
            "cv_fold_train_r2_mean": primary["fold_train_r2_mean"],
            "cv_generalization_gap": primary["generalization_gap"],
        }
        row.update({f"cv_{key}": value for key, value in primary.items() if key.startswith("time")})
        if secondary_cv_strategy is not None and str(secondary_cv_strategy) != "none":
            if str(secondary_cv_strategy) != "doi":
                raise ValueError(f"Unknown secondary CV strategy: {secondary_cv_strategy}")
            doi_folds = min(int(cv_folds), int(train["doi"].astype(str).nunique()))
            secondary = run_cv(make_group_cv_splits(train, group_column="doi", cv_folds=doi_folds))
            row.update(
                {
                    "doi_cv_oof_r2": secondary["oof_r2"],
                    "doi_cv_oof_rmse": secondary["oof_rmse"],
                    "doi_cv_fold_r2_mean": secondary["fold_r2_mean"],
                    "doi_cv_fold_r2_min": secondary["fold_r2_min"],
                    "doi_cv_fold_rmse_mean": secondary["fold_rmse_mean"],
                    "doi_cv_fold_train_r2_mean": secondary["fold_train_r2_mean"],
                    "doi_cv_generalization_gap": secondary["generalization_gap"],
                }
            )
            row.update({f"doi_cv_{key}": value for key, value in secondary.items() if key.startswith("time")})
        rows.append(row)
    return pd.DataFrame(rows)

def fit_selected_and_evaluate(matrix: ModelMatrix, spec: CandidateSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = matrix.role("train")
    if spec.family in {
        "transfer_blend",
        "calibrated_transfer_blend",
        "gated_transfer",
        "a90_gated_transfer",
        "a90_delta_gated_transfer",
        "risk_adjusted_gated_transfer",
        "disagreement_gated_transfer",
        "risk_routed_transfer",
    }:
        metric_rows = []
        prediction_rows = []
        for role in ["train", "test", "validation", "experimental"]:
            role_frame = matrix.role(role)
            y_true = role_frame["y"].to_numpy(dtype=float)
            if spec.family == "gated_transfer":
                y_pred = _fit_gated_transfer_and_predict_role(matrix, spec, role)
            elif spec.family == "calibrated_transfer_blend":
                y_pred = _fit_transfer_blend_and_predict_role(matrix, spec, role)
            elif spec.family == "a90_gated_transfer":
                y_pred = _fit_a90_gated_transfer_and_predict_role(matrix, spec, role)
            elif spec.family == "a90_delta_gated_transfer":
                y_pred = _fit_a90_delta_gated_transfer_and_predict_role(matrix, spec, role)
            elif spec.family == "risk_adjusted_gated_transfer":
                y_pred = _fit_risk_adjusted_gated_transfer_and_predict_role(matrix, spec, role)
            elif spec.family == "disagreement_gated_transfer":
                y_pred = _fit_disagreement_gated_transfer_and_predict_role(matrix, spec, role)
            elif spec.family == "risk_routed_transfer":
                y_pred = _fit_risk_routed_transfer_and_predict_role(matrix, spec, role)
            else:
                y_pred = _fit_transfer_blend_and_predict_role(matrix, spec, role)
            metric_rows.append(
                {
                    "candidate_id": spec.candidate_id,
                    "paired_counterpart_id": spec.paired_counterpart_id or "",
                    "family": spec.family,
                    "feature_group": spec.feature_group,
                    "feature_count": (
                        _a90_gated_feature_count(matrix, spec)
                        if spec.family == "a90_gated_transfer"
                        else _a90_delta_gated_feature_count(matrix, spec)
                        if spec.family == "a90_delta_gated_transfer"
                        else _risk_adjusted_gated_feature_count(matrix, spec)
                        if spec.family == "risk_adjusted_gated_transfer"
                        else _disagreement_gated_feature_count(matrix, spec)
                        if spec.family == "disagreement_gated_transfer"
                        else _risk_routed_feature_count(matrix, spec)
                        if spec.family == "risk_routed_transfer"
                        else _transfer_blend_feature_count(matrix, spec)
                        if spec.family in {"transfer_blend", "calibrated_transfer_blend"}
                        else _transfer_blend_feature_count(matrix, spec)
                    ),
                    "residual_baseline_column": spec.residual_baseline_column or "",
                    "residual_shrinkage": float(spec.residual_shrinkage),
                    "residual_ensemble_weight": float(spec.residual_ensemble_weight),
                    "calibration_method": spec.calibration_method,
                    "local_fingerprint": spec.local_fingerprint,
                    "descriptor_local_alpha": float(spec.descriptor_local_alpha),
                    "descriptor_local_topk": int(spec.descriptor_local_topk),
                    "descriptor_local_power": float(spec.descriptor_local_power),
                    "descriptor_local_feature_group": spec.descriptor_local_feature_group,
                    "descriptor_local_scaler": spec.descriptor_local_scaler,
                    "descriptor_local_time_h": (
                        int(spec.descriptor_local_time_h) if spec.descriptor_local_time_h is not None else ""
                    ),
                    "descriptor_local_min_delta": (
                        float(spec.descriptor_local_min_delta) if spec.descriptor_local_min_delta is not None else ""
                    ),
                    "time_local_alpha": float(spec.time_local_alpha),
                    "time_local_h": int(spec.time_local_h) if spec.time_local_h is not None else "",
                    "time_local_fingerprint": spec.time_local_fingerprint,
                    "time_local_blends": format_time_local_blends(spec),
                    "split_role": role,
                    "rows": int(len(role_frame)),
                    "r2": float(r2_score(y_true, y_pred)) if len(role_frame) >= 2 else np.nan,
                    "rmse": _rmse(y_true, y_pred),
                    "mae": float(np.mean(np.abs(y_true - y_pred))),
                }
            )
            for sample_id, canonical, time_h, observed, predicted in zip(
                role_frame["sample_id"],
                role_frame["canonical_smiles"],
                role_frame["time_h"],
                y_true,
                y_pred,
            ):
                prediction_rows.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "split_role": role,
                        "sample_id": sample_id,
                        "canonical_smiles": canonical,
                        "time_h": int(time_h),
                        "y_true": float(observed),
                        "y_pred": float(predicted),
                    }
                )
        return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)

    feature_columns = feature_columns_for_spec(matrix, spec)
    train_x = _feature_matrix_for_spec(train, feature_columns, spec)
    train_y = train["y"].to_numpy(dtype=float)
    train_baseline = residual_baseline_for_spec(train, spec)
    calibration_train_pred = None
    if spec.calibration_method != "none":
        calibration_train_pred = _oof_raw_predictions(
            train,
            train_x,
            train_y,
            train_baseline,
            spec,
            cv_folds=5,
        )
    estimator = _make_estimator(spec)
    estimator.fit(
        train_x,
        train_y - train_baseline,
    )
    plain_estimator = None
    residual_weight = float(spec.residual_ensemble_weight)
    if spec.residual_baseline_column is not None and residual_weight < 1.0:
        plain_estimator = _make_estimator(spec)
        plain_estimator.fit(train_x, train_y)

    metric_rows = []
    prediction_rows = []
    for role in ["train", "test", "validation", "experimental"]:
        role_frame = matrix.role(role)
        x = _feature_matrix_for_spec(role_frame, feature_columns, spec)
        y_true = role_frame["y"].to_numpy(dtype=float)
        role_baseline = residual_baseline_for_spec(role_frame, spec)
        y_pred = role_baseline + np.asarray(estimator.predict(x), dtype=float)
        if plain_estimator is not None:
            plain_pred = np.asarray(plain_estimator.predict(x), dtype=float)
            y_pred = (1.0 - residual_weight) * plain_pred + residual_weight * y_pred
        if spec.local_alpha > 0:
            local_pred = local_read_across_predict(
                fit_smiles=train["canonical_smiles"],
                fit_y=train["y"],
                query_smiles=role_frame["canonical_smiles"],
                fit_time_h=train["time_h"],
                query_time_h=role_frame["time_h"],
                topk=spec.local_topk,
                power=spec.local_power,
                exclude_self=role == "train",
                time_scale=spec.local_time_scale,
                statistic=spec.local_statistic,
                quantile=spec.local_quantile,
                fingerprint_kind=spec.local_fingerprint,
            )
            y_pred = (1.0 - spec.local_alpha) * y_pred + spec.local_alpha * local_pred
        time_blends = time_local_blends_for_spec(spec)
        if time_blends:
            y_pred = apply_time_specific_local_blends(
                base_pred=y_pred,
                fit_frame=train,
                fit_y=train_y,
                query_frame=role_frame,
                blends=time_blends,
                exclude_self=role == "train",
            )
        if spec.descriptor_local_alpha > 0:
            y_pred = apply_descriptor_local_blend(
                base_pred=y_pred,
                fit_frame=train,
                fit_y=train_y,
                query_frame=role_frame,
                spec=spec,
                exclude_self=role == "train",
                query_indices=role_frame.index.to_numpy() if role == "train" else None,
            )
        if calibration_train_pred is not None:
            y_pred = calibrate_predictions(
                calibration_train_pred,
                train_y,
                y_pred,
                method=spec.calibration_method,
                train_frame=train,
                query_frame=role_frame,
            )
        metric_rows.append(
            {
                "candidate_id": spec.candidate_id,
                "paired_counterpart_id": spec.paired_counterpart_id or "",
                "family": spec.family,
                "feature_group": spec.feature_group,
                "feature_count": len(feature_columns),
                "residual_baseline_column": spec.residual_baseline_column or "",
                "residual_shrinkage": float(spec.residual_shrinkage),
                "residual_ensemble_weight": float(spec.residual_ensemble_weight),
                "calibration_method": spec.calibration_method,
                "local_fingerprint": spec.local_fingerprint,
                "descriptor_local_alpha": float(spec.descriptor_local_alpha),
                "descriptor_local_topk": int(spec.descriptor_local_topk),
                "descriptor_local_power": float(spec.descriptor_local_power),
                "descriptor_local_feature_group": spec.descriptor_local_feature_group,
                "descriptor_local_scaler": spec.descriptor_local_scaler,
                "descriptor_local_time_h": (
                    int(spec.descriptor_local_time_h) if spec.descriptor_local_time_h is not None else ""
                ),
                "descriptor_local_min_delta": (
                    float(spec.descriptor_local_min_delta) if spec.descriptor_local_min_delta is not None else ""
                ),
                "time_local_alpha": float(spec.time_local_alpha),
                "time_local_h": int(spec.time_local_h) if spec.time_local_h is not None else "",
                "time_local_fingerprint": spec.time_local_fingerprint,
                "time_local_blends": format_time_local_blends(spec),
                "split_role": role,
                "rows": int(len(role_frame)),
                "r2": float(r2_score(y_true, y_pred)) if len(role_frame) >= 2 else np.nan,
                "rmse": _rmse(y_true, y_pred),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
            }
        )
        for sample_id, canonical, time_h, observed, predicted in zip(
            role_frame["sample_id"],
            role_frame["canonical_smiles"],
            role_frame["time_h"],
            y_true,
            y_pred,
        ):
            prediction_rows.append(
                {
                    "candidate_id": spec.candidate_id,
                    "split_role": role,
                    "sample_id": sample_id,
                    "canonical_smiles": canonical,
                    "time_h": int(time_h),
                    "y_true": float(observed),
                    "y_pred": float(predicted),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)

def candidate_by_id(candidates: list[CandidateSpec], candidate_id: str) -> CandidateSpec:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)

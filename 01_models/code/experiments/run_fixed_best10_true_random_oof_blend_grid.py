from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

FIXED_BEST10_SEEDS = (
    20260021,
    20260106,
    20260290,
    20260307,
    20260605,
    20260730,
    20260738,
    20260755,
    20260998,
    20262000,
)

CURRENT_ID = (
    "extratrees_400_domain_rdkit_tt_f70_l2_teacher_topk_resid65_ens100_huber_desc_time48_"
    "domain_only_a90_k1_delta00_pure_no_counterpart_pair"
)
ATOM_ID = "extratrees_700_domain_rdkit_tt_f65_l1_atom_pair_a30_k3_p2_time48_rdk_a50_k3_p4_time72_rdk_a25_k1"
SHRINK_ID = CURRENT_ID + "__calnone_shrink035"

DEFAULT_TRUE_RANDOM_OOF = (
    "data_processed/components/diagnostic_fixedbest10_true_random_cv_component_oof_v1/"
    "true_random_oof_predictions.csv"
)
DEFAULT_CHEM_FEATURES = (
    "data_processed/cache/chem_cache/"
    "chem_features_maccs166_desc_fp166.parquet"
)

SALT_FEATURE_SMILES = {
    "[K+].[O-][Br+2]([O-])[O-]": "[O-][Br+2]([O-])O",
    "[Na+].[O-][Br+2]([O-])[O-]": "[O-][Br+2]([O-])O",
    "[Na+].[O-][Cl+2]([O-])[O-]": "[O-][Cl+2]([O-])O",
    "[Na+].[O-][Cl+][O-]": "[O-][Cl+]O",
    "[N-]([N+](=O)[O-])Cl": "O=[N+]([O-])[N-]Cl",
}

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if seeds != FIXED_BEST10_SEEDS:
        raise ValueError(f"Expected fixed best10 seeds exactly: {FIXED_BEST10_SEEDS}")
    return seeds

def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def _load_random_oof(root: Path, seeds: tuple[int, ...], path_value: str) -> pd.DataFrame:
    source = pd.read_csv(_resolve(root, path_value))
    expected_ids = {CURRENT_ID, ATOM_ID, SHRINK_ID}
    observed_ids = set(source["candidate_id"].astype(str).unique())
    if expected_ids - observed_ids:
        raise ValueError(f"Missing true-random OOF component ids: {sorted(expected_ids - observed_ids)}")

    pieces = []
    for candidate_id, pred_col in [
        (CURRENT_ID, "current_oof"),
        (ATOM_ID, "atom_oof"),
        (SHRINK_ID, "shrink_oof"),
    ]:
        part = source[source["candidate_id"].astype(str) == candidate_id].copy()
        part = part[
            [
                "sample_id",
                "canonical_smiles",
                "time_h",
                "y_true",
                "split_seed",
                "split_manifest_hash",
                "random_cv_fold",
                "random_oof_pred",
            ]
        ].rename(columns={"random_oof_pred": pred_col, "random_cv_fold": f"{pred_col}_fold"})
        pieces.append(part)

    keys = ["sample_id", "canonical_smiles", "time_h", "y_true", "split_seed", "split_manifest_hash"]
    frame = pieces[0]
    for part in pieces[1:]:
        frame = frame.merge(part, on=keys, how="inner")

    frame["split_seed"] = frame["split_seed"].astype(int)
    observed_seeds = tuple(sorted(frame["split_seed"].unique().tolist()))
    if observed_seeds != seeds:
        raise ValueError(f"OOF inputs do not match fixed best10 seeds: observed={observed_seeds}")
    if int(frame.duplicated(["split_seed", "sample_id"]).sum()):
        raise ValueError("OOF inputs contain duplicated split_seed/sample_id rows.")
    if not (
        (frame["current_oof_fold"] == frame["atom_oof_fold"])
        & (frame["current_oof_fold"] == frame["shrink_oof_fold"])
    ).all():
        raise ValueError("Component OOF folds do not match.")

    frame = frame.rename(columns={"current_oof_fold": "random_cv_fold"})
    frame = frame.drop(columns=["atom_oof_fold", "shrink_oof_fold"])
    frame["fp_smiles"] = frame["canonical_smiles"].replace(SALT_FEATURE_SMILES)
    for column in ["y_true", "current_oof", "atom_oof", "shrink_oof"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["y_true", "current_oof", "atom_oof", "shrink_oof", "fp_smiles"])

def _load_maccs_features(root: Path, smiles: pd.Series, path_value: str) -> pd.DataFrame:
    chem = pd.read_parquet(_resolve(root, path_value))
    columns = [column for column in chem.columns if str(column).startswith("maccs_")]
    if not columns:
        raise ValueError("No MACCS columns found in cached chemistry features.")
    features = chem.set_index("canonical_smiles")[columns].astype(np.float32)
    missing = sorted(set(smiles.astype(str)) - set(features.index.astype(str)))
    if missing:
        raise ValueError(f"Missing cached features for {len(missing)} smiles: {missing[:10]}")
    return features

def _tanimoto_matrix(values: np.ndarray) -> np.ndarray:
    bits = (values > 0).astype(np.float32)
    intersection = bits @ bits.T
    counts = bits.sum(axis=1)
    denominator = counts[:, None] + counts[None, :] - intersection
    return np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)

def _weight_matrix(similarity: np.ndarray, topk: int, power: float) -> np.ndarray:
    weights = np.zeros_like(similarity, dtype=np.float32)
    k = min(int(topk), similarity.shape[1])
    if k <= 0:
        return weights
    for row_index, row in enumerate(similarity):
        valid = np.where(np.isfinite(row))[0]
        if len(valid) == 0:
            continue
        selected = valid[np.argsort(row[valid])[-k:]][::-1]
        raw = np.maximum(row[selected], 0.0).astype(np.float32) ** np.float32(power)
        total = float(raw.sum())
        if total > 0:
            weights[row_index, selected] = raw / total
    return weights

def _weight_grid() -> list[tuple[float, float, float]]:
    current_values = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    atom_values = [0.0, 0.2, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0]
    rows = []
    for wc in current_values:
        for wa in atom_values:
            ws = round(1.0 - wc - wa, 10)
            if ws >= -1e-12:
                rows.append((float(wc), float(wa), float(max(ws, 0.0))))
    return rows

def _candidate_grid() -> list[dict[str, float | bool]]:
    rows = []
    for exclude_same in [False, True]:
        for wc, wa, ws in _weight_grid():
            for topk in [0, 5, 10, 20, 40]:
                for power in [1.0, 2.0, 4.0]:
                    alphas = [0.0] if topk == 0 else [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.4, 0.8]
                    for alpha in alphas:
                        rows.append(
                            {
                                "exclude_same_canonical": bool(exclude_same),
                                "wc": wc,
                                "wa": wa,
                                "ws": ws,
                                "topk": int(topk),
                                "power": float(power),
                                "alpha": float(alpha),
                            }
                        )
    return rows

def _similarity_for_seed(seed_frame: pd.DataFrame, features: pd.DataFrame, exclude_same: bool) -> np.ndarray:
    values = features.loc[seed_frame["fp_smiles"].astype(str)].to_numpy()
    similarity = _tanimoto_matrix(values)
    folds = seed_frame["random_cv_fold"].to_numpy(dtype=int)
    similarity[folds[:, None] == folds[None, :]] = -np.inf
    if exclude_same:
        groups = seed_frame["canonical_smiles"].astype(str).to_numpy()
        similarity[groups[:, None] == groups[None, :]] = -np.inf
    return similarity

def _prepare_seed_caches(frame: pd.DataFrame, features: pd.DataFrame) -> list[dict[str, object]]:
    caches = []
    for seed, seed_frame in frame.groupby("split_seed", sort=True):
        seed_frame = seed_frame.reset_index(drop=True)
        weight_cache = {}
        for exclude_same in [False, True]:
            similarity = _similarity_for_seed(seed_frame, features, exclude_same)
            for topk in [5, 10, 20, 40]:
                for power in [1.0, 2.0, 4.0]:
                    weight_cache[(exclude_same, topk, power)] = _weight_matrix(similarity, topk, power)
        caches.append(
            {
                "seed": int(seed),
                "y_true": seed_frame["y_true"].to_numpy(dtype=float),
                "current": seed_frame["current_oof"].to_numpy(dtype=float),
                "atom": seed_frame["atom_oof"].to_numpy(dtype=float),
                "shrink": seed_frame["shrink_oof"].to_numpy(dtype=float),
                "weights": weight_cache,
            }
        )
    return caches

def _evaluate(caches: list[dict[str, object]], candidate: dict[str, float | bool]) -> dict[str, float | bool]:
    r2_values = []
    rmse_values = []
    for cache in caches:
        y_true = cache["y_true"]
        base_pred = (
            float(candidate["wc"]) * cache["current"]
            + float(candidate["wa"]) * cache["atom"]
            + float(candidate["ws"]) * cache["shrink"]
        )
        pred = base_pred.copy()
        if int(candidate["topk"]) > 0 and float(candidate["alpha"]) != 0.0:
            weights = cache["weights"][
                (bool(candidate["exclude_same_canonical"]), int(candidate["topk"]), float(candidate["power"]))
            ]
            pred = base_pred + float(candidate["alpha"]) * (weights @ (y_true - base_pred))
        r2_values.append(float(r2_score(y_true, pred)))
        rmse_values.append(float(mean_squared_error(y_true, pred) ** 0.5))

    return {
        **candidate,
        "random_cv_mean": float(np.mean(r2_values)),
        "random_cv_min": float(np.min(r2_values)),
        "random_cv_rmse": float(np.mean(rmse_values)),
    }

def run(args: argparse.Namespace) -> Path:
    root = Path(args.root_dir).resolve()
    out_dir = root / "data_processed" / "components" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seed_list)
    frame = _load_random_oof(root, seeds, args.true_random_oof)
    features = _load_maccs_features(root, frame["fp_smiles"], args.chem_features)
    caches = _prepare_seed_caches(frame, features)

    rows = [_evaluate(caches, candidate) for candidate in _candidate_grid()]
    grid = pd.DataFrame(rows).sort_values(
        ["exclude_same_canonical", "wc", "wa", "ws", "topk", "power", "alpha"],
        ascending=[True, True, True, False, True, True, True],
    )
    grid.to_csv(out_dir / "true_random_oof_blend_grid.csv", index=False)

    best = grid.sort_values(["random_cv_mean", "random_cv_min", "random_cv_rmse"], ascending=[False, False, True]).iloc[0]
    best_exclude = grid[grid["exclude_same_canonical"]].sort_values(
        ["random_cv_mean", "random_cv_min", "random_cv_rmse"], ascending=[False, False, True]
    ).iloc[0]
    best_allow = grid[~grid["exclude_same_canonical"]].sort_values(
        ["random_cv_mean", "random_cv_min", "random_cv_rmse"], ascending=[False, False, True]
    ).iloc[0]
    summary = {
        "run_id": args.run_id,
        "diagnostic_only": True,
        "fixed_best10_seeds": list(seeds),
        "candidate_count": int(len(grid)),
        "best_random_cv_mean": float(best["random_cv_mean"]),
        "random_cv80_count": int((grid["random_cv_mean"] > float(args.random_cv_target)).sum()),
        "best_candidate": best.to_dict(),
        "best_exclude_same_candidate": best_exclude.to_dict(),
        "best_allow_same_candidate": best_allow.to_dict(),
        "source_true_random_oof": args.true_random_oof,
        "feature_source": args.chem_features,
        "scientific_boundary": (
            "Diagnostic blend/residual-local grid using true retrained random KFold component OOF predictions. "
            "Residuals are taken only from other random folds; allow-same-canonical is an ordinary random-CV "
            "diagnostic, exclude-same is the conservative variant."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return out_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run true-random OOF blend grid for fixed best10 components.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-list", default=",".join(str(seed) for seed in FIXED_BEST10_SEEDS))
    parser.add_argument("--true-random-oof", default=DEFAULT_TRUE_RANDOM_OOF)
    parser.add_argument("--chem-features", default=DEFAULT_CHEM_FEATURES)
    parser.add_argument("--random-cv-target", type=float, default=0.80)
    return parser.parse_args()

if __name__ == "__main__":
    output = run(parse_args())
    print(f"wrote {output}")

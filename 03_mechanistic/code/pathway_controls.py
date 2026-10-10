from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "01_models"
SRC_ROOT = PROJECT_ROOT / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SEEDS = (20260021, 20260106, 20260290, 20260307, 20260605, 20260730, 20260738, 20260755, 20260998, 20262000)
ROLES = ("random_cv_oof", "test", "validation", "experimental")
PATHWAYS = (
    "kegg_KEGG_APOPTOSIS",
    "kegg_KEGG_P53_SIGNALING_PATHWAY",
    "kegg_KEGG_BASE_EXCISION_REPAIR",
    "kegg_KEGG_OXIDATIVE_PHOSPHORYLATION",
    "kegg_KEGG_GLUTATHIONE_METABOLISM",
    "kegg_KEGG_CELL_CYCLE",
    "kegg_KEGG_DNA_REPLICATION",
    "kegg_KEGG_MAPK_SIGNALING_PATHWAY",
    "kegg_KEGG_METABOLISM_OF_XENOBIOTICS_BY_CYTOCHROME_P450",
    "kegg_KEGG_RIBOSOME",
)
LABELS = {
    "kegg_KEGG_APOPTOSIS": "Apoptosis",
    "kegg_KEGG_P53_SIGNALING_PATHWAY": "p53 signaling",
    "kegg_KEGG_BASE_EXCISION_REPAIR": "Base excision repair",
    "kegg_KEGG_OXIDATIVE_PHOSPHORYLATION": "OxPhos",
    "kegg_KEGG_GLUTATHIONE_METABOLISM": "Glutathione metabolism",
    "kegg_KEGG_CELL_CYCLE": "Cell cycle",
    "kegg_KEGG_DNA_REPLICATION": "DNA replication",
    "kegg_KEGG_MAPK_SIGNALING_PATHWAY": "MAPK signaling",
    "kegg_KEGG_METABOLISM_OF_XENOBIOTICS_BY_CYTOCHROME_P450": "Xenobiotic P450",
    "kegg_KEGG_RIBOSOME": "Ribosome",
}

def leave_one_pathway_columns(columns: list[str] | tuple[str, ...], pathway: str) -> list[str]:
    target = pathway if str(pathway).startswith("pathway_transfer_") else f"pathway_transfer_{pathway}"
    return [str(col) for col in columns if str(col) != target]

def pathway_delta(full: dict[str, float], leave_one_out: dict[str, float]) -> dict[str, float]:
    return {
        "delta_r2": float(full["r2"] - leave_one_out["r2"]),
        "delta_rmse": float(leave_one_out["rmse"] - full["rmse"]),
    }

def pathway_prediction_shifts(
    predictions: pd.DataFrame,
    pathways: list[str] | tuple[str, ...],
    role: str = "test",
) -> pd.DataFrame:

    full = predictions[(predictions["model"] == "full") & (predictions["split_role"] == role)]
    rows = []
    for pathway in pathways:
        loo = predictions[(predictions["model"] == pathway) & (predictions["split_role"] == role)]
        if full.empty or loo.empty:
            continue
        paired = full[["split_seed", "sample_id", "prediction"]].rename(columns={"prediction": "full_prediction"}).merge(
            loo[["split_seed", "sample_id", "prediction"]].rename(columns={"prediction": "loo_prediction"}),
            on=["split_seed", "sample_id"],
            how="inner",
        )
        paired["pathway"] = pathway
        paired["pathway_label"] = LABELS.get(pathway, pathway)
        paired["prediction_shift"] = paired["full_prediction"] - paired["loo_prediction"]
        rows.append(paired)
    if not rows:
        return pd.DataFrame(columns=["split_seed", "sample_id", "full_prediction", "loo_prediction", "pathway", "pathway_label", "prediction_shift"])
    return pd.concat(rows, ignore_index=True)

def benjamini_hochberg(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(values[idx], kind="mergesort")]
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out

def _metric(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "r2": float(r2_score(y_true, prediction)) if len(y_true) > 1 else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
    }

def _compound_key(frame: pd.DataFrame) -> pd.Series:
    if "canonical_smiles" in frame:
        return frame["canonical_smiles"].astype(str)
    return frame["compound"].astype(str)

def compound_bootstrap_metric_delta(
    frame: pd.DataFrame,
    full_column: str,
    loo_column: str,
    n_boot: int = 1000,
    random_state: int = 20260903,
) -> pd.DataFrame:
    required = {full_column, loo_column, "y_true"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame missing columns: {sorted(missing)}")
    work = frame.copy()
    work["_compound"] = _compound_key(work)
    groups = [group for _, group in work.groupby("_compound", sort=True)]
    if not groups:
        return pd.DataFrame(columns=["delta_r2", "delta_rmse", "n_compounds"])
                                                                                  
    rng = np.random.default_rng(int(random_state))
    rows = []
    for _ in range(int(n_boot)):
        sampled = [groups[int(i)] for i in rng.integers(0, len(groups), size=len(groups))]
        draw = pd.concat(sampled, ignore_index=True)
        full = _metric(draw["y_true"], draw[full_column])
        loo = _metric(draw["y_true"], draw[loo_column])
        rows.append({"delta_r2": full["r2"] - loo["r2"], "delta_rmse": loo["rmse"] - full["rmse"], "n_compounds": len(groups)})
    return pd.DataFrame(rows)

def _fixed_estimator(seed: int) -> ExtraTreesRegressor:
    feature_fraction = 0.70
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=feature_fraction,
        random_state=20260605 + int(seed),
        n_jobs=-1,
    )

def _base_columns(feature_columns: list[str]) -> list[str]:
    prefixes = ("rdkit_", "maccs_", "tt_", "morgan_", "atom_", "domain_", "fg_")
    selected = [col for col in feature_columns if col.startswith(prefixes)]
    time = {"time_h_numeric", "time_h_log1p", *[f"time_h_{x}" for x in (3, 6, 12, 24, 48, 72)]}
    selected.extend(col for col in feature_columns if col in time)
    return list(dict.fromkeys(selected))

@lru_cache(maxsize=1)
def _pathway_bundle(root: str):
    from nci_cho_transfer.kegg import build_pathway_bundle

    return build_pathway_bundle(Path(root))

def _ensure_pathway_columns(root: Path, bundle, frame: pd.DataFrame, pathways: list[str]) -> pd.DataFrame:
    required = [f"pathway_transfer_{pathway}" for pathway in pathways]
    missing = [column for column in required if column not in frame.columns]
    if not missing:
        return frame
    from nci_cho_transfer.mechanistic_analysis import build_pathway_transfer_features

    pathway_bundle = _pathway_bundle(str(root.resolve()))
    extra = build_pathway_transfer_features(
        frame[["sample_id", "canonical_smiles", "time_h"]],
        bundle.nci_source_rows,
        pathway_bundle.pathway_activity,
        top_pathways=[column.removeprefix("pathway_transfer_") for column in missing],
    )
    return frame.merge(extra, on="sample_id", how="left")

def _fit_seed(root: Path, seed: int, pathways: list[str]) -> pd.DataFrame:
    from nci_cho_transfer.data_contract import load_data_contract
    from nci_cho_transfer.features import build_model_matrix
    from nci_cho_transfer.splits import build_frozen_split

    bundle = load_data_contract(root)
    split = build_frozen_split(bundle, split_seed=int(seed))
    matrix = build_model_matrix(bundle, split, use_cache=True)
    frame = _ensure_pathway_columns(root, bundle, matrix.frame.copy(), pathways)
    base = _base_columns(matrix.feature_columns)
    path_cols = [f"pathway_transfer_{p}" for p in pathways]
    path_cols = [col for col in path_cols if col in frame.columns]
    if not path_cols:
        raise ValueError("No pathway transfer columns found in model matrix")
    frame["_role"] = frame["split_role"].replace({"train": "random_cv_oof"})
    frame["_row_id"] = np.arange(len(frame))
    train = frame[frame["split_role"].eq("train")].copy().reset_index(drop=True)
    query_roles = {"random_cv_oof": train, "test": frame[frame["split_role"].eq("test")], "validation": frame[frame["split_role"].eq("validation")], "experimental": frame[frame["split_role"].eq("experimental")]}
    model_columns = {
        "full": base + path_cols,
        **{p: base + [col for col in path_cols if col != f"pathway_transfer_{p}"] for p in pathways},
        "base": base,
    }
    predictions: list[pd.DataFrame] = []
    for model_name, columns in model_columns.items():
        train_x = train[columns].to_numpy(dtype=float)
        train_y = train["y"].to_numpy(dtype=float)
        oof = np.full(len(train), np.nan, dtype=float)
        splitter = GroupKFold(n_splits=5)
        groups = train["canonical_smiles"].astype(str).to_numpy()
        for fit_idx, valid_idx in splitter.split(train_x, train_y, groups):
            fold_estimator = _fixed_estimator(seed)
            fold_estimator.fit(train_x[fit_idx], train_y[fit_idx])
            oof[valid_idx] = fold_estimator.predict(train_x[valid_idx])
        predictions.append(pd.DataFrame({
            "split_seed": int(seed), "sample_id": train["sample_id"].astype(str).to_numpy(),
            "split_role": "random_cv_oof", "canonical_smiles": train["canonical_smiles"].astype(str).to_numpy(),
            "time_h": train["time_h"].to_numpy(dtype=float), "y_true": train_y,
            "model": model_name, "pathway": model_name if model_name in pathways else "ALL",
            "prediction": oof,
        }))
        estimator = _fixed_estimator(seed)
        estimator.fit(train_x, train_y)
        for role in ("test", "validation", "experimental"):
            query = query_roles[role]
            if query.empty:
                continue
            pred = estimator.predict(query[columns].to_numpy(dtype=float))
            predictions.append(pd.DataFrame({
                "split_seed": int(seed), "sample_id": query["sample_id"].astype(str).to_numpy(),
                "split_role": role, "canonical_smiles": query["canonical_smiles"].astype(str).to_numpy(),
                "time_h": query["time_h"].to_numpy(dtype=float), "y_true": query["y"].to_numpy(dtype=float),
                "model": model_name, "pathway": model_name if model_name in pathways else "ALL",
                "prediction": pred,
            }))
    return pd.concat(predictions, ignore_index=True)

def _aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, model), group in predictions.groupby(["split_role", "model"], sort=True):
        metrics = _metric(group["y_true"].to_numpy(), group["prediction"].to_numpy())
        rows.append({"split_role": role, "model": model, "n": len(group), **metrics})
    return pd.DataFrame(rows)

def _build_leave_one_out(predictions: pd.DataFrame, pathways: list[str], n_boot: int) -> pd.DataFrame:
    rows = []
    full = predictions[predictions["model"].eq("full")]
    for role in ROLES:
        full_role = full[full["split_role"].eq(role)]
        if full_role.empty:
            continue
        full_metrics = _metric(full_role["y_true"], full_role["prediction"])
        for pathway in pathways:
            loo = predictions[(predictions["split_role"].eq(role)) & (predictions["model"].eq(pathway))]
            if loo.empty:
                continue
            loo_metrics = _metric(loo["y_true"], loo["prediction"])
            delta = pathway_delta(full_metrics, loo_metrics)
            paired = full_role[["split_seed", "sample_id", "canonical_smiles", "time_h", "y_true", "prediction"]].rename(columns={"prediction": "full"}).merge(
                loo[["split_seed", "sample_id", "prediction"]].rename(columns={"prediction": "loo"}), on=["split_seed", "sample_id"], how="inner"
            )
            boot = compound_bootstrap_metric_delta(paired, "full", "loo", n_boot=n_boot, random_state=20260903 + len(rows))
            p_r2 = float(2 * min((boot["delta_r2"] <= 0).mean(), (boot["delta_r2"] >= 0).mean())) if len(boot) else float("nan")
            p_rmse = float(2 * min((boot["delta_rmse"] <= 0).mean(), (boot["delta_rmse"] >= 0).mean())) if len(boot) else float("nan")
            rows.append({"split_role": role, "pathway": pathway, "pathway_label": LABELS.get(pathway, pathway), "n": len(full_role), "full_r2": full_metrics["r2"], "loo_r2": loo_metrics["r2"], "delta_r2": delta["delta_r2"], "delta_r2_ci_low": float(boot["delta_r2"].quantile(0.025)), "delta_r2_ci_high": float(boot["delta_r2"].quantile(0.975)), "full_rmse": full_metrics["rmse"], "loo_rmse": loo_metrics["rmse"], "delta_rmse": delta["delta_rmse"], "delta_rmse_ci_low": float(boot["delta_rmse"].quantile(0.025)), "delta_rmse_ci_high": float(boot["delta_rmse"].quantile(0.975)), "bootstrap_p_r2": p_r2, "bootstrap_p_rmse": p_rmse, "n_compounds": int(paired["canonical_smiles"].nunique())})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_r2"] = np.nan
        result["fdr_rmse"] = np.nan
        for role, idx in result.groupby("split_role").groups.items():
            result.loc[idx, "fdr_r2"] = benjamini_hochberg(result.loc[idx, "bootstrap_p_r2"].to_numpy())
            result.loc[idx, "fdr_rmse"] = benjamini_hochberg(result.loc[idx, "bootstrap_p_rmse"].to_numpy())
    return result

def _build_interactions(predictions: pd.DataFrame, pathways: list[str]) -> pd.DataFrame:
    rows = []
    base = predictions[predictions["model"].eq("base")]
    full = predictions[predictions["model"].eq("full")]
    for role in ROLES:
        b = base[base["split_role"].eq(role)]
        f = full[full["split_role"].eq(role)]
        if b.empty or f.empty:
            continue
        base_r2 = _metric(b.y_true, b.prediction)["r2"]
        full_r2 = _metric(f.y_true, f.prediction)["r2"]
        for pathway in pathways:
            p = predictions[(predictions["model"].eq(pathway)) & (predictions["split_role"].eq(role))]
            if p.empty:
                continue
            path_r2 = _metric(p.y_true, p.prediction)["r2"]
            rows.append({"split_role": role, "pathway": pathway, "pathway_label": LABELS.get(pathway, pathway), "base_r2": base_r2, "pathway_r2": path_r2, "full_r2": full_r2, "structure_pathway_increment": path_r2 - base_r2, "joint_increment": full_r2 - base_r2})
    return pd.DataFrame(rows)

def _shuffle_control(root: Path, seed: int, pathways: list[str]) -> pd.DataFrame:
    from nci_cho_transfer.data_contract import load_data_contract
    from nci_cho_transfer.features import build_model_matrix
    from nci_cho_transfer.splits import build_frozen_split

    bundle = load_data_contract(root)
    split = build_frozen_split(bundle, split_seed=int(seed))
    matrix = build_model_matrix(bundle, split, use_cache=True)
    frame = _ensure_pathway_columns(root, bundle, matrix.frame.copy(), pathways)
    base = _base_columns(matrix.feature_columns)
    path_cols = [f"pathway_transfer_{p}" for p in pathways if f"pathway_transfer_{p}" in frame.columns]
    train = frame[frame.split_role.eq("train")].copy()
    rng = np.random.default_rng(81000 + int(seed))
    rows = []
    for pathway_col in path_cols:
        shuffled = frame.copy()
        shuffled[pathway_col] = rng.permutation(shuffled[pathway_col].to_numpy())
        est = _fixed_estimator(seed)
        est.fit(train[base + path_cols].assign(**{pathway_col: shuffled.loc[train.index, pathway_col]})[base + path_cols], train.y)
        for role in ("test", "validation", "experimental"):
            q = shuffled[shuffled.split_role.eq(role)]
            if q.empty:
                continue
            rows.append({"split_seed": seed, "split_role": role, "pathway": pathway_col.removeprefix("pathway_transfer_"), "pathway_label": LABELS.get(pathway_col.removeprefix("pathway_transfer_"), pathway_col), "r2": _metric(q.y, est.predict(q[base + path_cols]))["r2"]})
    return pd.DataFrame(rows)

def _save_figure(fig: plt.Figure, output_path: Path, dpi: int = 1200) -> None:
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

def _plot_metric_panel(test: pd.DataFrame, metric: str, output_path: Path) -> None:
    lower = f"{metric}_ci_low"
    upper = f"{metric}_ci_high"
    ordered = test.assign(_abs=test[metric].abs()).sort_values("_abs", ascending=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.6), constrained_layout=True)
    values = ordered[metric].to_numpy(dtype=float)
    errors = np.vstack([values - ordered[lower].to_numpy(dtype=float), ordered[upper].to_numpy(dtype=float) - values])
    colors = ["#c44e52" if value < 0 else "#2f7fb8" for value in values]
    ax.barh(ordered.pathway_label, values, xerr=errors, capsize=2, color=colors, error_kw={"linewidth": 0.7})
    ax.axvline(0, color="#333333", lw=0.8)
    label = "Delta R2 (full - leave-one-pathway-out)" if metric == "delta_r2" else "Delta RMSE (leave-one-pathway-out - full)"
    ax.set_xlabel(label)
    ax.set_ylabel("KEGG pathway")
    ax.set_title("Conditional incremental contribution", fontsize=9, fontweight="bold")
    _save_figure(fig, output_path)
    plt.close(fig)

def plot_incremental_contribution(loo: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> None:
    test = loo[loo.split_role.eq("test")].copy()
    shifts = pathway_prediction_shifts(predictions, list(test.pathway))
    if test.empty or shifts.empty:
        return
    summary = shifts.groupby(["pathway", "pathway_label"], sort=False)["prediction_shift"].agg(
        mean_abs=lambda values: float(np.mean(np.abs(values))),
        median=lambda values: float(np.median(values)),
    ).reset_index()
    summary = summary.sort_values("mean_abs", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.8), gridspec_kw={"width_ratios": [0.95, 1.35]}, constrained_layout=True)
    axes[0].barh(summary.pathway_label, summary.mean_abs, color="#3d7594")
    for y, value in enumerate(summary.mean_abs):
        axes[0].text(value, y, f" {value:.3f}", va="center", ha="left", fontsize=7)
    axes[0].set_xlabel("Mean |prediction shift|")
    axes[0].set_ylabel("KEGG pathway")
    axes[0].set_title("Mean absolute contribution", fontsize=9, fontweight="bold")

    rng = np.random.default_rng(20260904)
    order = summary.pathway_label.tolist()
    for row, label in enumerate(order):
        values = shifts.loc[shifts.pathway_label.eq(label), "prediction_shift"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        colors = np.where(values >= 0, "#c44e52", "#2f7fb8")
        axes[1].scatter(values, row + jitter, s=9, alpha=0.55, c=colors, linewidths=0, rasterized=True)
    axes[1].axvline(0, color="#333333", lw=0.8)
    axes[1].set_yticks(range(len(order)), order)
    axes[1].set_xlabel("Prediction shift (full - leave-one-pathway-out)")
    axes[1].set_ylabel("")
    axes[1].set_title("Direction of pathway effects", fontsize=9, fontweight="bold")
    _save_figure(fig, output_path)
    plt.close(fig)

    _plot_metric_panel(test, "delta_r2", output_path.with_name("pathway_incremental_delta_r2.png"))
    _plot_metric_panel(test, "delta_rmse", output_path.with_name("pathway_incremental_delta_rmse.png"))

def run(root: Path, output_dir: Path, seeds: tuple[int, ...], n_boot: int = 500) -> Path:
    pathways = list(PATHWAYS)
    predictions = pd.concat([_fit_seed(root, seed, pathways) for seed in seeds], ignore_index=True)
    loo = _build_leave_one_out(predictions, pathways, n_boot=n_boot)
    interactions = _build_interactions(predictions, pathways)
    shuffled = pd.concat([_shuffle_control(root, seed, pathways) for seed in seeds], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "pathway_predictions.csv", index=False)
    _aggregate_metrics(predictions).to_csv(output_dir / "pathway_metrics.csv", index=False)
    loo.to_csv(output_dir / "pathway_leave_one_out.csv", index=False)
    interactions.to_csv(output_dir / "pathway_interactions.csv", index=False)
    shuffled.to_csv(output_dir / "pathway_shuffle_control.csv", index=False)
    plot_incremental_contribution(loo, predictions, output_dir / "pathway_incremental_contribution.png")
    summary = {
        "run_id": output_dir.name,
        "analysis": "KEGG pathway-specific conditional incremental predictive contribution",
        "causal_claim_boundary": "Observational model ablation; not biological causality.",
        "root_dir": "01_models",
        "seeds": list(seeds),
        "pathways": pathways,
        "same_frozen_split_per_seed": True,
        "compound_level_bootstrap": True,
        "leave_one_out_definition": "full model minus model with exactly one pathway_transfer column removed",
        "shuffle_negative_control": True,
        "primary_role": "test",
        "experimental_role": "descriptive_only",
        "outputs": [
            "pathway_leave_one_out.csv",
            "pathway_interactions.csv",
            "pathway_shuffle_control.csv",
            "pathway_incremental_contribution.png",
            "pathway_incremental_delta_r2.png",
            "pathway_incremental_delta_rmse.png",
        ],
    }
    (output_dir / "pathway_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir

def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-seed KEGG pathway leave-one-out incremental analysis")
    parser.add_argument("--root-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed-list", default=",".join(map(str, SEEDS)))
    parser.add_argument("--n-bootstrap", type=int, default=500)
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seed_list.split(",") if value.strip())
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"seed-list must be a non-empty subset of {SEEDS}")
    root = Path(args.root_dir).resolve()
    destination = Path(args.output_dir).resolve() if args.output_dir else root / "data_processed" / "components" / "kegg_pathway_incremental_20260903"
    print(run(root, destination, seeds, n_boot=args.n_bootstrap))

if __name__ == "__main__":
    main()

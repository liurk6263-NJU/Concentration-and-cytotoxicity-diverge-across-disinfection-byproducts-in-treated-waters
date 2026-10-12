

from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path(__file__).resolve().parents[2]
DBP_SOURCE = MODEL_ROOT / "04_screening/results/dbp_ad_expected_error_predictions.csv"
NSC_SOURCE = MODEL_ROOT / "04_screening/results/nsc_suspected_analogues_with_applicability_domain.csv"
RAW_CATALOG_SOURCE = MODEL_ROOT / "01_models/data_raw/dbp_raw_smiles_exact_dedup.csv"
CONFIRMED_SOURCE = MODEL_ROOT / "04_screening/data_raw/716DBPs.xlsx"
CONFIRMED_PAPER = "Confirmed DBP catalogue reference"
OUT_DIR = HERE.parent / "results"
EDITABLE_DIR = HERE.parent
QA_DIR = HERE.parent / "results"
PNG_OUT = HERE.parent / "Figure5c.png"
SVG_OUT = HERE.parent / "Figure5c.svg"
CSV_OUT = OUT_DIR / "embedding.csv"
MANIFEST_OUT = OUT_DIR / "embedding_parameters.json"
VALIDATION_OUT = OUT_DIR / "embedding_check.json"

FIG_WIDTH_MM = 210.0
FIG_HEIGHT_MM = FIG_WIDTH_MM * 8.2 / 14.5

CONFIRMED = "Standard-confirmed DBPs"
RESEARCH = "Research-reported DBPs"
LIKE = "DBPs-like"
CATEGORIES = [CONFIRMED, RESEARCH, LIKE]
MARKERS = {CONFIRMED: "o", RESEARCH: "^", LIKE: "D"}
AD_CUTOFF = 0.333

RDLogger.DisableLog("rdApp.*")

def canonical_smiles(smiles: object) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def load_confirmed_reference() -> tuple[set[str], dict[str, int]]:

    table = pd.read_excel(CONFIRMED_SOURCE, sheet_name="TableS1_716 Confirmed DBPs", header=None)
    header_row = table.iloc[1].tolist()
    confirmed = table.iloc[2:].copy()
    confirmed.columns = header_row
    smiles_column = header_row[6]
    confirmed["canonical_key"] = confirmed[smiles_column].map(canonical_smiles)
    valid = confirmed["canonical_key"].dropna()
    raw_catalog = pd.read_csv(RAW_CATALOG_SOURCE)
    raw_catalog["canonical_smiles_key"] = raw_catalog["dbp_raw_smiles"].map(canonical_smiles)
    raw_keys = set(raw_catalog["canonical_smiles_key"].dropna().astype(str))
    valid_in_catalog = valid.astype(str).isin(raw_keys)
    stats = {
        "reference_rows": int(len(confirmed)),
        "reference_rows_with_smiles": int(valid.shape[0]),
        "reference_unique_structure_keys": int(valid.nunique()),
        "reference_rows_with_smiles_in_raw_catalog": int(valid_in_catalog.sum()),
        "reference_unique_structure_keys_in_raw_catalog": int(valid[valid_in_catalog].nunique()),
        "raw_catalog_rows": int(len(raw_catalog)),
        "raw_catalog_connectivity_keys": int(raw_catalog["exact_molecule_key"].nunique()),
        "raw_catalog_canonical_smiles_keys": int(raw_catalog["canonical_smiles_key"].dropna().nunique()),
    }
    if stats["reference_rows"] != 716:
        raise ValueError(f"Confirmed DBP reference must contain 716 rows, found {stats['reference_rows']}")
    return set(valid.astype(str)), stats

def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

def load_records() -> pd.DataFrame:
    dbp = pd.read_csv(DBP_SOURCE)
    nsc = pd.read_csv(NSC_SOURCE)
    confirmed_keys, confirmed_stats = load_confirmed_reference()
    required_dbp = {"SMILES", "DBP_ID", "pLC50_mean", "pLC50_sd", "max_tanimoto", "composite_ad", "in_ocsr", "in_excel"}
    required_nsc = {"canonical_nsc_smiles", "query_id", "pLC50_mean", "pLC50_sd", "max_tanimoto", "composite_ad"}
    if not required_dbp.issubset(dbp.columns):
        raise ValueError(f"Missing DBP columns: {sorted(required_dbp - set(dbp.columns))}")
    if not required_nsc.issubset(nsc.columns):
        raise ValueError(f"Missing DBPs-like columns: {sorted(required_nsc - set(nsc.columns))}")

    dbp["in_excel"] = pd.to_numeric(dbp["in_excel"], errors="coerce").fillna(0).astype(int)
    dbp["in_ocsr"] = pd.to_numeric(dbp["in_ocsr"], errors="coerce").fillna(0).astype(int)
    dbp["confirmed716"] = dbp["SMILES"].map(canonical_smiles).isin(confirmed_keys)
    dbp["category"] = np.where(dbp["confirmed716"], CONFIRMED, RESEARCH)
    dbp_rows = pd.DataFrame(
        {
            "record_id": dbp["DBP_ID"].astype(str),
            "smiles": dbp["SMILES"].astype(str),
            "category": dbp["category"],
            "confirmed716": dbp["confirmed716"],
            "source_overlap": dbp["in_excel"].eq(1) & dbp["in_ocsr"].eq(1),
            "pLC50_mean": pd.to_numeric(dbp["pLC50_mean"], errors="coerce"),
            "pLC50_sd": pd.to_numeric(dbp["pLC50_sd"], errors="coerce"),
            "max_tanimoto": pd.to_numeric(dbp["max_tanimoto"], errors="coerce"),
            "composite_ad": _bool_series(dbp["composite_ad"]),
            "expected_mae": pd.to_numeric(dbp.get("expected_mae", np.nan), errors="coerce"),
            "source_table": "dbp_ad_expected_error_predictions.csv",
        }
    )
    dbp_keys = set(dbp["exact_molecule_key"].astype(str))
    nsc_rows = pd.DataFrame(
        {
            "record_id": nsc["query_id"].astype(str),
            "smiles": nsc["canonical_nsc_smiles"].astype(str),
            "category": LIKE,
            "source_overlap": nsc["canonical_nsc_smiles"].astype(str).isin(dbp_keys),
            "pLC50_mean": pd.to_numeric(nsc["pLC50_mean"], errors="coerce"),
            "pLC50_sd": pd.to_numeric(nsc["pLC50_sd"], errors="coerce"),
            "max_tanimoto": pd.to_numeric(nsc["max_tanimoto"], errors="coerce"),
            "composite_ad": _bool_series(nsc["composite_ad"]),
            "expected_mae": pd.to_numeric(nsc.get("expected_mae", np.nan), errors="coerce"),
            "source_table": "nsc_suspected_analogues_with_applicability_domain.csv",
        }
    )
    rows = pd.concat([dbp_rows, nsc_rows], ignore_index=True)
    rows.attrs["confirmed_reference_stats"] = confirmed_stats
    rows.attrs["confirmed_matched_prediction_rows"] = int(dbp_rows["confirmed716"].sum())
    numeric = ["pLC50_mean", "pLC50_sd", "max_tanimoto"]
    if rows[numeric].isna().any().any():
        raise ValueError("Prediction, SD or Tanimoto fields contain missing values")
    if rows["record_id"].duplicated().any():
        raise ValueError("Record identifiers are not unique across source tables")
    if rows["smiles"].duplicated().sum() == 0:
        raise ValueError("Expected source overlap between DBP and DBPs-like structures was not found")
    return rows

def fingerprints(rows: pd.DataFrame) -> tuple[list[Chem.Mol], np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols: list[Chem.Mol] = []
    matrix = np.zeros((len(rows), 2048), dtype=np.uint8)
    bad: list[str] = []
    for idx, (record_id, smiles) in enumerate(rows[["record_id", "smiles"]].itertuples(index=False, name=None)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bad.append(str(record_id))
            continue
        mols.append(mol)
        bit_vector = generator.GetFingerprint(mol)
        bit_vector.ToBitString()
        matrix[idx, :] = np.fromiter((int(bit) for bit in bit_vector.ToBitString()), dtype=np.uint8, count=2048)
    if bad:
        raise ValueError(f"SMILES parsing failed for {len(bad)} rows: {bad[:5]}")
    return mols, matrix

def embed(matrix: np.ndarray) -> np.ndarray:
                                                                                             
    n_components = min(50, matrix.shape[1] - 1, matrix.shape[0] - 1)
    compressed = TruncatedSVD(n_components=n_components, random_state=17).fit_transform(matrix)
    perplexity = min(40.0, max(15.0, (len(matrix) - 1) / 100.0))
    coordinates = TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=1000,
        random_state=17,
    ).fit_transform(compressed)
    coordinates = (coordinates - coordinates.mean(axis=0)) / coordinates.std(axis=0)
    return coordinates

def add_confidence_fields(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    sd_q95 = float(out["pLC50_sd"].quantile(0.95))
    sd_min = float(out["pLC50_sd"].min())
    denominator = max(sd_q95 - sd_min, 1e-9)
    stability = 1.0 - np.clip((out["pLC50_sd"] - sd_min) / denominator, 0.0, 1.0)
    tanimoto_support = np.clip(out["max_tanimoto"] / AD_CUTOFF, 0.0, 1.0)
    ad_support = np.where(out["composite_ad"], 1.0, 0.45 * tanimoto_support)
    out["relative_support_index"] = 0.55 * ad_support + 0.45 * stability
    out["plot_alpha"] = 0.22 + 0.72 * out["relative_support_index"]
    p_rank = out["pLC50_mean"].rank(pct=True)
    out["priority_score"] = 0.60 * p_rank + 0.40 * out["relative_support_index"]
    return out

def select_priority(rows: pd.DataFrame) -> list[int]:
    chosen: list[int] = []
    for category in CATEGORIES:
        block = rows[rows["category"].eq(category)].copy()
        eligible = block[block["composite_ad"]]
        if len(eligible) < 2:
            eligible = block
        chosen.extend(eligible.sort_values("priority_score", ascending=False).head(2).index.tolist())
    return chosen

def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

def draw(rows: pd.DataFrame, mols: list[Chem.Mol], priority: list[int]) -> None:
    configure_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4), dpi=220, facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.55, 1.78], wspace=0.16, left=0.055, right=0.985, top=0.90, bottom=0.10)
    ax = fig.add_subplot(grid[0, 0])
    side = fig.add_subplot(grid[0, 1])
    side.axis("off")

    cmap = mpl.colormaps.get_cmap("viridis")
    norm = Normalize(vmin=float(rows["pLC50_mean"].min()), vmax=float(rows["pLC50_mean"].max()))
    rows["x"] = rows["x"].to_numpy(float)
    rows["y"] = rows["y"].to_numpy(float)
    for category in CATEGORIES:
        mask = rows["category"].eq(category).to_numpy()
        colours = cmap(norm(rows.loc[mask, "pLC50_mean"].to_numpy(float)))
        ax.scatter(
            rows.loc[mask, "x"], rows.loc[mask, "y"],
            s=9 + 13 * rows.loc[mask, "relative_support_index"],
            c=colours, alpha=rows.loc[mask, "plot_alpha"], marker=MARKERS[category],
            edgecolors="#30343b", linewidths=0.22, rasterized=True, zorder=2,
        )

    label_offsets = {1: (-6, -28), 2: (28, 22), 3: (10, 10), 4: (-18, 29), 5: (-31, -17), 6: (29, -16)}
    for number, idx in enumerate(priority, 1):
        row = rows.loc[idx]
        ax.scatter(row["x"], row["y"], s=150, facecolors="none", edgecolors="#a51c30", linewidths=1.3, zorder=4)
        ax.annotate(str(number), (row["x"], row["y"]), xytext=label_offsets[number], textcoords="offset points",
                    fontsize=9.5, weight="bold", color="#8c1729", ha="center", va="center", zorder=5,
                    bbox={"boxstyle": "circle,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.95},
                    arrowprops={"arrowstyle": "-", "color": "#a51c30", "lw": 0.8})

    ax.set_xlabel("Chemical embedding 1", labelpad=8, fontsize=11.5)
    ax.set_ylabel("Chemical embedding", labelpad=8, fontsize=11.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(color="#d8dee3", linewidth=0.45, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color("#6d737a")
        spine.set_linewidth(0.8)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Predicted pLC50", labelpad=8, fontsize=11.0)
    cbar.ax.tick_params(labelsize=8.5, length=3)

    handles = [
        Line2D([0], [0], marker=MARKERS[CONFIRMED], color="w", markerfacecolor="#4c9f70", markeredgecolor="#30343b", markersize=8.0, label=f"{CONFIRMED} (source n={rows.attrs['confirmed_reference_stats']['reference_rows']})"),
        Line2D([0], [0], marker=MARKERS[RESEARCH], color="w", markerfacecolor="#4c9f70", markeredgecolor="#30343b", markersize=8.8, label=f"{RESEARCH} (n={int(rows['category'].eq(RESEARCH).sum())})"),
        Line2D([0], [0], marker=MARKERS[LIKE], color="w", markerfacecolor="#4c9f70", markeredgecolor="#30343b", markersize=8.6, label=f"{LIKE} (n={int(rows['category'].eq(LIKE).sum())})"),
        Line2D([0], [0], marker="o", color="#a51c30", markerfacecolor="none", markersize=10.0, label="Priority exemplar (outline)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.985), frameon=True, framealpha=0.93,
              facecolor="white", edgecolor="#d3d9de", fontsize=9.0, borderpad=0.75, handletextpad=0.55, labelspacing=0.5)
    side.text(0.00, 0.985, "Priority exemplars", va="top", fontsize=13.5, weight="bold", color="#202a30")
    y0, dy = 0.865, 0.145
    exemplar_names = pd.read_csv(OUT_DIR / 'exemplars.csv').set_index('record_id').compound_name.to_dict()
    for number, idx in enumerate(priority, 1):
        row = rows.loc[idx]
        y = y0 - (number - 1) * dy
        inset = side.inset_axes([0.00, y - 0.078, 0.24, 0.085])
        inset.imshow(Draw.MolToImage(mols[idx], size=(170, 92), kekulize=False))
        inset.axis("off")
        name_lines = wrap(exemplar_names.get(str(row["record_id"]), str(row["smiles"])), width=25,
                          break_long_words=True, break_on_hyphens=True)[:3]
        for line_index, line_text in enumerate(name_lines):
            side.text(0.27, y - 0.003 - line_index * 0.021, f"{number}  {line_text}" if line_index == 0 else line_text,
                      transform=side.transAxes, fontsize=7.9, weight="bold" if line_index == 0 else "normal", color="#26353d", va="center")
        category_y = y - 0.003 - len(name_lines) * 0.021 - 0.003
        side.text(0.27, category_y, row["category"], transform=side.transAxes, fontsize=7.2, color="#4f5961", va="center")
        side.text(0.27, category_y - 0.023, f"pLC50 {row['pLC50_mean']:.2f}  SD {row['pLC50_sd']:.2f}  T {row['max_tanimoto']:.2f}", transform=side.transAxes, fontsize=7.2, color="#26353d", va="center")

    fig.text(0.055, 0.972, "Figure 5", ha="left", va="top", fontsize=17, weight="bold", color="#1d2930")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if any(text.get_window_extent(renderer).x1 > fig.bbox.x1 - 2 for text in side.texts):
        raise ValueError("Priority exemplar text extends beyond the figure width")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EDITABLE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=300, facecolor="white")
    fig.savefig(SVG_OUT, format="svg", facecolor="white")
    plt.close(fig)

def main() -> None:
    rows = add_confidence_fields(load_records())
    mols, matrix = fingerprints(rows)
    coords = embed(matrix)
    rows[["x", "y"]] = coords
    priority = select_priority(rows)
    rows["priority_exemplar"] = False
    rows.loc[priority, "priority_exemplar"] = True
    ordered_columns = ["record_id", "category", "confirmed716", "source_overlap", "smiles", "pLC50_mean", "pLC50_sd", "max_tanimoto", "composite_ad", "expected_mae", "relative_support_index", "plot_alpha", "priority_score", "priority_exemplar", "x", "y", "source_table"]
    rows[ordered_columns].to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    draw(rows, mols, priority)
    manifest = {
        "figure": str(PNG_OUT),
        "editable_svg": str(SVG_OUT),
        "dimensions_mm": {"width": FIG_WIDTH_MM, "height": FIG_HEIGHT_MM},
        "source_tables": [str(DBP_SOURCE), str(NSC_SOURCE)],
        "confirmed_reference_source": str(CONFIRMED_SOURCE),
        "confirmed_reference_paper": str(CONFIRMED_PAPER),
        "mapping": {
            CONFIRMED: "DBP prediction rows whose canonical SMILES matches the 716-entry standard-confirmed DBP supplementary table",
            RESEARCH: "all remaining DBP prediction rows not matched to the confirmed supplementary table",
            LIKE: "all rows from nsc_suspected_analogues_with_applicability_domain.csv",
        },
        "record_count": int(len(rows)),
        "category_counts": {category: int(rows["category"].eq(category).sum()) for category in CATEGORIES},
        "confirmed_reference_count": int(rows.attrs["confirmed_reference_stats"]["reference_rows"]),
        "confirmed_reference_rows_with_smiles": int(rows.attrs["confirmed_reference_stats"]["reference_rows_with_smiles"]),
        "confirmed_reference_unique_structure_keys": int(rows.attrs["confirmed_reference_stats"]["reference_unique_structure_keys"]),
        "confirmed_reference_rows_with_smiles_in_raw_catalog": int(rows.attrs["confirmed_reference_stats"]["reference_rows_with_smiles_in_raw_catalog"]),
        "confirmed_reference_unique_structure_keys_in_raw_catalog": int(rows.attrs["confirmed_reference_stats"]["reference_unique_structure_keys_in_raw_catalog"]),
        "raw_catalog_rows": int(rows.attrs["confirmed_reference_stats"]["raw_catalog_rows"]),
        "raw_catalog_connectivity_keys": int(rows.attrs["confirmed_reference_stats"]["raw_catalog_connectivity_keys"]),
        "raw_catalog_canonical_smiles_keys": int(rows.attrs["confirmed_reference_stats"]["raw_catalog_canonical_smiles_keys"]),
        "confirmed_matched_prediction_rows": int(rows.attrs["confirmed_matched_prediction_rows"]),
        "confirmed_reference_keys_missing_prediction_export": int(rows.attrs["confirmed_reference_stats"]["reference_unique_structure_keys"] - rows.attrs["confirmed_matched_prediction_rows"]),
        "dbp_source_overlap_count": int(rows.loc[rows["source_table"].eq("dbp_ad_expected_error_predictions.csv"), "source_overlap"].sum()),
        "dbps_like_exact_overlap_count": int(rows.loc[rows["category"].eq(LIKE), "source_overlap"].sum()),
        "unique_smiles": int(rows["smiles"].nunique()),
        "missing_prediction_fields": {column: int(rows[column].isna().sum()) for column in ["pLC50_mean", "pLC50_sd", "max_tanimoto"]},
        "smiles_parse_success": int(len(mols)),
        "embedding": "Morgan radius 2, 2048 bits -> TruncatedSVD 50 components -> cosine t-SNE",
        "ad_boundary": AD_CUTOFF,
        "priority_ids": rows.loc[priority, "record_id"].tolist(),
        "confidence_encoding": "opacity combines max Tanimoto support, composite AD and 10-seed pLC50 SD; it is a relative support/stability index, not a calibrated probability",
        "occurrence_encoding": "not encoded because no occurrence field was present in the supplied prediction tables",
        "svg_editability": "text, axes, legend, markers and priority outlines remain native SVG elements; dense point cloud and RDKit structure thumbnails are fixed raster inserts anchored to the figure frame",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = {
        "figure_width_mm": FIG_WIDTH_MM,
        "figure_height_mm": FIG_HEIGHT_MM,
        "record_count": int(len(rows)),
        "category_counts": manifest["category_counts"],
        "confirmed_reference_count": int(rows.attrs["confirmed_reference_stats"]["reference_rows"]),
        "confirmed_matched_prediction_rows": int(rows.attrs["confirmed_matched_prediction_rows"]),
        "confirmed_reference_rows_with_smiles_in_raw_catalog": int(rows.attrs["confirmed_reference_stats"]["reference_rows_with_smiles_in_raw_catalog"]),
        "confirmed_reference_unique_structure_keys_in_raw_catalog": int(rows.attrs["confirmed_reference_stats"]["reference_unique_structure_keys_in_raw_catalog"]),
        "raw_catalog_canonical_smiles_keys": int(rows.attrs["confirmed_reference_stats"]["raw_catalog_canonical_smiles_keys"]),
        "required_text_absent_from_source": ["Chemical embedding 2", "High predicted toxicity + relative support", "SMILES:"],
        "priority_smiles": rows.loc[priority, "smiles"].tolist(),
    }
    VALIDATION_OUT.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

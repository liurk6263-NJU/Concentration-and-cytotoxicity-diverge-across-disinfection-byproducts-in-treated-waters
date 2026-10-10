from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = PACKAGE / "03_mechanistic" / "results"
MODEL_MATRIX_ROOT = PACKAGE / "03_mechanistic" / "data_processed"
MODEL_RUN = PACKAGE / "01_models" / "data_processed" / "components" / "et_s3"
sys.path.insert(0, str(PACKAGE / "01_models" / "code" / "src"))
from nci_cho_transfer.features import ModelMatrix
from nci_cho_transfer.formal_randomcv80_end_to_end import CURRENT_ID
from nci_cho_transfer.models import build_candidates, candidate_by_id, feature_columns_for_spec
SEED = 20260908
FEATURE_LABELS = {
    "rdkit_fr_ketone": "Ketone groups",
    "rdkit_NumAliphaticCarbocycles": "Aliphatic carbocycles",
    "domain_aliphatic_halogen_count": "Aliphatic halogen count",
    "rdkit_fr_C_O_noCOO": "Non-carboxylic C–O groups",
    "rdkit_fr_alkyl_halide": "Alkyl halide groups",
    "domain_halogen_atom_fraction": "Halogen atom fraction",
    "domain_halogen_fraction_mw": "Halogen mass fraction",
    "rdkit_SMR_VSA4": "Refractivity surface area (SMR_VSA4)",
    "time_h_72": "72-h exposure",
    "domain_max_halogen_z": "Maximum halogen atomic number",
    "rdkit_fr_allylic_oxid": "Allylic oxidation motifs",
    "rdkit_BCUT2D_MWHI": "High-mass BCUT descriptor",
    "rdkit_FractionCSP3": "sp³ carbon fraction",
    "rdkit_RingCount": "Ring count",
    "rdkit_AvgIpc": "Average information content",
    "rdkit_PEOE_VSA3": "Charge surface area (PEOE_VSA3)",
    "domain_bromine_iodine_count": "Bromine and iodine count",
}

def latest_oos_predictions() -> pd.DataFrame:
    holdout = pd.read_csv(MODEL_RUN / "formal_holdout_predictions.csv")
    random_cv = pd.read_csv(MODEL_RUN / "formal_random_cv_oof_predictions.csv")
    random_rows = random_cv[["split_seed", "sample_id", "blend_y_pred"]].copy()
    holdout_rows = holdout.loc[
        holdout["split_role"].ne("train"), ["split_seed", "sample_id", "blend_y_pred"]
    ].copy()
    oos = pd.concat([random_rows, holdout_rows], ignore_index=True)
    if oos.duplicated(["split_seed", "sample_id"]).any():
        raise ValueError("Best transfer model OOS table has duplicate seed/sample rows")
    counts = oos.groupby("sample_id").size()
    if len(counts) != 369 or not counts.eq(10).all():
        raise ValueError("Expected exactly 10 OOS predictions for each of 369 samples")
    return oos

def _model_matrix() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(MODEL_MATRIX_ROOT / "full_model_matrix.parquet")
    features = json.loads(
        (MODEL_MATRIX_ROOT / "full_model_matrix_features.json").read_text(encoding="utf-8")
    )
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Saved model matrix is missing {len(missing)} declared features")
    if frame["sample_id"].duplicated().any() or len(frame) != 369:
        raise ValueError("Saved model matrix must contain one row per labeled sample")
    return frame, features

def prepare_panel_a(oos: pd.DataFrame) -> dict[str, float | int]:
    frame, features = _model_matrix()
    spec = candidate_by_id(build_candidates("quick", None, 0), CURRENT_ID)
    candidate_columns = feature_columns_for_spec(ModelMatrix(frame, features), spec)
    interpretable_columns = [column for column in candidate_columns if not column.startswith("tt_")]
    target = (
        oos.groupby("sample_id", as_index=False)
        .agg(et_s3_oos_mean=("blend_y_pred", "mean"), et_s3_oos_sd=("blend_y_pred", "std"))
    )
    data = target.merge(frame, on="sample_id", how="inner", validate="one_to_one")
    x = data[interpretable_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    y = data["et_s3_oos_mean"].to_numpy(float)
    groups = data["canonical_smiles"].astype(str).to_numpy()

    cv_pred = np.zeros(len(data), dtype=float)
    group_cv = GroupKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(group_cv.split(x, y, groups)):
        model = ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=1,
            max_features=0.5,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        model.fit(x[train_idx], y[train_idx])
        cv_pred[valid_idx] = model.predict(x[valid_idx])
    fidelity = {
        "n_samples": int(len(data)),
        "n_unique_structures": int(pd.Series(groups).nunique()),
        "n_features": int(len(interpretable_columns)),
        "group_cv_folds": 5,
        "group_cv_r2": float(r2_score(y, cv_pred)),
        "group_cv_rmse": float(mean_squared_error(y, cv_pred) ** 0.5),
        "group_cv_mae": float(mean_absolute_error(y, cv_pred)),
    }
    if fidelity["group_cv_r2"] < 0.90:
        raise RuntimeError(f"Interpretable surrogate fidelity below 0.90: {fidelity}")

    final_model = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=1,
        max_features=0.5,
        random_state=SEED,
        n_jobs=-1,
    ).fit(x, y)
    shap_values = np.asarray(
        shap.TreeExplainer(final_model).shap_values(x, check_additivity=False), dtype=float
    )
    importance = np.abs(shap_values).mean(axis=0)
    ranked = np.argsort(-importance)
    summary = pd.DataFrame(
        {
            "feature": [interpretable_columns[index] for index in ranked],
            "feature_label": [
                FEATURE_LABELS.get(interpretable_columns[index], interpretable_columns[index])
                for index in ranked
            ],
            "mean_abs_shap": importance[ranked],
            "mean_shap": shap_values[:, ranked].mean(axis=0),
            "rank": np.arange(1, len(ranked) + 1),
        }
    )
    summary.to_csv(SOURCE / "SHAP_feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": data["sample_id"],
            "canonical_smiles": data["canonical_smiles"],
            "et_s3_oos_mean": y,
            "surrogate_groupcv_prediction": cv_pred,
        }
    ).to_csv(SOURCE / "SHAP_surrogate_fidelity.csv", index=False)

    long_parts = []
    for feature_index in ranked[:10]:
        values = x[:, feature_index]
        percentile = stats.rankdata(values, method="average") / len(values)
        long_parts.append(
            pd.DataFrame(
                {
                    "sample_id": data["sample_id"].astype(str),
                    "feature": interpretable_columns[feature_index],
                    "feature_label": FEATURE_LABELS.get(
                        interpretable_columns[feature_index], interpretable_columns[feature_index]
                    ),
                    "feature_value": values,
                    "feature_percentile": percentile,
                    "shap_value": shap_values[:, feature_index],
                    "mean_abs_shap": importance[feature_index],
                }
            )
        )
    pd.concat(long_parts, ignore_index=True).to_csv(
        SOURCE / "SHAP_specific_global_shap.csv", index=False
    )
    return fidelity

if __name__ == "__main__":
    SOURCE.mkdir(parents=True, exist_ok=True)
    print(prepare_panel_a(latest_oos_predictions()))

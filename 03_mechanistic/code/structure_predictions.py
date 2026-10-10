from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2] / "01_models"
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nci_cho_transfer.data_contract import load_data_contract
from nci_cho_transfer.features import ModelMatrix, build_model_matrix
from nci_cho_transfer.formal_randomcv80_end_to_end import CURRENT_ID
from nci_cho_transfer.models import build_candidates, candidate_by_id, fit_selected_and_evaluate
from nci_cho_transfer.splits import build_frozen_split

from structure_features import FEATURES, feature_values

warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")

SEEDS = (20260021, 20260106, 20260290, 20260307, 20260605, 20260730, 20260738, 20260755, 20260998, 20262000)
QUERIES = (
    ("cf_ketone_1", "2-butanone", "CCC(C)=O"),
    ("cf_ketone_2", "2,3-butanedione", "CC(=O)C(C)=O"),
    ("cf_ring_1", "tetralin", "c1ccc2c(c1)CCCC2"),
    ("cf_ring_2", "1,2,3,4,5,6,7,8-octahydroanthracene", "c1c2c(cc3c1CCCC3)CCCC2"),
    ("cf_noncarboxylic_co_1", "benzaldehyde", "O=Cc1ccccc1"),
    ("cf_noncarboxylic_co_2", "terephthaldehyde", "O=Cc1ccc(C=O)cc1"),
)

def query_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id, name, smiles in QUERIES:
        row = {column: np.nan for column in template.columns}
        row.update(
            {
                "source_row_number": -1,
                "name": name,
                "cas": "COUNTERFACTUAL",
                "raw_smiles": smiles,
                "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles)),
                "inchi_key": "COUNTERFACTUAL",
                "y": 0.0,
                "actual_lc50_m": 1.0,
                "time_h": 72,
                "source_domain": "counterfactual_prediction",
                "doi": "ET-S3_PREDICTION",
                "title_en": "controlled aliphatic-carbocycle counterfactual",
                "title_cn": "controlled aliphatic-carbocycle counterfactual",
                "sample_id": sample_id,
                "canonical_time_key": f"{Chem.MolToSmiles(Chem.MolFromSmiles(smiles))}|72",
                "split_role": "experimental",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=template.columns)

def main() -> None:
    bundle = load_data_contract(ROOT)
    base_split = build_frozen_split(bundle, split_seed=20260605)
    queries = query_rows(base_split.experimental)
    custom_split = replace(base_split, experimental=queries)
    print(f"Building ET-S3 feature matrix for {len(QUERIES)} controlled 72 h counterfactuals...")
    matrix = build_model_matrix(bundle, custom_split, use_cache=True)
    spec = candidate_by_id(build_candidates("quick", None, 0), CURRENT_ID)

    audit_rows = []
    for sample_id, name, smiles in QUERIES:
        values = feature_values(Chem.MolFromSmiles(smiles))
        audit_rows.append({"sample_id": sample_id, "name": name, "smiles": smiles, **values})
    audit = pd.DataFrame(audit_rows)
    print("\nDescriptor audit (top six):")
    print(audit[["name", "formula", *FEATURES]].to_string(index=False))

    all_predictions = []
    for seed in SEEDS:
        split = build_frozen_split(bundle, split_seed=seed)
        role_by_id = {}
        for role in ("train", "test", "validation"):
            for sample_id in getattr(split, role)["sample_id"].astype(str):
                role_by_id[sample_id] = role
        frame = matrix.frame.copy()
        frame["split_role"] = frame["sample_id"].astype(str).map(role_by_id).fillna("experimental")
        run_matrix = ModelMatrix(frame=frame, feature_columns=matrix.feature_columns)
        _, predictions = fit_selected_and_evaluate(run_matrix, spec)
        selected = predictions[predictions["sample_id"].isin([q[0] for q in QUERIES])].copy()
        selected["split_seed"] = seed
        all_predictions.append(selected)
        print(f"Finished seed {seed}")

    result = pd.concat(all_predictions, ignore_index=True)
    result.to_csv(HERE.parent / "results" / "structure_predictions_by_seed.csv", index=False, encoding="utf-8-sig")
    summary = (
        result.groupby(["sample_id", "canonical_smiles", "time_h"], as_index=False)["y_pred"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(HERE.parent / "results" / "structure_prediction_summary.csv", index=False, encoding="utf-8-sig")
    print("\nET-S3 prediction summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

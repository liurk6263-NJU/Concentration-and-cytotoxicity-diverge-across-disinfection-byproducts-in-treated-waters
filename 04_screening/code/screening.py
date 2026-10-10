from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent


def screening():
    raw = pd.read_csv(PACKAGE / '01_models/data_raw/dbp_raw_smiles_exact_dedup.csv')
    unique = pd.read_csv(ROOT / 'results/dbp_ad_expected_error_predictions.csv')
    unique['canonical_smiles'] = unique.SMILES
    columns = ['exact_molecule_key', 'canonical_smiles', 'DBP_ID', 'pLC50_mean', 'pLC50_sd', 'pLC50_n', 'max_tanimoto']
    records = raw.merge(unique[columns], on='exact_molecule_key', validate='many_to_one')
    assert len(records) == 3104 and records.exact_molecule_key.nunique() == 2795
    suspected = pd.read_csv(ROOT / 'results/nsc_suspected_analogues_with_applicability_domain.csv')
    summaries = []
    for label, frame in [('DBP records', records), ('DBP structures', unique), ('DBP-like structures', suspected)]:
        summaries.append({'category': label, 'n': len(frame), 'minimum_plc50': frame.pLC50_mean.min(), 'maximum_plc50': frame.pLC50_mean.max(), 'ge_4': int(frame.pLC50_mean.ge(4).sum()), 'ge_4_5': int(frame.pLC50_mean.ge(4.5).sum())})
    return records, pd.DataFrame(summaries)


if __name__ == '__main__':
    records, summary = screening()
    records.to_csv(ROOT / 'results/dbp_screening_records.csv', index=False)
    summary.to_csv(ROOT / 'results/screening_summary.csv', index=False)
    print(summary.to_string(index=False))

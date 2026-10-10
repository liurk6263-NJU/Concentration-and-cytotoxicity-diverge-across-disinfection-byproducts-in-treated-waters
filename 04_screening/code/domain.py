from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
from domain_helpers import _oos_rows_with_similarity, _aggregate_labeled_oos

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
MODEL = PACKAGE / '01_models'
sys.path.insert(0, str(MODEL / 'code/src'))
from nci_cho_transfer.ad_cell_transfer_analysis import build_expected_error_lookup, composite_ad_mask


def domain():
    comp = MODEL / 'data_processed/components'
    cv = pd.read_csv(comp / 'et_s3/formal_random_cv_oof_predictions.csv').assign(split_role='random_cv_oof')
    hold = pd.read_csv(comp / 'et_s3/formal_holdout_predictions.csv')
    s2 = pd.read_csv(comp / 'et_s2/et_s2_no_transfer_predictions.csv')
    keys = ['split_seed','split_role','sample_id']
    raw = pd.concat([cv,hold],ignore_index=True).drop(columns=['canonical_smiles'],errors='ignore')
    raw = raw.merge(s2[keys+['canonical_smiles','s2_no_transfer']],on=keys,validate='one_to_one').rename(columns={'s2_no_transfer':'S2_no_transfer'})
    oos = _oos_rows_with_similarity(raw,rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048),{})
    labeled = _aggregate_labeled_oos(oos)
    development = labeled.evaluation_set.eq('development')
    sd_cutoff = float(labeled.loc[development,'pLC50_sd'].median())
    sd_q75 = float(labeled.loc[development,'pLC50_sd'].quantile(.75))
    labeled['tanimoto_band'] = np.select([labeled.max_tanimoto.ge(.5),labeled.max_tanimoto.ge(.333),labeled.max_tanimoto.ge(.233)],['high_ge_0.50','within_0.333_0.50','borderline_0.233_0.333'],default='outside_lt_0.233')
    labeled['sd_band'] = np.select([labeled.pLC50_sd.le(sd_cutoff),labeled.pLC50_sd.le(sd_q75)],['low','moderate'],default='high')
    lookup = build_expected_error_lookup(labeled[development],min_rows=10)
    all_median = float(labeled.pLC50_sd.median())
    all_q75 = float(labeled.pLC50_sd.quantile(.75))
    local = labeled.copy()
    local['sd_band'] = np.select([local.pLC50_sd.le(all_median),local.pLC50_sd.le(all_q75)],['low','moderate'],default='high')
    local['error_square'] = local.abs_error**2
    local['large_error'] = local.abs_error.ge(local.abs_error.quantile(.75))
    dispersion = local.groupby('sd_band',as_index=False).agg(n=('sample_id','size'),mae=('abs_error','mean'),mse=('error_square','mean'),high_error_rate=('large_error','mean'))
    dispersion['rmse'] = dispersion.mse**.5
    cv_errors = oos[oos.split_role.eq('random_cv_oof')].copy()
    cv_errors['abs_error'] = abs(cv_errors.y_true-cv_errors.blend_y_pred)
    cv_errors['large_error'] = cv_errors.abs_error.ge(cv_errors.abs_error.quantile(.75))
    cv_errors['plot_band'] = np.select([cv_errors.max_tanimoto.ge(.5),cv_errors.max_tanimoto.ge(.333)],['high','within'],default='outside')
    similarity = cv_errors.groupby('plot_band',as_index=False).agg(n=('sample_id','size'),mae=('abs_error','mean'),rmse=('abs_error',lambda x:float(np.sqrt(np.mean(x**2)))),high_error_rate=('large_error','mean'))
    expected_similarity = pd.read_csv(PACKAGE/'05_supplementary/S5_local_error/similarity.csv').set_index('plot_band')
    expected_dispersion = pd.read_csv(PACKAGE/'05_supplementary/S5_local_error/dispersion.csv').set_index('sd_band')
    for actual,expected,key in [(similarity,expected_similarity,'plot_band'),(dispersion,expected_dispersion,'sd_band')]:
        aligned = actual.set_index(key).loc[expected.index]
        assert np.allclose(aligned[['n','mae','rmse','high_error_rate']],expected[['n','mae','rmse','high_error_rate']],rtol=0,atol=1e-10)
    parameters = {'candidate_sd_median':sd_cutoff,'candidate_sd_q75':sd_q75,'local_error_sd_median':all_median,'local_error_sd_q75':all_q75}
    return labeled,lookup,dispersion,parameters


if __name__ == '__main__':
    labeled,lookup,dispersion,parameters = domain()
    print(json.dumps(parameters,indent=2))
    print(dispersion.to_string(index=False))

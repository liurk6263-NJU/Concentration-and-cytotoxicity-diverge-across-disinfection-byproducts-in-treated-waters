from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def replay():
    seeds = pd.read_csv(ROOT/'data_processed/predictions_by_seed.csv')
    blend = (1-seeds.selected_weight)*seeds.randomcv80_pred+seeds.selected_weight*seeds.holdout80_pred
    assert np.allclose(blend,seeds.pLC50_pred,rtol=0,atol=1e-12)
    summary = seeds.groupby('query_id',as_index=False).agg(pLC50_mean=('pLC50_pred','mean'),pLC50_sd=('pLC50_pred','std'),pLC50_n=('pLC50_pred','size'),LC50_mean_mol_L=('LC50_mol_L_pred','mean'),LC50_sd_mol_L=('LC50_mol_L_pred','std'))
    assert summary.pLC50_n.eq(10).all()
    dbp = pd.read_csv(ROOT/'results/dbp_ad_expected_error_predictions.csv')
    ids = dbp.DBP_ID.astype(str)
    compared = pd.DataFrame({'query_id':ids,'expected':dbp.pLC50_mean}).merge(summary,on='query_id',validate='one_to_one')
    assert len(compared)==2795 and np.allclose(compared.expected,compared.pLC50_mean,rtol=0,atol=1e-12)
    expected = pd.read_csv(ROOT/'results/structure_matches.csv')
    query_ids = expected[['canonical_nsc_smiles','query_id']].drop_duplicates()
    pairs = pd.read_csv(ROOT/'data_processed/structure_pairs.csv').merge(query_ids,on='canonical_nsc_smiles',validate='many_to_one').merge(summary,on='query_id',validate='many_to_one')
    compared = pairs.merge(expected,on=['NSC','DBP_ID'],suffixes=('_replayed','_expected'),validate='one_to_one')
    assert len(compared)==2857 and np.allclose(compared.pLC50_mean_replayed,compared.pLC50_mean_expected,rtol=0,atol=1e-12)
    unique = pairs.drop_duplicates('query_id')
    rows = []
    for scope,frame in [('matching records',pairs),('unique candidate structures',unique)]:
        rows.append({'scope':scope,'n':len(frame),'ge_4':int(frame.pLC50_mean.ge(4).sum()),'ge_4_percent':float(frame.pLC50_mean.ge(4).mean()*100),'ge_4_5':int(frame.pLC50_mean.ge(4.5).sum()),'ge_4_5_percent':float(frame.pLC50_mean.ge(4.5).mean()*100)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    summary = replay()
    summary.to_csv(ROOT/'results/matching_summary.csv',index=False)
    print(summary.to_string(index=False))

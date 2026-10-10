from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def select():
    params = json.loads((ROOT/'data_processed/figure_parameters.json').read_text())
    dbp = pd.read_csv(ROOT/'results/dbp_ad_expected_error_predictions.csv')
    dbp = dbp[dbp.max_tanimoto.ge(params['tanimoto_minimum'])].copy()
    dbp['source_group'] = np.where(dbp.in_excel.eq(1), 'Standard-confirmed DBPs', 'Research-reported DBPs')
    blocks = []
    for flag, n in [(1,17),(0,83)]:
        block = dbp[dbp.in_excel.eq(flag)].sort_values(['pLC50_mean','DBP_ID'],ascending=[False,True]).head(n).copy()
        block['within_group_rank'] = np.arange(1,len(block)+1)
        blocks.append(block)
    a = pd.concat(blocks,ignore_index=True)
    a.insert(0,'plot_order',np.arange(1,len(a)+1))
    nsc = pd.read_csv(ROOT/'results/nsc_suspected_analogues_with_applicability_domain.csv')
    b = nsc[nsc.max_tanimoto.ge(params['tanimoto_minimum'])].sort_values(['pLC50_mean','query_id'],ascending=[False,True]).head(100).reset_index(drop=True)
    b.insert(0,'plot_order',np.arange(1,len(b)+1))
    return a,b


def verify():
    a,b = select()
    for name, actual, key in [('Figure5a_selected.csv',a,'DBP_ID'),('Figure5b_selected.csv',b,'query_id')]:
        frozen = pd.read_csv(ROOT/'results'/name)
        assert len(actual)==len(frozen)==100
        assert actual[key].tolist()==frozen[key].tolist()
        columns = ['pLC50_mean','pLC50_sd','max_tanimoto']
        assert np.allclose(actual[columns],frozen[columns],rtol=0,atol=1e-12)
    assert a.in_excel.value_counts().to_dict()=={0:83,1:17}
    return {'Figure5a':len(a),'Figure5b':len(b)}


if __name__=='__main__':
    print(json.dumps(verify()))

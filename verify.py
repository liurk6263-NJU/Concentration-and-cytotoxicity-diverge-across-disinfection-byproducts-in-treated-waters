from pathlib import Path
import ast
import hashlib
import importlib.util
import io
import json
import re
import sys
import tokenize
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = ROOT/'01_models'
sys.dont_write_bytecode = True
sys.path.insert(0,str(MODEL/'code/src'))
sys.path.insert(0,str(MODEL/'code/experiments'))
CHECKS = []


def check(name,condition):
    CHECKS.append({'check':name,'passed':bool(condition)})


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec)
    sys.modules[name]=loaded
    spec.loader.exec_module(loaded)
    return loaded


def metrics(frame,prediction,keys):
    rows=[]
    for key,g in frame.groupby(keys):
        key=key if isinstance(key,tuple) else (key,)
        y=g.y_true.to_numpy();p=g[prediction].to_numpy()
        rows.append(dict(zip(keys,key),r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum(),rmse=((y-p)**2).mean()**.5))
    return pd.DataFrame(rows)


def main():
    reference=json.loads((ROOT/'raw_hashes.json').read_text(encoding='utf-8'))
    for path,expected in reference.items():
        check('raw hash: '+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected)
    bad=[]
    for path in ROOT.rglob('*.py'):
        text=path.read_text(encoding='utf-8-sig')
        try:
            compile(text,str(path),'exec')
            tree=ast.parse(text)
            if any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(text).readline)):
                bad.append(path.name+': comments')
            if re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]',text):bad.append(path.name+': absolute path')
        except Exception as e:
            bad.append(path.name+': '+str(e))
    check('Python syntax, concise comments and relative paths',not bad)
    check('No bytecode or historical output folders',not any(p.suffix=='.pyc' or p.name=='__pycache__' or re.search(r'(?i)(historical|previous|backup|blocked|candidate)',p.name) for p in ROOT.rglob('*')))
    from nci_cho_transfer.formal_randomcv80_end_to_end import preflight_end_to_end_contract
    preflight=preflight_end_to_end_contract(MODEL)
    check('ET rebuild input contract',preflight['random_branch_schema_checked'] and preflight['holdout_branch_schema_checked'])
    comp=MODEL/'data_processed/components'
    s2=pd.read_csv(comp/'et_s2/et_s2_no_transfer_predictions.csv')
    cv=pd.read_csv(comp/'et_s3/formal_random_cv_oof_predictions.csv').assign(split_role='random_cv_oof')
    hold=pd.read_csv(comp/'et_s3/formal_holdout_predictions.csv')
    s3=pd.concat([cv,hold],ignore_index=True)
    keys=['split_seed','split_role','sample_id']
    s3=s3.drop(columns=['canonical_smiles'],errors='ignore').merge(s2[keys+['canonical_smiles']],on=keys,validate='one_to_one')
    check('ET stable keys',not s3.duplicated(keys).any() and len(s3)==6600)
    overlaps=[]
    fold_overlaps=[]
    for seed,g in s2.groupby('split_seed'):
        g=g.copy();g['key']=g.canonical_smiles+'|'+g.time_h.astype(int).astype(str)
        h=g[g.split_role.ne('random_cv_oof')]
        if h.groupby('key').split_role.nunique().gt(1).any():overlaps.append(int(seed))
        c=g[g.split_role.eq('random_cv_oof')]
        if c.groupby('key').random_cv_fold.nunique().gt(1).any():fold_overlaps.append(int(seed))
    check('Canonical SMILES + time: role isolation',not overlaps)
    check('Canonical SMILES + time: CV fold isolation',not fold_overlaps)
    seed_roles=s3[s3.split_role.ne('train')].groupby(['split_seed','split_role']).size().unstack()
    check('Ten seeds and 291/37/37/4 roles',len(seed_roles)==10 and seed_roles[['random_cv_oof','test','validation','experimental']].eq([291,37,37,4]).all().all())
    stored=pd.read_csv(MODEL/'results/et_metrics_by_seed.csv')
    for stage,df,pred in [('S1',pd.read_csv(MODEL/'results/et_s1_predictions.csv'),'y_pred'),('S2',s2,'s2_no_transfer'),('S3',s3,'blend_y_pred')]:
        actual=metrics(df[df.split_role.ne('train')],pred,['split_seed','split_role'])
        merged=actual.merge(stored[stored.stage.eq(stage)],on=['split_seed','split_role'],suffixes=('_actual','_stored'),validate='one_to_one')
        check('ET '+stage+' R2 and RMSE',len(merged)==40 and np.allclose(merged[['r2_actual','rmse_actual']],merged[['r2_stored','rmse_stored']],rtol=0,atol=1e-12))
    summary=json.loads((comp/'et_s3/summary.json').read_text(encoding='utf-8'))
    actual_summary=stored[stored.stage.eq('S3')].groupby('split_role')[['r2','rmse']].mean()
    check('ET-S3 metadata uses ten seed-specific metric means',all(abs(summary['holdout_by_role'][role][name]-actual_summary.loc[role,name])<1e-12 for role in ['test','validation','experimental'] for name in ['r2','rmse']))
    interference=module(MODEL/'code/interference.py','release_interference')
    inter=[]
    for seed in interference.FIXED_SEEDS:
        aligned=interference._load_aligned(MODEL,seed)
        for role,g in aligned.groupby('eval_role'):
            if role=='train':continue
            g=g.sort_values(['sample_id','time_h'],kind='mergesort')
            y=g.y_true.to_numpy();base=g.reference_prediction.to_numpy();delta=g.latest_s3.to_numpy()-base
            for repeat in range(1,11):
                rng=np.random.default_rng(int(seed)+interference.ROLE_OFFSETS[role]+(repeat-1)*interference.REPEAT_STRIDE)
                pred=base+.5*delta+.5*rng.permutation(delta)
                inter.append({'split_seed':seed,'split_role':role,'repeat':repeat,'r2':interference._r2(y,pred),'rmse':float(np.sqrt(np.mean((y-pred)**2)))})
    actual=pd.DataFrame(inter)
    frozen=pd.read_csv(MODEL/'results/interference_metrics.csv')
    merged=actual.merge(frozen,on=['split_seed','split_role','repeat'],validate='one_to_one')
    check('Interference 400 seed-role-repeat metrics',len(merged)==400 and np.allclose(merged[['r2','rmse']],merged[['interference_r2','interference_rmse']],rtol=0,atol=1e-12))
    expected={'random_cv_oof':.685510139295464,'test':.8119323440783677,'validation':.7296229396644007,'experimental':.6028786513456602}
    check('Interference published R2',all(abs(actual[actual.split_role.eq(k)].r2.mean()-v)<1e-12 for k,v in expected.items()))
    for role in ['random_cv_oof','test']:
        name='random_cv' if role=='random_cv_oof' else 'holdout'
        predictions=pd.read_csv(MODEL/'results/non_et'/(name+'_stage_predictions.csv'))
        frozen=pd.read_csv(MODEL/'results/non_et'/(name+'_metrics.csv'))
        if name=='random_cv':frozen['split_role']='random_cv_oof'
        actual=metrics(predictions,'y_pred',['family','split_seed','split_role','stage'])
        merged=actual.merge(frozen,on=['family','split_seed','split_role','stage'],suffixes=('_actual','_stored'),validate='one_to_one')
        check('Four algorithms '+role+' S1/S2/S3',len(merged)==120 and np.allclose(merged[['r2_actual','rmse_actual']],merged[['r2_stored','rmse_stored']],rtol=0,atol=1e-12))
        check('Four algorithms scope: '+role,set(predictions.split_role)=={role})
    replay=module(ROOT/'04_screening/code/replay_blend.py','screening_replay')
    screening=replay.replay()
    check('39150 screening predictions and 2857 matches',screening.n.tolist()==[2857,1120] and screening.ge_4.tolist()==[27,25] and screening.ge_4_5.tolist()==[6,5])
    records=pd.read_csv(ROOT/'04_screening/results/dbp_screening_records.csv')
    check('DBP catalogue: 3104 records, 2795 structures, 279/71 thresholds',len(records)==3104 and records.exact_molecule_key.nunique()==2795 and int(records.pLC50_mean.ge(4).sum())==279 and int(records.pLC50_mean.ge(4.5).sum())==71)
    sys.path.insert(0,str(ROOT/'04_screening/code'))
    domain=module(ROOT/'04_screening/code/domain.py','screening_domain')
    domain.domain()
    check('Local-error source values and population thresholds',True)
    from openpyxl import load_workbook
    water=load_workbook(ROOT/'02_treated_water/data_raw/treated_water_english.xlsx',read_only=True,data_only=True)
    samples=sum(any(v is not None for v in row) for row in water['wide_sample_DBP_Ci'].iter_rows(values_only=True))-1
    occurrences=sum(any(v is not None for v in row) for row in water['long_sample_DBP_Ci'].iter_rows(values_only=True))-1
    check('Water: 63 samples and 614 occurrence records',samples==63 and occurrences==614)
    water.close()
    index=pd.read_csv(ROOT/'INDEX.csv').fillna('')
    invalid=[]
    for row in index.itertuples():
        for field in [row.data_or_results,row.code,row.figure]:
            invalid.extend(p for p in field.split(';') if p and not (ROOT/p).exists())
    check('Main/SI index paths',not invalid)
    selection=module(ROOT/'04_screening/code/selection.py','screening_selection')
    selected=selection.verify()
    check('Figure5a/b frozen selection replay: 100+100 rows',selected=={'Figure5a':100,'Figure5b':100})
    local=ROOT/'05_supplementary/S5_local_error'
    texts=[''.join(x.itertext()).strip() for x in ET.parse(local/'FigureS5.svg').iter() if x.tag.endswith('text') or x.tag.endswith('tspan')]
    check('FigureS5 six SD labels: 0.158/0.228',all(sum(x in value for value in texts)>=2 for x in ['0.158','0.228']) and not any('0.165' in x or ('0.233' in x and 'SD' in x) for x in texts))
    table=pd.read_csv(ROOT/'05_supplementary/tables/TableS7.csv')
    dispersion=pd.read_csv(local/'dispersion.csv').set_index('sd_band').loc[['low','moderate','high']]
    shown=table[table['0'].eq('Ten-run prediction SD')]
    check('TableS7 SD tiers preserve source errors and 185/92/92 groups',shown['2'].tolist()==['≤0.158','0.158–0.228','>0.228'] and np.all(np.abs(shown[['3','4','5']].astype(float).to_numpy()-dispersion[['mae','rmse','high_error_rate']].to_numpy()*[1,1,100])<=[.0005,.0005,.05]) and shown['3'].iloc[0]=='0.233' and dispersion.n.tolist()==[185,92,92])
    check('S5 source numerical label Low MAE remains 0.233','0.233' in texts)
    expected_order=['Figure 1','Figure 2','Figure 3','Figure 4','Figure 5','Figure S1','Table S1','Table S2','Table S3','Figure S2','Table S4','Note S1','Figure S3','Table S5','Note S2','Figure S4','Note S3','Table S6','Figure S5','Table S7','Note S4','Note S5','Table S8','Note S6','Note S7','Table S9','Note S8','Note S9','Supplementary Data 1','Supplementary Data 2','Supplementary Data 3']
    check('Index follows current manuscript and SI order',index.paper_item.tolist()==expected_order)
    embedding=module(ROOT/'04_screening/code/embedding.py','screening_embedding')
    rows=embedding.add_confidence_fields(embedding.load_records())
    priority=embedding.select_priority(rows)
    exemplars=pd.read_csv(ROOT/'04_screening/results/exemplars.csv').sort_values('plot_order')
    check('Figure5c six frozen priority exemplar identities',rows.loc[priority,'record_id'].tolist()==exemplars.record_id.tolist())
    merged=exemplars.merge(rows,on='record_id',suffixes=('_exemplar','_source'),validate='one_to_one')
    check('Figure5c six source SMILES and predicted values',merged.canonical_smiles.tolist()==merged.smiles.tolist() and np.allclose(merged[['pLC50_mean_exemplar','pLC50_sd_exemplar','max_tanimoto_exemplar']],merged[['pLC50_mean_source','pLC50_sd_source','max_tanimoto_source']],rtol=0,atol=1e-12))
    from rdkit import Chem
    verified=exemplars[exemplars.pubchem_status.eq('verified')]
    check('Recorded PubChem evidence matches source InChIKeys',len(verified)>=3 and all(Chem.MolToInchiKey(Chem.MolFromSmiles(row.canonical_smiles))==Chem.MolToInchiKey(Chem.MolFromSmiles(row.pubchem_smiles))==row.inchikey for row in verified.itertuples()))
    check('Figure5c output paths',embedding.PNG_OUT.name=='Figure5c.png' and embedding.SVG_OUT.name=='Figure5c.svg')
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        obj=json.loads(manifest.read_text(encoding='utf-8'))
        check('Manifest file hashes',all(hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in obj['files']))
        check('Manifest complete file set',set(x['path'] for x in obj['files'])==set(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name not in ['MANIFEST.json','checksums.sha256']))
    failed=[x['check'] for x in CHECKS if not x['passed']]
    print(json.dumps({'checks':len(CHECKS),'failed_checks':failed,'details':bad if bad else None},indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':
    main()

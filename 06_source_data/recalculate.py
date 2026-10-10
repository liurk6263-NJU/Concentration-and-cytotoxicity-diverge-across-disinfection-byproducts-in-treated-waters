from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'06_source_data/mixture_water_chain'

water=pd.read_excel(ROOT/'02_treated_water/data_raw/treated_water.xlsx',sheet_name='long_sample_DBP_Ci')
converted=pd.read_csv(DATA/'Figure3_raw_occurrence_614.csv')
assert len(water)==len(converted)==614
expected_molar=water.Ci_value_ug_L.to_numpy()*1e-6/np.array([Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in water.DBPs])
assert np.allclose(expected_molar,converted.Ci_mol_L,rtol=0,atol=1e-14)
plotted=pd.read_csv(ROOT/'02_treated_water/results/Figure3_source_data.csv')
totals=water.groupby('sampleID').Ci_value_ug_L.sum()
display_order=[4,5,1,2,3,6,8,9,10,11,12,7]
display_labels=[f'S{i}' for i in range(1,13)]
assert plotted.groupby('sample').original_sample_id.first().loc[display_labels].tolist()==display_order
displayed=plotted.groupby('sample').concentration_ug_L.sum().loc[display_labels]
assert np.allclose(displayed.to_numpy(),[175.8,89.54,25.3,119.11,80.54,88.11,111.27,27.48,45.56,105.49,131.27,70.31],rtol=0,atol=1e-12)
group_map={'THMs / haloalkanes':'THMs','Halocarboxylic acids / organic acids':'Haloacids','Haloacids':'Haloacids',
           'Haloacetonitriles / cyanogen halides':'HANs/CNHs','Halonitromethanes / nitroalkanes':'HNMs',
           'Haloaldehydes':'Haloaldehydes','Haloketones':'Haloketones',
           'Halofuranones / MX-like DBPs':'Halofuranones','Halofuranones / mucohalic acids':'Halofuranones',
           'Carbonyl DBPs / diketones':'Other','Carbonyl DBPs / ketones':'Other','Oxygenated organics / ethers':'Other'}
selected=water[water.sampleID.isin(display_order)].copy()
selected['sample']=selected.sampleID.map(dict(zip(display_order,display_labels)))
selected['category']=selected.DBP_group.map(group_map)
assert selected.category.notna().all()
original_categories=selected.groupby(['sample','category']).Ci_value_ug_L.sum()
keys=pd.MultiIndex.from_frame(plotted[['sample','category']])
assert np.allclose(original_categories.reindex(keys,fill_value=0),plotted.concentration_ug_L,rtol=0,atol=1e-12)
expected_pct=plotted.concentration_ug_L.to_numpy()/plotted['sample'].map(displayed).to_numpy()*100
assert np.allclose(expected_pct,plotted.concentration_pct,rtol=0,atol=1e-12)
assert plotted[plotted['sample'].eq('S12')].treatment_group.eq('O3').all()

mixture=pd.read_csv(DATA/'FigureS3_raw_mixture_LC50_3.csv')
assert np.allclose(1/(mixture.measured_total_LC50_M*1e6),mixture.BCAT_from_raw_LC50,rtol=0,atol=1e-15)
d_exp=200*abs(mixture.assumed_total_LC50_M-mixture.measured_total_LC50_M)/(mixture.assumed_total_LC50_M+mixture.measured_total_LC50_M)
assert np.array_equal(d_exp.round(2),[9.43,14.51,6.57])
d_source=pd.read_csv(DATA/'FigureS3_prediction_D_source_latest_diagnostic.csv')
prediction=mixture.merge(d_source,left_on='group',right_on='Group',validate='one_to_one')
monomer=pd.read_csv(DATA/'FigureS3_prediction_D_monomer_seeds.csv')
mixture_seed=pd.read_csv(DATA/'FigureS3_prediction_D_mixture_seeds.csv')
assert len(monomer)==40 and len(mixture_seed)==30
assert np.allclose(np.power(10.,-monomer.predicted_pLC50)*1e6,monomer.predicted_LC50_uM,rtol=1e-12,atol=1e-9)
groups={'Group1':['exp_0001','exp_0002'],'Group2':['exp_0003','exp_0004'],'Group3':['exp_0001','exp_0002','exp_0003','exp_0004']}
calculations=[]
for seed,part in monomer.groupby('split_seed'):
    lookup=part.set_index('sample_id')
    for group,ids in groups.items():
        lc50=lookup.loc[ids,'predicted_LC50_uM']
        calculations.append({'split_seed':seed,'Group':group,'CAT':len(ids)/lc50.sum(),'LC50_uM':lc50.mean()})
calculated=pd.DataFrame(calculations).merge(mixture_seed,on=['split_seed','Group'],validate='one_to_one')
assert np.allclose(calculated.CAT,calculated.CAT_pred_latest_diagnostic,rtol=0,atol=1e-14)
model_lc50=calculated.groupby('Group').LC50_uM.mean().loc[prediction.group].to_numpy()
assert np.allclose(model_lc50,prediction.CAT_pred_equivalent_LC50_uM_mean_10seed,rtol=0,atol=1e-8)
measured_lc50=prediction.measured_total_LC50_M*1e6
d_pred=200*abs(model_lc50-measured_lc50)/(model_lc50+measured_lc50)
assert np.array_equal(d_pred.round(2),[2.16,16.35,15.82])

counts=pd.read_csv(ROOT/'04_screening/results/matching_summary.csv')
assert counts.n.tolist()==[2857,1120]
assert counts.ge_4.tolist()==[27,25] and counts.ge_4_5.tolist()==[6,5]
status=pd.read_csv(ROOT/'06_source_data/source_status.csv')
print(json.dumps({'Ci_molar_conversion':614,'displayed_concentration_categories':96,'experimental_D':d_exp.round(2).tolist(),'prediction_D':d_pred.round(2).tolist(),'pending_source_links':status[status.source_status.str.contains('pending')][['paper_item','panels']].to_dict('records')},indent=2))

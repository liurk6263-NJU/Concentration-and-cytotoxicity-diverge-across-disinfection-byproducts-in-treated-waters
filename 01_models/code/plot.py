from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
COLORS = ['#acc9db', '#337ea5', '#b7b7b7']


def figure2():
    metrics = pd.read_csv(ROOT / 'results/et_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    ax = axes[0, 0]
    for j, role in enumerate(['random_cv_oof', 'test']):
        g = metrics[metrics.split_role.eq(role)].set_index('stage').loc[['S1','S2','S3']]
        bars = ax.bar(np.arange(3)+(j-.5)*.32, g.r2_mean, width=.32, yerr=g.r2_sd, capsize=3, color=COLORS[j], edgecolor='#555555', label=['CV', 'Held-out'][j])
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(range(3), ['Baseline', 'Optimized', 'Cross-cell transfer'])
    ax.set_ylim(0, 1.06); ax.set_ylabel('$R^2$'); ax.set_title('a  Model-stage performance', loc='left', fontweight='bold'); ax.legend(frameon=False, ncol=2)
    for ax, role, letter, title in [(axes[0,1], 'validation','b','Independent literature validation'), (axes[1,0], 'experimental','c','Independent experimental compounds')]:
        g = pd.read_csv(ROOT / 'results' / (role+'_compounds.csv'))
        x = np.arange(len(g)); width = .4
        for j, field in enumerate(['measured_plc50', 'predicted_plc50']):
            err = g.prediction_sd if j else None
            ax.bar(x+(j-.5)*width, g[field], width, yerr=err, capsize=2, color=COLORS[j], edgecolor='#555555', label=['Measured', 'Predicted'][j])
        labels = g.canonical_smiles.astype(str).tolist()
        if role == 'experimental':
            names = {'Oc1c(Cl)cccc1Cl':'2,6-Dichlorophenol','O=[N+]([O-])c1ccc(O)c(Cl)c1':'2-Chloro-4-nitrophenol','O=Cc1cc(I)c(O)c(I)c1':'3,5-Diiodo-4-hydroxybenzaldehyde','O=C(O)c1cc(Br)ccc1O':'5-Bromosalicylic acid'}
            labels = [names.get(s,s) for s in labels]
        ax.set_xticks(x, labels, rotation=65 if role=='validation' else 20, ha='right', fontsize=6 if role=='validation' else 8)
        ax.set_ylim(0, 6 if role=='validation' else 4.5); ax.set_ylabel('pLC50'); ax.legend(frameon=False, ncol=2); ax.set_title(letter+'  '+title, loc='left', fontweight='bold')
    ax = axes[1,1]; g = pd.read_csv(ROOT / 'results/transfer_ablation.csv')
    for j, (field, sd, label, color) in enumerate([('no_transfer_r2','no_transfer_sd','No transfer','#b7b7b7'), ('interference_r2','interference_sd','Interference transfer','white'), ('authentic_transfer_r2','authentic_transfer_sd','Authentic transfer','#337ea5')]):
        bars = ax.bar(np.arange(4)+(j-1)*.25, g[field], width=.25, yerr=g[sd], capsize=3, color=color, edgecolor='#555555', label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_xticks(range(4), ['CV','Held-out','Literature','Experiment']); ax.set_ylim(-.12,1.12); ax.set_ylabel('$R^2$'); ax.legend(frameon=False, fontsize=8, ncol=3); ax.set_title('d  Transfer-control performance', loc='left', fontweight='bold')
    out = ROOT / 'figures'; out.mkdir(exist_ok=True)
    fig.savefig(out/'Figure2.png', dpi=220); fig.savefig(out/'Figure2.svg'); plt.close(fig)


def figure_s1():
    folder = ROOT.parent / '05_supplementary/S1_model_comparison'
    data = pd.read_csv(folder / 'results.csv')
    families = ['ExtraTrees','RandomForest','CatBoost','SVR','XGBoost']
    colors = ['#b7b7b7','#acc9db','#337ea5']
    fig, axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax, role, metric, letter in [(axes[0,0],'random_cv_oof','r2','a'),(axes[0,1],'test','r2','b'),(axes[1,0],'random_cv_oof','rmse','c'),(axes[1,1],'test','rmse','d')]:
        for j, stage in enumerate(['S1','S2','S3']):
            g = data[data.split_role.eq(role)&data.stage.eq(stage)].set_index('family').loc[families]
            bars = ax.bar(np.arange(5)+(j-1)*.25, g[metric+'_mean'], width=.25, yerr=g[metric+'_sd'], color=colors[j], capsize=2, label=stage)
            ax.bar_label(bars, fmt='%.2f', fontsize=6, padding=2)
        ax.set_xticks(range(5),families,rotation=15);ax.set_ylabel('$R^2$' if metric=='r2' else 'RMSE (pLC50)');ax.set_title(letter+'  '+('Random K-fold CV' if role=='random_cv_oof' else 'Held-out test'),loc='left',fontweight='bold'); ax.legend(frameon=False,fontsize=8)
    fig.savefig(folder/'FigureS1.png',dpi=220);fig.savefig(folder/'FigureS1.svg');plt.close(fig)


if __name__ == '__main__':
    figure2()
    figure_s1()

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','font.size':10})


def plot():
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,metric,letter in [(axes[0,0],'r2','a'),(axes[0,1],'rmse','b')]:
        data=pd.read_csv(ROOT/('identity_'+metric+'.csv'))
        for j,(field,label,color) in enumerate([('original_s3_'+metric,'Complete transfer','#acc9db'),('leave_identical_'+metric,'Identity excluded','#337ea5')]):
            bars=ax.bar(np.arange(3)+(j-.5)*.3,data[field],width=.3,color=color,label=label)
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
        ax.set_xticks(range(3),['Held-out','Literature','Experiment'])
        ax.set_ylabel('$R^2$' if metric=='r2' else 'RMSE (pLC50)')
        ax.set_title(letter+'  Identical-structure control',loc='left',fontweight='bold')
        ax.legend(frameon=False,fontsize=8)
        ax.set_ylim(0,1 if metric=='r2' else .6)
    data=pd.read_csv(ROOT/'channel_allocation.csv')
    for ax,field,letter in [(axes[1,0],'r2_gain_percent','c'),(axes[1,1],'rmse_reduction_percent','d')]:
        bars=ax.bar(range(3),data[field],color=['#b7b7b7','#acc9db','#337ea5'])
        ax.bar_label(bars,fmt='%.2f%%',padding=3,fontsize=9)
        ax.set_xticks(range(3),['Structure','Cellular state','Exposure time'])
        ax.set_ylabel('Allocated share (%)');ax.set_ylim(0,65)
        ax.set_title(letter+'  '+('R² gain' if letter=='c' else 'RMSE reduction'),loc='left',fontweight='bold')
    fig.savefig(ROOT/'FigureS2.png',dpi=220);fig.savefig(ROOT/'FigureS2.svg');plt.close(fig)


if __name__=='__main__':
    plot()

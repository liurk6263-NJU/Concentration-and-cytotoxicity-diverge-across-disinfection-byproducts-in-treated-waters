from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def plot():
    params = json.loads((ROOT/'parameters.json').read_text())
    lo,hi = params['local_error_sd_median'],params['local_error_sd_q75']
    similarity = pd.read_csv(ROOT/'similarity.csv').set_index('plot_band').loc[['high','within','outside']]
    dispersion = pd.read_csv(ROOT/'dispersion.csv').set_index('sd_band').loc[['low','moderate','high']]
    labels = [['T >= 0.50','T: 0.333-0.50','T < 0.333'],[f'SD <= {lo:.3f}',f'SD: {lo:.3f}-{hi:.3f}',f'SD > {hi:.3f}']]
    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','font.size':10})
    fig,axes = plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for row,(data,ticks) in enumerate(zip([similarity,dispersion],labels)):
        x = np.arange(3)
        for offset,metric,color in [(-0.18,'mae','#4477aa'),(0.18,'rmse','#66ccee')]:
            bars = axes[row,0].bar(x+offset,data[metric],width=0.36,label=metric.upper(),color=color)
            axes[row,0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        bars = axes[row,1].bar(x,data.high_error_rate*100,width=0.6,color='#4477aa')
        axes[row,1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        axes[row,0].set_ylabel('Error (pLC50)')
        axes[row,1].set_ylabel('Large-error fraction (%)')
        axes[row,0].legend(frameon=False)
        for col in range(2):
            axes[row,col].set_xticks(x,ticks)
            axes[row,col].set_ylim(0,float(axes[row,col].get_ylim()[1])*1.12)
            axes[row,col].text(-0.1,1.03,'abcd'[2*row+col],transform=axes[row,col].transAxes,fontweight='bold')
    fig.savefig(ROOT/'FigureS5.png',dpi=300)
    fig.savefig(ROOT/'FigureS5.svg')
    plt.close(fig)


if __name__=='__main__':
    plot()

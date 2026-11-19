import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('--out',default='reports/figures')
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.csv)
    if not {'policy','scenario','rate','queue_mean'} <= set(df): raise ValueError('episode CSV needs policy/scenario/rate/queue_mean columns')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for (policy,scenario),g in df.groupby(['policy','scenario']):
        m=g.groupby('rate').queue_mean.agg(['mean','std']).sort_index()
        ax.errorbar(m.index,m['mean'],yerr=m['std'].fillna(0),marker='o',label=f'{policy} / {scenario}')
    ax.set(xlabel='Demand per entrance (veh/h)',ylabel='Mean queued vehicles',title=f'Queue vs demand (episodes n={len(df)})')
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'queue_vs_demand.png',dpi=160)
if __name__=='__main__': main()

"""Rebuild explanatory figures from recorded values. No model training/inference."""
import csv
import json
from build_architecture import architecture, refinement
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, TEAL, MUTED = '#13213c', '#345ff4', '#168c87', '#66748b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': NAVY, 'text.color': NAVY, 'axes.titleweight': 'bold', 'svg.fonttype': 'none'})


def results():
    data=json.loads((ROOT/'results/recorded-metrics.json').read_text())
    rec=data['reconstruction']; gen=data['matched_generation']['rows']
    fig, axes=plt.subplots(2,2,figsize=(13,8))
    for ax, rows, field, title, direction, maximum in [
        (axes[0,0],rec,'pixel_mse','Validation reconstruction MSE','lower is better',.0045),
        (axes[0,1],rec,'ssim','Validation reconstruction SSIM','higher is better',1),
        (axes[1,0],gen,'fid','Matched generation FID','lower is better',260),
        (axes[1,1],gen,'kid_mean','Matched generation KID','lower is better',.34),
    ]:
        names=[r['model'] for r in rows]; vals=[r[field] for r in rows]
        colors=[BLUE if name=='H3B' else TEAL if name=='H3C' or name.startswith('H6A') else MUTED for name in names]
        bars=ax.bar(names,vals,color=colors,width=.5)
        ax.set_ylim(0,maximum);ax.set_title(title,loc='left',pad=22,fontsize=13)
        ax.text(0,1.02,direction,transform=ax.transAxes,fontsize=10,color=MUTED)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for bar,v in zip(bars,vals):
            label=f'{v:.6f}' if field=='pixel_mse' else f'{v:.4f}' if field in ('ssim','kid_mean') else f'{v:.2f}'
            ax.text(bar.get_x()+bar.get_width()/2,v+maximum*.025,label,ha='center',fontsize=11,weight='bold')
    fig.suptitle('Reconstruction quality and prior-sample fidelity are different questions',fontsize=17,x=.06,ha='left',weight='bold')
    fig.text(.06,.02,'Historical report values · Generation: 522 real / 522 prior samples, seed 2026 · Validation only; no significance claim',fontsize=10,color=MUTED)
    fig.tight_layout(rect=(.03,.06,.99,.95),h_pad=3)
    fig.savefig(OUT/'results.png',dpi=180,facecolor='white');fig.savefig(OUT/'results.svg',facecolor='white');plt.close(fig)


def history():
    with (ROOT/'results/H3B_training_history_from_notebook.csv').open() as f: rows=list(csv.DictReader(f))
    epochs=np.arange(1,51)
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    for ax,fields,title,ylabel in [
        (axes[0],['train_objective','val_objective'],'Composite objective','Hybrid objective'),
        (axes[1],['val_pixel_mse'],'Validation pixel MSE','MSE'),
        (axes[2],['val_ssim'],'Validation SSIM','SSIM'),
    ]:
        for field in fields:
            values=np.full(50,np.nan)
            for row in rows: values[int(row['epoch'])-1]=float(row[field])
            ax.plot(epochs,values,marker='.',markersize=5,linewidth=1.8,label=field.replace('_',' '))
        ax.set_title(title,loc='left',fontsize=13);ax.set_xlabel('Epoch');ax.set_ylabel(ylabel);ax.grid(alpha=.15)
        ax.legend(fontsize=8,frameon=False)
    fig.suptitle('H3B observed history · 36 of 50 epoch rows preserved',x=.06,ha='left',weight='bold',fontsize=17)
    fig.text(.06,.015,'Gaps indicate missing notebook output; no interpolation. Train/validation composite objectives use different evaluation conditions.',color=MUTED,fontsize=10)
    fig.tight_layout(rect=(.01,.06,1,.92));fig.savefig(OUT/'observed-history.png',dpi=170,facecolor='white');plt.close(fig)


if __name__ == '__main__':
    architecture();refinement();results();history()
    svg=OUT/'results.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print('Built architecture.svg, results.png/svg and observed-history.png from preserved evidence.')

"""Rebuild explanatory figures from recorded values. No model training/inference."""
import csv
import json
from html import escape
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


def architecture():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="740" viewBox="0 0 1280 740" role="img" aria-label="Hierarchical VAE prior and posterior paths">', '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#59719a" stroke-width="1.5"/></marker></defs>', '<rect width="1280" height="740" rx="24" fill="#f5f7fc"/>']
    def text(x, y, value, size=22, color=NAVY, weight=400):
        parts.append(f'<text x="{x}" y="{y}" font-family="Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')
    def box(x, y, title, lines, accent=BLUE):
        parts.append(f'<rect x="{x}" y="{y}" width="260" height="120" rx="15" fill="white" stroke="#cbd5e8"/>')
        text(x+20,y+35,title,23,accent,650)
        for i, value in enumerate(lines): text(x+20,y+69+27*i,value,19)
    def arrow(x, y):
        parts.append(f'<path d="M{x} {y} h34" stroke="#59719a" stroke-width="2" marker-end="url(#arrow)"/>')
    text(42,55,'Hierarchical Chest X-ray Generation',36,NAVY,700)
    text(42,90,'H3B architecture · independent prior samples and input-conditioned reconstruction',21,MUTED)
    text(42,145,'PRIOR GENERATION  /  NO INPUT IMAGE',18,TEAL,700)
    box(42,170,'Global prior',['z_g ~ N(0, I)','128 dimensions'],TEAL)
    box(354,170,'Conditional spatial prior',['p(z_l | z_g)','8 × 32 × 32 Gaussian'],TEAL)
    box(666,170,'Stochastic spatial code',['Sample z_l from the prior','No encoder features'],TEAL)
    box(978,170,'Residual decoder',['Inject local code at 32²','128 × 128 grayscale'],TEAL)
    for x in (309,621,933): arrow(x,230)
    text(42,355,'POSTERIOR RECONSTRUCTION  /  INPUT-CONDITIONED',18,BLUE,700)
    box(42,380,'Input + encoder',['128² → 64² → 32² → 8²','GroupNorm / SiLU'])
    box(354,380,'Global posterior',['q(z_g | x)','Reparameterized Gaussian'])
    box(666,380,'Spatial posterior',['q(z_l | x, z_g)','Prior + learned correction'])
    box(978,380,'Same decoder',['Bounded spatial injection','No deterministic skips'])
    for x in (309,621,933): arrow(x,440)
    parts.append('<rect x="42" y="552" width="1196" height="140" rx="18" fill="#13213c"/>')
    text(65,590,'STRUCTURE-AWARE TRAINING',18,'#a9c3ff',700)
    text(65,626,'MSE_sum + 48(1 − SSIM) + 32 Sobel-L1 + β(KL_global + 0.35 KL_local)',24,'#ffffff')
    text(65,661,'10-epoch KL warm-up · EMA · validation selection at final KL weight',20,'#c4d0e6')
    parts.append('</svg>')
    (OUT/'architecture.svg').write_text('\n'.join(parts),encoding='utf-8')


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
        bars=ax.bar(names,vals,color=[MUTED]*(len(rows)-1)+[BLUE if len(rows)==3 else TEAL],width=.5)
        ax.set_ylim(0,maximum);ax.set_title(title,loc='left',pad=22,fontsize=13)
        ax.text(0,1.02,direction,transform=ax.transAxes,fontsize=10,color=MUTED)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for bar,v in zip(bars,vals):
            label=f'{v:.5f}' if field=='pixel_mse' else f'{v:.4f}' if field in ('ssim','kid_mean') else f'{v:.2f}'
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
    architecture();results();history()
    print('Built architecture.svg, results.png/svg and observed-history.png from preserved evidence.')

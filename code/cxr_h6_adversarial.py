"""H6A: Decoder-only adversarial fine-tuning of the original H3B HVAE.

Real and prior generated X-rays are used for a small spectral-normalized
Patch Discriminator. Only H3B decoder blocks are updated; H3B encoder,
global mapping and conditional prior stay *identical* to the source checkpoint.
H6R is a controlled reconstruction-only discriminator ablation.

Requires cxr_h3_research.py and cxr_h4b_flow.py beside this file, Kaggle H3B
private dataset attached (extracted best.pth + config.json or H3B_EXPORT.zip),
original chest X-ray dataset attached, and pytorch-msssim installed.

Commands: unit-test, check, train, inspect, evaluate, export.
Not a medical image synthesis tool for clinical use.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import zipfile
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

import cxr_h3_research as h3
import cxr_h4b_flow as h4

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None


@dataclass
class Config:
    experiment: str = 'H6A'
    fake_mode: str = 'both'                 # both = H6A; recon = H6R
    seed: int = 2026
    epochs: int = 15                     # fixed schedule across staged resume
    stop_after: int = 5                  # may increase without changing schedule
    batch_size: int = 8
    g_lr: float = 1e-5
    d_lr: float = 5e-5
    weight_decay: float = 1e-5
    adv_ratio: float = 0.05             # target adv/reconstruction gradient ratio
    adv_max_weight: float = 20.0
    grad_probe_every: int = 30
    d_channels: int = 32
    d_grad_clip: float = 5.0
    g_grad_clip: float = 1.0
    ema_decay: float = 0.995
    export_every: int = 5
    data_root: str = '/kaggle/input'
    out_root: str = '/kaggle/working/cxr_hvae'


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False


def atomic_torch(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)


def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def log_row(folder, row):
    csvp = folder / 'history.csv'; exists = csvp.exists()
    with csvp.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists: w.writeheader()
        w.writerow(row); f.flush(); os.fsync(f.fileno())
    with (folder/'history.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())


def log_event(folder, evt, **kwargs):
    with (folder / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(event=evt, time=time.time(), **kwargs))+'\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def frozen_model(cfg, device):
    base, oldcfg, checksum = h4.load_base(
        h4.Config(data_root=cfg.data_root, output_root=cfg.out_root,
                  batch_size=cfg.batch_size), device)
    if oldcfg.experiment != 'H3B': raise RuntimeError('Only original H3B is supported')
    return base, oldcfg, checksum


DECODER_PREFIX = ('d8_res.', 'd16.', 'd16_res.', 'd32.', 'd32_res.',
                  'd64.', 'd128.', 'local_to_dec.', 'out.')


def unlock_decoder(net):
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith(DECODER_PREFIX) or name == 'local_gain')
    ps = [p for p in net.parameters() if p.requires_grad]
    assert len(ps)>0, 'No decoder parameters found'
    return ps


class PatchDiscriminator(nn.Module):
    """Small SN PatchGAN; no sigmoid (hinge logits)."""
    def __init__(self, c=32):
        super().__init__()
        conv = lambda ci, co, st: nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(ci, co, kernel_size=4, stride=st, padding=1))
        self.net = nn.Sequential(
            conv(1,c,2), nn.LeakyReLU(.2,inplace=True),
            conv(c,c*2,2), nn.LeakyReLU(.2,inplace=True),
            conv(c*2,c*4,2), nn.LeakyReLU(.2,inplace=True),
            conv(c*4,c*4,1), nn.LeakyReLU(.2,inplace=True),
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(c*4,1,3,padding=1))
        )
    def forward(self, x): return self.net(x)


def augment_for_disc(x, do=True):
    """Mild, anatomy-preserving augmentations used on both domains."""
    if not do: return x
    b = len(x)
    contrast = torch.empty(b,1,1,1,device=x.device).uniform_(.96,1.04)
    brightness = torch.empty(b,1,1,1,device=x.device).uniform_(-.02,.02)
    return ((x-.5)*contrast + .5 + brightness).clamp(0,1)


@torch.no_grad()
def posterior_latents(net, x, stochastic=True):
    f32, f8 = net.features(x)
    mg, lvg = net.global_posterior(f8)
    zg = net.reparam(mg,lvg,sample=stochastic)
    gm = net.global_map(zg)
    context, pm, plv = net.local_prior(gm)
    dm,dlv = net.posterior_stats(torch.cat([f32,context],dim=1)).chunk(2,dim=1)
    qm = pm + dm
    qlv = (plv+dlv).clamp(-9,6)
    zl = net.reparam(qm,qlv,sample=stochastic)
    return gm.detach(), zl.detach(), dict(mg=mg,lvg=lvg,qm=qm,qlv=qlv,pm=pm,plv=plv)


@torch.no_grad()
def prior_latents(net, n, device, global_rng=None, local_rng=None):
    zg = torch.randn((n,net.global_dim),device=device,generator=global_rng)
    gm = net.global_map(zg)
    _,pm,plv = net.local_prior(gm)
    noise = torch.randn(pm.shape,device=device,dtype=pm.dtype,generator=local_rng)
    zl = pm + torch.exp(.5*plv)*noise
    return gm.detach(), zl.detach()


def reconstruction_loss(rec,x):
    # Preserve H3B original loss normalization (KL is constant when frozen).
    mse = F.mse_loss(rec.float(),x.float(),reduction='sum')/len(x)
    st = 1 - ssim(rec.float(),x.float(),data_range=1.0,size_average=True)
    edge = F.l1_loss(h3.sobel(rec.float()),h3.sobel(x.float()))
    total = mse + 48.0*st + 32.0*edge
    return total, dict(mse_sum=mse, ssim=(1-st), edge=edge)


def hinge_disc(real_logits, reconstructed, prior_logits=None):
    real = F.relu(1.-real_logits).mean()
    if prior_logits is None:
        fake = F.relu(1.+reconstructed).mean()
    else:
        fake = .5*(F.relu(1.+reconstructed).mean()+F.relu(1.+prior_logits).mean())
    return real + fake


def adversarial_gen(rec_logits, prior_logits=None):
    return -rec_logits.mean() if prior_logits is None else -0.5*(rec_logits.mean()+prior_logits.mean())


def last_weight(net): return net.out[2].weight


def adaptive_coefficient(rec_loss, adv_loss, weight, target, ceiling):
    """Measure gradients on decoder's last convolution; detach scale."""
    gr = torch.autograd.grad(rec_loss, weight, retain_graph=True)[0].detach().norm()
    ga = torch.autograd.grad(adv_loss, weight, retain_graph=True)[0].detach().norm()
    scale = torch.clamp(target*gr/(ga+1e-8),0,ceiling).detach()
    return scale, float(gr), float(ga)


def create_loaders(cfg, original):
    original.data_root=cfg.data_root; original.out_root=cfg.out_root
    original.batch_size=cfg.batch_size; original.num_workers=0
    tr,vl,loader_rng,manifest=h3.data_loaders(original)
    original_manifest=Path(cfg.out_root)/'H3B'/'split_manifest.json'
    if original_manifest.exists():
        old=json.loads(original_manifest.read_text())
        for key in ('train_paths','val_paths'):
            if key in old and old[key]!=manifest[key]:
                raise RuntimeError('H3B original split mismatch: '+key)
    return tr,vl,loader_rng,manifest


def validate(net, loader, device):
    net.eval()
    sums=dict(mse=0.,ssim=0.,rec_objective=0.,edge=0.)
    n=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device); gm,zl,_=posterior_latents(net,x,stochastic=False)
            rec=net.decoder(gm,zl)
            objective,part=reconstruction_loss(rec,x)
            vals=dict(mse=part['mse_sum']/(128*128),ssim=part['ssim'],rec_objective=objective,edge=part['edge'])
            for k,v in vals.items(): sums[k]+=float(v)*len(x)
            n+=len(x)
    return {k:v/n for k,v in sums.items()}


def put_ema(ema, net, decay):
    with torch.no_grad():
        ep=dict(ema.named_parameters()); np_=dict(net.named_parameters())
        for key,p in ep.items():
            if np_[key].requires_grad:
                p.lerp_(np_[key],1.-decay)


def fixed_samples(net, n, seed, device):
    # Paired deterministic streams; stable across batch size and checkpoints.
    gr=torch.Generator(device=device).manual_seed(seed)
    lr=torch.Generator(device=device).manual_seed(seed+314)
    with torch.no_grad():
        gm,zl=prior_latents(net,n,device,gr,lr)
        return net.decoder(gm,zl)


def plot_curves(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=folder/'history.csv'
    if not p.exists(): return
    with p.open() as f: rows=list(csv.DictReader(f))
    if not rows: return
    def f(k):return [float(v[k]) for v in rows]
    ep=f('epoch');fig,ax=plt.subplots(1,3,figsize=(13,3.5))
    ax[0].plot(ep,f('val_mse'),label='Val MSE');ax[0].set(title='Deterministic reconstruction',xlabel='Epoch');ax[0].legend()
    ax[1].plot(ep,f('g_adv'),label='G adversarial');ax[1].plot(ep,f('d_loss'),label='D hinge');ax[1].legend();ax[1].set(xlabel='Epoch',title='GAN objectives')
    ax[2].plot(ep,f('gradient_ratio_actual'),label='Estimated G adv / recon gradient')
    ax[2].legend();ax[2].set(xlabel='Epoch',title='Last-layer gradient ratio')
    fig.tight_layout();fig.savefig(folder/'curves.png',dpi=180);plt.close(fig)


def export(cfg):
    folder=Path(cfg.out_root)/cfg.experiment
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/(cfg.experiment+'_EXPORT.zip')
    tmp=out.with_name(out.name+'.tmp')
    include={'last.pth','best.pth','history.csv','history.jsonl','events.jsonl',
             'config.json','split_manifest.json','curves.png','samples_comparison.png',
             'reconstruction_comparison.png','source_sha256.json'}
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(folder.iterdir()):
            if p.is_file() and (p.name in include or p.name.startswith(('evaluation_', 'samples_comparison_', 'reconstruction_', 'epoch_'))):
                z.write(p,p.name)
        for name in ('cxr_h3_research.py','cxr_h4b_flow.py',Path(__file__).name):
            path=Path(__file__).with_name(name)
            if path.exists(): z.write(path,'code/'+path.name)
    os.replace(tmp,out)
    print('BACKUP:',out,'(download; Kaggle /working is temporary)',flush=True)
    return out


def make_best_state(cfg,net,ema,epoch,original_sha,val):
    return dict(config=asdict(cfg),epoch=epoch,source_h3b_sha=original_sha,
                model_state_dict=net.state_dict(),ema_state_dict=ema.state_dict(),val=val)


def train(cfg):
    if ssim is None:raise ImportError('Install pytorch-msssim first')
    if cfg.stop_after>cfg.epochs or cfg.stop_after<1:raise ValueError('stop-after must be 1..epochs')
    if cfg.fake_mode not in ('both','recon'):raise ValueError(cfg.fake_mode)
    if cfg.experiment=='H6A' and cfg.fake_mode!='both':raise ValueError('H6A must use both; use --experiment H6R --fake-mode recon')
    seed_all(cfg.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base,original,bsha=frozen_model(cfg,device)
    tr,vl,loader_rng,manifest=create_loaders(cfg,original)
    net=base;trainables=unlock_decoder(net)
    ema=deepcopy(net).eval().requires_grad_(False)
    disc=PatchDiscriminator(cfg.d_channels).to(device)
    og=torch.optim.AdamW(trainables,lr=cfg.g_lr,betas=(.0,.99),weight_decay=cfg.weight_decay)
    od=torch.optim.Adam(disc.parameters(),lr=cfg.d_lr,betas=(.0,.99))
    sg=torch.optim.lr_scheduler.CosineAnnealingLR(og,T_max=cfg.epochs,eta_min=cfg.g_lr*.3)
    sd=torch.optim.lr_scheduler.CosineAnnealingLR(od,T_max=cfg.epochs,eta_min=cfg.d_lr*.3)
    folder=Path(cfg.out_root)/cfg.experiment;folder.mkdir(parents=True,exist_ok=True)
    last=folder/'last.pth';start=1;best=float('inf');best_epoch=0;coeff=0.
    guard_fields=('experiment','fake_mode','seed','epochs','batch_size','g_lr','d_lr',
                  'weight_decay','adv_ratio','adv_max_weight','grad_probe_every','d_channels','ema_decay')
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=True)
        for key in guard_fields:
            if ck['config'][key]!=getattr(cfg,key):raise RuntimeError('Resume config mismatch: '+key)
        if ck['source_h3b_sha']!=bsha:raise RuntimeError('H3B source SHA mismatch')
        net.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);disc.load_state_dict(ck['disc'])
        og.load_state_dict(ck['optim_g']);od.load_state_dict(ck['optim_d'])
        sg.load_state_dict(ck['sched_g']);sd.load_state_dict(ck['sched_d'])
        loader_rng.set_state(ck['loader_rng'].cpu());torch.set_rng_state(ck['torch_rng'].cpu())
        if torch.cuda.is_available() and ck.get('cuda_rng'):
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        random.setstate(ck['python_rng']);np.random.set_state((ck['numpy_rng'][0],np.array(ck['numpy_rng'][1],dtype=np.uint32),*ck['numpy_rng'][2:]))
        start=int(ck['epoch'])+1;best=float(ck['best']);best_epoch=int(ck['best_epoch']);coeff=float(ck['coeff'])
        # Recover a committed checkpoint's last log row if the process died before log append.
        logf=folder/'history.jsonl'
        logged=json.loads(logf.read_text().splitlines()[-1])['epoch'] if logf.exists() and logf.read_text().strip() else 0
        if logged < ck['epoch'] and 'row' in ck: log_row(folder,ck['row'])
        print('RESUMED from epoch',start-1,flush=True)
    else:
        if (folder/'history.csv').exists():raise RuntimeError('History exists but last.pth is missing; choose a fresh experiment folder')
        initval=validate(ema,vl,device);best=initval['rec_objective'];best_epoch=0
        atomic_json(asdict(cfg),folder/'config.json')
        atomic_json(manifest,folder/'split_manifest.json')
        atomic_json(dict(h3b_checkpoint_sha256=bsha),folder/'source_sha256.json')
        state=make_best_state(cfg,net,ema,0,bsha,initval)
        atomic_torch(state,folder/'epoch_000.pth');atomic_torch(state,folder/'best.pth')
        print('H6 epoch 0 unchanged H3B: val MSE',f'{initval["mse"]:.6f}', 'SSIM',f'{initval["ssim"]:.4f}',flush=True)
    print(cfg.experiment,'trainable decoder',sum(p.numel() for p in trainables),'train',len(tr.dataset),
          'val',len(vl.dataset),'device',device,'mode',cfg.fake_mode,flush=True)
    last_epoch=start-1
    for epoch in range(start,cfg.stop_after+1):
        last_epoch=epoch
        tic=time.time();net.train();disc.train()
        agg={k:0. for k in ('rec','d_loss','g_adv','mse','ssim','adv_coeff','gradient_ratio_actual','d_real','d_recon','d_prior')}
        nb=0;ratio_used=cfg.adv_ratio*min(1.,epoch/3.)
        for i,(x,_) in enumerate(tr,1):
            x=x.to(device,non_blocking=True)
            gm,zl,_=posterior_latents(net,x,stochastic=True)
            rec=net.decoder(gm,zl)
            gp,zp=prior_latents(net,len(x),device)
            fakeprior=net.decoder(gp,zp)
            # D update: detach both generated distributions. Neither encoder nor decoder sees D loss.
            disc.train();disc.requires_grad_(True);od.zero_grad(set_to_none=True)
            real_logits=disc(augment_for_disc(x))
            recon_logits=disc(augment_for_disc(rec.detach()))
            prior_logits=disc(augment_for_disc(fakeprior.detach())) if cfg.fake_mode=='both' else None
            dloss=hinge_disc(real_logits,recon_logits,prior_logits)
            if not torch.isfinite(dloss):raise FloatingPointError('D loss NaN/Inf')
            dloss.backward();nn.utils.clip_grad_norm_(disc.parameters(),cfg.d_grad_clip,error_if_nonfinite=True)
            od.step()
            # G update: D weights frozen, but gradients flow THROUGH D to generated pixels.
            disc.eval();disc.requires_grad_(False);og.zero_grad(set_to_none=True)
            reclos, parts=reconstruction_loss(rec,x)
            rec_gen_logits=disc(augment_for_disc(rec))
            prior_gen_logits=disc(augment_for_disc(fakeprior)) if cfg.fake_mode=='both' else None
            adv=adversarial_gen(rec_gen_logits,prior_gen_logits)
            if i==1 or i%cfg.grad_probe_every==0:
                coef,gr,ga=adaptive_coefficient(reclos,adv,last_weight(net),ratio_used,cfg.adv_max_weight)
                coeff=float(coef)
            total=reclos+coeff*adv
            if not torch.isfinite(total):raise FloatingPointError('G loss NaN/Inf')
            total.backward();nn.utils.clip_grad_norm_(trainables,cfg.g_grad_clip,error_if_nonfinite=True);og.step()
            put_ema(ema,net,cfg.ema_decay)
            values=dict(rec=float(reclos.detach()), d_loss=float(dloss.detach()),g_adv=float(adv.detach()),
                        mse=float(parts['mse_sum'].detach())/(128*128),ssim=float(parts['ssim'].detach()),
                        adv_coeff=coeff,gradient_ratio_actual=coeff*ga/(gr+1e-8),
                        d_real=float(real_logits.detach().mean()),d_recon=float(recon_logits.detach().mean()),
                        d_prior=float(prior_logits.detach().mean()) if prior_logits is not None else float('nan'))
            for k,v in values.items():
                if math.isfinite(v):agg[k]+=v
            nb+=1
            if i%75==0:print(f'{cfg.experiment} epoch {epoch}/{cfg.epochs} batch {i}/{len(tr)} | rec {values["rec"]:.2f} D {values["d_loss"]:.2f} adv_coef {coeff:.4f}',flush=True)
        sg.step();sd.step()
        val=validate(ema,vl,device)
        if not all(math.isfinite(v) for v in val.values()):raise FloatingPointError('Validation NaN')
        improved=val['rec_objective']<best-1e-5
        if improved:best=val['rec_objective'];best_epoch=epoch
        row=dict(epoch=epoch,mode=cfg.fake_mode,**{k:agg[k]/nb for k in agg},
                 **{'val_'+k:v for k,v in val.items()},lr_g=og.param_groups[0]['lr'],
                 lr_d=od.param_groups[0]['lr'],best_recon_epoch=best_epoch,seconds=round(time.time()-tic,1))
        # stable csv keys - duplicate val_mse/val_ssim avoided by dict overwriting
        state=dict(epoch=epoch,config=asdict(cfg),source_h3b_sha=bsha,model=net.state_dict(),
                   ema=ema.state_dict(),disc=disc.state_dict(),optim_g=og.state_dict(),optim_d=od.state_dict(),
                   sched_g=sg.state_dict(),sched_d=sd.state_dict(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   python_rng=random.getstate(),numpy_rng=(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),loader_rng=loader_rng.get_state(),
                   best=best,best_epoch=best_epoch,coeff=coeff,row=row)
        atomic_torch(state,last)
        if improved:atomic_torch(make_best_state(cfg,net,ema,epoch,bsha,val),folder/'best.pth')
        if epoch%cfg.export_every==0 or epoch==cfg.epochs:
            atomic_torch(make_best_state(cfg,net,ema,epoch,bsha,val),folder/f'epoch_{epoch:03d}.pth')
            with torch.no_grad():
                s1=fixed_samples(ema,16,2026,device).cpu()
                save_image(s1,folder/f'samples_epoch_{epoch:03d}.png',nrow=4,padding=2)
        # Recover file logs only after committed last snapshot; crash between last/log repaired on resume.
        if (folder/'history.jsonl').exists():
            existing=[json.loads(s)['epoch'] for s in (folder/'history.jsonl').read_text().splitlines() if s.strip()]
            if not existing or existing[-1] < epoch:log_row(folder,row)
        else:log_row(folder,row)
        plot_curves(folder)
        print(f'{cfg.experiment} {epoch:02d}/{cfg.epochs} | train rec {row["rec"]:.2f} D {row["d_loss"]:.3f} '
              f'val MSE {val["mse"]:.5f} SSIM {val["ssim"]:.4f} | best recon epoch {best_epoch} | {row["seconds"]:.0f}s',flush=True)
        if epoch%cfg.export_every==0 or epoch==cfg.epochs:export(cfg)
    if start>cfg.stop_after:print('Already completed requested stop-after. Use a larger --stop-after with same --epochs to resume.',flush=True)
    if last_epoch < start or (last_epoch%cfg.export_every!=0 and last_epoch!=cfg.epochs): export(cfg)
    print('TRAINING STAGE DONE. Select the best generation checkpoint using validation FID/KID (not reconstruction objective alone).',flush=True)


def chosen(cfg,device,which):
    net,original,bsha=frozen_model(cfg,device)
    p=Path(cfg.out_root)/cfg.experiment
    checkpoint=p/(f'epoch_{int(which):03d}.pth' if str(which).isdigit() else which+'.pth')
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['source_h3b_sha']!=bsha:raise RuntimeError('H3B hash mismatch')
    net.load_state_dict(ck['ema_state_dict'] if 'ema_state_dict' in ck else ck['ema'])
    net.eval().requires_grad_(False)
    return net,original,ck,bsha


def inspect(cfg,which):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net,original,ck,_=chosen(cfg,device,which)
    base,_,_=frozen_model(cfg,device)
    _,val,_,_=create_loaders(cfg,original)
    out=Path(cfg.out_root)/cfg.experiment
    with torch.no_grad():
        # Make sure same z_global and z_local are used for paired base/new comparison.
        a=fixed_samples(base,16,2026,device).cpu()
        b=fixed_samples(net,16,2026,device).cpu()
        save_image(torch.cat([a,b]),out/f'samples_comparison_{which}.png',nrow=4,padding=2)
        real,_=next(iter(val));real=real[:8].to(device)
        gm,zl,_=posterior_latents(net,real,False)
        rec=net.decoder(gm,zl)
        save_image(torch.cat([real.cpu(),rec.cpu()]),out/f'reconstruction_{which}.png',nrow=8,padding=2)
        print('PAIRED IMAGE DIFFERENCE',float((a-b).abs().mean()),'checkpoint',which,flush=True)
    print('INSPECT saved',out/f'samples_comparison_{which}.png', 'and',out/f'reconstruction_{which}.png',flush=True)


def as_u8(im):
    return (im.clamp(0,1).mul(255).round().to(torch.uint8)).repeat(1,3,1,1)


def evaluate(cfg, which, split='val',n_samples=522,seed=2026):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('Use Kaggle GPU T4 for FID/KID')
    net,original,ck,_=chosen(cfg,device,which)
    base,_,_=frozen_model(cfg,device)
    if split=='test':_,_,loader,_,_=h3.data_loaders(original,use_test=True)
    elif split=='val':_,loader,_,_=create_loaders(cfg,original)
    else:raise ValueError(split)
    n=min(n_samples,len(loader.dataset))
    if n<50:raise ValueError('Need >=50 samples')
    results=[]
    for name,model in [('H3B',base),(cfg.experiment+'_'+str(which),net)]:
        fid=FrechetInceptionDistance(feature=2048,normalize=False).to(device);fid.set_dtype(torch.float64)
        kid=KernelInceptionDistance(feature=2048,subsets=20,subset_size=50,normalize=False).to(device)
        inc=InceptionScore(normalize=False,splits=5).to(device)
        gr=torch.Generator(device=device).manual_seed(seed)
        lr=torch.Generator(device=device).manual_seed(seed+314)
        count=0;gradient=0.
        with torch.inference_mode():
            for xr,_ in loader:
                if count>=n:break
                xr=xr[:n-count].to(device)
                gm,zl=prior_latents(model,len(xr),device,gr,lr)
                fake=model.decoder(gm,zl)
                if not torch.isfinite(fake).all():raise FloatingPointError('Nonfinite generated image')
                fid.update(as_u8(xr),real=True);fid.update(as_u8(fake),real=False)
                kid.update(as_u8(xr),real=True);kid.update(as_u8(fake),real=False)
                inc.update(as_u8(fake))
                gradient+=float((fake[:,:,:,1:]-fake[:,:,:,:-1]).abs().mean())*len(xr)
                count+=len(xr)
        torch.manual_seed(seed+700)  # matched KID subset sampling for both models
        km,ks=kid.compute();im,ist=inc.compute()
        row=dict(model=name,checkpoint=str(which),seed=seed,split=split,n_real=count,n_generated=count,
                 FID2048=float(fid.compute()),KID_mean=float(km),KID_std=float(ks),
                 IS_mean=float(im),IS_std=float(ist),horizontal_gradient=gradient/count)
        print('EVALUATED',row,flush=True);results.append(row)
        del fid,kid,inc
        if torch.cuda.is_available():torch.cuda.empty_cache()
    out=Path(cfg.out_root)/cfg.experiment
    atomic_json(results,out/f'evaluation_{split}_{which}_seed{seed}.json')
    with (out/f'evaluation_{split}_{which}_seed{seed}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    export(cfg)


def unit_test():
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(123)
    net=h3.CXRHVAE32()
    dec=unlock_decoder(net)
    assert any(p.requires_grad for p in net.d128.parameters())
    assert not any(p.requires_grad for p in net.g_to_map.parameters())
    assert not any(p.requires_grad for p in net.posterior_stats.parameters())
    disc=PatchDiscriminator(8)
    x=torch.rand(2,1,128,128)
    gm,zl,_=posterior_latents(net,x)
    rec=net.decoder(gm,zl)
    gp,zp=prior_latents(net,2,'cpu')
    pri=net.decoder(gp,zp)
    d=hinge_disc(disc(x),disc(rec.detach()),disc(pri.detach()))
    assert d.isfinite()
    disc.eval().requires_grad_(False)
    rr=disc(rec);pp=disc(pri)
    total,parts=reconstruction_loss(rec,x)
    adv=adversarial_gen(rr,pp)
    c,gr,ga=adaptive_coefficient(total,adv,last_weight(net),.05,20.)
    (total+c*adv).backward()
    assert rec.shape==(2,1,128,128)
    assert all(p.grad is None for p in net.prior_stats.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in dec)
    print('UNIT TEST OK | frozen prior/encoder, decoder gradients, hinge loss, paired generation',flush=True)


def check(cfg):
    if ssim is None:raise ImportError('pip install pytorch-msssim')
    seed_all(cfg.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net,old,bsha=frozen_model(cfg,device)
    tr,vl,_,manifest=create_loaders(cfg,old)
    params=unlock_decoder(net)
    x,_=next(iter(tr));x=x[:2].to(device)
    gm,zl,_=posterior_latents(net,x)
    rec=net.decoder(gm,zl)
    recon,_=reconstruction_loss(rec,x)
    recon.backward()
    assert all(p.grad is None for p in net.posterior_stats.parameters())
    assert any(p.grad is not None for p in params)
    print('CHECK OK | base H3B sha',bsha[:14],'train',len(tr.dataset),'val',len(vl.dataset),
          'decoder trainable params',sum(p.numel() for p in params),'image shape',tuple(rec.shape),
          'device',device,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['unit-test','check','train','inspect','evaluate','export'])
    p.add_argument('--experiment',default='H6A');p.add_argument('--fake-mode',choices=['both','recon'],default='both')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--stop-after',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--checkpoint',default='5')
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--n-samples',type=int,default=522)
    p.add_argument('--data-root',default='/kaggle/input');p.add_argument('--out-root',default='/kaggle/working/cxr_hvae')
    args=p.parse_args()
    cfg=Config(experiment=args.experiment,fake_mode=args.fake_mode,epochs=args.epochs,
               stop_after=args.stop_after,batch_size=args.batch_size,seed=args.seed,
               data_root=args.data_root,out_root=args.out_root)
    if args.command=='unit-test':unit_test()
    elif args.command=='check':check(cfg)
    elif args.command=='train':train(cfg)
    elif args.command=='inspect':inspect(cfg,args.checkpoint)
    elif args.command=='evaluate':evaluate(cfg,args.checkpoint,args.split,args.n_samples,args.seed)
    else:export(cfg)


if __name__=='__main__':main()

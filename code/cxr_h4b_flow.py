"""H4B: spatially coupled, conditional normalizing-flow prior over frozen H3B local latents.

Post-hoc density calibration. Encoder, decoder, H3B Gaussian prior all frozen.
The new prior is EXACTLY the H3B conditional Gaussian at initialization.
Spatial checkerboard + channel affine coupling layers use convolutional conditioners.

Commands: unit-test, check, train, inspect, evaluate, export, restore.
Requires cxr_h3_research.py and original H3B/best.pth (or Kaggle H3B backup input).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

import cxr_h3_research as h3


@dataclass
class Config:
    experiment: str = 'H4B'
    seed: int = 2026
    epochs: int = 20
    batch_size: int = 16
    flow_steps: int = 6
    hidden_channels: int = 96
    learning_rate: float = 0.00012
    min_learning_rate: float = 0.000012
    weight_decay: float = 0.0001
    patience: int = 7
    min_delta: float = 0.0001
    grad_clip: float = 1.0
    log_scale_max: float = 0.8
    export_every: int = 5
    data_root: str = '/kaggle/input'
    output_root: str = '/kaggle/working/cxr_hvae'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_h3b(output_root, input_root):
    """Restore ONLY a verified original H3B backup, without overwriting weights."""
    dest = Path(output_root) / 'H3B'
    checkpoint = dest / 'best.pth'
    if checkpoint.exists():
        return checkpoint
    for arc in sorted(Path(input_root).rglob('H3B_EXPORT.zip')):
        with zipfile.ZipFile(arc) as zf:
            if not {'best.pth', 'config.json'}.issubset(set(zf.namelist())):
                continue
            if json.loads(zf.read('config.json')).get('experiment') != 'H3B':
                continue
            dest.mkdir(parents=True, exist_ok=True)
            for fn in ('best.pth', 'config.json', 'split_manifest.json', 'history.csv', 'history.jsonl'):
                if fn in zf.namelist() and not (dest / fn).exists():
                    (dest / fn).write_bytes(zf.read(fn))
            print('H3B restored from archive:', arc, flush=True)
            return checkpoint
    for source in sorted(Path(input_root).rglob('best.pth')):
        meta = source.with_name('config.json')
        if not meta.exists():
            continue
        try:
            if json.loads(meta.read_text()).get('experiment') != 'H3B':
                continue
        except (ValueError, OSError):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for fn in ('best.pth', 'config.json', 'split_manifest.json', 'history.csv', 'history.jsonl'):
            p = source.with_name(fn)
            if p.exists() and not (dest / fn).exists():
                shutil.copy2(p, dest / fn)
        print('H3B restored from extracted files:', source, flush=True)
        return checkpoint
    raise FileNotFoundError('Cannot find H3B best.pth. Attach private H3B backup in Kaggle Input.')


def load_base(cfg, device):
    p = ensure_h3b(cfg.output_root, cfg.data_root)
    ckpt = torch.load(p, map_location='cpu', weights_only=True)
    original = h3.Config(**ckpt['config'])
    if original.experiment != 'H3B' or original.global_dim != 128 or original.spatial_channels != 8 or original.image_size != 128:
        raise ValueError('Expected original H3B 128-d global and 8x32x32 local checkpoint.')
    original.data_root = cfg.data_root
    original.out_root = cfg.output_root
    original.batch_size = cfg.batch_size
    original.num_workers = 0
    net = h3.build_model(original).to(device)
    net.load_state_dict(ckpt['model_state_dict'], strict=True)
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, original, sha256(p)


class ConditionalCoupling(nn.Module):
    """RealNVP affine coupling with a spatial/channel mask and a spatial conditioner."""
    def __init__(self, channels=8, context_channels=96, hidden=96, kind='checker', parity=0, log_scale_max=0.8):
        super().__init__()
        self.channels = channels
        self.kind = kind
        self.parity = parity
        self.log_scale_max = log_scale_max
        if kind == 'checker':
            row = torch.arange(32)[:, None]
            col = torch.arange(32)[None, :]
            mask = ((row + col + parity) % 2).float().view(1, 1, 32, 32)
        elif kind == 'channel':
            mask = torch.zeros((1, channels, 1, 1))
            mask[:, parity::2] = 1.0
        else:
            raise ValueError(kind)
        self.register_buffer('mask', mask, persistent=True)
        # Depthwise convolution couples NEIGHBORING latent sites (unlike independent GMM).
        self.conditioner = nn.Sequential(
            nn.Conv2d(channels + context_channels + 1, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        # Exact identity at initialization; a rigorous paired baseline.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def _shift_scale(self, x, context):
        mask = self.mask
        if x.shape[-2:] != (32, 32) or context.shape[1:] != (96, 32, 32):
            raise ValueError(f'Expected 8x32x32 latent and 96x32x32 context; got {x.shape}, {context.shape}')
        masked = x * mask
        # Always supply one mask channel, even for channelwise masks.
        mask_feature = mask.expand(x.shape[0], self.channels, 32, 32).mean(dim=1, keepdim=True)
        raw_shift, raw_log_scale = self.conditioner(torch.cat([masked, context, mask_feature], dim=1)).chunk(2, dim=1)
        log_scale = self.log_scale_max * torch.tanh(raw_log_scale)
        return raw_shift, log_scale

    def forward(self, x, context):
        shift, scale = self._shift_scale(x, context)
        inv_mask = 1.0 - self.mask
        y = x * self.mask + inv_mask * (x * scale.exp() + shift)
        logdet = (inv_mask * scale).sum(dim=(1, 2, 3))
        return y, logdet

    def inverse(self, y, context):
        shift, scale = self._shift_scale(y, context)
        inv_mask = 1.0 - self.mask
        x = y * self.mask + inv_mask * ((y - shift) * (-scale).exp())
        return x


class ConditionalSpatialFlow(nn.Module):
    """p(z_local|z_global) = p_N(f((z-pm)/sigma|ctx)) |det df/du| / prod(sigma)."""
    def __init__(self, channels=8, steps=6, hidden=96, log_scale_max=0.8):
        super().__init__()
        if steps < 2 or steps % 2:
            raise ValueError('Use an even number of flow steps >=2 for complementary masks')
        self.layers = nn.ModuleList([
            ConditionalCoupling(channels, 96, hidden,
                                kind=('checker' if i % 2 == 0 else 'channel'),
                                parity=(i // 2) % 2,
                                log_scale_max=log_scale_max)
            for i in range(steps)
        ])

    def transform(self, u, ctx):
        ld = u.new_zeros(u.shape[0])
        for layer in self.layers:
            u, increment = layer(u, ctx)
            ld = ld + increment
        return u, ld

    def invert(self, v, ctx):
        for layer in reversed(self.layers):
            v = layer.inverse(v, ctx)
        return v

    def log_prob(self, z, context, pm, plv, return_logdet=False):
        sigma = (0.5 * plv).exp()
        u = (z - pm) / sigma
        v, ld = self.transform(u, context)
        gaussian = -0.5 * (math.log(2 * math.pi) + v.square())
        lp = gaussian.sum(dim=(1, 2, 3)) + ld - 0.5 * plv.sum(dim=(1, 2, 3))
        return (lp, ld) if return_logdet else lp

    def sample(self, context, pm, plv, noise=None, generator=None):
        v = noise if noise is not None else torch.randn(pm.shape, device=pm.device, dtype=pm.dtype, generator=generator)
        u = self.invert(v, context)
        return pm + (0.5 * plv).exp() * u


def baseline_log_prob(z, pm, plv):
    return (-0.5 * (math.log(2 * math.pi) + plv + (z - pm).square() * torch.exp(-plv))).sum(dim=(1, 2, 3))


@torch.no_grad()
def posterior_samples(base, x, rng=None):
    f32, f8 = base.features(x)
    mg, lvg = base.global_posterior(f8)
    zg = base.reparam(mg, lvg, sample=True, generator=rng)
    global_map = base.global_map(zg)
    context, pm, plv = base.local_prior(global_map)
    dm, dlv = base.posterior_stats(torch.cat((f32, context), dim=1)).chunk(2, dim=1)
    qm, qlv = pm + dm, (plv + dlv).clamp(-9, 6)
    zl = base.reparam(qm, qlv, sample=True, generator=rng)
    return context.detach(), pm.detach(), plv.detach(), zl.detach(), qm.detach(), qlv.detach()


def check_grads(flow, clip=1.0):
    return torch.nn.utils.clip_grad_norm_(flow.parameters(), clip, error_if_nonfinite=True)


def epoch_pass(flow, base, loader, device, optimizer=None, generator=None, progress_every=70, clip=1.0):
    training = optimizer is not None
    flow.train(training)
    counts = 0
    sums = dict(nll=0.0, h3b_nll=0.0, kl_mc=0.0, logdet_per_site=0.0)
    for j, (images, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        context, pm, plv, zl, qm, qlv = posterior_samples(base, images, generator)
        with torch.set_grad_enabled(training):
            lp, ld = flow.log_prob(zl, context, pm, plv, return_logdet=True)
            baseline_lp = baseline_log_prob(zl, pm, plv)
            nll = -lp.mean() / 1024.0
            # Diagnostic q-logp Monte Carlo: NOT an exact analytic KL.
            qlp = baseline_log_prob(zl, qm, qlv)
            kl_mc = (qlp - lp).mean()
            if not torch.isfinite(nll) or not torch.isfinite(kl_mc):
                raise FloatingPointError('Non-finite H4B training/validation likelihood')
            if training:
                optimizer.zero_grad(set_to_none=True)
                nll.backward()
                check_grads(flow, clip)
                optimizer.step()
        b = images.shape[0]
        counts += b
        for key, v in [('nll', nll), ('h3b_nll', -baseline_lp.mean()/1024.0),
                       ('kl_mc', kl_mc), ('logdet_per_site', ld.mean()/1024.0)]:
            sums[key] += float(v.detach()) * b
        if progress_every and j % progress_every == 0:
            print(('Train' if training else 'Val'), j, '/', len(loader), 'NLL=', round(float(nll.detach()), 5), flush=True)
    if counts == 0:
        raise ValueError('Empty loader')
    return {k:v/counts for k, v in sums.items()}


def atomic_json(obj, file):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + '.tmp')
    with open(temp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, file)


def atomic_torch(obj, file):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, file)


def append_jsonl(file, obj):
    with open(file, 'a') as f:
        f.write(json.dumps(obj) + '\n')
        f.flush(); os.fsync(f.fileno())


def append_csv(file, obj):
    exists = Path(file).exists()
    with open(file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(obj))
        if not exists:
            writer.writeheader()
        writer.writerow(obj)
        f.flush(); os.fsync(f.fileno())


def zip_export(cfg):
    run = Path(cfg.output_root) / cfg.experiment
    run.mkdir(parents=True, exist_ok=True)
    out = run / 'H4B_EXPORT.zip'
    stage = out.with_suffix('.zip.tmp')
    names = ['best.pth', 'last.pth', 'config.json', 'environment.json', 'split_manifest.json',
             'history.jsonl', 'history.csv', 'events.jsonl', 'curves.png',
             'samples_comparison.png', 'evaluation_val_seed2026.csv', 'evaluation_val_seed2026.json']
    with zipfile.ZipFile(stage, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name in names:
            if (run / name).exists():
                z.write(run / name, arcname=name)
        for extra in sorted(run.glob('evaluation_*')):
            if extra.name not in names and extra.is_file():
                z.write(extra, arcname=extra.name)
        z.write(Path(__file__), arcname='cxr_h4b_flow.py')
        z.write(Path(h3.__file__), arcname='cxr_h3_research.py')
    os.replace(stage, out)
    print('BACKUP:', out, '(download it; /kaggle/working is temporary)', flush=True)
    return out


def plot_history(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fn = run / 'history.jsonl'
    if not fn.exists(): return
    rows = [json.loads(line) for line in fn.read_text().splitlines() if line.strip()]
    if not rows: return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ('train_nll', 'val_nll', 'val_h3b_nll'):
        vals = [r.get(key, np.nan) for r in rows]
        axes[0].plot([r['epoch'] for r in rows], vals, label=key)
    axes[0].set(xlabel='Epoch', ylabel='NLL / spatial site', title='Conditional prior density fit')
    axes[0].legend()
    axes[1].plot([r['epoch'] for r in rows], [r['val_kl_mc'] for r in rows], label='MC local KL')
    axes[1].plot([r['epoch'] for r in rows], [r['val_logdet_per_site'] for r in rows], label='log-det / site')
    axes[1].set(xlabel='Epoch', title='Posterior-prior gap / Flow volume')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(run / 'curves.png', dpi=170); plt.close(fig)


def setup(cfg, device):
    print('Loading frozen H3B and original dataset...', flush=True)
    base, base_cfg, base_sha = load_base(cfg, device)
    tr, val, loader_rng, manifest = h3.data_loaders(base_cfg)
    flow = ConditionalSpatialFlow(8, cfg.flow_steps, cfg.hidden_channels, cfg.log_scale_max).to(device)
    # Make sure validation split is unchanged from the original H3B split file, if available.
    split_path = Path(cfg.output_root) / 'H3B' / 'split_manifest.json'
    if split_path.exists():
        known = json.loads(split_path.read_text())
        for k in ('train_paths', 'val_paths'):
            if known.get(k) != manifest.get(k):
                raise ValueError('Original H3B split differs from current dataset split: ' + k)
    return base, base_cfg, base_sha, flow, tr, val, loader_rng, manifest


def make_weight_state(cfg, base_sha, flow, epoch, score):
    return dict(epoch=epoch, config=asdict(cfg), h3b_sha256=base_sha,
                flow=flow.state_dict(), val_nll=score)


def train(cfg):
    seed_all(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Training needs Kaggle GPU. Select Accelerator: T4.')
    run = Path(cfg.output_root) / cfg.experiment
    run.mkdir(parents=True, exist_ok=True)
    base, base_cfg, sha, flow, tr, val, loader_rng, manifest = setup(cfg, device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_learning_rate)
    start = 1; best = float('inf'); best_epoch = -1; bad = 0
    ckpt_path = run / 'last.pth'
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if ck['config'] != asdict(cfg) or ck['h3b_sha256'] != sha:
            raise ValueError('Refusing resume: configuration or H3B checkpoint SHA mismatch')
        flow.load_state_dict(ck['flow'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng'])
        if device.type == 'cuda' and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        loader_rng.set_state(ck['loader_rng'])
        start = ck['epoch'] + 1; best = ck['best']; best_epoch = ck['best_epoch']; bad = ck['bad']
        rows = [json.loads(x) for x in (run/'history.jsonl').read_text().splitlines() if x.strip()] if (run/'history.jsonl').exists() else []
        if ck['row'] is not None and ck['epoch'] not in {r['epoch'] for r in rows}:
            append_jsonl(run/'history.jsonl', ck['row']); append_csv(run/'history.csv', ck['row'])
        print('RESUME after epoch', start - 1, flush=True)
    else:
        if (run/'history.jsonl').exists() or (run/'best.pth').exists():
            raise RuntimeError('Experiment files exist without last.pth; refusing to overwrite')
        atomic_json(asdict(cfg), run/'config.json')
        atomic_json(manifest, run/'split_manifest.json')
        atomic_json({'torch_version':torch.__version__, 'device':str(device), 'h3b_sha256':sha,
                     'flow_trainable_parameters':sum(p.numel() for p in flow.parameters()),
                     'model':'identity-initialized conditional spatial RealNVP prior',
                     'reconstruction_weights_frozen':True,
                     'validation_target':'MC negative log p_flow(z_l | z_g), posterior-sampled z_g; NOT image FID',
                     'prior_mismatch_warning':'Global prior remains N(0,I), training conditions on q(z_g|x)'},
                    run/'environment.json')
        # Explicit epoch-0 exact identity baseline selected if flow cannot beat H3B NLL.
        val_rng = torch.Generator(device=device).manual_seed(cfg.seed + 121212)
        init_val = epoch_pass(flow, base, val, device, generator=val_rng, progress_every=0)
        best = init_val['nll']; best_epoch = 0
        atomic_torch(make_weight_state(cfg,sha,flow,0,best),run/'best.pth')
        # Epoch-0 resumable state guards against interruption during the first epoch.
        atomic_torch(dict(**make_weight_state(cfg,sha,flow,0,best),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                          loader_rng=loader_rng.get_state(), best=best, best_epoch=0, bad=0, row=None),
                     ckpt_path)
        append_jsonl(run/'events.jsonl', {'event':'identity_baseline', 'val_nll':best,
                                          'h3b_nll':init_val['h3b_nll'], 'at':time.time()})
        print('H4B identity baseline | val NLL =',round(best,6),
              '| H3B NLL =',round(init_val['h3b_nll'],6),flush=True)
    append_jsonl(run/'events.jsonl', {'event':'start_or_resume','epoch':start,'at':time.time()})
    print('H4B spatial flow | trainable:',sum(p.numel() for p in flow.parameters()),
          '| train:',len(tr.dataset),'val:',len(val.dataset),'| device:',device,flush=True)
    for epoch in range(start, cfg.epochs + 1):
        tick = time.monotonic()
        train_m = epoch_pass(flow, base, tr, device, optimizer=optimizer, clip=cfg.grad_clip)
        val_rng = torch.Generator(device=device).manual_seed(cfg.seed + 121212)
        val_m = epoch_pass(flow, base, val, device, generator=val_rng, progress_every=0)
        if not all(math.isfinite(v) for v in [*train_m.values(), *val_m.values()]):
            raise FloatingPointError('Non-finite epoch diagnostics')
        scheduler.step()
        improved = val_m['nll'] < best - cfg.min_delta
        if improved:
            best = val_m['nll']; best_epoch = epoch; bad = 0
        else:
            bad += 1
        row = dict(epoch=epoch, train_nll=train_m['nll'], val_nll=val_m['nll'],
                   val_h3b_nll=val_m['h3b_nll'], train_kl_mc=train_m['kl_mc'],
                   val_kl_mc=val_m['kl_mc'], val_logdet_per_site=val_m['logdet_per_site'],
                   lr=optimizer.param_groups[0]['lr'], best_epoch=best_epoch,
                   seconds=round(time.monotonic()-tick,2))
        ck = dict(**make_weight_state(cfg,sha,flow,epoch,val_m['nll']),
                  optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                  torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                  loader_rng=loader_rng.get_state(), best=best, best_epoch=best_epoch, bad=bad, row=row)
        atomic_torch(ck, ckpt_path)
        if improved:
            atomic_torch(make_weight_state(cfg,sha,flow,epoch,best),run/'best.pth')
        append_jsonl(run/'history.jsonl',row)
        append_csv(run/'history.csv',row)
        plot_history(run)
        print(f"H4B {epoch:02d}/{cfg.epochs} | train_nll={train_m['nll']:.5f} | val_nll={val_m['nll']:.5f} "
              f"| H3B_nll={val_m['h3b_nll']:.5f} | MC_KL={val_m['kl_mc']:.2f} "
              f"| best={best_epoch} | seconds={row['seconds']:.0f}",flush=True)
        if epoch % cfg.export_every == 0 or epoch == cfg.epochs:
            zip_export(cfg)
        if bad >= cfg.patience:
            append_jsonl(run/'events.jsonl', {'event':'early_stop','epoch':epoch,'at':time.time()})
            print('EARLY STOP',epoch,flush=True)
            break
    print('TRAINING FINISHED | best epoch:',best_epoch, 'best validation NLL:',best,flush=True)
    zip_export(cfg)


def load_best(cfg, device):
    run = Path(cfg.output_root)/cfg.experiment
    best = torch.load(run/'best.pth', map_location='cpu', weights_only=True)
    selected = Config(**best['config'])
    base, bc, sha, flow, tr, val, lrng, manifest = setup(selected, device)
    if sha != best['h3b_sha256']:
        raise ValueError('H3B checkpoint mismatch vs H4B best.pth')
    flow.load_state_dict(best['flow'], strict=True)
    flow.eval()
    return base, bc, flow, val, best, selected


def batch_gradient(image):
    return 0.5*((image[:,:,:,1:]-image[:,:,:,:-1]).abs().mean() +
                (image[:,:,1:,:]-image[:,:,:-1,:]).abs().mean())


@torch.inference_mode()
def paired_generate(base, flow, count, seed, device, batch_size=16):
    rng_global = torch.Generator(device=device).manual_seed(seed)
    rng_local = torch.Generator(device=device).manual_seed(seed+314)
    a,b = [],[]
    for left in range(0,count,batch_size):
        n=min(batch_size,count-left)
        zg=torch.randn((n,base.global_dim),generator=rng_global,device=device)
        zbase=torch.randn((n,base.spatial_channels,32,32),generator=rng_local,device=device)
        gm=base.global_map(zg)
        ctx,pm,plv=base.local_prior(gm)
        z_gauss=pm+(0.5*plv).exp()*zbase
        z_flow=flow.sample(ctx,pm,plv,noise=zbase)
        a.append(base.decoder(gm,z_gauss).cpu())
        b.append(base.decoder(gm,z_flow).cpu())
    return torch.cat(a),torch.cat(b)


def inspect(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base, bc, flow, val, best, selected = load_best(cfg,device)
    a,b = paired_generate(base,flow,16,selected.seed,device)
    run=Path(selected.output_root)/selected.experiment
    save_image(torch.cat((a,b),dim=0),run/'samples_comparison.png',nrow=4,padding=2)
    metadata = dict(h3b_gradient=float(batch_gradient(a)),h4b_gradient=float(batch_gradient(b)),
                    mean_absolute_paired_diff=float((a-b).abs().mean()),best_epoch=best['epoch'])
    atomic_json(metadata,run/'sample_diagnostics.json')
    print('Saved:',run/'samples_comparison.png','| top 4 rows H3B; bottom 4 rows H4B',flush=True)
    print('Diagnostics:',metadata,flush=True)
    zip_export(selected)


def u8(x):
    return (x.clamp(0,1)*255).round().to(torch.uint8).repeat(1,3,1,1)


def evaluate(cfg,split='val',n=522,seed=2026):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('FID/KID evaluation requires a GPU')
    base,bc,flow,val,best,selected=load_best(cfg,device)
    if split=='val':
        loader=val
    else:
        _,_,loader,_,_=h3.data_loaders(bc,use_test=True)
    n=min(n,len(loader.dataset))
    if n<50: raise ValueError('Need >=50 samples')
    run=Path(selected.output_root)/selected.experiment
    rows=[]
    for name in ('H3B','H4B'):
        fid=FrechetInceptionDistance(feature=2048,normalize=False).to(device)
        fid.set_dtype(torch.float64)
        kid=KernelInceptionDistance(feature=2048,subsets=20,subset_size=50,normalize=False).to(device)
        inc=InceptionScore(normalize=False,splits=5).to(device)
        gglobal=torch.Generator(device=device).manual_seed(seed)
        glocal=torch.Generator(device=device).manual_seed(seed+314)
        seen=0; gradient_total=0.0
        with torch.inference_mode():
            for real,_ in loader:
                if seen>=n: break
                real=real[:n-seen].to(device)
                b=real.shape[0]
                zg=torch.randn((b,base.global_dim),device=device,generator=gglobal)
                noise=torch.randn((b,base.spatial_channels,32,32),device=device,generator=glocal)
                gmap=base.global_map(zg)
                ctx,pm,plv=base.local_prior(gmap)
                local=pm+torch.exp(0.5*plv)*noise if name=='H3B' else flow.sample(ctx,pm,plv,noise=noise)
                fake=base.decoder(gmap,local)
                if not torch.isfinite(fake).all(): raise FloatingPointError('Nonfinite image')
                fid.update(u8(real),real=True);fid.update(u8(fake),real=False)
                kid.update(u8(real),real=True);kid.update(u8(fake),real=False)
                inc.update(u8(fake))
                gradient_total += float(batch_gradient(fake))*b
                seen+=b
        km,ks=kid.compute();im,ist=inc.compute()
        row=dict(model=name,split=split,seed=seed,n_real=seen,n_generated=seen,
                 FID2048=float(fid.compute()),KID_mean=float(km),KID_std=float(ks),
                 IS_mean=float(im),IS_std=float(ist),gradient_energy=gradient_total/seen)
        rows.append(row)
        print('EVALUATED',row,flush=True)
        del fid,kid,inc
        torch.cuda.empty_cache()
    csv_file=run/f'evaluation_{split}_seed{seed}.csv'
    with open(csv_file,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json({'protocol':'522-val/624-test original split, grayscale repeated to RGB uint8, Inception v3, paired global/epsilon RNG',
                 'warning':'ImageNet features are not a clinical fidelity metric; small-sample FID biased',
                 'results':rows},run/f'evaluation_{split}_seed{seed}.json')
    zip_export(selected)
    return rows


def restore_archive(file,output_root):
    dst=Path(output_root)/'H4B'
    with zipfile.ZipFile(file) as z:
        if not {'best.pth','config.json'}.issubset(set(z.namelist())):
            raise ValueError('Not an H4B EXPORT ZIP')
        conf=json.loads(z.read('config.json'))
        if conf.get('experiment')!='H4B': raise ValueError('Wrong experiment')
        dst.mkdir(parents=True,exist_ok=True)
        for entry in z.infolist():
            if '/' in entry.filename or entry.filename.startswith('.') or entry.is_dir():
                continue
            if not (dst/entry.filename).exists():
                (dst/entry.filename).write_bytes(z.read(entry.filename))
    print('Restored to:', dst, '(H3B best.pth is required separately)',flush=True)


def unit_test():
    torch.manual_seed(3)
    # Uses small batch of 8x32x32 tensors, without dataset or checkpoint.
    flow=ConditionalSpatialFlow(8,steps=4,hidden=16)
    x=torch.randn(2,8,32,32)
    c=torch.randn(2,96,32,32)
    pm=torch.randn_like(x)*0.1
    lv=torch.randn_like(x)*0.1
    y,logdet=flow.transform(x,c)
    xr=flow.invert(y,c)
    inv=float((xr-x).abs().max().detach())
    baseline=baseline_log_prob(x,pm,lv)
    err=float((flow.log_prob(x,c,pm,lv)-baseline).abs().max().detach())
    print('identity max |log_prob_flow - Gaussian|=',err,flush=True)
    print('invertibility max error=',inv,flush=True)
    if inv>1e-5 or err>0.003:
        raise AssertionError('Flow identity/inverse incorrect')
    lp=flow.log_prob(x,c,pm,lv)
    (-lp.mean()/1024).backward()
    if not all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None):
        raise AssertionError('Nonfinite gradient')
    print('UNIT TEST OK: exact identity, inverse, log-density and gradients',flush=True)


def check(cfg):
    unit_test()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base,bc,sha,flow,tr,val,rng,manifest=setup(cfg,device)
    real,_=next(iter(tr));real=real[:2].to(device)
    ctx,pm,plv,z,qm,qlv=posterior_samples(base,real)
    nll=-flow.log_prob(z,ctx,pm,plv).mean()/1024.0
    nll.backward()
    print('CHECK OK | device:',device,'| image:',tuple(real.shape),'| latent:',tuple(z.shape),
          '| train/val:',len(tr.dataset),len(val.dataset),'| baseline SHA:',sha[:12],
          '| finite:',bool(torch.isfinite(nll)),flush=True)


def main():
    parser=argparse.ArgumentParser(description='H4B spatially coupled conditional RealNVP prior over frozen H3B')
    parser.add_argument('command',choices=['unit-test','check','train','inspect','evaluate','export','restore'])
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--steps',type=int,default=6)
    parser.add_argument('--hidden',type=int,default=96)
    parser.add_argument('--lr',type=float,default=1.2e-4)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--n-samples',type=int,default=522)
    parser.add_argument('--data-root',default='/kaggle/input')
    parser.add_argument('--out-root',default='/kaggle/working/cxr_hvae')
    parser.add_argument('--zip-path',default=None)
    args=parser.parse_args()
    cfg=Config(epochs=args.epochs,batch_size=args.batch_size,flow_steps=args.steps,
               hidden_channels=args.hidden,learning_rate=args.lr,seed=args.seed,
               data_root=args.data_root,output_root=args.out_root)
    if args.command=='unit-test':unit_test()
    elif args.command=='check':check(cfg)
    elif args.command=='train':train(cfg)
    elif args.command=='inspect':inspect(cfg)
    elif args.command=='evaluate':evaluate(cfg,args.split,args.n_samples,args.seed)
    elif args.command=='export':zip_export(cfg)
    elif args.command=='restore':
        if not args.zip_path: raise ValueError('--zip-path required')
        restore_archive(args.zip_path,args.out_root)


if __name__=='__main__':
    main()

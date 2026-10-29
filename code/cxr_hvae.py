"""CXR-HVAE: a reproducible hierarchical chest X-ray VAE experiment.

Run within Kaggle after attaching the Kaggle chest X-ray pneumonia dataset:
  python cxr_hvae.py check
  python cxr_hvae.py train --experiment H1 --epochs 50
  python cxr_hvae.py train --experiment H2 --epochs 50
  python cxr_hvae.py inspect --experiment H2
  python cxr_hvae.py evaluate --experiment H2  # held-out TEST; final reporting only
  python cxr_hvae.py export --experiment H2

Outputs are written to /kaggle/working/cxr_hvae by default.
IMPORTANT: Files in /kaggle/working are NOT durable across all Kaggle sessions.
Download exports after each experiment, and save a Kaggle Version with outputs.
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None


EXPERIMENTS = {
    "F0": {"spatial": False, "objective": "mse", "description": "Global-only residual VAE (architectural control)"},
    "H1": {"spatial": True,  "objective": "mse", "description": "Hierarchical VAE, reconstruction MSE"},
    "H2": {"spatial": True,  "objective": "hybrid", "description": "Hierarchical VAE, SSIM/edge-augmented reconstruction"},
}


@dataclass
class Config:
    experiment: str = "H2"
    seed: int = 2026
    image_size: int = 128
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.00015
    min_learning_rate: float = 0.00002
    weight_decay: float = 0.0001
    warmup_epochs: int = 10
    beta_target: float = 1.0
    ssim_weight: float = 48.0
    edge_weight: float = 32.0
    global_dim: int = 128
    spatial_channels: int = 8
    val_ratio: float = 0.1
    num_workers: int = 0
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    patience: int = 16
    save_samples_every: int = 5
    export_every: int = 5
    out_root: str = "/kaggle/working/cxr_hvae"
    data_root: str = "/kaggle/input"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def gn(channels):
    g = min(8, channels)
    while channels % g:
        g -= 1
    return nn.GroupNorm(g, channels)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            gn(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            gn(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + 0.5 * self.net(x)


class Down(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1), Residual(co))

    def forward(self, x):
        return self.seq(x)


class Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 3, padding=1)
        self.res = Residual(co)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.res(self.conv(x))


class CXRHVAE(nn.Module):
    """Two-level conditional Gaussian VAE; no deterministic encoder-to-decoder skip.

    p(z_g)=N(0,I), p(z_l|z_g) learned; q(z_g|x) and q(z_l|x,z_g).
    Sampling uses both PRIORS; reconstruction uses the two posteriors.
    """
    def __init__(self, global_dim=128, spatial_channels=8, use_spatial=True):
        super().__init__()
        self.global_dim = global_dim
        self.spatial_channels = spatial_channels
        self.use_spatial = use_spatial

        self.stem = nn.Conv2d(1, 32, 3, padding=1)
        self.e64 = Down(32, 48)
        self.e32 = Down(48, 80)
        self.e16 = Down(80, 128)
        self.e8 = Down(128, 192)
        self.e8_res = Residual(192)
        self.g_stats = nn.Linear(192, 2 * global_dim)

        self.g_to_map = nn.Linear(global_dim, 192 * 8 * 8)
        if use_spatial:
            self.prior_context = nn.Sequential(
                nn.Conv2d(192, 96, 3, padding=1), gn(96), nn.SiLU()
            )
            self.prior_stats = nn.Conv2d(96, 2 * spatial_channels, 1)
            self.posterior_stats = nn.Sequential(
                nn.Conv2d(128 + 96, 96, 3, padding=1),
                gn(96), nn.SiLU(), nn.Conv2d(96, 2 * spatial_channels, 1)
            )
            self.local_to_dec = nn.Conv2d(spatial_channels, 160, 1)
            for layer in (self.prior_stats, self.posterior_stats[-1]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.d8_res = Residual(192)
        self.d16 = Up(192, 160)
        self.d16_res = Residual(160)
        self.d32 = Up(160, 128)
        self.d64 = Up(128, 80)
        self.d128 = Up(80, 48)
        self.out = nn.Sequential(gn(48), nn.SiLU(), nn.Conv2d(48, 1, 3, padding=1), nn.Sigmoid())

    @staticmethod
    def reparam(mu, lv, sample=True, generator=None):
        if not sample:
            return mu
        eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        return mu + torch.exp(0.5 * lv) * eps

    def features(self, x):
        h = self.stem(x)
        h = self.e64(h)
        h = self.e32(h)
        f16 = self.e16(h)
        f8 = self.e8_res(self.e8(f16))
        return f16, f8

    def global_posterior(self, f8):
        stats = self.g_stats(F.adaptive_avg_pool2d(f8, 1).flatten(1))
        mu, lv = stats.chunk(2, dim=1)
        return mu, lv.clamp(-9, 6)

    def global_map(self, zg):
        return self.g_to_map(zg).reshape(-1, 192, 8, 8)

    def local_prior(self, global_map):
        context = self.prior_context(F.interpolate(global_map, scale_factor=2,
                                                    mode="bilinear", align_corners=False))
        pm, plv = self.prior_stats(context).chunk(2, dim=1)
        return context, pm, plv.clamp(-7, 5)

    def decoder(self, global_map, zl=None):
        h = self.d16(self.d8_res(global_map))
        if self.use_spatial:
            if zl is None:
                raise ValueError("Hierarchical decoder requires a spatial latent")
            h = h + self.local_to_dec(zl)
        h = self.d16_res(h)
        return self.out(self.d128(self.d64(self.d32(h))))

    def forward(self, x, stochastic=True):
        f16, f8 = self.features(x)
        mg, lvg = self.global_posterior(f8)
        zg = self.reparam(mg, lvg, sample=stochastic)
        gmap = self.global_map(zg)
        state = {"mg": mg, "lvg": lvg}
        if self.use_spatial:
            context, pm, plv = self.local_prior(gmap)
            dm, dlv = self.posterior_stats(torch.cat([f16, context], dim=1)).chunk(2, dim=1)
            qm = pm + dm
            qlv = (plv + dlv).clamp(-9, 6)
            zl = self.reparam(qm, qlv, sample=stochastic)
            state.update(qm=qm, qlv=qlv, pm=pm, plv=plv)
        else:
            zl = None
        return self.decoder(gmap, zl), state

    @torch.no_grad()
    def generate(self, batch_size, generator=None, device=None):
        device = device or next(self.parameters()).device
        zg = torch.randn((batch_size, self.global_dim), device=device, generator=generator)
        gmap = self.global_map(zg)
        if self.use_spatial:
            _, pm, plv = self.local_prior(gmap)
            zl = self.reparam(pm, plv, sample=True, generator=generator)
        else:
            zl = None
        return self.decoder(gmap, zl)


def kl_diag(qm, qlv, pm=None, plv=None):
    if pm is None:
        pm = torch.zeros_like(qm)
        plv = torch.zeros_like(qlv)
    # Computation is FP32 and log variances are bounded above.
    var_ratio = torch.exp(qlv - plv)
    mean_term = (qm - pm).square() * torch.exp(-plv)
    kl = 0.5 * (plv - qlv + var_ratio + mean_term - 1.0)
    return kl.flatten(1).sum(dim=1).mean()


def sobel(x):
    k = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    kx = k.reshape(1, 1, 3, 3)
    return torch.cat((F.conv2d(x, kx, padding=1),
                      F.conv2d(x, kx.transpose(-1, -2), padding=1)), dim=1)


def loss_terms(rec, x, latents, beta, objective, cfg):
    mse_sum = F.mse_loss(rec.float(), x.float(), reduction="sum") / x.size(0)
    kl_g = kl_diag(latents["mg"], latents["lvg"])
    if "qm" in latents:
        kl_l = kl_diag(latents["qm"], latents["qlv"], latents["pm"], latents["plv"])
    else:
        kl_l = mse_sum.new_zeros(())
    if objective == "hybrid":
        if ssim is None:
            raise ImportError("Run pip install pytorch-msssim before hybrid training")
        structure = 1 - ssim(rec.float(), x.float(), data_range=1.0, size_average=True)
        edge = F.l1_loss(sobel(rec.float()), sobel(x.float()))
    else:
        structure, edge = mse_sum.new_zeros(()), mse_sum.new_zeros(())
    total = mse_sum + beta * (kl_g + kl_l) + cfg.ssim_weight * structure + cfg.edge_weight * edge
    return {"total": total, "mse_sum": mse_sum, "kl_global": kl_g,
            "kl_spatial": kl_l, "ssim_penalty_raw": structure, "edge_l1": edge}


def find_dataset(root):
    root = Path(root)
    candidates = list(root.glob("**/chest_xray/train"))
    candidates += list(root.glob("**/train"))
    candidates = [p for p in candidates if (p / "NORMAL").is_dir() and (p / "PNEUMONIA").is_dir()
                  and (p.parent / "test").is_dir()]
    if not candidates:
        raise FileNotFoundError("Chest X-ray train/test not found under " + str(root)
                                + ". Attach paultimothymooney/chest-xray-pneumonia via Add Input.")
    return sorted(set(candidates), key=lambda p: len(str(p)))[0].parent


def data_loaders(cfg, use_test=False):
    base = find_dataset(cfg.data_root)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])
    full = datasets.ImageFolder(str(base / "train"), transform=transform)
    rng = np.random.default_rng(cfg.seed)
    train_ids, val_ids = [], []
    labels = np.asarray(full.targets)
    for cls in np.unique(labels):
        ids = np.where(labels == cls)[0]
        rng.shuffle(ids)
        count = max(1, int(round(len(ids) * cfg.val_ratio)))
        val_ids.extend(ids[:count].tolist())
        train_ids.extend(ids[count:].tolist())
    train_ids.sort()
    val_ids.sort()
    manifest = {"data_root": str(base), "seed": cfg.seed, "val_ratio": cfg.val_ratio,
                "train_count": len(train_ids), "val_count": len(val_ids),
                "train_paths": [str(Path(full.samples[i][0]).relative_to(base)) for i in train_ids],
                "val_paths": [str(Path(full.samples[i][0]).relative_to(base)) for i in val_ids]}
    gen = torch.Generator().manual_seed(cfg.seed + 333)
    train_loader = DataLoader(Subset(full, train_ids), batch_size=cfg.batch_size,
                              shuffle=True, generator=gen, num_workers=cfg.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(full, val_ids), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=torch.cuda.is_available())
    if not use_test:
        return train_loader, val_loader, gen, manifest
    test_set = datasets.ImageFolder(str(base / "test"), transform=transform)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader, gen, manifest


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log_csv(path, row):
    # This is the append-only experimental record, updated after EVERY epoch.
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def evaluate_recon(net, loader, device, cfg, objective, beta=1.0):
    net.eval()
    sums = {"total": 0., "mse_sum": 0., "kl_global": 0., "kl_spatial": 0.,
            "ssim_penalty_raw": 0., "edge_l1": 0., "mse_pixel": 0., "ssim": 0.}
    n = 0
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            rec, state = net(x, stochastic=False)
            vals = loss_terms(rec, x, state, beta, objective, cfg)
            vals["mse_pixel"] = F.mse_loss(rec, x, reduction="mean")
            if ssim is not None:
                vals["ssim"] = ssim(rec, x, data_range=1.0, size_average=True)
            else:
                vals["ssim"] = rec.new_tensor(float("nan"))
            if not torch.isfinite(torch.stack([v for k, v in vals.items() if k != "ssim"])).all():
                raise FloatingPointError("Non-finite validation values")
            bs = x.size(0)
            for key, value in vals.items():
                sums[key] += float(value.item()) * bs
            n += bs
    return {k: v / n for k, v in sums.items()}


def plot_logs(history_path, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not Path(history_path).exists():
        return
    rows = [json.loads(line) for line in Path(history_path).read_text().splitlines() if line.strip()]
    if not rows:
        return
    x = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, [r["train_total"] for r in rows], label="Train")
    axes[0].plot(x, [r["val_total"] for r in rows], label="Val")
    axes[0].set(title="Optimization objective", xlabel="Epoch", ylabel="Loss per image")
    axes[0].legend()
    axes[1].plot(x, [r["val_mse_pixel"] for r in rows], label="Val MSE")
    axes[1].set(title="Held-out validation reconstruction", xlabel="Epoch", ylabel="Pixel MSE")
    ax2 = axes[1].twinx()
    ax2.plot(x, [r["val_ssim"] for r in rows], color="tab:orange", label="Val SSIM")
    ax2.set_ylabel("SSIM")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


@torch.no_grad()
def sample_grid(net, cfg, dest, device, n=16):
    net.eval()
    gen = torch.Generator(device=device).manual_seed(2026)
    images = net.generate(n, generator=gen, device=device)
    if not torch.isfinite(images).all():
        raise FloatingPointError("Generated samples contain NaN/Inf")
    save_image(images.float().cpu(), dest, nrow=4, padding=2)


def zip_experiment(directory, script_path, filename=None, include_last=True):
    directory = Path(directory)
    filename = Path(filename or directory / f"{directory.name}_EXPORT.zip")
    select = ["config.json", "split_manifest.json", "environment.json", "history.jsonl",
              "history.csv", "events.jsonl", "best.pth", "curves.png", "samples_best.png",
              "evaluation.json", "evaluation_samples.png", "reconstruction_comparison.png"]
    if include_last:
        select.append("last.pth")
    tmp = filename.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in select:
            p = directory / name
            if p.exists():
                zf.write(p, arcname=name)
        if Path(script_path).exists():
            zf.write(script_path, arcname="cxr_hvae.py")
    os.replace(tmp, filename)
    print("EXPORT CREATED:", filename)
    print("Download this ZIP or save a Kaggle Version with outputs; /kaggle/working is temporary.")
    return filename


def train(cfg, fresh=False):
    if cfg.image_size != 128:
        raise ValueError("This implementation requires --image-size 128.")
    if ssim is None:
        raise ImportError("Please install pytorch-msssim before training so validation SSIM can be recorded.")
    exp = EXPERIMENTS[cfg.experiment]
    run = Path(cfg.out_root) / cfg.experiment
    run.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, vl, loader_rng, manifest = data_loaders(cfg)
    model = CXRHVAE(cfg.global_dim, cfg.spatial_channels, exp["spatial"]).to(device)
    ema = deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                           eta_min=cfg.min_learning_rate)
    start = 1
    best = float("inf")
    best_epoch = 0
    bad_epochs = 0
    last_path = run / "last.pth"
    old_cfg_path = run / "config.json"
    if last_path.exists() and not fresh:
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        old_cfg = ck["config"]
        for key in ("experiment", "seed", "batch_size", "epochs", "global_dim", "spatial_channels",
                    "learning_rate", "ssim_weight", "edge_weight", "warmup_epochs", "val_ratio"):
            if old_cfg[key] != getattr(cfg, key):
                raise ValueError(f"Resume config mismatch for {key}: {old_cfg[key]} != {getattr(cfg, key)}")
        model.load_state_dict(ck["model_state_dict"])
        ema.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng"]])
        loader_rng.set_state(ck["loader_rng"].cpu())
        start = int(ck["epoch"]) + 1
        best, best_epoch = ck["best_val"], ck["best_epoch"]
        bad_epochs = ck["bad_epochs"]
        # Recover the last epoch's log line after a crash between checkpoint
        # save and JSONL/CSV writes; never duplicate an already logged epoch.
        history_file = run / "history.jsonl"
        existing = [json.loads(line) for line in history_file.read_text().splitlines() if line.strip()] if history_file.exists() else []
        if (not existing or existing[-1]["epoch"] < ck["epoch"]) and "epoch_log" in ck:
            log_jsonl(history_file, ck["epoch_log"])
            log_csv(run / "history.csv", ck["epoch_log"])
        print(f"RESUMED {cfg.experiment} from epoch {start - 1} (optimizer/scheduler/RNG restored)")
    elif fresh and any(run.iterdir()):
        raise RuntimeError("--fresh cannot overwrite an existing experiment directory; choose new --out-root.")
    elif not last_path.exists() and (run / "history.jsonl").exists():
        raise RuntimeError("Existing log but no last.pth checkpoint; refusing to silently overwrite logs.")
    else:
        atomic_json_save(asdict(cfg), run / "config.json")
        atomic_json_save(manifest, run / "split_manifest.json")
        atomic_json_save({"python": sys.version, "torch": torch.__version__,
                          "torchvision": __import__("torchvision").__version__,
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                          "params": sum(p.numel() for p in model.parameters()),
                          "notes": "No pretrained generator or discriminator; test split excluded from training"},
                         run / "environment.json")
    log_jsonl(run / "events.jsonl", {"event": "start_or_resume", "time": time.time(),
                                       "epoch": start, "experiment": cfg.experiment})
    print(f"{cfg.experiment}: {exp['description']} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Training {len(tr.dataset)} | validation {len(vl.dataset)} | device={device}")
    print("CHECKPOINTS: last.pth every epoch, best.pth when validation improves")
    for epoch in range(start, cfg.epochs + 1):
        tic = time.monotonic()
        model.train()
        beta = cfg.beta_target * min(1.0, epoch / max(1, cfg.warmup_epochs))
        totals = {k: 0.0 for k in ("total", "mse_sum", "kl_global", "kl_spatial", "ssim_penalty_raw", "edge_l1")}
        count = 0
        for batch, (x, _) in enumerate(tr):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Pure FP32 by design (T4 FP16 caused NaN during prior R2 run).
            rec, state = model(x, stochastic=True)
            parts = loss_terms(rec, x, state, beta, exp["objective"], cfg)
            if not all(torch.isfinite(p).all().item() for p in parts.values()):
                msg = f"NON-FINITE LOSS: epoch={epoch} batch={batch}: " + ", ".join(
                    f"{k}={float(v.detach()):.4g}" for k, v in parts.items())
                log_jsonl(run / "events.jsonl", {"event": "nonfinite_loss", "message": msg})
                raise FloatingPointError(msg)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                           error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(p, 1.0 - cfg.ema_decay)
            count += x.size(0)
            for key in totals:
                totals[key] += parts[key].detach().item() * x.size(0)
        train_avg = {k: val / count for k, val in totals.items()}
        # Always evaluate with beta_target=1 for checkpoint selection, even during KL warm-up.
        val = evaluate_recon(ema, vl, device, cfg, exp["objective"], beta=cfg.beta_target)
        if not all(math.isfinite(v) for v in val.values()):
            log_jsonl(run / "events.jsonl", {"event": "nonfinite_validation", "epoch": epoch})
            raise FloatingPointError(f"Non-finite EMA validation at epoch {epoch}")
        scheduler.step()
        improved = val["total"] < best - 1e-6
        if improved:
            best = val["total"]
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "beta": round(beta, 6), "lr": optimizer.param_groups[0]["lr"],
               "train_total": train_avg["total"], "train_mse_sum": train_avg["mse_sum"],
               "train_kl_global": train_avg["kl_global"], "train_kl_spatial": train_avg["kl_spatial"],
               "train_ssim_raw": train_avg["ssim_penalty_raw"], "train_edge_l1": train_avg["edge_l1"],
               "val_total": val["total"], "val_mse_pixel": val["mse_pixel"],
               "val_ssim": val["ssim"], "val_kl_global": val["kl_global"],
               "val_kl_spatial": val["kl_spatial"], "val_ssim_raw": val["ssim_penalty_raw"],
               "val_edge_l1": val["edge_l1"], "best_epoch": best_epoch,
               "seconds": round(time.monotonic() - tic, 2)}
        # Persist a resumable checkpoint and logs after *every* completed epoch.
        full = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_val": best,
                "best_epoch": best_epoch, "bad_epochs": bad_epochs, "config": asdict(cfg),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_rng": loader_rng.get_state(), "epoch_log": row}
        atomic_torch_save(full, last_path)
        if improved:
            atomic_torch_save({"epoch": epoch, "model_state_dict": ema.state_dict(),
                               "config": asdict(cfg), "val_metrics": val,
                               "model_kind": "EMA"}, run / "best.pth")
            sample_grid(ema, cfg, run / "samples_best.png", device)
        log_jsonl(run / "history.jsonl", row)
        log_csv(run / "history.csv", row)
        plot_logs(run / "history.jsonl", run / "curves.png")
        print(f"{cfg.experiment} {epoch:02d}/{cfg.epochs} beta={beta:.2f} | "
              f"train={train_avg['total']:.2f} | val={val['total']:.2f} | "
              f"val_mse={val['mse_pixel']:.5f} | val_ssim={val['ssim']:.4f} | "
              f"KLg={val['kl_global']:.2f} KLl={val['kl_spatial']:.2f} | "
              f"best={best_epoch} | {row['seconds']:.0f}s", flush=True)
        if epoch % cfg.save_samples_every == 0:
            sample_grid(ema, cfg, run / f"samples_epoch_{epoch:03d}.png", device)
        if epoch % cfg.export_every == 0 or epoch == cfg.epochs:
            zip_experiment(run, Path(__file__))
            print("IMPORTANT: download the EXPORT ZIP now if you need durable weights.")
        if bad_epochs >= cfg.patience:
            log_jsonl(run / "events.jsonl", {"event": "early_stop", "epoch": epoch,
                                               "best_epoch": best_epoch})
            print("Early stopping: no validation objective improvement for", cfg.patience, "epochs")
            break
    print(f"COMPLETE {cfg.experiment} | best={best_epoch} | best objective={best:.4f}")
    zip_experiment(run, Path(__file__))


def load_best(cfg, device):
    path = Path(cfg.out_root) / cfg.experiment / "best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Restore from your downloaded ZIP or saved Kaggle Version.")
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = Config(**ck["config"])
    use_spatial = EXPERIMENTS[cfg.experiment]["spatial"]
    net = CXRHVAE(cfg.global_dim, cfg.spatial_channels, use_spatial).to(device)
    net.load_state_dict(ck["model_state_dict"])
    if not all(torch.isfinite(p).all().item() for p in net.parameters()):
        raise FloatingPointError("Non-finite weights in best.pth")
    net.eval()
    return net, cfg, ck


def inspect(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg, ck = load_best(cfg, device)
    _, val_loader, _, _ = data_loaders(cfg)
    score = evaluate_recon(net, val_loader, device, cfg,
                           EXPERIMENTS[cfg.experiment]["objective"])
    print("Best epoch:", ck["epoch"], "| validation:", json.dumps(score, indent=2))
    folder = Path(cfg.out_root) / cfg.experiment
    sample_grid(net, cfg, folder / "samples_best.png", device)
    x, _ = next(iter(val_loader))
    x = x[:8].to(device)
    with torch.inference_mode():
        rec, _ = net(x, stochastic=False)
    save_image(torch.cat([x.cpu(), rec.cpu()]), folder / "reconstruction_comparison.png", nrow=8)
    print("Saved samples_best.png and reconstruction_comparison.png")


def evaluate_final(cfg, n_samples=624):
    """Use only after selecting your final method on held-out VALIDATION data."""
    if n_samples < 2:
        raise ValueError("Need at least 2 evaluation images")
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg, ck = load_best(cfg, device)
    _, _, test_loader, _, _ = data_loaders(cfg, use_test=True)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.set_dtype(torch.float64)
    inc = InceptionScore(normalize=False, splits=5).to(device)
    torch_gen = torch.Generator(device=device).manual_seed(2026)
    count = 0
    folder = Path(cfg.out_root) / cfg.experiment
    with torch.inference_mode():
        for real, _ in test_loader:
            if count >= n_samples:
                break
            real = real[:n_samples - count].to(device)
            fake = net.generate(real.size(0), generator=torch_gen, device=device)
            if not torch.isfinite(fake).all():
                raise FloatingPointError("Non-finite images during FID evaluation")
            to_u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8).repeat(1, 3, 1, 1)
            real_u8, fake_u8 = to_u8(real), to_u8(fake)
            fid.update(real_u8, real=True)
            fid.update(fake_u8, real=False)
            inc.update(fake_u8)
            count += real.size(0)
    fid_val = float(fid.compute().item())
    inc_mean, inc_std = inc.compute()
    output = {"experiment": cfg.experiment, "best_epoch": ck["epoch"],
              "test_real_images": count, "test_generated_images": count,
              "FID_2048": fid_val, "IS_mean": float(inc_mean.item()),
              "IS_std": float(inc_std.item()),
              "caveats": "624 samples yield noisy FID; ImageNet Inception is not a clinical diagnostic measure"}
    atomic_json_save(output, folder / "evaluation.json")
    sample_grid(net, cfg, folder / "evaluation_samples.png", device)
    print(json.dumps(output, indent=2))
    zip_experiment(folder, Path(__file__))


def self_check(cfg):
    seed_everything(cfg.seed)
    print("DEVICE:", "cuda" if torch.cuda.is_available() else "cpu")
    path = find_dataset(cfg.data_root)
    print("DATASET:", path)
    tr, vl, _, manifest = data_loaders(cfg)
    print("TRAIN/VAL:", len(tr.dataset), len(vl.dataset))
    model = CXRHVAE(128, 8, use_spatial=True)
    print("HIERARCHICAL PARAMETERS:", f"{sum(p.numel() for p in model.parameters()):,}")
    model.train()
    x = torch.rand(2, 1, 128, 128)
    rec, state = model(x)
    losses = loss_terms(rec, x, state, beta=0.1, objective="mse", cfg=cfg)
    losses["total"].backward()
    print("SHAPES:", tuple(x.shape), tuple(rec.shape),
          "GLOBAL:", tuple(state["mg"].shape), "LOCAL:", tuple(state["qm"].shape))
    print("BACKWARD OK:", math.isfinite(losses["total"].item()))
    with torch.inference_mode():
        generated = model.eval().generate(2)
    print("PRIOR GENERATION:", tuple(generated.shape))


def main():
    parser = argparse.ArgumentParser(description="Hierarchical VAE for Kaggle chest X-rays")
    parser.add_argument("command", choices=["check", "train", "inspect", "evaluate", "export"])
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default="H2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00015)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-root", default="/kaggle/working/cxr_hvae")
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--n-samples", type=int, default=624)
    parser.add_argument("--include-last", action="store_true")
    args = parser.parse_args()
    cfg = Config(experiment=args.experiment, epochs=args.epochs,
                 batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed,
                 num_workers=args.num_workers, out_root=args.out_root, data_root=args.data_root)
    if args.command == "check":
        self_check(cfg)
    elif args.command == "train":
        train(cfg, fresh=args.fresh)
    elif args.command == "inspect":
        inspect(cfg)
    elif args.command == "evaluate":
        evaluate_final(cfg, n_samples=args.n_samples)
    elif args.command == "export":
        zip_experiment(Path(cfg.out_root) / cfg.experiment, Path(__file__),
                       include_last=args.include_last)


if __name__ == "__main__":
    main()

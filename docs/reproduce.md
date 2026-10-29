# Reproduction guide

## Local implementation checks

Use Python 3.12 and a compatible PyTorch/torchvision pair. Install requirements into a virtual environment, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python code/cxr_h6_adversarial.py unit-test
python tools/build_figures.py
```

The tests use random tensors and a tiny temporary image dataset. They check latent/output shapes, finite hybrid gradients, prior-only generation, KL behavior, frozen-parameter invariance during a decoder update, state-dictionary reload, and repeatable/disjoint image-level splits. They need neither trained weights nor ImageNet weights. The generated tensors are not research samples.

## Dataset layout

Download the dataset from its source and keep it outside Git. Point `--data-root` to a parent directory containing this structure (recursive discovery is supported):

```text
inputs/
  chest_xray/
    train/
      NORMAL/*.jpeg
      PNEUMONIA/*.jpeg
    test/
      NORMAL/*.jpeg
      PNEUMONIA/*.jpeg
```

The loader creates its own 522-image validation partition from the original train directory. It requires the separate test directory to be present for dataset discovery, but does not use it for model selection.

## H3B training and reconstruction

Kaggle example (the defaults below match the script layout):

```bash
python code/cxr_h3_research.py check --experiment H3B --data-root /kaggle/input --out-root /kaggle/working/cxr_hvae
python code/cxr_h3_research.py train --experiment H3B --epochs 50 --batch-size 16 --seed 2026 --data-root /kaggle/input --out-root /kaggle/working/cxr_hvae
python code/cxr_h3_research.py inspect --experiment H3B --data-root /kaggle/input --out-root /kaggle/working/cxr_hvae
python code/cxr_h3_research.py export --experiment H3B --out-root /kaggle/working/cxr_hvae --include-last
```

For Windows or another host, replace **both roots on every command** with your local input and output directories. The defaults are Kaggle paths, not automatically discovered Windows locations. Avoid `--fresh` unless intentionally starting a separate run; preserve completed logs and checkpoints.

The implementation includes RNG state and optimizer/scaler state for resumable training. Nevertheless, library and device differences can change results. A successful fresh training command does not guarantee exactly reproducing the historical table.

## Restore the original H3B for H6A

H6A needs a source artifact that is not included in this repository. Either restore it under the output root:

```text
cxr_hvae/
  H3B/
    best.pth
    config.json
    split_manifest.json    # preserve when available
```

or attach a trusted `H3B_EXPORT.zip`/extracted backup under the input root. The H4 helper searches it, checks the experiment/configuration, and copies the expected files without overwriting existing artifacts. Do not load an untrusted checkpoint or archive.

## Restricted refinement and paired generation evaluation

```bash
python code/cxr_h6_adversarial.py check --data-root /kaggle/input --out-root /kaggle/working/cxr_hvae
python code/cxr_h6_adversarial.py train --experiment H6A --fake-mode both --epochs 15 --stop-after 15 --batch-size 8 --seed 2026 --data-root /kaggle/input --out-root /kaggle/working/cxr_hvae
python code/cxr_h6_adversarial.py evaluate --experiment H6A --checkpoint 15 --split val --n-samples 522 --seed 2026 --data-root /kaggle/input --out-root /kaggle/working/cxr_hvae
```

The original evaluator requires CUDA. Install `torchmetrics[image]` and `torch-fidelity`; allow the official feature-weight download. The single command evaluates both H3B and H6A with matched streams and writes JSON/CSV with the count, seed, split, and metrics. Keep the exported source hash and environment with those outputs.

For staged sessions, run the same configuration with `--stop-after 5`, then 10, then 15. Leave `--epochs 15` unchanged to retain the intended schedule. Save/download exports; Kaggle working storage is not a durable backup.

## Report and figures

```bash
cd docs/report
pdflatex -interaction=nonstopmode HW1_CXR_VAE_Report.tex
pdflatex -interaction=nonstopmode HW1_CXR_VAE_Report.tex
```

The original PDF is included unchanged and has two pages including references. LaTeX is optional for reading it. `tools/build_figures.py` rebuilds the public architecture, result comparison, and observed-history charts from source data without training any neural model.

## What can and cannot be reproduced from this release alone

| Operation | Available from this repository alone? |
|---|---|
| Architecture / gradient / serialization checks | Yes, with dependencies |
| Rebuild comparison figures from recorded metrics | Yes |
| Read and rebuild the original report | Yes; LaTeX needed only to rebuild |
| Train H3B from scratch | Requires the external dataset and compute |
| Re-evaluate the exact historical H3B/H6A models | Requires original checkpoints, dataset, and compatible environment |
| Recover missing training-history rows | No |
| Claim held-out test or clinical performance | No; no such evaluation was supplied |

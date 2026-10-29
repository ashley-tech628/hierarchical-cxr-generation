# Experiments and evidence provenance

## Scope

This release preserves the original course submission's report, scripts, figures, and metric summaries. Training and image-quality evaluation were conducted before repository preparation. The original trained weights and full notebook/ZIP logs were not supplied in this directory.

`results/recorded-metrics.json` transcribes the supplied numerical summaries. It is not an output from a new evaluation. The technical report rounds H1/H2/H3B reconstruction values and H3B/H6A generation values; the original submission README supplies additional reconstruction precision.

## Data partition

- Source: Kaggle Chest X-Ray Images (Pneumonia), original `train` directory.
- 5,216 images split by class, seed 2026: 4,694 train / 522 validation.
- The loader shuffles indices separately for each class and rounds the validation count per class, then sorts the selected indices.
- Inputs: direct square resize, grayscale, floating-point [0,1].
- No patient-disjoint split was established. Class balance is preserved approximately; class identity does not condition the generative model.
- The dataset's supplied `test` directory is separate and was not scored in the report. Its supplied `val` directory is not the 522-image validation partition used here.

## Reconstruction protocol

Validation uses deterministic posterior means. Pixel MSE and SSIM assess agreement with the corresponding input image. H1→H2 adds structure terms; H2→H3B changes local-grid size, prior/posterior modules, local injection, and the local KL multiplier. H3A would be the closer grid-size control, but its quantitative result is not supplied. Do not attribute the entire H1→H3B improvement to any single change.

## Matched generation protocol

The authoritative comparison is the paired `evaluate` routine in `cxr_h6_adversarial.py`:

| Setting | Value |
|---|---|
| Models | Restored H3B source and H6A epoch 15 |
| Split | 522-image validation partition |
| Sample count | 522 real / 522 generated per model |
| Global RNG | seed 2026, reset per model |
| Spatial RNG | seed 2026 + 314, reset per model |
| KID subset RNG | seed 2026 + 700, reset before computing KID |
| FID features | 2,048-dimensional ImageNet Inception features; float64 metric accumulation |
| KID | 20 subsets, subset size 50 |
| Inception Score | 5 splits |
| Input conversion | clamp [0,1], multiply by 255, round to uint8, repeat grayscale to 3 channels |

The paired routine restores both checkpoints and checks the source-H3B checksum. Its two latent RNG streams avoid interleaving global/local draws in a single stream. Batch size, device, library versions, preprocessing, and evaluation implementation should still be held fixed and recorded when repeating comparisons.

The standalone H3B evaluator has a different default sample count, a single latent RNG stream, and no KID calculation. It should not be mixed with H6A values as a paired ablation. Use the H6A evaluator to score both models together.

## History coverage

The CSV contains 36 observed rows from 50 epochs: 1–6, 11–28, 34–42, and 48–50. Earlier notebook output was truncated. The original training-objective PNG is retained intact; a newly rendered history chart places visible gaps at unobserved epochs. Missing values are never filled or reconstructed from a smooth curve.

Train and validation objectives have different stochastic/augmentation/evaluation conditions. Their absolute ordering alone should not be interpreted as evidence of leakage.

## Interpretation

H3B's recorded reconstruction metrics are stronger than the H1/H2 progression. H6A's FID improves from 216.52 to 210.14, KID from 0.2725 to 0.2612, and IS from 2.287 to 2.321 on the reported protocol. The changes are modest and do not establish significance. Raw per-run uncertainty, repeated seeds, or clinical expert ratings are not supplied.

FID/KID/IS rely on natural-image features rather than radiograph-specific features. Good scores do not establish pathology correctness, patient diversity, privacy, diagnostic usefulness, or clinical safety. The report also describes a detail-latent experiment with sharper reconstructions and worse prior samples, but no quantitative values are available here; it remains a qualitative research observation.

## Follow-up experiments

1. Repeat matched generation at multiple seeds and larger sample sizes, retaining raw metric outputs.
2. Compare H3A/H3B with a fixed architecture and only the local KL multiplier changed.
3. Evaluate patient-disjoint partitions when identifiers permit, without tuning on test results.
4. Examine radiograph-specific feature distances, nearest-neighbor similarity, and diversity in addition to ImageNet metrics.
5. Quantify the reconstruction/adversarial tradeoff over the refinement schedule, including failed or degraded variants.

These are proposed follow-ups, not completed claims.

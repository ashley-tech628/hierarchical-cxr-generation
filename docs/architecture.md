# Architecture and optimization

## Two stochastic levels

| Stage | H3B tensor / operation | Purpose |
|---|---|---|
| Input | `B × 1 × 128 × 128` | Grayscale radiograph in [0,1] |
| Encoder stem | `1 → 32`, 3×3 convolution | Initial local features |
| Downsampling | 48@64² → 80@32² → 128@16² → 192@8² | Spatial hierarchy with residual blocks |
| Global posterior | pooled 192-vector → two 128-vectors | Mean and clipped log variance of `q(z_g | x)` |
| Global map | 128-vector → `192 × 8 × 8` | Decoder and conditional-prior context |
| Local prior context | upsample to 32²; 192→96 convolution + residual | Context for `p(z_l | z_g)` |
| Local prior | two `8 × 32 × 32` maps | Conditional mean and log variance |
| Local posterior | concatenate 80-channel encoder feature and 96-channel context | Residual corrections to prior parameters |
| Local injection | `8 → 128` 3×3 projection, multiplied by `tanh(local_gain)` | Bounded contribution at the decoder's 32² stage |
| Decoder | 192@8² → 160@16² → 128@32² → 80@64² → 48@128² | Bilinear upsampling, convolution and residual refinement |
| Output | GroupNorm → SiLU → 3×3 convolution → sigmoid | One-channel image |

`CXRHVAE32` subclasses the baseline model to preserve compatible component names while replacing the local-prior, posterior, injection, and decoder path. Some inherited legacy modules are not used by H3B's forward path; parameter counts should not be presented as counts of active computation without checking the module graph.

Gaussian reparameterization is `z = mu + exp(0.5 * logvar) * epsilon`. Posterior log variances are clipped to [-9,6], and prior log variances to [-7,5]. These bounds constrain numerical extremes; they do not establish calibrated uncertainty.

## Reconstruction versus generation

**Reconstruction:** encode an image; sample or take means from its global and spatial posteriors; decode. Validation reconstruction uses posterior means (`stochastic=False`). Training uses stochastic posterior draws.

**Generation:** sample the standard-normal global prior; compute its global map and conditional spatial prior; sample that spatial prior; decode. The encoder is never needed. There is no class embedding and no deterministic encoder-to-decoder skip.

The encoder's 32² features enter only the posterior distribution. Treating that connection as a U-Net skip would misdescribe the architecture and the independent-generation path.

## Objective and numerical scale

H3B optimizes batch-averaged summed image MSE plus `48 × (1 − SSIM)`, `32 × Sobel L1`, global KL, and `0.35 × local KL`. The scalar warm-up multiplier reaches 1 after ten epochs. The local KL sums all spatial positions and channels per image before the batch mean, so its dimensionality makes its relative weighting consequential.

Diagonal-Gaussian KL is computed analytically in FP32. Structural terms also operate in FP32. Validation reports both the composite objective and per-pixel MSE; these have different units. Checkpoint selection fixes KL beta at 1 so early warm-up checkpoints are not favored simply because their penalty is smaller.

H3C adds a multi-scale high-pass reconstruction penalty, implemented at image scales 128/64/32. The current report records H3C MSE 0.002469 and SSIM 0.7547, compared with H3B 0.002467 / 0.7542. The small SSIM increase and slightly higher MSE do not establish a generation improvement.

## Restricted adversarial adaptation

H6A starts from an H3B source checkpoint and records its SHA-256. Its trainable allowlist contains decoder blocks, the local-to-decoder projection, the output head, and `local_gain`. The encoder, global-posterior head, `g_to_map`, conditional prior, and local-posterior network remain frozen. The frozen posterior/prior paths produce detached latent tensors.

1. **Discriminator update:** spectral-normalized patch logits receive real images and detached generated images. Hinge loss is averaged over posterior reconstructions and independent prior samples in `fake_mode=both`.
2. **Decoder update:** discriminator parameters are frozen, while gradients through its operations reach the decoder's pixels. Reconstruction and adversarial gradients are measured at the decoder's final convolution.
3. **Adaptive coefficient:** `lambda = clamp(0.05 * ||g_rec|| / (||g_adv|| + 1e-8), 0, 20)`, detached from autograd. This is a local gradient-balancing heuristic, not a convergence guarantee.
4. **EMA:** only trainable model parameters are updated in the H6A EMA. Generator/decoder clipping is 1.0; discriminator clipping is 5.0.

The small discriminator uses 4×4 strided convolutions, LeakyReLU, spectral normalization, and a final patch-logit head. Brightness/contrast augmentation is applied to both domains. No pathology-preservation claim is established by that augmentation choice.

## Artifact and resume discipline

H3 training writes configuration, split manifest, environment notes, per-epoch JSONL/CSV, resumable checkpoint state, and EMA-selected best weights. Writes use temporary files and replacement. Resume guards compare important hyperparameters and refuse an existing history without the corresponding last checkpoint.

H6 stores the H3B source hash, validates restored split paths, and guards schedule and optimizer settings across staged training. Its `stop_after` parameter can advance from 5 to 10 to 15 epochs while retaining a single 15-epoch schedule.

These mechanisms improve experiment traceability. Exact cross-device or cross-version bitwise reproducibility is not promised; the base script enables cuDNN benchmarking.

## Conditional-flow module

`cxr_h4b_flow.py` contains exploratory conditional affine-coupling flow code with checkerboard/channel masks and identity initialization. H6 imports its checkpoint-restore helpers. **Running H6A does not require H4B-trained flow weights**, and no H4B score is attributed to H3B or H6A.

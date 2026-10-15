# D-LAP

Code for D-LAP, a parameter-neighborhood method for transferable targeted adversarial attacks, built on the FTM implementation. D-LAP averages image gradients from temporarily perturbed surrogate models and uses target-logit gradient memory to guide the sampled parameter directions.

## Environment

The completed experiments used:

| Component | Version / configuration |
| --- | --- |
| Python | 3.8.20 |
| PyTorch / torchvision | 2.4.1 / 0.19.1 |
| timm | 1.0.20 |
| NumPy / Pillow | 1.24.4 / 10.4.0 |
| SciPy / Matplotlib | 1.10.1 / 3.7.2 |
| Platform | Windows 11, CUDA 12.4, cuDNN 9.1 |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 GB |

```bash
conda create -n img_code python=3.8.20
conda activate img_code
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.24.4 Pillow==10.4.0 timm==1.0.20 scipy==1.10.1 matplotlib==3.7.2
```

`requirements.txt` lists the pinned packages, but currently sets the PyTorch CUDA wheel index for the entire file. The two installation commands above separate the PyTorch wheels from packages installed through the default package index. GPU execution is expected for the reported experiments.

## Data and pretrained weights

Default layout (download the images separately; labels.csv is already included in the repository):

```text
data/ILSVRC2012/
  images/
  labels.csv
vit_weight/                 # optional local timm transformer checkpoints
```

The ImageNet dataset and pretrained model weights are publicly obtainable from their original providers. Download the images and model weights separately; these large files are not bundled with this repository. The label file, `data/ILSVRC2012/labels.csv`, has already been uploaded and contains the image identifiers, true classes, and assigned target classes used by the experiments. `utils.load_ground_truth` accepts either header:

```csv
ImageId,TrueLabel,TargetClass
```

or:

```csv
filename,label,targeted_label
```

Preserve the original sample order and targets when reproducing a saved experiment. The loader independently subtracts one from a label column if its minimum is at least one; inspect the recorded label-normalization metadata, especially for custom subsets that omit class zero. A saved `sample_manifest.csv` records the resolved samples; `--sample_manifest_input PATH` can reuse their order during generation.

CNNs are loaded through torchvision/timm pretrained builders, which may download weights into their caches. Merely placing CNN weights in an arbitrary `checkpoints/` folder does not configure the root loader. For supported local transformer models, use `--model_weights_dir ./vit_weight`; the architecture-to-filename mapping is in `utils.LOCAL_VIT_MODEL_ARCH_AND_FILE`. Add `--prefer_local_weights` to require the mapped local weights. Record the actual checkpoint identity and hash when reproducing results: the historical Inception-v3 full1000 run used a dedicated local loader, rather than assuming every Inception-v3 checkpoint is interchangeable.

## Generate D-LAP adversarial examples

Run commands from the repository root. The following single-line command works in Bash and PowerShell and explicitly selects the completed RN50 D-LAP configuration (M=2, K=4, S=1):

```bash
python main.py --model_name ResNet50 --attack_method RTMF --config_idx 1 --seed 42 --img_size 299 --epsilon 16 --max_iterations 300 --num_images 1000 --batch_size 32 --beta 0.01 --momentum_decay 1.0 --use_dlap --dlap_control_mode sparse_state_m2 --dlap_guidance_grad_norm normalized --dlap_state_refresh_interval 4 --dlap_param_grad_samples_per_refresh 1 --lap_num_samples 2 --lap_num_samples_interval 1 --lap_rho 0.05 --lap_rho_min 0.02 --use_lap_adaptive_radius --lap_beta 0.9 --lap_warmup 5 --lap_cv_temp 2.0 --lap_mode cv --lap_grad_smooth_kernel 0 --dlap_alpha 0.85 --save_dir ./exp/rn50_dlap
```

RTMF combines resized diverse inputs, translation-invariant gradient smoothing, momentum, and feature tuning/mixup. With FTM enabled, the clean-feature cache consumes one nominal iteration: `--max_iterations 300` produces **299 effective image updates**. The image-space budget is 16/255 and the step size is 1.6/255. Match effective image updates when comparing hosts with and without FTM.

Generation batch size is part of the protocol because parameter gradients and directional memory depend on the batch. The completed CNN runs used batch size 32 for RN50 and 16 for Inception-v3. Changing the source model name alone does not reproduce the historical Inception-v3 weight-loading and serialization protocol.

### Key D-LAP settings

| Option | Meaning / main configuration |
| --- | --- |
| `--use_dlap` | Enable the D-LAP/control implementation |
| `--dlap_control_mode sparse_state_m2` | Two image-gradient samples per effective step, with sparse parameter-state updates |
| `--dlap_state_refresh_interval 4` | Refresh parameter state every fourth effective step |
| `--dlap_param_grad_samples_per_refresh 1` | Request parameter gradients from the first sample at a refresh |
| `--dlap_guidance_grad_norm normalized` | Module-group normalized gradients enter the directional EMA |
| `--lap_rho 0.05 --lap_rho_min 0.02` | Exponential base-radius schedule |
| `--use_lap_adaptive_radius --lap_mode cv` | Scale base radii using module-group gradient statistics |
| `--lap_beta 0.9` | Decay for directional memory and radius statistics |
| `--lap_warmup 5 --lap_cv_temp 2.0` | Radius adaptation warmup and CV scaling exponent |
| `--dlap_alpha 0.85` | Final random-direction mixing coefficient; initially 1, then annealed after 60% of the steps |
| `--beta 0.01` | FTM coefficient, separate from `--lap_beta` |

The default control mode is `legacy` and the default guidance normalization is `raw`. Therefore `--use_dlap` alone does **not** select the main configuration above.

## Evaluate saved examples

Generation and evaluation are separate invocations. Use a fresh output directory for each generation run; existing evaluation artifacts are protected against overwriting.

```bash
python main.py --eval --save_dir ./exp/rn50_dlap --image_csv ./data/ILSVRC2012/labels.csv --img_size 299 --batch_size 32 --target_model_names ResNet18,DenseNet121,mobilenet_v2
```

This example evaluates three targets. Omit `--target_model_names` to use the target list in `config.py`, or supply the exact list for the comparison being reproduced. `--use_local_vit_targets --model_weights_dir ./vit_weight` instead selects available supported local transformer targets. Keep preprocessing, model weights, sample order, and targets fixed across methods.

The root evaluator reports the fraction of evaluated images classified as their assigned target class. It does not apply a defended-model clean-correct denominator; defense experiments use their dedicated evaluation protocol.

## Baselines and controls

For the FTM baseline, use the same data, source model, seed, batch size, and image-update settings while omitting all D-LAP flags:

```bash
python main.py --model_name ResNet50 --attack_method RTMF --seed 42 --img_size 299 --epsilon 16 --max_iterations 300 --num_images 1000 --batch_size 32 --beta 0.01 --momentum_decay 1.0 --save_dir ./exp/rn50_ftm
```

Starting from the full D-LAP command:

- **Isotropic-M2:** change the control mode to `isotropic_sparse_m2` and set `--dlap_alpha 1.0`. It retains random module-normalized neighborhoods and adaptive radii without reading directional EMA guidance.
- **RepeatGrad-M2:** change the control mode to `repeatgrad_m2`. It averages two image-gradient samples without applying parameter perturbations.
- **Dense D-LAP:** retain `sparse_state_m2`, but set `--dlap_state_refresh_interval 1 --dlap_param_grad_samples_per_refresh 2`.

These commands describe the root implementation. Exact paper-table reproduction also requires the corresponding sample manifests, checkpoint identities, PNG serialization, target panel, and experiment-specific evaluation scripts.

## Implementation details

- **Objective:** maximize the positive batch sum of assigned target-class raw logits (`LogitLoss` in `attacks.py`). There is no softmax, cross-entropy, or additional feature-loss term in this objective.
- **Parameter groups:** trainable tensors are grouped by their module-name prefix; a module's selected weight and bias share a joint L2/Frobenius normalization denominator with epsilon 1e-12.
- **Parameter selection:** the name filter excludes literal substrings `pos_embed`, `cls_token`, `patch_embed`, and `norm`. This is a name-based filter, not a blanket exclusion of BatchNorm/LayerNorm classes. Ordinary CNN BatchNorm affine tensors named `bn1.weight`, for example, can be selected.
- **Model state:** surrogate models use evaluation mode, freezing BatchNorm running statistics. Selected affine parameters can still receive temporary perturbations. The implementation adds and subtracts parameter displacements per sample; floating-point restoration is not guaranteed to be bitwise identical.
- **Directional memory:** module-normalized target-logit parameter gradients update an EMA. The EMA is normalized again before direction mixing. At a refresh, the first sample updates memory before the second sample reads it. Memory is initialized per generation batch.
- **Adaptive radii:** the exponentially scheduled base radius is multiplied by module-specific CV-derived factors after warmup; the effective radii are not a single global weight-space radius.
- **Image update:** average the sampled image gradients, apply the host TI smoothing and per-image L1-normalized momentum, then ascend by the sign and project to the clean-image L-infinity box and valid pixel range.

### PNG serialization

The root entrypoint calls `utils.save_images`, which multiplies by 255 and casts to uint8. It does **not** automatically invoke `strict_serialization.py`. The strict helper additionally projects saved integer pixels around the canonical clean uint8 image. Historical strict results used dedicated writers/projection wrappers; using the root command alone must not be described as reproducing that strict serialization pipeline. Keep the clean-image resizing and integer anchor identical when comparing saved-PNG results.

## Outputs

Generation writes `adv_imgs/`, `attack_params.json`, `attack_timing.json`, `sample_manifest.csv`, and compute/memory records. Evaluation writes `eval_results.txt`, `evaluation_model_weights.json`, `evaluation_config.json`, and evaluation memory records. The saved metadata records effective steps, label normalization, configuration, and available checkpoint identifiers/hashes.

## Source files for a runnable release

Upload these **12 root Python files together** when releasing the current entrypoint without code changes:

```text
main.py
config.py
utils.py
attacks.py
dlap_attack.py
memory_utils.py
clean_repeatgrad.py
tsnr_attack.py
temporal_decomp.py
stateful_param_walk.py
cross_space_reliability.py
strict_serialization.py
```

The first six form the main entrypoint and D-LAP/FTM utilities. The next five are imported unconditionally by the current `main.py`, even when their historical experimental modes are disabled. `strict_serialization.py` is additionally needed to expose the strict integer-budget writer used by paper experiment wrappers. Include `p02_diagnostics.py` only if releasing the optional `--p02_diagnostic` mode.

Also include this README, `requirements.txt`, the existing `LICENSE` and upstream attribution, and the sample/target metadata and configuration records needed for the claimed experiments. The 12 files above support the root generation/evaluation commands; they do not by themselves reproduce all defense, external-baseline, mechanism, and figure experiments. Those require their corresponding scripts and metadata. Some historical runners contain absolute local paths and must be made portable before being advertised as runnable release entrypoints.

Do not include generated adversarial images, ImageNet image files, checkpoint binaries, caches, temporary files, or private manuscript/revision material in the source upload.

## Acknowledgement and license

This repository builds on the public FTM implementation. We thank the original authors for releasing their code. Retain the upstream copyright and MIT license in `LICENSE`, along with attribution for any additional third-party code included in a release.


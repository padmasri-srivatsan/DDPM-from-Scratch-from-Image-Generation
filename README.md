# Denoising Diffusion on CIFAR-10

A PyTorch project that builds the core components of a **Denoising Diffusion Probabilistic Model (DDPM)** to generate small colour images. The implementation lives in a Jupyter notebook and Metal Performance Shaders (MPS), as well as CUDA and CPU execution.

The project covers data preparation, forward noising, a time-conditioned U-Net with attention, training, checkpointing, and DDPM/DDIM sampling. It also includes evaluation and baseline-comparison code, with those results still pending in the saved artifacts.

**Main notebook:** [DDPM_CIFAR10_Mac_Local.ipynb](DDPM_CIFAR10_Mac_Local.ipynb)

## Project objective

Explore whether a diffusion model built from its core components can generate higher-quality images and train more consistently than autoencoder (AE), variational autoencoder (VAE), and deep convolutional generative adversarial network (DCGAN) baselines on CIFAR-10.

The saved artifacts demonstrate a trained prototype and learning progress. They do not yet establish better image quality than the baselines.

## Current results

The following values come from the saved checkpoint histories:

| Run | Training images | Epochs | Base channels | Parameters | First-epoch MSE | Final-epoch MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small-data sanity test | 64 | 300 | 64 | 9,283,587 | 1.0895 | 0.0969 |
| Main cosine + attention run | 47,500 | 18 | 32 | 2,329,603 | 0.3554 | 0.0636 |

The main resume checkpoint records **6,678 optimizer steps**. Its average training noise-prediction mean squared error (MSE) decreased by approximately **82.1%** between the first and final recorded epochs.

Training MSE measures how accurately the network predicts added noise. It does not directly measure the quality of generated images. The sanity test and main run also use different model widths, so they are not a controlled architecture comparison.

### Saved sample grid

![Generated CIFAR-10 sample grid after 18 epochs](diffusion_project/samples/cosine_attn_epoch018.png)

The notebook's logging path uses exponential moving average (EMA) weights with 50 DDIM sampling steps. The saved grid shows colour and texture, but many tiles have substantial artifacts and unclear object structure.

### Evaluation status

| Experiment or output | Status in the supplied project |
| --- | --- |
| Main diffusion model | Checkpoints, training history, and sample grids available |
| FID and Inception Score | Evaluation code available, no saved scores |
| Generative precision and recall | Evaluation code available, no saved scores |
| Sampling runtime comparison | Code available, no saved timing results |
| Linear/cosine and attention ablations | Code available, disabled by default |
| AE, VAE, and DCGAN baselines | Code available, no retained trained outputs or scores |
| Classifier-free guidance | Optional code available, disabled by default |
| Web application or production deployment | Not included |

## Dataset and preparation

The notebook loads the **50,000-image CIFAR-10 training partition** through `torchvision.datasets.CIFAR10`.

- **Image shape:** 3 × 32 × 32, with red, green, and blue channels.
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- **Training split:** 47,500 images.
- **Reference split:** 2,500 images, selected using a seed-42 random permutation.
- **Preprocessing:** tensor conversion and pixel scaling from `[0, 1]` to `[-1, 1]`.
- **Augmentation:** random horizontal flips for training only.
- **Batching:** 128 images per batch, with the incomplete final batch dropped.

The reference set receives no random flips and is exported to PNG files for distribution-based evaluation. The code asserts that training and reference indices do not overlap. The saved checkpoints do not retain the historical split indices.

The main model is **unconditional**: it generates images without receiving a class label. Its training target is the added Gaussian noise, rather than the CIFAR-10 class.

## How the model works

1. Select a clean training image and a random timestep.
2. Add Gaussian noise according to the chosen noise schedule.
3. Give the noisy image and timestep to the U-Net.
4. Train the network to predict the noise using MSE.
5. Generate new images by starting with random noise and repeatedly applying the learned denoising function.

The implementation includes:

- Linear and cosine noise schedules across 1,000 timesteps.
- Sinusoidal timestep embeddings and time-conditioned residual blocks.
- A U-Net with skip connections and attention at selected resolutions.
- Adam optimization, learning-rate warm-up, and gradient clipping.
- EMA weights, checkpoint saving, and training resumption.
- DDPM ancestral sampling and DDIM sampling with configurable step counts.

DDIM-50 uses 20 times fewer model evaluations than DDPM-1000. Actual runtime and image-quality differences still require measurement.

## Project structure

```text
DDPM/
├── README.md
├── DDPM_CIFAR10_Mac_Local.ipynb
├── requirements.txt
├── data/                              # Downloaded CIFAR-10 files
├── diffusion_project/
│   ├── checkpoints/
│   │   ├── cosine_attn_final.pt       # Main weights, EMA, history, partial config
│   │   ├── cosine_attn_last.pt        # Main resume state
│   │   └── overfit_test_final.pt      # Small-data sanity test
│   └── samples/                       # Saved generated grids
├── fid_eval/
│   └── real_holdout/                  # 2,500 reference PNGs
└── output/                            # Project presentations
```

This structure describes the local project. Checkpoints, datasets, and generated files may need separate distribution if they are excluded from a Git checkout. The notebook writes `diffusion_project/results.csv` after successful evaluation and final aggregation; that file is not currently present.

## Setup

Run these commands from the project directory using a Python environment compatible with PyTorch and torchvision:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas matplotlib tqdm Pillow ipython torch-fidelity jupyterlab ipykernel
python -m ipykernel install --user --name ddpm-local --display-name "Python (DDPM)"
jupyter lab DDPM_CIFAR10_Mac_Local.ipynb
```

Select **Python (DDPM)** as the notebook kernel. For a CUDA-specific installation, use the installation instructions appropriate to your hardware on the [PyTorch website](https://pytorch.org/get-started/locally/).

**Dependency-file limitation:** the current `requirements.txt` includes names such as `DataLoader`, `Subset`, `Image`, and `display`. These are objects imported from other libraries, rather than the intended standalone dependencies. It also omits an explicit `torchvision` entry. Use the installation command above and skip the notebook's `%pip install -r requirements.txt` cell. Dependencies are not version-pinned, so this is not a locked, fully reproduced environment.

The dataset cell downloads CIFAR-10 into `data/` if needed. Metric calculation may also download pretrained feature-extractor weights.

## Running the notebook

### Explore or train from scratch

Follow the numbered notebook sections:

| Steps | Stage |
| --- | --- |
| 1–4 | Environment, data loading, reference split, and exploratory analysis |
| 5–6 | Noise schedules and forward diffusion |
| 7–10 | Timestep embedding, residual blocks, attention, and U-Net |
| 11–13 | Samplers, training utilities, sanity test, and main training |
| 14–15 | Image generation and sampling-step experiments |
| 16–18 | Evaluation, ablations, and baseline comparisons |
| 19–20 | Optional class conditioning and final result export |

Review the settings before executing training cells. Step 12 runs a 300-epoch sanity test, and Step 18 trains the baseline models when executed. These are separate experiments with additional compute costs.

The notebook chooses CUDA first, then MPS, then CPU. Mixed precision activates only on CUDA; the MPS path uses float32. CPU training can be slow.

### Load the existing main checkpoint

**Set `BASE_CH = 32` before loading the saved main model.** The notebook currently defaults to 64, while `cosine_attn_final.pt` and `cosine_attn_last.pt` contain a 32-channel model. Leaving the default unchanged causes state-dictionary shape mismatches.

1. Install the dependencies and run the setup and data cells.
2. Set `BASE_CH = 32` and run the model, sampler, and training-function definitions through Step 11.
3. Skip the Step 12 sanity-training cell when you only want to inspect the existing model.
4. Run Step 13 to read the saved main history. It skips training when the final checkpoint exists.
5. Generate a small sample batch with the following notebook code:

```python
main = load_run("cosine_attn")

images = ddim_sample(
    main["ema"],
    main["sched"],
    n=16,
    steps=50,
    generator=fixed_generator(),
)
show_grid(images, "Saved main model: EMA, DDIM-50", nrow=4)
```

The saved sanity-test checkpoint uses **64 base channels**. Model width must match the particular checkpoint being loaded. Architecture settings currently depend on global notebook variables and are not fully stored in each checkpoint.

### Main training settings

| Setting | Current notebook value |
| --- | ---: |
| Random seed | 42 |
| Diffusion timesteps | 1,000 |
| Batch size | 128 |
| Learning rate | 0.0002 |
| Learning-rate warm-up | 200 optimizer steps |
| Gradient-norm limit | 1.0 |
| Maximum EMA decay | 0.999 |
| Base channels | 64, or 32 for the saved main checkpoint |
| Epoch upper limit | 1,000 |
| Main training budget | 75 minutes per invocation |
| Generated images per metric evaluation | 2,500 |

The training loop checks its time budget at epoch boundaries, so the setting is not a strict execution-time limit. Existing `*_last.pt` files support resumption, and an existing main `*_final.pt` file makes Step 13 skip training. Use a distinct run name for a new experiment to keep earlier artifacts separate.

## Evaluation and comparison

The evaluation code uses `torch-fidelity` to compare generated images with the saved real reference set:

- **Fréchet Inception Distance (FID):** compares real and generated feature distributions. Lower is better.
- **Inception Score (IS):** summarizes confidence and diversity of class predictions for generated images.
- **Generative precision and recall:** assess sample fidelity and coverage of the real distribution.
- **Generation time:** records the elapsed time for image generation and export.

Keep the reference set and generated-image count consistent when comparing runs. On the MPS workflow, the current metric call uses CPU rather than CUDA.

The following expensive experiments are disabled by default:

| Flag | Experiment |
| --- | --- |
| `RUN_DDPM_FID = False` | Full 1,000-step DDPM metric evaluation |
| `RUN_ABLATIONS = False` | Linear/cosine schedule and attention comparisons |
| `RUN_CFG = False` | Class-conditioned training with classifier-free guidance |
| `RUN_FLOOR = False` | Real-versus-real metric calibration |

The real-versus-real comparison is a calibration measurement, not a universal lower bound on FID. A single seed and 2,500-image evaluation also limit conclusions about small metric differences. Equal epochs across different architectures do not imply equal compute.

## Limitations and next steps

- Resolve the checkpoint-width mismatch and store complete architecture settings per run.
- Record dataset identity, split indices, package versions, device, and elapsed time.
- Improve the generated images and retain reproducible quality metrics and sampling timings.
- Complete controlled ablations and baseline comparisons with explicit compute budgets.
- Repeat experiments across seeds before claiming small performance improvements.
- Validate downstream usefulness before using the model for synthetic augmentation or an application.

This is a research and learning prototype. Business impact, privacy guarantees, and production readiness have not been demonstrated.

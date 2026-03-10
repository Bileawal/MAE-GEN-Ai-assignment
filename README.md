# MAE-GEN-Ai-assignment
# 🎭 Masked Autoencoder (MAE) — Self-Supervised Image Representation Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)
![Kaggle](https://img.shields.io/badge/Kaggle-T4×2_GPU-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A pure PyTorch implementation of Masked Autoencoders (MAE) for self-supervised visual representation learning.**  
Trained on TinyImageNet using dual NVIDIA T4 GPUs on Kaggle.

[📓 Kaggle Notebook](#) · [🤗 Live Demo](#) · [📊 Results](#results)

</div>

---

## 📌 Overview

Masked Autoencoders learn powerful visual features **without any labels**. The idea:

1. Split a `224×224` image into `196` patches of `16×16` pixels
2. Randomly **mask 75%** of patches (147 out of 196)
3. Feed only the **visible 25%** (49 patches) through a large ViT encoder
4. A lightweight decoder reconstructs the **full image** from visible tokens + learnable mask tokens
5. Loss is **MSE on masked patches only** — forcing the model to understand global image structure

> This approach, introduced by He et al. (2021), achieves state-of-the-art transfer learning performance with no supervised labels.

---

## 🏗️ Architecture

```
Input Image (224×224)
        │
        ▼
  ┌─────────────┐
  │  Patchify   │  → 196 patches of 16×16
  └─────────────┘
        │  random mask 75%
        ▼
  ┌─────────────────────────────┐
  │   MAE ENCODER (ViT-Base)   │  ← only 49 visible patches
  │   768D · 12H · 12L · 86M  │
  └─────────────────────────────┘
        │  latent representations
        ▼
  ┌─────────────────────────────┐
  │   MAE DECODER (ViT-Small)  │  ← visible tokens + 147 mask tokens
  │   384D · 6H · 12L · 22M   │
  └─────────────────────────────┘
        │
        ▼
  Reconstructed Image (MSE loss on masked patches only)
```

### Encoder — ViT-Base/16

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Patch Size | 16 × 16 |
| Number of Patches | 196 (14×14 grid) |
| Visible Patches (25%) | 49 |
| Hidden Dimension | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Parameters | ~86M |

### Decoder — ViT-Small/16

| Parameter | Value |
|---|---|
| Hidden Dimension | 384 |
| Transformer Layers | 12 |
| Attention Heads | 6 |
| Output per Patch | 768 values (16×16×3) |
| Parameters | ~22M |

---

## 📊 Results

| Metric | Score |
|---|---|
| **PSNR** | `[YOUR SCORE]` dB |
| **SSIM** | `[YOUR SCORE]` |
| **Val Loss** | `[YOUR SCORE]` |
| **Epochs** | 20 (fast) / 100 (full) |
| **Training Time** | ~30 min (fast) / ~8 hrs (full) |

### Reconstruction Samples

> Masked Input (75% removed) → Model Reconstruction → Ground Truth

*(Add your reconstructions.png here after training)*

```
![Reconstructions](reconstructions.png)
```

### Loss Curve

```
![Loss Curve](loss_curve.png)
```

---

## 🗂️ Project Structure

```
masked-autoencoder-mae/
│
├── MAE_FastTrain_Kaggle.ipynb   # Main notebook (fast training ~30 min)
├── MAE_TinyImageNet_COMPLETE.ipynb  # Full training notebook (100 epochs)
│
├── loss_curve.png               # Train/val loss vs epochs
├── reconstructions.png          # 8 qualitative reconstruction samples
├── metrics.png                  # PSNR & SSIM distribution plots
│
├── mae_fast.pth                 # Saved model checkpoint
└── README.md                    # This file
```

---

## ⚙️ Implementation Details

### Key Components (Pure PyTorch — No timm)

| Component | Implementation |
|---|---|
| `PatchEmbed` | `Conv2d(3, 768, kernel=16, stride=16)` |
| `MHSA` | Multi-Head Self-Attention from scratch |
| `Block` | Pre-LN Transformer block (Attention + MLP + residuals) |
| `Positional Embed` | Fixed 2D Sine-Cosine embeddings |
| `random_masking()` | Noise-based random patch selection |
| `patchify()` | `(B,3,H,W)` → `(B,196,768)` reshape |
| `unpatchify()` | `(B,196,768)` → `(B,3,H,W)` reshape |

### Training Techniques

| Technique | Detail |
|---|---|
| **Loss** | MSE on masked patches only (normalised pixel targets) |
| **Optimizer** | AdamW — decay/no-decay parameter groups |
| **Scheduler** | Cosine LR + 3-epoch linear warm-up |
| **Mixed Precision** | `torch.cuda.amp` — `autocast` + `GradScaler` |
| **Gradient Clipping** | `clip_grad_norm_(params, 1.0)` |
| **Multi-GPU** | `nn.DataParallel` — Kaggle T4×2 |
| **Batch Size** | 128 (fast) / 64 (full) |
| **Weight Decay** | 0.05 on weight matrices only |

---

## 🚀 How to Run

### On Kaggle (Recommended)

1. Open [Kaggle](https://kaggle.com) → Create New Notebook
2. Add dataset: **TinyImageNet** → `akash2sharma/tiny-imagenet`
3. Set accelerator: **GPU T4 × 2**
4. Upload `MAE_FastTrain_Kaggle.ipynb`
5. Click **Run All** — done in ~30 minutes

### Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/masked-autoencoder-mae.git
cd masked-autoencoder-mae

# Install dependencies
pip install torch torchvision einops scikit-image gradio matplotlib

# Download TinyImageNet and update TINY_ROOT path in Cell 3
# Then run the notebook
jupyter notebook MAE_FastTrain_Kaggle.ipynb
```

### Dependencies

```
torch >= 2.0
torchvision
einops
scikit-image
gradio
matplotlib
numpy
Pillow
```

---

## 🎛️ Configuration

All hyperparameters are in **Cell 3** of the notebook:

```python
IMG_SIZE      = 128     # image size (128=fast, 224=full spec)
PATCH_SIZE    = 16      # always 16×16
MASK_RATIO    = 0.75    # 75% masking (assignment requirement)
ENC_DIM       = 768     # encoder hidden dim (ViT-Base spec)
ENC_DEPTH     = 6       # encoder layers (6=fast, 12=full spec)
DEC_DIM       = 384     # decoder hidden dim (ViT-Small spec)
DEC_DEPTH     = 4       # decoder layers (4=fast, 12=full spec)
EPOCHS        = 20      # training epochs
BATCH_SIZE    = 128     # per-iteration batch size
```

To run the **full assignment spec**, change:
```python
IMG_SIZE = 224 | ENC_DEPTH = 12 | DEC_DEPTH = 12 | BATCH_SIZE = 64
```

---

## 🖥️ Gradio Demo App

The notebook includes a live Gradio app (Cell 15):

- 📂 **Upload** any image
- 🎚️ **Slider** to choose masking ratio (10%–95%)
- ⚡ **Auto-runs** reconstruction on slider change
- 🖼️ Shows: Masked Input | Reconstruction | Ground Truth
- 🔗 `share=True` generates a public URL directly from Kaggle

---

## 📚 Reference

```bibtex
@article{he2021masked,
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  author  = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao 
             and Dollár, Piotr and Girshick, Ross},
  journal = {arXiv preprint arXiv:2111.06377},
  year    = {2021}
}
```

---

## 📋 Assignment Requirements Coverage

| Requirement | Status |
|---|---|
| ViT-Base Encoder (~86M) | ✅ 768D · 12H · patch16 |
| ViT-Small Decoder (~22M) | ✅ 384D · 6H · patch16 |
| 75% masking (147/196 patches) | ✅ `random_masking()` |
| Encoder: only visible patches | ✅ No mask tokens in encoder |
| Sincos positional embeddings | ✅ `sincos_pos_embed()` |
| Learnable mask token | ✅ `nn.Parameter` |
| MSE loss on masked patches only | ✅ `loss * mask / mask.sum()` |
| AdamW + Cosine LR + Warmup | ✅ All implemented |
| Mixed Precision (AMP) | ✅ `autocast` + `GradScaler` |
| Gradient clipping | ✅ `clip_grad_norm_` |
| DataParallel T4×2 | ✅ `nn.DataParallel` |
| Loss curve plot | ✅ Linear + log scale |
| ≥5 reconstruction samples | ✅ 8 samples |
| PSNR & SSIM evaluation | ✅ `skimage.metrics` |
| Gradio app (upload + slider) | ✅ Cell 15 |
| Pure base PyTorch | ✅ No timm used |

---

<div align="center">

Made with ❤️ using PyTorch · Trained on Kaggle T4×2 GPUs

</div>

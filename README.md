# SAGE-DINOv3: Shape-Adapting Gated Experts with DINOv3 for Crack Segmentation

Official PyTorch implementation of **SAGE-DINOv3**, an advanced Mixture of Local Experts (MoLEx) architecture tailored for fine-grained boundary and crack segmentation. SAGE-DINOv3 integrates **DINOv3 vision foundation models** (ConvNeXt-Tiny + ViT-Small) with dynamic sparse routing, pairwise cross-space adaptation (SA-Hub), and boundary-aware optimization.

---

## 🔬 Implementation Highlights — Key Code Architecture

### 1. Hybrid DINOv3 Backbone Integration
SAGE-DINOv3 combines two complementary DINOv3 foundation backbones:
* **ConvNeXt-Tiny DINOv3** (`convnext_tiny.dinov3_lvd1689m`): 4 convolution stages ($C \in [96, 192, 384, 768]$) capturing local high-frequency crack edges and textures.
* **ViT-Small DINOv3** (`vit_small_patch16_dinov3.lvd1689m`): 4 Vision Transformer blocks ($D=384$) equipped with Dynamic RoPE (Rotary Position Embeddings) capturing global topological connectivity.

Weights are loaded automatically from the official `timm` / Hugging Face hub repository:
```python
# sage/networks/convnextv2_vit_hybrid.py
self.convnext = timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=True)
vit_full = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True)
```

### 2. In-Place SAGE Layer Injection & Expert Pool
SAGE wraps the 4 CNN stages and 4 ViT blocks in-place into `SageLayer` modules, constructing a **heterogeneous pool of 8 experts**:
```python
# sage/networks/b2_unet.py
self.expert_pool = inject_sage_layers(
    convnext=self.backbone.convnext,
    transformer_blocks=self.backbone.transformer_blocks,
    encoder_channels=[96, 192, 384, 768],
    transformer_dim=384,
    sage_config=self.sage_config,
)
```
Each layer combines:
$$\mathbf{y} = \alpha \cdot \mathbf{y}_{\text{main}} + (1 - \alpha) \cdot \mathbf{y}_{\text{expert}}$$
Where $\mathbf{y}_{\text{expert}}$ is dynamically computed by routing patches to the Top-$2$ most relevant experts selected from the 8-expert pool.

### 3. Cross-Space SA-Hub Adapters
To allow CNN tokens to pass through ViT blocks and vice-versa, the **Shape-Adapting Hub (SA-Hub)** instantiates pairwise projection adapters:
* **CNN $\to$ ViT:** 1x1 Conv $\to$ Flatten spatial dimensions $(H, W) \to$ Patch sequence.
* **ViT $\to$ CNN:** Reshape token sequence to 2D $\to$ Bilinear Interpolation to target spatial resolution $\to$ 1x1 Conv.

---

## ⚖️ Trainable vs. Frozen Parameter Regimes

SAGE-DINOv3 supports two verified training protocols on the **Crack500** benchmark:

### Parameter Status by Module and Stage

| Module / Layer | Config A (Stage 1 & 2) | Config B (Stage 1: Ep 1-17) | Config B (Stage 2: Ep 18-35) | Optimizer Learning Rate |
| :--- | :---: | :---: | :---: | :---: |
| **DINOv3 ConvNeXt Stages** (`.main_block`) | **Frozen** (`grad=None`) | **Frozen** (`grad=None`) | **Trainable** (Unfrozen) | $\text{LR} = 10^{-5}$ *(Config B Stage 2)* |
| **DINOv3 ViT Blocks** (`.main_block`) | **Frozen** (`grad=None`) | **Frozen** (`grad=None`) | **Trainable** (Unfrozen) | $\text{LR} = 10^{-5}$ *(Config B Stage 2)* |
| **Bridge Projections** (`convnext_to_transformer`) | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 10^{-4}$ |
| **8 SAGE Routers** | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 2 \times 10^{-4}$ (S1) / $10^{-4}$ (S2) |
| **SA-Hub Pairwise Adapters** | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 2 \times 10^{-4}$ (S1) / $10^{-4}$ (S2) |
| **UNet Decoder** | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 10^{-4}$ |
| **Soft Boundary IoU Loss** | *N/A (Loss function)* | *N/A (Loss function)* | *N/A (Loss function)* | *No parameters to update* |

---

## 📊 Empirical Benchmarks on NVIDIA Tesla T4 (Google Colab)

Detailed empirical characterization of memory footprint and training speed on Tesla T4 (see full report in [`docs/dinov3_verification_t4.md`](docs/dinov3_verification_t4.md)):

| Experiment Mode | Config File | Batch Size | Peak VRAM | Speed | 1 Epoch Time | Full 35 Epochs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A** *(Full Freeze)* | `configs/dinov3_b2_crack500_frozen_full.yaml` | **16** | **9,353 MB** | **9.83 img/s** | ~3.2 min | **~1.88 hours** |
| **Config B** *(Stage 2 Unfreeze)* | `configs/dinov3_b2_crack500_frozen_stage1.yaml` | **10** | **10,477 MB** | **7.01 img/s** | ~4.5 min | **~2.64 hours** |

* **Hardware Safety:** Config B uses `BS=10` to maintain a **>4.0 GB VRAM safety buffer**, preventing transient out-of-memory errors caused by dynamic sparse expert routing.

---

## 📁 Repository Structure

```text
sage_dinov3/
├── configs/                  # Active training configurations
│   ├── dinov3_b2_crack500.yaml               # Canonical Base configuration
│   ├── dinov3_b2_crack500_frozen_full.yaml   # Config A: Full Freeze (BS=16)
│   └── dinov3_b2_crack500_frozen_stage1.yaml # Config B: Stage 2 Unfreeze 1e-5 (BS=10)
├── docs/                     # Verification and empirical characterization reports
│   └── dinov3_verification_t4.md             # T4 VRAM, throughput & stability report
├── sage/                     # Core library implementation
│   ├── components/           # SAGE Routers, SA-Hub adapters, SageLayer wrappers
│   ├── networks/             # B2 U-Net, ConvNeXt+ViT Hybrid, Decoders
│   └── utils/                # Dataloader, metrics, evaluation utilities
├── scripts/                  # Training and evaluation entry points
│   ├── train_crack.py        # Two-stage training engine with dynamic unfreeze
│   ├── evaluate_crack_official.py # Direct-resize official Crack500 benchmark
│   └── tests/                # Precision and protocol verification tests
│       ├── test_amp_stability.py       # FP32/AMP FP16 & adapter gradient audit
│       ├── test_boundary_iou_loss.py   # Loss correctness tests
│       └── test_two_stage_protocol.py  # Checkpoint continuity tests
├── tools/                    # Post-training analysis & visualization
│   ├── analyze_routing.py    # Per-sample dynamic routing decisions audit
│   └── visualize_expert_routing_decisions.py # Gate & token routing plots
├── prepare_data/             # Dataset download & preprocessing
│   └── prepare_crack500.py   # Automated Crack500 setup
└── requirements.txt          # Python dependencies
```

---

## ⚡ Reproduction Guide (Google Colab / Tesla T4)

### Step 1: Environment Setup & Dataset Preparation
```bash
%cd /content
!rm -rf /content/sage_dinov3
!git clone https://github.com/bach0823/sage_dinov3.git /content/sage_dinov3
%cd /content/sage_dinov3

# Download and extract Crack500 (1896 training samples)
!python prepare_data/prepare_crack500.py
```

### Step 2: Launch Training

* **To train Config A (Full Freeze — Fast & Lightweight):**
```bash
%cd /content/sage_dinov3
!python scripts/train_crack.py \
  --config configs/dinov3_b2_crack500_frozen_full.yaml \
  --data-root /content/dataset/Crack500
```

* **To train Config B (Delayed Unfreeze — Foundation Fine-Tuning):**
```bash
%cd /content/sage_dinov3
!python scripts/train_crack.py \
  --config configs/dinov3_b2_crack500_frozen_stage1.yaml \
  --data-root /content/dataset/Crack500
```

### Step 3: Post-Training Routing Diagnostics & Archival
Following SAGE project standards, post-training evaluation, diagnostics, and artifact downloading are executed in an **All-in-One Cell**:
```python
%cd /content/sage_dinov3
from google.colab import files
import shutil

# 1. Routing Diagnostics on Validation Set
!python tools/analyze_routing.py \
    --config configs/dinov3_b2_crack500_frozen_stage1.yaml \
    --checkpoint results/dinov3_sage_b2_frozen_stage1/best_model_b2_stage2.pth \
    --output_dir results/dinov3_sage_b2_frozen_stage1/routing_diagnostics \
    --split val \
    --data-root /content/dataset/Crack500

# 2. Package checkpoints (Preserves BOTH best_model and last_model) + Logs
archive_name = "/content/dinov3_sage_b2_frozen_stage1_run"
shutil.make_archive(archive_name, "zip", "results", "dinov3_sage_b2_frozen_stage1")

# 3. Trigger browser download to local workstation
files.download(f"{archive_name}.zip")
```

---

## 🧪 Verification & Stability Tests

Verify AMP FP16 numerical stability, cuDNN Tensor Core compatibility, and active adapter tracking:
```bash
python scripts/tests/test_amp_stability.py
```

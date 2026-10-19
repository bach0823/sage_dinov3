# SAGE-DINOv3: Shape-Adapting Gated Experts with DINOv3 for Crack Segmentation

Official PyTorch implementation of **SAGE-DINOv3**, an advanced Mixture of Local Experts (MoLEx) architecture tailored for fine-grained boundary and crack segmentation. SAGE-DINOv3 replaces the legacy backbones with high-capacity **DINOv3 vision foundation models** (ConvNeXt-Tiny + ViT-Small), combined with dynamic routing and shape-adaptive cross-attention.

---

## 🚀 Key Architectural Innovations

1. **Hybrid Foundation Backbone:**
   - **CNN Stages (Local Edge & Texture):** DINOv3 ConvNeXt-Tiny (`convnext_tiny.dinov3_lvd1689m`) with channel stages $[96, 192, 384, 768]$.
   - **Transformer Blocks (Global Context & Topology):** DINOv3 ViT-Small (`vit_small_patch16_dinov3.lvd1689m`) with 4 Dynamic RoPE attention blocks (embedding dim $384$).
2. **Heterogeneous Expert Pool (8 Experts & 8 Routers):**
   - 4 ConvNeXt Stage Experts + 4 ViT Block Experts.
   - 8 Hierarchical SAGE Routers (Top-2 dynamic gating, Sigmoid modulation, load balancing $\lambda=0.01$).
3. **Shape-Adapting Hub (SA-Hub):**
   - Pairwise cross-space adapters enabling seamless feature exchange between CNN spatial maps and ViT token representations.
4. **Boundary IoU Objective:**
   - Differentiable Soft Boundary IoU ($\lambda=0.50, d=2$) preserving narrow crack continuity.

---

## 📁 Repository Structure

```text
sage_dinov3/
├── configs/                  # Standard training configurations
│   ├── dinov3_b2_crack500.yaml               # Canonical Base configuration
│   ├── dinov3_b2_crack500_frozen_full.yaml   # Config A: Full Freeze (BS=16)
│   └── dinov3_b2_crack500_frozen_stage1.yaml # Config B: Stage 2 Unfreeze 1e-5 (BS=10)
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

## 🔬 Training Configurations

We provide two verified training regimes on the **Crack500** benchmark:

| Mode | Config File | Batch Size | Stage 1 (Epoch 1-17) | Stage 2 (Epoch 18-35) | Peak VRAM | Speed |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **Config A** *(Full Freeze)* | `configs/dinov3_b2_crack500_frozen_full.yaml` | **16** | Backbone Frozen | Backbone Frozen | 9.3 GB | ~10.2 img/s |
| **Config B** *(Delayed Unfreeze)* | `configs/dinov3_b2_crack500_frozen_stage1.yaml` | **10** | Backbone Frozen | Backbone Unfrozen ($\text{LR}=10^{-5}$) | 10.5 GB | ~12.5 img/s |

---

## ⚡ Quick Start (Google Colab / Tesla T4)

### 1. Setup & Environment

```bash
# Clone repository
!git clone https://github.com/bach0823/sage_dinov3.git /content/sage_dinov3
%cd /content/sage_dinov3

# Install dependencies and download Crack500 dataset
!pip install -q -r requirements.txt
!python prepare_data/prepare_crack500.py
```

### 2. Run Training

* **Train Config A (Full Freeze - Fast & Lightweight):**
```bash
%cd /content/sage_dinov3
!python scripts/train_crack.py \
  --config configs/dinov3_b2_crack500_frozen_full.yaml \
  --data-root /content/dataset/Crack500
```

* **Train Config B (Delayed Unfreeze - Foundation Fine-tuning):**
```bash
%cd /content/sage_dinov3
!python scripts/train_crack.py \
  --config configs/dinov3_b2_crack500_frozen_stage1.yaml \
  --data-root /content/dataset/Crack500
```

### 3. Post-Training Routing Diagnostics & Archival

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

# 2. Package checkpoints (Best + Last models) and logs
archive_name = "/content/dinov3_sage_b2_frozen_stage1_run"
shutil.make_archive(archive_name, "zip", "results", "dinov3_sage_b2_frozen_stage1")

# 3. Download to local machine
files.download(f"{archive_name}.zip")
```

---

## 🧪 Preflight & Verification Suite

Before launching extensive runs, verify AMP FP16 numerical stability and sparse adapter routing:

```bash
python scripts/tests/test_amp_stability.py
```

Expected output:
- `FP32 Baseline`: Loss finite, NaN = 0, Inf = 0.
- `AMP FP16`: Stable gradient backpropagation with cuDNN enabled.
- `SA-Hub Adapter Audit`: Active adapters tracked across batches.

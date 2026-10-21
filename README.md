# SAGE-DINOv3: Shape-Adapting Gated Experts with DINOv3 for Crack Boundary Segmentation

*Advancing Mixture of Local Experts (MoLEx) with Meta DINOv3 Vision Foundation Backbones*

---

Official PyTorch implementation of **SAGE-DINOv3**, an advanced Mixture of Local Experts (MoLEx) architecture tailored for fine-grained boundary and crack segmentation. SAGE-DINOv3 inherits the foundational theoretical principles of **SAGE** and upgrades the feature extraction pipeline with **Meta DINOv3 vision foundation backbones** (ConvNeXt-Tiny + ViT-Small with Dynamic RoPE), cross-space adaptation (SA-Hub), and boundary-aware optimization.

> [!NOTE]
> **Academic Origin & Lineage:**  
> This project builds upon the original work:  
> *“SAGE: Shape-Adapting Gated Experts for Adaptive Histopathology Image Segmentation”* (CVPR 2026 Findings) by Gia Huy Thai, Hoang-Nguyen Vu, Anh-Minh Phan, Quang-Thinh Ly, Thi-Ngoc-Truc Nguyen, and Nhat Ho.  
> SAGE-DINOv3 adapts the SAGE framework from medical histopathology to industrial and civil infrastructure surface crack inspection using modern self-supervised foundation representations.

---

## 🔬 Implementation Highlights — Key Code Architecture

Integration follows a modular, non-destructive design pattern. Pretrained foundation representations from Meta DINOv3 provide rich multi-scale feature hierarchies without requiring full end-to-end retraining from scratch.

### 1. Hybrid DINOv3 Dual-Stream Backbone

`HybridDINOv3Backbone` (`sage/networks/convnextv2_vit_hybrid.py`) instantiates two complementary DINOv3 vision foundation backbones loaded directly via `timm`:
* **ConvNeXt-Tiny DINOv3** (`convnext_tiny.dinov3_lvd1689m`): 4 convolution stages ($C \in [96, 192, 384, 768]$) capturing local high-frequency crack edges, micro-texture fissures, and concrete surface roughness.
* **ViT-Small DINOv3** (`vit_small_patch16_dinov3.lvd1689m`): 4 Vision Transformer blocks ($D=384$) equipped with Dynamic RoPE (Rotary Position Embeddings) capturing global topological crack continuity.

```python
# sage/networks/convnextv2_vit_hybrid.py
self.convnext = timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=pretrained)
vit_full = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=pretrained)
# Extract the first 4 transformer blocks
self.transformer_blocks = nn.ModuleList([vit_full.blocks[i] for i in range(num_transformer_layers)])
```

Spatial-to-sequence bridge projections align the 2D feature maps from ConvNeXt Stage 2 ($384$ channels) into the 1D token space expected by the ViT Transformer blocks:
```python
# sage/networks/convnextv2_vit_hybrid.py
self.convnext_to_transformer = nn.Sequential(
    nn.Conv2d(384, 384, kernel_size=1),
    nn.BatchNorm2d(384),
    nn.GELU()
)
```

### 2. In-Place SAGE Layer Injection & Heterogeneous 8-Expert Pool

In `B2ConvNeXtViTUNet` (`sage/networks/b2_unet.py`), the 4 CNN stages and 4 ViT blocks are wrapped in-place into `SageLayer` modules, constructing a **heterogeneous pool of 8 experts**:

```python
# sage/networks/b2_unet.py — inject_sage_layers()
self.expert_pool = inject_sage_layers(
    convnext=self.backbone.convnext,
    transformer_blocks=self.backbone.transformer_blocks,
    encoder_channels=[96, 192, 384, 768],
    transformer_dim=384,
    sage_config=self.sage_config,
)
```

Inside each `SageLayer` (`sage/components/sage_layer.py`), the main layer computation is dynamically modulated by routed expert outputs:
$$\mathbf{y} = \alpha \cdot \mathbf{y}_{\text{main}} + (1 - \alpha) \cdot \mathbf{y}_{\text{expert}}$$

Where $\mathbf{y}_{\text{expert}}$ is dynamically computed by routing patches to the **Top-2** most relevant experts selected from the 8-expert pool via `SageRouter` (`sage/components/router.py`).

### 3. Cross-Space SA-Hub Adapters

To allow 2D CNN feature maps to be processed by 1D ViT blocks and vice-versa, the **Shape-Adapting Hub** (`sage/components/sa_hub.py`) implements pairwise cross-space projection adapters:
* **CNN $\to$ ViT:** $1 \times 1\text{ Conv} \to \text{Flatten spatial dimensions } (H, W) \to \text{Patch sequence } (N, D)$.
* **ViT $\to$ CNN:** $\text{Reshape token sequence to 2D } \to \text{Bilinear Interpolation to target resolution } (H, W) \to 1 \times 1\text{ Conv}$.

---

## ⚖️ Trainable vs. Frozen Parameter Regimes

SAGE-DINOv3 supports two verified parameter regimes on the **Crack500** benchmark:

### Parameter Status by Module and Stage

| Module / Component | Config A (Stage 1 & 2) | Config B (Stage 1: Ep 1–17) | Config B (Stage 2: Ep 18–35) | Optimizer Learning Rate |
| :--- | :---: | :---: | :---: | :---: |
| **DINOv3 ConvNeXt Stages** (`.main_block`) | **Frozen** (`grad=None`) | **Frozen** (`grad=None`) | **Trainable** (Unfrozen) | $\text{LR} = 10^{-5}$ *(Config B Stage 2)* |
| **DINOv3 ViT Blocks** (`.main_block`) | **Frozen** (`grad=None`) | **Frozen** (`grad=None`) | **Trainable** (Unfrozen) | $\text{LR} = 10^{-5}$ *(Config B Stage 2)* |
| **Bridge Projections** (`convnext_to_transformer`) | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 10^{-4}$ |
| **8 SAGE Routers** (`SageRouter`) | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 2 \times 10^{-4}$ (S1) / $10^{-4}$ (S2) |
| **SA-Hub Pairwise Adapters** | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 2 \times 10^{-4}$ (S1) / $10^{-4}$ (S2) |
| **Progressive UNet Decoder** | **Trainable** | **Trainable** | **Trainable** | $\text{LR} = 10^{-4}$ |
| **Soft Boundary IoU Loss** | *N/A (Loss function)* | *N/A (Loss function)* | *N/A (Loss function)* | *No parameters to update* |

---

## 📐 Loss Formulation & Gradient Path

The total optimization objective $\mathcal{L}_{\text{total}}$ consists of three complementary components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}}(\mathbf{z}, \mathbf{y}) + \lambda_{\text{bound}} \cdot \mathcal{L}_{\text{bound}}(\mathbf{z}, \mathbf{y}) + \lambda_{\text{lb}} \cdot \mathcal{L}_{\text{lb}}(\mathbf{G})$$

### 1. Primary Segmentation Loss ($\mathcal{L}_{\text{seg}}$)
Combines Binary Cross-Entropy with Dice Loss:
$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{BCE}}(\sigma(\mathbf{z}), \mathbf{y}) + \mathcal{L}_{\text{Dice}}(\sigma(\mathbf{z}), \mathbf{y})$$

### 2. Soft Boundary IoU Loss ($\mathcal{L}_{\text{bound}}$)
Tailored specifically for elongated, thin crack fissures. A differentiable morphological distance transform extracts boundary shells $\mathcal{B}(\mathbf{y})$:
$$\mathcal{L}_{\text{bound}} = 1 - \frac{|\mathcal{B}(\sigma(\mathbf{z})) \cap \mathcal{B}(\mathbf{y})| + \epsilon}{|\mathcal{B}(\sigma(\mathbf{z})) \cup \mathcal{B}(\mathbf{y})| + \epsilon}$$
With boundary weight $\lambda_{\text{bound}} = 0.50$ and dilation radius $d = 2$.

### 3. Load Balancing Loss ($\mathcal{L}_{\text{lb}}$)
Enforces uniform expert utilization across the 8 heterogeneous experts to prevent router collapse:
$$\mathcal{L}_{\text{lb}} = E \sum_{i=1}^{E} f_i \cdot P_i$$
Where $f_i$ is the fraction of tokens dispatched to expert $i$, and $P_i$ is the average gating probability assigned to expert $i$ ($\lambda_{\text{lb}} = 0.01$).

---

## 📊 Empirical Benchmarks on NVIDIA Tesla T4

| Experiment Mode | Config File | Batch Size | Peak VRAM | Speed | 1 Epoch Time | Full 35 Epochs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A** *(Full Freeze)* | `configs/dinov3_b2_crack500_frozen_full.yaml` | **16** | **9,353 MB** | **9.83 img/s** | ~3.2 min | **~1.88 hours** |
| **Config B** *(Stage 2 Unfreeze)* | `configs/dinov3_b2_crack500_frozen_stage1.yaml` | **10** | **10,477 MB** | **7.01 img/s** | ~4.5 min | **~2.64 hours** |

---

## 📁 Repository Structure

```text
sage_dinov3/
├── configs/                  # Active training configurations
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
├── prepare_data/             # Dataset preprocessing scripts
│   └── prepare_crack500.py   # Automated Crack500 setup
└── requirements.txt          # Python dependencies
```

---

## ⚡ Execution Guide

### Step 1: Environment Setup
```bash
git clone https://github.com/Lewis-26/sage_dinov3.git
cd sage_dinov3
pip install -r requirements.txt
python prepare_data/prepare_crack500.py
```

### Step 2: Training

* **Config A (Full Freeze):**
```bash
python scripts/train_crack.py --config configs/dinov3_b2_crack500_frozen_full.yaml
```

* **Config B (Stage 2 Dynamic Unfreeze):**
```bash
python scripts/train_crack.py --config configs/dinov3_b2_crack500_frozen_stage1.yaml
```

### Step 3: Evaluation
```bash
python scripts/evaluate_crack_official.py \
    --config configs/dinov3_b2_crack500_frozen_stage1.yaml \
    --checkpoint results/dinov3_sage_b2_frozen_stage1/best_model_b2_global.pth
```



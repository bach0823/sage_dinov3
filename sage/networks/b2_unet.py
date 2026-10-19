"""
Baseline Ladder B2: Full SAGE-DINOv3 Model
ConvNeXt-Tiny DINOv3 + ViT-Small DINOv3 Hybrid UNet with SAGE 8 Experts & 8 Routers.

Clean architecture:
- No S2-Gate, PointRend, CGSR, PLU, Tangent head, or Strip Pooling.
- Fully compatible with DINOv3 channels [96, 192, 384, 768] and ViT-Small (384).
- 8 Heterogeneous Experts (4 CNN + 4 ViT).
- 8 SAGE Routers (top_k=2, shared_expert_indices=[0, 1, 2, 3]).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnextv2_vit_hybrid import ConvNeXtV2ViTHybrid
from .decoder_block import UNetDecoder
from .sage_injection import inject_sage_layers, pre_populate_sa_hubs
from ..components.sage_layer import SageLayer
from ..components.router import SageRouter

logger = logging.getLogger(__name__)

DEFAULT_SAGE_CONFIG: Dict[str, Any] = {
    "top_k": 2,
    "gating_type": "sigmoid",
    "shared_expert_indices": [0, 1, 2, 3],
    "router_hidden_dim": 64,
    "load_balance_factor": 0.01,
    "logit_modulation": True,
    "expert_dropout": 0.1,
    "fusion_type": "residual",
    "residual_scale": 0.1,
    "adaptive_alpha": 0.9,
}


class B2ConvNeXtViTUNet(nn.Module):
    """
    SAGE-DINOv3 Model: ConvNeXt-Tiny + ViT-Small with 8 Experts & 8 Routers.
    """

    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 448,
        num_transformer_layers: int = 4,
        freeze_encoder: bool = False,
        freeze_transformer: bool = False,
        use_dwsc: bool = False,
        pretrained: bool = True,
        sage_config: Optional[Dict[str, Any]] = None,
        weights_dir: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_transformer_layers = num_transformer_layers

        # 1. Merge SAGE config with defaults
        self.sage_config = dict(DEFAULT_SAGE_CONFIG)
        if sage_config is not None:
            self.sage_config.update(sage_config)

        # 2. Backbone: DINOv3 ConvNeXt-Tiny + ViT-Small (4 blocks)
        self.backbone = ConvNeXtV2ViTHybrid(
            img_size=img_size,
            convnext_model_name="convnext_tiny.dinov3_lvd1689m",
            vit_model_name="vit_small_patch16_dinov3.lvd1689m",
            num_transformer_layers=num_transformer_layers,
            freeze_encoder=freeze_encoder,
            freeze_transformer=freeze_transformer,
            pretrained=pretrained,
            weights_dir=weights_dir,
        )

        # 3. Pure UNet Decoder (Clean: No S2G, No PointRend, No CGSR)
        self.decoder = UNetDecoder(
            encoder_channels=self.backbone.encoder_channels,
            num_classes=num_classes,
            use_dwsc=use_dwsc,
        )

        # 4. Inject SAGE wrappers in-place and construct shared expert pool (8 experts: 4 CNN + 4 ViT)
        self.expert_pool = inject_sage_layers(
            convnext=self.backbone.convnext,
            transformer_blocks=self.backbone.transformer_blocks,
            encoder_channels=self.backbone.encoder_channels,
            stem_channels=self.backbone.stem_channels,
            transformer_dim=self.backbone.transformer_dim,
            sage_config=self.sage_config,
            pe_owner=None,
        )

        # 5. Pre-populate all SA-Hub adapters eagerly (O(D^2) pairwise)
        pre_populate_sa_hubs(
            model=self,
            encoder_channels=self.backbone.encoder_channels,
            stem_channels=self.backbone.stem_channels,
            transformer_dim=self.backbone.transformer_dim,
        )

    @property
    def num_sage_experts(self) -> int:
        return self.backbone.num_sage_experts

    def set_shared_experts(self, shared_expert_indices: List[int]) -> None:
        self.sage_config["shared_expert_indices"] = list(shared_expert_indices)
        for module in self.modules():
            if isinstance(module, SageRouter):
                module.shared_expert_indices = list(shared_expert_indices)
                if hasattr(module, "_update_shared_mask"):
                    module._update_shared_mask()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        feat_dict = self.backbone(x)
        skips = feat_dict["skips"]
        bottleneck = feat_dict["bottleneck"]

        out = self.decoder(
            bottleneck=bottleneck,
            skips=skips,
            target_size=target_size,
        )
        return out

    def forward_with_routing_info(self, x: torch.Tensor) -> Dict[str, Any]:
        logits = self.forward(x)

        cnn_routing_infos = []
        for stage in self.backbone.convnext.stages:
            if isinstance(stage, SageLayer):
                cnn_routing_infos.append(stage.get_last_routing_info())

        transformer_routing_infos = []
        for blk in self.backbone.transformer_blocks:
            if isinstance(blk, SageLayer):
                transformer_routing_infos.append(blk.get_last_routing_info())

        all_routing_infos = cnn_routing_infos + transformer_routing_infos

        return {
            "logits": logits,
            "routing_infos": {
                "cnn": cnn_routing_infos,
                "transformer": transformer_routing_infos,
                "all": all_routing_infos,
            },
        }

    def compute_total_load_balance_loss(
        self,
        routing_infos: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> torch.Tensor:
        if isinstance(routing_infos, dict):
            if "all" in routing_infos:
                infos = routing_infos["all"]
            else:
                infos = routing_infos.get("cnn", []) + routing_infos.get("transformer", [])
        elif isinstance(routing_infos, list):
            infos = routing_infos
        else:
            infos = []

        total_loss = None
        for info in infos:
            if info and isinstance(info, dict) and "load_balance_loss" in info:
                lb = info["load_balance_loss"]
                if isinstance(lb, torch.Tensor):
                    total_loss = lb if total_loss is None else (total_loss + lb)

        if total_loss is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return total_loss

    def get_expert_usage_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, List[Dict[str, Any]]] = {
            "cnn_stages": [],
            "transformer_blocks": [],
        }
        for stage in self.backbone.convnext.stages:
            if isinstance(stage, SageLayer) and hasattr(stage, "get_stats"):
                stats["cnn_stages"].append(stage.get_stats())

        for blk in self.backbone.transformer_blocks:
            if isinstance(blk, SageLayer) and hasattr(blk, "get_stats"):
                stats["transformer_blocks"].append(blk.get_stats())
        return stats

    def get_model_info(self) -> Dict[str, Union[int, str, List[int]]]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sage_layers = [m for m in self.modules() if isinstance(m, SageLayer)]

        return {
            "model_name": "B2ConvNeXtViTUNet (SAGE-DINOv3 Clean)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "num_transformer_layers": self.num_transformer_layers,
            "num_sage_experts": self.num_sage_experts,
            "num_injected_routers": len(sage_layers),
            "expert_pool_size": len(self.expert_pool) if self.expert_pool is not None else 0,
            "top_k": self.sage_config.get("top_k", 2),
            "router_hidden_dim": self.sage_config.get("router_hidden_dim", 64),
            "gating_type": self.sage_config.get("gating_type", "sigmoid"),
            "fusion_type": self.sage_config.get("fusion_type", "residual"),
            "residual_scale": self.sage_config.get("residual_scale", 0.1),
        }

    def load_stage1_state_dict(self, state_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        state_dict = {k: v for k, v in state_dict.items() if 'p3_refinement' not in k}
        missing, unexpected = self.load_state_dict(state_dict, strict=True)
        logger.info("Successfully loaded Stage 1 checkpoint into Stage 2 (strict=True, 0 missing, 0 unexpected).")
        return missing, unexpected


def create_b2_unet(
    num_classes: int = 1,
    img_size: int = 448,
    num_transformer_layers: int = 4,
    freeze_encoder: bool = False,
    freeze_transformer: bool = False,
    use_dwsc: bool = False,
    pretrained: bool = True,
    sage_config: Optional[Dict[str, Any]] = None,
    weights_dir: Optional[str] = None,
    **kwargs,
) -> B2ConvNeXtViTUNet:
    return B2ConvNeXtViTUNet(
        num_classes=num_classes,
        img_size=img_size,
        num_transformer_layers=num_transformer_layers,
        freeze_encoder=freeze_encoder,
        freeze_transformer=freeze_transformer,
        use_dwsc=use_dwsc,
        pretrained=pretrained,
        sage_config=sage_config,
        weights_dir=weights_dir,
    )

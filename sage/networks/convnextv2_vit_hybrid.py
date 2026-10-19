"""
ConvNeXt-Tiny + ViT-Small Hybrid Backbone for Crack Segmentation (SAGE-DINOv3)

Integrated for DINOv3 architecture:
- ConvNeXt-Tiny (DINOv3 pretrained, channels: [96, 192, 384, 768])
- ViT-Small (DINOv3 pretrained Eva blocks, embed_dim: 384, default depth: 4 blocks)
- Dynamic RoPE attention routing for 14x14 bottleneck and cross-resolution calls
- Strictly excludes CLS and register tokens for dense prediction
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Union

from safetensors.torch import load_file
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CONVNEXT_TINY_DINOV3_CHANNELS: List[int] = [96, 192, 384, 768]
VIT_SMALL_DINOV3_EMBED_DIM: int = 384
DEFAULT_IMG_SIZE: int = 448
BOTTLENECK_GRID_SIZE: int = 14  # 448 / 32 = 14 -> 196 tokens


class DynamicEvaBlockRoPEWrapper(nn.Module):
    """
    Wrapper for DINOv3 EvaBlock that computes Dynamic RoPE based on sequence length.
    Supports both 14x14 (196 tokens) in bottleneck and 28x28 (784 tokens) in S0/S1 cross-calls.
    """

    def __init__(self, block: nn.Module, rope_module: nn.Module):
        super().__init__()
        self.block = block
        self.rope_module = rope_module
        if hasattr(self.block, "attn") and hasattr(self.block.attn, "num_prefix_tokens"):
            self.block.attn.num_prefix_tokens = 0

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        seq_len = x.shape[1]
        grid = int(math.isqrt(seq_len))
        rope = self.rope_module.get_embed(shape=(grid, grid))
        return self.block(x, rope=rope)

    def __getattr__(self, name: str):
        protected = {
            "block",
            "rope_module",
            "_modules",
            "_parameters",
            "_buffers",
            "_backward_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_modules_containers",
        }
        if name in protected:
            return super().__getattr__(name)
        return getattr(self.block, name)


class ConvNeXtV2ViTHybrid(nn.Module):
    """
    DINOv3 Hybrid Backbone: ConvNeXt-Tiny + ViT-Small (4 blocks) for SAGE-Lite.
    """

    def __init__(
        self,
        img_size: int = DEFAULT_IMG_SIZE,
        convnext_model_name: str = "convnext_tiny.dinov3_lvd1689m",
        vit_model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        num_transformer_layers: int = 4,
        freeze_encoder: bool = False,
        freeze_transformer: bool = False,
        pretrained: bool = True,
        weights_dir: Optional[str] = None,
    ):
        super().__init__()

        self.img_size = img_size
        self.num_transformer_layers = num_transformer_layers
        if weights_dir is None:
            # Default weights dir relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "weights"))

        # ====================================================================
        # 1. ENCODER: ConvNeXt-Tiny DINOv3 Backbone
        # ====================================================================
        logger.info(f"Loading ConvNeXt-Tiny DINOv3: {convnext_model_name}")
        self.convnext = timm.create_model(
            convnext_model_name,
            pretrained=False,
            features_only=False,
        )

        conv_weight_path = os.path.join(weights_dir, "convnext_tiny_dinov3.safetensors")
        loaded_conv = False
        if pretrained and os.path.exists(conv_weight_path) and os.path.getsize(conv_weight_path) > 1024 * 1024:
            try:
                sd_c = load_file(conv_weight_path)
                self.convnext.load_state_dict(sd_c, strict=False)
                logger.info(f"Loaded pretrained weights for ConvNeXt from local {conv_weight_path}")
                loaded_conv = True
            except Exception as e:
                logger.warning(f"Failed loading local {conv_weight_path}: {e}")

        if pretrained and not loaded_conv:
            logger.info(f"Downloading official pretrained weights for ConvNeXt-Tiny ({convnext_model_name}) from timm/HF...")
            m_temp = timm.create_model(convnext_model_name, pretrained=True)
            self.convnext.load_state_dict(m_temp.state_dict(), strict=False)
            del m_temp
            logger.info("Successfully loaded official timm pretrained weights for ConvNeXt.")

        dummy_input = torch.randn(1, 3, img_size, img_size)
        self.encoder_channels: List[int] = []
        self.encoder_spatial_sizes: List[int] = []

        with torch.no_grad():
            x = self.convnext.stem(dummy_input)
            self.stem_channels = x.shape[1]
            for stage in self.convnext.stages:
                x = stage(x)
                self.encoder_channels.append(x.shape[1])
                self.encoder_spatial_sizes.append(x.shape[2])

        assert self.encoder_channels == CONVNEXT_TINY_DINOV3_CHANNELS, (
            f"Channel contract mismatch: Expected {CONVNEXT_TINY_DINOV3_CHANNELS}, got {self.encoder_channels}"
        )

        if hasattr(self.convnext, "head"):
            del self.convnext.head

        if freeze_encoder:
            for param in self.convnext.parameters():
                param.requires_grad = False
            logger.info("ConvNeXt encoder parameters frozen")

        # ====================================================================
        # 2. BOTTLENECK: ViT-Small DINOv3 (Configurable blocks, default=4)
        # ====================================================================
        logger.info(f"Loading ViT-Small DINOv3: {vit_model_name}")
        vit_full = timm.create_model(vit_model_name, pretrained=False)
        vit_weight_path = os.path.join(weights_dir, "vit_small_dinov3.safetensors")
        loaded_vit = False
        if pretrained and os.path.exists(vit_weight_path) and os.path.getsize(vit_weight_path) > 1024 * 1024:
            try:
                sd_v = load_file(vit_weight_path)
                vit_full.load_state_dict(sd_v, strict=False)
                logger.info(f"Loaded pretrained weights for ViT-Small from local {vit_weight_path}")
                loaded_vit = True
            except Exception as e:
                logger.warning(f"Failed loading local {vit_weight_path}: {e}")

        if pretrained and not loaded_vit:
            logger.info(f"Downloading official pretrained weights for ViT-Small ({vit_model_name}) from timm/HF...")
            m_temp = timm.create_model(vit_model_name, pretrained=True)
            vit_full.load_state_dict(m_temp.state_dict(), strict=False)
            del m_temp
            logger.info("Successfully loaded official timm pretrained weights for ViT-Small.")

        self.transformer_dim = vit_full.embed_dim  # 384
        self.num_heads = getattr(vit_full, "num_heads", 6)
        self.rope = vit_full.rope

        all_blocks = vit_full.blocks
        assert 1 <= self.num_transformer_layers <= len(all_blocks), (
            f"Requested {self.num_transformer_layers} blocks, but {vit_model_name} only has {len(all_blocks)} blocks"
        )

        # Wrap first num_transformer_layers blocks with Dynamic RoPE wrapper
        self.transformer_blocks = nn.ModuleList([
            DynamicEvaBlockRoPEWrapper(all_blocks[i], self.rope)
            for i in range(self.num_transformer_layers)
        ])
        logger.info(f"Configured {len(self.transformer_blocks)} DINOv3 ViT blocks with Dynamic RoPE")

        del vit_full

        if freeze_transformer:
            for block in self.transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            logger.info("Transformer blocks frozen")

        # ====================================================================
        # 3. INTERFACE PROJECTIONS: ConvNeXt <-> ViT
        # ====================================================================
        self.bottleneck_channels = self.encoder_channels[-1]  # 768
        self.bottleneck_spatial_size = self.encoder_spatial_sizes[-1]  # 14

        # Linear projection: ConvNeXt channels (768) -> ViT dimension (384)
        self.convnext_to_transformer = nn.Linear(self.bottleneck_channels, self.transformer_dim)
        self.pre_transformer_norm = nn.LayerNorm(self.transformer_dim)

        # Reprojection: ViT dimension (384) -> Bottleneck channels (768)
        self.transformer_to_decoder = nn.Linear(self.transformer_dim, self.bottleneck_channels)
        self.post_transformer_norm = nn.LayerNorm(self.bottleneck_channels)

    @property
    def num_sage_experts(self) -> int:
        return len(self.encoder_channels)

    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x_feat = self.convnext.stem(x)
        for stage in self.convnext.stages:
            x_feat = stage(x_feat)
            features.append(x_feat)
        return features

    def forward_bottleneck(self, stage3_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = stage3_feat.shape
        num_spatial_patches = h * w  # 14 * 14 = 196

        # 1. Flatten to spatial tokens: (B, 768, H, W) -> (B, H*W, 768)
        tokens = stage3_feat.flatten(2).transpose(1, 2)
        assert tokens.shape[1] == num_spatial_patches

        # 2. Linear projection: 768 -> 384
        tokens = self.convnext_to_transformer(tokens)
        tokens = self.pre_transformer_norm(tokens)

        # 3. Forward through ViT blocks (Dynamic RoPE injected automatically)
        for blk in self.transformer_blocks:
            tokens = blk(tokens)

        # 4. Reproject back to bottleneck channels: 384 -> 768
        out_tokens = self.transformer_to_decoder(tokens)
        out_tokens = self.post_transformer_norm(out_tokens)

        # 5. Reshape back to 2D spatial grid: (B, 768, H, W)
        bottleneck_spatial = out_tokens.transpose(1, 2).reshape(b, c, h, w)
        return bottleneck_spatial, tokens

    def forward(self, x: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        encoder_features = self.get_encoder_features(x)
        skips = encoder_features[:-1]  # Stage 0 (96), Stage 1 (192), Stage 2 (384)
        stage3_feat = encoder_features[-1]  # Stage 3 (768)

        bottleneck_spatial, tokens = self.forward_bottleneck(stage3_feat)

        return {
            "skips": skips,
            "bottleneck": bottleneck_spatial,
            "tokens": tokens,
        }

    def get_model_info(self) -> Dict[str, Union[int, str, List[int]]]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "ConvNeXtV2ViTHybrid (SAGE-DINOv3)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_channels": self.encoder_channels,
            "num_transformer_layers": len(self.transformer_blocks),
            "num_sage_experts": self.num_sage_experts,
        }


def create_convnextv2_vit_hybrid(
    img_size: int = DEFAULT_IMG_SIZE,
    num_transformer_layers: int = 4,
    freeze_encoder: bool = False,
    freeze_transformer: bool = False,
    pretrained: bool = True,
    weights_dir: Optional[str] = None,
) -> ConvNeXtV2ViTHybrid:
    return ConvNeXtV2ViTHybrid(
        img_size=img_size,
        num_transformer_layers=num_transformer_layers,
        freeze_encoder=freeze_encoder,
        freeze_transformer=freeze_transformer,
        pretrained=pretrained,
        weights_dir=weights_dir,
    )

"""
UNet Decoder Blocks and Clean Decoder Architecture for SAGE-DINOv3

Pure UNet Decoder without S2-Gate, PointRend, CGSR, PLU, Tangent head, or Strip Pooling.
Channels are dynamically configured based on encoder_channels ([96, 192, 384, 768]).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def make_conv3x3(in_ch: int, out_ch: int, use_dwsc: bool = False) -> nn.Module:
    """Constructs a 3x3 convolution layer."""
    if use_dwsc:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
        )
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)


class DecoderBlock(nn.Module):
    """
    Single UNet Decoder Block with 2x upsampling, skip concatenation, and double conv3x3.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_dwsc: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.use_dwsc = use_dwsc

        # 2x Transposed Convolution for spatial upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )

        combined_channels = in_channels + skip_channels

        self.conv1 = nn.Sequential(
            make_conv3x3(combined_channels, out_channels, use_dwsc=use_dwsc),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            make_conv3x3(out_channels, out_channels, use_dwsc=use_dwsc),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder block.
        """
        x_up = self.upsample(x)

        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(
                x_up,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        out = torch.cat([x_up, skip], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class UNetDecoder(nn.Module):
    """
    Pure UNet Decoder for SAGE-DINOv3.
    """

    def __init__(
        self,
        encoder_channels: List[int] = [96, 192, 384, 768],
        num_classes: int = 1,
        use_dwsc: bool = False,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_classes
        self.use_dwsc = use_dwsc

        # Decoder Blocks:
        # Block 0: in = 768, skip = 384 (Stage 2) -> out = 384 @ 28x28
        # Block 1: in = 384, skip = 192 (Stage 1) -> out = 192 @ 56x56
        # Block 2: in = 192, skip = 96  (Stage 0) -> out = 96  @ 112x112
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                in_channels=encoder_channels[3],
                skip_channels=encoder_channels[2],
                out_channels=encoder_channels[2],
                use_dwsc=use_dwsc,
            ),
            DecoderBlock(
                in_channels=encoder_channels[2],
                skip_channels=encoder_channels[1],
                out_channels=encoder_channels[1],
                use_dwsc=use_dwsc,
            ),
            DecoderBlock(
                in_channels=encoder_channels[1],
                skip_channels=encoder_channels[0],
                out_channels=encoder_channels[0],
                use_dwsc=use_dwsc,
            ),
        ])

        # Segmentation Head: Stage 0 out (96) -> 48 -> num_classes (1) + 4x Bilinear Upsample
        head_mid_channels = encoder_channels[0] // 2
        self.segmentation_head = nn.Sequential(
            make_conv3x3(encoder_channels[0], head_mid_channels, use_dwsc=use_dwsc),
            nn.BatchNorm2d(head_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_mid_channels, num_classes, kernel_size=1),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        reversed_skips = list(reversed(skips))
        assert len(reversed_skips) == len(self.decoder_blocks), (
            f"Expected {len(self.decoder_blocks)} skip connections, got {len(reversed_skips)}"
        )

        x_dec = bottleneck
        for i, block in enumerate(self.decoder_blocks):
            skip = reversed_skips[i]
            x_dec = block(x_dec, skip)

        logits_low = self.segmentation_head(x_dec)

        if target_size is not None:
            logits = F.interpolate(
                logits_low,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            logits = F.interpolate(
                logits_low,
                scale_factor=4.0,
                mode="bilinear",
                align_corners=False,
            )

        return logits

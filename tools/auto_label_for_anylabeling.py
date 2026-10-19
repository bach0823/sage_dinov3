#!/usr/bin/env python3
"""
SAGE-Lite Auto-Labeling Tool for AnyLabeling / LabelMe
======================================================
High-Resolution Batch Inference on crack images using SAGE-Lite Phase 7 model.
Features:
- Tiling / Sliding Window with 50% Overlap and Snap-to-Edge (Zero fake padding).
- Mini-batch GPU inference with automatic Turing GTX 16xx AMP-safe fallback.
- Smooth probability accumulation and border artifact elimination.
- High-precision polygon contour vectorization via cv2.approxPolyDP.
- Full compliance with AnyLabeling / LabelMe JSON standard (v0.4.43).

Author: Special Subject AI Team
Date: October 2026
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.networks.b2_unet import create_b2_unet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AutoLabeling")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-labeling crack dataset using SAGE-Lite (Tiling Overlap 50% + Snap-to-Edge)."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=None,
        help="Path to folder containing images to be labeled (e.g. C:/Users/.../dataset/train)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image to test (overrides --input-dir if provided)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSON files. If None, saves JSONs directly alongside original images."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml",
        help="Path to SAGE-Lite model config YAML."
    )
    parser.add_argument(
        "-k", "--checkpoint",
        type=str,
        default="results/phase7_combination/phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth",
        help="Path to model checkpoint (.pth)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tiling", "direct"],
        default="tiling",
        help="Inference mode: 'tiling' (50%% overlap + snap-to-edge, default) or 'direct' (downscale 448x448)."
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=448,
        help="Sliding window tile size (default: 448)."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="Sliding window overlap ratio in range [0.0, 0.9] (default: 0.20, i.e., 20%% overlap)."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding window stride. If None, auto-computed as int(tile_size * (1 - overlap)) (default: 358 for 20%% overlap)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for GPU tile inference (default: 16)."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarization (default: 0.5)."
    )
    parser.add_argument(
        "-m", "--min-area",
        type=float,
        default=30.0,
        help="Minimum contour area in pixels to filter out noise (default: 30.0)."
    )
    parser.add_argument(
        "-e", "--epsilon-ratio",
        type=float,
        default=0.002,
        help="Ramer-Douglas-Peucker epsilon multiplier (epsilon = ratio * arcLength, default: 0.002)."
    )
    parser.add_argument(
        "-l", "--label",
        type=str,
        default="crack",
        help="Label name for detected polygons (default: 'crack')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files. If false, skips already labeled images."
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also export predicted binary masks as PNGs alongside JSON."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export overlay visualizations with polygon boundaries drawn on image."
    )
    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Build SAGE-Lite B2 UNet and load trained weights."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_file}")
    vit_depth = int(config.get("num_transformer_layers", 12))
    sage_cfg = config.get("sage_config", {})
    use_plu_head = config.get("use_plu_head", False)

    logger.info(
        f"Instantiating B2 architecture: vit_depth={vit_depth}, use_plu_head={use_plu_head}, "
        f"s2_gate={config.get('use_s2_gate', False)}, s2_gate_b2={config.get('use_s2_gate_block2', False)}"
    )

    model = create_b2_unet(
        num_transformer_layers=vit_depth,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode=config.get("p3_mode"),
        use_plu_head=use_plu_head,
        use_s2_gate=config.get("use_s2_gate", False),
        s2_gate_kernel_size=config.get("s2_gate_kernel_size", 3),
        use_s2_gate_block2=config.get("use_s2_gate_block2", False),
        s2_gate_block2_kernel_size=config.get("s2_gate_block2_kernel_size", 3),
        s2_gate_block2_dilation=config.get("s2_gate_block2_dilation", 1),
    ).to(device)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_absolute():
        ckpt_file = PROJECT_ROOT / ckpt_file

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    logger.info(f"Loading checkpoint weights from {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    logger.info(f"Model state dict loaded cleanly. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.eval()
    return model


def get_snap_coords(length: int, tile_size: int = 448, stride: int = 224) -> List[int]:
    """
    Generate 1D grid coordinates with overlap and snap-to-edge guarantee.
    Zero artificial padding: the final tile snaps exactly to (length - tile_size).
    """
    if length <= tile_size:
        return [0]
    coords = list(range(0, length - tile_size, stride))
    last_coord = length - tile_size
    if not coords or coords[-1] != last_coord:
        coords.append(last_coord)
    return coords


def predict_probability_map_tiling(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    tile_size: int = 448,
    stride: int = 224,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Full-Resolution Inference via Tiling (50% Overlap + Snap-to-Edge).
    All pixels processed are authentic pixels from image_bgr.
    Includes GTX 16xx safe AMP fallback to avoid FP16 NaN underflow.
    """
    H, W = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. 1D Snap-to-edge coordinates
    y_coords = get_snap_coords(H, tile_size=tile_size, stride=stride)
    x_coords = get_snap_coords(W, tile_size=tile_size, stride=stride)

    # 2. Extract patches
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    patches = []
    positions = []
    for y in y_coords:
        for x in x_coords:
            patch = img_rgb[y : y + tile_size, x : x + tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patches.append(patch_tensor)
            positions.append((y, x))

    accum_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Detect Turing GTX 1650/1660 which can produce NaNs in FP16 autocast
    is_gtx16xx = False
    if device.type == "cuda":
        dname = torch.cuda.get_device_name(device).lower()
        if "1650" in dname or "1660" in dname:
            is_gtx16xx = True

    # 3. Mini-batch GPU inference
    total_patches = len(patches)
    for i in range(0, total_patches, batch_size):
        batch = torch.stack(patches[i : i + batch_size]).to(device, non_blocking=True)
        batch = (batch - mean) / std

        with torch.no_grad():
            if device.type == "cuda" and not is_gtx16xx:
                with torch.amp.autocast("cuda"):
                    out = model(batch)
            else:
                out = model(batch)

            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)

        # 4. Accumulate into full canvas
        for j, prob_tile in enumerate(probs):
            y, x = positions[i + j]
            accum_map[y : y + tile_size, x : x + tile_size] += prob_tile
            count_map[y : y + tile_size, x : x + tile_size] += 1.0

    # 5. Average probability across overlapping regions
    final_prob_map = accum_map / np.maximum(count_map, 1.0)
    return final_prob_map


def predict_probability_map_direct(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    target_size: int = 448,
) -> np.ndarray:
    """Fast direct downscale inference (448x448) with bilinear interpolation back."""
    orig_h, orig_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    tensor = (tensor - mean) / std

    with torch.no_grad():
        out = model(tensor)
        logits = out["logits"] if isinstance(out, dict) else out
        probs_448 = torch.sigmoid(logits)
        probs_orig = F.interpolate(
            probs_448,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        prob_map = probs_orig[0, 0].cpu().numpy().astype(np.float32)
    return prob_map


def mask_to_polygons(
    binary_mask: np.ndarray,
    min_area: float = 15.0,
    epsilon_ratio: float = 0.002
) -> List[List[List[float]]]:
    """
    Vectorize binary mask into list of valid polygon coordinates.
    Filters contours with area < min_area and simplifies via approxPolyDP.
    Each polygon: [[x1, y1], [x2, y2], ...]
    """
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS
    )

    polygons: List[List[List[float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        if epsilon_ratio > 0.0:
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
        else:
            approx = cnt

        # Must have at least 3 points to form a polygon
        if len(approx) < 3:
            continue

        pts = [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in approx]
        polygons.append(pts)

    return polygons


def build_anylabeling_json(
    image_filename: str,
    orig_h: int,
    orig_w: int,
    polygons: List[List[List[float]]],
    label: str = "crack"
) -> Dict[str, Any]:
    """Construct AnyLabeling / LabelMe 0.4.43 compliant JSON dictionary."""
    shapes = []
    for pts in polygons:
        shape_dict = {
            "label": label,
            "points": pts,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        shapes.append(shape_dict)

    data = {
        "version": "0.4.43",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(orig_h),
        "imageWidth": int(orig_w)
    }
    return data


def draw_overlay(image_bgr: np.ndarray, polygons: List[List[List[float]]]) -> np.ndarray:
    """Draw detected polygons on image for visualization."""
    overlay = image_bgr.copy()
    for pts in polygons:
        np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [np_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.fillPoly(overlay, [np_pts], color=(0, 140, 255))
    return cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)


def process_single_image(
    img_file: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace
) -> Tuple[bool, int]:
    """Process a single image, infer mask, extract polygons, and save JSON."""
    target_json_path = output_path / f"{img_file.stem}.json"
    if target_json_path.exists() and not args.overwrite:
        return False, 0

    # 1. Read Image safely
    img_bytes = np.fromfile(str(img_file), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.warning(f"Failed to read image: {img_file.name}. Skipping.")
        return False, 0

    orig_h, orig_w = img_bgr.shape[:2]

    # 2. Probability prediction
    if args.mode == "tiling":
        prob_map = predict_probability_map_tiling(
            model=model,
            image_bgr=img_bgr,
            device=device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size
        )
    else:
        prob_map = predict_probability_map_direct(
            model=model,
            image_bgr=img_bgr,
            device=device,
            target_size=args.tile_size
        )

    # 3. Binary Thresholding
    binary_mask = (prob_map >= args.threshold).astype(np.uint8) * 255

    # 4. Extract Polygons
    polygons = mask_to_polygons(
        binary_mask,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio
    )

    # 5. Build AnyLabeling JSON
    labelme_data = build_anylabeling_json(
        image_filename=img_file.name,
        orig_h=orig_h,
        orig_w=orig_w,
        polygons=polygons,
        label=args.label
    )

    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)

    # 6. Optional Outputs
    if args.save_mask:
        mask_out_path = output_path / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(mask_out_path), binary_mask)

    if args.visualize:
        vis_img = draw_overlay(img_bgr, polygons)
        vis_out_path = output_path / f"{img_file.stem}_vis.jpg"
        cv2.imwrite(str(vis_out_path), vis_img)

    return True, len(polygons)


def main():
    args = parse_args()

    if args.stride is None:
        args.stride = max(1, int(round(args.tile_size * (1.0 - args.overlap))))

    # Determine files to process
    if args.image:
        single_path = Path(args.image)
        if not single_path.exists():
            logger.error(f"Image not found: {single_path}")
            sys.exit(1)
        image_files = [single_path]
        output_path = Path(args.output_dir) if args.output_dir else single_path.parent
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_path}")
            sys.exit(1)
        output_path = Path(args.output_dir) if args.output_dir else input_path
        all_files = sorted(list(input_path.iterdir()))
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    else:
        logger.error("Must specify either --input-dir (-i) or --image.")
        sys.exit(1)

    if not image_files:
        logger.warning("No supported image files found to process.")
        sys.exit(0)

    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"Inference Mode: {args.mode.upper()} (Tile: {args.tile_size}x{args.tile_size}, Stride: {args.stride}, Batch: {args.batch_size})")

    # Load model
    model = load_model(args.config, args.checkpoint, device)

    # Optional progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(image_files, desc=f"Auto-Labeling [{args.mode}]", unit="img")
    except ImportError:
        iterator = image_files

    processed_count = 0
    skipped_count = 0
    total_polygons = 0

    for img_file in iterator:
        success, poly_count = process_single_image(
            img_file=img_file,
            output_path=output_path,
            model=model,
            device=device,
            args=args
        )
        if success:
            processed_count += 1
            total_polygons += poly_count
        else:
            skipped_count += 1

    logger.info("=" * 60)
    logger.info(f"Auto-Labeling Completed! ({args.mode.upper()})")
    logger.info(f"Total images found:     {len(image_files)}")
    logger.info(f"Processed images:        {processed_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Total polygons detected: {total_polygons}")
    logger.info(f"Annotations saved to:    {output_path.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

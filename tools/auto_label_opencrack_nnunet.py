#!/usr/bin/env python3
"""
OpenCrack nnU-Net Auto-Labeling Tool for AnyLabeling / LabelMe
=============================================================
Batch inference on crack images using OpenCrack nnU-Net v2 model
(https://huggingface.co/fadeevla/opencrack-nnunet).

Features:
- Official nnU-Net v2 2D PlainConvUNet engine trained on OpenCrack consolidated benchmark.
- Native Gaussian-weighted sliding window / tiling inference.
- Configurable overlap (tile_step_size, e.g. 0.8 for 20% overlap).
- High-precision polygon contour vectorization via cv2.approxPolyDP and min_area filtering.
- 100% compliant AnyLabeling / LabelMe JSON export.

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
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OpenCrack-nnUNet")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "opencrack_nnunet" / "Dataset501_OpenCrack" / "nnUNetTrainer__nnUNetPlans__2d"


def register_custom_trainer():
    """Register nnUNetTrainerSaveEvery10 dynamically to satisfy checkpoint requirements."""
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_mod
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    if not hasattr(trainer_mod, "nnUNetTrainerSaveEvery10"):
        class nnUNetTrainerSaveEvery10(nnUNetTrainer):
            pass
        setattr(trainer_mod, "nnUNetTrainerSaveEvery10", nnUNetTrainerSaveEvery10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-labeling crack dataset using OpenCrack nnU-Net (fadeevla/opencrack-nnunet)."
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
        "-m", "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Path to nnU-Net model folder containing plans.json and fold_0."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_ep0500.pth",
        help="Checkpoint filename inside fold_0 (default: checkpoint_ep0500.pth)."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="Sliding window overlap ratio in range [0.0, 0.9] (default: 0.20, i.e., 20%% overlap)."
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=30.0,
        help="Minimum contour area in pixels to filter out noise (default: 30.0)."
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.002,
        help="Ramer-Douglas-Peucker epsilon multiplier (default: 0.002)."
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
        "--use-mirroring",
        action="store_true",
        help="Enable test-time augmentation (mirroring). Slower but slightly higher accuracy."
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


def load_nnunet_predictor(model_dir: str, checkpoint_name: str, overlap: float, device: torch.device, use_mirroring: bool):
    """Initialize nnU-Net v2 predictor from local model folder."""
    register_custom_trainer()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # tile_step_size = 1.0 - overlap (e.g. 1.0 - 0.20 = 0.80)
    tile_step_size = max(0.1, min(1.0, 1.0 - overlap))
    logger.info(f"Initializing nnUNetPredictor with tile_step_size={tile_step_size:.2f} (overlap={overlap*100:.0f}%) on {device}")

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=False,
        device=device
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=(0,),
        checkpoint_name=checkpoint_name
    )
    logger.info("OpenCrack nnU-Net model initialized successfully!")
    return predictor


def mask_to_polygons(
    binary_mask: np.ndarray,
    min_area: float = 30.0,
    epsilon_ratio: float = 0.002
) -> List[List[List[float]]]:
    """
    Vectorize binary mask into list of valid polygon coordinates.
    Filters contours with area < min_area and simplifies via approxPolyDP.
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
    """Draw detected polygons on image for visual inspection."""
    overlay = image_bgr.copy()
    for pts in polygons:
        np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [np_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.fillPoly(overlay, [np_pts], color=(0, 140, 255))
    return cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)


def process_single_image(
    img_file: Path,
    output_path: Path,
    predictor: Any,
    rw: Any,
    args: argparse.Namespace
) -> Tuple[bool, int]:
    """Run prediction on 1 image using OpenCrack nnU-Net, vectorize to polygon, save JSON."""
    target_json_path = output_path / f"{img_file.stem}.json"
    if target_json_path.exists() and not args.overwrite:
        return False, 0

    # 1. Read Image via NaturalImage2DIO
    try:
        img_arr, props = rw.read_images([str(img_file)])
    except Exception as e:
        logger.warning(f"Failed to read image {img_file.name} with NaturalImage2DIO: {e}")
        return False, 0

    # 2. Predict with nnUNetPredictor
    pred = predictor.predict_single_npy_array(img_arr, props)
    # pred is shape (1, H, W) where 1 indicates crack class
    binary_mask = (pred[0] == 1).astype(np.uint8) * 255
    orig_h, orig_w = binary_mask.shape[:2]

    # 3. Vectorize to Polygons
    polygons = mask_to_polygons(
        binary_mask,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio
    )

    # 4. Build AnyLabeling JSON
    labelme_data = build_anylabeling_json(
        image_filename=img_file.name,
        orig_h=orig_h,
        orig_w=orig_w,
        polygons=polygons,
        label=args.label
    )

    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)

    # 5. Optional outputs
    if args.save_mask:
        mask_out_path = output_path / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(mask_out_path), binary_mask)

    if args.visualize:
        # Read BGR for drawing overlay
        img_bytes = np.fromfile(str(img_file), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            vis_img = draw_overlay(img_bgr, polygons)
            vis_out_path = output_path / f"{img_file.stem}_vis.jpg"
            cv2.imwrite(str(vis_out_path), vis_img)

    return True, len(polygons)


def main():
    args = parse_args()

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

    # Load model
    predictor = load_nnunet_predictor(
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint,
        overlap=args.overlap,
        device=device,
        use_mirroring=args.use_mirroring
    )

    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    rw = NaturalImage2DIO()

    # Progress loop
    try:
        from tqdm import tqdm
        iterator = tqdm(image_files, desc="OpenCrack-nnUNet Auto-Labeling", unit="img")
    except ImportError:
        iterator = image_files

    processed_count = 0
    skipped_count = 0
    total_polygons = 0

    for img_file in iterator:
        success, poly_count = process_single_image(
            img_file=img_file,
            output_path=output_path,
            predictor=predictor,
            rw=rw,
            args=args
        )
        if success:
            processed_count += 1
            total_polygons += poly_count
        else:
            skipped_count += 1

    logger.info("=" * 60)
    logger.info("Auto-Labeling Completed Successfully! (OpenCrack nnU-Net)")
    logger.info(f"Total images found:     {len(image_files)}")
    logger.info(f"Processed images:        {processed_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Total polygons detected: {total_polygons}")
    logger.info(f"Annotations saved to:    {output_path.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

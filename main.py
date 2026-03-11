"""
Main entry point for the Deepfake Image Detection system.

Provides CLI subcommands for:
    - extract:    Extract frames from video datasets
    - preprocess: Detect and crop faces from extracted frames
    - train:      Train the multi-branch deepfake detector
    - evaluate:   Evaluate on FF++ C23 test split
    - explain:    Generate explainability visualizations
"""

import os
import sys
import argparse

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger("deepfake_detector")


def cmd_extract(args):
    """Extract frames from video datasets."""
    from datasets.frame_extractor import (
        build_split_structure,
        extract_celebdf_frames,
    )

    cfg = Config()
    timestamps = cfg.frame_timestamps

    if args.dataset in ("ffpp", "both"):
        assert args.ffpp_root, "Must provide --ffpp_root for FF++ extraction"
        build_split_structure(
            ffpp_root=args.ffpp_root,
            output_root=args.output or cfg.frames_dir,
            timestamps=timestamps,
            compression=args.compression,
            num_workers=args.workers,
        )

    if args.dataset in ("celebdf", "both"):
        assert args.celebdf_root, "Must provide --celebdf_root for Celeb-DF extraction"
        extract_celebdf_frames(
            celebdf_root=args.celebdf_root,
            output_root=args.output or os.path.join(cfg.frames_dir, "celebdf"),
            timestamps=timestamps,
            num_workers=args.workers,
        )

    logger.info("Frame extraction complete!")


def cmd_preprocess(args):
    """Detect and crop faces from extracted frames."""
    from datasets.face_cropper import process_dataset_splits

    cfg = Config()

    process_dataset_splits(
        frames_root=args.input or cfg.frames_dir,
        output_root=args.output or cfg.frames_dir,
        face_size=cfg.face_size,
        margin=cfg.face_margin,
        device=args.device or cfg.device,
    )

    logger.info("Face preprocessing complete!")


def cmd_train(args):
    """Train the multi-branch deepfake detector."""
    from training.train import train

    cfg = Config()

    # Override config with CLI args
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.workers:
        cfg.num_workers = args.workers
    if args.no_amp:
        cfg.use_amp = False

    # Set seed
    cfg.set_seed()

    logger.info(repr(cfg))

    train(cfg, resume_path=args.resume)


def cmd_evaluate(args):
    """Evaluate model on FF++ C23 test split."""
    from evaluation.test_ffpp import evaluate_test_set

    cfg = Config()

    if args.data_root:
        cfg.frames_dir = args.data_root

    evaluate_test_set(cfg, checkpoint_path=args.checkpoint)


def cmd_explain(args):
    """Generate explainability visualizations."""
    import torch
    from models.fusion_model import MultiBranchDetector

    cfg = Config()
    device = torch.device(cfg.device)

    # Load model
    checkpoint_path = args.checkpoint or os.path.join(cfg.checkpoints_dir, "best_model.pt")
    model = MultiBranchDetector(
        spatial_embed_dim=cfg.spatial_embed_dim,
        frequency_embed_dim=cfg.frequency_embed_dim,
        srm_embed_dim=cfg.srm_embed_dim,
        hidden_dim=cfg.fusion_hidden_dim,
        dropout_rate=cfg.dropout_rate,
        pretrained=False,
    )

    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")

    model = model.to(device).eval()

    output_dir = os.path.join(cfg.outputs_dir, "explainability")
    os.makedirs(output_dir, exist_ok=True)

    # Grad-CAM
    if args.type in ("gradcam", "all"):
        from explainability.gradcam import DeepfakeGradCAM

        gradcam = DeepfakeGradCAM(model, device=str(device))

        if args.image:
            gradcam.visualize_and_save(
                args.image,
                os.path.join(output_dir, "gradcam_single.png"),
                cfg,
            )
        elif args.image_dir:
            gradcam.batch_visualize(
                args.image_dir,
                os.path.join(output_dir, "gradcam"),
                cfg,
                max_images=args.max_images,
            )

    # Frequency maps
    if args.type in ("frequency", "all"):
        from explainability.frequency_maps import (
            visualize_frequency_comparison,
            visualize_single_spectrum,
        )

        if args.real_image and args.fake_image:
            visualize_frequency_comparison(
                args.real_image,
                args.fake_image,
                os.path.join(output_dir, "frequency_comparison.png"),
            )
        elif args.image:
            visualize_single_spectrum(
                args.image,
                os.path.join(output_dir, "frequency_spectrum.png"),
            )

    # Patch importance
    if args.type in ("patch", "all"):
        from explainability.patch_importance import PatchImportanceAnalyzer

        analyzer = PatchImportanceAnalyzer(
            model, device=str(device), patch_size=args.patch_size
        )

        if args.image:
            analyzer.visualize_and_save(
                args.image,
                os.path.join(output_dir, "patch_importance.png"),
                cfg,
            )
        elif args.image_dir:
            analyzer.batch_visualize(
                args.image_dir,
                os.path.join(output_dir, "patch_importance"),
                cfg,
                max_images=args.max_images,
            )

    logger.info(f"Explainability outputs saved to: {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deepfake Image Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames from FF++
  python main.py extract --dataset ffpp --ffpp_root /data/FF++ --output datasets/frames

  # Preprocess faces
  python main.py preprocess --input datasets/frames --output datasets/faces

  # Train the model
  python main.py train --epochs 25 --batch_size 32

  # Evaluate on test set
  python main.py evaluate --checkpoint checkpoints/best_model.pt

  # Generate Grad-CAM visualization
  python main.py explain --type gradcam --image test_face.png
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Extract ────────────────────────────────────────────────────────
    p_extract = subparsers.add_parser("extract", help="Extract frames from videos")
    p_extract.add_argument("--dataset", choices=["ffpp", "celebdf", "both"], required=True)
    p_extract.add_argument("--ffpp_root", type=str, default="")
    p_extract.add_argument("--celebdf_root", type=str, default="")
    p_extract.add_argument("--output", type=str, default=None)
    p_extract.add_argument("--compression", type=str, default="c23")
    p_extract.add_argument("--workers", type=int, default=4)

    # ── Preprocess ─────────────────────────────────────────────────────
    p_preprocess = subparsers.add_parser("preprocess", help="Detect and crop faces")
    p_preprocess.add_argument("--input", type=str, default=None)
    p_preprocess.add_argument("--output", type=str, default=None)
    p_preprocess.add_argument("--device", type=str, default=None)

    # ── Train ──────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train the deepfake detector")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch_size", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--workers", type=int, default=None)
    p_train.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    p_train.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    # ── Evaluate ───────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("evaluate", help="Evaluate on FF++ C23 test set")
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--data_root", type=str, default=None)

    # ── Explain ────────────────────────────────────────────────────────
    p_explain = subparsers.add_parser("explain", help="Generate explainability visualizations")
    p_explain.add_argument(
        "--type",
        choices=["gradcam", "frequency", "patch", "all"],
        default="all",
    )
    p_explain.add_argument("--checkpoint", type=str, default=None)
    p_explain.add_argument("--image", type=str, default=None, help="Single image path")
    p_explain.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    p_explain.add_argument("--real_image", type=str, default=None, help="Real image for comparison")
    p_explain.add_argument("--fake_image", type=str, default=None, help="Fake image for comparison")
    p_explain.add_argument("--patch_size", type=int, default=32)
    p_explain.add_argument("--max_images", type=int, default=20)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "extract": cmd_extract,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "explain": cmd_explain,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

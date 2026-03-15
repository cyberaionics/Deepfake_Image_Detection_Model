"""
Frame Extractor for FaceForensics++ and Celeb-DF v2.

Extracts frames from videos at specified timestamps (0.5s, 1.0s, 1.5s).
Falls back to the middle frame for short videos.
Saves frames as PNG files.
"""

import os
import cv2
import glob
import argparse
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("frame_extractor")


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    timestamps: list[float] = [0.5, 1.0, 1.5],
    frame_format: str = "png",
) -> list[str]:
    """
    Extract frames from a single video at the given timestamps.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        timestamps: List of timestamps (in seconds) to extract.
        frame_format: Image format (default: png).

    Returns:
        List of paths to saved frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        logger.warning(f"Invalid video metadata: {video_path} (fps={fps}, frames={total_frames})")
        cap.release()
        return []

    duration = total_frames / fps
    video_name = Path(video_path).stem
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    # Determine which frames to extract
    frame_indices = []
    for ts in timestamps:
        if ts < duration:
            idx = int(fps * ts)
            frame_indices.append((ts, min(idx, total_frames - 1)))

    # If video is too short for any timestamp, use the middle frame
    if len(frame_indices) == 0:
        mid_idx = total_frames // 2
        frame_indices.append(("mid", mid_idx))

    for ts, frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(
                f"Failed to read frame {frame_idx} from {video_path}"
            )
            continue

        ts_label = f"{ts}s" if isinstance(ts, float) else ts
        filename = f"{video_name}_t{ts_label}.{frame_format}"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, frame)
        saved_paths.append(save_path)

    cap.release()
    return saved_paths


def _extract_worker(args: tuple) -> list[str]:
    """Worker function for multiprocessing."""
    video_path, output_dir, timestamps, frame_format = args
    return extract_frames_from_video(video_path, output_dir, timestamps, frame_format)


def _get_ffpp_video_paths(ffpp_root: str, compression: str, metadata_csv: str = "") -> tuple[list[str], list[str]]:
    """
    Robustly finds real and fake videos in FF++ dataset, handling official
    nested structure, flat structures, Kaggle variations, and explicit CSV metadata.
    """
    real_vids = []
    fake_vids = []
    
    # --- 0. Metadata CSV Parsing ---
    if metadata_csv and os.path.isfile(metadata_csv):
        logger.info(f"Parsing explicit metadata CSV: {metadata_csv}")
        # Build lookup table of basenames from directory
        all_mp4s = glob.glob(os.path.join(ffpp_root, "**", "*.mp4"), recursive=True)
        vid_map = {os.path.basename(p): p for p in all_mp4s}
        
        found_real, found_fake = 0, 0
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # E.g. File Path: DeepFakeDetection/01_02__meeting_serious__YVGY8LOK.mp4
                if 'File Path' not in row or 'Label' not in row:
                    logger.warning("CSV must contain 'File Path' and 'Label' columns.")
                    break
                    
                fp = row['File Path']
                label = row['Label'].strip().upper()
                basename = os.path.basename(fp)
                
                if basename in vid_map:
                    if label == "FAKE":
                        fake_vids.append(vid_map[basename])
                        found_fake += 1
                    elif label == "REAL":
                        real_vids.append(vid_map[basename])
                        found_real += 1
                        
        if real_vids or fake_vids:
            logger.info(f"Resolved from CSV: {found_real} real, {found_fake} fake")
            return sorted(list(set(real_vids))), sorted(list(set(fake_vids)))
        else:
            logger.warning("Failed to resolve any videos from CSV, falling back to path heuristics.")
            
    # --- Fallback Heuristics ---
    def _find(bd):
        vids = []
        if not os.path.isdir(bd): return vids
        sd = os.path.join(bd, compression, "videos")
        if os.path.isdir(sd):
            vids.extend(glob.glob(os.path.join(sd, "*.mp4")))
            if vids: return vids
        cd = os.path.join(bd, compression)
        if os.path.isdir(cd):
            vids.extend(glob.glob(os.path.join(cd, "*.mp4")))
            if vids: return vids
        return glob.glob(os.path.join(bd, "**", "*.mp4"), recursive=True)

    # 1. Official paths
    off_real = os.path.join(ffpp_root, "original_sequences", "youtube")
    if os.path.isdir(off_real):
        real_vids.extend(_find(off_real))
        for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            fake_vids.extend(_find(os.path.join(ffpp_root, "manipulated_sequences", m)))
        if real_vids or fake_vids:
            return sorted(list(set(real_vids))), sorted(list(set(fake_vids)))

    # 2. Flattened paths
    for rd in ["youtube", "real", "original"]:
        d = os.path.join(ffpp_root, rd)
        if os.path.isdir(d): real_vids.extend(_find(d))
        
    for fd in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "fake", "manipulated"]:
        d = os.path.join(ffpp_root, fd)
        if os.path.isdir(d): fake_vids.extend(_find(d))
        
    if real_vids or fake_vids:
        return sorted(list(set(real_vids))), sorted(list(set(fake_vids)))

    # 3. Ultimate Fallback
    all_mp4s = glob.glob(os.path.join(ffpp_root, "**", "*.mp4"), recursive=True)
    for vp in all_mp4s:
        p_low = vp.lower()
        if any(k in p_low for k in ["deepfake", "face2face", "faceswap", "neuraltexture", "manipulate", "fake"]):
            fake_vids.append(vp)
        elif any(k in p_low for k in ["youtube", "original", "real"]):
            real_vids.append(vp)
            
    return sorted(list(set(real_vids))), sorted(list(set(fake_vids)))


def extract_ffpp_frames(
    ffpp_root: str,
    output_root: str,
    timestamps: list[float] = [0.5, 1.0, 1.5],
    compression: str = "c23",
    num_workers: int = 4,
    metadata_csv: str = "",
):
    """
    Extract frames from FaceForensics++ dataset.
    """
    logger.info(f"Extracting FF++ frames from: {ffpp_root}")

    real_vids, fake_vids = _get_ffpp_video_paths(ffpp_root, compression, metadata_csv)
    
    tasks = []
    
    # --- Real videos ---
    real_output = os.path.join(output_root, "real")
    if real_vids:
        for vp in real_vids:
            tasks.append((vp, real_output, timestamps, "png"))
        logger.info(f"  Real videos found: {len(real_vids)}")
    else:
        logger.warning(f"  Real videos not found in {ffpp_root}")

    # --- Fake videos ---
    fake_output = os.path.join(output_root, "fake")
    if fake_vids:
        for vp in fake_vids:
            tasks.append((vp, fake_output, timestamps, "png"))
        logger.info(f"  Fake videos found: {len(fake_vids)}")
    else:
        logger.warning(f"  Fake videos not found in {ffpp_root}")

    # Extract with multiprocessing
    logger.info(f"  Extracting frames with {num_workers} workers...")
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(_extract_worker, tasks), total=len(tasks), desc="FF++ Extraction"))


def extract_celebdf_frames(
    celebdf_root: str,
    output_root: str,
    timestamps: list[float] = [0.5, 1.0, 1.5],
    num_workers: int = 4,
):
    """
    Extract frames from Celeb-DF v2 dataset.

    Expected Celeb-DF structure:
        celebdf_root/
            Celeb-real/
            Celeb-synthesis/
            YouTube-real/
            List_of_testing_videos.txt

    Args:
        celebdf_root: Root directory of Celeb-DF v2.
        output_root: Output directory for extracted frames.
        timestamps: Timestamps to extract.
        num_workers: Number of parallel workers.
    """
    logger.info(f"Extracting Celeb-DF v2 frames from: {celebdf_root}")

    # Real videos
    real_dirs = ["Celeb-real", "YouTube-real"]
    real_output = os.path.join(output_root, "real")

    # Fake videos
    fake_dirs = ["Celeb-synthesis"]
    fake_output = os.path.join(output_root, "fake")

    tasks = []

    for rd in real_dirs:
        d = os.path.join(celebdf_root, rd)
        if os.path.isdir(d):
            vids = sorted(glob.glob(os.path.join(d, "*.mp4")))
            for vp in vids:
                tasks.append((vp, real_output, timestamps, "png"))
            logger.info(f"  Real videos in {rd}: {len(vids)}")

    for fd in fake_dirs:
        d = os.path.join(celebdf_root, fd)
        if os.path.isdir(d):
            vids = sorted(glob.glob(os.path.join(d, "*.mp4")))
            for vp in vids:
                tasks.append((vp, fake_output, timestamps, "png"))
            logger.info(f"  Fake videos in {fd}: {len(vids)}")

    logger.info(f"  Extracting frames with {num_workers} workers...")
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(_extract_worker, tasks), total=len(tasks), desc="Celeb-DF Extraction"))


def build_split_structure(
    ffpp_root: str,
    output_root: str,
    timestamps: list[float] = [0.5, 1.0, 1.5],
    compression: str = "c23",
    num_workers: int = 4,
    metadata_csv: str = "",
):
    """
    Build the train/val/test split structure using FaceForensics++ official splits.

    FF++ provides train/val/test splits via text files. This function reads those
    split files and organizes extracted frames accordingly.

    Args:
        ffpp_root: Root directory of FaceForensics++.
        output_root: Output directory (datasets/frames/).
        timestamps: Frame extraction timestamps.
        compression: Compression level.
        num_workers: Number of workers.
        metadata_csv: Optional metadata CSV mapping file.
    """
    splits_dir = os.path.join(ffpp_root, "splits")

    split_files = {
        "train": os.path.join(splits_dir, "train.json"),
        "val": os.path.join(splits_dir, "val.json"),
        "test": os.path.join(splits_dir, "test.json"),
    }

    # Check if split files exist
    have_splits = all(os.path.isfile(f) for f in split_files.values())

    if have_splits:
        import json
        logger.info("Using official FF++ split files")

        for split_name, split_file in split_files.items():
            with open(split_file, "r") as f:
                video_pairs = json.load(f)

            # Flatten pairs to get video IDs
            video_ids = set()
            for pair in video_pairs:
                video_ids.update(pair)

            split_output = os.path.join(output_root, split_name)

            # Extract videos for this split
            real_vids, fake_vids = _get_ffpp_video_paths(ffpp_root, compression, metadata_csv)
            
            real_output = os.path.join(split_output, "real")
            fake_output = os.path.join(split_output, "fake")
            tasks = []

            for vid_path in real_vids:
                vid_name = Path(vid_path).stem
                if vid_name in video_ids:
                    tasks.append((vid_path, real_output, timestamps, "png"))

            for vid_path in fake_vids:
                vid_name = Path(vid_path).stem
                source_id = vid_name.split("_")[0]
                if source_id in video_ids:
                    tasks.append((vid_path, fake_output, timestamps, "png"))

            logger.info(f"  {split_name}: {len(tasks)} videos to extract")
            with Pool(num_workers) as pool:
                list(tqdm(
                    pool.imap(_extract_worker, tasks),
                    total=len(tasks),
                    desc=f"FF++ {split_name}",
                ))
    else:
        # Fallback: Extract all videos into a flat structure
        # User can manually create splits afterward
        logger.info("No official split files found, extracting all frames to flat structure")
        logger.info("You can create train/val/test splits manually later")
        extract_ffpp_frames(ffpp_root, output_root, timestamps, compression, num_workers, metadata_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from deepfake datasets")
    parser.add_argument("--dataset", choices=["ffpp", "celebdf", "both"], required=True)
    parser.add_argument("--ffpp_root", type=str, default="")
    parser.add_argument("--celebdf_root", type=str, default="")
    parser.add_argument("--output", type=str, required=True, help="Output directory for frames")
    parser.add_argument("--compression", type=str, default="c23", choices=["c23", "c40", "raw"])
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--metadata_csv", type=str, default="", help="Optional Kaggle CSV metadata")
    parser.add_argument(
        "--timestamps",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
    )
    args = parser.parse_args()

    if args.dataset in ("ffpp", "both"):
        assert args.ffpp_root, "Must provide --ffpp_root for FF++ extraction"
        build_split_structure(
            args.ffpp_root, args.output, args.timestamps, args.compression, args.workers, args.metadata_csv
        )

    if args.dataset in ("celebdf", "both"):
        assert args.celebdf_root, "Must provide --celebdf_root for Celeb-DF extraction"
        extract_celebdf_frames(
            args.celebdf_root, args.output, args.timestamps, args.workers
        )

    logger.info("Frame extraction complete!")

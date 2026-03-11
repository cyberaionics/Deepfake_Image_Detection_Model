# 🔍 Deepfake Image Detection System

A research-grade **Multi-Branch Deepfake Detector** implemented in PyTorch, trained on **FaceForensics++** and **Celeb-DF v2**, with explainable AI components.

## Architecture

The model uses three complementary feature extraction branches:

```
Input (3×224×224)
    ├── Spatial Branch (EfficientNet-B4)  → 1792-dim
    ├── Frequency Branch (FFT + CNN)     →  256-dim
    └── SRM Branch (Noise Residual + CNN) →  128-dim
                                             ↓
        Concatenate → 2176-dim → FC(512) → ReLU → Dropout → FC(1) → Sigmoid
```

| Branch     | What it detects                                            |
|------------|-----------------------------------------------------------|
| **Spatial**    | Texture inconsistencies, face blending, eye/mouth errors  |
| **Frequency**  | High-frequency artifacts from GAN-based synthesis         |
| **SRM Noise**  | Noise residual patterns via forensic SRM filters          |

## Project Structure

```
deepfake_detector/
├── main.py                         # CLI entry point
├── requirements.txt
├── datasets/
│   ├── frame_extractor.py          # Video → frame extraction
│   ├── face_cropper.py             # MTCNN face detect + crop
│   └── video_dataset.py            # PyTorch Dataset + augmentations
├── models/
│   ├── spatial_branch.py           # EfficientNet-B4 backbone
│   ├── frequency_branch.py         # FFT → CNN branch
│   ├── srm_branch.py               # SRM filter → CNN branch
│   └── fusion_model.py             # Multi-branch fusion + classifier
├── training/
│   ├── train.py                    # Training loop (AMP, checkpointing)
│   └── validation.py               # Validation with AUC tracking
├── evaluation/
│   ├── metrics.py                  # Accuracy, AUC, F1, EER, plots
│   └── test_ffpp.py                # FF++ C23 test evaluation
├── explainability/
│   ├── gradcam.py                  # Grad-CAM heatmaps
│   ├── frequency_maps.py           # FFT spectrum visualization
│   └── patch_importance.py          # Patch occlusion importance
└── utils/
    ├── config.py                   # Centralized configuration
    └── logger.py                   # Logging utility
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

**FaceForensics++**: Request access at [faceforensics.com](https://github.com/ondyari/FaceForensics)
- Download the C23 (light compression) version
- Include all manipulation methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures

**Celeb-DF v2**: Download from [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics)

### 3. Expected Dataset Layout

```
/path/to/FaceForensics++/
├── original_sequences/youtube/c23/videos/
├── manipulated_sequences/
│   ├── Deepfakes/c23/videos/
│   ├── Face2Face/c23/videos/
│   ├── FaceSwap/c23/videos/
│   └── NeuralTextures/c23/videos/
└── splits/
    ├── train.json
    ├── val.json
    └── test.json

/path/to/Celeb-DF-v2/
├── Celeb-real/
├── Celeb-synthesis/
├── YouTube-real/
└── List_of_testing_videos.txt
```

## Usage

All commands are run via `main.py`:

### Step 1: Extract Frames

Extracts frames at 0.5s, 1.0s, and 1.5s timestamps from each video.

```bash
# FaceForensics++
python main.py extract --dataset ffpp --ffpp_root /path/to/FF++ --output datasets/frames

# Celeb-DF v2
python main.py extract --dataset celebdf --celebdf_root /path/to/Celeb-DF-v2 --output datasets/frames

# Both
python main.py extract --dataset both --ffpp_root /path/to/FF++ --celebdf_root /path/to/Celeb-DF-v2
```

### Step 2: Face Preprocessing

Detects and crops faces using MTCNN, resizes to 224×224.

```bash
python main.py preprocess --input datasets/frames --output datasets/frames
```

### Step 3: Train

```bash
# Basic training
python main.py train

# Custom settings
python main.py train --epochs 25 --batch_size 32 --lr 1e-4

# Resume from checkpoint
python main.py train --resume checkpoints/last_checkpoint.pt

# Disable mixed precision
python main.py train --no_amp
```

### Step 4: Evaluate on FF++ C23 Test Split

```bash
python main.py evaluate --checkpoint checkpoints/best_model.pt
```

**Reported metrics:**
- Accuracy (%)
- AUC-ROC
- F1-score (macro)
- Precision / Recall (fake class)
- Equal Error Rate (EER)
- Confusion Matrix
- ROC Curve

### Step 5: Explainability Visualizations

```bash
# Grad-CAM
python main.py explain --type gradcam --image path/to/face.png

# Frequency analysis (real vs fake comparison)
python main.py explain --type frequency --real_image real.png --fake_image fake.png

# Patch importance
python main.py explain --type patch --image path/to/face.png --patch_size 32

# All explainability methods
python main.py explain --type all --image path/to/face.png --real_image real.png --fake_image fake.png

# Batch processing
python main.py explain --type gradcam --image_dir datasets/frames/test/fake --max_images 20
```

## Training Configuration

| Parameter       | Value              |
|----------------|--------------------|
| Loss            | BCEWithLogitsLoss  |
| Optimizer       | AdamW              |
| Learning Rate   | 1e-4               |
| Weight Decay    | 1e-2               |
| Scheduler       | Cosine Annealing   |
| Batch Size      | 32                 |
| Epochs          | 25                 |
| Mixed Precision | Enabled (AMP)      |
| Image Size      | 224×224            |

## Data Augmentation

- Horizontal flip (p=0.5)
- Color jitter (p=0.3)
- Random resized crop (p=0.3)
- Gaussian blur (p=0.2)
- JPEG compression simulation (p=0.3)

## Explainability Examples

### Grad-CAM
Shows which facial regions the model focuses on. Hot regions (red) indicate areas with detected artifacts — commonly around eyes, mouth, and face boundaries.

### Frequency Spectrum
Compares FFT magnitude spectra of real vs fake images. Fake images often show distinct high-frequency patterns absent in real images, visible as periodic peaks in the radial frequency profile.

### Patch Importance
Reveals which image patches most influence the prediction. Masking important patches causes the largest prediction shift, highlighting critical artifact regions.

## Cloud GPU Support

The system automatically adapts to the available hardware:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Compatible with:
- **Kaggle GPU** (P100/T4)
- **Google Colab** (T4/A100)
- **RunPod** (A100/H100)
- **Lambda Labs**

Features:
- CUDA auto-detection with CPU fallback
- Mixed precision training (`torch.cuda.amp`)
- Efficient dataloading with configurable `num_workers`
- Checkpoint saving for training resume
- Gradient accumulation for effective batch size scaling

## Reproducibility

Fixed random seeds across:
- Python `random`
- NumPy
- PyTorch (CPU + CUDA)
- cuDNN deterministic mode

```bash
# Default seed: 42 (configurable in config.py)
```

## License

For research and educational purposes.

import os
import torch

# Base Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "DCSASS Dataset")
LABELS_DIR = os.path.join(DATASET_DIR, "Labels")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
VIDEOS_DIR = os.path.join(STORAGE_DIR, "videos")
EVIDENCE_DIR = os.path.join(STORAGE_DIR, "evidence")

# Database & Backend Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(PROJECT_ROOT, 'visioniq.db')}")
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "visioniq")
MONGODB_TIMEOUT_MS = int(os.environ.get("MONGODB_TIMEOUT_MS", 2000))

DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.50))
DEFAULT_WINDOW_SEC = float(os.environ.get("WINDOW_SEC", 2.0))
DEFAULT_STRIDE_SEC = float(os.environ.get("STRIDE_SEC", 1.0))
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 500))

# Ensure required runtime directories exist
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# 7 Target Activity Classes
CLASSES = [
    "Normal",
    "Fighting",
    "Robbery",
    "Shoplifting",
    "Burglary",
    "RoadAccidents",
    "Vandalism"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Input Video Dimensions & Sampling
NUM_FRAMES = 16          # Uniformly sample 16 frames per clip
IMG_SIZE = 112           # 112x112 spatial resolution optimal for R(2+1)D on 4GB VRAM
FPS_TARGET = 30.0

# Pretrained Video Normalization (Kinetics-400 / ImageNet)
NORM_MEAN = [0.43216, 0.394666, 0.37645]
NORM_STD = [0.22803, 0.22145, 0.216989]

# Training Configuration
BATCH_SIZE = 4           # Micro-batch size to guarantee safety on 4GB GTX 1650
GRAD_ACCUM_STEPS = 2     # Effective batch size = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 15
PATIENCE = 5             # Early stopping patience
USE_AMP = True           # Automatic Mixed Precision (fp16)
RANDOM_SEED = 42

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0          # Windows multiprocessing safety with OpenCV VideoCapture

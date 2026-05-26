# ============================================================
# Configuration for UNISCP Urban Driving VQA Inference
# ============================================================

# ----- Model Settings -----
MODEL_DIR = "/home/user/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"  # Update this to your model path
TORCH_DTYPE = "bfloat16"       # "float16", "bfloat16", or "float32"
MAX_PIXELS = 1280 * 28 * 28    # Max pixel count for image processing
MAX_NEW_TOKENS = 1024           # Max tokens to generate

# ----- Dataset Settings -----
# Root path of the UNISCP dataset
DATASET_ROOT = "./UNISCP"  # Update this

# Sequences to process (set to None to process all RURAL_* folders automatically)
# Example: ["RURAL_A0", "RURAL_A1", "RURAL_A2"] or None for all
SEQUENCES = ["RURAL_B0", "CP_MSCLIKE", "GARDEN_MSCLIKE", "LOOP1_MSCLIKE"]

# Relative path from sequence root to left camera images
IMAGE_SUBPATH = "1_IMAGE/1_IMAGE/LEFT"

# Output folder name (will be created under each sequence)
OUTPUT_FOLDER_NAME = "6_CAPTION"

# ----- Processing Settings -----
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
SKIP_EXISTING = True            # Skip images that already have JSON output
INDENT = 2                      # JSON indentation
BATCH_LOG_INTERVAL = 100        # Log progress every N images

# # ============================================================
# # Configuration for UNISCP Urban Driving VQA Inference
# # ============================================================

# # ----- Model Settings -----
# MODEL_DIR = "/home/user/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
# TORCH_DTYPE = "bfloat16"       # "float16", "bfloat16", or "float32"
# MAX_PIXELS = 1280 * 28 * 28    # Max pixel count for image processing
# MAX_NEW_TOKENS = 1024           # Max tokens to generate

# # ----- Dataset Settings -----
# DATASET_ROOT = "./UNISCP"
# SEQUENCES = ["FENDUAN_1"]
# IMAGE_SUBPATH = "1_IMAGE/1_IMAGE/LEFT"
# OUTPUT_FOLDER_NAME = "6_CAPTION"

# # ----- Processing Settings -----
# IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
# SKIP_EXISTING = True
# INDENT = 2
# BATCH_LOG_INTERVAL = 100

# # ----- Batch Inference Settings -----
# # Number of images to process simultaneously in one forward pass.
# # Increase to better utilize GPU. Start with 4, increase if GPU memory allows.
# # If OOM, reduce this number.
# BATCH_SIZE = 4


# ============================================================
# Configuration for 4Cams+1LiDAR Urban Driving VQA Inference
# ============================================================

# ----- Model Settings -----
MODEL_DIR = "/home/user/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
TORCH_DTYPE = "bfloat16"
MAX_PIXELS = 1280 * 28 * 28
MAX_NEW_TOKENS = 1024

# ----- Dataset Settings -----
DATASET_ROOT = "/media/user/hdd4t/self_collected_data/2026_0430_4cams1lidar"
SEQUENCES = ["DAY_0430"]
SEQUENCE_PREFIX = "DAY_"
IMAGE_SUBPATH = "1_IMAGE/1_IMAGE/FRONT"
OUTPUT_FOLDER_NAME = "6_CAPTION"

# ----- Processing Settings -----
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
SKIP_EXISTING = True
INDENT = 2
BATCH_LOG_INTERVAL = 100

# ----- Batch Inference Settings -----
BATCH_SIZE = 4
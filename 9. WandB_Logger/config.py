# Training Hyperparams
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MIN_NUM_EPOCHS = 1
MAX_NUM_EPOCHS = 3

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute relatied
PROFILER = None
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "16-mixed"
STRATEGY = "deepspeed"
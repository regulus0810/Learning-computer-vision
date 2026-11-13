import os
import torch

# 1. Datapath
KAGGLE_PATH = "/kaggle/input/datasets/ayuraj/asl-dataset/asl_dataset/asl_dataset"

DATA_DIR = KAGGLE_PATH

# Path to save results
OUTPUT_DIR = "/kaggle/working/"
MODEL_CNN_SAVE_PATH = os.path.join(OUTPUT_DIR, "asl_cnn_model.pth")
PREDICTION_SAVE_PATH = os.path.join(OUTPUT_DIR, "prediction.png")

# 2. Dataset Parameters
# IMAGE_SIZE = (64, 64)
IMAGE_SIZE = (128, 128) 
CHANNELS = 3
NUM_CLASSES = 36

TEST_SPLIT = 0.2
RANDOM_SEED = 42

# 3. hYPERPARAMETER
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Datapath
DATA_TRAIN_PATH = "/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/Train"
DATA_TEST_PATH = "/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/Test"
DATA_META_PATH = "/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/Meta"

TRAIN_CSV_PATH = "/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/Train.csv"
TEST_CSV_PATH = "/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/Test.csv"
META_CSV_PATH = "/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/Meta.csv"

# Path to save results
OUTPUT_DIR = "/kaggle/working/"
MODEL_CNN_SAVE_PATH = os.path.join(OUTPUT_DIR, "GTSRB_cnn_model.pth")
MODEL_RESNET18_SAVE_PATH = os.path.join(OUTPUT_DIR, "GTSRB_resnet18_model.pth")
PREDICTION_SAVE_PATH = os.path.join(OUTPUT_DIR, "prediction.png")

# 2. Dataset Parameters
# IMAGE_SIZE = (64, 64)
IMAGE_SIZE = (128, 128) 
CHANNELS = 3
NUM_CLASSES = 43

# 3. hYPERPARAMETER
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Normalization mean and std
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
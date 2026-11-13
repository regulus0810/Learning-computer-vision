
# --- Parameter of mlp model ---
class MlpConfig:
    CHANNELS = 1
    HEIGHT = 28
    WIDTH = 28
    INPUT_SIZE = CHANNELS * HEIGHT * WIDTH  # 784
    
    # Kích thước các tầng
    HIDDEN_1 = 128
    HIDDEN_2 = 64
    NUM_CLASSES = 10



# --- Train ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
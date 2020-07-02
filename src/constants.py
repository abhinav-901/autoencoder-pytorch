DATA_DIR = './data'
BATCH_SIZE = 6
LR = 0.001
WEIGHT_DECAY = 1e-5
PRINT_EVERY = 1
NUM_EPOCHS = 500
TRAIN_OUT_DIR = './out/train'
TEST_OUT_DIR = './out/test'
DROP_RATE = 0.02

# Directories for different purpose
# model path dir
CNN_AUTO_EN_MODEL_PATH = './out/model/cnn_autoen'
# out directory for training results
CNN_AUTO_EN_TRAIN_OUT_DIR = './out/train/cnn_autoen'
# out directory for testing results
CNN_AUTO_EN_TEST_OUT_DIR = './out/test/cnn_autoen'
# similar directories for mlp autoencoder
MLP_AUTO_EN_MODEL_PATH = './out/model/mlp_autoen'
MLP_AUTO_EN_TRAIN_OUT_DIR = './out/train/mlp_autoen'
MLP_AUTO_EN_TEST_OUT_DIR = './out/test/mlp_autoen'
# from below choice make any one True
RUN_CNN_AUTOEN = False
RUN_UPSCALE_AUTOEN = True
RUN_MLP_AUTOEN = False
SAVE_CHECKPOINT = 1

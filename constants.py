import torch


DATA_PATH = "./data/"
TRAIN_X_PATH = DATA_PATH + "gsn_img_uint8.npy"
TRAIN_Y_PATH = DATA_PATH + "gsn_msk_uint8.npy"
TEST_X_PATH = DATA_PATH + "test_gsn_image.npy"
TEST_Y_PATH = DATA_PATH + "test_gsn_mask.npy"

IMG_SIZE = 128
IMG_CHANNELS = 3

DEFAULT_MB_SIZE = 32
DEFAULT_STAT_PERIOD = 10

DEFAULT_NUM_EPOCHS = 15
DEFAULT_PATIENCE = 3
DEFAULT_EPOCH_TRAIN_EVAL = False
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 0.001

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available():
    print('WARNING! CUDA is not available - running on CPU.')

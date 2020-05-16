import sys

from constants import *
from input import get_dataloader
from network import SegTrainer

X_PATH = TEST_X_PATH
Y_PATH = TEST_Y_PATH


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        X_PATH = sys.argv[1]
        Y_PATH = sys.argv[2]

    trainer = SegTrainer()
    trainer.load_snapshot()
    dataloader = get_dataloader(x_path=X_PATH, y_path=Y_PATH, mb_size=64, drop_last=False, shuffle=False)
    trainer.run_evaluation(dataloader=dataloader, ds_name='EVALUATION')

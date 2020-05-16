import sys

from matplotlib import pyplot as plt

from constants import *
from input import get_dataloader
from network import SegTrainer

X_PATH = TRAIN_X_PATH
Y_PATH = TRAIN_Y_PATH
MODEL_PATH = BEST_SNAPSHOT_PATH
TOP_LOSSES_DIR = './report/img/train/'


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        X_PATH = sys.argv[1]
        Y_PATH = sys.argv[2]
    if len(sys.argv) >= 4:
        TOP_LOSSES_DIR = sys.argv[3]

    trainer = SegTrainer()
    trainer.load_snapshot(path=MODEL_PATH)
    dataloader = get_dataloader(x_path=X_PATH, y_path=Y_PATH, mb_size=64, drop_last=False, shuffle=False)
    trainer.run_evaluation(dataloader=dataloader, ds_name='EVALUATION', store_top_losses=50)
    for i, (loss, img) in enumerate(trainer.top_losses):
        print(loss.item())
        plt.imshow(img)
        plt.savefig(f'{TOP_LOSSES_DIR}{i}.png')
        plt.show()

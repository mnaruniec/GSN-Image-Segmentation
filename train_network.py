import numpy as np

from constants import *
from input import load_file
from network import SegTrainer
from utils import show_censored_img


def main():
    trainer = SegTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
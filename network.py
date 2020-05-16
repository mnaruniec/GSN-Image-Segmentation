from copy import deepcopy
from datetime import datetime
from functools import partial

from albumentations import Blur, JpegCompression, HueSaturationValue, GaussNoise
from torch import nn, optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from constants import *
from input import get_dataloaders
from transforms import *
from utils import iou


DEFAULT_TRAIN_AUGMENTATIONS = [Identity(), HorizontalFlip(), Rotation90(), Rotation270()]
DEFAULT_SELF_AUGMENTATIONS = DEFAULT_TRAIN_AUGMENTATIONS
DEFAULT_FLY_ALBUMENTATIONS = [
    Blur(p=0.15),
    JpegCompression(quality_lower=40, quality_upper=80, p=0.15),
    HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, p=0.15),
    GaussNoise(p=0.15),
]


class SegNet(nn.Module):
    def __init__(
            self,
            conv_channels=DEFAULT_CONV_CHANNELS,
            convs_per_block=DEFAULT_CONVS_PER_BLOCK,
            max_pool_sizes=DEFAULT_MAX_POOL_SIZES,
    ):
        super().__init__()

        assert len(max_pool_sizes) == len(conv_channels) - 1

        conv_blocks = []
        for in_channels, out_channels in zip([IMG_CHANNELS] + conv_channels, conv_channels):
            block_layers = []
            for i in range(convs_per_block):
                in_channels = out_channels if i else in_channels

                block_layers += [
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                ]

            conv_blocks.append(nn.Sequential(*block_layers))

        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.pool_layers = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=size) for size in max_pool_sizes]
        )

        upconv_blocks = []
        for out_channels in reversed(conv_channels[:-1]):
            block_layers = []
            for i in range(convs_per_block):
                in_channels = out_channels if i else 2 * out_channels

                block_layers += [
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                ]

            upconv_blocks.append(nn.Sequential(*block_layers))

        self.upconv_blocks = nn.ModuleList(upconv_blocks)

        self.upsample_layers = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=size, stride=size)
                for in_channels, out_channels, size in reversed(list(zip(
                    conv_channels[1:],
                    conv_channels,
                    max_pool_sizes
                )))
            ]
        )

        self.last_layer = nn.Conv2d(in_channels=conv_channels[0], out_channels=2, kernel_size=1)

    def forward(self, x):
        conv_tensors = []

        for conv, pool in zip(self.conv_blocks, self.pool_layers):
            x = conv(x)
            conv_tensors.append(x)
            x = pool(x)

        x = self.conv_blocks[-1](x)

        for upsample, upconv, skip in zip(
                self.upsample_layers,
                self.upconv_blocks,
                reversed(conv_tensors),
        ):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = upconv(x)

        x = self.last_layer(x)

        return x


class SegTrainer:
    def __init__(
            self,
            optimizer_lambda = partial(optim.Adam, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY, amsgrad=True),
            mb_size=DEFAULT_MB_SIZE,
            num_epochs=DEFAULT_NUM_EPOCHS,
            patience=DEFAULT_PATIENCE,
            stat_period=DEFAULT_STAT_PERIOD,
            stat_mbs=DEFAULT_STAT_MBS,
            epoch_train_eval=DEFAULT_EPOCH_TRAIN_EVAL,
            self_augmentations=DEFAULT_SELF_AUGMENTATIONS,
            train_augmentations=DEFAULT_TRAIN_AUGMENTATIONS,
            fly_albumentations=DEFAULT_FLY_ALBUMENTATIONS,
            **net_kwargs,
    ):
        assert self_augmentations
        assert train_augmentations

        self.net_kwargs = net_kwargs

        self.optimizer_lambda = optimizer_lambda

        self.mb_size = mb_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.stat_period = stat_period
        self.stat_mbs = stat_mbs
        self.epoch_train_eval = epoch_train_eval
        self.self_augmentations = self_augmentations

        self.net = None
        self.criterion = None
        self.optimizer = None

        self.train_dl, self.valid_dl, self.test_dl = get_dataloaders(
            train_augmentations=train_augmentations,
            fly_albumentations=fly_albumentations,
        )

    def init_net(self):
        self.net = SegNet(**self.net_kwargs)
        self.net.to(DEVICE)
        self.net.train()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer_lambda(self.net.parameters())

    def evaluate_on(self, dataloader: DataLoader, full=False) -> (float, int, float, float):
        """ Returns (pixel_acc, pixel_count, avg_loss, avg_iou) """
        with torch.no_grad():
            net = self.net
            net.eval()

            correct = 0
            pixel_total = 0

            running_loss = 0.
            i = 0

            iou_sum = 0.
            img_total = 0

            for data in dataloader:
                i += 1
                images, labels = data

                outputs = 0
                for aug in self.self_augmentations:
                    aug_images = aug.apply(images)
                    aug_outputs = net(aug_images)
                    outputs += aug.reverse(aug_outputs)

                outputs /= len(self.self_augmentations)

                _, predicted = torch.max(outputs.data, 1)

                pixel_total += labels.size(0) * labels.size(1) * labels.size(2)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                img_total += len(labels)
                for target, pred in zip(labels, predicted):
                    iou_sum += iou(target, pred)

                if not full and i >= self.stat_mbs:
                    break

        net.train()
        return correct / pixel_total, pixel_total, running_loss / i, iou_sum / img_total

    def run_evaluation(self, dataloader, ds_name: str = ''):
        acc, total, loss, iou = self.evaluate_on(dataloader, full=True)

        print(f'{ds_name} stats: acc: {(100 * acc):.2f}%, iou: {(100 * iou):.2f}%, loss: {loss:.4f}')

        return acc, iou, loss

    def train_batch(self, data):
        inputs, labels = data

        self.net.train()
        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, reset_net=True, plot_loss=True):
        if reset_net:
            self.init_net()

        train_losses = []
        valid_losses = []
        epoch_losses = []
        epoch_x = 0
        epoch_xs = []

        last_epoch = 0
        best_state_dict = None
        best_epoch = 0
        best_epoch_loss = 10 ** 9

        try:
            for epoch in range(1, self.num_epochs + 1):
                print(f'EPOCH {epoch}')

                train_loss = 0.0
                last_epoch = epoch

                for i, data in enumerate(self.train_dl, 0):
                    train_loss += self.train_batch(data)

                    if i % self.stat_period == self.stat_period - 1:
                        epoch_x += 1

                        train_loss = train_loss / self.stat_period
                        train_losses.append(train_loss)

                        acc, total, valid_loss, iou = self.evaluate_on(self.valid_dl)

                        valid_losses.append(valid_loss)

                        print(f'Epoch {epoch}, batch {i + 1}, train loss: {train_loss:.4f}, '
                              f'valid acc: {100 * acc:.2f}%, valid iou: {100 * iou:.2f}%, valid loss: {valid_loss:.4f}')

                        train_loss = 0.0

                _, _, epoch_loss = self.run_evaluation(self.valid_dl, 'VALID')

                epoch_losses.append(epoch_loss)
                epoch_xs.append(epoch_x)

                # early stopping & snapshotting
                if epoch_loss < best_epoch_loss:
                    best_epoch_loss = epoch_loss
                    best_epoch = epoch
                    best_state_dict = deepcopy(self.net.state_dict())
                elif len(epoch_losses) > self.patience:
                    if all((l > best_epoch_loss for l in epoch_losses[-(self.patience + 1):])):
                        print(f'No improvement in last {self.patience + 1} epochs, early stopping.')
                        break

                if self.epoch_train_eval:
                    self.run_evaluation(self.train_dl, 'TRAIN')

        finally:
            if plot_loss:
                plt.plot(
                    range(len(train_losses)), train_losses, 'r',
                    range(len(valid_losses)), valid_losses, 'b',
                    epoch_xs, epoch_losses, 'g',
                )
                plt.show()

            if best_state_dict and best_epoch != last_epoch:
                print(f'Restoring snapshot from epoch {best_epoch} with valid loss: {best_epoch_loss:.4f}')
                self.net.load_state_dict(best_state_dict)

            acc, iou, loss = self.run_evaluation(self.test_dl, 'TEST')

            snapshot_path = SNAPSHOT_PATH\
                            + f'Snap_a{10000 * acc:.0f}_i{10000 * iou:.0f}_{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
            torch.save(best_state_dict, snapshot_path)
            #self.run_evaluation(self.train_dl, 'TRAIN')

            return acc, loss

    def load_snapshot(self, path: str = BEST_SNAPSHOT_PATH):
        self.init_net()
        self.net.load_state_dict(torch.load(path))

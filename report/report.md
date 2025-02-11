---
output:
  html_document: default
  pdf_document: default
---
Michał Naruniec\
360386

# Deep Neural Networks
## Assignment 2

### Usage

There are two scripts: train_network and evaluate_network.

The first one runs the training  of a fresh network.

The second one loads a snapshot, evaluates it on the given dataset and gathers 50 samples with highest loss.

### Architecture

I used almost original U-Net described [here](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
I added batch normalization before each convolution and paddings, so that only max pooling and transpose convolution changes the size.
Below you can see my diagram of the architecture:

![arch](architecture.jpg)

### Data augmentation

I implemented two "levels" of augmentation plus self-augmentation.

First, I extend dataset four times, by applying identity, 90 and 270 degree rotations and horizontal flip.
I use the same four transformations for self-augmentation.

On top of that, I use 4 transformations from the Albumentations library on the fly.
This means that in each epoch I use 8000 images, but each of them has 0.15 chance to have any of four transformations applied: Blur, JpegCompression, HueSaturation (aka. ColorJitter), GaussNoise.

However, the best model I trained used only one on-the-fly augmentation (Blur). Further ones slightly decreased the evaluation metrics.

### Evaluation and analysis

#### Test set metrics

The best model achieved 96.74% pixel accuracy, and 93.83% IoU on the test set.

#### Worst case scenarios

I have gathered images with largest loss values from both training and test datasets.
I haven't noticed anything special about the training samples

| | | | |
|:---:|:---:|:---:|:---:|
| ![](img/train/46.png) | ![](img/train/47.png) | ![](img/train/48.png) | ![](img/train/49.png) |

The test samples do reveal a pattern. They display people which are pasted onto the background.
Their edges are irregular, especially when you can see singular hairs.
Sometimes there are also occlusions, like in the pictures with the girl of Asian descendance.
The network probably has a hard time on the edges, and might be also confused by differences in the lighting.
The train dataset samples did not have these anomalies, so it might be the case that it does not have such images to train on.

| | | | |
|:---:|:---:|:---:|:---:|
| ![](img/test/49.png) | ![](img/test/48.png) | ![](img/test/47.png) | ![](img/test/46.png) |
| ![](img/test/45.png) | ![](img/test/44.png) | ![](img/test/43.png) | ![](img/test/42.png) |
| ![](img/test/41.png) | ![](img/test/40.png) | ![](img/test/39.png) | ![](img/test/38.png) |
| ![](img/test/37.png) | ![](img/test/36.png) | ![](img/test/35.png) | ![](img/test/34.png) |
| ![](img/test/33.png) | ![](img/test/32.png) | ![](img/test/31.png) | ![](img/test/30.png) |

#### Pixel prediction entropy

The histograms below show the entropy of values returned by the softmax on the outputs from the network.
Large entropy values mean distributions similar to uniform, low ones - almost constant distributions.

As you can see , most ot the correctly classified pixels come from cases, where the network is pretty sure of its decision.
In general, there were about 140k false positives, 130k false negatives, 4600k true positives and 3300k true negatives.
The misclassified pixels unsurprisingly come mostly from the pixels with almost uniform distribution.
However, there is plenty of false positives with large commitment.
The model seems to be biased towards positive detections, which is not suprising, considering that most of the target pixels are actually true-valued.

| | TRUE | FALSE|
|:---:|:---:|:---:|
| **POSITIVE** | ![TP](img/entropy/TP.png) | ![FP](img/entropy/FP.png) |
| **NEGATIVE** | ![TN](img/entropy/TN.png) | ![FN](img/entropy/FN.png) |

#### Conclusion

The hardest cases for the network are probably the ones in which the person is artificially pasted onto the background in a sloppy way, is occluded or has some holes and is on minority of the picture.

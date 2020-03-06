# Computer Vision

## COCO

Image Classification - Common Objects in Context

Convolution Neural Network - COCO21

| Layer                | Filters | Size/Stride |    Input    |   Output   |
|:--------------------:|:-------:|:-----------:|:-----------:|:----------:|
| Convolution2D        |      32 |       3x3/1 |  224x224x3  | 224x224x32 |
| Convolution2D        |      32 |       3x3/1 |  224x224x32 | 224x224x32 |
|  MaxPooling2D        |         |       3x3/2 |  224x224x32 | 112x112x32 |
| Convolution2D        |      64 |       3x3/1 |  112x112x32 | 112x112x64 |
| Convolution2D        |      64 |       3x3/1 |  112x112x64 | 112x112x64 |
|  MaxPooling2D        |         |       3x3/2 |  112x112x64 |  56x56x64  |
| Convolution2D        |     128 |       3x3/1 |   56x56x64  |  56x56x128 |
| Convolution2D        |      64 |       1x1/1 |   56x56x128 |  56x56x64  |
| Convolution2D        |     128 |       3x3/1 |   56x56x64  |  56x56x128 |
|  MaxPooling2D        |         |       3x3/2 |   56x56x128 |  28x28x128 |
| Convolution2D        |     256 |       3x3/1 |   28x28x128 |  28x28x256 |
| Convolution2D        |     128 |       1x1/1 |   28x28x256 |  28x28x128 |
| Convolution2D        |     256 |       3x3/1 |   28x28x128 |  28x28x256 |
|  MaxPooling2D        |         |       3x3/2 |   28x28x256 |  14x14x256 |
| Convolution2D        |     512 |       3x3/1 |   14x14x256 |  14x14x512 |
| Convolution2D        |     256 |       1x1/1 |   14x14x512 |  14x14x256 |
| Convolution2D        |     512 |       3x3/1 |   14x14x256 |  14x14x512 |
| Convolution2D        |     256 |       1x1/1 |   14x14x512 |  14x14x256 |
| Convolution2D        |     512 |       3x3/1 |   14x14x256 |  14x14x512 |
|  MaxPooling2D        |         |       3x3/2 |   14x14x512 |   7x7x512  |
| Convolution2D        |    1024 |       3x3/1 |    7x7x512  |   7x7x1024 |
| Convolution2D        |     512 |       1x1/1 |    7x7x1024 |   7x7x512  |
| Convolution2D        |    1024 |       3x3/1 |    7x7x512  |   7x7x1024 |
| Convolution2D        |     512 |       1x1/1 |    7x7x1024 |   7x7x512  |
| Convolution2D        |    1024 |       3x3/1 |    7x7x512  |   7x7x1024 |
| Convolution2D        |      80 |       1x1/1 |    7x7x1024 |   7x7x80   |
| GlobalAveragePooling |         |      global |    7x7x80   |       80   |
| Softmax              |         |             |        80   |       80   |

## MNIST

Image Classification - Fashion-MNIST

```
Test Accuracy 93.70%
```

## NICS

Image Caption - Nueral Image Caption System

<img src="NICS/nics300x300_better.png" align="center">

## SSMD

Object Detection - Single Shot Multi Detector

<p align="center">
  <img src="SSMD/ssmd.gif">
</p>

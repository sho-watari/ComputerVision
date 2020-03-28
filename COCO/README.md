# COCO

Image Classification - Common Objects in Context

## coco21

| Layer                | Filters | Size/Stride |    Input    |   Output   |
|:--------------------:|:-------:|:-----------:|:-----------:|:----------:|
| Convolution2D        |      32 |       3x3/1 |  3x224x224  | 32x224x224 |
| Convolution2D        |      32 |       3x3/1 |  32x224x224 | 32x224x224 |
|  MaxPooling2D        |         |       3x3/2 |  32x224x224 | 32x112x112 |
| Convolution2D        |      64 |       3x3/1 |  32x112x112 | 64x112x112 |
| Convolution2D        |      64 |       3x3/1 |  64x112x112 | 64x112x112 |
|  MaxPooling2D        |         |       3x3/2 |  64x112x112 |  64x56x56  |
| Convolution2D        |     128 |       3x3/1 |   64x56x56  |  128x56x56 |
| Convolution2D        |      64 |       1x1/1 |   128x56x56 |  64x56x56  |
| Convolution2D        |     128 |       3x3/1 |   64x56x56  |  128x56x56 |
|  MaxPooling2D        |         |       3x3/2 |   128x56x56 |  128x28x28 |
| Convolution2D        |     256 |       3x3/1 |   128x28x28 |  256x28x28 |
| Convolution2D        |     128 |       1x1/1 |   256x28x28 |  128x28x28 |
| Convolution2D        |     256 |       3x3/1 |   128x28x28 |  256x28x28 |
|  MaxPooling2D        |         |       3x3/2 |   256x28x28 |  256x14x14 |
| Convolution2D        |     512 |       3x3/1 |   256x14x14 |  512x14x14 |
| Convolution2D        |     256 |       1x1/1 |   512x14x14 |  256x14x14 |
| Convolution2D        |     512 |       3x3/1 |   256x14x14 |  512x14x14 |
| Convolution2D        |     256 |       1x1/1 |   512x14x14 |  256x14x14 |
| Convolution2D        |     512 |       3x3/1 |   256x14x14 |  512x14x14 |
|  MaxPooling2D        |         |       3x3/2 |   512x14x14 |   512x7x7  |
| Convolution2D        |    1024 |       3x3/1 |    512x7x7  |   1024x7x7 |
| Convolution2D        |     512 |       1x1/1 |    1024x7x7 |   512x7x7  |
| Convolution2D        |    1024 |       3x3/1 |    512x7x7  |   1024x7x7 |
| Convolution2D        |     512 |       1x1/1 |    1024x7x7 |   512x7x7  |
| Convolution2D        |    1024 |       3x3/1 |    512x7x7  |   1024x7x7 |
| Convolution2D        |      80 |       1x1/1 |    1024x7x7 |   80x7x7   |
| GlobalAveragePooling |         |      global |    80x7x7   |       80   |
| Softmax              |         |             |        80   |       80   |

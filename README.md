# CV-Pytorch-cls-det
基于pytorch的CV入门教程（图像分类与目标检测）

Author: Ruilong Ren

Date: 2022-1-12

GPU: 4 × GTX 1080Ti


## Backbones
- LetNet
- AlexNet
- ResNet18/34/50/101/152
- GoogLeNet
- VGG16/19
- DenseNet
- ResNeXt
- SENet
- MobileNetv2
- EfficientNetB0
- Darknet53
- CSPDarknet53

## Datasets
- MNIST
- CIFAR10
- PASCAL VOC2007/2012

## Output(Accurancy)
### CIFAR10
- EPOCH: 100
- LR: CosineAnnealing


|    **METHODS**    | **BEST-ACC** | **TIME(s)** |
| :---------------: | :----------: | :---------: |
|      LetNet       |    71.8%     | **1754.7**  |
|      AlexNet      |    70.09%    |   6292.1    |
|   **ResNet18**    |  **95.23%**  | **3987.9**  |
|     ResNet50      |    95.2%     |   8468.4    |
|     GoogLeNet     |    95.02%    | **13144.3** |
|       VGG16       |    93.8%     |    9618     |
|    DenseNet121    |    94.55%    |   7735.3    |
|  ResNeXt29_32×4d  |    95.18%    |   9573.5    |
|      SENet18      |   95.22 %    |   6988.3    |
|    MobileNetv2    |    93.37%    |   5995.6    |
|  EfficientNetB0   |    90.49%    |   7210.8    |
|     Darknet53     |    91.85%    |   8440.4    |
|   CSPDarknet53    |    88.75%    |   5898.9    |
|    MobileNetv3    |              |             |
| ShuffleNetv2_x0.5 |              |             |



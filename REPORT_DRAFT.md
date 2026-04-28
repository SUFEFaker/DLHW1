# HW1 实验报告初稿：从零实现三层 MLP 进行 EuroSAT 地表覆盖分类

## 1. 实验目标

本次作业目标是在 EuroSAT_RGB 遥感图像数据集上，从零实现一个三层神经网络分类器，用于 10 类地表覆盖图像分类。实现过程中不使用 PyTorch、TensorFlow、JAX 等自动微分框架，仅使用 NumPy 完成矩阵计算、自动微分、反向传播、SGD 优化、学习率衰减、交叉熵损失、L2 正则化、模型选择和测试评估。

代码仓库链接：TODO: 填写 Public GitHub Repo 链接

模型权重下载地址：TODO: 填写 Google Drive 或其他网盘链接

## 2. 数据集与预处理

实验使用 EuroSAT_RGB 数据集，包含 10 个类别：

AnnualCrop、Forest、HerbaceousVegetation、Highway、Industrial、Pasture、PermanentCrop、Residential、River、SeaLake。

数据集按 0.70 / 0.15 / 0.15 划分为训练集、验证集和测试集，具体数量如下：

| Split | 样本数 |
|---|---:|
| Train | 18900 |
| Validation | 4050 |
| Test | 4050 |

图像输入尺寸为 64 x 64 x 3。每张图像首先转换为 RGB 格式，并缩放到指定尺寸，然后将像素值归一化到 [0, 1]。训练集统计得到的通道均值和标准差用于所有 split 的标准化：

| Channel | Mean | Std |
|---|---:|---:|
| R | 0.3445 | 0.2037 |
| G | 0.3803 | 0.1365 |
| B | 0.4079 | 0.1148 |

## 3. 模型结构与实现

模型为三层全连接 MLP，结构如下：

```text
Input(64 x 64 x 3)
-> Linear(input_dim, hidden_dim_1)
-> Activation
-> Linear(hidden_dim_1, hidden_dim_2)
-> Activation
-> Linear(hidden_dim_2, 10)
-> Softmax Cross Entropy
```

本实验实现了 `ReLU`、`Sigmoid` 和 `Tanh` 激活函数，并支持通过命令行切换隐藏层维度、激活函数、学习率、学习率衰减、权重衰减和训练轮数。

自动微分部分实现了轻量级 Tensor 计算图，支持加减乘除、矩阵乘法、sum、mean、reshape 以及激活函数的反向传播。交叉熵损失使用数值稳定的 softmax cross entropy 实现。优化器使用 SGD，并在每个 epoch 后进行指数学习率衰减。训练过程中根据验证集准确率保存最佳模型权重。

主要代码模块如下：

| 模块 | 功能 |
|---|---|
| `hw1/autograd.py` | Tensor 计算图和反向传播 |
| `hw1/data.py` | 数据加载、划分、归一化和 batch 生成 |
| `hw1/nn.py` | Linear 层和 ThreeLayerMLP |
| `hw1/losses.py` | Cross entropy 和 L2 regularization |
| `hw1/optim.py` | SGD 和学习率衰减 |
| `hw1/engine.py` | 单个 epoch 的训练、验证、测试逻辑 |
| `train.py` | 训练入口 |
| `evaluate.py` | 加载最优模型并测试 |
| `search.py` | 网格搜索或随机搜索 |
| `analyze.py` | 权重可视化和错例分析 |

## 4. Baseline 实验

Baseline 使用如下配置：

| 参数 | 值 |
|---|---|
| hidden_dims | [256, 128] |
| activation | ReLU |
| learning_rate | 0.05 |
| lr_decay | 0.95 |
| weight_decay | 1e-4 |
| batch_size | 64 |
| epochs | 20 |

Baseline 结果如下：

| 指标 | 值 |
|---|---:|
| Best epoch | 18 |
| Best validation accuracy | 0.3795 |
| Test accuracy | 0.3756 |
| Test loss | 1.8018 |

Baseline 的测试准确率较低，说明默认学习率、隐藏层规模和正则化组合并不适合当前任务。由于输入为原始像素展开后的向量，MLP 无法像 CNN 一样直接利用局部空间结构，因此对超参数较敏感。

## 5. 超参数搜索

本实验使用随机搜索调节学习率、隐藏层大小、正则化强度和激活函数。搜索命令如下：

```powershell
python search.py --data-root EuroSAT_RGB --search-mode random --num-trials 10 --epochs 8 --batch-size 64 --learning-rates 0.1 0.05 0.02 0.01 --weight-decays 0 1e-4 5e-4 --hidden-dim-options 256,128 512,256 128,128 --activations relu tanh
```

10 组随机搜索结果如下：

| Trial | Learning Rate | Hidden Dims | Weight Decay | Activation | Best Val Acc |
|---|---:|---|---:|---|---:|
| 001 | 0.01 | [128, 128] | 1e-4 | ReLU | 0.6081 |
| 002 | 0.02 | [128, 128] | 0 | Tanh | 0.5620 |
| 003 | 0.10 | [256, 128] | 5e-4 | Tanh | 0.5212 |
| 004 | 0.02 | [512, 256] | 0 | ReLU | 0.6341 |
| 005 | 0.05 | [512, 256] | 5e-4 | ReLU | 0.5296 |
| 006 | 0.05 | [512, 256] | 5e-4 | Tanh | 0.5644 |
| 007 | 0.02 | [128, 128] | 0 | ReLU | 0.5965 |
| 008 | 0.10 | [512, 256] | 0 | ReLU | 0.1111 |
| 009 | 0.10 | [128, 128] | 1e-4 | ReLU | 0.1111 |
| 010 | 0.01 | [256, 128] | 5e-4 | ReLU | 0.6215 |

搜索结果显示，学习率为 0.1 时模型训练明显不稳定，验证准确率接近随机分类水平。ReLU 整体优于 Tanh，较大的隐藏层 [512, 256] 在学习率 0.02 下获得最高验证准确率。基于搜索结果，选择 `learning_rate=0.02, hidden_dims=[512, 256], weight_decay=0, activation=ReLU` 作为最终模型配置。

## 6. 最终模型训练结果

最终模型配置如下：

| 参数 | 值 |
|---|---|
| hidden_dims | [512, 256] |
| activation | ReLU |
| learning_rate | 0.02 |
| lr_decay | 0.95 |
| weight_decay | 0 |
| batch_size | 64 |
| epochs | 12 |

最终模型结果如下：

| 指标 | 值 |
|---|---:|
| Best epoch | 6 |
| Best validation accuracy | 0.6447 |
| Test accuracy | 0.6373 |
| Test loss | 1.0083 |

与 baseline 相比，最终模型测试准确率从 0.3756 提升到 0.6373，说明合适的学习率和更大的隐藏层容量对当前 MLP 分类器非常重要。

训练曲线如下：

![Training curves](runs/best_lr002_hd512x256_relu/training_curves.png)

从曲线可以看到，训练集 loss 随 epoch 稳定下降，训练准确率持续上升。验证集准确率在第 6 个 epoch 达到最高值 0.6447，之后验证集表现出现波动，而训练集仍继续提升。这说明模型在第 6 个 epoch 后开始出现一定程度的过拟合，因此保存验证集最佳模型是必要的。

## 7. 测试集混淆矩阵

测试集混淆矩阵如下：

![Confusion matrix](runs/best_lr002_hd512x256_relu/test_confusion_matrix.png)

从混淆矩阵可以观察到，模型对 Forest、Industrial、Residential、SeaLake 等类别识别较好，但对视觉纹理相近的类别仍存在明显混淆。例如 HerbaceousVegetation 容易被分为 PermanentCrop，Highway 容易被分为 River，PermanentCrop 容易被分为 AnnualCrop。这些类别在遥感图像中都可能包含细长纹理、绿色植被或规则农田纹理，而 MLP 使用展平后的像素向量，缺少对局部空间结构和平移不变性的建模能力。

## 8. 第一层权重可视化

将第一层权重恢复为 64 x 64 x 3 的图像形式，得到如下可视化结果：

![First layer weights](runs/best_lr002_hd512x256_relu/first_layer_weights.png)

第一层权重图中可以观察到明显的颜色响应差异。部分神经元对绿色通道或蓝色通道更敏感，可能对应植被、水体等遥感图像中常见的颜色模式。也有一些权重呈现较弱的空间块状或条纹状分布，说明模型在一定程度上学习到了全局颜色和粗糙空间纹理。然而，由于 MLP 将图像直接展平，第一层权重并不像卷积核那样专门捕捉局部边缘或局部纹理，因此其空间模式不够清晰，难以形成稳定的局部特征检测器。

## 9. 错例分析

为了避免只展示单一类别的错例，本实验从测试集中选取了 5 类高频混淆对的代表样本：

![Typical misclassified examples](runs/best_lr002_hd512x256_relu/typical_misclassified_examples.png)

选取的错例如下：

| 图像 | 真实类别 | 预测类别 | 可能原因 |
|---|---|---|---|
| `HerbaceousVegetation_182.jpg` | HerbaceousVegetation | PermanentCrop | 二者都包含大面积绿色植被纹理，且纹理重复性较强，MLP 难以区分自然草本覆盖和规则作物区域。 |
| `Highway_1171.jpg` | Highway | River | 道路和河流在遥感图中都可能呈现细长连续结构，颜色或背景复杂时容易被展平像素模型混淆。 |
| `Forest_1181.jpg` | Forest | SeaLake | 若图像中存在较暗区域或蓝绿色色调，模型可能更依赖颜色统计而非真实地物边界，导致森林被误判为水体相关类别。 |
| `PermanentCrop_621.jpg` | PermanentCrop | AnnualCrop | PermanentCrop 与 AnnualCrop 都是农田类场景，具有规则纹理和相似颜色分布，类别边界本身较细。 |
| `Residential_1497.jpg` | Residential | Highway | 住宅区中可能存在道路网络、灰色屋顶和规则线性结构，模型可能把这些线性结构当作 Highway 特征。 |

这些错例说明，MLP 能学习一定的颜色和全局纹理统计，但对遥感图像中细粒度空间结构的建模能力有限。道路、水体边界、农田纹理和植被纹理都依赖局部形状和空间排列，仅使用展平像素向量会丢失图像局部邻域的归纳偏置。

## 10. 总结

本实验从零实现了一个基于 NumPy 的三层 MLP 分类器，完成了自动微分、反向传播、SGD 优化、学习率衰减、L2 正则化、验证集最优模型保存、超参数搜索、测试集评估、混淆矩阵、第一层权重可视化和错例分析。

实验结果表明，超参数对 MLP 在 EuroSAT 上的表现影响显著。Baseline 测试准确率为 0.3756，而经过随机搜索后，最终模型测试准确率提升到 0.6373。最佳模型使用 ReLU 激活、[512, 256] 隐藏层和 0.02 初始学习率。尽管该结果明显优于 baseline，但模型仍在植被、农田、道路和水体等视觉相似类别上存在混淆。主要原因是 MLP 缺少卷积结构，不能有效利用图像的局部空间关系。因此，在不改变作业要求的前提下，当前实现已经完成三层 MLP 分类器的训练和分析；若允许使用卷积神经网络，预计可以进一步提升遥感图像分类性能。

## 附录：复现实验命令

Baseline：

```powershell
python train.py --data-root EuroSAT_RGB --run-name baseline --epochs 20 --batch-size 64 --learning-rate 0.05 --lr-decay 0.95 --weight-decay 1e-4 --activation relu --hidden-dims 256 128
```

随机搜索：

```powershell
python search.py --data-root EuroSAT_RGB --search-mode random --num-trials 10 --epochs 8 --batch-size 64 --learning-rates 0.1 0.05 0.02 0.01 --weight-decays 0 1e-4 5e-4 --hidden-dim-options 256,128 512,256 128,128 --activations relu tanh
```

最终模型：

```powershell
python train.py --data-root EuroSAT_RGB --run-name best_lr002_hd512x256_relu --epochs 12 --batch-size 64 --learning-rate 0.02 --lr-decay 0.95 --weight-decay 0 --activation relu --hidden-dims 512 256
```

权重可视化：

```powershell
python analyze.py --checkpoint runs/best_lr002_hd512x256_relu/best_model.npz --metadata runs/best_lr002_hd512x256_relu/best_model.json weights
```

错例分析：

```powershell
python analyze.py --checkpoint runs/best_lr002_hd512x256_relu/best_model.npz --metadata runs/best_lr002_hd512x256_relu/best_model.json errors
```

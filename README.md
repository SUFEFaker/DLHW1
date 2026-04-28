# HW1: 从零开始实现三层 MLP 分类 EuroSAT

本项目使用 `NumPy` 从零实现了一个三层神经网络分类器，用于 EuroSAT 遥感图像分类，满足作业中对以下模块的要求：

- 数据加载与预处理
- 三层 MLP 模型定义
- 自动微分与反向传播
- 训练循环、SGD、学习率衰减、L2 正则化
- 验证集选最优模型
- 测试集评估与混淆矩阵
- 网格搜索 / 随机搜索
- 第一层权重可视化与错例分析辅助脚本

## 1. 环境依赖

建议使用 Python 3.10+。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖说明：

- `numpy`：矩阵运算、模型训练
- `Pillow`：读取 EuroSAT 图像
- `matplotlib`：绘制训练曲线、权重图、混淆矩阵热力图和错例图

## 2. 数据集放置方式

当前目录下应包含：

```text
hw1.pdf
EuroSAT_RGB/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

默认数据根目录为 `EuroSAT_RGB`。

## 3. 项目结构

```text
hw1/
  autograd.py       # Tensor / Parameter / 自动微分
  data.py           # 数据集划分、归一化、batch 加载
  engine.py         # 训练/验证/测试共享逻辑
  experiments.py    # 训练实验主流程
  losses.py         # softmax cross-entropy / L2
  metrics.py        # accuracy / confusion matrix
  nn.py             # Linear / ThreeLayerMLP
  optim.py          # SGD / LR decay
  utils.py          # JSON / checkpoint / 随机种子
  visualize.py      # 训练曲线、权重图、错例图
train.py            # 训练脚本
evaluate.py         # 导入最优权重并测试
search.py           # 超参数搜索
analyze.py          # 权重可视化、错例分析
requirements.txt
README.md
```

## 4. 训练

一个基础训练命令示例：

```bash
python train.py \
  --data-root EuroSAT_RGB \
  --run-name baseline \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.05 \
  --lr-decay 0.95 \
  --weight-decay 1e-4 \
  --activation relu \
  --hidden-dims 256 128
```

常用参数：

- `--hidden-dims 256 128`：两层隐藏层宽度
- `--activation relu|sigmoid|tanh`：激活函数切换
- `--learning-rate`：初始学习率
- `--lr-decay`：每个 epoch 的学习率衰减系数
- `--weight-decay`：L2 正则化强度
- `--train-ratio --val-ratio --test-ratio`：数据划分比例
- `--max-train-samples / --max-val-samples / --max-test-samples`：调试用子集

训练输出默认保存在 `runs/<run_name>/`，主要文件包括：

- `best_model.npz`：验证集最优权重
- `best_model.json`：模型与数据配置
- `history.json`：每个 epoch 的 loss / accuracy
- `training_curves.png`：训练/验证集 loss 曲线 + accuracy 曲线
- `test_metrics.json`：测试集指标
- `test_confusion_matrix.csv`：测试集混淆矩阵
- `test_confusion_matrix.png`：混淆矩阵热力图

## 5. 导入最优模型做测试

```bash
python evaluate.py \
  --checkpoint runs/baseline/best_model.npz
```

如果 `best_model.json` 不在同目录，可以显式指定：

```bash
python evaluate.py \
  --checkpoint runs/baseline/best_model.npz \
  --metadata runs/baseline/best_model.json
```

可选参数：

- `--split train|val|test`
- `--batch-size 128`
- `--output-dir runs/baseline`

## 6. 超参数搜索

网格搜索示例：

```bash
python search.py \
  --search-mode grid \
  --epochs 8 \
  --learning-rates 0.1 0.05 0.01 \
  --weight-decays 0.0 1e-4 5e-4 \
  --hidden-dim-options 256,128 512,256 128 \
  --activations relu tanh
```

随机搜索示例：

```bash
python search.py \
  --search-mode random \
  --num-trials 6 \
  --epochs 8 \
  --learning-rates 0.1 0.05 0.01 \
  --weight-decays 0.0 1e-4 5e-4 \
  --hidden-dim-options 256,128 512,256 128 \
  --activations relu tanh
```

搜索结果会写入：

- `runs/search/<search_name>/search_results.json`
- `runs/search/<search_name>/best_trial.json`

## 7. 生成报告所需图像

### 7.1 第一层权重可视化

```bash
python analyze.py \
  --checkpoint runs/baseline/best_model.npz \
  weights \
  --max-filters 25
```

输出：

- `runs/baseline/first_layer_weights.png`

### 7.2 错例分析图

```bash
python analyze.py \
  --checkpoint runs/baseline/best_model.npz \
  errors \
  --split test \
  --num-samples 16 \
  --cols 4
```

输出：

- `runs/baseline/misclassified_examples.png`
- `runs/baseline/misclassified_examples.json`

其中 JSON 会记录每张错分图像的：

- 相对路径
- 真实类别
- 预测类别

## 8. 实现细节说明

### 自动微分

`hw1/autograd.py` 中实现了一个轻量 Tensor 计算图，支持本作业需要的基础运算：

- 加减乘除
- 矩阵乘法
- `sum` / `mean`
- `reshape`
- `ReLU` / `Sigmoid` / `Tanh`

损失函数 `softmax_cross_entropy` 单独实现了数值稳定版本，并将梯度回传到 logits。

### 模型结构

默认模型结构为：

```text
Input -> Linear -> Activation -> Linear -> Activation -> Linear -> Softmax
```

即三层线性层，对应两个隐藏层和一个输出层。

### 优化与正则化

- 优化器：SGD
- 学习率衰减：指数衰减
- 正则化：L2 regularization
- 模型选择：按验证集准确率保存最优权重

## 9. 与作业要求的对应关系

- 自动微分与反向传播：`hw1/autograd.py`
- 数据加载与预处理：`hw1/data.py`
- 模型定义：`hw1/nn.py`
- 训练循环：`train.py` + `hw1/engine.py` + `hw1/experiments.py`
- 测试评估：`evaluate.py`
- 超参数搜索：`search.py`
- 混淆矩阵：`evaluate.py` / `train.py` 输出
- 权重可视化：`analyze.py weights`
- 错例分析：`analyze.py errors`

## 10. 说明

- 当前核心训练逻辑不依赖 PyTorch / TensorFlow / JAX。
- 若只想快速检查流程是否正常，可以通过 `--max-train-samples` 等参数先跑一个小子集。
- 正式提交时，建议将实验报告、GitHub Repo 链接和训练好权重的下载地址一并整理。

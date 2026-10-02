# PyTorch 基础复习

本文围绕一个分类任务，串联 Tensor、`nn.Module`、模型结构、Dataset、DataLoader、损失函数、自动求导和优化器。

**示例任务：每个样本有 4 个特征，模型判断其属于类别 0 还是类别 1。**

## 训练流程

```text
Dataset 提供样本
    ↓
DataLoader 把样本组成一批 Tensor
    ↓
模型前向计算，得到预测分数
    ↓
损失函数衡量预测误差
    ↓
自动求导计算参数梯度
    ↓
优化器更新参数
    ↓
处理下一个批次
```

## 1. Tensor 基本操作

Tensor 是 PyTorch 存储和计算数据的基本对象，可以是标量、向量、矩阵或更高维数组。首先关注三个属性：**形状 `shape`、数据类型 `dtype`、设备 `device`**。

```python
import torch

x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])

print(x.shape)   # torch.Size([2, 3])
print(x.dtype)   # 默认浮点类型为 torch.float32 时：torch.float32
print(x.device)  # cpu
```

这里可以把 `[2, 3]` 理解为 2 个样本，每个样本有 3 个特征。

| 操作 | 含义 | 此例结果形状 |
|---|---|---|
| `x[:, 0]` | 取所有样本的第一个特征 | `[2]` |
| `x.mean(dim=0)` | 沿第 0 维求均值，得到每个特征的平均值 | `[3]` |
| `x.mean(dim=1)` | 沿第 1 维求均值，得到每个样本的平均值 | `[2]` |
| `x.T` | 二维矩阵转置 | `[3, 2]` |
| `x.reshape(3, 2)` | 重新组织形状 | `[3, 2]` |
| `x * x` | 逐元素相乘 | `[2, 3]` |
| `x @ x.T` | 矩阵乘法 | `[2, 2]` |
| `torch.cat([x, x], dim=0)` | 沿已有维度拼接 | `[4, 3]` |
| `torch.stack([x, x], dim=0)` | 新增一个维度后堆叠 | `[2, 2, 3]` |

注意：

- `dim` 指定操作沿哪个维度进行；求均值默认会去掉该维度，`keepdim=True` 可以保留它。
- `reshape` 与转置不同。例如本例中，`x.reshape(3, 2)` 的第一行为 `[1, 2]`，`x.T` 的第一行为 `[1, 4]`。
- 普通模型输入通常使用浮点数；类别编号通常使用 `torch.long`。
- Tensor 的 `.to(device)` 返回转换后的张量，通常需要接住返回值：`x = x.to(device)`。

## 2. nn.Module

`nn.Module` 是模型和网络层的基础类，负责管理子模块、参数以及训练／评估模式。

```python
from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)
```

- `__init__`：创建模型时定义有哪些层。
- `forward`：定义每次前向计算时，数据如何经过这些层。
- `model(x)`：正常调用模型，并让 PyTorch 处理相关模块调用机制。
- `model.parameters()`：取得注册的模型参数，供优化器使用。
- `model.to(device)`：把模型参数及注册的缓冲区移到指定设备。
- `model.train()`／`model.eval()`：切换训练／评估模式。

把网络层赋给 `self.fc1` 等属性，会将这些子模块及其参数注册到模型中。可训练参数默认需要梯度。

## 3. 定义模型结构：代码、形状与公式

上面的网络结构为：

```text
输入 [B, 4]
    ↓ Linear(4, 16)
隐藏表示 [B, 16]
    ↓ ReLU
隐藏表示 [B, 16]
    ↓ Linear(16, 2)
输出分数 [B, 2]
```

其中 $B$ 是一个批次的样本数量。数学表达式为：

$$
H = \operatorname{ReLU}(XW_1^\top + b_1),
\qquad
Z = HW_2^\top + b_2.
$$

| 对象 | 含义 | 形状 |
|---|---|---|
| $X$ | 输入数据 | $B \times 4$ |
| $W_1$ | 第一层权重 | $16 \times 4$ |
| $b_1$ | 第一层偏置 | $16$ |
| $H$ | 隐藏表示 | $B \times 16$ |
| $W_2$ | 第二层权重 | $2 \times 16$ |
| $b_2$ | 第二层偏置 | $2$ |
| $Z$ | 两个类别的原始分数 | $B \times 2$ |

偏置通过广播加到每个样本上。ReLU 逐元素计算 $\operatorname{ReLU}(u)=\max(0,u)$。

**`nn.Linear(输入维度, 输出维度)` 的权重形状是 `[输出维度, 输入维度]`，因此批量计算写成 $XW^\top+b$。**

这个模型共有：

$$
4 \times 16 + 16 + 16 \times 2 + 2 = 114
$$

个可训练参数。ReLU 没有参数，但提供非线性；若去掉它，这两层仿射变换可以合并为一层。

## 4. Dataset 和 DataLoader

对于常见的按索引读取的数据集：

- **Dataset 管样本**：提供样本数量以及第 $i$ 个样本的数据和标签。自定义时通常实现 `__len__` 与 `__getitem__`。
- **DataLoader 管批次**：把样本组成批次，并支持打乱顺序等功能。

```python
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(128, 4)
y = (X[:, 0] > 0).long()

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

这里标签规则是：第一个特征大于 0 时属于类别 1，否则属于类别 0。`TensorDataset` 已经实现按索引读取样本的接口。

```text
整个数据集：X [128, 4]，y [128]
一个批次： xb [16, 4]，yb [16]
```

- **批次**：一次处理的一组样本，本例为 16 个。
- **训练迭代**：本例中处理一个批次，并进行一次参数更新。
- **训练轮次（epoch）**：遍历整个训练集一次，本例包含 $128/16=8$ 次迭代。

## 5. 损失函数

损失函数把预测与目标之间的差异变成可以优化的数值。

| 任务 | 常用损失函数 | 传入的模型输出 |
|---|---|---|
| 连续数值回归 | `nn.MSELoss()` | 预测数值 |
| 二分类或多标签分类 | `nn.BCEWithLogitsLoss()` | 每个标签的原始分数 |
| 互斥的多类别分类 | `nn.CrossEntropyLoss()` | 每个类别的原始分数 |

本例用两个输出分数表示两个互斥类别，所以选择 `CrossEntropyLoss`：

```python
loss_fn = nn.CrossEntropyLoss()
logits = model(xb)           # [B, 2]
loss = loss_fn(logits, yb)   # 默认求平均，得到标量
```

采用类别编号标签时：

```text
logits：浮点数，[B, C]
标签：torch.long，[B]，取值为 0 到 C-1
```

其中 $C$ 是类别数。设 $z_{b,c}$ 是样本 $b$ 对类别 $c$ 的原始分数，$y_b$ 是其正确类别。普通、不加权交叉熵为：

$$
p_{b,c} = \frac{e^{z_{b,c}}}{\sum_{j=0}^{C-1} e^{z_{b,j}}},
\qquad
L = -\frac{1}{B}\sum_{b=1}^{B}\log p_{b,y_b}.
$$

**传入 `CrossEntropyLoss` 前不要做 softmax 或 argmax。** 它内部完成对应的稳定计算；softmax 可以用于展示预测概率，argmax 用于选择预测类别。

## 6. 自动求导机制

PyTorch 在前向计算时记录相关运算，反向计算时应用链式法则，将参数梯度存入 `.grad`。

```python
w = torch.tensor(2., requires_grad=True)
loss = (3 * w - 1) ** 2
loss.backward()

print(w.grad)  # tensor(30.)
print(w)       # 数值仍然为 2
```

因为：

$$
L(w)=(3w-1)^2,
\qquad
\frac{dL}{dw}=2(3w-1)\times3.
$$

在 $w=2$ 时，梯度为 30。

需要记住：

- **`loss.backward()` 计算梯度，不更新参数。**
- 梯度默认累积，因此普通训练中，每批求导前需要清除旧梯度。
- 普通训练输入通常不需要设置 `requires_grad=True`，模型参数需要。
- `torch.no_grad()` 关闭代码块内的梯度记录，常用于预测。

## 7. 优化器

优化器根据梯度更新指定的参数。例如：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

`model.parameters()` 指定需要更新的参数，`lr` 是控制更新幅度的学习率。

不带动量和权重衰减的 SGD 更新为：

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L_t.
$$

其中 $\theta_t$ 为当前参数，$\eta$ 为学习率，$\nabla_\theta L_t$ 为当前批次的损失梯度。Adam 和 AdamW 会进一步使用梯度的历史信息调整更新方式。

| 操作 | 作用 |
|---|---|
| `optimizer.zero_grad(set_to_none=True)` | 清除旧梯度，将对应的 `.grad` 设为 `None` |
| `loss.backward()` | 计算并累积当前梯度 |
| `optimizer.step()` | 使用梯度更新参数 |

## 8. 完整训练与预测示例

以下代码可以独立保存为 Python 脚本，不依赖前面的变量。

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)


# 1. 创建练习数据：第一个特征大于 0，标签就是 1。
torch.manual_seed(42)
X = torch.randn(128, 4)
y = (X[:, 0] > 0).long()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 2. 创建模型、损失函数和优化器。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 3. 训练。
for epoch in range(20):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)  # 清除旧梯度
        logits = model(xb)                    # 前向计算
        loss = loss_fn(logits, yb)            # 计算损失
        loss.backward()                      # 计算梯度
        optimizer.step()                     # 更新参数

        total_loss += loss.item() * xb.size(0)

    average_loss = total_loss / len(dataset)
    print(f"第 {epoch + 1} 轮，平均损失：{average_loss:.4f}")

# 4. 对新样本进行预测。
model.eval()
with torch.no_grad():
    new_x = torch.randn(3, 4).to(device)
    logits = model(new_x)                  # [3, 2]
    probabilities = logits.softmax(dim=1)  # 每个样本的类别概率
    predictions = logits.argmax(dim=1)     # [3]，预测类别编号

print("类别概率：", probabilities.cpu())
print("预测类别：", predictions.cpu())
```

`loss.item()` 取出损失数值用于记录。将每批平均损失乘以该批样本数，再除以总样本数，得到整轮的平均损失。

这个示例用于理解训练流程；真实任务还需要独立的验证／测试数据来评估泛化表现。

## 9. 容易混淆的地方

| 常见混淆 | 正确理解 |
|---|---|
| `*` 与 `@` | 前者逐元素相乘，后者做矩阵乘法 |
| `reshape` 与转置 | 结果形状相同，也不代表元素排列相同 |
| `forward` 与 `backward` | 前者计算预测，后者计算梯度 |
| `backward()` 与 `step()` | 前者求梯度，后者更新参数 |
| `zero_grad()` 与参数初始化 | 清除梯度不会重置模型参数 |
| `train()` 与执行训练 | `train()` 只切换模式，训练仍需前向、求导和更新 |
| `eval()` 与 `no_grad()` | 前者改变部分层的行为，后者关闭梯度记录 |
| 输出分数与概率 | logits 可以为任意实数，softmax 后才得到类别概率 |

`eval()` 主要影响 Dropout、BatchNorm 等层。本例没有这些层，但预测时仍保留 `eval()` 与 `no_grad()` 的常见写法。

## 自测问题

1. `nn.Linear(4, 16)` 的权重与偏置形状分别是什么？
2. 批次输入是 `[32, 4]` 时，本例模型的输出形状是什么？
3. 为什么 `CrossEntropyLoss` 的输入是 `[B, 2]`，类别标签却是 `[B]`？
4. `backward()`、`step()`、`zero_grad()` 分别改变什么？
5. 为什么预测时经常同时使用 `eval()` 和 `no_grad()`？
6. 128 个样本、批次大小 16 时，一轮训练有多少次参数更新？

## 官方参考资料

- [Tensor 基本操作](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Dataset 与 DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [自动求导基础](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)
- [求导模式与评估模式](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [完整训练流程](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

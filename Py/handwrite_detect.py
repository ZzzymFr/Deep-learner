import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

plt.rcParams['font.sans-serif'] = ['Heiti TC','PingFang SC', 'Songti SC','STHeiti', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

'''
数据预处理
- ToTensor() 把图片从像素值 0 到 255 转成浮点张量，并除以 255，数值落到 0 到 1。黑白图只有一个通道，张量形状会变成 [1, 高, 宽]，通道放在最前面。
- Normalize((0.1307,), (0.3081,)) 对每个像素做 (像素 - 0.1307) / 0.3081。0.1307 是这类手写数字图片的平均亮度，
0.3081 是它们的标准差；括号里各只有一个数，因为只有一个通道。缩放之后，像素大致分布在 0 附近，后面训练时的梯度更稳。
0.3081和0.1307是业界标准
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#下载并保存数据
train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

#分成128张图一份，一共有60000/128份，训练时打开shuffle以防记住顺序
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

print(f'train_set: {len(train_loader)}, test_loader: {len(test_loader)}')
print(f'{train_set[0][0].shape} channels*height*width')

fig ,axes = plt.subplots(3,3, figsize = (10,10))
for i,x in enumerate(axes.flat):
    img,label = train_set[i]
    x.imshow(img.squeeze()*0.3801+0.1307,cmap='gray')
    x.set_title(f'label: {label}', fontsize=12)
    x.axis('off')
fig.suptitle('MNIST training set', fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

torch.manual_seed(1)
'''
w1 = (torch.randn(28*28,512)*(2.0/784)**0.5).requires_grad_(True)
b1 = torch.zeros(10, requires_grad=True)
w2 = (torch.randn(512,128)*(2.0/512)**0.5).requires_grad_(True)
b2 = torch.zeros(10, requires_grad=True)
w3 = torch.randn(128,10)*(2.0/128)**0.5
b3 = torch.zeros(10, requires_grad=True)

params = [w1, b1, w2, b2, w3, b3]
lr = 0.01

print(f'网络共{sum(p.numel() for p in params):,}个参数')
'''

model = nn.Sequential(
    nn.Linear(28*28, 512),
    nn.SiLU(),
    nn.Linear(512, 256),
    nn.SiLU(),
    nn.Linear(256, 128),
    nn.SiLU(),
    nn.Linear(128,10)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)
model.train()
# 每一轮都会走完 60000 张训练图，几轮就能看到准确率。
for epoch in range(10):

    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        # Linear 要的是每一行 784 个数，图片现在是 [批大小, 1, 28, 28]。
        flat = images.view(images.size(0), -1)
        optimizer.zero_grad()
        logits = model(flat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    print(f"epoch: {epoch}, 训练损失： {total_loss / total:7.4f}, 训练准确率： {correct / total:7.4f}")

model.eval()
test_loss = 0.0
correct = 0
total = 0

for images, labels in test_loader:
        flat = images.view(images.size(0), -1)
        logits = model(flat)
        loss = criterion(logits, labels)
        test_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
print(f"测试损失： {test_loss / total:7.4f}, 测试准确率： {correct / total:7.4f}")

with torch.no_grad():
    images, labels = next(iter(test_loader))
    pred = model(images.view(images.size(0), -1)).argmax(dim=1)
for i in range(30):
    print(f"样例 {i}: 实际 {labels[i].item()}，预测 {pred[i].item()}")
import numpy as np
import matplotlib
import torch
import torch.nn as nn




x = torch.tensor([[1.0]])
y = torch.tensor([[0.0]])

#-------------model----------------
model = nn.Sequential(nn.Linear(1, 1, bias=False),
                          nn.ReLU(),
                          nn.Linear(1, 1, bias=False),
                          nn.ReLU(),
                          nn.Linear(1, 1, bias=False))
with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[2].weight.fill_(2.0)
        model[4].weight.fill_(-1)
criterion = nn.MSELoss()#均方差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#优化器，这里是随机梯度下降 SGD。它根据 loss 的梯度去改 model 里的可学习参数（那些 Linear 的权重）。

for epoch in range(100):
        loss = criterion(model(x), y)#计算损失
        optimizer.zero_grad()#清空之前计算的梯度
        loss.backward()#反向传播，计算梯度
        optimizer.step()#自动更新权重（+梯度*lr）

        if epoch %10 == 0:
            print(f"epoch: {epoch}, 预测值： {model(x).item():7.4f}, 损失： {loss.item():7.4f}")
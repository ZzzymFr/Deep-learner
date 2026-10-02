import numpy as np
import matplotlib
import torch
import torch.nn as nn


class mlp_bp_sgd():
    inp : torch.Tensor
    out : torch.Tensor
    model : nn.Sequential
    lr : float

    def __init__(self, inp : torch.Tensor, out: torch.Tensor, model: nn.Sequential, lr : float):
        self.inp = inp
        self.out = out
        self.model = model
        self.lr = lr
    def run(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        for epoch in range(100):
            loss = criterion(self.model(self.inp), self.out)#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch %10 == 0:
                preds = self.model(self.inp).view(-1)
                text = ", ".join(f"{v:7.4f}" for v in preds.tolist())
                print(f"epoch: {epoch}, 预测值： {text}, 损失： {loss.item():7.4f}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).item()*100

x = torch.tensor([[1.0]])
y = torch.tensor([[0.0]])

#-------------model----------------
model = nn.Sequential(nn.Linear(1, 1, bias=True),
                          nn.SELU(),
                          nn.Linear(1, 1, bias=False),
                          nn.SELU(),
                          nn.Linear(1, 1, bias=False))
with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[2].weight.fill_(2.0)
        model[4].weight.fill_(-1)
item = mlp_bp_sgd(x, y, model, 0.01)
item.run()

# 一套二手房：86 平方米、房龄 8 年、距地铁 1.2 公里，成交价 168 万。
# 三个输入和房价都缩放过，数值才落在这个学习率跟得上的范围里。
torch.manual_seed(0)
x = torch.tensor([[86.0 / 100, 8.0 / 30, 1.2 / 5],
                  [150.0 / 100,2.0/30, 1/5]])
y = torch.tensor([[168.0 / 100],
                  [430.0/100]])

#-------------model----------------
model = nn.Sequential(nn.Linear(3, 16),
                      nn.SiLU(),
                      nn.Linear(16, 16),
                      nn.SiLU(),
                      nn.Linear(16, 1))
item = mlp_bp_sgd(x, y, model, 0.05)
item.run()
# 训练已经结束，这里只做预测，no_grad 表示不再记录梯度。
with torch.no_grad():
    # model(x) 的形状是 [2, 1]：两套房子，每套一个预测。
    # view(-1) 把它摊成一串数，避免对多个数调用 .item()。
    # 训练时房价除过 100，这里乘回去，单位变回万元。
    trained = model(x).view(-1) * 100
# 两个预测各格式化成一位小数，用逗号连成一行。
print("成交价预测： " + ", ".join(f"{v:.1f} 万元" for v in trained.tolist()))
pred = item.predict(torch.tensor([[200.0 / 100,10/30, 0.5/5]]))
print(f"新房子预测： {pred:.1f} 万元")
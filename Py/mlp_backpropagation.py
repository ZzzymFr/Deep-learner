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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for epoch in range(100):
            loss = criterion(self.model(self.inp), self.out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch %10 == 0:
                print(f"epoch: {epoch}, 预测值： {self.model(self.inp).item():7.4f}, 损失： {loss.item():7.4f}")

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
item = mlp_bp_sgd(x,y,model,0.01)
item.run()
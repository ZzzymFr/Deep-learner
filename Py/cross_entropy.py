import torch
import torch.nn as nn

# 三个特征：每周学习小时/10、作业完成率、出勤率。
# 标签是类别编号，CrossEntropyLoss 要的是这个，不是分数本身。
# 0 不及格，1 及格，2 优秀。最后一层 Linear(20, 3) 的 3 就是这三类。
x = torch.tensor([
    [0.20, 0.10, 0.30],
    [0.30, 0.20, 0.20],
    [0.10, 0.30, 0.40],
    [0.40, 0.15, 0.25],
    [0.55, 0.60, 0.60],
    [0.50, 0.70, 0.55],
    [0.70, 0.50, 0.65],
    [0.60, 0.55, 0.70],
    [1.20, 0.95, 0.90],
    [1.00, 0.90, 1.00],
    [1.10, 1.00, 0.85],
    [0.90, 0.85, 0.95],
])
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

# 测试数据不参与训练，用来看新学生会被分到哪一类。
test_x = torch.tensor([
    [0.25, 0.20, 0.15],
    [0.15, 0.10, 0.35],
    [0.65, 0.60, 0.50],
    [0.50, 0.65, 0.75],
    [1.05, 0.90, 0.95],
    [0.95, 1.00, 0.80],
])
test_y = torch.tensor([0, 0, 1, 1, 2, 2])
grades = ["不及格", "及格", "优秀"]

model = nn.Sequential(nn.Linear(3,20),
                      nn.SiLU(),
                      nn.Linear(20,20),
                      nn.SiLU(),
                      nn.Linear(20,3))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)
for epoch in range(150):
    optimizer.zero_grad()
    loss = criterion(model.forward(x), y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        # 输出是每名学生在三类上的分数，argmax 取出分数最高的那一类。
        pred = model(x).argmax(dim=1)
        text = ", ".join(str(v) for v in pred.tolist())
        print(f"epoch: {epoch}, 预测类别： {text}, 损失： {loss.item():7.4f}")

with torch.no_grad():
    test_pred = model(test_x).argmax(dim=1)
for hours, homework, attend, pred, true in zip(test_x[:, 0] * 10, test_x[:, 1], test_x[:, 2], test_pred, test_y):
    print(f"学习 {hours:.1f} 小时, 作业 {homework:.2f}, 出勤 {attend:.2f} -> 预测 {grades[pred]}, 实际 {grades[true]}")

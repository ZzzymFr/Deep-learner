"""教学用字符级 GPT：只依赖 PyTorch 和 Python 标准库，默认在 CPU 运行。

运行：python minimal_transformer.py --steps 200
检查：python minimal_transformer.py --self-check --steps 200

固定规模：表示维度 D=64，注意力头 H=4，层数 L=2，上下文 T=32。
语料很小，只演示机制；验证句子与训练句子不同，不能据此评价实用能力。
省去 dropout、学习率调度、混合精度和 KV cache，使关键张量过程可见。
"""

import argparse
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


CONTEXT_LENGTH = 32
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2

TRAIN_TEXT = """学习从一个简单的问题开始。
神经网络把输入转换为输出。
输入是一组数字，输出也是一组数字。
权重决定数字之间怎样相互影响。
训练通过误差调整权重。
梯度告诉我们应该向哪个方向调整。
一个词可以用一个向量表示。
一句话可以用一排向量表示。
每个位置保存自己的向量。
注意力让不同位置交换信息。
模型根据已有的文字预测后面的文字。
因果遮罩阻止模型读取未来的文字。
位置向量帮助模型区分文字的顺序。
多个注意力头在不同空间计算关系。
前馈网络分别处理每个位置的向量。
所有位置共享同一组前馈网络权重。
残差连接把输入加回输出。
归一化帮助计算保持稳定。
多层网络逐步形成新的表示。
概率分布表示模型对下一字符的判断。
交叉熵衡量预测与答案之间的差异。
我们可以并行计算训练片段中各个位置的预测。
生成文字时，新字符被接到已有文字后面。
好的实验需要检查数据、计算和结果。
"""

VALID_TEXT = """我们用不同的话检查模型的预测。
注意力根据当前输入计算位置之间的关系。
训练时每个位置只能读取自己和前面的字符。
模型的权重通过梯度逐步调整。
小语料上的结果不能代表模型处理新问题的能力。
"""


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("表示维度必须能被注意力头数整除")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        # True 表示可读取的位置；包括对角线，因为输入位置 t 预测 t+1。
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask[None, None, :, :], persistent=False)

    def forward(self, x):
        batch, time, dim = x.shape                    # x: [B, T, D]
        q, k, v = self.qkv(x).chunk(3, dim=-1)        # 各为 [B, T, D]
        q = q.reshape(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, H, T, dh]，其中 dh = D / H。
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, H, T, T]；行是查询位置，列是被读取的位置。
        scores = scores.masked_fill(~self.causal_mask[:, :, :time, :time], float("-inf"))
        weights = F.softmax(scores, dim=-1)           # 对每行的读取位置归一化
        mixed = weights @ v                          # [B, H, T, dh]
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, dim)
        return self.output(mixed)                    # 合并各头后投影：[B, T, D]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_length):
        super().__init__()
        self.ln_attention = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, context_length)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Pre-LN：先归一化，再计算子层，最后加回残差。
        x = x + self.attention(self.ln_attention(x))
        # Linear 只作用于最后一维；每个位置共享同一个 MLP。
        x = x + self.ffn(self.ln_ffn(x))
        return x


class CharacterGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.context_length = CONTEXT_LENGTH
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.blocks = nn.Sequential(*[
            TransformerBlock(D_MODEL, N_HEADS, CONTEXT_LENGTH) for _ in range(N_LAYERS)
        ])
        self.final_ln = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, token_ids, targets=None):
        if token_ids.ndim != 2:
            raise ValueError("token_ids 应为 [batch, time]")
        _, time = token_ids.shape
        if not 1 <= time <= self.context_length:
            raise ValueError(f"序列长度必须在 1 到 {self.context_length} 之间")
        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        # token: [B, T, D]；position: [T, D]，自动广播到每个样本。
        x = self.blocks(x)
        logits = self.lm_head(self.final_ln(x))        # [B, T, V]，尚未 softmax
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets 必须与 token_ids 形状相同")
            # Cross entropy 内部计算 log-softmax；这里传入原始 logits。
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, new_tokens, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature 必须大于零")
        was_training = self.training
        self.eval()
        try:
            for _ in range(new_tokens):
                # 没有 KV cache：每次重新计算最近 T 个字符。
                # 截断后的位置重新编号为 0..T-1，窗口外的历史不再可见。
                context = token_ids[:, -self.context_length:]
                logits, _ = self(context)
                probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)  # [B, 1]
                token_ids = torch.cat((token_ids, next_id), dim=1)
            return token_ids
        finally:
            self.train(was_training)                 # 恢复调用前的模式


def sample_batch(data, batch_size):
    if len(data) <= CONTEXT_LENGTH:
        raise ValueError("语料长度必须大于上下文长度")
    starts = torch.randint(0, len(data) - CONTEXT_LENGTH, (batch_size,))
    x = torch.stack([data[i:i + CONTEXT_LENGTH] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + CONTEXT_LENGTH + 1] for i in starts.tolist()])
    # x: c0,c1,...,c31；y: c1,c2,...,c32。每个位置预测下一个字符。
    return x, y


@torch.no_grad()
def evaluate(model, data):
    """确定性遍历验证文本；按目标字符数加权平均，不随机抽验证窗口。"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_targets = 0
    try:
        for start in range(0, len(data) - 1, CONTEXT_LENGTH):
            segment = data[start:start + CONTEXT_LENGTH + 1]
            x, y = segment[:-1][None, :], segment[1:][None, :]
            _, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_targets += y.numel()
        if total_targets == 0:
            raise ValueError("验证文本至少需要两个字符")
        return total_loss / total_targets
    finally:
        model.train(was_training)


def self_check(model, train_data):
    """在副本上检查因果性、有限梯度和小批次记忆能力，不改动主模型。"""
    probe = copy.deepcopy(model)
    probe.eval()
    x = train_data[:CONTEXT_LENGTH][None, :]
    boundary = CONTEXT_LENGTH // 2
    altered = x.clone()
    altered[:, boundary:] = (altered[:, boundary:] + 1) % model.vocab_size
    with torch.no_grad():
        original_logits, _ = probe(x)
        altered_logits, _ = probe(altered)
    # 改变未来字符，之前各位置的输出应保持一致。
    torch.testing.assert_close(
        original_logits[:, :boundary, :], altered_logits[:, :boundary, :],
        rtol=1e-5, atol=1e-6,
    )
    if torch.allclose(original_logits[:, boundary:, :], altered_logits[:, boundary:, :]):
        raise AssertionError("检查输入改变后，模型的后半段输出没有变化")

    probe.train()
    x_small, y_small = sample_batch(train_data, batch_size=2)
    probe.zero_grad(set_to_none=True)
    _, loss = probe(x_small, y_small)
    if not torch.isfinite(loss).item():
        raise AssertionError("损失不是有限值")
    loss.backward()
    gradients = [parameter.grad for parameter in probe.parameters()]
    if any(grad is None or not torch.isfinite(grad).all().item() for grad in gradients):
        raise AssertionError("存在缺失或非有限的梯度")
    if not any(grad.abs().sum().item() > 0 for grad in gradients):
        raise AssertionError("所有梯度均为零")

    # 固定同一个小批次，检查完整训练路径是否能明显降低损失。
    initial_loss = loss.item()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        _, loss = probe(x_small, y_small)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
        optimizer.step()
        if loss.item() < 0.1:
            break
    with torch.no_grad():
        _, final_loss = probe(x_small, y_small)
    target = min(0.5, initial_loss * 0.25)
    if not torch.isfinite(final_loss).item() or final_loss.item() >= target:
        raise AssertionError(
            f"小批次未充分拟合：{initial_loss:.3f} -> {final_loss.item():.3f}，要求 < {target:.3f}"
        )
    print(f"自检通过：因果遮罩、有限梯度；小批次损失 {initial_loss:.3f} -> {final_loss.item():.3f}")


def main():
    parser = argparse.ArgumentParser(description="最小字符级 GPT，默认 CPU")
    parser.add_argument("--steps", type=int, default=200, help="训练步数，默认 200")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tokens", type=int, default=120, help="生成的新字符数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.tokens < 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("steps、tokens 必须非负；batch-size、learning-rate 必须为正")

    torch.manual_seed(args.seed)
    # 小矩阵不需要大量线程；所有模型、数据和计算均留在 CPU。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # 字符表作为已知输入格式；验证文本不参与梯度或权重更新。
    # 仅出现在验证集的字符没有训练样例，会使验证损失更难降低。
    characters = sorted(set(TRAIN_TEXT + VALID_TEXT))
    char_to_id = {char: index for index, char in enumerate(characters)}
    encode = lambda text: torch.tensor([char_to_id[char] for char in text], dtype=torch.long)
    train_data, valid_data = encode(TRAIN_TEXT), encode(VALID_TEXT)
    model = CharacterGPT(len(characters))
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"CPU | 字符表 {len(characters)} | 参数 {count:,} | D=64 H=4 L=2 T=32")
    if args.self_check:
        self_check(model, train_data)
    # 自检消耗了随机数；统一采样种子，方便比较开启与关闭自检的训练。
    torch.manual_seed(args.seed + 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    model.train()
    print(f"初始验证损失：{evaluate(model, valid_data):.3f}")
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train_data, args.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            valid_loss = evaluate(model, valid_data)
            print(f"步数 {step:4d} | 当前训练批次损失 {loss.item():.3f} | 验证损失 {valid_loss:.3f}")

    prompt = "学习"
    output_ids = model.generate(encode(prompt)[None, :], args.tokens)
    output = "".join(characters[index] for index in output_ids[0].tolist())
    print("\n生成示例（小语料与少量训练下可能不连贯）：")
    print(output)


if __name__ == "__main__":
    main()

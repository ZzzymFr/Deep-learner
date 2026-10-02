"""教学用字符级 GPT：只依赖 PyTorch 和 Python 标准库，在 CUDA 上运行。

运行：python minimal_transformer.py --steps 200
检查：python minimal_transformer.py --self-check --steps 200

固定规模：表示维度 D=64，注意力头 H=4，层数 L=2，上下文 T=32。
语料很小，只演示机制；验证句子与训练句子不同，不能据此评价实用能力。
训练时用 dropout 随机丢掉一部分激活。省去学习率调度、混合精度和 KV cache，使关键张量过程可见。
"""

import argparse
import array
import copy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CONTEXT_LENGTH = 64
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.06

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
重复有助于小模型记住常见的说法。
训练语料中的每个字符都会成为下一次预测的条件。
当前字符只汇总它前面的信息。
当上下文变长时，模型仍然只看最近的一段。
上面提到的注意力、残差和归一化会在每一层重复。
一个小实验也能说明梯度怎样更新权重。
语料里的句子越整齐，下一个字符就越容易预测。
模型能把相似的句子映射到接近的表示。
代表顺序的位置向量和代表字符的词向量加在一起。
我们用训练语料学习规律，再用新的句子查看预测。
损失下降表示预测分布更接近真实的下一个字符。
采样时从当前分布里抽出一个字符，再接到已有文字后面。
同一批里的每条片段都有自己的上下文。
优化器根据平均梯度同时调整全部权重。
学习率决定每一次更新走多远。
权重衰减让过大的权重慢慢变小。
遮罩把未来位置的分数变成负无穷，softmax 之后它们的概率为零。
查询、键和值来自同一段文字，所以称为自注意力。
多头把表示拆开，各自比较不同的关系，再合并回来。
早上七点，闹钟响了两次他才起身。
窗外的梧桐叶子发黄，风一吹就落到自行车筐里。
他烧了一壶水，撕开面包，把果酱涂得一边厚一边薄。
地铁里人很多，播报声被谈话盖住，他只好看屏幕上的站名。
出站后要过一条窄马路，红灯还要再等四十秒。
办公室的窗没有擦，阳光在灰尘里变成一道斜线。
中午他们去巷口吃面，老板记得他不要香菜。
下午突然下雨，他把外套顶在头上跑进便利店。
便利店的灯很白，关东煮的热气糊住了玻璃门。
晚上回家时楼梯间的灯坏了一盏，他摸着扶手慢慢走。
猫在门口等着，尾巴绕住他的小腿，直到罐头被打开。
春天的河水发浑，孩子们把纸船放下去，看它卡在桥墩旁边。
夏天的蝉从中午叫到天黑，楼上的人把窗户关严。
秋天适合晒被子，被子晒完有太阳和灰尘混在一起的味道。
冬天早晨的哈气碰到围巾就湿了一小块。
菜市场最里面的摊主把鱼鳞刮到桶外，水花溅到买菜人的鞋上。
一斤番茄三块二，她挑了四个软一点的，准备晚上做汤。
汤里先放姜片，水开了再下番茄，最后撒一把葱花。
米饭煮得稍微硬，用锅铲翻到底下那一层锅巴。
朋友敲门时带了一袋橘子，橘子皮剥开，屋子里立刻亮了一点。
他们下棋下到第九格就吵起来，谁也不肯承认自己看错了一步。
书摊上的旧小说缺了最后三页，结尾只好靠读者自己补。
图书馆靠窗的位子最难抢，下午四点就被斜阳晒得不能坐。
公交车拐进老街时会颠一下，站着的人一起抓住拉环。
司机说前面修路，这一趟要绕去河边，全程多花十二分钟。
河边的跑步者逆着风，步子很碎，呼吸白茫茫的。
小孩蹲在沙坑里埋一把蓝色塑料铲，起身时忘记带走。
看门人把铲子插回桶里，桶上写着请用完放回原处。
夜市的灯泡用红线串起来，风大的时候整排灯一起摇。
烤玉米抹了辣酱，第一口烫，第二口才吃出甜。
收摊的人把桌子折好，铁腿碰地，声音传过半条街。
月亮升起来的时候，楼与楼之间只剩一条窄窄的光。
他在笔记本上记下今天买了什么，字迹越写越斜。
明天若是晴天，就把晾在廊里的衬衫收进来。
地理课上，老师把长江从西往东画在黑板上，入海口标在上海旁边。
黄河拐了几个大弯，课本里说它携带的泥沙很多。
赤道附近终年炎热，两极则长年覆盖冰雪，企鹅靠拢在一起取暖。
月球本身不发光，我们看见的亮面是它反射的太阳光。
一颗种子先冒出根，再顶开土，长出两片叶子。
蜜蜂在油菜花之间飞，后腿上沾着黄色的花粉。
足球赛进球后，看台上的人同时站起来，哨声过了很久才重新坐下。
游泳课要求先在浅水区闷气，再学习把头侧向一边换气。
钢琴每天练习一刻钟，今天的曲子反复错在同一个小节。
弟弟把积木搭到第八层就倒了，他数清散落的块数，决定换一种搭法。
邮局里要填一张表，收件人、地址和邮编各占一行。
信封粘得不牢，柜员递来胶水，说这样寄到外地不容易开口。
医院走廊很安静，叫号屏跳到三十五号时，一位老人扶着墙慢慢走进去。
药房把药片分成早中晚三小格，标签上写着饭后服用。
工地外围挂着安全帽的图案，卡车倒车时嘀嘀响，路人绕到对面人行道。
桥下有人在钓鱼，浮漂半天不动，桶里只剩半瓶水。
山路转到第三道弯能看见海，海的颜色比天空更深。
旅馆的房间朝北，毛巾是干的，水龙头要拧两圈才有热水。
回程的火车票是靠窗的位子，隧道里手机没有信号，出隧道才收到消息。
博物馆禁止闪光灯，玻璃柜里的青铜鼎上还能看见绿色的锈。
讲解员说这只鼎用来煮肉，不是用来装水，大家这才停止猜测。
周末的菜市场比工作日吵，讨价还价的声音一层压过一层。
裁缝量了袖长，用粉笔在布料背面做了两个记号，说明周四可以来取。
雨停之后柏油路还反光，骑车的人只好把车速放得很慢。
社区通知明天停水六个小时，从上午九点到下午三点，请提前接好。
楼道里新增了一盆绿萝，叶子擦过就亮，土干了要浇透一次。
小孩学会自己系鞋带，左脚总是比右脚松，走两步就蹲下去重系。
老收音机只能收到两个台，一个播新闻，一个放很慢的曲子。
停电的晚上大家把蜡烛点在桌上，说话声音不自觉放轻。
蜡烛烧到只剩短短一截时，电来了，屋里突然亮得刺眼。

I am Dario.
Frank is a friend of mine.
He is nice.
Sherry is his girlfriend.
She is a nice person.
Frank bought a bag yesterday.
The bag was manufactured by Louis Vuitton, and it was absolutely gorgeous.
He also has an vehicle, the vehicle was manufactured by Porsche, and its model is 911.
It is an fascinating car.
Milk is white.
Apple is red.
Frank's vehicle is red as well.
Apple produces electronics, such as phones, laptops.
AMD produces cpu and gpu.
Students go to school everyday to gain experience.
University of Toronto is a great school.
Machine learning is essential.
Deep learning and neutral networks can understand human language by translating tokens.
These tokens are then turned into vectors.
Create vector space.
Acceleration is determined by force and mass.
Velocity is different from speed, velocity has direction, and it is a vector.
Deep learning is a branch of machine learning.
C++ is a programming language, and in fact a low level language, which makes it different from language like python.
It needs a compiler to work, rather than python, it needs a interpreter while program is running. In general C++ is harder than python.
However, while dealing with a lot of data, Python is usually cheaper and easier for its easy syntax. 
"""

VALID_TEXT = """我们用不同的话检查模型的预测。
注意力根据当前输入计算位置之间的关系。
训练时每个位置只能读取自己和前面的字符。
模型的权重通过梯度逐步调整。
小语料上的结果不能代表模型处理新问题的能力。
我喜欢吃番茄，我买了番茄，花了十二块钱。

I love dreaming.
I hope I can have a bmw someday, and a big house as well.
Python is easier than C++.
"""


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("表示维度必须能被注意力头数整除")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
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
        weights = self.attn_dropout(weights)         # 训练时随机丢掉一部分注意力
        mixed = weights @ v                          # [B, H, T, dh]
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, dim)
        return self.output(mixed)                    # 合并各头后投影：[B, T, D]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()
        self.ln_attention = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN：先归一化，再计算子层，最后加回残差。
        x = x + self.resid_dropout(self.attention(self.ln_attention(x)))
        # Linear 只作用于最后一维；每个位置共享同一个 MLP。
        x = x + self.resid_dropout(self.ffn(self.ln_ffn(x)))
        return x


class CharacterGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.context_length = CONTEXT_LENGTH
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.embed_dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[
            TransformerBlock(D_MODEL, N_HEADS, CONTEXT_LENGTH, DROPOUT) for _ in range(N_LAYERS)
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
        x = self.embed_dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
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
        token_ids = token_ids.to(next(self.parameters()).device)
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


def wiki_title_paths():
    """all-titles 含全部名字空间，其中 0 号就是条目名，不必再读 ns0 文件。"""
    root = Path(__file__).resolve().parents[1] / "data"
    all_titles = root / "zhwiki-latest-all-titles" / "zhwiki-latest-all-titles"
    ns0 = root / "zhwiki-latest-all-titles-in-ns0" / "zhwiki-latest-all-titles-in-ns0"
    if all_titles.is_file():
        return [all_titles]
    if ns0.is_file():
        return [ns0]
    raise FileNotFoundError(f"在 {root} 下没有找到维基标题文件")


def iter_wiki_titles(path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        has_namespace = "page_namespace" in header
        for line in handle:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if has_namespace:
                _, _, line = line.partition("\t")
            title = line.replace("_", " ").strip()
            if title:
                yield title


def load_wiki_title_chunks():
    chunks = []
    pending = []
    characters = set()
    title_count = 0

    def flush():
        nonlocal pending
        if not pending:
            return
        text = "".join(pending)
        chunks.append(text)
        characters.update(text)
        pending = []

    for path in wiki_title_paths():
        for title in iter_wiki_titles(path):
            pending.append(title)
            pending.append("\n")
            title_count += 1
            if title_count % 200_000 == 0:
                flush()
    flush()
    return chunks, characters, title_count


def encode_text(text, char_to_id):
    ids = array.array("I", (char_to_id[char] for char in text))
    return torch.tensor(ids, dtype=torch.long)


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
            device = next(model.parameters()).device
            _, loss = model(x.to(device), y.to(device))
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
    device = next(probe.parameters()).device
    x = train_data[:CONTEXT_LENGTH][None, :].to(device)
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
    x_small, y_small = x_small.to(device), y_small.to(device)
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
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=0.01)
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
    parser = argparse.ArgumentParser(description="最小字符级 GPT，在 CUDA 上训练")
    parser.add_argument("--steps", type=int, default=200, help="训练步数，默认 200")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tokens", type=int, default=120, help="生成的新字符数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.tokens < 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("steps、tokens 必须非负；batch-size、learning-rate 必须为正")

    if not torch.cuda.is_available():
        parser.error("当前 PyTorch 不能使用 CUDA，训练不会在 CPU 上继续")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 字符表作为已知输入格式；验证文本不参与梯度或权重更新。
    # 仅出现在验证集的字符没有训练样例，会使验证损失更难降低。
    wiki_chunks, wiki_characters, title_count = load_wiki_title_chunks()
    characters = sorted(set(TRAIN_TEXT + VALID_TEXT) | wiki_characters)
    char_to_id = {char: index for index, char in enumerate(characters)}
    encode = lambda text: encode_text(text, char_to_id)
    train_parts = [encode(TRAIN_TEXT + "\n")]
    train_parts.extend(encode(chunk) for chunk in wiki_chunks)
    train_data = torch.cat(train_parts)
    valid_data = encode(VALID_TEXT)
    model = CharacterGPT(len(characters)).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{torch.cuda.get_device_name(device)} | 字符表 {len(characters)} | 参数 {count:,} | "
        f"训练字符 {len(train_data):,} | 维基标题 {title_count:,} | "
        f"D={D_MODEL} H={N_HEADS} L={N_LAYERS} T={CONTEXT_LENGTH}"
    )
    if args.self_check:
        self_check(model, train_data)
    # 自检消耗了随机数；统一采样种子，方便比较开启与关闭自检的训练。
    torch.manual_seed(args.seed + 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    model.train()
    print(f"初始验证损失：{evaluate(model, valid_data):.3f}")
    for step in range(1, 5500):
        x, y = sample_batch(train_data, args.batch_size)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            valid_loss = evaluate(model, valid_data)
            print(f"步数 {step:4d} | 当前训练批次损失 {loss.item():.3f} | 验证损失 {valid_loss:.3f}")

    prompt = "学习"
    output_ids = model.generate(encode(prompt)[None, :].to(device), args.tokens)
    output = "".join(characters[index] for index in output_ids[0].tolist())
    print("\n生成示例（小语料与少量训练下可能不连贯）：")
    print(output)


if __name__ == "__main__":
    main()

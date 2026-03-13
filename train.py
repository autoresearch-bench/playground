"""
Bare-minimum transformer pretraining baseline.
Usage: uv run train.py
"""

import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)


def _round_to_64(x):
    return int(math.ceil(x / 64) * 64)


class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = _round_to_64(4 * n_embd * 2 / 3)
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = SwiGLU(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction=reduction,
            )
            return loss
        return logits

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_EMBD = 512
N_HEAD = 8
N_LAYER = 6
BATCH_SIZE = 64
LR = 6e-4
WARMUP_STEPS = 100

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

model = GPT(vocab_size, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e6:.1f}M")

model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
train_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "train")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

step = 0
total_training_time = 0

while True:
    t0 = time.time()

    x, y, epoch = next(train_loader)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # LR schedule: linear warmup then cosine decay
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    if step < WARMUP_STEPS:
        lr = LR * (step + 1) / WARMUP_STEPS
    else:
        lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        total_training_time += dt

    if step % 50 == 0:
        print(f"step {step:05d} | loss: {loss.item():.4f} | lr: {lr:.2e} | dt: {dt*1000:.0f}ms | remaining: {max(0, TIME_BUDGET - total_training_time):.0f}s")

    step += 1
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_params_M:     {num_params / 1e6:.1f}")

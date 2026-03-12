"""
DNABERT-2 style baseline for the Genome LM Speedrun.

Architecture: 12-layer bidirectional transformer with RoPE, Pre-RMSNorm,
SwiGLU FFN, and masked language modeling (15 % masking) — ~86 M parameters.

The model also exposes `.encode(input_ids) -> (B, D)` (CLS-token repr)
so that prepare.evaluate_downstream() can run linear probing.

Usage: python train.py
"""

import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, PAD_TOKEN,
    make_dataloader, evaluate_mlm_loss, evaluate_downstream,
)

# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _build_rope_cache(head_dim: int, max_len: int,
                      base: int = 10_000, device=None) -> tuple:
    inv_freq = 1.0 / (base ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    ))
    t    = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (T, D/2)
    emb   = torch.cat([freqs, freqs], dim=-1) # (T, D)
    return emb.cos(), emb.sin()


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    T   = q.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, D)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    q   = q * cos + _rotate_half(q) * sin
    k   = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.qkv  = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim,     bias=False)
        # QK-norm for training stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                attn_bias=None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k    = _apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias,
                                           is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (no bias)."""
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DNABERTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn  = MultiHeadAttention(hidden_dim, num_heads)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn   = SwiGLUFFN(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                attn_bias=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class DNABERT2(nn.Module):
    """
    DNABERT-2 style masked language model.

    Default config: ~86 M parameters.
        layers=12, hidden=768, heads=12, intermediate=2048

    Supports:
        forward(input_ids, labels=None, reduction='mean')
            → loss (if labels given) or logits
        encode(input_ids) → (B, hidden_dim) CLS-token representations
            used by evaluate_downstream() for linear probing
    """

    def __init__(
        self,
        vocab_size:       int = VOCAB_SIZE,
        hidden_dim:       int = 768,
        num_heads:        int = 12,
        num_layers:       int = 12,
        intermediate_dim: int = 2048,
        max_seq_len:      int = MAX_SEQ_LEN,
        rope_base:        int = 10_000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embed   = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.blocks  = nn.ModuleList([
            DNABERTBlock(hidden_dim, num_heads, intermediate_dim)
            for _ in range(num_layers)
        ])
        self.norm_f  = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

        head_dim = hidden_dim // num_heads
        cos, sin = _build_rope_cache(head_dim, max_seq_len,
                                     base=rope_base, device="cpu")
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _attn_bias(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build (B, 1, 1, T) float bias: 0 for real tokens, -inf for PAD."""
        pad_mask  = (input_ids == PAD_TOKEN)              # (B, T)
        bias      = input_ids.new_zeros(
            input_ids.size(0), 1, 1, input_ids.size(1), dtype=torch.float
        )
        bias.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        return bias

    def _run_encoder(self, input_ids: torch.Tensor) -> torch.Tensor:
        attn_bias = self._attn_bias(input_ids)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, attn_bias)
        return self.norm_f(x)                # (B, T, D)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return CLS-token representation for downstream linear probing."""
        with torch.no_grad():
            hidden = self._run_encoder(input_ids)
        return hidden[:, 0, :]              # (B, D)

    def forward(self, input_ids: torch.Tensor,
                labels=None, reduction: str = "mean"):
        hidden = self._run_encoder(input_ids)
        logits = self.lm_head(hidden)       # (B, T, V)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM       = 768
NUM_HEADS        = 12
NUM_LAYERS       = 12
INTERMEDIATE_DIM = 2048
BATCH_SIZE       = 64
LR               = 3e-4
WARMUP_STEPS     = 300
WEIGHT_DECAY     = 0.01
GRAD_CLIP        = 1.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda")

model = DNABERT2(
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_dim=INTERMEDIATE_DIM,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e6:.1f}M")

model = torch.compile(model)

optimizer    = torch.optim.AdamW(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
train_loader = make_dataloader(BATCH_SIZE, MAX_SEQ_LEN, "train")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

step                = 0
total_train_time    = 0.0

while True:
    t0 = time.time()

    x, y, _ = next(train_loader)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    # LR schedule: linear warmup → cosine decay to 10 % of peak
    if step < WARMUP_STEPS:
        lr = LR * (step + 1) / WARMUP_STEPS
    else:
        progress = min(total_train_time / TIME_BUDGET, 1.0)
        lr = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    for g in optimizer.param_groups:
        g["lr"] = lr

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        total_train_time += dt

    if step % 50 == 0:
        print(
            f"step {step:05d} | loss: {loss.item():.4f} | lr: {lr:.2e}"
            f" | dt: {dt*1000:.0f}ms | remaining: {max(0, TIME_BUDGET - total_train_time):.0f}s"
        )

    step += 1
    if step > 5 and total_train_time >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    val_loss = evaluate_mlm_loss(model, BATCH_SIZE)

downstream = evaluate_downstream(model, batch_size=BATCH_SIZE)

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_loss:         {val_loss:.6f}")
for key, val in downstream.items():
    print(f"{key:<26}{val:.6f}")
print(f"training_seconds: {total_train_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_params_M:     {num_params / 1e6:.1f}")

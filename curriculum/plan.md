# Research Plan

## Overall Goal
Minimize val_bpb on a ~28M-param GPT model trained for 300s on H100.

## Phase 1: Training Dynamics (COMPLETE)
**Best config**: LR=6e-3, grad_clip=1.0, wd=0.1, betas=(0.9, 0.95), min_lr=6e-4
**Result**: val_bpb = 1.2231 (31.6% improvement over baseline 1.7876)
**10 experiments**, key findings: LR peak at 6e-3, batch=64 optimal, torch.compile hurts, LR-batch coupling confirmed.

## Phase 2: Architecture (NEXT)
Lock Phase 1 best training config, explore architecture:

### Priority Order
1. **Weight tying** - share tok_emb and lm_head weights. Reduces params ~4M, acts as regularizer.
2. **SwiGLU activation** - replace GELU MLP with SwiGLU (8/3 hidden ratio to match param count).
3. **RMSNorm** - replace LayerNorm with RMSNorm (faster, works well in practice).
4. **RoPE** - replace learned positional embeddings with rotary position embeddings.
5. **Depth/width** - try deeper/narrower or shallower/wider variants.
6. **Combined best** - all architecture improvements together.

### Expected Impact
- Prior experience: SwiGLU + RMSNorm + RoPE + weight tying together reached ~1.167.

## Phase 3: Synthesis (Planned)
- Combine best Phase 1 training + best Phase 2 architecture
- Re-optimize LR for new architecture (may need different LR with fewer params from weight tying)

## Phase 4: Advanced (Planned)
- Muon optimizer, QK-norm, exotic techniques

# Research Plan

## Objective
Minimize val_bpb on a 28.3M param GPT model trained for 300s on H100.

## Phase 1: Training Dynamics (CURRENT)
Baseline: LR=3e-4, AdamW(default betas), batch=64, warmup=100, cosine decay to 0, no grad clip, no torch.compile

### Strategy
1. **Baseline** — Run unmodified code to establish val_bpb reference
2. **Aggressive LR + stability** — LR=3e-3, grad_clip=1.0, wd=0.1, betas=(0.9,0.95), min_lr floor
3. **Higher LR** — LR=5e-3, same stability settings
4. **torch.compile** — Enable for ~20% wall-time savings (more steps in budget)
5. **Batch size** — Test 32 or 128 for throughput/convergence tradeoff
6. **Combined best** — Merge all improvements

### Key Insights from Prior Work
- Default LR=3e-4 is ~10-16x too low for this setup → expect ~1.80 bpb
- LR=3e-3 with grad_clip=1.0, wd=0.1 should reach ~1.26
- LR=5e-3 with torch.compile, betas=(0.9,0.95) reached ~1.24 previously
- Gradient clipping essential at higher LR
- min_lr floor at 10% of peak prevents LR from going to 0

## Phase 2: Architecture (Pending)
Lock Phase 1 best training config, explore:
- RMSNorm vs LayerNorm
- SwiGLU vs GELU MLP
- RoPE vs learned positional embeddings
- Weight tying (tok_emb = lm_head)
- Depth/width tradeoffs

## Phase 3: Synthesis (Pending)
Combine best training + best architecture, fine-tune interactions.

## Phase 4: Advanced (Pending)
RoPE scaling, QK-norm, Muon optimizer, exotic techniques.

# Research Plan

## Goal
Minimize val_bpb for a 28.3M param GPT model on climbmix data, 300s training budget on H100.

## Phase 1: Training Dynamics (Current)
**Strategy**: Systematically increase LR from the too-low default, add gradient clipping and weight decay, enable torch.compile, tune optimizer betas, and explore batch size.

### Experiment Queue
1. **Baseline** (no changes) - establish starting point (expect ~1.8 bpb from prior experience)
2. **LR=1e-3** - 3x default, expect significant improvement
3. **LR=3e-3 + grad_clip=1.0 + wd=0.1** - aggressive LR with stability measures
4. **LR=5e-3 + grad_clip=1.0 + wd=0.1 + betas=(0.9,0.95) + min_lr=5e-4** - near-optimal from prior work
5. **torch.compile** added to best config - expect ~20% speedup → more steps
6. **Batch size sweep** (32, 128) with best config
7. **LR=8e-3 or higher** - push boundary further

### Key Hypotheses
- Default LR=3e-4 is ~10-16x too low for this model size/budget
- Gradient clipping essential at high LR to prevent divergence
- torch.compile gives free speedup → more training steps in budget
- betas=(0.9, 0.95) better than default (0.9, 0.999) for LLM pretraining
- min_lr floor prevents learning rate from going too close to 0

## Phase 2: Architecture (Planned)
- RMSNorm vs LayerNorm
- SwiGLU vs GELU
- Weight tying (tok_emb = lm_head)
- RoPE vs learned positional embeddings
- Depth/width trade-offs

## Phase 3: Synthesis (Planned)
- Combine best training + best architecture
- Fine-tune interactions

## Phase 4: Advanced (Planned)
- Muon optimizer
- QK-norm
- Other exotic techniques

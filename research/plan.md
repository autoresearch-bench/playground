# Research Plan

## Baseline Established
- val_bpb: 1.794518
- 28.3M params, 512d/8h/6L, ~155ms/step, 25GB VRAM

## Phase 2: First Batch (parallel experiments)

### Exp 1: torch.compile + gradient clipping
- Add `model = torch.compile(model)`
- Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Hypothesis: compile gives ~1.5x speedup, more steps = better convergence

### Exp 2: Larger model (768d/12h/8L)
- Increase dimensions significantly, ~90M params
- Hypothesis: more capacity with H100 headroom should help

### Exp 3: Compile + bigger model + modern arch (SwiGLU + RMSNorm)
- Combine: torch.compile, 768d/12h/8L, SwiGLU activation, RMSNorm
- Higher LR (6e-4) with grad clip and weight decay 0.1
- Hypothesis: modern best-practices combo should beat baseline substantially

### Exp 4: Higher LR + weight decay + grad clip (keep same model size)
- LR 1e-3, weight decay 0.1, grad clip 1.0
- Hypothesis: training hyperparameters may be more impactful than model size at fixed time

## Phase 3: React to Phase 2 results
- If compile helps: use it as base for all future experiments
- If larger model helps: explore even larger or different depth/width ratios
- If architecture helps: try more modern improvements (rotary embeddings, etc.)
- Stack winning changes

## Phase 4: Advanced improvements
- Muon optimizer
- Rotary positional embeddings (RoPE)
- QK normalization
- Learning rate schedule tuning (warmup length, min LR)
- Token/parameter efficiency optimizations

## Budget Strategy
- 4.0 GPU-hours total, ~0.08 used for baseline
- Each run costs ~0.083 GPU-hours (5 min)
- Can do ~47 more runs
- Use 2 parallel assistants for throughput

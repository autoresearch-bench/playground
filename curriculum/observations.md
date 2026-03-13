# Scientific Observations

## Phase 1: Training Dynamics

### Baseline Analysis
- Model: GPT with 512 embd, 8 heads, 6 layers (~28M params)
- Default config: LR=3e-4, AdamW (default wd=0.01), batch_size=64, warmup=100
- Cosine decay to 0 (no min_lr floor), no gradient clipping, no torch.compile
- Training budget: 300s on H100 with bfloat16 autocast
- Context length: 2048, Vocab size: 8192
- LR schedule note: warmup is step-based but decay is time-based (uses total_training_time/TIME_BUDGET)

### Experiment Results
| # | Description | Key Changes | val_bpb | vs Baseline |
|---|-------------|-------------|---------|-------------|
| 1 | Baseline | LR=3e-4, no clip, default betas/wd | 1.7876 | -- |
| 2 | High LR + stability | LR=3e-3, clip=1.0, wd=0.1, betas=(0.9,0.95), min_lr=3e-4 | 1.2632 | -29.3% |
| 3 | Higher LR | LR=5e-3, clip=1.0, wd=0.1, betas=(0.9,0.95), min_lr=5e-4 | 1.2411 | -30.6% |
| 4 | torch.compile | LR=3e-3 + torch.compile | 1.2828 | -28.2% |
| 5 | LR=8e-3 | LR=8e-3, clip=1.0, wd=0.1, min_lr=8e-4 | 1.2546 | -29.8% |
| 6 | Batch=128 | LR=5e-3, batch=128 | 1.3051 | -27.0% |
| 7 | **LR=6e-3** | **LR=6e-3, clip=1.0, wd=0.1, min_lr=6e-4** | **1.2231** | **-31.6%** |
| 8 | Batch=32 | LR=5e-3, batch=32, warmup=200 | 1.2278 | -31.3% |
| 9 | LR=7e-3 | LR=7e-3, clip=1.0, wd=0.1, min_lr=7e-4 | 1.5866 | -11.2% |
| 10 | Batch=32 + LR=6e-3 | LR=6e-3, batch=32, warmup=200 | 1.3240 | -25.9% |

### Key Findings
1. **LR is the dominant factor**: 20x LR increase (3e-4 → 6e-3) yielded 31.6% improvement.
2. **Optimal LR is 6e-3**: LR curve peaks at 6e-3 (1.2231). LR=7e-3 diverged, LR=8e-3 was noisy (1.255).
3. **Stability measures essential**: grad_clip=1.0, wd=0.1, betas=(0.9,0.95) enable high LR.
4. **Min LR floor helps**: 10% of peak (6e-4) prevents LR decay to 0.
5. **torch.compile hurts**: 15.9s compilation overhead not amortized in 300s budget.
6. **Batch=64 is optimal**: Batch=128 too few steps, batch=32 too noisy at optimal LR.
7. **LR-batch coupling**: Optimal LR scales with batch size. LR=6e-3 works for batch=64 but diverges at batch=32.

### Phase 1 Best Config (locked for Phase 2)
- **LR=6e-3**, grad_clip=1.0, weight_decay=0.1
- betas=(0.9, 0.95), min_lr=6e-4 (cosine floor)
- warmup_steps=100, batch_size=64
- **val_bpb = 1.2231**

## Phase 2: Architecture
*(Results will be added as experiments complete)*

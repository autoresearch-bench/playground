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
| 4 | torch.compile | LR=3e-3 + torch.compile (same stability config) | 1.2828 | -28.2% |

### Key Findings
1. **Baseline confirms low LR**: val_bpb=1.7876 with default LR=3e-4. ~2000 steps in 300s budget.
2. **10x LR is transformative**: LR=3e-3 with stability measures → 1.2632, a 29.3% improvement.
3. **Higher LR keeps helping**: LR=5e-3 → 1.2411, another 1.7% over LR=3e-3. Trend suggests more room.
4. **torch.compile HURTS for short runs**: 15.9s compilation overhead eats into 300s budget. Fewer total steps (~1640 vs ~1830). No per-step speedup. Don't use for this setup.

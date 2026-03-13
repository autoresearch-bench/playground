# Observations

## Baseline Result
- **val_bpb: 1.794518** (baseline to beat)
- Model: GPT, 512 embed, 8 heads, 6 layers, 28.3M params
- Training: ~1930 steps in 300s, ~155ms/step avg
- Peak VRAM: 24.9 GB out of 80 GB (31% utilization)
- Batch size: 64, tokens/batch: 64*2048 = 131K tokens/step

## Key Insights from Baseline
1. **H100 massively underutilized** - only 25/80 GB VRAM used. Room for 2-3x larger model or batch.
2. **No torch.compile** - significant overhead from Python dispatch. Compile could give 1.5-2x speedup.
3. **Step time ~155ms** - for a 28M param model on H100, this is slow without compile.
4. **LR schedule bug** - LR is set AFTER optimizer.step(), so it takes effect next step. Minor issue.
5. **No gradient clipping** - may cause instability, especially at higher LR.

## Potential Improvements (prioritized by expected impact)
1. **torch.compile** - Free speedup → more training steps → better convergence
2. **Larger model** - 768d/12h/8L or similar, use the VRAM headroom
3. **Higher learning rate** - With grad clipping, can push LR higher (6e-4 to 1e-3)
4. **Architecture: SwiGLU + RMSNorm** - Modern improvements, better quality per param
5. **Larger batch size** - 128 or more, better gradient estimates
6. **Weight decay 0.1** - Standard for LLM training
7. **Gradient clipping** - Stabilize training, enable higher LR
8. **Combined changes** - Stack winning improvements

## Experiment Log
| # | Branch | Change | val_bpb | Delta | Status |
|---|--------|--------|---------|-------|--------|
| 0 | baseline | Original train.py | 1.794518 | - | BASELINE |

# Scientific Observations

## Phase 1: Training Dynamics

### Experiment Log
(Results will be recorded here as experiments complete)

### Prior Knowledge (from previous sessions on similar setup)
- LR=3e-4 → ~1.80 bpb (way too low)
- LR=3e-3 + clip=1.0 + wd=0.1 → ~1.26
- LR=5e-3 + compile + betas=(0.9,0.95) → ~1.24
- SwiGLU + RMSNorm + RoPE + weight tying → ~1.17 (architecture changes, Phase 2)

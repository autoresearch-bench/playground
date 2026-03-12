# genome-lm-speedrun (autoresearch)

Train a DNA language model on synthetic human-genome-like sequences and
evaluate on three NucleL-style downstream genomic tasks, as fast as possible.

## Goal

**Minimize val_loss** (masked language modeling cross-entropy on held-out
genomic sequences). Lower is better.

The downstream tasks (linear probing) serve as mechanistic checks that the
pretraining signal actually learns useful genomic representations:

| Task | Type | Chance | Target |
|------|------|--------|--------|
| Promoter detection | Binary | 50 % | ≥ 85 % acc |
| Splice site detection | 3-class | 33 % | ≥ 80 % acc |
| CpG island detection | Binary | 50 % | ≥ 90 % acc |

## What you have

- `prepare.py` — **read-only**. Contains:
  - Fixed constants: `MAX_SEQ_LEN = 512`, `TIME_BUDGET = 300` (5 min),
    `VOCAB_SIZE = 10` (A, T, G, C, N + 5 special tokens)
  - `Tokenizer` class — fixed nucleotide tokenizer
  - `make_dataloader(B, T, split)` — infinite generator of
    `(input_ids, labels, attn_mask)` MLM batches on CUDA
  - `evaluate_mlm_loss(model, batch_size)` — fixed evaluation on the val set.
    Your model must accept `model(x, y, reduction='none')` and return a flat
    tensor of per-token cross-entropy losses (0 at non-masked positions).
  - `evaluate_downstream(model, batch_size)` — linear probing on three tasks.
    Your model must expose `model.encode(input_ids) → (B, D)` (CLS-token repr).
    Returns `{"promoter_acc": float, "splice_acc": float, "cpg_acc": float}`.
  - Data is at `~/.cache/genome-lm-speedrun/`

- A **Modal API key** — use `modal` to provision GPU compute.

- `druids` — the orchestration client library.

## What you must build

Create or modify `train.py` that:

1. Imports from `prepare.py` for data loading and evaluation
2. Defines a DNA language model architecture
3. Trains it within the 5-minute time budget
4. Evaluates using `evaluate_mlm_loss` and `evaluate_downstream`
5. Prints results in this **exact** format:

```
---
val_loss:         <float>
promoter_acc:     <float>
splice_acc:       <float>
cpg_acc:          <float>
training_seconds: <float>
total_seconds:    <float>
peak_vram_mb:     <float>
num_params_M:     <float>
```

(Downstream lines are omitted if `model.encode` is not implemented.)

## Constraints

- **Do not modify `prepare.py`.** It is read-only infrastructure.
- Active parameter budget: **≤ 150 M parameters**.
- Training time budget: **5 minutes wall clock** (`TIME_BUDGET` in `prepare.py`).
- No pretrained weights. No external data beyond what `prepare.py` provides.
- Model interfaces:
  - `model(x, y, reduction='none')` → flat per-token losses
  - `model.encode(input_ids)` → `(B, D)` CLS-token representations (optional but recommended)

## Baseline architecture (train.py)

DNABERT-2 style transformer:

| Component | Value |
|-----------|-------|
| Parameters | ~86 M |
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Intermediate dim | 2048 (SwiGLU) |
| Positional enc. | RoPE |
| Normalization | Pre-RMSNorm |
| Attention | QK-norm + bidirectional SDPA |
| Activation | SwiGLU |
| Weight tying | lm_head ↔ embed |
| Vocabulary | 10 tokens (nucleotide) |
| Objective | MLM (15 % masked) |

## Genomic context — why this benchmark is interesting

Unlike text or protein models, DNA language models face unique challenges:

- **Extreme vocabulary compression**: only 4 nucleotides + N vs. 50 k BPE tokens.
  The model must learn structure at k-mer, motif, and long-range levels.
- **Long-range dependencies**: promoters, enhancers, and splice sites interact
  over thousands of base pairs.
- **Strand symmetry**: the reverse complement of a sequence has equivalent
  biological meaning. Equivariant architectures may help.
- **Repetitive elements**: ~45 % of the human genome is repetitive (TEs, SINEs,
  LINEs). The model must learn to distinguish signal from noise.
- **CpG depletion**: methylation causes C→T mutation at CpG sites, creating
  strong dinucleotide biases the model should capture.

## Compute

Use Modal to spin up GPU instances. The 5-minute budget is a wall-clock
constraint on the training loop (from first forward pass to the step where
budget is exceeded). Compilation and data loading count toward `total_seconds`
but not `training_seconds`.

## Results tracking

Log results to `results.tsv` (tab-separated):

```
commit	val_loss	promoter_acc	splice_acc	cpg_acc	memory_gb	status	description
```

## The loop

LOOP FOREVER:

1. Look at current state: results.tsv, current train.py
2. Decide what to try. What might improve val_loss?
3. Modify train.py (architecture, optimizer, hyperparams, etc.)
4. Call `run_experiment` with a short description.
5. The tool commits, runs on Modal, parses metrics, keeps/discards.
6. Plan your next experiment.
7. Repeat.

## Research directions to explore

Key question: which speedrun techniques transfer from text/protein to DNA?

- **k-mer tokenization**: replace single-nucleotide tokens with overlapping
  6-mers (DNABERT-1 style) or BPE on nucleotide sequences. Larger effective
  context at the cost of vocabulary size.
- **Reverse complement augmentation**: randomly RC-flip sequences during
  training to enforce strand symmetry.
- **Optimizer**: Muon vs AdamW, Lion, schedule-free
- **Architecture**: deeper vs wider, GQA (grouped-query attention), ALiBi,
  Hyena / Mamba (SSM-based for long-range genomics)
- **Masking strategies**: span masking (mask contiguous k-mers), structure-aware
  masking (bias toward CpG / conserved motifs), higher mask rate
- **RoPE tuning**: lower base frequency for genomic scale
- **Batch/LR**: larger batch + higher LR, gradient accumulation
- **Compilation**: `torch.compile` with `mode='max-autotune'`
- **Flash attention**: already via SDPA, but try explicit flash-attn kernel

**NEVER STOP.** Continue experimenting until manually stopped or budget runs out.

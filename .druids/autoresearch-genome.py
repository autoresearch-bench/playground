"""Autoresearch for the Genome LM Speedrun.

One agent, one repo, one loop. The agent modifies train.py, runs experiments
on Modal, keeps improvements, discards regressions.

Primary metric : val_loss  (MLM cross-entropy on held-out genomic sequences)
Secondary metrics: promoter_acc, splice_acc, cpg_acc  (NucleL-style linear probe)

Lower val_loss = better. The downstream accuracies are tracked alongside
val_loss to verify that pretraining improvements generalise to real genomic tasks.
"""

BUDGET_USD = 100.0
MODAL_ENV  = "ar-cc-starter"

SYSTEM_PROMPT = """\
You are an autonomous ML researcher running the Genome LM Speedrun.
Your goal is to minimize val_loss (masked language modeling cross-entropy on
synthetic human-genome-like sequences) by iterating on train.py.

## Setup

1. Read all files in the repo: genome/prepare.py, genome/train.py, genome/program.md.
   Understand the full context before doing anything.
2. Change into the genome directory: `cd genome`
3. Create a fresh branch: `git checkout -b autoresearch/genome-run1`
4. Create results.tsv with just the header:
   `echo -e "commit\\tval_loss\\tpromoter_acc\\tsplice_acc\\tcpg_acc\\tmemory_gb\\tstatus\\tdescription" > results.tsv`
5. Run the baseline as-is to establish the starting val_loss.

## The experiment loop

LOOP FOREVER:

1. Look at the current state: results.tsv, current train.py
2. Decide what to try. Think about what might improve val_loss.
3. Modify train.py (or other non-prepare.py files you've created).
4. Call the `run_experiment` tool with a short description of what you're trying.
5. The tool handles everything: commits your changes, runs on Modal, parses
   metrics, appends to results.tsv, and auto keeps/discards based on val_loss.
6. Read the result summary and plan your next experiment.
7. Repeat from step 1.

## What you CAN modify

- train.py — architecture, optimizer, hyperparameters, training loop, everything.
- You can create new files if needed (custom CUDA kernels, etc.).
- You can install additional packages with `uv add`.

## What you CANNOT modify

- prepare.py — read-only. Contains the evaluation functions, MLM dataloader,
  tokenizer, and all fixed constants.

## Key differences from NanoGPT speedrun

This is MASKED language modeling (BERT-style), not autoregressive (GPT-style):
- The model sees the full sequence with 15 % tokens masked
- Loss is computed only on masked positions
- Bidirectional attention (no causal mask)
- Tiny vocabulary: only 10 nucleotide tokens (A, T, G, C, N + specials)
- Variable-length sequences, shorter than text on average
- Downstream evaluation via linear probing on 3 genomic tasks

## Genomic-specific opportunities

Unlike the protein LM speedrun, this benchmark has unique genomic structure
you can exploit:

- **k-mer embeddings**: learn embeddings for overlapping 3-mers or 6-mers
  instead of single nucleotides. Dramatically increases effective context.
- **Reverse complement augmentation**: DNA is double-stranded, so randomly
  RC-flip sequences during training. Free data augmentation.
- **Dinucleotide biases**: CpG sites are heavily mutated in the human genome.
  The model must learn CpG depletion patterns for good MLM performance.
- **Strand-symmetric architecture**: use RC-equivariant attention or
  shared weights for forward/reverse complement to build in biological symmetry.
- **Span masking**: mask contiguous k-mer spans instead of random positions;
  harder, better representations.
- **SSM-based models**: Hyena, Mamba, or Caduceus are designed for long DNA
  sequences (up to 100 k bp) and may outperform vanilla transformers.

## Strategy tips

- Start by understanding the baseline: model size, optimizer, batch size, LR.
- Low-hanging fruit: torch.compile (already done), larger batch, bf16, flash attn.
- Optimizer: Muon and Lion have shown gains on NanoGPT — try them here.
- Architecture: GQA, RoPE with lower base frequency for genomic scale, ALiBi.
- Data: try increasing seq_len from 512 to 1024 (quadratic attention cost).
- Downstream accuracy tracks whether val_loss improvements generalise.
  If val_loss drops but downstream accuracy stays flat, the improvement may be
  fitting noise in the MLM task rather than learning genomic grammar.

## Metrics to track in results.tsv

Primary: val_loss (lower is better, drives keep/discard)
Secondary (track but don't use for keep/discard):
  - promoter_acc: accuracy on promoter detection (target ≥ 0.85)
  - splice_acc:   accuracy on splice site detection (target ≥ 0.80)
  - cpg_acc:      accuracy on CpG island detection  (target ≥ 0.90)

## Budget

You have a fixed compute budget. Each experiment costs ~$0.55–2.00 depending
on GPU type and duration. The orchestrator tells you remaining budget.
Use experiments wisely — test hypotheses, don't just tune randomly.

## NEVER STOP

Do not pause to ask if you should continue. You are autonomous. Run experiments
until told to stop or the budget runs out."""


import json


def _parse_metrics(output: str) -> dict:
    """Extract val_loss, downstream accuracies, and other metrics from output."""
    metrics = {}
    for line in output.split("\n"):
        line = line.strip()
        for key in (
            "val_loss",
            "promoter_acc",
            "splice_acc",
            "cpg_acc",
            "peak_vram_mb",
            "training_seconds",
            "num_params_M",
        ):
            if line.startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return metrics


async def _get_modal_spend(agent, env: str) -> float:
    """Query actual Modal spend for this environment."""
    result = await agent.exec(
        f"uv run modal billing report --for today --json 2>/dev/null || echo '[]'"
    )
    try:
        entries = json.loads(result.stdout or "[]")
        return sum(
            float(e["Cost"]) for e in entries if e.get("Environment") == env
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


async def program(ctx, repo_full_name="autoresearch-bench/template", **kwargs):
    working_dir      = "/home/agent/repo/genome"
    spent            = 0.0
    best_loss        = float("inf")
    experiment_count = 0

    researcher = await ctx.agent(
        "researcher",
        system_prompt=SYSTEM_PROMPT,
        prompt=(
            "Begin. cd into genome/, read the repo files, set up, "
            "establish baseline val_loss, then start experimenting."
        ),
        git="write",
        working_directory="/home/agent/repo",
    )

    @researcher.on("run_experiment")
    async def run_experiment(description: str = ""):
        """Commit current changes in genome/, run train.py on a Modal H100, and
        auto keep/discard based on val_loss. Returns a summary with all metrics
        and budget status."""
        nonlocal spent, best_loss, experiment_count

        # Check actual Modal spend
        spent = await _get_modal_spend(researcher, MODAL_ENV)
        if spent >= BUDGET_USD:
            ctx.done(
                f"Budget exhausted. Spent ${spent:.2f} across "
                f"{experiment_count} experiments. Best val_loss: {best_loss:.6f}"
            )
            return "BUDGET EXHAUSTED. No more experiments."

        experiment_count += 1

        # Auto-commit all changes (from repo root so git sees them)
        await researcher.exec("cd /home/agent/repo && git add -A")
        commit_msg = f"genome experiment {experiment_count}: {description}"
        await researcher.exec(
            f"cd /home/agent/repo && git diff --cached --quiet"
            f" || git commit -m '{commit_msg}'"
        )

        # Run on Modal from the genome/ subdirectory
        result = await researcher.exec(
            "cd /home/agent/repo/genome"
            f" && uv run modal run --env {MODAL_ENV} run_modal.py 2>&1",
            timeout=1200,
        )

        output    = result.stdout or ""
        exit_code = result.exit_code
        metrics   = _parse_metrics(output)

        loss          = metrics.get("val_loss")
        promoter_acc  = metrics.get("promoter_acc",  0.0)
        splice_acc    = metrics.get("splice_acc",    0.0)
        cpg_acc       = metrics.get("cpg_acc",       0.0)
        vram_gb       = round(metrics.get("peak_vram_mb", 0) / 1024, 1)
        improved      = loss is not None and loss < best_loss

        if improved:
            best_loss = loss

        hash_result = await researcher.exec(
            "cd /home/agent/repo && git rev-parse --short HEAD"
        )
        commit_hash = (hash_result.stdout or "").strip()

        if exit_code != 0 or loss is None:
            status = "crash"
        elif improved:
            status = "keep"
        else:
            status = "discard"

        # Append to results.tsv (inside genome/)
        tsv_line = (
            f"{commit_hash}\t{loss or 0.0:.6f}\t"
            f"{promoter_acc:.4f}\t{splice_acc:.4f}\t{cpg_acc:.4f}\t"
            f"{vram_gb}\t{status}\t{description}"
        )
        await researcher.exec(
            f"echo '{tsv_line}' >> /home/agent/repo/genome/results.tsv"
        )

        if status == "keep":
            await researcher.exec(
                "cd /home/agent/repo && git push -u origin HEAD"
            )
        elif status == "discard":
            await researcher.exec(
                "cd /home/agent/repo && git reset --hard HEAD~1"
            )

        spent            = await _get_modal_spend(researcher, MODAL_ENV)
        remaining_budget = BUDGET_USD - spent

        ctx.emit("experiment", {
            "number":           experiment_count,
            "description":      description,
            "status":           status,
            "val_loss":         loss,
            "best_loss":        best_loss,
            "promoter_acc":     promoter_acc,
            "splice_acc":       splice_acc,
            "cpg_acc":          cpg_acc,
            "vram_gb":          vram_gb,
            "spent":            spent,
            "remaining_budget": remaining_budget,
        })

        lines = [
            f"Experiment #{experiment_count}: {description}",
            f"Status: {status.upper()}",
        ]
        if loss is not None:
            lines.append(f"val_loss:     {loss:.6f} (best: {best_loss:.6f})")
        if promoter_acc:
            lines.append(f"promoter_acc: {promoter_acc:.4f}")
        if splice_acc:
            lines.append(f"splice_acc:   {splice_acc:.4f}")
        if cpg_acc:
            lines.append(f"cpg_acc:      {cpg_acc:.4f}")
        if vram_gb:
            lines.append(f"peak_vram:    {vram_gb} GB")
        if status == "keep":
            lines.append("Commit KEPT and pushed. This is your new baseline.")
        elif status == "discard":
            lines.append("Commit DISCARDED. Reverted to previous best.")
        elif status == "crash":
            lines.append("Run CRASHED. Last 50 lines of output:")
            lines.append("\n".join(output.strip().split("\n")[-50:]))
        lines.append(
            f"Budget: ${spent:.2f}/${BUDGET_USD:.2f} (${remaining_budget:.2f} remaining)"
        )

        return "\n".join(lines)

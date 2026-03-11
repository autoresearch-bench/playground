"""Autoresearch -- single-agent autonomous pretraining research.

One agent, one repo, one loop. The agent modifies train.py, runs
experiments on Modal, keeps improvements, discards regressions.
This is the Karpathy autoresearch format encoded as a druids program.

Budget enforcement: the program tracks cumulative spend and stops
the agent when the budget is exhausted.
"""

BUDGET_USD = 100.0
MODAL_ENV = "ar-cc-starter"

SYSTEM_PROMPT = """\
You are an autonomous ML researcher. Your goal is to minimize val_bpb \
(bits per byte) on a language modeling task by iterating on train.py.

## Setup

1. Read all files in the repo: prepare.py, train.py, program.md. \
Understand the full context before doing anything.
2. Create a fresh branch: `git checkout -b autoresearch/run1`
3. Create results.tsv with just the header: \
`echo -e "commit\\tval_bpb\\tmemory_gb\\tstatus\\tdescription" > results.tsv`
4. Run the baseline as-is to establish the starting val_bpb.

## The experiment loop

LOOP FOREVER:

1. Look at the current state: results.tsv, current train.py
2. Decide what to try. Think about what might improve val_bpb.
3. Modify train.py (or other files you've created).
4. Call the `run_experiment` tool with a short description of what you're trying.
5. The tool handles everything: commits your changes, runs on Modal, parses \
metrics, appends to results.tsv, and auto keeps/discards based on val_bpb.
6. Read the result summary and plan your next experiment.
7. Repeat from step 1.

## What you CAN modify

- train.py -- architecture, optimizer, hyperparameters, training loop, everything.
- You can create new files if needed.
- You can install additional packages with `uv add`.

## What you CANNOT modify

- prepare.py -- read-only. Contains the evaluation function, dataloader, tokenizer.

## Strategy tips

- Start by understanding the baseline. What's the model size? What optimizer? \
What learning rate?
- Low-hanging fruit first: bigger model, better LR, gradient accumulation, \
torch.compile, flash attention.
- Track what works and what doesn't. Look for patterns in results.tsv.
- If you're stuck, try more radical changes: different architecture, different \
optimizer (Muon, Lion), RoPE, RMSNorm, etc.
- Simpler is better, all else being equal. A small improvement from deleting \
code is better than a small improvement from adding complexity.

## Budget

You have a fixed compute budget. Each experiment costs ~$0.55. The orchestrator \
will tell you when the budget is exhausted. Use your experiments wisely.

## NEVER STOP

Do not pause to ask if you should continue. Do not ask "should I keep going?" \
The human might be asleep. You are autonomous. Run experiments until you are \
told to stop or the budget runs out."""


import json
import shlex


def _parse_metrics(output):
    """Extract val_bpb and peak_vram_mb from training output."""
    metrics = {}
    for line in output.split("\n"):
        line = line.strip()
        for key in ("val_bpb", "peak_vram_mb", "training_seconds", "num_params_M"):
            if line.startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return metrics


async def _get_modal_spend(agent, env):
    """Query actual Modal spend for this environment."""
    result = await agent.exec(
        f"uv run modal billing report --for today --json 2>/dev/null"
        f" || echo '[]'",
    )
    try:
        entries = json.loads(result.stdout or "[]")
        return sum(
            float(e["Cost"]) for e in entries
            if e.get("Environment") == env
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


async def program(ctx, repo_full_name="autoresearch-bench/ar-cc-starter", **kwargs):
    working_dir = "/home/agent/repo"
    spent = 0.0
    best_bpb = float("inf")
    experiment_count = 0

    researcher = await ctx.agent(
        "researcher",
        system_prompt=SYSTEM_PROMPT,
        prompt="Begin. Read the repo, set up, establish baseline, then start experimenting.",
        git="write",
        working_directory=working_dir,
    )

    @researcher.on("run_experiment")
    async def run_experiment(description: str = ""):
        """Commit your current changes, run train.py on a Modal H100, and
        auto keep/discard based on val_bpb. Returns a summary with metrics
        and budget status."""
        nonlocal spent, best_bpb, experiment_count

        # Check actual Modal spend
        spent = await _get_modal_spend(researcher, MODAL_ENV)
        if spent >= BUDGET_USD:
            ctx.done(f"Budget exhausted. Spent ${spent:.2f} across {experiment_count} experiments. Best val_bpb: {best_bpb}")
            return "BUDGET EXHAUSTED. No more experiments."

        experiment_count += 1

        # Auto-commit all changes
        await researcher.exec("cd /home/agent/repo && git add -A")
        commit_msg = shlex.quote(f"experiment {experiment_count}: {description}")
        await researcher.exec(
            f"cd /home/agent/repo && git diff --cached --quiet || git commit -m {commit_msg}",
        )

        # Run on Modal
        result = await researcher.exec(
            f"cd /home/agent/repo && uv run modal run --env {MODAL_ENV} run_modal.py 2>&1",
            timeout=900,
        )

        output = result.stdout or ""
        exit_code = result.exit_code
        metrics = _parse_metrics(output)

        bpb = metrics.get("val_bpb")
        vram_gb = round(metrics.get("peak_vram_mb", 0) / 1024, 1)
        improved = bpb is not None and bpb < best_bpb

        if improved:
            best_bpb = bpb

        # Get commit hash
        hash_result = await researcher.exec("git rev-parse --short HEAD")
        commit_hash = (hash_result.stdout or "").strip()

        # Determine status
        if exit_code != 0 or bpb is None:
            status = "crash"
        elif improved:
            status = "keep"
        else:
            status = "discard"

        # Append to results.tsv
        tsv_line = f"{commit_hash}\t{bpb or 0.0:.6f}\t{vram_gb}\t{status}\t{description}"
        await researcher.exec(f"printf '%s\\n' {shlex.quote(tsv_line)} >> /home/agent/repo/results.tsv")

        # Auto keep/discard
        if status == "keep":
            await researcher.exec("cd /home/agent/repo && git push -u origin HEAD")
        elif status == "discard":
            await researcher.exec("cd /home/agent/repo && git reset --hard HEAD~1")

        # Refresh actual spend
        spent = await _get_modal_spend(researcher, MODAL_ENV)
        remaining_budget = BUDGET_USD - spent

        ctx.emit("experiment", {
            "number": experiment_count,
            "description": description,
            "status": status,
            "val_bpb": bpb,
            "best_bpb": best_bpb,
            "vram_gb": vram_gb,
            "spent": spent,
            "remaining_budget": remaining_budget,
        })

        # Build a clean summary for the agent
        summary_lines = [
            f"Experiment #{experiment_count}: {description}",
            f"Status: {status.upper()}",
        ]
        if bpb is not None:
            summary_lines.append(f"val_bpb: {bpb:.6f} (best: {best_bpb:.6f})")
        if vram_gb:
            summary_lines.append(f"peak_vram: {vram_gb} GB")
        if status == "keep":
            summary_lines.append("Commit KEPT and pushed. This is your new baseline.")
        elif status == "discard":
            summary_lines.append("Commit DISCARDED. Reverted to previous best.")
        elif status == "crash":
            summary_lines.append("Run CRASHED. Last 50 lines of output:")
            tail = "\n".join(output.strip().split("\n")[-50:])
            summary_lines.append(tail)
        summary_lines.append(f"Budget: ${spent:.2f}/${BUDGET_USD:.2f} (${remaining_budget:.2f} remaining)")

        return "\n".join(summary_lines)

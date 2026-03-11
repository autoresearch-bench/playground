"""Test program: verify the run_experiment tool works end-to-end."""

import json
import shlex

MODAL_ENV = "ar-cc-starter"


def _parse_metrics(output):
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


async def program(ctx, **kwargs):
    working_dir = "/home/agent/repo"
    best_bpb = float("inf")

    researcher = await ctx.agent(
        "researcher",
        system_prompt="You are testing the experiment pipeline. Do exactly what the prompt says.",
        prompt=(
            "Test the run_experiment tool. Do these steps exactly:\n"
            "1. Run `git checkout -b test/pipeline-check`\n"
            "2. Call the `run_experiment` tool with description 'baseline test'\n"
            "3. Report back exactly what the tool returned."
        ),
        git="write",
        working_directory=working_dir,
    )

    @researcher.on("run_experiment")
    async def run_experiment(description: str = ""):
        """Run train.py on Modal and return results."""
        nonlocal best_bpb

        # Commit
        await researcher.exec("cd /home/agent/repo && git add -A")
        commit_msg = shlex.quote(f"test: {description}")
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

        if bpb is not None and bpb < best_bpb:
            best_bpb = bpb

        # Get commit hash
        hash_result = await researcher.exec("cd /home/agent/repo && git rev-parse --short HEAD")
        commit_hash = (hash_result.stdout or "").strip()

        # Determine status
        if exit_code != 0 or bpb is None:
            status = "crash"
        else:
            status = "keep"

        # Try git push
        push_result = await researcher.exec("cd /home/agent/repo && git push -u origin HEAD 2>&1")
        push_ok = push_result.exit_code == 0

        summary = [
            f"=== PIPELINE TEST RESULTS ===",
            f"Exit code: {exit_code}",
            f"Status: {status}",
            f"val_bpb: {bpb}",
            f"peak_vram: {vram_gb} GB",
            f"commit: {commit_hash}",
            f"git push: {'OK' if push_ok else 'FAILED'}",
            f"push output: {(push_result.stdout or '')[:200]}",
        ]
        if status == "crash":
            summary.append(f"Last 20 lines:\n{''.join(output.strip().split(chr(10))[-20:])}")

        result_text = "\n".join(summary)
        ctx.done(result_text)
        return result_text

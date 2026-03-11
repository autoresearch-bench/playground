"""Test program: verify the run_experiment tool works end-to-end."""

import json
import shlex
import traceback

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

    researcher = await ctx.agent(
        "researcher",
        system_prompt="You are testing the experiment pipeline. Call the run_experiment tool exactly once with description 'baseline test'. Report back exactly what it returns, verbatim.",
        prompt="Call the run_experiment tool now with description 'baseline test'. Copy its entire return value into your response.",
        git="write",
        working_directory=working_dir,
    )

    @researcher.on("run_experiment")
    async def run_experiment(description: str = ""):
        """Run train.py on a Modal H100 and return results. Call this with a short description of the experiment."""
        try:
            # Step 1: verify exec works at all
            whoami = await researcher.exec("whoami")
            step1 = f"whoami: {whoami.stdout.strip()} (exit {whoami.exit_code})"

            # Step 2: check Modal
            modal_check = await researcher.exec("cd /home/agent/repo && uv run modal token info 2>&1")
            step2 = f"modal token: exit {modal_check.exit_code}, output: {(modal_check.stdout or '')[:100]}"

            # Step 3: git branch
            await researcher.exec("cd /home/agent/repo && git checkout -b test/pipeline-check 2>&1 || true")
            await researcher.exec("cd /home/agent/repo && git add -A")
            commit_msg = shlex.quote(f"test: {description}")
            await researcher.exec(f"cd /home/agent/repo && git diff --cached --quiet || git commit -m {commit_msg}")

            branch = await researcher.exec("cd /home/agent/repo && git branch --show-current")
            step3 = f"branch: {branch.stdout.strip()}"

            # Step 4: run Modal
            result = await researcher.exec(
                f"cd /home/agent/repo && uv run modal run --env {MODAL_ENV} run_modal.py 2>&1",
                timeout=900,
            )
            output = result.stdout or ""
            metrics = _parse_metrics(output)
            step4 = f"modal exit: {result.exit_code}, val_bpb: {metrics.get('val_bpb', 'N/A')}, last 5 lines: {chr(10).join(output.strip().split(chr(10))[-5:])}"

            # Step 5: try git push
            push = await researcher.exec("cd /home/agent/repo && git push -u origin HEAD 2>&1")
            step5 = f"push exit: {push.exit_code}, output: {(push.stdout or push.stderr or '')[:200]}"

            summary = f"""=== PIPELINE TEST ===
{step1}
{step2}
{step3}
{step4}
{step5}
=== END ==="""
            ctx.done(summary)
            return summary

        except Exception as e:
            error_msg = f"TOOL ERROR: {e}\n{traceback.format_exc()}"
            ctx.done(error_msg)
            return error_msg

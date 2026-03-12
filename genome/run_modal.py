"""Run genome LM training on a Modal GPU.

Single H100 (default):
    modal run run_modal.py

Multi-GPU speedrun (8×H100):
    Change gpu="H100" to gpu="H100:8" and update train.py for DDP.
"""

import modal

app = modal.App("genome-lm-speedrun")

# Shared volume — prepare once, reuse across runs
data_volume = modal.Volume.from_name("genome-lm-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy")
    .add_local_file("prepare.py", "/root/speedrun/prepare.py", copy=True)
    .add_local_file("train.py",   "/root/speedrun/train.py",   copy=True)
)


@app.function(
    image=image,
    gpu="H100",           # change to "H100:8" for full multi-GPU speedrun
    volumes={"/data": data_volume},
    timeout=60 * 20,      # 20 min total (5 min training + overhead)
)
def train():
    import os
    import subprocess

    os.chdir("/root/speedrun")
    cache_dir = os.path.expanduser("~/.cache/genome-lm-speedrun")
    data_dir  = os.path.join(cache_dir, "data")

    # Use cached data from volume if available, otherwise prepare fresh
    if os.path.exists("/data/data") and os.listdir("/data/data"):
        print("Using cached data from volume.")
        os.makedirs(cache_dir, exist_ok=True)
        data_link = os.path.join(cache_dir, "data")
        if not os.path.exists(data_link):
            os.symlink("/data/data", data_link)
    else:
        print("=== Running prepare.py (first time) ===")
        result = subprocess.run(
            [
                "python", "prepare.py",
                "--num-train", "100000",
                "--num-val",   "5000",
            ],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            raise RuntimeError("prepare.py failed")

        # Cache prepared data to volume for future runs
        os.makedirs("/data/data", exist_ok=True)
        subprocess.run(["cp", "-r", data_dir, "/data/"])
        data_volume.commit()
        print("Data cached to volume.")

    print("\n=== Running train.py ===")
    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True, text=True,
        timeout=900,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-3000:])
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed (exit code {result.returncode})")

    return result.stdout


@app.local_entrypoint()
def main():
    output = train.remote()
    print("\n=== Final output ===")
    print(output)

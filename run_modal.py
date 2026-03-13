"""Run train.py on a single Modal GPU."""

import modal

app = modal.App("autoresearch-playground")

# Shared data volume — pre-load once, all research orgs read from it
data_volume = modal.Volume.from_name("autoresearch-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("pyarrow", "requests", "rustbpe", "tiktoken", "numpy")
    .add_local_file("prepare.py", "/root/autoresearch/prepare.py")
    .add_local_file("train.py", "/root/autoresearch/train.py")
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": data_volume},
    timeout=60 * 15,
)
def train():
    import subprocess
    import os

    os.chdir("/root/autoresearch")
    cache_dir = os.path.expanduser("~/.cache/autoresearch")

    # Symlink from volume if data exists, otherwise prepare and cache
    if os.path.exists("/data/data") and os.path.exists("/data/tokenizer"):
        print("Using cached data from volume.")
        os.makedirs(cache_dir, exist_ok=True)
        os.symlink("/data/data", os.path.join(cache_dir, "data"))
        os.symlink("/data/tokenizer", os.path.join(cache_dir, "tokenizer"))
    else:
        print("=== Running prepare.py (first time) ===")
        result = subprocess.run(
            ["python", "prepare.py", "--num-shards", "2"],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            raise RuntimeError("prepare.py failed")
        # Cache to volume
        subprocess.run(["cp", "-r", os.path.join(cache_dir, "data"), "/data/data"])
        subprocess.run(["cp", "-r", os.path.join(cache_dir, "tokenizer"), "/data/tokenizer"])
        data_volume.commit()
        print("Data cached to volume.")

    print("\n=== Running train.py ===")
    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True, text=True,
        timeout=600,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed with return code {result.returncode}")

    return result.stdout


@app.local_entrypoint()
def main():
    output = train.remote()
    print("\n=== Final output ===")
    print(output)


import glob
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/your-org/your-repo.git"
REPO_DIR = "rwkv_bio"


def run(cmd, **kwargs):
    print(f"[download_repo] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def clone_or_pull_repo():
    repo_path = Path(REPO_DIR)
    if (repo_path / ".git").is_dir():
        print(f"[download_repo] {REPO_DIR} already exists, pulling latest...")
        run(["git", "-C", REPO_DIR, "pull"])
    else:
        print(f"[download_repo] cloning {REPO_URL} -> ./{REPO_DIR}")
        run(["git", "clone", REPO_URL, REPO_DIR])


def install_requirements():
    print("[download_repo] installing requirements.txt...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO_DIR)


def has_tpu():
    return len(glob.glob("/dev/accel*")) > 0


def has_gpu():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() != b""


def count_gpus():
    result = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return len(lines)


def torch_xla_installed():
    check = subprocess.run(
        [sys.executable, "-c", "import torch_xla"],
        cwd=REPO_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return check.returncode == 0


def detect_accelerator():
    if has_tpu():
        return "tpu"
    if has_gpu():
        return "gpu"
    return "cpu"


def main():
    extra_args = sys.argv[1:]

    clone_or_pull_repo()
    install_requirements()

    accel = detect_accelerator()
    print(f"[download_repo] detected accelerator: {accel}")

    if accel == "tpu":
        if not torch_xla_installed():
            print("[download_repo] installing torch_xla for TPU...")
            run([
                sys.executable, "-m", "pip", "install", "torch_xla[tpu]",
                "-f", "https://storage.googleapis.com/libtpu-releases/index.html",
            ], cwd=REPO_DIR)
        print("[download_repo] launching train_tpu.py")
        run([sys.executable, "train_tpu.py", *extra_args], cwd=REPO_DIR)

    elif accel == "gpu":
        n_gpus = count_gpus()
        print(f"[download_repo] found {n_gpus} GPU(s), launching train_gpu.py")
        run([
            "torchrun", "--standalone", f"--nproc_per_node={n_gpus}",
            "train_gpu.py", *extra_args,
        ], cwd=REPO_DIR)

    else:
        print(
            "[download_repo] no TPU or GPU detected on this machine.\n"
            "[download_repo] train_tpu.py/train_gpu.py both assume an "
            "accelerator; exiting.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
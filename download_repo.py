
import glob
import os
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



def has_tpu():
    if len(glob.glob("/dev/accel*")) > 0:
        return True
    probe = (
        "try:\n"
        "    import torch_xla.core.xla_model as xm\n"
        "    print(len(xm.get_xla_supported_devices()))\n"
        "except Exception:\n"
        "    pass\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    try:
        return int(result.stdout.strip()) > 0
    except ValueError:
        return False


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


def is_notebook():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


def _ensure_repo_on_path():
    base = os.getcwd()
    for d in (base, os.path.join(base, REPO_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)


def run_in_process(extra_args):
    _ensure_repo_on_path()
    import train_cpu
    return train_cpu.main(extra_args)


def run_tpu_notebook(extra_args):
    _ensure_repo_on_path()
    import train_tpu
    return train_tpu.main_notebook(extra_args)


def main():
    extra_args = sys.argv[1:]

    clone_or_pull_repo()
    accel = detect_accelerator()
    print(f"[download_repo] detected accelerator: {accel}")


    if is_notebook():
        if accel == "tpu":
            print(
                "[download_repo] TPU notebook detected; launching torch_xla "
                "training in-process (8 TPU cores, fork start method)."
            )
            return run_tpu_notebook(extra_args)
        print(
            "[download_repo] interactive notebook detected; running "
            "single-process training in-process (works on CPU and GPU)."
        )
        return run_in_process(extra_args)

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
            "[download_repo] no TPU or GPU detected; running single-process "
            "CPU training in-process."
        )
        return run_in_process(extra_args)


if __name__ == "__main__":
    main()
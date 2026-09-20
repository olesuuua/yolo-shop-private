"""Create both pinned environments and restore the released six-class detector."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def python_version(executable):
    return tuple(json.loads(subprocess.check_output(
        [str(executable), "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
        text=True)))


def prepare_environment(executable, directory, expected):
    if python_version(executable) != expected:
        raise ValueError(f"Use Python {expected[0]}.{expected[1]} for {directory.name}.")
    python = directory / "bin/python"
    if directory.exists():
        if not python.is_file() or python_version(python) != expected:
            raise ValueError(f"Incompatible environment at {directory}; move it aside before retrying.")
    else:
        subprocess.run([str(executable), "-m", "venv", str(directory)], check=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddle-python", default="python3.11", help="Python 3.11 executable for Paddle")
    parser.add_argument("--no-cache", action="store_true", help="Download Python packages without using pip's cache")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 14):
        parser.error("Run setup with Python 3.14: python3.14 setup_lightstore.py")
    supported = {("Darwin", "arm64"), ("Linux", "x86_64")}
    if (platform.system(), platform.machine()) not in supported:
        parser.error("This setup targets Apple Silicon macOS and Linux x86_64 (including WSL2). Native Windows, Intel Macs and Raspberry Pi need a different runtime.")
    if shutil.which("git") is None:
        parser.error("Install Git first.")
    paddle_python = shutil.which(args.paddle_python)
    if paddle_python is None:
        parser.error("Install Python 3.11 or pass --paddle-python /path/to/python3.11.")
    environment = os.environ.copy()
    environment.update({"PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    # Keep the interpreter environments separate: the detector needs NumPy 1.x.
    app_python = prepare_environment(sys.executable, ROOT / ".venv", (3, 14))
    worker_python = prepare_environment(paddle_python, ROOT / ".venv-ppyolo-export", (3, 11))
    pip_options = ["--no-cache-dir"] if args.no_cache else []

    def pip(python, *arguments):
        subprocess.run([str(python), "-m", "pip", *arguments, *pip_options],
                       cwd=ROOT, env=environment, check=True)

    if platform.system() == "Linux":
        # Avoid the default Linux CUDA dependencies; inference uses CPU only.
        pip(app_python, "install", "--index-url", "https://download.pytorch.org/whl/cpu",
            "torch==2.14.0", "torchvision==0.29.0")
    for python, requirements in ((app_python, "requirements.txt"),
                                 (worker_python, "requirements-paddle.txt")):
        pip(python, "install", "--prefer-binary", "-r", str(ROOT / requirements))
        subprocess.run([str(python), "-m", "pip", "check"], env=environment, check=True)
    subprocess.run([str(app_python), str(ROOT / "run_lightstore.py"), "--check"],
                   cwd=ROOT, env=environment, check=True)
    print("\nSetup and real CPU inference passed. Start with: python3 run_lightstore.py", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Setup failed: {error}") from error

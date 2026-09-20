"""Launch the local GPU experiment, loading credentials only in the backend."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
python = ROOT / ".venv/bin/python"
if Path(sys.prefix) != ROOT / ".venv":
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
os.environ.setdefault("LIGHTSTORE_DEVICE", "gpu:0")
from run_lightstore import main
main()

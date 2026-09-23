"""Generate development review artifacts on a clean checkout; never touch holdout."""
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture(scope="session", autouse=True)
def development_review_artifacts():
    root = Path(__file__).resolve().parents[1]
    if not (root/"data/v2/validation/307.csv").exists():
        subprocess.run([sys.executable, str(root/"main.py"), "--develop"], cwd=root, check=True)

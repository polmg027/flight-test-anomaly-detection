"""Freeze records and first-evaluation receipts; no reselection after freeze."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import numpy as np

CORE_FILES = ("main.py", "src/v2/config.py", "src/v2/data.py", "src/v2/features.py",
              "src/v2/models.py", "src/v2/events.py", "src/v2/evaluation.py",
              "src/v2/experiments.py", "src/v2/protocol.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def source_hashes(root):
    # Normalize checkout line endings so Git's Windows conversion is not a code change.
    return {p: hashlib.sha256((root/p).read_text(encoding="utf-8").encode()).hexdigest() for p in CORE_FILES}


def environment():
    return {"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scipy", "scikit-learn")}}


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def fit_fingerprint(detector):
    h = hashlib.sha256()
    for a in (detector.imputer.statistics_, detector.scaler.center_, detector.scaler.scale_, detector.feature_scale):
        h.update(np.asarray(a).tobytes())
    h.update(json.dumps(detector.normalizers, sort_keys=True).encode())
    for tree in detector.models["isolation_forest"].estimators_:
        for a in (tree.tree_.feature, tree.tree_.threshold, tree.tree_.n_node_samples):
            h.update(a.tobytes())
    h.update(detector.models["local_outlier_factor"]._fit_X.tobytes())
    return h.hexdigest()


def verify_freeze(root, manifest, detector):
    if source_hashes(root) != manifest["source_sha256"]:
        raise RuntimeError("Frozen code changed. Use a new version and future holdout, not this benchmark.")
    if environment() != manifest["environment"]:
        raise RuntimeError("Numerical environment differs; install requirements-lock.txt for exact reproduction.")
    if fit_fingerprint(detector) != manifest["fitted_state_sha256"]:
        raise RuntimeError("Reconstructed detector differs from freeze.")

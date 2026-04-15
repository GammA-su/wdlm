from __future__ import annotations

import os
from pathlib import Path


# Keep the default test suite CPU-only and low-thread so it behaves like a fast
# unit suite rather than a benchmark run on developer hardware.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("WDLM_TEST_MODE", "1")


_INTEGRATION_TEST_FILES = {
    "test_cli_contract.py",
    "test_validate_dataset.py",
    "test_eval_suite.py",
    "test_smoke_script.py",
    "test_reproducibility.py",
}


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # type: ignore[override]
    """Skip expensive integration modules unless explicitly enabled."""

    if os.environ.get("WDLM_RUN_INTEGRATION_TESTS") == "1":
        return False
    return collection_path.name in _INTEGRATION_TEST_FILES

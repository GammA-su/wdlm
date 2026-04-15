from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("WDLM_RUN_INTEGRATION_TESTS") != "1",
    reason="Integration smoke test is opt-in. Set WDLM_RUN_INTEGRATION_TESTS=1 to enable.",
)


def test_smoke_deliverable_b_script_exits_successfully() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/smoke_deliverable_b.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Deliverable B smoke passed:" in result.stdout

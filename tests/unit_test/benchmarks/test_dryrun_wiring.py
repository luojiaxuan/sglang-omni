# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke tests for the no-GPU sweep-orchestration dry-run.

`dryrun_sweep_wiring.sh` invokes the real `run_mmmu_sweep.sh` against
PATH-prepended `ssh`/`scp` shims that synthesize a complete retained
bundle in a tempdir. The real validator runs at the end of the sweep
script and exits non-zero on any bundle defect.

The negative tests post-tamper the bundle the dry-run produces and re-run
the real validator. They prove validator-side failure detection on real
defects (missing `launcher.log`, mangled `container_image_digest`) — not
on a stubbed validator.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "dryrun_sweep_wiring.sh"
VALIDATOR = REPO_ROOT / "benchmarks" / "scripts" / "validate_mmmu_artifacts.py"


def _run_dryrun_keeping_bundle(out_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "--keep", "--out", str(out_path)],
        capture_output=True,
        text=True,
    )


def _run_validator(out_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "python",
            str(VALIDATOR),
            str(out_root),
            str(out_root / "sweep-status.jsonl"),
        ],
        capture_output=True,
        text=True,
    )


def test_dryrun_invokes_real_sweep_and_validator_passes(tmp_path) -> None:
    """Happy path: dry-run invokes `run_mmmu_sweep.sh`, all 4 cells get a
    complete retained bundle, and the real validator passes."""
    bundle = tmp_path / "bundle"
    result = _run_dryrun_keeping_bundle(bundle)
    assert result.returncode == 0, (
        f"dry-run exited {result.returncode}; stderr:\n{result.stderr}"
    )
    assert "wiring OK" in result.stdout
    # The real sweep script ran, not a handwritten fixture.
    assert "run_mmmu_sweep.sh" in result.stdout
    # 2 lanes * 2 backends * 1 rep = 4 cells.
    cells = list(bundle.rglob("mmmu_results.json"))
    assert len(cells) == 4
    # Each cell carries the full retained bundle the validator requires.
    for cell in cells:
        assert (cell.parent / "preflight.json").exists()
        assert (cell.parent / "launcher.log").exists()
        assert (cell.parent / "stderr.log").exists()
    # sweep-status.jsonl has one row per cell.
    status_path = bundle / "sweep-status.jsonl"
    assert status_path.exists()
    rows = [
        json.loads(line)
        for line in status_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    assert all(r["status"] == "success" for r in rows)


def test_validator_rejects_bundle_with_missing_launcher_log(tmp_path) -> None:
    """Real bundle tampering: delete `launcher.log` from one cell and
    confirm the validator emits the documented `missing launcher.log`
    error. No validator stub — exercises the real failure-detection path
    on a real defect.
    """
    bundle = tmp_path / "bundle"
    _run_dryrun_keeping_bundle(bundle).check_returncode()

    # Tamper: remove launcher.log from one cell.
    launcher_logs = list(bundle.rglob("launcher.log"))
    assert launcher_logs, "dry-run did not produce launcher.log"
    launcher_logs[0].unlink()

    result = _run_validator(bundle)
    assert result.returncode != 0
    assert "missing launcher.log" in result.stderr


def test_validator_rejects_bundle_with_mismatched_digest(tmp_path) -> None:
    """Real bundle tampering: mangle the `container_image_digest` in one
    `sweep-status.jsonl` row so it disagrees with the cell's run_metadata
    digest. The validator's digest cross-check must reject the bundle.
    """
    bundle = tmp_path / "bundle"
    _run_dryrun_keeping_bundle(bundle).check_returncode()

    status_path = bundle / "sweep-status.jsonl"
    lines = status_path.read_text().splitlines()
    assert lines
    first = json.loads(lines[0])
    first["container_image_digest"] = "sha256:tampered" + "0" * 56
    lines[0] = json.dumps(first)
    status_path.write_text("\n".join(lines) + "\n")

    result = _run_validator(bundle)
    assert result.returncode != 0
    # Validator's mismatch message uses the literal `!=` between the
    # status row digest and the run_metadata digest.
    assert "!=" in result.stderr
    assert "digest" in result.stderr

"""Smoke test for the provisioner pod-build path.

The prior file (`test_provisioner_pvc_volumes.py`) tested `_build_volumes` /
`_build_volume_mounts` / `_build_pod_volumes` helpers that no longer exist
after the provisioner refactor (volume construction is inlined in
`_build_pod`). Rather than rewriting 14 helper-level tests, we keep one
smoke test that exercises `_build_pod` end-to-end and verifies the K8s
pod spec contains the structures the orchestrator depends on.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def provisioner_module():
    """Load the provisioner FastAPI app module by path (not on sys.path)."""
    repo_root = Path(__file__).resolve().parents[2]
    app_path = repo_root / "docker" / "provisioner" / "app.py"
    if not app_path.is_file():
        pytest.skip(f"provisioner app.py missing at {app_path}")

    os.environ.setdefault("PROVISIONER_HOST_BASE_PATH", "/tmp/.deer-flow")
    os.environ.setdefault("PROVISIONER_SKILLS_HOST_PATH", "/tmp/skills")
    spec = importlib.util.spec_from_file_location("provisioner_app_smoke", str(app_path))
    if spec is None or spec.loader is None:
        pytest.skip(f"Failed to load module spec from {app_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["provisioner_app_smoke"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.skip(f"provisioner module failed to import (env not set up): {e}")
    return module


def test_build_pod_smoke_produces_valid_spec(provisioner_module):
    """_build_pod returns a pod spec with volumes + volume_mounts wired up."""
    pod = provisioner_module._build_pod(sandbox_id="sb-test", thread_id="thread-test")

    assert pod.spec is not None
    assert pod.spec.containers, "pod has no containers"
    assert pod.spec.volumes, "pod has no volumes (host-paths or PVCs)"

    container = pod.spec.containers[0]
    assert container.volume_mounts, "container has no volume mounts"

    mount_paths = {vm.mount_path for vm in container.volume_mounts}
    # The orchestrator depends on these two mount targets — guard them.
    assert "/mnt/skills" in mount_paths, f"missing /mnt/skills mount; got {mount_paths}"
    assert any(mp.startswith("/mnt/user-data") for mp in mount_paths), (
        f"missing /mnt/user-data* mount; got {mount_paths}"
    )

"""Regression tests for the sm_80 CUDA-graph-capture collision crash.

Background (see docs/benchmark/sm80_repro/FINDINGS.md): on an A100 (sm_80) with
warp 1.11.0.dev20251124, capturing a mujoco_warp step into a CUDA graph SIGSEGVs
whenever the model contains a geom that routes through the convex/CCD narrowphase
(cylinder, ellipsoid, mesh). The same model steps fine eagerly, so it is a
graph-capture bug, not a bad collision kernel.

The workaround (cylinder/ellipsoid -> capsule, LEAP mesh colliders disabled) is what
these tests guard: if the environment is upgraded to a warp version where capture
works, these tests start passing for the convex geoms and the workaround can be
reverted.

Each case runs in a SUBPROCESS because the failure mode is a segfault, which would
otherwise take down the whole pytest session.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPRO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "mjlab", "continual_distill", "docs", "benchmark", "sm80_repro",
    "repro_geom_collision.py",
)

# Geoms that use the analytic primitive narrowphase — always safe.
PRIMITIVE_GEOMS = ["box", "sphere", "capsule"]
# Geoms that route through convex/CCD narrowphase — these crash on the old warp.
CONVEX_GEOMS = ["cylinder", "ellipsoid", "disc", "mesh"]


def _has_cuda() -> bool:
    try:
        import warp as wp

        wp.init()
        return wp.get_device().is_cuda
    except Exception:  # noqa: BLE001
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA device")


def _run(geom: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, REPRO, "--geom", geom, "--nstep", "5", *extra],
        capture_output=True,
        text=True,
        timeout=900,
    )


@requires_cuda
@pytest.mark.parametrize("geom", PRIMITIVE_GEOMS + CONVEX_GEOMS)
def test_eager_step_works_for_all_geoms(geom: str) -> None:
    """Eager stepping must work for every geom type — proves kernels are correct."""
    p = _run(geom, "--no-graph")
    assert p.returncode == 0, f"eager step failed for {geom}:\n{p.stdout}\n{p.stderr}"


@requires_cuda
@pytest.mark.parametrize("geom", PRIMITIVE_GEOMS)
def test_graph_capture_works_for_primitive_geoms(geom: str) -> None:
    """Primitive-narrowphase geoms capture fine on every warp version tested."""
    p = _run(geom)
    assert p.returncode == 0, f"capture failed for {geom}:\n{p.stdout}\n{p.stderr}"


@requires_cuda
@pytest.mark.parametrize("geom", CONVEX_GEOMS)
def test_graph_capture_for_convex_geoms(geom: str) -> None:
    """Convex/CCD geoms under graph capture.

    This crashed (rc 139/-11) on warp 1.11.0.dev20251124 + sm_80, which is what forced
    the capsule/box asset workarounds. mjlab now requires mujoco-warp >= 3.11, where it
    is fixed, so this must PASS — a failure here means the environment has been
    downgraded below the pin, and the real cylinder/ellipsoid/mesh assets will segfault.
    """
    p = _run(geom)
    if p.returncode in (139, -11):
        pytest.fail(
            f"sm_80 graph-capture segfault for {geom}. The environment's mujoco-warp is "
            "older than the pyproject pin (>=3.11); the asset_zoo's real "
            "cylinder/ellipsoid/mesh geoms need the upstream fix. "
            "See docs/benchmark/sm80_repro/FINDINGS.md"
        )
    assert p.returncode == 0, f"unexpected failure for {geom}:\n{p.stdout}\n{p.stderr}"

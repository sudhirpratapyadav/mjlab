"""Probe: is the crash caused by GPU allocation during CUDA-graph capture?

`mujoco_warp/_src/collision_convex.py::convex_narrowphase` calls wp.empty(...) ~20x
on every invocation. Allocating inside a stream capture is only legal when it is
served by a capture-safe CUDA mempool. This probe tests, in isolation from mujoco:

  1. does a bare wp.empty() inside wp.ScopedCapture() crash on this GPU?
  2. does it stop crashing when the mempool is disabled/enabled?

Run under srun on the target GPU.
"""

import faulthandler
import os
import sys

faulthandler.enable()

import warp as wp


def probe(label: str, fn) -> None:
    print(f"\n=== {label} ===", flush=True)
    try:
        fn()
        print(f"[{label}] OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{label}] EXCEPTION {type(e).__name__}: {e}", flush=True)


def main() -> int:
    wp.init()
    dev = wp.get_device()
    print(f"device={dev} arch={getattr(dev, 'arch', '?')}", flush=True)
    print(f"warp={wp.config.version}", flush=True)
    print(f"is_mempool_supported={wp.is_mempool_supported(dev)}", flush=True)
    print(f"is_mempool_enabled={wp.is_mempool_enabled(dev)}", flush=True)
    try:
        print(
            "is_mempool_access_supported(self)="
            f"{wp.is_mempool_access_supported(dev, dev)}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"mempool_access probe failed: {e}", flush=True)

    @wp.kernel
    def touch(a: wp.array(dtype=float)):
        i = wp.tid()
        a[i] = a[i] + 1.0

    # baseline: capture with NO allocation inside
    def no_alloc():
        pre = wp.zeros(1024, dtype=float)
        with wp.ScopedCapture() as cap:
            wp.launch(touch, dim=1024, inputs=[pre])
        wp.capture_launch(cap.graph)
        wp.synchronize()

    # the suspect: allocate INSIDE the capture, exactly like convex_narrowphase
    def alloc_inside():
        with wp.ScopedCapture() as cap:
            tmp = wp.empty(shape=(4096, 32), dtype=wp.vec3)
            wp.launch(touch, dim=1024, inputs=[wp.zeros(1024, dtype=float)])
            del tmp
        wp.capture_launch(cap.graph)
        wp.synchronize()

    probe("capture, no alloc inside", no_alloc)
    probe("capture, wp.empty INSIDE capture", alloc_inside)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

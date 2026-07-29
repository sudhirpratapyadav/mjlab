"""Probe: how much shared memory / how many registers does the CCD kernel request?

Crash frame is `wp_cuda_launch_kernel(..., hooks.forward_smem_bytes, ...)`. On sm_80 a
kernel may use at most 48KB of *static* shared memory; beyond that it needs an explicit
cudaFuncSetAttribute opt-in (max 163KB/SM on A100). If mujoco_warp's CCD kernel requests
more dynamic smem than the driver grants, the launch fails -- and because warp disables
error verification during graph capture, that failure surfaces as a SIGSEGV instead of
a clean CUDA error.

This prints, per convex geom pair, the kernel's smem request and register count, and
compares against the device limits.
"""

import faulthandler
import os
import sys

faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_geom_collision import build_xml  # noqa: E402


def main() -> int:
    geom = sys.argv[1] if len(sys.argv) > 1 else "cylinder"

    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    wp.init()
    dev = wp.get_device()
    print(f"[smem] device={dev} arch={dev.arch}", flush=True)

    import ctypes

    # device limits via the CUDA driver
    core = wp.context.runtime.core
    for name, attr in [
        ("MAX_SHARED_MEMORY_PER_BLOCK", 8),
        ("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", 97),
        ("MAX_SHARED_MEMORY_PER_MULTIPROCESSOR", 39),
        ("MAX_REGISTERS_PER_BLOCK", 12),
    ]:
        try:
            fn = core.wp_cuda_device_get_attribute
            fn.argtypes = [ctypes.c_int, ctypes.c_int]
            fn.restype = ctypes.c_int
            print(f"[smem] {name} = {fn(dev.ordinal, attr)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[smem] {name}: unavailable ({e})", flush=True)
            break

    mj_model = mujoco.MjModel.from_xml_string(build_xml(geom, "box"))
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    m = mjw.put_model(mj_model)
    d = mjw.put_data(mj_model, mj_data, nworld=8)

    # force module load WITHOUT capture, then inspect the loaded CCD kernel hooks
    mjw.step(m, d)
    wp.synchronize()

    print(f"\n[smem] loaded modules for geom={geom}:", flush=True)
    for mod_name, module in sorted(wp.context.user_modules.items()):
        if "ccd" not in mod_name.lower() and "narrowphase" not in mod_name.lower():
            continue
        exec_mod = module.get_exec_module(dev) if hasattr(module, "get_exec_module") else None
        print(f"  module {mod_name}", flush=True)
        for kname, kernel in getattr(module, "kernels", {}).items():
            try:
                hooks = module.get_kernel_hooks(kernel, dev)
                print(
                    f"    kernel {kname}: forward_smem_bytes="
                    f"{getattr(hooks, 'forward_smem_bytes', '?')} "
                    f"backward_smem_bytes={getattr(hooks, 'backward_smem_bytes', '?')}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"    kernel {kname}: hook probe failed: {e}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

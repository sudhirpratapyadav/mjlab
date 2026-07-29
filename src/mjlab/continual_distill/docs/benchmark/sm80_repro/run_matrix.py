"""Run the sm_80 collision repro across every geom pair, one subprocess each.

Each case runs isolated so a SIGSEGV (139) is recorded, not fatal to the sweep.
Prints a pass/fail matrix and writes results.json next to this file.

Usage:
    python run_matrix.py [--nworld 8] [--nstep 20] [--no-graph] [--only cylinder,mesh]
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(HERE, "repro_geom_collision.py")

FREE_GEOMS = ["box", "sphere", "capsule", "cylinder", "ellipsoid", "disc", "mesh"]
PARTNERS = ["box", "mesh"]


def run_case(free: str, partner: str, args) -> dict:
    cmd = [sys.executable, REPRO, "--geom", free, "--partner", partner,
           "--nworld", str(args.nworld), "--nstep", str(args.nstep)]
    if args.no_graph:
        cmd.append("--no-graph")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    tail = (p.stdout + p.stderr).strip().splitlines()
    return {
        "free": free,
        "partner": partner,
        "returncode": p.returncode,
        "ok": p.returncode == 0,
        "segfault": p.returncode in (139, -11),
        "tail": tail[-25:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=8)
    ap.add_argument("--nstep", type=int, default=20)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated subset of free geoms")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()

    frees = args.only.split(",") if args.only else FREE_GEOMS
    results = []
    for partner in PARTNERS:
        for free in frees:
            r = run_case(free, partner, args)
            status = "SEGFAULT" if r["segfault"] else ("PASS" if r["ok"] else f"FAIL({r['returncode']})")
            print(f"{free:>10} vs {partner:<6} : {status}", flush=True)
            if not r["ok"]:
                for line in r["tail"][-8:]:
                    print(f"             | {line}", flush=True)
            results.append(r)

    with open(args.out, "w") as f:
        json.dump({"graph": not args.no_graph, "nworld": args.nworld,
                   "nstep": args.nstep, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed; "
          f"{sum(r['segfault'] for r in results)} segfaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

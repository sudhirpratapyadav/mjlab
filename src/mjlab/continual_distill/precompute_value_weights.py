#!/usr/bin/env python3
"""Precompute per-sample |ΔV(s)| distillation weights (A3) from an RL teacher
checkpoint's critic, and cache them next to the dataset as value_weights.npy.

The distill run then uses `--distill-weight-mode delta_value`, which loads this
cache (see compute_distill_weights). Kept OFFLINE so the distill runtime needs no
critic / raw .pt dependency.

Usage:
  python -m mjlab.continual_distill.precompute_value_weights \
    --checkpoint logs/rsl_rl/franka_lift_cube/2026-01-09_23-19-02/model_2900.pt \
    --dataset src/mjlab/continual_distill/teacher_datasets/Mjlab_Lift_Cube_Franka_model_2900_20260706_185155
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    args = ap.parse_args()

    dpath = Path(args.dataset)
    data = pickle.load(open(dpath / "data.pkl", "rb"))
    obs = data["observations"].astype(np.float32)
    meta = data.get("metadata", {})
    E = int(meta.get("num_envs", 0))
    N = obs.shape[0]
    assert E > 0 and N % E == 0, f"need num_envs in metadata; got E={E}, N={N}"
    S = N // E
    obs_dim = obs.shape[1]

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    critic = _mlp([obs_dim] + list(args.hidden) + [1])
    msd = {}
    lin = [k for k in critic.state_dict() if "weight" in k]
    for j, lk in enumerate(sorted(lin, key=lambda x: int(x.split(".")[0]))):
        msd[lk] = sd[f"critic.{j*2}.weight"]
        msd[lk.replace("weight", "bias")] = sd[f"critic.{j*2}.bias"]
    critic.load_state_dict(msd)
    critic.eval()

    onm = sd["critic_obs_normalizer._mean"].numpy()
    ostd = sd["critic_obs_normalizer._std"].numpy()
    with torch.no_grad():
        V = critic(torch.tensor((obs - onm) / (ostd + 1e-8))).squeeze(-1).numpy()

    # |ΔV| per (step, env), step-major layout
    Ve = V.reshape(S, E)
    dV = np.zeros((S, E), dtype=np.float64)
    dV[1:] = np.abs(Ve[1:] - Ve[:-1])
    dV[0] = dV[1]
    w_raw = dV.reshape(-1)  # raw |ΔV| per sample; floor/clip applied at load time

    out = dpath / "value_weights.npy"
    np.save(out, w_raw.astype(np.float32))
    prof = dV.mean(1)
    gr = float(prof[:47].mean()); ho = float(prof[60:150].mean()) if S >= 150 else float(prof[60:].mean())
    print(f"saved {out}  (N={N}, S={S}, E={E})")
    print(f"|ΔV| grasp(t<47)={gr:.4f}  hold(t60+)={ho:.4f}  ratio={gr/(ho+1e-9):.1f}x")


if __name__ == "__main__":
    main()

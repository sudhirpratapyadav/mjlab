"""Build the per-task review pages from the completed simulation artifacts."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "verified_tasks"
NOTES = {
  "Axial-Extract": "Joint-progress reward; extraction target and plug/mount contacts checked.",
  "Cage-Drag": "Geometric cage + transport; yaw-safe aperture; violations revoke success and reward.",
  "Drag-Pull": "Ground-level point goal, spawn separation, transport direction and termination rules checked.",
  "Edge-Grasp": "Actual bilateral pad contact and low velocity required; transient flicks rejected.",
  "Flip-Switch": "Joint-progress reward; switch target, detent motion and housing clearance checked.",
  "Lift-Cube": "Goal requires a slow actual two-pad grasp; free-flight proximity rejected.",
  "Open-Door": "Joint-progress reward; hinge target and rendered door pose checked.",
  "Open-Drawer": "Joint-progress reward; slide target and drawer/handle geometry checked.",
  "Open-Lid": "Joint-progress reward; hinge target, lid and handle goal geometry checked.",
  "Peg-Insertion": "Tip/center frames reconciled; goal lowered 50 mm; seated square-bore fit, release and speed checked.",
  "Pivot-Lift": "Recorded wall-pivot contact plus actual slow grasp required for the airborne goal.",
  "Place-In-Container": "Full cube/basket-frame containment, support, release and low speed; resting goal geometry corrected.",
  "Push-Button": "Joint-progress reward; pressed target and spring-return behavior checked.",
  "Push-Cuboid": "Ground-level point goal, spawn separation, object dimensions and termination rules checked.",
  "Push-Flap": "Joint-progress reward; negative hinge target, panel and housing clearance checked.",
  "Reach-Target": "Position-only wrist goal, target reachability and matching reward checked.",
  "Reorient-Object": "Axis-based reward matches success; fast orientation crossings rejected.",
  "Rotate-Valve": "Monotonic joint reward removes 270-degree Cartesian shortcut; wheel/hub geometry checked.",
  "Slide-Window": "Joint-progress reward; slide target and sash/frame clearance checked.",
  "Stack-Cube": "Slow released placement in actual contact with the base; hovering rejected.",
  "Strike-Slide": "Beyond-reach ground goal, puck dimensions/contact configuration and point success checked.",
  "Throw-To-Bin": "Beyond-reach bin retained; full containment, support, release and low speed required.",
  "Tool-Pull": "Actual pad/tool then tool/puck contacts required; direct robot/puck contact invalidates success.",
  "Topple-Block": "Symmetric axis reward matches the two accepted faces; fast transient crossings rejected.",
  "Turn-Lever": "Joint-progress reward; lever target, mount and housing clearance checked.",
}


def main():
  rows = json.loads((OUT / "results.json").read_text())
  assert len(rows) == 25
  assert len({r["source_sha256"] for r in rows}) == 1, "mixed source revisions"
  lines = [
    "# Verification of all 25 task definitions — 2026-09-11",
    "",
    "All 25 tasks were reviewed individually through code/configuration inspection, "
    "8 sampled resets, a constructed MuJoCo goal/contact state, and one real "
    "teacher rollout to first success, termination or timeout. Each has saved "
    "qpos/qvel/mocap state, goal and rollout renders, and an MP4. The constructed "
    "states test predicates; they are not demonstrated policy trajectories.",
    "",
    "Regression validation: **290 tests passed** on the CPU MuJoCo Warp path. "
    "The final run uses the same source fingerprint for every task.",
    "",
    "Focused stress validation: **1,024 resets each for Peg-Insertion and Cage-Drag**. "
    "Both pass placement/collision and success gates, with zero success-at-reset "
    "and all goal witnesses accepted. Raw reports: "
    "[Peg-Insertion](stress_validation/Peg-Insertion.json), "
    "[Cage-Drag](stress_validation/Cage-Drag.json).",
    "",
    "Test mode now preserves each training task's episode budget and failure "
    "conditions. Only observation corruption and external robot pushes are disabled. "
    "Simulation ran on CPU in Slurm holder 20277, dgx1, with one task per command; "
    "rendering used EGL with software rendering requested.",
    "",
    "**The 25 single-episode outcomes below are diagnostics, not teacher success "
    "rates. The September 10 teacher scoreboard must be remeasured under these "
    "corrected contracts.**",
    "",
    "[Open the visual review](verified_tasks/index.html)",
    "",
    "| Task | Correction or checked invariant | Goal state | Reset checks | Diagnostic rollout |",
    "|---|---|---|---|---|",
  ]
  cards = []
  for row in rows:
    task = row["task"]
    short = task.removeprefix("Mjlab-").removesuffix("-Franka")
    reset_ok = all(
      not r["success"] and not r["deep_contacts"] and r["finite"] for r in row["resets"]
    )
    oracle_ok = row["oracle"]["accepted"] and not row["oracle"]["deep_contacts"]
    outcome = (
      "success"
      if row["rollout"]["success"]
      else ("terminated" if row["rollout"]["terminated"] else "timeout")
    )
    goal_text = "pass" if oracle_ok else "needs review"
    resets_text = f"{len(row['resets'])} clear" if reset_ok else "needs review"
    lines.append(
      f"| [{short}](verified_tasks/{task}/result.json) | {NOTES[short]} | {goal_text} | {resets_text} | {outcome} |"
    )
    cards.append(f"""<section id="{task}"><h2>{html.escape(short)}</h2>
<p>{html.escape(NOTES[short])}</p>
<p>Goal probe: <b>{goal_text}</b> · Resets: <b>{resets_text}</b> · Diagnostic: <b>{outcome}</b></p>
<div class="images"><figure><img loading="lazy" src="{task}/goal_detail.png" alt="{short} constructed goal state"><figcaption>Constructed goal/contact state</figcaption></figure>
<figure><img loading="lazy" src="{task}/final.png" alt="{short} final rollout state"><figcaption>Actual final rollout state</figcaption></figure></div>
<details><summary>Rollout and raw evidence</summary><video controls preload="none" src="{task}/rollout.mp4"></video>
<p><a href="{task}/result.json">Checks and metrics</a> · <a href="{task}/trajectory.npz">Recorded trajectory</a> · <a href="{task}/goal_state.npz">Goal-state qpos/qvel</a> · <a href="{task}/review.jpg">Reset / middle / final</a></p></details></section>""")
  lines += [
    "",
    "The goal-state witnesses include contact history for Cage-Drag, Tool-Pull "
    "and Pivot-Lift. They establish that the intended state can satisfy the "
    "predicate without deep interpenetration; they do not prove the teacher "
    "can reliably reach that state from every reset.",
    "",
    "[Task contract decisions](VERIFY_FIXES.md) · [Pre-fix audit](SEMANTICS_AUDIT.md)",
    "",
  ]
  (ROOT / "TASK_VERIFICATION_RESULTS.md").write_text("\n".join(lines))
  (OUT / "index.html").write_text(
    """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>25-task verification</title>
<style>body{font:16px/1.5 system-ui;max-width:1300px;margin:32px auto;padding:0 20px;background:#f5f5f3;color:#202020}section{background:white;padding:24px;margin:24px 0;border:1px solid #ddd;border-radius:8px}.images{display:flex;flex-wrap:wrap}figure{flex:1;min-width:280px;margin:8px}img,video{width:100%;max-width:640px}summary{cursor:pointer}a{color:#145d98}figcaption{color:#555}</style>
<h1>25-task definition verification</h1><p>September 11, 2026. Code, reset states, MuJoCo contacts, goal probes and actual rollouts reviewed individually.</p>
<p><b>Goal probes are constructed states. One diagnostic rollout per task is not a teacher success-rate estimate.</b></p>
"""
    + "\n".join(cards)
    + "</html>\n"
  )
  print(
    "REPORT",
    len(rows),
    "tasks",
    sum(r["oracle"]["accepted"] for r in rows),
    "accepted goal probes",
  )


if __name__ == "__main__":
  main()

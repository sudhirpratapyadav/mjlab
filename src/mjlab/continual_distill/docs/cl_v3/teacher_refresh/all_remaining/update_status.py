"""Refresh completed measurements through the official scoreboard writer."""
import json,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
updater=here.parents[1]/'update_status.py'
for folder in sorted((here/'baseline32').iterdir()):
 p=folder/'result.json'
 if not p.exists():continue
 name=folder.name[6:-7]
 baseline128=here/'baseline128'/folder.name/'result.json'
 r=json.loads((baseline128 if baseline128.exists() else p).read_text())
 best=r
 video=here/'videos'/folder.name/'result.json'
 published=here/'videos'/folder.name/'publication_verification.json'
 strategy=None
 candidates={
  'Tool-Pull': ('retained_check32','Decode shaft heading from its true axis, including when the round shaft rolls. No success-rate improvement; broader candidates rejected.'),
  'Peg-Insertion': ('peg_brake128','Brake pickup descent using FK height and guard actual lowest hand clearance; retain insertion geometry and transport.'),
  'Open-Lid': ('estimate128','Filter the stationary knob position in the robot-base frame before grasping; retain the hinge lift controller.'),
  'Push-Button': ('vertical128','Vertical press after alignment; avoid wedging the pad against the cap side.'),
  'Lift-Cube': ('brake128','Use FK grasp height, brake final descent, and guard lowest hand clearance.'),
  'Stack-Cube': ('recovery128','Safe pickup, raised transport, hand-frame grasp estimate, slow placement, and fixed joint command while opening.'),
  'Topple-Block': ('anchor128','Store the punch waypoint in the robot-base frame; subtract current FK instead of repeatedly commanding the same displacement.'),
  'Open-Drawer': ('gravity128','Add model-based gravity torque divided by actuator stiffness to position commands.'),
  'Cage-Drag': ('centered128','Center and align above the cube, lower, and transport with open fingers, fixed heading and gravity support.'),
  'Drag-Pull': ('videos','Analytical gravity support replaces vertical bias integration and feed-forward.'),
 }
 if name in candidates:
  directory,strategy=candidates[name]
  candidate=here/directory/folder.name/'result.json'
  if candidate.exists():best=json.loads(candidate.read_text())
 if name=='Place-In-Container':
  confirmation=here/'confirmation128'/folder.name/'result.json'
  if confirmation.exists():best=json.loads(confirmation.read_text())
 args=[sys.executable,str(updater),'--task',name,'--owner','teacher-refresh','--gate','B='+('ok' if r['n']==128 else '~'),'--gate','T='+('ok' if best['n']==128 and best['sr']>=.90 else '~'),'--gate','V='+('ok' if published.exists() else '~'),'--set',f"Baseline SR (n, HEAD)={r['sr']:.4f} ({r['n']}, {r['source_sha256'][:12]})",'--set',f"Current SR (n, HEAD)={best['sr']:.4f} ({best['n']}, {best['source_sha256'][:12]})",'--notes',f"September 11 corrected definitions, strict first episodes, seed {best['stats_seed']}. {best['num_success']}/{best['n']}; "+('diagnostic iteration batch only; n=128 confirmation pending. ' if best['n']==32 else '')+'See teacher_refresh/all_remaining/WORK_LOG.md and BASELINES.md; experiment sources/traces archived.']
 confirmation_dir={'Push-Button':'confirmation_small_gains128','Topple-Block':'confirmation_small_gains128','Peg-Insertion':'confirmation_peg_brake128','Stack-Cube':'confirmation_recovery128','Open-Lid':'confirmation_estimate128','Place-In-Container':'confirmation128'}.get(name)
 if confirmation_dir:
  confirmation=here/confirmation_dir/folder.name/'result.json'
  if confirmation.exists():
   c=json.loads(confirmation.read_text())
   notes_index=args.index('--notes')+1
   args[notes_index]+=f" Independent seed {c['stats_seed']}: {c['num_success']}/{c['n']} ({c['sr']:.1%})."
 if strategy:args+=['--set','Strategy='+strategy,'--gate','A=ok','--gate','S=ok']
 subprocess.run(args,check=True,capture_output=True)

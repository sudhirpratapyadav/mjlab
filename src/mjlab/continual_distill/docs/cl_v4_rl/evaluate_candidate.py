"""Run the frozen two-batch gate and render its actual validation trajectory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--task',required=True)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--label',required=True)
  parser.add_argument('--diagnostics',action='store_true')
  parser.add_argument('--confirmation-seed',type=int,default=20260915,help='Preregister before training/evaluation; use a fresh seed after a failed confirmation')
  args = parser.parse_args()
  if args.confirmation_seed<0 or args.confirmation_seed==20260914:
    parser.error('Confirmation seed must be nonnegative and distinct from validation20260914')
  if Path(args.label).name != args.label:
    parser.error('label must be a single path component')
  if not args.checkpoint.is_file():
    parser.error('checkpoint does not exist')
  trace = HERE/'runs/verified'/args.label
  results = []
  for kind,seed in [('val',20260914),('confirm',args.confirmation_seed)]:
    output = HERE/'evidence'/f'{args.label}-{kind}-{seed}.json'
    cmd = [sys.executable,str(HERE/'evaluate_teacher.py'),'--task',args.task,'--checkpoint',str(args.checkpoint.resolve()),'--seed',str(seed),'--episodes','128','--output',str(output)]
    if kind == 'val':
      cmd.extend(['--trace-dir',str(trace)])
    if args.diagnostics:
      cmd.append('--diagnostics')
    subprocess.run(cmd,check=True)
    result = json.loads(output.read_text())
    results.append(result)
    print(kind,result['successes'],result['episodes'],flush=True)
    if result['success_rate'] <= .90:
      break
  subprocess.run([sys.executable,str(HERE/'render_evaluation.py'),str(HERE/'evidence'/f'{args.label}-val-20260914.json')],check=True)
  print('RATE_GATE',len(results)==2 and all(r['success_rate']>.90 for r in results),'VIDEO_REVIEW_REQUIRED',flush=True)

if __name__ == '__main__':
  main()

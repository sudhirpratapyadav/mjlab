import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
x=json.loads((a.baseline/'result.json').read_text());y=json.loads((a.candidate/'result.json').read_text())
for k in ['task_id','n','stats_seed','stats_batch_size','device','evaluation_protocol','task_revision']:assert x[k]==y[k],k
b=np.load(a.baseline/'trace.npz');c=np.load(a.candidate/'trace.npz')
for k in ['qpos','qvel','mocap_pos','mocap_quat','obs']:np.testing.assert_array_equal(b[k][0],c[k][0],err_msg=k)
print(json.dumps(dict(task=x['task_id'],before=x['num_success'],after=y['num_success'],initial_states_and_observations_identical=True)))

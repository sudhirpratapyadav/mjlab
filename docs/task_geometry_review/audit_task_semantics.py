"""Read-only task semantics probes; run on CPU inside a cluster allocation."""
import json
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.mdp.rewards import object_at_goal_reward


def main():
    results = {}
    source = Path('src/mjlab/continual_distill/docs/cl_v3/physics_refresh/results.json')
    inventory = []
    for row in json.loads(source.read_text()):
        cfg = load_env_cfg(row['task_id'], test=True)
        offsets = {}
        for name, entity in cfg.scene.entities.items():
            if name in ('robot', 'mocap_goal'):
                continue
            model = entity.spec_fn().compile()
            try:
                offsets[name] = model.site('object_site').pos.tolist()
            except KeyError:
                pass
        inventory.append(dict(task=row['task_id'], commands={k:type(v).__name__ for k,v in cfg.commands.items()}, site_offsets=offsets,
                              rewards={k:dict(func=v.func.__name__,weight=v.weight) for k,v in cfg.rewards.items()}))
    results['inventory'] = inventory
    for task, asset in [('Peg-Insertion', 'object'), ('Cage-Drag', 'cube'),
                        ('Reorient-Object', 'cylinder'), ('Topple-Block', 'block')]:
        cfg = load_env_cfg(f'Mjlab-{task}-Franka', test=True)
        cfg.scene.num_envs = 2
        env = ManagerBasedRlEnv(cfg, device='cpu')
        try:
            env.reset()
            env.sim.forward()
            cmd = next(iter(env.command_manager._terms.values()))
            obj = env.scene[asset]
            out = {}
            def place(pos):
                quat = torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
                obj.write_root_link_pose_to_sim(torch.cat((pos,quat),dim=-1))
                obj.write_root_link_velocity_to_sim(torch.zeros(2,6))
                env.sim.forward()
                cmd._update_metrics()
                cmd._update_command()
                env.sim.forward()
            def state():
                return dict(success=cmd.compute_success().tolist(), latched=cmd.episode_success.tolist(),
                            goal=(cmd.target_pos-env.scene.env_origins).tolist(),
                            root=(obj.data.root_link_pos_w-env.scene.env_origins).tolist(),
                            site=(obj.data.site_pos_w[:,obj.site_names.index('object_site')]-env.scene.env_origins).tolist(),
                            precise_reward=object_at_goal_reward(env,next(iter(env.command_manager._terms)),asset,max_dist=.35).tolist())
            if task == 'Peg-Insertion':
                place(cmd.target_pos.clone())
                out['seated'] = state()
                marker = env.scene['mocap_goal'].cfg.spec_fn().compile()
                out['marker_geom_local_positions'] = marker.geom_pos.tolist()
                out['source_geom_local_positions'] = obj.cfg.spec_fn().compile().geom_pos.tolist()
                raised = cmd.target_pos.clone(); raised[:,2] += .05
                place(raised)
                out['tip_at_goal_but_50mm_above_seated'] = state()
                partial = cmd.target_pos.clone(); partial[:,2] += .014
                place(partial)
                obj.write_root_link_velocity_to_sim(torch.tensor([[0.,0.,1.,0.,0.,0.]]).repeat(2,1))
                env.sim.forward(); cmd._update_metrics()
                out['14mm_unseated_moving_up_1mps'] = state()
            elif task == 'Cage-Drag':
                place(cmd.target_pos.clone())
                out['object_at_goal_without_gripper_caging'] = state()
                out['gripper_distance'] = cmd.metrics['gripper_object_distance'].tolist()
                q = env.scene['robot'].data.joint_pos.clone()
                q[:,list(cmd.finger_idx)] = 0.
                env.scene['robot'].write_joint_position_to_sim(q)
                env.sim.forward(); cmd._update_metrics()
                out['closed_after_success'] = state()
                q[:,list(cmd.finger_idx)] = .04
                env.scene['robot'].write_joint_position_to_sim(q)
                env.sim.forward(); cmd._update_metrics()
                out['reopened_after_violation'] = state()
            else:
                pos = cmd.target_pos.clone()
                pos[:, 2] = .2  # compare orientations at the same position
                place(pos)
                out['upright'] = state()
                quat = torch.tensor([[2**-.5,0.,2**-.5,0.]]).repeat(2,1)
                obj.write_root_link_pose_to_sim(torch.cat((pos,quat),dim=-1))
                env.sim.forward(); cmd._update_metrics()
                out['rotated_90_degrees'] = state()
            results[task] = out
        finally:
            env.close()
    Path('docs/task_geometry_review/semantics_audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k!='inventory'},indent=2))


if __name__ == '__main__':
    main()

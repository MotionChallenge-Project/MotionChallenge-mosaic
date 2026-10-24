"""Replay pinned RobotBridge reset/step methods on the same SDK states."""
import argparse
import ast
import collections
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from adapter import MOSAIC
from motion_challenge.physics import World
from motion_challenge.sdk import ReferenceFrame
from motion_challenge.validation import external_output_path


def load_author(source):
    namespace = {'np': np, 'sRot': R,
                 'logger': NS(debug=lambda *a, **k: None, info=lambda *a, **k: None)}
    selections = {
        'deploy/utils/dof.py': {'DoFAdapter': None},
        'deploy/utils/transformation.py': {'matrix_from_quat': None, 'subtract_frame_transforms': None,
                                          'quat_rotate_inverse': None},
        'deploy/utils/dataset.py': {'TransformAligner': None, 'MotionDataset': {
            'command', 'joint_pos', 'anchor_pos_w', 'anchor_quat_w', 'robot_anchor_pos_w', 'robot_anchor_quat_w',
            'body_pos_w_aligned', 'reset', 'post_step_callback', 'get_data'}},
        'deploy/envs/base_env.py': {'BaseEnv': {
            'reset', '_reset_envs', 'step', '_pre_physics_step', '_physics_step',
            '_post_physics_step', '_update_obs'}},
        'deploy/envs/mosaic.py': {'MosaicEnv': {
            'reset', 'step', '_get_command', 'compute_observation', '_post_physics_step',
            '_policy_vector_to_sim', '_sim_vector_to_policy'}},
    }
    for path, names in selections.items():
        nodes = []
        for node in ast.parse((source/path).read_text()).body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)) or node.name not in names:
                continue
            selected = names[node.name]
            if selected is not None:
                node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in selected]
                assert {n.name for n in node.body} == selected
            nodes.append(node)
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+nodes, []),
                     str(source/path), 'exec'), namespace)
    return namespace


def transform(packet, rotation):
    def body(b):
        return replace(b, position_world_m=tuple(rotation.apply(b.position_world_m)),
                       orientation_xyzw=tuple((rotation*R.from_quat(b.orientation_xyzw)).as_quat()),
                       linear_velocity_world_m_s=tuple(rotation.apply(b.linear_velocity_world_m_s)),
                       angular_velocity_world_rad_s=tuple(rotation.apply(b.angular_velocity_world_rad_s)))
    return replace(packet, root_position_world_m=tuple(rotation.apply(packet.root_position_world_m)),
        root_orientation_xyzw=tuple((rotation*R.from_quat(packet.root_orientation_xyzw)).as_quat()),
        root_linear_velocity_world_m_s=tuple(rotation.apply(packet.root_linear_velocity_world_m_s)),
        root_angular_velocity_world_rad_s=tuple(rotation.apply(packet.root_angular_velocity_world_rad_s)),
        bodies=tuple(body(b) for b in packet.bodies))


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'weights', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root/'native-sources.json').read_text())
    for path, digest in pins['files'].items():
        assert hashlib.sha256((args.source/path).read_bytes()).hexdigest() == digest
    native = load_author(args.source)
    runtime = json.loads((root/'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    assert runtime['reference_offsets_s'] == [-.02]
    policy = MOSAIC(root, runtime)
    rotated_policy = MOSAIC(root, runtime)
    metadata = policy.session.get_modelmeta().custom_metadata_map
    for values, key in ((policy.default, 'default_joint_pos'), (policy.scale, 'action_scale'),
                        (policy.kp, 'joint_stiffness'), (policy.kd, 'joint_damping')):
        np.testing.assert_array_equal(values, np.asarray(metadata[key].split(','), dtype=np.float32))
    world = World(root/'MotionChallenge-Core/robots/g1-29dof-cpu-v0.json')
    order = [world.joint_names.index(n) for n in policy.names]
    inverse = [policy.names.index(n) for n in world.joint_names]
    rng = np.random.default_rng(241024)
    permutation = rng.permutation(29)
    shuffled_names = tuple(np.asarray(world.joint_names)[permutation])
    canonical = [shuffled_names.index(n) for n in world.joint_names]
    errors = dict(observation=0., history=0., action=0., motor_target=0., global_yaw_observation=0.)
    captured = {}
    original_infer = policy.infer
    def capture(inputs):
        captured['obs'] = inputs['obs'].copy()
        return original_infer(inputs)
    policy.infer = capture
    def compare(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a)-b))))
    def shuffled(packet):
        kw = dict(joint_position_rad=tuple(np.asarray(packet.joint_position_rad)[permutation]),
                  joint_velocity_rad_s=tuple(np.asarray(packet.joint_velocity_rad_s)[permutation]))
        if hasattr(packet, 'joint_names'):
            kw['joint_names'] = shuffled_names
        return replace(packet, **kw)
    seen_indices = []
    for reset in range(4):
        refs, states = [], []
        for step in range(40):
            rotation = R.from_euler('xyz', rng.uniform(-1., 1., 3))
            world.data.qpos[:] = np.r_[rng.normal(0., .3, 2), .8,
                rotation.as_quat()[[3, 0, 1, 2]], rng.normal(0., .2, 29)]
            world.data.qvel[:] = rng.normal(0., .4, 35)
            world.data.time = step*.02
            mujoco.mj_forward(world.model, world.data)
            state = world.state()
            refs.append(ReferenceFrame(step*.02, step*.02, state.joint_position_rad,
                state.root_position_world_m, state.root_orientation_xyzw, state.joint_velocity_rad_s,
                state.root_linear_velocity_world_m_s, state.root_angular_velocity_world_rad_s, state.bodies))
            world.data.qpos[7:] += rng.normal(0., .1, 29)
            world.data.qvel[:] = rng.normal(0., .6, 35)
            mujoco.mj_forward(world.model, world.data)
            states.append(world.state())
        refs = refs[:34]
        states.append(states[-1])  # Allows the final native post-step observation.
        body_names = [b.name for b in refs[0].bodies]
        anchor_index = body_names.index('torso_link')
        initial = R.from_quat(refs[0].bodies[anchor_index].orientation_xyzw)
        heading_inverse = R.from_euler('z', initial.as_euler('xyz')[2]).inv()
        class Simulator:
            index = 0
            def update_obs(self):
                state = transform(states[self.index], heading_inverse)
                anchor = next(b for b in state.bodies if b.name == 'torso_link')
                self.torso_trans = np.asarray(anchor.position_world_m)
                self.torso_quat = np.asarray(anchor.orientation_xyzw)
                return dict(base_ang_vel=R.from_quat(state.root_orientation_xyzw).inv().apply(state.root_angular_velocity_world_rad_s),
                    base_lin_vel=np.zeros(3), dof_pos=np.asarray(state.joint_position_rad),
                    dof_vel=np.asarray(state.joint_velocity_rad_s))
            def calibrate(self, *a):
                self.index = 0
            def apply_action(self, target):
                self.target = target.copy()
                self.index += 1
        sim = Simulator()
        loader = native['MotionDataset'].__new__(native['MotionDataset'])
        loader.simulator = sim
        loader.override_robot_anchor_pos = False
        loader.motion_anchor_body_index = anchor_index
        loader.command_horizon = 1
        loader.command_velocity = True
        loader.motion_init_align = native['TransformAligner'](yaw_only=True, xy_only=True)
        loader._update_metrics = lambda: None  # Author metrics do not enter inference.
        loader.motion = NS(time_step_total=len(refs),
            joint_pos=np.asarray([r.joint_position_rad for r in refs], np.float32)[:, order],
            joint_vel=np.asarray([r.joint_velocity_rad_s for r in refs], np.float32)[:, order],
            body_pos_w=np.asarray([[b.position_world_m for b in r.bodies] for r in refs]),
            body_quat_w=np.asarray([[b.orientation_xyzw for b in r.bodies] for r in refs])[..., [3, 0, 1, 2]])
        env = native['MosaicEnv'].__new__(native['MosaicEnv'])
        env.simulator, env.motion_loader = sim, loader
        env.cfg = NS(control=NS(use_teleop=False, action_clip_value=10000.))
        env.policy_cfg = {}
        env.policy_model_meta = True
        env._policy_dim = 29
        env.action_beta = 1.
        env.policy_action_scales = policy.scale
        env.policy_default_joint_pos = policy.default
        env.policy_default_joint_pos_sim = policy.default[inverse]
        env._sim_to_policy = np.asarray(inverse)
        env._policy_to_sim_adapter = native['DoFAdapter'](policy.names, world.joint_names)
        env._sim_to_policy_adapter = native['DoFAdapter'](world.joint_names, policy.names)
        env.history_handler = NS(reset=lambda: None)
        env.init_ref_dof_pos = None
        env.first_obs_received = False
        env.episode_length_buf = np.zeros(1)
        env.action = np.zeros((1, 29))
        env.base_ang_vel = env.base_lin_vel = env.dof_pos = env.dof_vel = None
        env.eval_mode = True
        env.use_estimator = False
        env._check_termination = lambda: None
        env.gravity_w = np.asarray([0., 0., -1.])
        env.time_step, env.playback_speed, env.max_timestep = 0, 1., -1
        names = ('obs_command_buffer', 'obs_motion_anchor_ori_b_buffer', 'obs_base_ang_vel_buffer',
                 'obs_joint_pos_rel_buffer', 'obs_joint_vel_rel_buffer', 'obs_prev_policy_action_buffer')
        for name, size in zip(names, (58, 6, 3, 29, 29, 29), strict=True):
            setattr(env, name, collections.deque([np.zeros(size) for _ in range(5)], maxlen=5))
        original_obs = env.reset()
        extra_yaw = R.from_euler('z', 1.37)
        policy.reset(shuffled(states[0]), (shuffled(refs[0]),), reset)
        rotated_policy.reset(shuffled(transform(states[0], extra_yaw)),
                             (shuffled(transform(refs[0], extra_yaw)),), reset)
        for step in range(40):
            ref_index = min(max(0, step-1), len(refs)-1)
            seen_indices.append(ref_index)
            ref = replace(refs[ref_index], requested_time_s=(step-1)*.02)
            inputs = {k: v.astype(np.float32) for k, v in original_obs.items()}
            native_action = policy.session.run(None, inputs)[0].reshape(29)
            control = policy.act(shuffled(states[step]), (shuffled(ref),))
            compare('observation', captured['obs'], inputs['obs'])
            for buffer, name in zip(policy.history, names, strict=True):
                compare('history', np.asarray(buffer), np.asarray(getattr(env, name)))
            compare('action', policy.last, native_action)
            # Keep previous actions identical when testing coordinate invariance.
            if step:
                rotated_policy.last = previous_action.copy()
            rotated_obs = rotated_policy.observations(shuffled(transform(states[step], extra_yaw)),
                                                      (shuffled(transform(ref, extra_yaw)),))
            compare('global_yaw_observation', rotated_obs['obs'], inputs['obs'])
            original_obs = env.step(native_action)
            compare('motor_target', np.asarray(control.joint_position_target_rad)[canonical], sim.target[0])
            previous_action = native_action
    policy.close()
    rotated_policy.close()
    report = dict(status='passed' if max(errors.values()) < 2e-5 else 'failed', source=pins,
        cases=160, resets=4, endpoint_cases=24, shuffled_joint_order=True,
        native_reset_and_step_methods=True, reference_indices_first_clip=seen_indices[:40],
        common_world_yaw_invariance=True, max_abs_errors=errors, tolerance=2e-5)
    output = external_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    assert report['status'] == 'passed'


if __name__ == '__main__':
    main()

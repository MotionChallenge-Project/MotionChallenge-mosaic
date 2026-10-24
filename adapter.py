"""MOSAIC GMT: original ONNX, term-major history and native reference timing."""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_challenge.adapters.onnx import ONNXController
from motion_challenge.sdk import ControlCommand


class MOSAIC(ONNXController):
    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        # RobotBridge parses the checkpoint metadata as float32.
        self.default = self.default.astype(np.float32)
        self.scale = self.scale.astype(np.float32)
        self.kp = self.kp.astype(np.float32)
        self.kd = self.kd.astype(np.float32)

    def frame(self, state, reference):
        q, qd, omega, _, _ = self.proprio(state)
        ref = reference[0]
        actual = next(b for b in state.bodies if b.name == self.cfg['anchor_body_name'])
        target = next(b for b in ref.bodies if b.name == self.cfg['anchor_body_name'])
        # Express both packets in the author's initial-reference heading frame.
        # This is a common coordinate change; it cannot change tracking errors.
        actual_q = (self.heading_inverse * R.from_quat(actual.orientation_xyzw)).as_quat().astype(np.float32)
        target_q = (self.heading_inverse * R.from_quat(target.orientation_xyzw)).as_quat().astype(np.float32)
        relative = (R.from_quat(actual_q).inv() * R.from_quat(target_q)).as_quat().astype(np.float32)
        orientation = R.from_quat(relative).as_matrix()[:, :2].flatten()
        command = np.concatenate([
            np.asarray(ref.joint_position_rad)[self.order],
            np.asarray(ref.joint_velocity_rad_s)[self.order],
        ]).astype(np.float32)
        return (command, orientation, omega, q.astype(np.float32)-self.default,
                qd.astype(np.float32), self.last.copy())

    def append(self, state, reference):
        for buffer, term in zip(self.history, self.frame(state, reference), strict=True):
            buffer.append(term)

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        anchor = next(b for b in reference[0].bodies if b.name == self.cfg['anchor_body_name'])
        initial_yaw = R.from_quat(anchor.orientation_xyzw).as_euler('xyz')[2]
        self.heading_inverse = R.from_euler('z', initial_yaw).inv()
        self.history = [deque([np.zeros(n) for _ in range(5)], maxlen=5)
                        for n in (58, 6, 3, 29, 29, 29)]
        # BaseEnv.reset and MosaicEnv.reset each compute one observation.
        # Fresh buffers per clip reproduce a fresh author environment instance.
        self.append(state, reference)
        self.append(state, reference)
        self.first = True

    def observations(self, state, reference):
        if not self.first:
            self.append(state, reference)
        self.first = False
        return {'obs': np.concatenate([np.asarray(b).reshape(-1) for b in self.history])[None].astype(np.float32)}

    def act(self, state, reference):
        raw = self.infer(self.observations(state, reference))['actions'].reshape(29)
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite MOSAIC action')
        self.last = raw.astype(np.float32)
        target = np.clip(self.last*self.scale+self.default, -10000., 10000.)
        def ordered(array):
            return tuple(float(x) for x in array[self.inverse])
        return ControlCommand(ordered(target), (0.,)*29, ordered(self.kp), ordered(self.kd), (0.,)*29)


def create(root, config):
    return MOSAIC(root, config)

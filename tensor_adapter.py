"""MOSAIC GPU observations and five-frame term-major history."""
import json
import re
from pathlib import Path
import torch
from motion_challenge.networks.torch_graph import TensorGraph
from motion_challenge.simulation.tensor_math import normalize, conjugate, multiply, rotate, matrix


def vector(value, names, device):
    if isinstance(value, dict):
        result = []
        for name in names:
            values = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
            if len(values) > 1: raise ValueError("Ambiguous named joint value")
            result.append(values[0] if values else 0.)
        value = result
    x = torch.as_tensor(value, dtype=torch.float32, device=device)
    return x.expand(len(names)) if x.ndim == 0 else x


def axis_rotation(angle, axis):
    values = [torch.zeros_like(angle) for _ in range(3)]
    values[axis] = (angle/2).sin()
    return torch.stack((*values, (angle/2).cos()), dim=-1)


class TensorMOSAIC:
    def __init__(self, root, runtime, joint_names, slots, device, converted_weights):
        self.device = torch.device(device)
        cfg = json.loads((Path(root)/"policy.json").read_text())
        if cfg["anchor_body_name"] != "torso_link": raise ValueError("Expected G1 torso anchor")
        names = cfg["policy_joint_names"]
        self.order = [joint_names.index(n) for n in names]
        self.inverse = [names.index(n) for n in joint_names]
        self.waist = [joint_names.index("waist_"+axis+"_joint") for axis in ("yaw","roll","pitch")]
        self.default = vector(cfg["default_joint_pos"], names, device)
        self.scale = vector(cfg["action_scale"], names, device)
        self.kp = vector(cfg["joint_kp"], names, device)[self.inverse]
        self.kd = vector(cfg["joint_kd"], names, device)[self.inverse]
        self.model = TensorGraph(converted_weights, device, runtime["weights_sha256"]).eval()
        self.buffers = [torch.zeros((slots,5,n), device=device, dtype=torch.float64) for n in (58,6,3,29,29,29)]
        self.last = torch.zeros((slots,29), device=device)
        self.heading_inverse = torch.zeros((slots,4), device=device, dtype=torch.float64)
        self.fresh = torch.ones(slots, device=device, dtype=torch.bool)

    def torso(self, root_orientation, joints):
        # G1 torso chain: pelvis -> Z yaw -> X roll -> Y pitch.
        rotation = normalize(root_orientation.double())
        for index, axis in zip(self.waist, (2,0,1)):
            rotation = multiply(rotation, axis_rotation(joints[...,index].double(), axis))
        return normalize(rotation)

    def reset(self, slots):
        self.last[slots] = 0
        self.fresh[slots] = True

    def observations(self, state, reference):
        q = state.joint_position_rad[:,self.order]
        qd = state.joint_velocity_rad_s[:,self.order]
        actual_root = normalize(state.root_orientation_xyzw.double())
        omega = rotate(conjugate(actual_root),state.root_angular_velocity_world_rad_s.double())
        actual = self.torso(actual_root,state.joint_position_rad)
        target = self.torso(reference.root_orientation_xyzw[:,0],reference.joint_position_rad[:,0])
        m = matrix(target)
        initial_yaw = torch.atan2(m[:,1,0],m[:,0,0])
        initial_inverse = axis_rotation(-initial_yaw,2)
        self.heading_inverse.copy_(torch.where(self.fresh[:,None],initial_inverse,self.heading_inverse))
        actual = normalize(multiply(self.heading_inverse,actual).float().double())
        target = normalize(multiply(self.heading_inverse,target).float().double())
        relative = normalize(multiply(conjugate(actual),target).float().double())
        orientation = matrix(relative)[...,:,:2].flatten(1)
        command = torch.cat((reference.joint_position_rad[:,0,self.order],reference.joint_velocity_rad_s[:,0,self.order]),dim=-1).float()
        terms = (command,orientation,omega,q.float()-self.default,qd.float(),self.last.clone())
        for buffer,term in zip(self.buffers,terms):
            normal = torch.cat((buffer[:,1:],term[:,None].double()),dim=1)
            initial = torch.zeros_like(buffer)
            initial[:,-2:] = term[:,None]
            buffer.copy_(torch.where(self.fresh[:,None,None],initial,normal))
        self.fresh.zero_()
        return {"obs":torch.cat([h.flatten(1) for h in self.buffers],dim=-1).float()}

    @torch.inference_mode()
    def act(self, state, reference):
        action = self.model(**self.observations(state,reference))["actions"]
        self.last.copy_(action)
        target = (action*self.scale+self.default).clamp(-10000.,10000.)[:,self.inverse]
        zero = torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1)

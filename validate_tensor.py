
import sys,json,pathlib,types
import numpy as np,torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
import argparse
from motion_challenge.resources import resource_root
p=argparse.ArgumentParser()
p.add_argument("--data",type=pathlib.Path,required=True)
p.add_argument("--weights",type=pathlib.Path,required=True)
p.add_argument("--converted",type=pathlib.Path,required=True)
p.add_argument("--device",default="cuda:0")
cli=p.parse_args()
root=pathlib.Path(__file__).resolve().parent
device=cli.device
from motion_challenge.data.prepare import prepare
prepared=prepare("lafan",cli.data,cli.converted.parent/"prepared")
meta=json.loads((prepared/"dataset.json").read_text())
poses=np.load(prepared/"poses.npy",mmap_mode="r")
from motion_challenge.simulation.tensor_reference import TensorMotionBank,TensorReference
from motion_challenge.simulation.physics import World
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket,ReferenceFrame
from tensor_adapter import TensorMOSAIC
world=World(resource_root()/"robots/g1-29dof-mjwarp-v1.json")
slots=7
bank=TensorMotionBank(np.array(poses),meta["starts"],meta["lengths"],np.full(40,30.),meta["names"],device)
refs=TensorReference(world,bank,56,device)
runtime=json.loads((root/"runtime.mjwarp.json").read_text())
adapter=TensorMOSAIC(root,runtime,world.joint_names,slots,device,cli.converted)
adapter.reset(torch.arange(slots,device=device))
sys.modules.setdefault("onnxruntime",types.ModuleType("onnxruntime"))
from adapter import MOSAIC
from motion_challenge.adapters.onnx import vector
original=[]
for _ in range(slots):
 a=MOSAIC.__new__(MOSAIC);a.cfg=json.loads((root/"policy.json").read_text())
 a.runtime=runtime;a.names=a.cfg["policy_joint_names"];a.default=vector(a.cfg["default_joint_pos"],a.names).astype(np.float32);original.append(a)
rng=np.random.default_rng(1997);maximum=0.;action_error=0.
from onnx.reference import ReferenceEvaluator
source=str(cli.weights)
oracle=ReferenceEvaluator(source)
for step in range(24):
 times=torch.tensor(np.arange(slots)*.031+step*.02,device=device,dtype=torch.float64)[:,None]+torch.tensor(runtime["reference_offsets_s"],device=device,dtype=torch.float64)[None]
 r=refs.motion(torch.arange(slots,device=device),times)
 q=rng.normal(0,.1,(slots,29));qd=rng.normal(0,.1,(slots,29));quat=rng.normal(size=(slots,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True);omega=rng.normal(0,.1,(slots,3))
 if step==11:adapter.reset(torch.tensor([1,4],device=device))
 state=DeployableStateBatch(world.joint_names,torch.full((slots,),step*.02,device=device),*[torch.tensor(x,device=device) for x in (q,qd,quat,omega)])
 inputs=adapter.observations(state,r)
 cpu={n:getattr(r,n).cpu().numpy() for n in ("requested_time_s","source_time_s","joint_position_rad","root_position_world_m","root_orientation_xyzw","joint_velocity_rad_s")}
 expected_actions=[]
 for row,a in enumerate(original):
  packet=DeployableStatePacket(step*.02,world.joint_names,tuple(q[row]),tuple(qd[row]),tuple(quat[row]),tuple(omega[row]))
  reference=[]
  for k in range(1):
   pos=tuple(cpu["root_position_world_m"][row,k]);rot=tuple(cpu["root_orientation_xyzw"][row,k])
   world.data.qpos[:]=world.model.qpos0
   world.data.qpos[:3]=pos;world.data.qpos[3:7]=[rot[3],*rot[:3]];world.data.qpos[world.qadr]=cpu["joint_position_rad"][row,k]
   import mujoco
   mujoco.mj_kinematics(world.model,world.data);bid=world.model.body("torso_link").id
   body=types.SimpleNamespace(name="torso_link",position_world_m=world.data.xpos[bid].copy(),orientation_xyzw=world.data.xquat[bid,[1,2,3,0]].copy())
   reference.append(ReferenceFrame(float(cpu["requested_time_s"][row,k]),float(cpu["source_time_s"][row,k]),tuple(cpu["joint_position_rad"][row,k]),pos,rot,joint_velocity_rad_s=tuple(cpu["joint_velocity_rad_s"][row,k]),bodies=(body,)))
  world.data.qpos[:]=world.model.qpos0
  world.data.qpos[:3]=0;world.data.qpos[3:7]=[quat[row,3],*quat[row,:3]];world.data.qpos[world.qadr]=q[row]
  mujoco.mj_kinematics(world.model,world.data)
  actual_body=types.SimpleNamespace(name="torso_link",position_world_m=world.data.xpos[bid].copy(),orientation_xyzw=world.data.xquat[bid,[1,2,3,0]].copy())
  packet=types.SimpleNamespace(time_s=step*.02,joint_names=world.joint_names,joint_position_rad=q[row],joint_velocity_rad_s=qd[row],root_orientation_xyzw=quat[row],root_angular_velocity_world_rad_s=omega[row],bodies=(actual_body,))
  if step==0 or (step==11 and row in [1,4]):a.reset(packet,reference,0)
  expected={k:np.asarray(v,np.float32) for k,v in a.observations(packet,reference).items()}
  for name,value in inputs.items():
   actual=value[row:row+1].cpu().numpy();maximum=max(maximum,float(np.max(abs(actual-expected[name]))))
   np.testing.assert_allclose(actual,expected[name],atol=2e-6,rtol=2e-6,err_msg=name)
  action=oracle.run(["actions"],expected)[0].reshape(-1);expected_actions.append(action);a.last=action.astype(np.float32)
 with torch.inference_mode():actions=adapter.model(**inputs)["actions"]
 adapter.last.copy_(torch.tensor(np.stack(expected_actions),device=device))
 error=float(np.max(abs(actions.cpu().numpy()-np.stack(expected_actions))));action_error=max(action_error,error)
 np.testing.assert_allclose(actions.cpu().numpy(),np.stack(expected_actions),atol=2e-5,rtol=2e-5)
report={"status":"passed","steps":24,"batch":slots,"partial_reset_slots":[1,4],"partial_reset_step":11,"observation_max_abs_error":maximum,"network_max_abs_error":action_error}
(root/"validation/tensor-observation-parity.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))


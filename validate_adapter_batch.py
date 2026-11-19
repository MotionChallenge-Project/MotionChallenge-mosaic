"""Check full-batch inference and partial resets against independent slots."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from motion_challenge.resources import resource_root
from motion_challenge.data.prepare import prepare
from motion_challenge.simulation.physics import World
from motion_challenge.simulation.tensor_reference import TensorMotionBank, TensorReference
from motion_challenge.sdk import DeployableStateBatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parent
    runtime = json.loads((root / "runtime.mjwarp.json").read_text())
    world = World(resource_root() / "robots/g1-29dof-mjwarp-v1.json")
    prepared = prepare("lafan", args.data, args.converted.parent / "prepared")
    meta = json.loads((prepared / "dataset.json").read_text())
    poses = np.load(prepared / "poses.npy", mmap_mode="r")
    bank = TensorMotionBank(np.asarray(poses), meta["starts"], meta["lengths"],
        np.full(len(meta["names"]), meta["fps"]), meta["names"], args.device)
    reference = TensorReference(world, bank, 1, args.device)
    module, name = runtime["tensor_entrypoint"].split(":")
    factory = getattr(importlib.import_module(module), name)
    large = factory(root, runtime, world.joint_names, 4096, args.device, args.converted)
    small = factory(root, runtime, world.joint_names, 7, args.device, args.converted)
    selected = torch.tensor([0, 1, 2, 3, 4, 5, 4095], device=args.device)
    large.reset(torch.arange(4096, device=args.device))
    small.reset(torch.arange(7, device=args.device))
    seen = []
    def record(module, positional, keyword):
        inputs = list(positional) + list(keyword.values())
        seen.append(next(x.shape[0] for x in inputs if torch.is_tensor(x)))
    hook = large.model.register_forward_pre_hook(record, with_kwargs=True)
    rng = np.random.default_rng(718)
    maximum = 0.
    elapsed = []
    offsets = torch.tensor(runtime["reference_offsets_s"], device=args.device, dtype=torch.float64)
    ids = torch.arange(4096, device=args.device) % len(meta["names"])
    for step in range(4):
        if step == 2:
            large.reset(selected[[1, 6]])
            small.reset(torch.tensor([1, 6], device=args.device))
        arrays = [rng.normal(0, .1, (4096, n)) for n in (29, 29, 4, 3)]
        arrays[2] /= np.linalg.norm(arrays[2], axis=1, keepdims=True)
        values = [torch.as_tensor(x, device=args.device) for x in arrays]
        times = torch.full((4096,), step * .02, device=args.device)
        full_state = DeployableStateBatch(world.joint_names, times, *values)
        gold_state = DeployableStateBatch(world.joint_names, times[selected], *[x[selected] for x in values])
        clock = times.double()[:, None] + offsets
        full_ref = reference.motion(ids, clock)
        gold_ref = reference.motion(ids[selected], clock[selected])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record(torch.cuda.current_stream(args.device))
            full = large.act(full_state, full_ref)
            end.record(torch.cuda.current_stream(args.device))
            gold = small.act(gold_state, gold_ref)
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
        difference = float((full[selected] - gold).abs().max())
        maximum = max(maximum, difference)
        torch.testing.assert_close(full[selected], gold, atol=5e-5, rtol=2e-5)
        if not torch.isfinite(full).all():
            raise AssertionError("Non-finite full-batch output")
    hook.remove()
    if seen != [4096] * 4:
        raise AssertionError(f"Inference was not full-batch: {seen}")
    report = {"status": "passed", "batch": 4096, "forward_batches": seen,
        "steps": 4, "partial_reset_slots": [1, 4095], "partial_reset_step": 2,
        "independent_slots": selected.tolist(), "max_pd_abs_error": maximum,
        "act_ms": elapsed}
    (root / "validation/full-batch-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

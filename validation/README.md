# Author inference validation

The original `gmt.onnx` is the general tracker identified in the authors'
[model card](https://huggingface.co/BAAI-Humanoid/MOSAIC_Model/blob/2ab795d7420cf339bb803bfb145c18faf8137cb8/README.md).
The Noitom and PICO teleoperation adaptors are separate models.

`validate_inference.py` executes unmodified methods extracted from the pinned
RobotBridge source. It compares complete reset/step sequencing, term-major
histories, observations, actions, joint ordering and float32 PD targets.
Source files are checked against `native-sources.json` before execution.

```bash
git clone https://github.com/BAAI-Humanoid/RobotBridge.git /tmp/RobotBridge
git -C /tmp/RobotBridge checkout 175287dd559c098d21b287a8a27c6bb48d4606ff
uv run validate_inference.py --source /tmp/RobotBridge \
  --weights /path/to/gmt.onnx --output /tmp/mosaic-native-parity.json
```

The report covers 160 states, four independent resets, 24 clamped endpoint
cases, shuffled joint names and a common 1.37-radian world-yaw rotation.
Observation, action and motor-target maximum absolute differences are zero.
Intermediate float64 history values differ by at most 6.7e-16.

Offline reference tracking selects the author's `MosaicEnv` path with
`use_teleop=false`, `use_estimator=false`, `eval_mode=true`, five history samples
and one command frame. This explicit override is necessary because the released
control YAML enables a live teleoperation input by default. No teleop smoothing
or residual adaptor is applied to the general tracker.

RobotBridge computes observations twice during reset, then computes each next
observation before advancing its reference index. The adapter retains this
sequence, including the one-control-period reference delay. Each independent
benchmark clip gets new buffers, as a freshly constructed author environment
would. The final reference is held to the canonical motion endpoint.

The author's loader removes the initial reference torso yaw. The adapter
expresses both actual and reference orientations in that same fixed frame;
relative orientation and the benchmark's world trajectory remain unchanged.
RobotBridge's NPZ `[1:]` file slicing is not applied to the canonical dataset.
The benchmark still starts at its first frame and keeps every scored frame.

Only state transport and author-side metric/logging/simulator facilities are
replaced in this inference audit. Benchmark physics, initial state, torque
limits, failure conditions and metric calculation remain owned by the Core.

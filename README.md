# MotionChallenge · mosaic

[Author project](https://baai-humanoid.github.io/MOSAIC/) · [Checkpoint](https://huggingface.co/BAAI-Humanoid/MOSAIC_Model/blob/2ab795d7420cf339bb803bfb145c18faf8137cb8/gmt.onnx) · [Provenance](provenance.json)

## Evaluate

Use Linux x86-64, Python 3.12 and an NVIDIA GPU compatible with the locked CUDA dependencies.

~~~sh
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-mosaic.git
cd MotionChallenge-mosaic
uv sync --locked
uv run --locked python evaluate.py --dataset lafan --data /data/lafan --device cuda:0 --output results/lafan
uv run --locked python evaluate.py --dataset seed --data /data/seed/g1.tar.gz --device cuda:0 --output results/seed
~~~

Obtain the original G1 data from [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) or [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed). The LAFAN directory contains g1/*.csv. Keep the SEED g1.tar.gz archive intact; Core applies the single published list of 129,561 motions. Motion data remains local.

The command downloads the pinned original checkpoint, verifies its SHA256, converts weights locally for batch inference, checks conversion parity, evaluates the complete dataset and independently recomputes trajectory metrics on CPU.

--cache /local-ssd/MotionChallenge shares original weights, converted weights, prepared motions and scoring FK across runs. --weights /path/to/original.onnx selects an existing checkpoint; --offline uses cached artifacts. --data-format lerobot-v3 selects an existing local LeRobot v3 dataset.

## Evaluation

LAFAN uses 40 environments; SEED uses 4096. Inference runs the full environment batch. Finished slots are refilled with the next motion, longest motions first.

Control and scoring run every 20 ms, with four 5 ms simulation steps. Contact sliding friction is 1.0. The Newton solver uses up to 50 iterations and 20 line-search iterations.

Policy observations exclude actual global position and actual linear velocity in every frame. Policy references provide root/joint information; the adapter constructs its reference features. Core precomputes reference FK for scoring, and reads actual body poses from simulation.

All motions run to the endpoint. Results include success, completion and full-motion/prefix tracking metrics. Numerical simulation failures do not produce a full-suite ranked score.

BONES-SEED evaluation uses a feasibility-filtered subset of the original G1 motion data. Download the original archive; Core applies one public inclusion list of 129,561 motions shared by every model.

## Validation

[Admission](admission.mjwarp.json) · [Native inference](validation/native-parity.json) · [Tensor observations](validation/tensor-observation-parity.json) · [Full batch](validation/full-batch-parity.json)

~~~sh
uv run --locked motionchallenge-verify results/lafan/result.json --jobs 8
uv run --locked motionchallenge-reproduce results/lafan results/lafan-repeat --output reproduction.json
~~~

Each result binds the protocol, data, weights, adapter, simulator and recorded trajectories. Reproduction reports pipeline identity and numerical differences separately. The pinned MotionChallenge-Core submodule supplies preparation, simulation, scoring and verification.

<!-- evaluation-evidence -->

[LAFAN full evaluation](validation/lafan-evaluation.json) · [SEED full evaluation](validation/seed-evaluation.json) · [LAFAN pipeline reproduction](validation/lafan-reproduction.json)

[Independent original-SEED one-command run](validation/seed-one-command.json)

<!-- current-evaluation-evidence -->

## Current evaluation evidence

[LAFAN1 full evaluation](validation/lafan-evaluation.json) · [BONES-SEED full evaluation](validation/seed-evaluation.json)

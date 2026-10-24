# MOSAIC GMT

[Author project](https://baai-humanoid.github.io/MOSAIC/) · [Checkpoint](https://huggingface.co/BAAI-Humanoid/MOSAIC_Model/blob/2ab795d7420cf339bb803bfb145c18faf8137cb8/gmt.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-mosaic.git
cd MotionChallenge-mosaic
uv sync
uv run evaluate.py
```

Produces success, completion and six MPBPE values in a verified `result.json`.
MuJoCo 3.17 CPU batch; `--jobs 8` sets parallelism. Missing data and author
weights are downloaded and SHA256-checked. Results and traces stay outside Git.

Use local artifacts with `--weights PATH --data PATH --offline`.
The root `MotionChallenge-Core` submodule owns data, physics, scoring and verification.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](MotionChallenge-Core/docs/OBSERVATIONS.md).

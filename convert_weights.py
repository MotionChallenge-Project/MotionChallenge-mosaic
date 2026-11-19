"""Convert the pinned author checkpoint and validate dynamic-batch inference."""
from motion_challenge.networks.torch_graph import convert, validate


def prepare(root, runtime, source, output, artifacts, device):
    outputs = runtime["tensor_outputs"]
    convert(source, output, runtime["weights_sha256"], outputs)
    return validate(source, output, devices=("cpu", device))

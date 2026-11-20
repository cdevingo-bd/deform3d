"""Small batched deformation example; no simulator, downloads or visual assets.

Run after installing the package: python examples/quickstart.py --device cpu
Use --device cuda:0 on an NVIDIA GPU.
"""

import argparse

import numpy as np
import warp as wp

from deform3d_warp import DeviceBindings, enqueue_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    worlds = np.stack((rest, rest * [1.2, .8, 1.1])).astype(np.float32)
    weights = np.array([[.25, .25, .25, .25], [.1, .2, .3, .4]], dtype=np.float32)
    bindings = DeviceBindings(
        np.tile([0, 1, 2, 3], (2, 1)), weights, source_vertex_count=4, device=args.device
    )
    source = wp.array(worlds, dtype=wp.vec3f, device=args.device)
    output = wp.empty((2, 2), dtype=wp.vec3f, device=args.device)
    enqueue_embedding(source, bindings, output)
    actual = output.numpy()  # Readback is only for this example's verification.
    expected = np.einsum("vi,wic->wvc", weights, worlds)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    print(f"PASS: two worlds, two embedded points per world ({args.device})")
    print(actual)


if __name__ == "__main__":
    main()

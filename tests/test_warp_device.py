"""Simulator-independent parity tests: python tests/test_warp_device.py --device cuda:0.

Requires numpy and warp-lang. Also works with pytest (CPU by default; set
DEFORM3D_WARP_DEVICE=cuda:0 to exercise CUDA and graph replay).
"""

from pathlib import Path
import argparse
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from deform3d_warp import (DeviceBindings, DeviceSurfaceBindings, SurfaceTopology,
                          enqueue_embedding, enqueue_surface_embedding, enqueue_normals)


def reference_normals(points, faces, epsilon=1e-20):
    """Independent float64 area-weighted reference."""
    points = np.asarray(points, dtype=np.float64)
    total = np.zeros_like(points)
    for a, b, c in faces:
        cross = np.cross(points[:, b] - points[:, a], points[:, c] - points[:, a])
        for vertex in (a, b, c):
            total[:, vertex] += cross
    length = np.linalg.norm(total, axis=-1, keepdims=True)
    return np.divide(total, length, out=np.zeros_like(total), where=length > epsilon)


class WarpDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = wp.get_device(os.environ.get("DEFORM3D_WARP_DEVICE", "cpu"))

    def xyz(self, values):
        return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.vec3f, device=self.device)

    def test_affine_embedding_world_and_vertex_windows(self):
        rng = np.random.default_rng(741)
        source = rng.normal(size=(5, 19, 3)).astype(np.float32)
        ids = rng.integers(0, 11, size=(67, 4), dtype=np.int32)
        weights = rng.uniform(size=(67, 4)).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        # Intentional extrapolation remains affine-covariant.
        weights[0] = [-0.25, 0.5, 0.5, 0.25]
        binding = DeviceBindings(ids, weights, source_vertex_count=11, device=self.device)
        sentinel = np.full((6, 75, 3), -987.0, dtype=np.float32)
        output = self.xyz(sentinel)
        enqueue_embedding(self.xyz(source), binding, output, world_count=3,
                          source_world_offset=1, output_world_offset=2,
                          source_vertex_offset=4, output_vertex_offset=5)
        expected = sentinel.copy()
        expected[2:5, 5:72] = np.sum(source[1:4, 4 + ids].astype(np.float64)
                                      * weights[None, :, :, None], axis=2)
        np.testing.assert_allclose(output.numpy(), expected, atol=5e-7, rtol=5e-6)

        # A general affine map of the cage must induce the same map of the mesh.
        matrix = np.array([[1.3, 0.4, -0.2], [0.1, 0.8, 0.3], [0.2, -0.1, 1.1]], np.float32)
        shift = np.array([3.0, -2.0, 4.0], np.float32)
        transformed = source @ matrix.T + shift
        enqueue_embedding(self.xyz(transformed), binding, output, world_count=3,
                          source_world_offset=1, output_world_offset=2,
                          source_vertex_offset=4, output_vertex_offset=5)
        np.testing.assert_allclose(output.numpy()[2:5, 5:72],
                                   expected[2:5, 5:72] @ matrix.T + shift,
                                   atol=2e-6, rtol=5e-6)

    def test_normals_world_windows_area_weights_and_repeatability(self):
        # Two faces of unequal area and direction distinguish area weighting
        # from averaging unit face normals; the last vertex is isolated.
        points = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 1], [7, 7, 7]], np.float32)
        faces = np.array([[0, 1, 2], [0, 3, 1]], np.int32)
        worlds = np.stack([points, points * [1, 2, 3], points @
                           np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])])
        source = np.full((5, 9, 3), 123.0, np.float32)
        source[1:4, 2:7] = worlds
        topology = SurfaceTopology(faces, vertex_count=5, device=self.device)
        sentinel = np.full((6, 11, 3), 78.0, np.float32)
        output = self.xyz(sentinel)
        scratch = wp.empty((3, 2), dtype=wp.vec3f, device=self.device)
        positions = self.xyz(source)

        def launch():
            enqueue_normals(positions, topology, scratch, output, world_count=3,
                            source_world_offset=1, output_world_offset=2,
                            source_vertex_offset=2, output_vertex_offset=4)

        launch()
        actual = output.numpy()
        expected = sentinel.copy()
        expected[2:5, 4:9] = reference_normals(worlds, faces)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
        for _ in range(5):
            launch()
        np.testing.assert_array_equal(output.numpy(), actual)
        np.testing.assert_array_equal(topology.offsets.numpy(), [0, 2, 4, 5, 6, 6])
        np.testing.assert_array_equal(topology.incident_faces.numpy(), [0, 1, 0, 1, 0, 1])

    def test_degenerate_cancelling_small_and_empty_faces(self):
        points = self.xyz([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]]])
        output = wp.empty((1, 4), dtype=wp.vec3f, device=self.device)
        for faces in (np.array([[0, 1, 3], [0, 0, 0]], np.int32),
                      np.array([[0, 1, 2], [2, 1, 0]], np.int32),
                      np.empty((0, 3), np.int32)):
            topology = SurfaceTopology(faces, vertex_count=4, device=self.device)
            scratch = wp.empty((1, len(faces)), dtype=wp.vec3f, device=self.device)
            enqueue_normals(points, topology, scratch, output)
            np.testing.assert_array_equal(output.numpy(), np.zeros((1, 4, 3)))

        topology = SurfaceTopology([[0, 1, 2]], vertex_count=4, device=self.device)
        scratch = wp.empty((1, 1), dtype=wp.vec3f, device=self.device)
        tiny_points = self.xyz(points.numpy() * 1e-8)
        enqueue_normals(tiny_points, topology, scratch, output, epsilon=1e-15)
        np.testing.assert_array_equal(output.numpy(), np.zeros((1, 4, 3)))
        enqueue_normals(tiny_points, topology, scratch, output, epsilon=0)
        np.testing.assert_allclose(output.numpy()[0, :3], [[0, 0, 1]] * 3, atol=1e-7)

    def test_setup_rejects_invalid_bindings_and_topology(self):
        valid_ids = [[0, 1, 2, 3]]
        for ids, weights in (([[-1, 1, 2, 3]], [[0.25] * 4]),
                             ([[0, 1, 2, 4]], [[0.25] * 4]),
                             ([[0.0, 1, 2, 3]], [[0.25] * 4]),
                             (valid_ids, [[0.5] * 4]),
                             (valid_ids, [[float("nan"), 0, 0, 1]])):
            with self.assertRaises(ValueError):
                DeviceBindings(ids, weights, source_vertex_count=4, device=self.device)
        for faces in ([[0, 1, 4]], [[0, -1, 2]], [[0.0, 1, 2]], [[0, 1]]):
            with self.assertRaises(ValueError):
                SurfaceTopology(faces, vertex_count=4, device=self.device)

    def test_frame_rejects_capacity_alias_and_invalid_epsilon(self):
        points = self.xyz(np.zeros((2, 4, 3)))
        output = wp.empty_like(points)
        binding = DeviceBindings([[0, 1, 2, 3]], [[0.25] * 4], source_vertex_count=4, device=self.device)
        for options in ({"world_count": 3}, {"source_vertex_offset": 1},
                        {"output_vertex_offset": 4}, {"source_world_offset": -1},
                        {"world_count": 0.5}):
            with self.assertRaises(ValueError):
                enqueue_embedding(points, binding, output, **options)
        with self.assertRaises(ValueError):
            enqueue_embedding(points, binding, points)
        overlapping_output = wp.array(ptr=output.ptr, dtype=wp.vec3f, shape=(2, 4),
                                      strides=(12, 12), device=self.device, copy=False)
        with self.assertRaises(ValueError):
            enqueue_embedding(points, binding, overlapping_output)
        topology = SurfaceTopology([[0, 1, 2]], vertex_count=4, device=self.device)
        scratch = wp.empty((2, 1), dtype=wp.vec3f, device=self.device)
        for epsilon in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                enqueue_normals(points, topology, scratch, output, epsilon=epsilon)
        with self.assertRaises(ValueError):
            enqueue_normals(points, topology, output, output)
        with self.assertRaises(ValueError):
            enqueue_normals(points, topology, scratch[:1], output)

    def test_empty_bindings_and_zero_worlds_are_noops(self):
        binding = DeviceBindings(np.empty((0, 4), np.int32), np.empty((0, 4), np.float32),
                                 source_vertex_count=0, device=self.device)
        source = wp.empty((2, 0), dtype=wp.vec3f, device=self.device)
        output = self.xyz(np.full((2, 1, 3), 5))
        enqueue_embedding(source, binding, output)
        enqueue_embedding(source, binding, output, world_count=0)
        np.testing.assert_array_equal(output.numpy(), np.full((2, 1, 3), 5))

    def test_external_views_require_float32_alignment(self):
        owner = self.xyz(np.zeros((2, 8, 3)))
        source = self.xyz(np.zeros((2, 4, 3)))
        output = wp.empty_like(source)
        binding = DeviceBindings([[0, 1, 2, 3]], [[0.25] * 4],
                                 source_vertex_count=4, device=self.device)
        # These deliberately malformed raw-pointer wrappers are metadata only.
        # Trap launch so a regression cannot execute an unsafe memory access.
        with patch("deform3d_warp.wp.launch", side_effect=AssertionError("unexpected kernel launch")):
            for byte_offset, strides in ((1, (96, 12)), (0, (64, 13)), (0, (49, 12))):
                bad = wp.array(ptr=owner.ptr + byte_offset, dtype=wp.vec3f,
                               shape=(2, 4), strides=strides, device=self.device, copy=False)
                with self.assertRaisesRegex(ValueError, "aligned"):
                    enqueue_embedding(bad, binding, output)
                with self.assertRaisesRegex(ValueError, "aligned"):
                    enqueue_embedding(source, binding, bad)
            # A singleton dimension's stride is unused and need not align.
            singleton = wp.array(ptr=owner.ptr, dtype=wp.vec3f, shape=(1, 4),
                                 strides=(1, 12), device=self.device, copy=False)
            enqueue_embedding(singleton, binding, output, world_count=0)
            oversized_stride = wp.array(ptr=owner.ptr, dtype=wp.vec3f, shape=(2, 1),
                                        strides=(2**32 + 12, 12), device=self.device, copy=False)
            with self.assertRaisesRegex(ValueError, "int32 stride ABI"):
                enqueue_embedding(source, binding, oversized_stride)

    def test_positive_strided_engine_views_preserve_neighbor_data(self):
        source_host = np.arange(2 * 16 * 3, dtype=np.float32).reshape(2, 16, 3)
        source_owner = self.xyz(source_host)
        source = source_owner[:, ::2]
        output_owner = self.xyz(np.full((2, 8, 3), -12.0, np.float32))
        output = output_owner[:, 1::2]
        ids = np.array([[0, 2, 4, 7], [1, 3, 5, 6]], np.int32)
        weights = np.array([[0.25] * 4, [0.1, 0.2, 0.3, 0.4]], np.float32)
        binding = DeviceBindings(ids, weights, source_vertex_count=8, device=self.device)
        enqueue_embedding(source, binding, output)
        expected = np.full((2, 8, 3), -12.0, np.float32)
        expected[:, [1, 3]] = np.sum(source_host[:, ::2][:, ids] * weights[None, :, :, None], axis=2)
        np.testing.assert_allclose(output_owner.numpy(), expected, atol=2e-5)
        with self.assertRaises(ValueError):
            enqueue_embedding(source_owner[:, :8], binding, source_owner[:, 4:12])

    def test_surface_affine_weights_batched_windows_and_strided_validity(self):
        triangle = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], np.float32)
        transforms = np.array([np.eye(3), [[1, .2, .3], [.1, .8, -.2], [0, .4, 1.1]]], np.float32)
        source = np.full((4, 7, 3), 45.0, np.float32)
        source[1:3, 2:5] = triangle @ transforms.transpose(0, 2, 1) + [1, -2, 3]
        weights = np.array([[.2, .3, .5], [-.2, .6, .6]], np.float32)
        binding = DeviceSurfaceBindings([[0, 1, 2]] * 2, weights,
                                        source_vertex_count=3, device=self.device)
        output = self.xyz(np.full((4, 6, 3), -77.0, np.float32))
        validity_owner = wp.full((4, 12), -1, dtype=wp.int32, device=self.device)
        validity = validity_owner[:, ::2]
        enqueue_surface_embedding(self.xyz(source), binding, output, validity, world_count=2,
                                  source_world_offset=1, output_world_offset=2,
                                  source_vertex_offset=2, output_vertex_offset=3)
        expected = np.full((4, 6, 3), -77.0, np.float32)
        expected[2:4, 3:5] = np.einsum("bvc,pv->bpc", source[1:3, 2:5], weights)
        np.testing.assert_allclose(output.numpy(), expected, atol=1e-6)
        flags = np.full((4, 12), -1, np.int32)
        flags[2:4, [6, 8]] = 1
        np.testing.assert_array_equal(validity_owner.numpy(), flags)

    def test_surface_signed_offsets_follow_winding_and_keep_world_distance(self):
        source = np.array([[[0, 0, 0], [2, 0, 1], [0, 3, 2]],
                           [[0, 0, 0], [8, 0, 4], [0, 3, 2]]], np.float32)
        ids = np.array([[0, 1, 2], [0, 2, 1], [0, 1, 2]], np.int32)
        weights = np.full((3, 3), 1 / 3, np.float32)
        offsets = np.array([.25, .25, -.1], np.float32)
        binding = DeviceSurfaceBindings(ids, weights, offsets, source_vertex_count=3, device=self.device)
        output = wp.empty((2, 3), dtype=wp.vec3f, device=self.device)
        validity = wp.empty((2, 3), dtype=wp.int32, device=self.device)
        enqueue_surface_embedding(self.xyz(source), binding, output, validity)
        triangles = source[:, ids].astype(np.float64)
        base = (triangles * weights[None, :, :, None]).sum(axis=2)
        normal = np.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0])
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        expected = base + normal * offsets[None, :, None]
        np.testing.assert_allclose(output.numpy(), expected, atol=5e-7)
        np.testing.assert_allclose(np.linalg.norm(output.numpy() - base, axis=-1),
                                   np.broadcast_to(np.abs(offsets), (2, 3)), atol=5e-7)
        np.testing.assert_array_equal(validity.numpy(), np.ones((2, 3), np.int32))

    def test_surface_invalid_geometry_preserves_output_and_reports_on_device(self):
        source = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                           [[0, 0, 0], [1, 0, 0], [1, 1e-7, 0]],
                           [[0, 0, 0], [float("nan"), 0, 0], [0, 1, 0]],
                           [[0, 0, 0], [1e20, 0, 0], [0, 1e20, 0]],
                           [[0, 0, 0], [1e-20, 0, 0], [0, 1e-20, 0]]], np.float32)
        binding = DeviceSurfaceBindings([[0, 1, 2]], [[.25, .25, .5]], [0.0],
                                        source_vertex_count=3, device=self.device)
        output = self.xyz(np.full((6, 1, 3), -99.0, np.float32))
        validity = wp.empty((6, 1), dtype=wp.int32, device=self.device)
        enqueue_surface_embedding(self.xyz(source), binding, output, validity)
        np.testing.assert_array_equal(validity.numpy().reshape(-1), [1, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(output.numpy()[1:4], np.full((3, 1, 3), -99.0))
        self.assertTrue(np.isfinite(output.numpy()).all())
        for offsets in ([float("nan")], [0, 1]):
            with self.assertRaises(ValueError):
                DeviceSurfaceBindings([[0, 1, 2]], [[.25, .25, .5]], offsets,
                                      source_vertex_count=3, device=self.device)

    def test_surface_cuda_graph_handles_changed_positions_and_collapsed_triangle(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA graph replay requires a CUDA device")
        source = self.xyz([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
        binding = DeviceSurfaceBindings([[0, 1, 2]], [[.25, .25, .5]], [.1],
                                        source_vertex_count=3, device=self.device)
        output = wp.empty((1, 1), dtype=wp.vec3f, device=self.device)
        validity = wp.empty((1, 1), dtype=wp.int32, device=self.device)
        stream = wp.Stream(self.device)
        wp.synchronize_device(self.device)

        def frame():
            enqueue_surface_embedding(source, binding, output, validity, stream=stream)

        frame()
        wp.synchronize_stream(stream)
        wp.capture_begin(stream=stream)
        try:
            frame()
        finally:
            graph = wp.capture_end(stream=stream)
        for triangle in ([[[0, 0, 0], [1, 0, 0], [0, 0, 2]]],
                         [[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]):
            upload = self.xyz(triangle)
            wp.synchronize_device(self.device)
            previous = output.numpy().copy()
            wp.copy(source, upload, stream=stream)
            wp.capture_launch(graph, stream=stream)
            wp.synchronize_stream(stream)
            if triangle[0][2] == [2, 0, 0]:
                np.testing.assert_array_equal(validity.numpy(), [[0]])
                np.testing.assert_array_equal(output.numpy(), previous)
            else:
                np.testing.assert_array_equal(validity.numpy(), [[1]])
                np.testing.assert_allclose(output.numpy(), [[[.25, -.1, 1.0]]], atol=1e-6)

    def test_cuda_graph_replays_updated_source_on_explicit_stream(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA graph replay requires a CUDA device")
        source_host = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                [[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]]], np.float32)
        binding = DeviceBindings([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
                                 [[0.25] * 4] * 3, source_vertex_count=4, device=self.device)
        topology = SurfaceTopology([[0, 1, 2]], vertex_count=3, device=self.device)
        source = self.xyz(source_host)
        output = wp.empty((2, 3), dtype=wp.vec3f, device=self.device)
        normals = wp.empty_like(output)
        scratch = wp.empty((2, 1), dtype=wp.vec3f, device=self.device)
        stream = wp.Stream(self.device)

        def frame():
            enqueue_embedding(source, binding, output, stream=stream)
            enqueue_normals(output, topology, scratch, normals, stream=stream)

        wp.synchronize_device(self.device)  # Setup uploads precede custom-stream reads.
        frame()  # Compile/load kernels before capture.
        wp.synchronize_stream(stream)
        wp.capture_begin(stream=stream)
        try:
            frame()
        finally:
            graph = wp.capture_end(stream=stream)
        for scale in (1.0, 2.0, -1.0):
            updated = source_host.copy()
            updated[..., 2] += updated[..., 1] * scale
            # Copies are explicit test setup; the captured frame stays resident.
            upload = self.xyz(updated)
            wp.synchronize_device(self.device)
            wp.copy(source, upload, stream=stream)
            wp.capture_launch(graph, stream=stream)
            wp.synchronize_stream(stream)
            np.testing.assert_allclose(output.numpy(), updated[:, :3], atol=1e-6)
            np.testing.assert_allclose(normals.numpy(), reference_normals(updated[:, :3], [[0, 1, 2]]), atol=1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args, unittest_args = parser.parse_known_args()
    os.environ["DEFORM3D_WARP_DEVICE"] = args.device
    unittest.main(argv=[sys.argv[0]] + unittest_args)

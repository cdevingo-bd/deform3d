"""Dense geometry updates from an existing MuJoCo Warp or MJLab simulation.

This adapter does not step physics, patch a solver, or change a renderer. Bind
once to the compiled model; rebuild after topology or reference-pose changes.
"""

from pathlib import Path

import warp as wp

from . import DeviceBindings, DeviceSurfaceBindings, enqueue_embedding, enqueue_surface_embedding
from .bindings import read_bindings
from .surface_bindings import read_surface_bindings


class FlexEmbedding:
    """Own prepared bindings and batched output, while the host owns simulation.

    ``model`` is the CPU MjModel used to export the binding file. ``data`` is the
    matching Warp Data (MJLab's ``sim.wp_data``). ``positions`` and, for shells,
    ``validity`` remain on the device. A zero validity entry means an invalid
    surface result; inspect it in diagnostics before using that output.
    """

    def __init__(self, model, data, binding_file):
        binding_file = Path(binding_file)
        with binding_file.open("rb") as stream:
            magic = stream.read(8)
        if magic == b"D3EMB003":
            flex_id, offset, records = read_bindings(binding_file, model)
            binding_type = DeviceBindings
        elif magic == b"D3SUR001":
            flex_id, offset, records = read_surface_bindings(binding_file, model)
            binding_type = DeviceSurfaceBindings
        else:
            raise ValueError("unknown deform3d binding format")
        source = data.flexvert_xpos
        if source.ndim != 2 or source.shape[1] != model.nflexvert:
            raise ValueError("Warp data does not match the CPU model's flex vertex count")
        kwargs = dict(source_vertex_count=int(model.flex_vertnum[flex_id]), device=source.device)
        if binding_type is DeviceSurfaceBindings:
            kwargs["normal_offsets"] = records["normal_offset"]
        self.bindings = binding_type(records["indices"], records["weights"], **kwargs)
        self.flex_id = flex_id
        self.vertex_offset = offset
        self._source_shape = source.shape
        self.positions = wp.empty((source.shape[0], len(records)), dtype=wp.vec3f, device=source.device)
        self.validity = (
            wp.empty(self.positions.shape, dtype=wp.int32, device=source.device)
            if binding_type is DeviceSurfaceBindings else None
        )

    def update(self, model, data, *, refresh=True, stream=None):
        """Update from integrated qpos, without advancing simulation time.

        ``model``/``data`` are matching Warp objects (MJLab's ``wp_model`` and
        ``wp_data``). Kinematics/flex refresh is enabled because ``step`` leaves
        derived geometry at the pre-integration state. Set refresh=False only
        after the host has already refreshed it. Keep all producers/consumers
        on this stream, or insert explicit synchronization in the application.
        """
        if data.flexvert_xpos.shape != self._source_shape:
            raise ValueError("world count or topology changed; rebuild the embedding")
        device = self.positions.device
        if data.flexvert_xpos.device != device:
            raise ValueError("simulation and embedding must use the same device")
        if stream is not None and stream.device != device:
            raise ValueError("stream and embedding must use the same device")
        with wp.ScopedDevice(device):
            if stream is not None:
                with wp.ScopedStream(stream):
                    self._update(model, data, refresh)
            else:
                self._update(model, data, refresh)
        return self.positions

    def _update(self, model, data, refresh):
        if refresh:
            import mujoco_warp as mjw
            mjw.kinematics(model, data)
            mjw.flex(model, data)
        kwargs = dict(source_vertex_offset=self.vertex_offset)
        if self.validity is None:
            enqueue_embedding(data.flexvert_xpos, self.bindings, self.positions, **kwargs)
        else:
            enqueue_surface_embedding(
                data.flexvert_xpos, self.bindings, self.positions, self.validity, **kwargs
            )

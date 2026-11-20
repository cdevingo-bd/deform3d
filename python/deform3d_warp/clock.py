"""Simulator-independent device clock for reproducible control scheduling.

Time is accumulated in compensated float64 from the actual device timestep.
Float32 simulator timestamps are rounded from that time, never accumulated.
The clock owns time: after resetting a host simulation, reset this object too.
"""
import numbers
import numpy as np
import warp as wp

from . import _no_overlap


@wp.kernel(enable_backward=False)
def _write(elapsed: wp.array[wp.float64], status: wp.array[int], output: wp.array[float]):
    w = wp.tid()
    value = wp.float32(elapsed[w])
    if not wp.isfinite(value):
        status[w] = status[w] | 1
    elif status[w] == 0:
        output[w] = value


@wp.kernel(enable_backward=False)
def _advance(timestep: wp.array[float], elapsed: wp.array[wp.float64],
             compensation: wp.array[wp.float64], status: wp.array[int], output: wp.array[float]):
    w = wp.tid()
    dt = wp.float64(timestep[w % timestep.shape[0]])
    old = elapsed[w]
    if dt == wp.float64(0.0):
        # Pausing must preserve compensation too; adding zero through Kahan's
        # recurrence can otherwise consume a pending half-ulp correction.
        rounded_old = wp.float32(old)
        if not wp.isfinite(rounded_old) or not wp.isfinite(compensation[w]):
            status[w] = status[w] | 1
        elif status[w] == 0:
            output[w] = rounded_old
        return
    y = dt - compensation[w]
    updated = old + y
    correction = (updated - old) - y
    rounded = wp.float32(updated)
    if not wp.isfinite(dt) or dt < wp.float64(0.0) or not wp.isfinite(rounded) or not wp.isfinite(correction):
        status[w] = status[w] | 1
    elif status[w] == 0:
        elapsed[w] = updated
        compensation[w] = correction
        output[w] = rounded


class DeviceSimulationClock:
    """One float64 time/compensation pair and sticky status per world (20 bytes).

    Controls can consume ``elapsed`` directly. Call ``enqueue_write`` before
    the first step, then ``enqueue_advance`` exactly once after each successful
    simulated step. A zero timestep pauses a world. Timesteps are float32,
    matching the simulator's actual integration parameter, and may change on
    replay. Invalid/negative steps set bit 1 and leave that world's clock and
    output unchanged; clear/reset explicitly after correcting the caller.

    Setup/reset are host operations. Enqueue performs no allocations, transfers,
    or synchronization. Keep storage alive and use a single ordered stream.
    This reduces cumulative clock error; it does not increase state precision
    or the resolution of a float32 simulator's own time-dependent functions.
    """
    def __init__(self, nworld=1, *, initial_time=0., device='cuda:0'):
        if isinstance(nworld, bool) or not isinstance(nworld, numbers.Integral) or not 1 <= nworld <= 2147483647:
            raise ValueError('positive integer world count required')
        self.nworld, self.device = int(nworld), wp.get_device(device)
        initial = self._times(initial_time)
        self.elapsed = wp.array(initial, dtype=wp.float64, device=self.device)
        self.compensation = wp.zeros(nworld, dtype=wp.float64, device=self.device)
        self.status = wp.zeros(nworld, dtype=int, device=self.device)
        self.workspace_bytes = 20*self.nworld

    def _times(self, value):
        times = np.asarray(value, np.float64)
        if times.ndim == 0:
            times = np.full(self.nworld, times, np.float64)
        if times.shape != (self.nworld,) or not np.isfinite(times).all() or np.any(np.abs(times) > np.finfo(np.float32).max):
            raise ValueError('initial times must be finite and representable as float32, one per world')
        return np.ascontiguousarray(times)

    def reset(self, times=0.):
        initial = self._times(times)
        if self.device.is_cuda and wp.get_stream(self.device).is_capturing:
            raise RuntimeError('reset the clock outside graph capture')
        self.elapsed.assign(initial)
        self.compensation.zero_()
        self.status.zero_()

    def _validate(self, output, timestep=None):
        arrays = [('output', output, wp.float32, (self.nworld,)),
                  ('elapsed', self.elapsed, wp.float64, (self.nworld,)),
                  ('compensation', self.compensation, wp.float64, (self.nworld,)),
                  ('status', self.status, wp.int32, (self.nworld,))]
        if timestep is not None:
            if not isinstance(timestep, wp.array) or timestep.shape not in ((1,), (self.nworld,)):
                raise ValueError('timestep must contain one value or one per world')
            arrays.append(('timestep', timestep, wp.float32, timestep.shape))
        for i, (name, array, dtype, shape) in enumerate(arrays):
            if (not isinstance(array, wp.array) or array.dtype != dtype or array.device != self.device
                or not array.is_contiguous or array.shape != shape
                or array.ptr % wp.types.type_size_in_bytes(dtype)):
                raise ValueError(name+' has invalid dtype, shape, alignment or device')
            for _, other, _, _ in arrays[:i]:
                _no_overlap(array, other, 'clock storage, output and timestep must not overlap')

    def enqueue_write(self, output):
        self._validate(output)
        wp.launch(_write, self.nworld, inputs=[self.elapsed, self.status, output], device=self.device)

    def enqueue_advance(self, timestep, output):
        self._validate(output, timestep)
        wp.launch(_advance, self.nworld,
                  inputs=[timestep, self.elapsed, self.compensation, self.status, output], device=self.device)

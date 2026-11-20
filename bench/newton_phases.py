"""CUDA phase costs for a frozen Newton system; no simulation is advanced."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from deform3d_warp.mujoco_newton import _Workspace, _kernels, _transport_kernels, _register_diagonal


def run(args):
    with np.load(args.system) as archive:
        a, b = archive['matrix'], archive['rhs']
    worlds, nv = b.shape
    with wp.ScopedDevice('cuda:0'):
        ctx = SimpleNamespace(h=wp.array(a.astype(np.float32)), grad=wp.array(b.astype(np.float32)),
                              search=wp.zeros((worlds, nv)), search_dot=wp.zeros(worlds),
                              newton_decrement=wp.zeros(worlds), done=wp.zeros(worlds, dtype=wp.bool))
        ws = _Workspace(nv, worlds, 'cuda:0', args.precision, args.tile, args.diagonal)
        count = wp.ones(worlds, dtype=wp.int32)
        ws.enqueue(ctx, count, count)
        diagonal, panel, trailing, substitute, finish = _kernels(nv, args.precision, args.tile)
        diagonal_threads = 64
        if args.diagonal == 'register':
            diagonal = _register_diagonal(args.precision,args.tile)
            diagonal_threads = 32
        copy_upper, rhs = _transport_kernels(args.precision)

        def factor():
            wp.launch(copy_upper, (worlds, ws.size, ws.size), [nv, ctx.h, ws.refactor, ws.work])
            for k in range(0, ws.size, ws.tile_size):
                wp.launch_tiled(diagonal, dim=worlds, inputs=[k, ws.refactor, ws.work], block_dim=diagonal_threads)
                remaining = (ws.size-k)//ws.tile_size-1
                if remaining:
                    wp.launch_tiled(panel, dim=(worlds, remaining), inputs=[k, ws.refactor, ws.work], block_dim=64)
                    wp.launch_tiled(trailing, dim=(worlds, remaining, remaining), inputs=[k, ws.refactor, ws.work], block_dim=64)

        def solve():
            wp.launch(rhs, (worlds, ws.size), [nv, ws.active, ctx.grad, ws.x])
            wp.launch_tiled(substitute, dim=worlds, inputs=[ws.active,ws.work,ws.x_tiles], block_dim=64)

        def check():
            wp.launch_tiled(finish, dim=worlds, inputs=[ws.active, ctx.h, ctx.grad, ws.x, ws.work,
                ctx.search, ctx.search_dot, ctx.newton_decrement, ws.status, ws.max_relative_residual],
                block_dim=max(32, 1 << (nv-1).bit_length()))

        records = {}
        for name, fn in [('full', lambda: ws.enqueue(ctx, count, count)),
                         ('copy_and_factor', factor), ('rhs_and_substitute', solve), ('residual_and_summary', check)]:
            fn()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                for _ in range(args.repeat):
                    fn()
            records[name] = (capture.graph, [])
        start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for sample in range(args.samples):
            for name in list(records)[::1 if sample % 2 == 0 else -1]:
                graph, values = records[name]
                wp.record_event(start)
                wp.capture_launch(graph)
                wp.record_event(end)
                values.append(wp.get_event_elapsed_time(start, end) / args.repeat)
        x = -ctx.search.numpy().astype(np.float64)
        relative = np.linalg.norm((a @ x[..., None])[..., 0] - b, axis=1) / np.linalg.norm(b, axis=1)
        np.testing.assert_array_equal(ws.status.numpy(), 0)
        if not np.isfinite(relative).all() or np.any(relative > 1e-4):
            raise RuntimeError('independent frozen-system residual failed')
        report = dict(nv=nv, worlds=worlds, precision=args.precision, tile=args.tile, diagonal=args.diagonal,
                      samples=args.samples, repeat=args.repeat, gpu=wp.get_device().name,
                      scope='frozen matrix; CUDA graphs timed independently; no physics/assembly/rendering',
                      relative_residual=relative.tolist(),
                      phases_cuda_ms={name: dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)))
                                      for name, (_, values) in records.items()})
        if args.detail_factor:
            events = []
            def marked(name, fn):
                begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                wp.record_event(begin)
                fn()
                wp.record_event(end)
                events.append((name, begin, end))
            with wp.ScopedCapture() as capture:
                wp.launch(copy_upper, (worlds, ws.size, ws.size), [nv, ctx.h, ws.refactor, ws.work])
                for k in range(0, ws.size, ws.tile_size):
                    marked('diagonal', lambda: wp.launch_tiled(diagonal, dim=worlds,
                        inputs=[k, ws.refactor, ws.work], block_dim=diagonal_threads))
                    remaining = (ws.size-k)//ws.tile_size-1
                    if remaining:
                        marked('panel', lambda: wp.launch_tiled(panel, dim=(worlds, remaining),
                            inputs=[k, ws.refactor, ws.work], block_dim=64))
                        marked('trailing', lambda: wp.launch_tiled(trailing, dim=(worlds, remaining, remaining),
                            inputs=[k, ws.refactor, ws.work], block_dim=64))
            groups = {name: [] for name in ('diagonal', 'panel', 'trailing')}
            for _ in range(args.samples):
                wp.capture_launch(capture.graph)
                sample = dict.fromkeys(groups, 0.)
                for name, begin, end in events:
                    sample[name] += wp.get_event_elapsed_time(begin, end)
                for name, value in sample.items():
                    groups[name].append(value)
            report['factor_instrumented_cuda_ms'] = {
                name: dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)))
                for name, values in groups.items()}
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('system', type=Path)
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--tile', type=int, choices=[16, 32], default=32)
    p.add_argument('--diagonal', choices=['tiled','register'], default='tiled')
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--repeat', type=int, default=10)
    p.add_argument('--detail-factor', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.samples, args.repeat) < 1:
        p.error('samples and repeat must be positive')
    run(args)

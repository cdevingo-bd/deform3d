"""Render the measured solver speed and the still-failing trajectory gate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    previous = read('mjwarp_strong_cooperative.json')
    first = read('mjwarp_strong_parallel_newton_0.5.json')
    final = read('mjwarp_strong_parallel_newton_final_0.5.json')
    linear = read('newton_linear_0.5.json')['results']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    fig.suptitle('deform3d 0.5 · faster Newton solve; trajectory parity still fails',
                 fontsize=16, fontweight='bold')
    colors = ['#758498', '#287cba', '#168574']
    names = ['Upstream', 'Parallel FP32', 'Parallel FP64']
    values = [linear[k]['cuda_ms']['median'] for k in
              ('upstream', 'parallel_float32_tile32', 'parallel_float64_tile16')]
    axes[0].bar(names, values, color=colors)
    axes[0].set(title='Identical 456-DOF contact Hessian', ylabel='Median linear solve (ms)', ylim=(0, 9))
    for i, value in enumerate(values):
        axes[0].text(i, value + .15, f'{value:.2f}', ha='center')
    axes[0].text(.02, .98, 'Assembly, physics and rendering excluded',
                 transform=axes[0].transAxes, va='top', fontsize=9)

    runs = [previous, first, final]
    values = [r['host_step_ms']['p50'] for r in runs]
    p95 = [r['host_step_ms']['p95'] for r in runs]
    axes[1].bar(['0.4', '0.5 first', '0.5 final'], values, color=colors)
    axes[1].scatter(range(3), p95, color='#40464e', marker='_', s=160, label='p95')
    axes[1].axhline(1, color='#b6403b', linestyle='--', linewidth=1, label='Real-time budget: 1 ms')
    axes[1].set(title='Strong squeeze: full 3-second cycle', ylabel='Wall time per 1 ms step (ms)', ylim=(0, 78))
    for i, value in enumerate(values):
        axes[1].text(i, value + 1.0, f'{value:.2f}', ha='center')
    axes[1].legend(frameon=False, fontsize=9, loc='upper right')

    for run, color, label in ((first, colors[1], '0.5 first'), (final, colors[2], '0.5 final')):
        samples = run['checkpoints']
        axes[2].plot([s['time_s'] for s in samples],
                     [s['max_flex_delta_m'] * 1000 for s in samples], color=color, label=label, linewidth=1)
    axes[2].axhline(5, color='#b6403b', linestyle='--', label='Acceptance: 5 mm')
    axes[2].axvspan(.7, 2.5, color='#c8cdd3', alpha=.16)
    axes[2].set(title='Maximum CPU/GPU flex difference', xlabel='Simulation time (s)', ylabel='Position difference (mm)', ylim=(0, 27))
    axes[2].legend(frameon=False, fontsize=9, loc='upper right')
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.supxlabel('Quadro P2000 · shared desktop · full-step timings exclude CPU reference, readbacks and rendering', fontsize=10)
    fig.savefig(RESULTS / 'solver_progress_0.5.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

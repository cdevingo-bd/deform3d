"""Reproduce the 0.6 numerical-validation figure from raw measured reports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = [
        ('GPU CCD 1e-6', 'mjwarp_ccd_full_0.6'),
        ('GPU CCD 1e-8', 'mjwarp_tight_ccd_full_0.6'),
        ('Repeat / video', 'mjwarp_showcase_0.6'),
        ('Two worlds', 'mjwarp_tight_ccd_two_world_0.6'),
        ('Local contacts', 'mjwarp_local_ccd_full_0.6'),
        ('Two worlds / ALL', 'mjwarp_deterministic_showcase_0.6')]
    data = [json.loads((ROOT/'results'/f'{name}.json').read_text()) for _, name in cases]
    convergence = json.loads((ROOT/'results/cpu_contact_convergence_0.6.json').read_text())
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    figure.suptitle('deform3d 0.6  |  Contact accuracy and its remaining cost', fontsize=19, weight='bold')
    cpu = convergence['results'][:-1]
    ax = axes[0, 0]
    values = [d['max_flex_delta_m']*1000 for d in cpu]
    ax.barh(range(len(cpu)), values, color=['#be5d36', '#be5d36', '#21846b'])
    ax.set_yticks(range(len(cpu)), [f"CPU tolerance {d['ccd_tolerance']:.0e}" for d in cpu])
    ax.set_xscale('log'); ax.set_xlim(.005, 45)
    ax.set_xlabel('Maximum shape difference vs CPU 1e-10 (mm, log scale)')
    ax.set_title('The previous CPU baseline was not converged', loc='left')
    for y, value in enumerate(values): ax.text(value*1.1, y, f'{value:.3f}', va='center')
    for ax, key, scale, gate, title, xlabel in [
        (axes[0, 1], 'max_flex_cpu_delta_m', 1000, 5, 'Full 3 s GPU cycles: shape', 'Maximum difference from converged CPU (mm)'),
        (axes[1, 0], 'pad_force_relative_rms_delta', 100, 10, 'Full 3 s GPU cycles: gripper force', 'Relative RMS signed pad-force difference (%)')]:
        vals = [d[key]*scale for d in data]
        ax.barh(np.arange(len(cases)), vals, color=['#21846b' if v <= gate else '#be5d36' for v in vals])
        ax.axvline(gate, color='#243647', linestyle='--', label=f'Unchanged gate: {gate:g}')
        ax.set_yticks(np.arange(len(cases)), [label for label, _ in cases]); ax.invert_yaxis()
        ax.set_xlim(0, max(vals)*1.2); ax.set_xlabel(xlabel+f'\nDashed line: unchanged gate {gate:g}'); ax.set_title(title, loc='left')
        for y, val in enumerate(vals): ax.text(val+max(vals)*.02, y, f'{val:.2f}', va='center', fontsize=10)
    ax = axes[1, 1]
    vals = [d['cuda_step_ms']['p50'] for d in data]
    ax.barh(np.arange(len(cases)), vals, color='#326d91')
    ax.set_yticks(np.arange(len(cases)), [label for label, _ in cases]); ax.invert_yaxis()
    ax.set_xlim(0, max(vals)*1.25)
    ax.axvline(1, color='#243647', linestyle='--', label='1 ms simulation step')
    for y, val in enumerate(vals): ax.text(val+max(vals)*.02, y, f'{val:.2f}', va='center', fontsize=10)
    ax.set_title('Quadro P2000: full graph latency', loc='left')
    ax.set_xlabel('Median CUDA ms per step (both worlds where specified)\nDashed line: 1 ms simulation step')
    figure.supxlabel('Every-step diagnostics; 456 DOFs and 34,272 dense vertices per world.\n'
                        'Timings exclude setup, readback and rendering. Materials uncalibrated.', fontsize=10)
    figure.savefig(ROOT/'results/contact_validation_0.6.png', dpi=180)
    figure.savefig(ROOT/'results/contact_validation_0.6.svg')


if __name__ == '__main__': main()

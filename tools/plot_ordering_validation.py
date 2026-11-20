"""Plot 0.7 ordering and solver evidence from the unmodified measured reports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(name):
        return json.loads((ROOT/'results'/f'{name}.json').read_text())
    prior = read('mjwarp_deterministic_showcase_0.6')
    fast = read('mjwarp_canonical_fast_0.7')
    deterministic = read('mjwarp_canonical_deterministic_0.7')
    double = read('mjwarp_canonical_double_0.7')
    linear = read('newton_linear_0.7')['results']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(14,9),layout='constrained')
    fig.suptitle('deform3d 0.7  |  Canonical constraints and measured solver cost',fontsize=18,weight='bold')
    ax = axes[0,0]
    for r,label,color in ((fast,'1 world / ordinary atomics','#bd6b3c'),
                          (deterministic,'2 worlds / deterministic FP32','#286d93'),
                          (double,'2 worlds / deterministic FP64','#29856b')):
        ax.plot([p['time_s'] for p in r['checkpoints']],
                [p['max_flex_delta_m']*1000 for p in r['checkpoints']],label=label,color=color,lw=1.4)
    ax.axhline(5,ls='--',color='#303b46',label='Unchanged 5 mm gate')
    ax.set_title('Shape difference from the converged CPU reference',loc='left')
    ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Maximum vertex difference (mm)');ax.legend(fontsize=9)
    ax = axes[0,1]
    values = [prior['max_identical_world_flex_spread_m']*1000,
              deterministic['max_identical_world_flex_spread_m']*1000,
              double['max_identical_world_flex_spread_m']*1000]
    ax.barh([0,1,2],values,color=['#bd6b3c','#286d93','#29856b'])
    ax.set_yticks([0,1,2],['0.6 / upstream flags','0.7 / canonical FP32','0.7 / canonical FP64']);ax.invert_yaxis()
    ax.set_xlim(0,max(1.,max(values))*1.3)
    for y,v in enumerate(values):ax.text(v+max(1.,max(values))*.02,y,f'{v:.6g} mm',va='center')
    ax.set_title('Full-cycle difference between the two GPU worlds',loc='left')
    ax.set_xlabel('Maximum flex-vertex spread (mm)\nAll use ALL and bounded scratch; all 3,000 steps checked')
    ax = axes[1,0]
    names = ['parallel_float32_tile32','register_float32_tile32']
    values = [linear[n]['cuda_ms']['median'] for n in names]
    ax.barh([0,1],values,color=['#737e89','#286d93']);ax.invert_yaxis()
    ax.set_yticks([0,1],['Tiled diagonal','Register diagonal']);ax.set_xlim(0,max(values)*1.25)
    for y,v in enumerate(values):ax.text(v+.04,y,f'{v:.3f} ms',va='center')
    ax.set_title('Identical frozen 456-DOF Newton system',loc='left')
    ax.set_xlabel('Complete linear solve: median CUDA ms\nIncludes residual monitoring; excludes physics and assembly')
    ax = axes[1,1]
    names = ['Fast FP32 / 1 world','ALL FP32 / 2 worlds','ALL FP64 / 2 worlds']
    values = [fast['cuda_step_ms']['p50'],deterministic['cuda_step_ms']['p50'],double['cuda_step_ms']['p50']]
    ax.barh([0,1,2],values,color=['#bd6b3c','#286d93','#29856b']);ax.invert_yaxis()
    ax.set_yticks([0,1,2],names);ax.set_xlim(0,max(values)*1.25)
    for y,v in enumerate(values):ax.text(v+max(values)*.02,y,f'{v:.2f} ms',va='center')
    ax.axvline(1,color='#303b46',ls='--')
    ax.set_title('Full graph remains slower than real time',loc='left')
    ax.set_xlabel('Median CUDA ms per 1 ms simulated step\nController + physics + dense geometry + normals; no rendering')
    fig.supxlabel('Quadro P2000, MuJoCo/MJWarp 4.0.1, Warp 1.19.0. Zero world spread is separate from CPU parity.\n'
                  'All three new full-cycle configurations fail overall shape/force acceptance. Materials are uncalibrated.',fontsize=10)
    fig.savefig(ROOT/'results/ordering_validation_0.7.png',dpi=180)
    fig.savefig(ROOT/'results/ordering_validation_0.7.svg')


if __name__ == '__main__':
    main()

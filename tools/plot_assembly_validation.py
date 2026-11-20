"""Scientific figure from recorded 0.8 assembly and trajectory measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'


def main():
    reports={name:json.loads((R/file).read_text()) for name,file in {
        '0.7 deterministic':'mjwarp_canonical_deterministic_0.7.json',
        '0.8 deterministic':'mjwarp_owned_deterministic_0.8.json',
        '0.8 ordinary atomics':'mjwarp_owned_fast_0.8.json'}.items()}
    micro=json.loads((R/'contact_assembly_0.8.json').read_text())['results']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.4),layout='constrained')
    colors=['#8796a5','#098977','#b87012']
    for (name,r),color in zip(reports.items(),colors):
        points=r['checkpoints']
        axes[0,0].plot([p['time_s'] for p in points],
            [1000*p['max_flex_delta_m'] for p in points],label=name,color=color,lw=1.15,alpha=.95)
    axes[0,0].axhline(5,color='#b52c36',ls='--',lw=1.4,label='5 mm gate')
    axes[0,0].set(title='Full-cycle mesh agreement',xlabel='Simulation time (s)',ylabel='Maximum flex-position difference (mm)',ylim=(0,None))
    axes[0,0].legend(fontsize=8,loc='upper left')
    r=reports['0.8 deterministic']; points=r['checkpoints']
    t=[p['force_time_s'] for p in points]
    axes[0,1].plot(t,[abs(p['cpu_pad_forces_N'][1]) for p in points],color='#26384b',label='CPU MuJoCo',lw=1.5)
    axes[0,1].plot(t,[abs(p['gpu_pad_forces_N'][0][1]) for p in points],color=colors[1],label='0.8 GPU, world 0',lw=1,alpha=.8)
    axes[0,1].set(title=f"Right pad force; both-pad RMS error {100*r['pad_force_relative_rms_delta']:.2f}%",xlabel='Simulation time (s)',ylabel='Magnitude of net axial pad force (N)')
    axes[0,1].legend(fontsize=9)
    keys=['atomic','sorted_deterministic','owned_with_prepare','owned_reuse']
    vals=[micro[k]['cuda_median_ms'] for k in keys]
    axes[1,0].barh(['Ordinary atomics','Sorted deterministic','Owned, including setup','Owned, reuse'],vals,color=[colors[0],'#617084',colors[1],'#59b5a5'])
    axes[1,0].invert_yaxis(); axes[1,0].set_xscale('log')
    axes[1,0].set(title='Same frozen contact operators: two worlds',xlabel='Median CUDA assembly time (ms, logarithmic)')
    axes[1,0].set_xlim(.04,32)
    for i,v in enumerate(vals):axes[1,0].text(v*1.1,i,f'{v:.3f}',va='center',fontsize=9)
    labels=[];steps=[]
    for name,r in reports.items():
        labels.append(f"{name}\n{r['worlds']} world"+('s' if r['worlds']>1 else '')
                      + ('\nComparison PASS' if r['validation_passed'] else '\nComparison FAIL'))
        steps.append(r['cuda_step_ms']['p50'])
    axes[1,1].bar(labels,steps,color=colors,width=.6)
    for i,v in enumerate(steps):axes[1,1].text(i,v+2,f'{v:.2f} ms',ha='center',fontsize=10)
    axes[1,1].axhline(1,color='#b52c36',ls='--',lw=1,label='1 ms real-time budget')
    axes[1,1].set(title='Evolving full frame: 1 ms of simulation',ylabel='Median CUDA frame time (ms)',ylim=(0,max(steps)*1.22))
    axes[1,1].tick_params(axis='x',labelsize=8); axes[1,1].legend(fontsize=8,loc='upper right')
    for ax in axes.flat: ax.grid(alpha=.16); ax.set_axisbelow(True)
    fig.suptitle('deform3d 0.8  |  Output-owned GPU contact assembly',fontsize=19,fontweight='bold')
    fig.supxlabel('Quadro P2000 • 456 physical DOFs • uncalibrated materials • frame timings exclude rendering and diagnostic readback',fontsize=9)
    fig.savefig(R/'assembly_validation_0.8.png',dpi=180)
    fig.savefig(R/'assembly_validation_0.8.svg')


if __name__=='__main__':main()

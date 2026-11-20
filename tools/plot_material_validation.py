"""Scientific performance and trajectory figure from retained measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'


def main():
    old=json.loads((R/'mjwarp_owned_deterministic_0.8.json').read_text())
    new=json.loads((R/'mjwarp_material_deterministic_0.9.json').read_text())
    failed=json.loads((R/'mjwarp_material_fp64_prototype_0.9.json').read_text())
    micro=json.loads((R/'material_operators_0.9.json').read_text())['results']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(13,8.4),layout='constrained')
    colors=['#687b92','#be791b','#078b78']
    reports=[old,failed,new]
    labels=['0.8 accepted','FP64 prototype\nshape gate FAILED','0.9 binned gather']
    times=[r['cuda_step_ms']['p50'] for r in reports]
    ax[0,0].bar(labels,times,color=colors,width=.6)
    for i,t in enumerate(times):ax[0,0].text(i,t+1.2,f'{t:.2f} ms',ha='center')
    ax[0,0].axhline(1,color='#b42d3b',ls='--',label='Real-time budget: 1 ms')
    ax[0,0].set(title='Full simulation + dense mesh: two worlds',ylabel='Median CUDA frame time (ms)',ylim=(0,max(times)*1.18))
    ax[0,0].legend(fontsize=8)
    phases=['stiffness_build','passive_forces','force_shift']; xs=np.arange(3)
    for offset,mode,color,label in [(-.2,'sorted_deterministic',colors[0],'Upstream deterministic'),(.2,'owned_deterministic',colors[2],'Owned, same binned arithmetic')]:
        vals=[micro[mode+'/'+p]['cuda_median_ms'] for p in phases]
        ax[0,1].bar(xs+offset,vals,width=.36,color=color,label=label)
        for i,v in enumerate(vals):ax[0,1].text(i+offset,v*1.12,f'{v:.3f}',ha='center',fontsize=8)
    ax[0,1].set_yscale('log'); ax[0,1].set_xticks(xs,['Stiffness build','Passive forces','Force shift'])
    ax[0,1].set(title='Paired operators at the same compressed state',ylabel='Median CUDA time (ms, logarithmic)')
    ax[0,1].legend(fontsize=8,loc='upper left'); ax[0,1].margins(y=.25)
    for r,color,label in [(failed,colors[1],'Rejected FP64 prototype'),(new,colors[2],'Accepted binned gather')]:
        points=r['checkpoints']; t=[p['time_s'] for p in points]
        ax[1,0].plot(t,[1000*p['max_flex_delta_m'] for p in points],color=color,label=label,lw=1.1)
    ax[1,0].axhline(5,color='#b42d3b',ls='--',label='Unchanged 5 mm gate')
    ax[1,0].set(title='Shape comparison across squeeze and release',xlabel='Simulated time (s)',ylabel='Maximum flex-position difference (mm)')
    ax[1,0].legend(fontsize=8)
    points=new['checkpoints'];t=[p['force_time_s'] for p in points]
    ax[1,1].plot(t,[abs(p['cpu_pad_forces_N'][1]) for p in points],color='#26384b',label='CPU MuJoCo',lw=1.4)
    ax[1,1].plot(t,[abs(p['gpu_pad_forces_N'][0][1]) for p in points],color=colors[2],label='0.9 GPU',lw=1,alpha=.8)
    ax[1,1].set(title=f"Right pad force; both-pad RMS difference {100*new['pad_force_relative_rms_delta']:.2f}%",xlabel='Simulated time (s)',ylabel='Axial pad-force magnitude (N)')
    ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.grid(alpha=.15);a.set_axisbelow(True)
    fig.suptitle('deform3d 0.9  |  Deterministic material assembly without global sorting',fontsize=18,fontweight='bold')
    fig.supxlabel('Quadro P2000 • 456 physical DOFs • 1 ms timestep • two worlds • materials uncalibrated • rendering excluded',fontsize=9)
    fig.savefig(R/'material_validation_0.9.png',dpi=180)
    fig.savefig(R/'material_validation_0.9.svg')


if __name__=='__main__': main()

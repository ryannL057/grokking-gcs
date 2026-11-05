"""Run archived plotting code against a temporary copy of verified inputs."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


FIGURES = {
    'fig01':('figure1.py',[],'figure1_path_coherence'),
    'fig02':('figure2.py',['STAGE'],'depth_coherence'),
    'fig03':('figure3.py',[],'functional_compatibility_figure_tminus'),
    'fig04':('appendix.py',['STAGE','probe_positions'],'probe_positions_readable'),
    'fig05':('appendix.py',['STAGE','reference_trajectories'],'reference_trajectories'),
    'fig09':('appendix.py',['STAGE','detached_attention'],'detached_attention'),
    'fig10':('appendix.py',['STAGE','cohort_mlp'],'cohort_mlp'),
    'fig11':('appendix.py',['STAGE','cohort_attention'],'cohort_attention'),
    'fig12':('appendix.py',['STAGE','attention_controls'],'attention_controls_readable'),
    'fig13':('appendix.py',['STAGE','attention_maps'],'attention_maps'),
    'fig14':('appendix.py',['STAGE','matched_mlp'],'matched_mlp'),
}


def render(target: str,ref: Path,output: Path) -> list[str]:
    if target in {'fig06', 'fig07'}:
        from .ablations import render as render_ablation
        return render_ablation(target,ref,output)
    if target == 'fig08':
        return render_neighbors(ref,output)
    script,args,stem=FIGURES[target]
    with tempfile.TemporaryDirectory(prefix='gcs-render-') as temp:
        stage=Path(temp)
        shutil.copytree(ref/'source_data',stage/'source_data')
        (stage/'figures/reviewed').mkdir(parents=True)
        (stage/'audit').mkdir()
        env=dict(os.environ,GCS_RENDER_ROOT=str(stage),MPLBACKEND='Agg',MPLCONFIGDIR=str(stage/'mpl'))
        run=subprocess.run([sys.executable,str(Path(__file__).parent/'plots'/script),
                            *[str(stage) if a=='STAGE' else a for a in args]],
                           env=env,capture_output=True,text=True)
        (output/f'{target}.log').write_text(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(f'{target} plotting failed: see {output/str(target+".log")}\n{run.stderr[-1500:]}')
        paths=[]
        for file in (stage/'figures').rglob(stem+'.*'):
            dest=output/(target+file.suffix)
            shutil.copyfile(file,dest);paths.append(dest.name)
        if not paths:raise RuntimeError(f'No output produced for {target}')
        return sorted(paths)


def render_neighbors(ref: Path,output: Path) -> list[str]:
    import numpy as np
    os.environ['MPLBACKEND']='Agg'
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'gcs-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=np.load(ref/'source_data/reviewed_appendix_figures/neighbor_count.npz',allow_pickle=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,path in zip(axes,['overall','attention','mlp']):
        for k in data['param_values']:
            raw=data[f'k{k}_{path}_raw']
            y=np.zeros_like(raw) if np.ptp(raw)<1e-10 else (raw-raw.min())/np.ptp(raw)
            np.testing.assert_allclose(y,data[f'k{k}_{path}_norm'],atol=1e-12)
            ax.plot(data[f'k{k}_{path}_epochs'],y,label=f'k={k}')
        ax.set(xlabel='Epoch',ylabel='Scaled GCS',title=path.capitalize())
    axes[-1].legend(frameon=False)
    for suffix in ['pdf','png','svg']:fig.savefig(output/f'fig08.{suffix}',dpi=180)
    plt.close(fig)
    return [f'fig08.{suffix}' for suffix in ['pdf','png','svg']]

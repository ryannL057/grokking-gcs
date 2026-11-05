#!/usr/bin/env python3
"""Run a preserved full-experiment protocol with an explicit data root."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
runners=json.loads((root/'manifests/experiment_runners.json').read_text())['runners']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--list',action='store_true')
p.add_argument('--data-root',type=Path)
p.add_argument('runner',nargs='?',choices=runners)
p.add_argument('arguments',nargs=argparse.REMAINDER)
a=p.parse_args()
if a.list:
    print('\n'.join(runners));raise SystemExit(0)
if not a.runner or not a.data_root:p.error('runner and --data-root are required')
data=a.data_root.resolve()
if data.is_relative_to((root/'results/reference').resolve()):p.error('Experiment data-root must not be immutable reference data')
data.mkdir(parents=True,exist_ok=True)
arguments=a.arguments[1:] if a.arguments and a.arguments[0]=='--' else a.arguments
env=dict(os.environ,GCS_DATA_ROOT=str(data),MPLBACKEND='Agg',
         MPLCONFIGDIR=str(Path(tempfile.gettempdir())/'gcs-experiment-mpl'))
raise SystemExit(subprocess.call([sys.executable,'-m',f'gcs.legacy.{a.runner}',*arguments],env=env,cwd=data))

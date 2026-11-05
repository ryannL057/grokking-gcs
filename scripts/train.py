#!/usr/bin/env python3
"""Train one explicit configuration, or a finite named primary run."""
import argparse
import json
from pathlib import Path

from gcs.runner import train

parser=argparse.ArgumentParser(description=__doc__)
choice=parser.add_mutually_exclusive_group(required=True)
choice.add_argument('--config',type=Path)
choice.add_argument('--run',help='Stable ID from manifests/primary_runs.json')
choice.add_argument('--list',action='store_true')
parser.add_argument('--output',type=Path)
parser.add_argument('--device',default='cpu')
parser.add_argument('--dry-run',action='store_true',help='Print the resolved configuration and schedule without training')
parser.add_argument('--threads',type=int,default=1)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
runs=json.loads((root/'manifests/primary_runs.json').read_text())
if args.list:
    for row in runs: print(row['run_id'],row['horizon'])
else:
    if args.output is None and not args.dry_run: parser.error('--output is required')
    if args.run:
        matched=[r for r in runs if r['run_id']==args.run]
        if not matched: parser.error(f'Unknown run: {args.run}')
        config_path=root/matched[0]['config']
    else: config_path=args.config
    config=json.loads(config_path.read_text())
    if args.dry_run:
        print(json.dumps({'config':config,'schedule':matched[0] if args.run else None,'device':args.device},indent=2))
    else:
        import torch
        torch.set_num_threads(args.threads)
        if args.run:
            from gcs.primary import train_primary
            result=train_primary(config,matched[0],args.output,device=args.device)
        else:
            result=train(config,args.output,device=args.device)
        print(json.dumps(result,indent=2))

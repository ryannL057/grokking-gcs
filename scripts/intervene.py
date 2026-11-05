#!/usr/bin/env python3
"""Recompute immediate attention and MLP checkpoint-reset damage."""
import argparse
import json
from pathlib import Path
import torch
from gcs.runner import reset_experiment

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--host',type=Path,required=True)
p.add_argument('--donor',type=Path,required=True)
p.add_argument('--config',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--threads',type=int,default=1)
a=p.parse_args();torch.set_num_threads(a.threads)
print(json.dumps(reset_experiment(a.host,a.donor,a.output,config=json.loads(a.config.read_text()) if a.config else None,device=a.device),indent=2))

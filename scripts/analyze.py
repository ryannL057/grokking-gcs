#!/usr/bin/env python3
"""Compute three-path GCS using an explicit fixed reference."""
import argparse
import json
from pathlib import Path
import torch
from gcs.runner import analyze

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--config',type=Path)
r=p.add_mutually_exclusive_group(required=True)
r.add_argument('--geometry',type=Path,help='Saved numerical reference geometry (.npz)')
r.add_argument('--reference',type=Path,help='Reference checkpoint for building a new geometry')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--threads',type=int,default=1)
p.add_argument('--probe',choices=['final','operands','all'],default='final')
p.add_argument('--alignment',choices=['cosine','procrustes','rank_permuted'],default='cosine')
a=p.parse_args();torch.set_num_threads(a.threads)
print(json.dumps(analyze(a.checkpoint,a.output,config=json.loads(a.config.read_text()) if a.config else None,
      reference=a.reference,geometry_path=a.geometry,device=a.device,token_probe=a.probe,alignment=a.alignment),indent=2))

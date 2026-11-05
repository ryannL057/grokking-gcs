#!/usr/bin/env python3
"""Continue a fixed-component arm with the paper's matched additional budget."""
import argparse,json
from pathlib import Path
import torch
from gcs.runner import continue_fixed

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=['VV','VL','LV','LL','0L','0V','0V_long','V_unfrozen','L_attn_frozen'],required=True)
p.add_argument('--trough',type=Path,required=True)
p.add_argument('--late',type=Path,required=True,help='t-minus checkpoint')
p.add_argument('--initial',type=Path,help='True pre-update initialization; required for 0 arms')
p.add_argument('--final-epoch',type=int,required=True)
p.add_argument('--t-minus',type=int,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--threads',type=int,default=1)
a=p.parse_args();torch.set_num_threads(a.threads)
budget=a.final_epoch+1-a.t_minus
if budget<=0:p.error('final-epoch + 1 must exceed t-minus')
if a.arm.startswith('0') and a.initial is None:p.error('0 arms require --initial (strictly pre-update weights)')
if a.arm=='V_unfrozen':host,donor,component=a.trough,None,'none'
elif a.arm=='L_attn_frozen':host,donor,component=a.late,None,'attention'
else:
    hosts={'V':a.trough,'L':a.late};donors={'0':a.initial,'V':a.trough,'L':a.late}
    host,donor,component=hosts[a.arm[1]],donors[a.arm[0]],'mlp'
result=continue_fixed(host,donor,a.output,component=component,updates=budget*(3 if a.arm=='0V_long' else 1),
    config=json.loads(a.config.read_text()),device=a.device,stop_after_success=1000 if a.arm=='0V_long' else None)
print(json.dumps({k:v for k,v in result.items() if k!='curve'},indent=2))

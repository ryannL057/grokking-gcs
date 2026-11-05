"""Registry-driven reproduction with explicit incomplete/failure reporting."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import pandas as pd

from ..artifacts import require_inputs,sha256
from .render import render
from .tables import table


def main(root: Path,argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list',action='store_true')
    mode.add_argument('--all',action='store_true')
    mode.add_argument('--target',action='append',help='Repeat to select several stable paper IDs')
    p.add_argument('--output',type=Path,default=Path('outputs/reproduction'))
    args=p.parse_args(argv)
    targets=json.loads((root/'manifests/paper_map.json').read_text())['targets']
    if args.list:
        for item in targets:print(f"{item['id']:8} {item['status']:25} {item['title']}")
        return 0
    names={t['id'] for t in targets}
    if args.target and set(args.target)-names:p.error(f'Unknown targets: {sorted(set(args.target)-names)}')
    selected=targets if args.all else [t for t in targets if t['id'] in args.target]
    output=args.output.resolve()
    if output.is_relative_to((root/'results/reference').resolve()):p.error('Output must not be inside immutable reference inputs')
    output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for item in selected:
        started=time.monotonic();ident=item['id']
        record={'target':ident,'title':item['title']}
        try:
            if item['status']=='missing_original_input':raise FileNotFoundError(item['gap'])
            require_inputs(root,item['inputs'])
            if ident.startswith('fig'):
                paths=render(ident,root/'results/reference',output)
            elif item['kind']=='appendix_analysis':
                from .appendix import reproduce
                result=reproduce(ident,root/'results/reference')
                (output/f'{ident}.json').write_text(json.dumps(_json_clean(result),indent=2,default=_json_default,allow_nan=False)+'\n')
                paths=[f'{ident}.json']
            else:
                rows,result=table(int(ident[3:]),root)
                (output/f'{ident}.json').write_text(json.dumps(_json_clean(result),indent=2,default=_json_default,allow_nan=False)+'\n')
                pd.DataFrame(rows).to_csv(output/f'{ident}.csv',index=False)
                paths=[f'{ident}.json',f'{ident}.csv']
            record.update(status='passed',outputs={name:sha256(output/name) for name in paths})
            print(ident,'passed',flush=True)
        except Exception as error:
            record.update(status='incomplete' if item['status']=='missing_original_input' else 'failed',error=str(error))
            print(ident,record['status'],str(error)[:300],flush=True)
        record['seconds']=round(time.monotonic()-started,3);reports.append(record)
    report={'paper_sha256':json.loads((root/'manifests/paper_map.json').read_text())['paper_sha256'],
            'complete':all(r['status']=='passed' for r in reports),'targets':reports}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['complete'] else 1


def _json_default(value):
    if hasattr(value,'item'):return value.item()
    raise TypeError(f'Not JSON serializable: {type(value)}')


def _json_clean(value):
    if isinstance(value,dict):return {k:_json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

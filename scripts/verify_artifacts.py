#!/usr/bin/env python3
"""Verify every bundled numerical input, configuration and example checkpoint."""
import json
from pathlib import Path
from gcs.artifacts import verify

root=Path(__file__).resolve().parents[1]
records=json.loads((root/'manifests/artifacts.json').read_text())['artifacts']
errors=verify(root,records)
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Verified {len(records)} artifacts ({sum(r["size"] for r in records):,} bytes).')

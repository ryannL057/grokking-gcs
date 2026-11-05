"""Verify immutable input records without importing model dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, records: list[dict]) -> list[str]:
    root = Path(root).resolve()
    errors = []
    for row in records:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root):
            errors.append(f"outside artifact root: {row['path']}")
        elif not path.is_file():
            errors.append(f"missing input: {row['path']}")
        elif path.stat().st_size != row['size'] or sha256(path) != row['sha256']:
            errors.append(f"checksum mismatch: {row['path']}")
    return errors


def require_inputs(root: Path, paths: list[str] | None = None) -> list[dict]:
    records = json.loads((root / 'manifests/artifacts.json').read_text())['artifacts']
    if paths is not None:
        selected = set(paths)
        records = [r for r in records if r['path'] in selected]
        absent = selected - {r['path'] for r in records}
        if absent:
            raise ValueError(f'Inputs absent from manifest: {sorted(absent)}')
    errors = verify(root, records)
    if errors:
        raise ValueError('\n'.join(errors))
    return records

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def validator():
    try:
        return importlib.import_module('gcs.artifacts').verify
    except ModuleNotFoundError:
        pytest.fail('Artifact verification is not implemented')


def test_corruption_missing_file_and_escape_are_rejected(tmp_path):
    verify = validator()
    payload = tmp_path / 'input.csv'
    payload.write_bytes(b'a,b\n1,2\n')
    record = {'path': 'input.csv', 'size': 8, 'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}
    assert verify(tmp_path, [record]) == []
    payload.write_bytes(b'a,b\n9,2\n')
    assert 'checksum' in verify(tmp_path, [record])[0]
    payload.unlink()
    assert 'missing' in verify(tmp_path, [record])[0]
    assert 'outside' in verify(tmp_path, [{**record, 'path': '../outside.csv'}])[0]


def test_symlink_cannot_escape_artifact_root(tmp_path):
    verify = validator()
    root = tmp_path / 'bundle'
    root.mkdir()
    secret = tmp_path / 'outside'
    secret.write_bytes(b'abc')
    (root / 'linked').symlink_to(secret)
    errors = verify(root, [{'path': 'linked', 'size': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}])
    assert 'outside' in errors[0]


@pytest.mark.parametrize('problem', ['corrupt', 'missing'])
def test_verification_cli_exits_unsuccessfully_for_invalid_inputs(tmp_path, problem):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'manifests').mkdir()
    script = tmp_path / 'scripts/verify_artifacts.py'
    shutil.copy2(Path(__file__).resolve().parents[1] / 'scripts/verify_artifacts.py', script)
    payload = tmp_path / 'input.csv'
    payload.write_bytes(b'a,b\n1,2\n')
    (tmp_path / 'manifests/artifacts.json').write_text(json.dumps({'artifacts': [{
        'path': 'input.csv', 'size': 8,
        'sha256': hashlib.sha256(b'a,b\n1,2\n').hexdigest(),
    }]}))
    valid = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    assert 'Verified 1 artifacts' in valid.stdout
    if problem == 'corrupt':
        payload.write_bytes(b'a,b\n9,2\n')
    else:
        payload.unlink()
    invalid = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert invalid.returncode != 0
    assert 'Verified' not in invalid.stdout
    assert 'input.csv' in invalid.stderr

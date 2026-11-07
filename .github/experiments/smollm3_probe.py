"""One model-only experiment. Uses unchanged qualified runtime and harness.

Not a default switch, release, or native/core requalification. Any failed axis stays failed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

BASE_COMMIT = 'f803d5d16add2edb0963b181ee530e75cbcf91d8'
BASE_TREE = '7f371e3d8acf46c6091527cad19b02faed68ef0a'
MODEL_REPO = 'ggml-org/SmolLM3-3B-GGUF'
MODEL_FILE = 'SmolLM3-Q4_K_M.gguf'
MODEL_REVISION = '4965cb60b150737b68a0408c36aeefb65078f894'
MODEL_SHA = '8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e'
MODEL_SIZE = 1915305312
HARNESS_SHA = 'a96392bef56afc1ff8faf1fbc89648364c5e2fc2d7f12ba657d5d4ca0ba53da0'
MODEL_URL = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'root', 'wheel', 'native-evidence', 'deadline-file'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    source = Path(args.source_root).resolve(strict=True)
    root = Path(args.root).resolve()
    if os.name != 'nt':
        raise RuntimeError('Windows required')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Baseline commit changed')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}'], text=True).strip() != BASE_TREE:
        raise RuntimeError('Baseline tree changed')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Baseline source is dirty')
    hp = source / '.github/qualification/interactive_core.py'
    if sha(hp) != HARNESS_SHA:
        raise RuntimeError('Baseline harness changed')
    spec = importlib.util.spec_from_file_location('unchanged_interactive_core', hp)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.MODEL_SHA256 = MODEL_SHA
    core.MODEL_REL = Path('.nexus-agent/models/ggml-org--SmolLM3-3B-GGUF') / MODEL_FILE
    original_env = core.child_environment
    end = json.loads(Path(args.deadline_file).read_text(encoding='utf-8-sig'))['work_deadline_utc']

    def model_only_env(home, temporary, workspace):
        env = original_env(home, temporary, workspace)
        env['NEXUS_AGENT_MODEL_REPO'] = MODEL_REPO
        env['NEXUS_AGENT_MODEL_FILENAME'] = MODEL_FILE
        env['NEXUS_AGENT_ENABLE_THINKING'] = '1'
        target = Path(home) / core.MODEL_REL
        if target.exists():
            raise RuntimeError('Model destination not fresh')
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + '.part')
        count = 0
        h = hashlib.sha256()
        start = time.monotonic()
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, part.open('xb') as f:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                    if time.time() >= end - 600 or time.monotonic() - start > 1200:
                        raise TimeoutError('Bounded model download exhausted; no retry')
                    count += len(block)
                    if count > MODEL_SIZE:
                        raise RuntimeError('Model exceeds pinned bytes')
                    h.update(block)
                    f.write(block)
            if count != MODEL_SIZE or h.hexdigest() != MODEL_SHA:
                raise RuntimeError('Pinned model digest/size mismatch')
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
        (root / 'output/model-experiment.json').write_text(json.dumps({
            'kind': 'ISOLATED_MODEL_EXPERIMENT_NOT_RELEASE_QUALIFICATION',
            'baseline_commit': BASE_COMMIT, 'baseline_tree': BASE_TREE,
            'harness_sha256': HARNESS_SHA, 'model_repo': MODEL_REPO,
            'model_revision': MODEL_REVISION, 'model_filename': MODEL_FILE,
            'model_sha256': MODEL_SHA, 'model_bytes': count,
            'download_seconds': time.monotonic() - start,
            'default_changed': False, 'app_source_changed': False,
            'runtime': 'existing llama.cpp b7075, compatibility unproven until observed'}, indent=2), encoding='utf-8')
        return env

    core.child_environment = model_only_env
    args.commit, args.tree = BASE_COMMIT, BASE_TREE
    try:
        return core.run_driver(args)
    finally:
        # Diagnostic capture cannot mask the protocol outcome or skip its cleanup.
        output = root / 'output'
        try:
            output.mkdir(parents=True, exist_ok=True)
            logs = root / 'home/.nexus-agent/llama-server'
            if logs.is_symlink() or not logs.resolve().is_relative_to(root.resolve()):
                raise RuntimeError('Unsafe owned log directory')
            for index, log in enumerate(sorted(logs.glob('server-*.log')) if logs.exists() else []):
                if index >= 4 or log.is_symlink() or not log.is_file() or not log.resolve().is_relative_to(logs.resolve()):
                    continue
                cap = 2 * 1024 * 1024
                core.check_evidence_cap(output, reserve=cap + core.MANIFEST_RESERVE)
                with log.open('rb') as stream:
                    raw = stream.read(cap + 1)
                # Child environment contains no tokens and corpus is synthetic.
                # Still redact conventional credential-shaped assignments defensively.
                import re
                text = raw[:cap].decode('utf-8', errors='replace')
                text = re.sub(r'(?i)(authorization|api[_-]?key|access[_-]?token|password)([=: ]+)[^\s,;]+', r'\1\2[REDACTED]', text)
                encoded = text.encode('utf-8')[:cap]
                (output / f'model-server-{index}.log').write_bytes(encoded)
                (output / f'model-server-{index}-capture.json').write_text(json.dumps({
                    'owned_file': log.name, 'original_bytes': log.stat().st_size,
                    'captured_bytes': len(encoded), 'truncated': len(raw) > cap,
                    'meaning': 'Actual load/template/thinking diagnostics, not a separate model invocation'}, indent=2), encoding='utf-8')
            core.check_evidence_cap(output)
        except Exception as error:
            # Log collection failure is evidence incompleteness, never a changed model score.
            print('MODEL_LOG_CAPTURE_INCOMPLETE:' + type(error).__name__, file=sys.stderr)



if __name__ == '__main__':
    raise SystemExit(main())

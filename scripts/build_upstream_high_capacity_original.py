#!/usr/bin/env python3
"""Compile the exact isolated1600 preset; called only under its resource owner."""
import os
from pathlib import Path
import shutil
import subprocess
import upstream_repro as u

REPORT = u.ROOT / 'experiments/upstream-high-capacity-original'
RUN = u.ROOT / 'artifacts/upstream-high-capacity-original'
TREE = RUN / 'source'


def main():
    receipt = REPORT / 'original-build-receipt.json'
    if receipt.exists():
        raise ValueError('Existing isolated build must be explicitly verified, not replaced')
    plan = u.read(REPORT / 'execution-plan.json')
    for rel, digest in plan['bindings'].items():
        if u.sha(u.ROOT / rel) != digest:
            raise ValueError('Complete original source/data input changed before build')
    env = u.environment()
    env.update(CARGO_TARGET_DIR=str(RUN / 'target'), CARGO_BUILD_JOBS='1')
    cargo = shutil.which('cargo', path=env['PATH'])
    if not cargo:
        raise ValueError('Pinned workspace Rust toolchain unavailable')
    command = [cargo, 'build', '--release', '--locked', '--features', 'blas', '--bin', 'tsvdx4-new']
    subprocess.run(command, cwd=TREE, env=env, check=True)
    binary = RUN / 'target/release/tsvdx4-new'
    # Recheck exact complete source/data closure after compilation too.
    for rel, digest in plan['bindings'].items():
        if u.sha(u.ROOT / rel) != digest:
            raise ValueError('Complete original source/data input changed during build')
    u.write(receipt, dict(source_commit=u.PIN, command=command, cwd=str(TREE),
            preparation_sha256=u.sha(REPORT / 'execution-plan.json'),
            builder_sha256=u.sha(Path(__file__)),
            cargo=subprocess.check_output([cargo, '--version'], env=env, text=True).strip(),
            environment={key:env[key] for key in ['CARGO_TARGET_DIR','CARGO_BUILD_JOBS','RUSTFLAGS'] if key in env},
            binary_path=str(binary), binaries={binary.name:u.sha(binary)},
            original_source_and_complete_data_unchanged=True, training_launched=False))
    print('Complete isolated source-pinned1600 binary build; no training launched', flush=True)


if __name__ == '__main__':
    main()

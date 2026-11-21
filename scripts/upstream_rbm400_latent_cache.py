"""Authenticated read-only latent cache IO, without production admission or CLI.

The caller supplies a separately frozen manifest digest and an ownership guard.
Source/own16 admission and real-model feature replay remain external requirements.
Maps stay open only inside the context and are bound to the hashed file identity.
"""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import stat
import numpy as np
from upstream_factored_rbm_inputs import verify_files, digest_ok
from upstream_factored_rbm_admission import file_identity
from upstream_rbm400_latent_context_math import Context, FACTORS

FORMAT = 'own_rbm400_latent_cache_v1'
FIELDS = {phase + '/' + name for phase in ('known', 'qualifying')
          for name in ('users', 'items', 'covered')} | {'item_map', 'user_map'}


def confined(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Safe relative cache path required')
    actual = root / path
    if actual.resolve() != actual:
        raise ValueError('Cache paths must not traverse symlinks')
    return actual


@contextmanager
def mapped(path, record, dtype, shape, guard):
    """Map the same authenticated FD; reject trailing bytes and changed files."""
    guard()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or file_identity(before) != record['identity']:
            raise ValueError('Exact authenticated regular cache file required')
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError('Primitive NPYv1/v2 cache required')
        actual_shape, fortran, actual_dtype = readers[version](stream, max_header_size=4096)
        if actual_shape != shape or actual_dtype != np.dtype(dtype) or fortran:
            raise ValueError('Exact C-order cache dtype and dimensions required')
        offset = stream.tell()
        if before.st_size != offset + np.prod(shape, dtype=np.int64) * actual_dtype.itemsize:
            raise ValueError('Exact header and complete cache payload required')
        array = np.memmap(stream, dtype=actual_dtype, mode='r', offset=offset, shape=shape)
        try:
            yield array
        finally:
            array._mmap.close()
            guard()
            if (file_identity(before) != file_identity(os.fstat(stream.fileno()))
                    or file_identity(before) != file_identity(path.stat())):
                raise ValueError('Cache identity changed during use')


@contextmanager
def load(root, manifest_path, expected_sha256, guard):
    """Load both coordinate-bound contexts; does not admit a scientific provider.

    Each reference hash is part of the caller-authenticated manifest. The caller
    must separately validate those references against the declared source/model
    receipts, rather than treating this IO-level authentication as admission.
    """
    root = Path(root).resolve()
    path = confined(root, manifest_path)
    if not callable(guard) or not digest_ok(expected_sha256):
        raise ValueError('External guard and predeclared manifest SHA256 required')
    records = verify_files(root, {manifest_path: expected_sha256}, guard)
    with path.open('rb') as stream:
        identity = records[manifest_path]['identity']
        if file_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError('Authenticated manifest identity required')
        manifest = json.load(stream)
        if (file_identity(os.fstat(stream.fileno())) != identity
                or file_identity(path.stat()) != identity):
            raise ValueError('Manifest changed while reading')
    if (manifest['format'] != FORMAT or set(manifest['arrays']) != FIELDS
            or manifest['qualifying_labels_read'] is not False
            or manifest['production_provider_admitted'] is not False):
        raise ValueError('Exact pending label-free cache manifest required')
    users, items = manifest['users'], manifest['items']
    if type(users) is not int or type(items) is not int or users < 1 or items < FACTORS:
        raise ValueError('Explicit positive canonical user/movie dimensions required')
    references = manifest['references']
    if not references or not all(digest_ok(v) for v in references.values()):
        raise ValueError('Explicit frozen source/model/proof references required')
    rows = manifest['arrays']
    bindings = dict(references)
    for row in rows.values():
        relative, digest = row['path'], row['sha256']
        confined(root, relative)
        if relative in bindings or not digest_ok(digest):
            raise ValueError('Distinct authenticated cache payloads required')
        bindings[relative] = digest
    for relative in bindings:
        confined(root, relative)
    records = verify_files(root, bindings, guard)
    with ExitStack() as stack:
        arrays = {}
        for name, row in rows.items():
            leaf = name.split('/')[-1]
            dtype = np.bool_ if leaf == 'covered' else ('<f8' if leaf.endswith('_map') else '<f4')
            shape = ((users,) if leaf == 'covered' else
                     ((FACTORS, FACTORS) if leaf.endswith('_map') else
                      (users if leaf == 'users' else items, FACTORS)))
            arrays[name] = stack.enter_context(mapped(
                root / row['path'], records[row['path']], dtype, shape, guard))
        contexts = {}
        for phase in ('known', 'qualifying'):
            guard()
            contexts[phase] = Context(*(arrays[phase + '/' + n] for n in ('users', 'items', 'covered')))
        a, b = arrays['item_map'], arrays['user_map']
        condition = np.linalg.cond(a)
        if (not np.isfinite(a).all() or not np.isfinite(b).all()
                or not np.isfinite(condition) or condition > 1e6
                or not np.allclose(a @ b.T, np.eye(FACTORS), rtol=0, atol=1e-9)):
            raise ValueError('Finite full-rank declared contragredient maps required')
        guard()
        try:
            yield contexts, manifest
        finally:
            # Rehash on caller exceptions too; still check the manifest if a
            # payload/reference check fails. ExitStack closes maps in all cases.
            try:
                verify_files(root, bindings, guard)
            finally:
                verify_files(root, {manifest_path: expected_sha256}, guard)

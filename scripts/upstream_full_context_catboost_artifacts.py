"""Exclusive guarded saves for the future owned full-context CPU tree study.

No source or production-model admission occurs here. Namespace ownership and
full remaining-output reservation must be provided by the external supervisor.
"""
import hashlib
import os
from pathlib import Path
import stat
import numpy as np
import upstream_full_context_catboost as backend
from upstream_factored_rbm_admission import file_identity
from upstream_factored_rbm_terminal_metal import durable_directory, sync_directory

MODEL_RESERVE = 64 * 2**20


def confined(path, root):
    path, root = Path(path), Path(root)
    if root.resolve() != root or path.resolve() != path or not path.is_relative_to(root) or path == root:
        raise ValueError('Canonical owned output path without source aliases required')
    return path, root


def validate_signature(model, seed, iterations, width):
    if (type(seed) is not int or not 0 <= seed < 2**31
            or type(iterations) is not int or not 1 <= iterations <= backend.MAX_ROUNDS
            or type(width) is not int or width not in backend.WIDTHS):
        raise ValueError('Original positive selected full-refit declaration required')
    signature = backend.model_signature(model)
    expected = dict(backend.PARAMETERS, random_seed=seed, iterations=iterations)
    if (signature['tree_count'] != iterations or len(signature['feature_names']) != width
            or any(signature['flat_parameters'].get(key) != value for key, value in expected.items())):
        raise ValueError('Complete original declared-width CPU tree configuration required')
    return signature


def save_model(path, model, *, seed, iterations, width, root, guard):
    """Save through an exclusively opened FD, then publish without overwrite.

    A failed staging file stays available for diagnosis. This never resumes,
    overwrites or substitutes another fit. The guard must account for the
    complete consumer group and preserve the same stack when it suspends.
    """
    if not callable(guard):
        raise ValueError('External owned whole-group resource guard required')
    path, root = confined(path, root)
    signature = validate_signature(model, seed, iterations, width)
    staging = path.with_name(path.name + '.staging')
    if path.exists() or staging.exists():
        raise ValueError('Existing model or failed staging preserved; no retry')
    guard(extra_disk_bytes=MODEL_RESERVE)
    durable_directory(path.parent, root, guard)
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, 'O_NOFOLLOW'): flags |= os.O_NOFOLLOW
    descriptor = os.open(staging, flags, 0o600)
    try:
        original = os.fstat(descriptor)
        if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
            raise ValueError('Exclusive regular model staging required')
        model.save_model(fname='/dev/fd/%s' % descriptor, format='cbm')
        guard(extra_disk_bytes=MODEL_RESERVE)
        os.fsync(descriptor)
        complete = os.fstat(descriptor)
        if (complete.st_dev != original.st_dev or complete.st_ino != original.st_ino
                or complete.st_nlink != 1 or not 0 < complete.st_size <= MODEL_RESERVE
                or file_identity(complete) != file_identity(staging.stat())):
            raise ValueError('Complete bounded original model bytes and staging identity required')
        # Authenticate bytes from the very FD passed to CatBoost, not a path
        # reopened after the native writer returns.
        os.lseek(descriptor, 0, os.SEEK_SET)
        hashed = hashlib.sha256()
        while True:
            guard(); block = os.read(descriptor, 1048576)
            if not block: break
            hashed.update(block)
        if file_identity(complete) != file_identity(os.fstat(descriptor)) or file_identity(complete) != file_identity(staging.stat()):
            raise ValueError('Original saved tree changed during hashing')
        guard()
        os.link(staging, path)
        linked = path.stat()
        if linked.st_dev != complete.st_dev or linked.st_ino != complete.st_ino:
            raise ValueError('Original authenticated model inode required at publication')
        sync_directory(path.parent)
        staging.unlink(); sync_directory(path.parent)
        final = path.stat()
        if final.st_nlink != 1 or file_identity(final) != file_identity(os.fstat(descriptor)):
            raise ValueError('Published original single-link model identity required')
        guard()
        return dict(path=str(path), bytes=final.st_size, sha256=hashed.hexdigest(),
                    identity=file_identity(final), signature=signature,
                    seed=seed, iterations=iterations, width=width)
    finally:
        os.close(descriptor)

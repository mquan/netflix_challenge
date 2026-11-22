"""Parent-observed producer exit component; no spawning or production admission.

An external owned supervisor captures its actual direct child while it is
waiting at bootstrap, connects the pipe and consumer, then polls this record
without blocking on stdout. Only a reaped zero exit permits receipt creation.
Whole-family lanes, bootstrap authorization and stop/resume remain external.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_consumer as consumer


@dataclass(frozen=True)
class OwnedProducer:
    handle: subprocess.Popen
    producer: dict
    owner: dict
    group: int
    contract: dict
    metadata_sha256: str
    source_sha256: dict


def same_lifetime(expected, actual):
    return bool(actual and actual['pid']==expected['pid'] and actual['started']==expected['started'])


def owner_alive(owner):
    actual=consumer.identity.process(os.getpid())
    if (owner['pid']!=os.getpid() or not same_lifetime(owner,actual)
            or actual['command']!=owner['command'] or actual['pgid']!=owner['pgid']):
        raise ValueError('Same actual original parent owner must capture and reap its producer')


def capture(handle, *, owner, group, expected_contract, metadata_sha256, source_sha256, guard):
    consumer.no_torch();guard();owner_alive(owner)
    if (not isinstance(handle,subprocess.Popen) or type(group) is not int or group<=0
            or not consumer.wire.digest_ok(metadata_sha256)
            or expected_contract['admission_sha256']!=metadata_sha256 or not source_sha256):
        raise ValueError('Actual owned handle, original group, full source closure and stream binding required')
    if handle.poll() is not None:raise ValueError('Capture actual waiting bootstrap before producer exit')
    actual=consumer.identity.process(handle.pid)
    if (not actual or actual['ppid']!=os.getpid() or actual['pgid']!=group):
        raise ValueError('Actual original direct producer child in its authorized scientific group required')
    consumer.verify_files(study.ROOT,source_sha256,guard)
    # Copies prevent later changes to caller-owned metadata from altering the
    # binding that was captured with the real child lifetime.
    import copy
    return OwnedProducer(handle,copy.deepcopy(actual),copy.deepcopy(owner),group,
        copy.deepcopy(expected_contract),metadata_sha256,dict(source_sha256))


def poll(record, path, *, guard):
    """Return None while live, or a durable actual-zero-exit gate after reaping.

    A failed producer raises and leaves the scientific namespace untouched.
    Its owner must preserve/stop all surviving roles; this helper never kills,
    retries, restarts, releases lanes or declares a production study complete.
    """
    consumer.no_torch();guard()
    if not isinstance(record,OwnedProducer):raise ValueError('Captured actual original producer handle required')
    owner_alive(record.owner)
    if record.handle.pid!=record.producer['pid']:
        raise ValueError('Actual child handle must match captured producer identity')
    path=Path(path)
    if path.exists():raise ValueError('Original producer receipt preserved; no overwrite or retry')
    if (path.resolve()!=path or not path.is_relative_to(study.ROOT)
            or path==study.ROOT):
        raise ValueError('Canonical exclusive owned producer receipt path required')
    status=record.handle.poll()  # waitpid(WNOHANG): never wait on a full pipe.
    actual=consumer.identity.process(record.producer['pid'])
    if status is None:
        if (not same_lifetime(record.producer,actual) or actual['ppid']!=record.owner['pid']
                or actual['pgid']!=record.group):
            raise ValueError('Original live producer lifetime, parent and group must remain unchanged')
        return None
    if status!=0:raise ValueError('Actual producer failed; no successful exit receipt may be published')
    if same_lifetime(record.producer,actual):
        raise ValueError('Original producer must actually be reaped, not merely close its output pipe')
    value=dict(state='complete_owned_original_tree_producer_zero_exit',exit_code=0,
        producer=record.producer,owner=record.owner,scientific_group=record.group,
        contract_sha256=hashlib.sha256(consumer.wire.encoded(record.contract)).hexdigest(),
        source_admission_sha256=record.metadata_sha256,source_sha256=record.source_sha256,
        observed_by_actual_parent_waitpid=True,qualifying_labels_read=False,
        production_study_admitted=False)
    consumer.verify_files(study.ROOT,record.source_sha256,guard)
    study.save_json(path,value,guard)
    result=dict(path=path,sha256=study.sha(path),producer=record.producer)
    consumer.producer_exit(path,result['sha256'],record.producer,record.contract,
        record.metadata_sha256,study.ROOT,guard)
    return result

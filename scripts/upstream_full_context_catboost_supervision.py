"""Whole-family guard/readmission/watchdog component; no CLI or admission.

The future owner supplies its source/phase authorization and full remaining
output reservation. All original resource limits and observer scope stay fixed.
Numerical roles self-stop the same process group and return to the same frame
only after the live original owner readmits three consecutive full reservations.
"""
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import time
import upstream_full_context_catboost_family as family
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_consumer as consumer
import upstream_cf96_memory_resume as memory


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Control:
    def __init__(self,root,roles,*,repo,baseline_path,owner,watchdog,check_sources,remaining_output,emergency_reserve,request_path):
        self.root=copy.deepcopy(root);self.roles_provider=roles;self.repo=Path(repo)
        self.baseline_path=Path(baseline_path);self.owner=copy.deepcopy(owner);self.watchdog=copy.deepcopy(watchdog)
        self.check_sources=check_sources;self.remaining_output=remaining_output;self.emergency_reserve=emergency_reserve
        self.request_path=Path(request_path);self.cached_roles={};self.last_reserve=emergency_reserve
        if (any(not callable(f) for f in (roles,check_sources,remaining_output))
                or self.repo.resolve()!=self.repo or root['pgid']!=root['pid']
                or type(emergency_reserve) is not int or emergency_reserve<family.RESERVE
                or any(p.resolve()!=p or not p.is_relative_to(self.repo) for p in (self.baseline_path,self.request_path))
                or not self.request_path.parent.is_dir()):
            raise ValueError('Original root/group, owned metadata and full declared output reservation required')
        self.baseline=consumer.verified_json(self.baseline_path,study.BASELINE_SHA,self.repo,lambda:None)
        if set(self.baseline)!={'compressor_bytes','swap_used_bytes'}:
            raise ValueError('Exact original scientific baseline required')

    def roles(self):
        values=self.roles_provider()
        if not isinstance(values,dict) or 'root' in values or any(not isinstance(v,dict) for v in values.values()):
            raise ValueError('All original scientific role captures required')
        self.cached_roles=copy.deepcopy(values);return values

    def check(self):
        if self.check_sources() is False:raise ValueError('Original source/phase authorization unavailable')
        if (not family.identity.same_owner(self.owner,family.identity.process(self.owner['pid']))
                or not family.identity.same_owner(self.watchdog,family.identity.process(self.watchdog['pid']))):
            raise RuntimeError('Same live original owner and watchdog required')
        current=consumer.verified_json(self.baseline_path,study.BASELINE_SHA,self.repo,lambda:None)
        if current!=self.baseline:raise ValueError('Original baseline cannot be reset or replaced')

    def reserve(self,extra=0):
        if type(extra) is not int or extra<0:raise ValueError('Nonnegative complete extra output reservation required')
        declared=self.remaining_output()
        if type(declared) is not int or declared<family.RESERVE:
            raise ValueError('Complete original remaining-output reservation required')
        self.last_reserve=max(declared,extra,family.RESERVE);return self.last_reserve

    def sample(self,extra=0,*,readmission=False):
        self.check();roles=self.roles();reserve=self.reserve(extra)
        workers,group=family.measured(self.root,roles);current=memory.memory();free=shutil.disk_usage(self.repo).free
        predicate=family.readmission if readmission else family.safe
        args=(workers,group,self.root['pid'],current,self.baseline,free) if readmission else (workers,group,current,self.baseline,free)
        allowed=predicate(*args,disk_reserve=reserve)
        return allowed,dict(workers=workers,family_pids=sorted(group),family_usage=family.usage(workers,group),
            memory=current,free_bytes=free,complete_remaining_output_reservation=reserve,
            full_family_readmission_reservation=family.FAMILY_LIMIT if readmission else None)

    def stop(self,reason):
        try:self.roles()
        except Exception:pass  # Keep the last actual role identities on accounting failure.
        try:status(self.request_path,dict(state='same_original_family_preservation_requested',root=self.root,
            roles=self.cached_roles,reason=reason,complete_remaining_output_reservation=self.last_reserve,
            original_baseline_sha256=study.BASELINE_SHA,observed=stamp(),automatic_retry=False))
        except OSError:pass
        if not family.stop_family(self.root,self.cached_roles):
            raise RuntimeError('Original family no longer available for same-frame preservation')

    def wait_then_resume(self,extra=0):
        # Called by the external owner, never by a stopped scientific role.
        if os.getpid()!=self.owner['pid']:raise ValueError('Only the original external owner may readmit its family')
        good=0
        while good<3:
            # Owner/watchdog/source loss is terminal for readmission. A live
            # orphan remains stopped with its lane descriptors retained.
            self.check()
            if not family.matches(self.root,family.identity.process(self.root['pid']),self.root['pid']):
                raise RuntimeError('Never resume a surviving role after original scientific root loss')
            try:allowed,sample=self.sample(extra,readmission=True)
            except Exception:allowed=False
            good=good+1 if allowed else 0
            if good<3:time.sleep(10)
        self.check()
        actual=family.identity.process(self.root['pid'])
        if not family.matches(self.root,actual,self.root['pid']) or 'T' not in actual['state']:
            raise RuntimeError('Same preserved original scientific root required for continuation')
        os.killpg(self.root['pid'],signal.SIGCONT)
        return sample


class Guard:
    def __init__(self,control):
        self.control=control;self.last=0.;self.stops=0
        roles=control.roles();actual=family.live_roles(control.root,roles)
        if os.getpid() not in {row['pid'] for row in actual.values()} or os.getpgrp()!=control.root['pid']:
            raise ValueError('Guard belongs only to an original live numerical family role')

    def __call__(self,extra_disk_bytes=0):
        if type(extra_disk_bytes) is not int or extra_disk_bytes<0:
            raise ValueError('Nonnegative complete extra output reservation required')
        c=self.control
        while True:
            now=time.monotonic()
            try:
                if (now-self.last<3 and shutil.disk_usage(c.repo).free>=family.FLOOR+max(c.last_reserve,extra_disk_bytes)):
                    return
                allowed,sample=c.sample(extra_disk_bytes)
            except Exception as error:
                allowed=False;sample=dict(accounting_or_authorization_error=repr(error))
                c.last_reserve=max(c.last_reserve,c.emergency_reserve,extra_disk_bytes)
            if allowed:self.last=now;return
            self.stops+=1;c.stop(dict(self_stop_number=self.stops,sample=sample))
            # SIGCONT returns here; root, source, matrices, optimizer/RNG and
            # current batch/write frames remain in their original processes.


def watch(owner,root_provider,release_provider,keep_fds,*,record=None):
    """Keep inherited lane descriptions until every known actual role is terminal.

    Metadata errors retain the descriptors. On owner loss or an early release
    request, stop every surviving original captured group, including an orphan
    source after root loss. No model retry, signal to a recycled PID or resume.
    """
    if any(not callable(f) for f in (root_provider,release_provider)):
        raise ValueError('Owned root/role and release metadata providers required')
    captured={}
    def live():
        return [(root,roles) for root,roles in captured.values() if family.live_roles(root,roles)]
    def preserve(values):
        return [root['pid'] for root,roles in values if family.stop_family(root,roles)]
    try:
        while True:
            try:
                value=root_provider()
                if value is not None:
                    root,roles=value;key=(root['pid'],root['started'])
                    previous=captured.get(key,(root,{}))[1]
                    captured[key]=(copy.deepcopy(root),dict(previous,**copy.deepcopy(roles)))
                released=release_provider()
                owner_live=family.identity.same_owner(owner,family.identity.process(owner['pid']))
                alive=live()
                if not owner_live or released or len(alive)>1:
                    if not alive:return
                    stopped=preserve(alive)
                    if record is not None:record(dict(state='original_families_stopped_lanes_retained',
                        families=[dict(root=r,roles=p) for r,p in alive],owner_live=owner_live,
                        early_release_request=released,actual_group_stops_sent=stopped,observed=stamp()))
            except Exception as error:
                # A malformed/missing current pointer must never erase a
                # previously captured live root or orphan source lifetime.
                try:stopped=preserve(live())
                except Exception:stopped=[]
                if record is not None:
                    try:record(dict(state='watchdog_metadata_uncertain_lanes_retained',error=repr(error),
                        actual_group_stops_sent=stopped,observed=stamp()))
                    except Exception:pass
            time.sleep(2)
    finally:
        for fd in keep_fds:os.close(fd)

#!/usr/bin/env python3
"""Fixed rank16 context NN on all91 own predictors, with verified reused trees."""
import argparse
import datetime as dt
import fcntl
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import upstream_cf12_context as preceding
import upstream_source_extension as source
import upstream_extension_run_current as gate
import upstream_cf96_memory_resume as memory

context,n,u,verifier=preceding.context,preceding.n,preceding.u,preceding.verifier
RUN=u.ROOT/'artifacts/upstream-extension-context'
REPORT=u.ROOT/'experiments/upstream-extension-context'
CONFIG=dict(context.CONFIG,architecture='context-weighted-residual-all91-v1')
SIGNALS=list(range(63))+list(range(68,91))+list(range(221,228))
CONTEXT=list(range(63,68))+list(range(91,221))


def model_factory(width,cfg):
    import torch
    if width!=228 or cfg!=CONFIG:raise ValueError('Frozen all91 context layout/config required')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(SIGNALS))
            self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),
                torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),torch.nn.GELU(),
                torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:
                torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals))
            weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()


def neural_model(checkpoint):
    import torch
    torch.set_num_threads(CONFIG['threads'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=model_factory(saved['feature_width'],saved['config'])
    model.load_state_dict(saved['state']);return model


def code_hashes():
    paths=[Path(__file__),Path(preceding.__file__),Path(source.__file__),Path(gate.__file__),Path(memory.__file__)]
    return {**preceding.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def configure():
    context.model_factory=model_factory
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,CONFIG,code_hashes
    n.train_nn=context.train_nn;verifier.neural_model=neural_model
    # Reuse the unchanged generic four-arm assembler/scorer, with this process's
    # paths. No earlier source file or on-disk experiment is modified.
    preceding.RUN,preceding.REPORT=RUN,REPORT


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial all91 context preparation preserved')
    source.configure();n.CONFIG=u.read(source.REPORT/'manifest.json')['config'];pm=n.verify_prepared()
    evaluation=source.REPORT/'evaluation.json';proof_path=source.REPORT/'independent-verification.json'
    proof=u.read(proof_path)
    if (proof['state']!='complete' or proof['evaluation_sha256']!=u.sha(evaluation) or
            not proof['fixed_average_bitwise_equal'] or pm['nonlinear_features']!=227 or
            pm['prediction_columns']!=91 or pm['new_native_predictions']!=source.NEW):
        raise ValueError('Complete verified fixed91 parent required')
    replay_path=source.REPORT/'tree-replay.json';replay=u.read(replay_path)
    if len(replay['records'])!=5 or not all(r['bitwise_equal'] for r in replay['records']):
        raise ValueError('All five saved parent trees must replay')
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in
        [source.REPORT/'manifest.json',evaluation,proof_path,replay_path]}
    for fold in range(5):
        path=source.REPORT/'tree'/f'{fold}.json';receipt=u.read(path)
        if receipt['manifest_sha256']!=u.sha(source.REPORT/'manifest.json'):raise ValueError('Parent tree recipe differs')
        for name,digest in receipt['outputs'].items():
            p=source.RUN/'tree'/str(fold)/name
            if u.sha(p)!=digest:raise ValueError('Parent tree artifact changed')
            sources[str(p.relative_to(u.ROOT))]=digest
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    tree=next(r for r in u.read(evaluation)['records'] if r['arm']=='tree')
    if u.sha(source.RUN/'tree.f32')!=tree['prediction_sha256']:raise ValueError('Parent assembled tree differs')
    reference=gate.development_reference(prepared=source.RUN)
    reference_root=Path(reference['report'])
    for name in ['manifest.json','evaluation.json','independent-verification.json']:
        path=reference_root/name;sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    parent_best=next(r['rmse'] for r in u.read(evaluation)['records'] if r['arm']=='fixed-half-tree-nn')
    for relative,digest in pm['prepared_sha256'].items():
        if u.sha(source.RUN/relative)!=digest:raise ValueError('Prepared parent changed')
    configure();RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True)
    for name,digest in {**pm['prepared_sha256'],'tree.f32':tree['prediction_sha256']}.items():
        shutil.copyfile(source.RUN/name,RUN/name)
        if u.sha(RUN/name)!=digest:raise ValueError('Copied parent artifact differs')
    manifest=dict(config=CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=pm['rows'],baseline_rmse=pm['baseline_rmse'],prediction_columns=91,
        design_columns=167,nonlinear_features=227,signal_indices=SIGNALS,context_indices=CONTEXT,
        previous_best_rmse=min(reference['rmse'],parent_best),development_reference=reference,
        reused_tree_prediction_sha256=tree['prediction_sha256'],qualifying_labels_read=False,
        arms=['control','tree','nn','fixed-half-tree-nn'],isolation=pm['isolation'],
        scope='Adaptive known-ProbeX only. Fixed rank16 context architecture on all six predeclared additions; original nested user folds, optimizer, seeds and stopping rules. Reuses verified matching91 trees. No automatic production or recovered-label audit.')
    u.write(REPORT/'manifest.json',manifest);return manifest


def predecessors():
    names={'upstream_extension_run_current.py','upstream_extension_production_current.py',
           'upstream_cf12_matching_production.py'}
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines();found={}
    for row in rows:
        pid,command=row.strip().split(None,1)
        if any(Path(arg).name in names for arg in command.split()[1:]):found[pid]=command
    return found


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=memory.memory();good=0
        while True:
            prior=predecessors();current=memory.memory()
            owned=memory.footprints() if not prior and current['compressor_bytes']<10*2**30 else {}
            admitted=not prior and bool(owned) and memory.admitted(previous,current,sum(r['bytes'] for r in owned.values()))
            # An empty worker set is also safe when its scan actually ran.
            if not prior and current['compressor_bytes']<10*2**30 and not owned:
                admitted=memory.admitted(previous,current,0)
            good=good+1 if admitted else 0
            u.write(REPORT/'status.json',dict(state='waiting_for_prior_fits_and_memory',
                observed=dt.datetime.now(dt.timezone.utc).isoformat(),predecessors=prior,
                memory=current,owned_workers=owned,consecutive_good_samples=good))
            if good>=2:break
            previous=current;time.sleep(30)
        with (source.REPORT/'run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
            prepare();u.write(REPORT/'status.json',dict(state='nested_nn_training'))
            preceding.guarded([sys.executable,str(Path(__file__)),'--worker'],REPORT/'nn.log')
            u.write(REPORT/'status.json',dict(state='saved_nn_replay'))
            preceding.guarded([sys.executable,str(Path(__file__)),'--replay'],REPORT/'nn-replay.log')
            result=preceding.summarize()
            u.write(REPORT/'status.json',dict(state='development_complete',records=result['records'],goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args()
    configure()
    if a.worker:n.worker('nn')
    elif a.replay:verifier.replay('nn')
    else:
        try:run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise

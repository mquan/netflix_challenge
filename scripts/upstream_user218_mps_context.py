#!/usr/bin/env python3
"""Fixed larger rank32/hidden128 context NN on own107 plus all90 original latent supports."""
import argparse
import math
from pathlib import Path
import subprocess
import numpy as np
import upstream_extension_context as base
import upstream_user218_features as data
import upstream_mps_context as mps_backend

u,n=base.u,base.n
BASE_FILE=Path(base.__file__)
PREVIOUS_REPORT=u.ROOT/'experiments/upstream-rbm50-context'
PREVIOUS_RUN=u.ROOT/'artifacts/upstream-rbm50-context'
ORIGINAL_HASHES=base.code_hashes
ORIGINAL_PREDECESSORS=base.predecessors
RUN=u.ROOT/'artifacts/upstream-user218-mps-context'
REPORT=u.ROOT/'experiments/upstream-user218-mps-context'
CONFIG=dict(base.CONFIG,architecture='context-weighted-residual-own107-user218-mps-v1',neural_backend='torch-mps-float32',nn_hidden=128,context_rank=32)
SIGNALS=list(range(63))+list(range(68,91))+list(range(221,243))+[461]
CONTEXT=list(base.CONTEXT)+list(range(243,461))


def model_factory(width,cfg):
    import torch
    if width!=462 or cfg!=CONFIG:raise ValueError('Frozen own107+user218 larger context recipe required')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(SIGNALS));self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),
                torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:
                torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals));weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()


def code_hashes():
    return {**ORIGINAL_HASHES(),**data.code_hashes(),str(BASE_FILE.relative_to(u.ROOT)):u.sha(BASE_FILE),
        str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),
        'scripts/verify_upstream_cf48_final_screen.py':u.sha(u.ROOT/'scripts/verify_upstream_cf48_final_screen.py'),str(Path(mps_backend.__file__).relative_to(u.ROOT)):u.sha(Path(mps_backend.__file__))}


def predecessors():
    found=ORIGINAL_PREDECESSORS()
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        pid,command=row.strip().split(None,1)
        if any(Path(arg).name==BASE_FILE.name for arg in command.split()[1:]):found[pid]=command
    return found


def configure():
    # Rebind only this interpreter's controller. Existing source and experiments
    # remain unchanged; all reused implementation files are hash-bound.
    base.RUN,base.REPORT,base.CONFIG=RUN,REPORT,CONFIG
    base.SIGNALS,base.CONTEXT=SIGNALS,CONTEXT
    base.model_factory,base.code_hashes,base.prepare,base.predecessors=model_factory,code_hashes,prepare,predecessors
    base.memory.admitted=context_admitted
    base.__file__=str(Path(__file__).resolve())
    base.configure()
    mps_backend.MODEL_FACTORY=model_factory
    n.train_nn=mps_backend.train_nn;n.nn_predict=mps_backend.predict
    base.verifier.neural_model=neural_model


def neural_model(checkpoint):
    return mps_backend.load_model(checkpoint,CONFIG)


def context_admitted(previous,current,footprint):
    # Reserve eight GiB for the wider462-input meta worker, separately from
    # the native jobs. Original fitting/stopping remains unchanged. Do not
    # alter the shared memory helper on disk.
    return (footprint<18*2**30 and current['compressor_bytes']<5*2**30 and
            current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20)


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial own107+user218 context cache preserved')
    dm=u.read(data.REPORT/'manifest.json')
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=data.RUN,data.REPORT,dm['config'],data.code_hashes
    n.verify_prepared()
    ridge_proof=data.REPORT/'cache-column-verification.json';checked=u.read(ridge_proof)
    independent_ridge=data.REPORT/'ridge-verification.json'
    if (checked['state']!='complete_cache_columns' or not checked['outer_users_disjoint'] or
            checked['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or
            checked['screen_sha256']!=u.sha(data.REPORT/'ridge-screen.json') or
            checked['source_ridge_verification_sha256']!=u.sha(independent_ridge) or
            dm['new_native_predictions']!=data.NEW or dm['prediction_columns']!=107 or
            dm.get('support_columns')!=218 or dm.get('tx_first_phase_epochs_complete')!=20 or dm.get('rbm_first_phase_epochs_complete')!=20 or
            not checked.get('original_support_floats_preserved_bitwise')):
        raise ValueError('Verified fixed107+user218 inputs required')
    source=base.source;sm=u.read(source.REPORT/'manifest.json');evaluation=source.REPORT/'evaluation.json'
    proof=source.REPORT/'independent-verification.json';replay=source.REPORT/'tree-replay.json'
    if (u.read(proof)['evaluation_sha256']!=u.sha(evaluation) or
            not u.read(proof)['fixed_average_bitwise_equal'] or
            [r['fold'] for r in u.read(replay)['records']]!=list(range(5)) or
            not all(r['bitwise_equal'] for r in u.read(replay)['records'])):
        raise ValueError('Complete reused-tree verification required')
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in
        [data.REPORT/'manifest.json',ridge_proof,independent_ridge,data.REPORT/'ridge-screen.json',source.REPORT/'manifest.json',evaluation,proof,replay]}
    for key in ['labels.npy','users.npy','groups.npy']:
        if dm['prepared_sha256'][key]!=sm['prepared_sha256'][key]:raise ValueError('Own107/reused-tree identities differ')
    for fold in range(5):
        path=source.REPORT/'tree'/f'{fold}.json';receipt=u.read(path)
        if receipt['manifest_sha256']!=u.sha(source.REPORT/'manifest.json'):raise ValueError('Reused tree receipt differs')
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        for name,digest in receipt['outputs'].items():
            artifact=source.RUN/'tree'/str(fold)/name
            if u.sha(artifact)!=digest:raise ValueError('Actual reused tree model changed')
            sources[str(artifact.relative_to(u.ROOT))]=digest
    tree=next(r for r in u.read(evaluation)['records'] if r['arm']=='tree')
    if u.sha(source.RUN/'tree.f32')!=tree['prediction_sha256']:raise ValueError('Reused tree vector differs')
    combined_path=u.ROOT/'experiments/upstream-tree91-context86/verification.json';combined=u.read(combined_path)
    if combined['qualifying_labels_read'] is not False or not combined['fixed_average_bitwise_equal']:
        raise ValueError('Known-label global reference required')
    for relative,digest in combined['source_sha256'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Global reference source changed')
        sources[relative]=digest
    sources[str(combined_path.relative_to(u.ROOT))]=u.sha(combined_path);reference=combined['rmse']
    previous_context=None;previous_proof=PREVIOUS_REPORT/'independent-verification.json'
    if previous_proof.exists():
        previous_evaluation=PREVIOUS_REPORT/'evaluation.json';old=u.read(previous_proof)
        if old['state']!='complete' or old['evaluation_sha256']!=u.sha(previous_evaluation):
            raise ValueError('Previous context result binding differs')
        previous_context=min((r for r in old['records'] if r['arm'] in {'nn','fixed-half-tree-nn'}),key=lambda r:r['independent_rmse'])
        vector=PREVIOUS_RUN/(previous_context['arm']+'.f32')
        if u.sha(vector)!=previous_context['prediction_sha256']:raise ValueError('Previous context vector differs')
        for path in [previous_proof,previous_evaluation,vector]:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        reference=min(reference,previous_context['independent_rmse'])
    configure();RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True)
    for name,digest in dm['prepared_sha256'].items():
        u.link_or_copy(data.RUN/name,RUN/name)
        if u.sha(RUN/name)!=digest:raise ValueError('Copied own107+user218 input differs')
    u.link_or_copy(source.RUN/'tree.f32',RUN/'tree.f32')
    manifest=dict(config=CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=dm['rows'],baseline_rmse=dm['baseline_rmse'],prediction_columns=107,design_columns=183,nonlinear_features=461,
        signal_indices=SIGNALS,context_indices=CONTEXT,previous_best_rmse=reference,
        previous_context103=previous_context,reused_tree_prediction_sha256=tree['prediction_sha256'],
        rbm_user_support_columns=50,tx_user_item_support_columns=40,total_learned_support_columns=218,own_encoder_user_support_columns=64,own_item_pca_support_columns=64,rbm_first_phase_epochs_complete=20,rbm_full_two_phase_completion_claimed=True,tx_full_two_phase_completion_claimed=True,
        cf48_full95_model_complete=True,cf96_full95_model_complete=False,
        arms=['control','tree','nn','fixed-half-tree-nn'],qualifying_labels_read=False,isolation=dm['isolation'],
        scope='Adaptive known-ProbeX only. Fixed larger rank32/hidden128 context architecture on own107 plus all90 original RBM/Tx context features plus64 fixed own PCA item context features and64 fixed own epoch20 encoder user features, same five nested user folds/seeds/optimizer/stopping; separately declared float32 MPS training/inference, with device-specific dropout RNG. Reuses actual verified tree91, without synthetic fitting receipts. No production or qualifying audit launched.')
    u.write(REPORT/'manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();configure()
    if a.worker:n.worker('nn')
    elif a.replay:base.verifier.replay('nn')
    else:
        import time
        REPORT.mkdir(parents=True,exist_ok=True)
        while not (data.REPORT/'cache-column-verification.json').exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_verified_user218_support_cache'));time.sleep(30)
        try:base.run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise

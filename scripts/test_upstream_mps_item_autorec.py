#!/usr/bin/env python3
"""Isolated transposed-axis, leakage, resume and actual inference fixture."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import upstream_mps_item_autorec as model
import verify_upstream_mps_item_autorec_data as independent

u,data=model.u,model.data
REPORT=u.ROOT/'experiments/upstream-mps-item-autorec-fixture'

def configure(root):
    if not (root/'fixture-only').exists():raise ValueError('Isolated fixture marker required')
    u.FRESH=root/'fresh';data.RUN=root/'csr';data.REPORT=root/'csr-reports'
    data.USERS,data.ITEMS=4,70005
    data.PHASES={'trainx':('probex',6,4),'fulltrain':('qual',10,5)}
    model.RUN,model.REPORT=root/'model',root/'model-reports'
    model.CONFIG=dict(model.CONFIG,rows=4,columns=70005,hidden=5,batch_rows=2,epochs_per_phase=3,snapshot_epochs=[1,2,3])
    model.HISTORIES.clear()

def save_phase(root,phase,triples):
    out=root/'fresh/data'/phase;out.mkdir(parents=True)
    # Reverse the source order to require an actual independent transpose/sort.
    a=np.array(list(reversed(triples)),dtype=np.int64)
    for name,values in [('item_idxs',a[:,0].astype(np.int16)),('user_idxs',a[:,1].astype(np.int32)),('ratings',a[:,2].astype(np.int8)),('item_cnts',np.bincount(a[:,0],minlength=4).astype(np.int32))]:np.save(out/(name+'.npy'),values)

def run():
    import torch
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/'verification.json').exists():raise ValueError('Existing fixture evidence preserved')
    histories=[(0,0,1),(0,70000,5),(1,2,3),(1,70004,4),(2,69999,2),(3,1,4)]
    known=[(0,69999,4),(1,1,2),(2,70000,5),(3,70004,1)]
    qual=[(3,70000),(0,70004),(2,0),(1,69999),(3,2)]
    with tempfile.TemporaryDirectory(prefix='item-autorec-fixture-',dir=u.ROOT/'artifacts') as folder:
        root=Path(folder);(root/'fixture-only').touch();configure(root)
        save_phase(root,'trainx',histories);save_phase(root,'fulltrain',histories+known)
        for query,triples in [('probex',known),('qual',qual)]:
            out=root/'fresh/data'/query;out.mkdir(parents=True)
            np.save(out/'item_idxs.npy',np.array([r[0] for r in triples],dtype=np.int16))
            np.save(out/'user_idxs.npy',np.array([r[1] for r in triples],dtype=np.int32))
            np.save(out/'ratings.npy',np.array([r[2] for r in known],dtype=np.int8) if query=='probex' else np.zeros(len(qual),dtype=np.int8))
        for phase in data.PHASES:data.build_phase(phase);independent.verify_phase(phase)
        columns=np.load(data.RUN/'trainx/columns.npy')
        if columns.dtype!=np.uint32 or columns.max()!=70004:raise ValueError('High user IDs were truncated')
        # Simulate a producer bug, including a matching claimed payload hash:
        # the independent tuple replay must still reject the wrong column.
        rp=data.REPORT/'trainx-receipt.json';original_receipt=rp.read_bytes()
        p=data.RUN/'trainx/columns.npy';a=np.load(p,mmap_mode='r+');before=int(a[0]);a[0]=5;a.flush()
        bad=u.read(rp);bad['outputs']['columns.npy']=u.sha(p);u.write(rp,bad)
        rejected=False
        try:independent.verify_phase('trainx')
        except ValueError as error:rejected='every-row CSR replay differs' in str(error)
        finally:a[0]=before;a.flush();del a;rp.write_bytes(original_receipt)
        if not rejected:raise ValueError('Independent wrong-column rejection required')
        save_phase(root,'leak',histories+[known[0]])
        original_run,original_phases=data.RUN,data.PHASES
        data.RUN=root/'leak-csr';data.PHASES={**original_phases,'leak':('probex',7,4)};leak_rejected=False
        try:data.build_phase('leak')
        except ValueError as error:leak_rejected='Query pair appears in training history' in str(error)
        finally:data.RUN,data.PHASES=original_run,original_phases
        if not leak_rejected:raise ValueError('Query leakage rejection required')
        model.prepare()
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            for epoch in range(1,4):
                for phase in data.PHASES:
                    with (REPORT/f'{phase}-{epoch}.log').open('x') as log:
                        subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker-root',str(root),'--phase',phase,'--epoch',str(epoch)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                model.paired(epoch)
            # Independent canonical gather: construct dense rows directly from
            # the original unsorted triples, with no CSR/query-order helper.
            for phase,query,triples in [('trainx','probex',histories),('fulltrain','qual',histories+known)]:
                state=torch.load(model.RUN/phase/'snapshots/epoch-00003/model.pt',map_location='cpu',weights_only=False)
                fitted=model.make_model(model.CONFIG).to('mps');fitted.load_state_dict(state['model']);fitted.eval()
                dense=np.zeros((4,70005),dtype=np.float32)
                for item,user,rating in triples:dense[item,user]=np.float32(rating)-np.float32(model.CONFIG['center'])
                parts=[]
                with torch.no_grad():
                    for lo in range(0,4,2):parts.append(fitted(torch.from_numpy(dense[lo:lo+2]).to('mps')).cpu().numpy())
                all_rows=np.concatenate(parts);ids=np.load(u.FRESH/'data'/query/'item_idxs.npy');cols=np.load(u.FRESH/'data'/query/'user_idxs.npy')
                prediction=np.clip(all_rows[ids,cols]+np.float32(model.CONFIG['center']),1,5).astype(np.float32)
                stored=np.load(model.RUN/phase/'snapshots/epoch-00003/prediction.npy')
                if not np.array_equal(prediction.view(np.uint32),stored.view(np.uint32)):raise ValueError('Independent original-order query gather differs')
                pointer=model.latest(phase)
                if pointer['epoch']!=3 or [p.name for p in (model.RUN/phase).glob('epoch-*.pt')]!=['epoch-00003.pt']:
                    raise ValueError('Declared one-checkpoint retention differs')
                if len(list((model.RUN/phase/'snapshots').glob('*/model.pt')))!=3:raise ValueError('Snapshot weights were not preserved')
                del fitted
        sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(model.__file__),Path(data.__file__),Path(independent.__file__)]}
        u.write(REPORT/'verification.json',dict(state='complete_isolated_item_oriented_fixture',code_sha256=sources,
            row_items=4,column_users=70005,high_user_ids_uint32_preserved=True,independent_original_tuple_replay=True,
            independent_wrong_column_rejected=True,query_leakage_rejected=True,fresh_process_optimizer_rng_resumes=4,
            actual_saved_snapshot_query_replays=6,independent_original_query_gather_bitwise=True,
            one_checkpoint_per_phase_and_all_snapshots_preserved=True,fsync_and_atomic_pointer_path_exercised=True,
            qualifying_labels_read=False,real_data_training_performed=False,scope='Isolated tiny fixture only; no real-data trained predictor,50-epoch completion or accuracy claim.'))
        print('Complete transposed fixture: uint32 high IDs, independent corruption/leak checks, six actual replays and fresh-process resumes',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-root');p.add_argument('--phase');p.add_argument('--epoch',type=int);a=p.parse_args()
    if a.worker_root:configure(Path(a.worker_root));model.worker(a.phase,a.epoch)
    else:run()

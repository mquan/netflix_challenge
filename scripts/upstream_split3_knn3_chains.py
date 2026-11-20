#!/usr/bin/env python3
"""Exact published KNN3 chains on our already completed original split3 bases."""
import argparse
import fcntl
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
import upstream_split3_pilot as pilot
import upstream_latent90_local as resources
u=pilot.u
RUN=u.ROOT/'artifacts/upstream-split3-knn3-chains'
TREE=RUN/'source'
REPORT=u.ROOT/'experiments/upstream-split3-knn3-chains'
JOBS={'s3-dnn32nm__knn3':'s3-dnn32nm','s3-nade12s__knn3':'s3-nade12s'}
COUNTS={'train_3':98720013,'probe_3':1760494,'fulltrain':100480507,'qual':2817131}
BASE_TREE=u.ROOT/'artifacts/upstream-split3-diversity/source'
WAIT_NAMES={'upstream_item154_tree.py','upstream_item154_tree_supervisor.py','upstream_itemae_virtual_queue.py',
 'upstream_attention2_virtual_queue.py','upstream_itemae_virtual_mps_context.py','upstream_attention2_virtual_mps_context.py'}

def stamp():return pilot.stamp()

def prepare():
    if (REPORT/'plan.json').exists():
        plan=u.read(REPORT/'plan.json')
        for n,h in plan['bindings'].items():
            if u.sha(u.ROOT/n)!=h:raise ValueError('Original frozen source/base changed: '+n)
        return plan
    if RUN.exists() or REPORT.exists():raise ValueError('Partial chain preparation preserved')
    source=pilot.prepare();registry=u.read(pilot.REGISTRY)['3']['jobs'];bindings={};recipes={};parents={}
    for name,base in JOBS.items():
        expected=f'cargo build --release --bin fit && RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4  ./target/release/fit chain {name} -p pipeline-3.toml --final --preds preds_3 '
        if registry[name]['command']!=expected:raise ValueError('Original published chain command differs')
        recipes[name]=registry[name];path=u.ROOT/f'experiments/upstream-split3-diversity/{base}-verification.json';proof=u.read(path)
        if (proof['state']!='complete_original_two_phase_split3_model' or proof['source_pin']!=u.PIN or
            proof['epochs_per_phase']!=(7 if base=='s3-dnn32nm' else 8) or not proof['both_full_epoch_traces'] or
            not proof['full_known_projection_recomputed'] or not proof['known_training_disjointness_bound_to_unchanged_original_data'] or
            not proof['qualifying_targets_zero'] or proof['qualifying_labels_read'] is not False or proof['public_prediction_values_read'] is not False):
            raise ValueError('Actual complete original own parent models required')
        for n,h in proof['bindings'].items():
            if n in bindings and bindings[n]!=h:raise ValueError('Conflicting original parent lineage')
            bindings[n]=h
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path);parents[base]=proof
    for n,h in bindings.items():
        if u.sha(u.ROOT/n)!=h:raise ValueError('Original complete parent binding changed')
    RUN.mkdir();REPORT.mkdir();TREE.mkdir()
    def bind(p,h=None):
        actual=u.sha(p)
        if h is not None and actual!=h:raise ValueError('Source/input differs')
        bindings[str(p.relative_to(u.ROOT))]=actual
    for n,h in source['source_files_sha256'].items():
        a=pilot.TREE/n;b=TREE/n;bind(a,h);b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b);bind(b,h)
    (TREE/'target/release').mkdir(parents=True);shutil.copy2(pilot.TREE/'target/release/fit',TREE/'target/release/fit');bind(TREE/'target/release/fit',source['binary_sha256']['fit'])
    (TREE/'data').mkdir();(TREE/'preds_3').mkdir()
    for split in COUNTS:
        directory=pilot.TREE/'data'/split;(TREE/'data'/split).symlink_to(directory.resolve(),target_is_directory=True)
        for p in directory.glob('*.npy'):bind(p)
    for base,proof in parents.items():
        for split,count in COUNTS.items():
            a=BASE_TREE/'preds_3'/f'{base}.{split}.npy';h=proof['bindings'].get(str(a.relative_to(u.ROOT)))
            if h is None:raise ValueError('Actual original parent terminal vector not bound')
            values=np.load(a,mmap_mode='r')
            if values.shape!=(count,) or values.dtype!=np.float32:raise ValueError('Complete original base vector schema differs')
            for lo in range(0,count,1_000_000):
                if not np.isfinite(values[lo:lo+1_000_000]).all():raise ValueError('Nonfinite parent predictor')
            b=TREE/'preds_3'/a.name;u.link_or_copy(a,b);bind(b,h)
    if np.any(np.load(TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    for p in [Path(__file__),Path(resources.__file__),Path(pilot.__file__),pilot.REGISTRY,pilot.REPORT/'plan.json',pilot.REPORT/'data-verification.json',pilot.REPORT/'completion-verification.json',pilot.RUN/'probex-row-map.npy']:bind(p)
    plan=dict(created=stamp(),source_pin=u.PIN,bindings=bindings,original_registry_jobs=recipes,parent_models=list(parents),
        published_outputs=list(JOBS),original_parameters='Unmodified Knn3Config::default; source environment whitelist supplies only original4 Rayon/4 OpenBLAS threads. Full co-rated-neighbor construction for both complete original datasets; no epoch/model/data subsampling.',
        admission='Wait for actual current CPU tree and both preceding full-size neural coordinators to finish. Then two stable samples: other native/meta physical footprints below16GiB, compressor below5GiB, swap growth below32MiB/30s and disk above20GiB. Reserve8GiB; chains serialized.',
        base_predictor_values_shared_by_hardlink=True,all_four_parent_terminal_vectors_per_model_finite_and_source_bound=True,
        known_training_disjointness_bound_to_original_data=True,qualifying_targets_zero=True,qualifying_labels_read=False,public_prediction_values_read=False)
    u.write(REPORT/'plan.json',plan);return plan

def predecessors():
    found={}
    for row in subprocess.check_output(['ps','-axo','pid=,stat=,command='],text=True).splitlines():
        r=row.strip().split(None,2)
        if len(r)==3 and not r[1].startswith('Z') and WAIT_NAMES.intersection(Path(a).name for a in r[2].split()):found[r[0]]=r[2]
    return found

def wait_memory():
    previous=resources.memory.memory();good=0
    while good<2:
        current=resources.memory.memory();workers=resources.footprints();busy=predecessors();free=shutil.disk_usage(u.ROOT).free
        good=good+1 if (not busy and sum(r['bytes'] for r in workers.values())<16*2**30 and current['compressor_bytes']<5*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>20*2**30) else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_prior_full_studies_and_original_native_headroom',predecessors=busy,workers=workers,memory=current,disk_free_bytes=free,good_samples=good,observed=stamp()))
        if good<2:previous=current;time.sleep(30)

def phase_trace(log):
    phases=re.findall(r'^(train_3 => probe_3|fulltrain => qual)\s*$',log,re.M)
    counts=[tuple(map(int,m)) for m in re.findall(r'KNN3: (\d+) items with neighbors, (\d+) total weights',log)]
    if phases!=['train_3 => probe_3','fulltrain => qual'] or len(counts)!=2 or 'skipped, files supplied by hand' in log or '(no_fulltrain)' in log:
        raise ValueError('Both full original non-iterative KNN3 builds required')
    if any(not (0<a<=17770 and 0<b<=a*60) for a,b in counts):raise ValueError('Original complete neighbor build counts differ')
    cfg=next((line for line in log.splitlines() if line.startswith('Knn3Config ')),None)
    if cfg is None:raise ValueError('Original native KNN3 configuration log required')
    for k,v in {'threshold':.25,'k_min':10,'k_max':60,'shrinkage':20000.,'reg':.01,'x':.8,'bl_reg_m':25.,'bl_reg_u':10.}.items():
        match=re.search(r'\b'+k+r': ([\d.eE+-]+)',cfg)
        if not match or float(match[1])!=v:raise ValueError('Original default configuration differs')
    return counts

def run():
    plan=prepare()
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);mapping=np.load(pilot.RUN/'probex-row-map.npy')
        for name,base in JOBS.items():
            receipt=REPORT/(name+'-receipt.json')
            if receipt.exists():raise ValueError('Existing chain output preserved; explicit recovery required')
            if list((TREE/'preds_3').glob(name+'*')):raise ValueError('Partial chain outputs preserved')
            wait_memory();prepare();command=['./target/release/fit','chain',name,'-p','pipeline-3.toml','--final','--preds','preds_3'];started=stamp();lp=REPORT/(name+'.log')
            with lp.open('x') as log:
                child=subprocess.Popen(command,cwd=TREE,env=resources.model_environment({}),stdout=log,stderr=subprocess.STDOUT)
                u.write(REPORT/(name+'-launch.json'),dict(pid=child.pid,started=started,command=command))
                u.write(REPORT/'status.json',dict(state='original_native_knn3_chain_running',job=name,pid=child.pid,started=started))
                if child.wait():raise RuntimeError('Original KNN3 chain failed; partial outputs preserved')
            counts=phase_trace(lp.read_text());output=TREE/'preds_3';shutil.copyfile(lp,output/(name+'.out'))
            outputs={str(p.relative_to(TREE)):u.sha(p) for p in output.glob(name+'*') if p.is_file()}
            if not set(plan['original_registry_jobs'][name]['outputs']).issubset(outputs):raise ValueError('Original declared chain output missing')
            for split,count in [('probe_3',1760494),('qual',2817131)]:
                a=np.load(output/f'{name}.{split}.npy',mmap_mode='r')
                if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Complete finite native chain queries required')
            known=RUN/(name+'.probex.npy');np.save(known,np.load(output/f'{name}.probe_3.npy')[mapping]);qual=output/f'{name}.qual.npy'
            u.write(receipt,dict(state='complete_original_chain_pending_independent_verification',job=name,parent=base,source_pin=u.PIN,plan_sha256=u.sha(REPORT/'plan.json'),
                started=started,finished=stamp(),command=command,full_native_phase_build_counts=counts,outputs=outputs,
                projected_pair=dict(name=name,probex=str(known.relative_to(u.ROOT)),qual=str(qual.relative_to(u.ROOT)),probex_sha256=u.sha(known),qual_sha256=u.sha(qual)),
                qualifying_labels_read=False,public_prediction_values_read=False))
        u.write(REPORT/'status.json',dict(state='both_original_knn3_chains_complete_pending_independent_checks',observed=stamp()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare() if a.prepare else run()

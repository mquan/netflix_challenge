"""Join complete own-trajectory operators to authenticated RBM400 contexts.

No model training, recovered rating access, queue launch, or provider admission
occurs here. Source declaration requires completed original extraction proofs.
The scalar alias decision uses both entire query streams, without labels.
"""
from contextlib import contextmanager, ExitStack
import hashlib
from pathlib import Path
import numpy as np
import upstream_complete_trajectory_data as preceding
import upstream_rbm400_latent_execution as extraction
import upstream_rbm400_latent_provider as provider
import upstream_rbm400_latent_cache as cache
import upstream_rbm400_latent_operators as operators
import upstream_factored_rbm_admission as admission

ROOT=extraction.ROOT
COUNTS={'probex':1128164,'qual':2817131}
PHASE={'probex':'known','qual':'qualifying'}


def paired_alias(existing,new_pair,identities):
    if (set(existing)!=set(COUNTS) or set(new_pair)!=set(COUNTS)
            or len(identities)!=486 or len(set(identities))!=486
            or any(len(existing[q])!=486 for q in COUNTS)
            or any(not extraction.inputs.digest_ok(h) for q in COUNTS for h in existing[q]+[new_pair[q]])):
        raise ValueError('Complete486 original clipped pairs and both complete new hashes required')
    for column,identity in enumerate(identities):
        if all(existing[q][column]==new_pair[q] for q in COUNTS):
            return dict(retain_scalar=False,alias_of=identity,clipped_pair_sha256=dict(new_pair))
    return dict(retain_scalar=True,alias_of=None,clipped_pair_sha256=dict(new_pair))


def clipped_hash(values,guard):
    if values.ndim!=1 or values.dtype!=np.dtype('<f4') or not len(values):
        raise ValueError('Complete original native float32 vector required')
    h=hashlib.sha256()
    for lo in range(0,len(values),8192):
        guard();part=values[lo:lo+8192]
        if not np.isfinite(part).all():raise ValueError('Finite raw scalar predictions required before clipping')
        h.update(np.clip(part,0,6).astype('<f4').tobytes())
    guard();return h.hexdigest()


def parent_pairs(declaration):
    candidates=declaration['parent_plan']['candidates']
    retained=declaration['parent']['retained']
    if (len(candidates)!=128 or len(retained)!=95 or len({r['id'] for r in candidates})!=128
            or len({r['id'] for r in retained})!=95):
        raise ValueError('Every original128 candidate and95 paired-retained addition required')
    indexes={row['id']:i for i,row in enumerate(candidates)}
    if any(row['id'] not in indexes or row!=candidates[indexes[row['id']]] for row in retained):
        raise ValueError('Retained candidate identities/files must exactly match original declaration')
    if [indexes[row['id']] for row in retained]!=sorted(indexes[row['id']] for row in retained):
        raise ValueError('Original candidate order must survive paired alias removal')
    identities=['original391:%03d'%i for i in range(391)]+[row['id'] for row in retained]
    hashes={}
    for q in COUNTS:
        full=declaration['input_manifest']['clipped_sha256'][q]
        if len(full['original'])!=391 or len(full['candidates'])!=128:
            raise ValueError('Every original519 complete scalar hash required')
        hashes[q]=list(full['original'])+[full['candidates'][indexes[row['id']]] for row in retained]
    return hashes,identities


class QueryInputs:
    def __init__(self,parent,prediction,users,items,context,guard,*,retain_scalar,latent):
        operators.dimensions(retain_scalar,latent)
        if (not callable(guard) or prediction.ndim!=1 or prediction.dtype!=np.dtype('<f4')
                or users.shape!=prediction.shape or items.shape!=prediction.shape
                or users.dtype.kind not in 'iu' or items.dtype.kind not in 'iu'
                or (latent and context is None) or (not latent and context is not None)):
            raise ValueError('Exact row-aligned scalar, canonical addresses and explicit arm required')
        self.parent,self.prediction,self.users,self.items=parent,prediction,users,items
        self.context,self.guard,self.retain_scalar,self.latent=context,guard,retain_scalar,latent

    def block(self,lo,hi):
        if type(lo) is not int or type(hi) is not int or not 0<=lo<hi<=len(self.prediction):
            raise ValueError('Nonempty original query row interval required')
        self.guard();z,f=self.parent.block(lo,hi)
        context=self.context.values(self.users[lo:hi],self.items[lo:hi]) if self.latent else None
        result=operators.extend(z,f,self.prediction[lo:hi],context,
                                retain_scalar=self.retain_scalar,latent=self.latent)
        self.guard();return result


def bind(bindings,path,digest=None):
    provider.original.bind(bindings,path,digest)


def source_declaration(guard):
    """Build and fully authenticate the complete real provider source contract."""
    if not callable(guard):raise ValueError('External owned3GiB data-worker guard required')
    guard();mp=preceding.REPORT/'manifest.json';vp=preceding.REPORT/'independent-verification.json'
    ip=preceding.REPORT/'input-manifest.json'
    pp=preceding.REPORT/'plan.json';ivp=preceding.REPORT/'input-verification.json'
    m,v,im,parent_plan,iv=[extraction.read(path) for path in (mp,vp,ip,pp,ivp)]
    if (v['state']!='complete_independent_all_trajectory_operators_statistics_ridge_and_stored_predictions'
            or v['manifest_sha256']!=extraction.inputs.sha(mp) or v['source_sha256']!=m['source_sha256']
            or v['every_full_operator_statistic_solve_and_prediction_bit_equal'] is not True
            or v['matched_B_known_and_qualifying'] is not True or v['qualifying_labels_read'] is not False
            or v['neural_models_fitted']!=0 or m['qualifying_labels_read'] is not False
            or m['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or m['input_manifest_sha256']!=extraction.inputs.sha(ip)
            or m['plan_sha256']!=extraction.inputs.sha(pp)
            or m['input_verification_sha256']!=extraction.inputs.sha(ivp)
            or v['input_verification_sha256']!=extraction.inputs.sha(ivp)
            or iv['state']!='complete_independent_all519_scalar_pairs_and_both_operators'
            or iv['input_manifest_sha256']!=extraction.inputs.sha(ip)
            or iv['every_complete_parent_operator_bit_equal'] is not True
            or iv['qualifying_labels_read'] is not False or iv['models_fitted']!=0
            or iv['source_sha256']!=m['source_sha256'] or parent_plan['source_sha256']!=m['source_sha256']
            or parent_plan['all128_candidates_before_alias_removal'] is not True
            or im['source_sha256']!=m['source_sha256'] or im['retained']!=m['retained']
            or [m[k] for k in ('raw_predictions','design_columns','nonlinear_columns','nn_inputs')]!=[486,562,712,713]
            or len(m['retained'])!=95):
        raise ValueError('All519 admitted parent streams and full486-column operator proof required')
    extraction.check_sources(extraction.read(extraction.REPORT/'plan.json'))
    done_path=extraction.REPORT/'completion.json';paired_path=extraction.STUDY/'paired-verification.json'
    done,paired=[extraction.read(p) for p in (done_path,paired_path)]
    plan_sha=extraction.inputs.sha(extraction.REPORT/'plan.json')
    extraction.validate_result('assemble',paired,plan_sha)
    if (done['state']!='complete_full_own_rbm400_latent_extraction_and_independent_queries_and_features'
            or done['verification_sha256']!=extraction.inputs.sha(paired_path)
            or done['qualifying_labels_read'] is not False
            or paired['actual_complete_models_reopened']!=2 or paired['independent_full_phase_replays']!=2):
        raise ValueError('Complete both-phase actual-model/native/cache/feature replays required')
    bindings=dict(m['source_sha256'])
    expected_proofs={str((extraction.STUDY/extraction.RESULTS[phase][0]).relative_to(ROOT))
                     for phase in extraction.PHASES[:-1]}
    if set(paired['proof_sha256'])!=expected_proofs:
        raise ValueError('Every extraction and independent replay proof must be bound')
    for name,digest in extraction.read(extraction.REPORT/'plan.json')['source_sha256'].items():bind(bindings,ROOT/name,digest)
    for name,digest in paired['proof_sha256'].items():bind(bindings,ROOT/name,digest)
    manifest_row=paired['manifest'];manifest_path=Path(manifest_row['path'])
    if manifest_path!=extraction.ART/'cache-manifest.json':raise ValueError('Exact owned paired-cache path required')
    bind(bindings,manifest_path,manifest_row['sha256'])
    cm=extraction.read(manifest_path)
    if cm['users']!=480189 or cm['items']!=17770:raise ValueError('Complete canonical coordinate universe required')
    for name,digest in cm['references'].items():bind(bindings,ROOT/name,digest)
    for row in cm['arrays'].values():bind(bindings,ROOT/row['path'],row['sha256'])
    for phase in ('known','qualifying'):
        proof=extraction.read(extraction.STUDY/(phase+'-independent-verification.json'))
        extraction.validate_result('verify_'+phase,proof,plan_sha)
        if (proof['manifest_sha256']!=manifest_row['sha256'] or proof['all_actual_cache_bits_match'] is not True
                or proof['all_query_feature_bits_match'] is not True or proof['query_rows']!=extraction.COUNTS[phase]):
            raise ValueError('Every original native query and cache bit required')
    # The new scalar uses the original known projection, not the larger native
    # split3 holdout. Canonical known/qualifying addresses enter separately.
    scalar_paths={'probex':provider.rbm.ART/'known.npy','qual':provider.rbm.ART/'qualifying-native.npy'}
    coordinates={q:{k:'artifacts/upstream-repro/fresh/data/'+q+'/'+k+'.npy'
                    for k in ('user_idxs','item_idxs')} for q in COUNTS}
    for path in scalar_paths.values():
        relative=str(path.relative_to(ROOT))
        if relative not in bindings:raise ValueError('Original paired scalar hashes must be inherited, never rebased')
    for fields in coordinates.values():
        if any(name not in bindings for name in fields.values()):raise ValueError('Original canonical query hashes required')
    for name,digest in m['prepared_sha256'].items():bind(bindings,preceding.RUN/name,digest)
    for path in (mp,vp,ip,pp,ivp,done_path,paired_path,extraction.REPORT/'plan.json',Path(__file__)):bind(bindings,path)
    records=provider.original.verify_bindings(bindings,guard)
    result=dict(parent=m,parent_plan=parent_plan,input_manifest=im,cache_manifest=manifest_row,
                scalar_paths={q:str(path.relative_to(ROOT)) for q,path in scalar_paths.items()},
                coordinates=coordinates,source_sha256=bindings)
    parent_pairs(result)
    return result,records


@contextmanager
def open_query(query,declaration,records,guard,*,retain_scalar,latent):
    if query not in COUNTS:raise ValueError('Exact canonical query required')
    operators.dimensions(retain_scalar,latent)
    row=declaration['cache_manifest'];relative=str(Path(row['path']).relative_to(ROOT))
    with ExitStack() as stack:
        contexts,_=stack.enter_context(cache.load(ROOT,relative,row['sha256'],guard))
        coordinates=stack.enter_context(admission.arrays(ROOT,declaration['coordinates'][query],guard,records))
        p=declaration['scalar_paths'][query]
        prediction=stack.enter_context(admission.mapped(ROOT/p,records[p]))
        if prediction.shape!=(COUNTS[query],):raise ValueError('Complete original projected query scalar required')
        if any(value.shape!=prediction.shape for value in coordinates.values()):raise ValueError('All canonical row addresses required')
        parent_path=preceding.parent.REPORT/'manifest.json'
        if str(parent_path.relative_to(ROOT)) not in records:
            raise ValueError('Full B391 parent declaration must be in authenticated closure')
        parent_manifest=extraction.read(parent_path)
        parent=preceding.Inputs(query,parent_manifest,declaration['parent']['retained'])
        stack.callback(parent.close)
        context=contexts[PHASE[query]] if latent else None
        try:
            yield QueryInputs(parent,prediction,coordinates['user_idxs'],coordinates['item_idxs'],context,guard,
                              retain_scalar=retain_scalar,latent=latent)
        finally:
            if provider.original.verify_bindings(declaration['source_sha256'],guard)!=records:
                raise ValueError('Complete source identity must remain unchanged through query use')

"""Declaration and provenance for a metadata-only RBM400 observer successor.

No launch or signal on import. The retirement program must finish before a
successor plan can be created. Original scientific modules remain immutable.
"""
import ast
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import upstream_factored_rbm_execution as original
import upstream_factored_rbm_execution_native as previous

ROOT = original.ROOT
REPORT = ROOT/'experiments/upstream-rbm400-native-observer-v2-recovery'
PREVIOUS_PLAN_SHA = 'd39966728bf04320ced9a6ce4b450d50d4118b93dad1eb11e85c92d48263f6bb'
ORIGINAL_PLAN_SHA = '48fc8377c22e2b5bbbe9a1ef22881c4b0ee94a4ff0c3b2aedbcfe99548c980c2'
REPLACEMENTS = {
    'upstream_factored_rbm_execution': 'upstream_factored_rbm_execution_native_v2',
    'upstream_factored_rbm_worker': 'upstream_factored_rbm_worker_native_v2',
    'upstream_local_rbm_blend': 'upstream_local_rbm_blend_native_v2',
    'verify_upstream_local_rbm_blend': 'verify_upstream_local_rbm_blend_native_v2',
    'upstream_local_rbm_blend_queue': 'upstream_local_rbm_blend_queue_native_v2',
}


def original_plan():
    path = original.REPORT/'plan.json'
    if original.inputs.sha(path) != ORIGINAL_PLAN_SHA:
        raise ValueError('Exact original producer plan required')
    plan = original.read(path)
    original.check_sources(plan)
    witness = original.read(ROOT/'experiments/upstream-rbm400-native-observer-candidate/verification.json')
    if original.inputs.sha(original.REPORT/'first-scientific-baseline.json') != witness['original_baseline_sha256']:
        raise ValueError('Original first scientific baseline changed')
    return plan


def previous_plan():
    path=previous.REPORT/'plan.json'
    if original.inputs.sha(path)!=PREVIOUS_PLAN_SHA:
        raise ValueError('Exact preceding observer attempt required')
    plan=original.read(path);previous.check_sources(plan)
    if (previous.REPORT/'first-scientific-baseline.json').read_bytes()!=(original.REPORT/'first-scientific-baseline.json').read_bytes():
        raise ValueError('Both attempts must share original baseline bytes')
    return plan


def inherited_baseline():
    original_plan()
    return original.read(original.REPORT/'first-scientific-baseline.json')


def copy_baseline(path):
    original_plan()
    payload = (original.REPORT/'first-scientific-baseline.json').read_bytes()
    original.durable_payload(path, lambda stream: stream.write(payload), ROOT, original.metadata_guard)
    if Path(path).read_bytes() != payload:
        raise ValueError('Inherited baseline bytes differ')


def equivalent_worker():
    old = (ROOT/'scripts/upstream_factored_rbm_worker.py').read_text()
    new = (ROOT/'scripts/upstream_factored_rbm_worker_native_v2.py').read_text()
    expected = old.replace('import upstream_factored_rbm_execution as execution',
                           'import upstream_factored_rbm_execution_native_v2 as execution')
    if ast.dump(ast.parse(expected)) != ast.dump(ast.parse(new)):
        raise ValueError('Scientific worker differs beyond its execution import')


def retirement():
    value = original.read(REPORT/'retirement.json')
    if (value['state'] != 'complete_exact_metadata_only_producer_and_idle_consumer_retirement'
            or value['original_plan_sha256'] != ORIGINAL_PLAN_SHA
            or value['previous_plan_sha256'] != PREVIOUS_PLAN_SHA
            or value['learned_state_existed'] is not False
            or value['original_baseline_sha256'] != original.inputs.sha(original.REPORT/'first-scientific-baseline.json')
            or value['all_original_processes_terminal'] is not True
            or value['both_lanes_verified_free'] is not True):
        raise ValueError('Exact completed metadata-only retirement required')
    for key, captured in value['processes'].items():
        actual = original.identity.process(captured['pid'])
        same = original.same_child(captured, actual) if key == 'worker' else original.identity.same_owner(captured, actual)
        if same:
            raise ValueError('Retired original process is still alive')
    return value


def recovery_fields():
    return dict(original_plan_sha256=ORIGINAL_PLAN_SHA,
                previous_plan_sha256=PREVIOUS_PLAN_SHA,
                original_baseline_sha256=original.inputs.sha(original.REPORT/'first-scientific-baseline.json'),
                retirement_sha256=original.inputs.sha(REPORT/'retirement.json'),
                metadata_pass_restarted=True, learned_state_discarded=False,
                numerical_recipe_changed=False, resource_limits_changed=False,
                baseline_reset=False,accounting_scope='exact_original_vmmap_research_predicate')


def check_declaration(plan):
    old = original_plan()
    preceding = previous_plan()
    equivalent_worker()
    retirement()
    if plan.get('recovery') != recovery_fields() or plan['policy'] != old['policy']:
        raise ValueError('Exact inherited baseline/caps and explicit recovery provenance required')
    for name, digest in old['source_sha256'].items():
        if plan['source_sha256'].get(name) != digest:
            raise ValueError('Original full source closure omitted or changed')
    for name,digest in preceding['source_sha256'].items():
        if plan['source_sha256'].get(name)!=digest:
            raise ValueError('Preceding attempt source closure omitted or changed')


def frozen_plan(execution):
    old = original_plan()
    preceding = previous_plan()
    equivalent_worker()
    retirement()
    files = [ROOT/'scripts'/f'{name}.py' for name in REPLACEMENTS.values()]
    files += [Path(__file__), ROOT/'scripts/upstream_research_footprints_native.py',
              ROOT/'scripts/upstream_research_footprints_native_original_scope.py',
              ROOT/'scripts/test_upstream_research_footprints_native_original_scope.py',
              ROOT/'OWN_RBM400_ORIGINAL_SCOPE_CORRECTION.md',
              ROOT/'scripts/upstream_rbm400_metadata_retirement_v2.py',
              ROOT/'scripts/test_upstream_rbm400_observer_recovery_v2.py',
              ROOT/'scripts/test_upstream_factored_rbm_execution_native_v2.py',
              ROOT/'scripts/test_upstream_local_rbm_blend_native_v2.py',
              ROOT/'scripts/test_upstream_research_footprints_native.py',
              ROOT/'OWN_RBM400_OBSERVER_RECOVERY_PLAN.md',
              ROOT/'OWN_RBM400_NATIVE_OBSERVER_V2_EXECUTION.md',
              ROOT/'OWN_LOCAL_RBM_NATIVE_OBSERVER_V2_BLEND_PLAN.md',
              ROOT/'experiments/upstream-rbm400-original-scope-candidate/verification.json',
              ROOT/'experiments/upstream-rbm400-original-scope-candidate/review.json',
              REPORT/'protocol-verification.json', REPORT/'review.json',
              REPORT/'retirement.json', original.REPORT/'plan.json',
              previous.REPORT/'plan.json',
              original.REPORT/'first-scientific-baseline.json']
    sources = dict(old['source_sha256'])
    sources.update(preceding['source_sha256'])
    sources.update({str(p.relative_to(ROOT)): original.inputs.sha(p) for p in files})
    return dict(state='declared_original_gpu_rbm400_metadata_only_native_observer_successor',
                created=original.stamp(), configuration=asdict(execution.Config()),
                phases=list(execution.PHASES), environment=execution.ENV,
                source_sha256=sources,
                libraries={n: importlib.metadata.version(n) for n in ('numpy', 'torch')},
                policy=old['policy'], recovery=recovery_fields(),
                scope='Restart only incomplete input hashing. Full original15+15 recipe, inputs, numerical sources, caps and first baseline retained. New namespaces and honest downstream producer provenance; no learned state discarded.')

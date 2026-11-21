"""Complete-source, read-only actual-model extraction adapter; no CLI or owner.

Only a separately reviewed supervisor may call this with its live guard after
both the original RBM and own16 queues complete. This adapter never fits models,
opens qualifying rating vectors, writes caches, or admits a neural provider.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import upstream_local_rbm_blend_metal as original
import upstream_local_rbm_blend_queue_metal as consumer
import upstream_factored_rbm_worker_metal as worker
import upstream_factored_rbm_admission as admission
import upstream_rbm400_latent_extraction as core

rbm = original.rbm
inputs = original.inputs
ROOT = rbm.ROOT


def require_consumer_completion(completion, canonical, canonical_sha256, freeze_sha256):
    """Lifecycle only: recovered numerical scores never influence extraction."""
    if (completion['state'] != 'complete_local_rbm16_known_only_calibration_freeze_full_replay_and_independent_recovered_audit'
            or completion['verification_sha256'] != canonical_sha256
            or completion['qualifying_labels_used_for_fitting_or_selection'] is not False
            or canonical['state'] != 'complete_independent_local_rbm16_full_source_models_coefficients_and_frozen_group_scores'
            or canonical['qualifying_labels_used_for_fitting_or_selection'] is not False
            or canonical['all_six_vectors_independently_verified'] is not True
            or canonical['full_inherited_source_closure_rehashed'] is not True
            or canonical['simplex_supports_examined'] != 65535
            or canonical['ridge_equations'] != 17
            or canonical['primary'] != original.CANDIDATES[0]
            or canonical['bindings'][str((original.REPORT / 'frozen.json').relative_to(ROOT))] != freeze_sha256):
        raise ValueError('Full original own16 lifecycle must complete before latent extraction')


def sources(guard):
    if not callable(guard):
        raise ValueError('External original ownership/resource guard required')
    guard()
    bindings = {}
    # Validates all15+15 epochs, exact source/recipe, complete saved models,
    # actual MPS proof, every native query and independent paired replay.
    original.gpu_sources(bindings)
    consumer.check_sources(rbm.read(consumer.REPORT / 'plan.json'))
    done_path = consumer.REPORT / 'completion.json'
    canonical_path = original.REPORT / 'independent-canonical-verification.json'
    frozen_path = original.REPORT / 'frozen.json'
    done, canonical, frozen = [rbm.read(p) for p in (done_path, canonical_path, frozen_path)]
    require_consumer_completion(done, canonical, inputs.sha(canonical_path), inputs.sha(frozen_path))
    if canonical['source_sha256'] != frozen['source_sha256']:
        raise ValueError('Original own16 fully frozen source closure required')
    for name, digest in canonical['source_sha256'].items():
        original.bind(bindings, ROOT / name, digest)
    for name, digest in canonical['bindings'].items():
        original.bind(bindings, ROOT / name, digest)
    plan = rbm.read(consumer.REPORT / 'plan.json')
    for name, digest in plan['source_sha256'].items():
        original.bind(bindings, ROOT / name, digest)
    for path in (done_path, canonical_path, consumer.REPORT / 'plan.json'):
        original.bind(bindings, path)
    # This shared verifier rejects recovered-outcome files in fitting closure.
    records = original.verify_bindings(bindings, guard)
    guard()
    return bindings, records


@contextmanager
def terminal_payload(path, expected, guard):
    """Deserialize the authenticated FD and preserve the original learned file."""
    import torch
    path = Path(path)
    guard()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or path.resolve() != path
                or admission.file_identity(before) != expected['identity']):
            raise ValueError('Exact fully authenticated original terminal file required')
        record = torch.load(stream, map_location='cpu', weights_only=True)
        try:
            guard()
            yield record
        finally:
            guard()
            if (admission.file_identity(before) != admission.file_identity(os.fstat(stream.fileno()))
                    or admission.file_identity(before) != admission.file_identity(path.stat())):
                raise ValueError('Original terminal file changed during extraction')


def extract_phase(phase, guard):
    """Replay and extract a complete phase under an external owned8GiB gate.

    No CPU mode exists here. Synthetic CPU tests target the separate core and
    FD/lifecycle contracts, never authorize a production call to this adapter.
    """
    if phase not in ('known', 'qualifying'):
        raise ValueError('Exact original complete phase required')
    if any(os.environ.get(k) != v for k, v in rbm.ENV.items()):
        raise ValueError('Original runtime environment required before Torch import')
    bindings, records = sources(guard)
    torch = worker.torch_runtime()
    model_path = rbm.ART / (phase + '.pt')
    native_path = rbm.ART / (phase + '-native.npy')
    relative = lambda p: str(p.relative_to(ROOT))
    try:
        with worker.reader_for(phase, guard) as (reader, _, declaration, data_sha):
            with terminal_payload(model_path, records[relative(model_path)], guard) as record:
                with admission.mapped(native_path, records[relative(native_path)]) as expected:
                    value = core.extract(record, reader, phase, data_sha,
                                         declaration['exact_config'], expected, guard, 'mps')
        torch.mps.synchronize()
        value['metadata']['original_full_source_and_own16_lifecycle_verified'] = True
        # External cache/provider declaration and independently repeated feature
        # extraction still required: never change production_provider_admitted.
        return value, bindings
    finally:
        # Preserve integrity checks even if model decoding/inference fails.
        original.verify_bindings(bindings, guard)

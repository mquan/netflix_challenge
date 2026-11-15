---
kind: workflow
source: events.jsonl
sequence: 10
---

# Problem-specific workflow

Version: 1

## Problem frame

This is a plan-only research deliverable. Assess the existing day-gated household-persona proposal against recorded experiments and source notes; do not implement or score a model.

## Immediate objective

Audit the existing draft's claim that the proposed method is untried and that its stated development gap follows from the recorded results.

## Evidence and validation

Trace method comparisons to the workspace research logs and code inventory. Recalculate the 0.85725 prize threshold and the stated development gap from recorded scores, keeping the Phase 0 transfer offset explicitly provisional. Check that primary-research citations and the proposed validation protocol are represented accurately in the available workspace material. Submit the resulting document to the registered external evaluator; treat its result as the success signal.

## Checkpoints

### prior_work — Establish whether the proposed recurring-persona mechanism is distinct from recorded user-day factors, session features, and sequence models.

- Signal: qualitative
- Depends on: 
- Verify: Map each closest prior method to its workspace record and identify the specific constraint or inference step absent from it; flag any matching prior implementation.

### target_math — Make the target, best recorded development score, and remaining gap auditable.

- Signal: proxy
- Depends on: 
- Verify: Recompute the threshold and gap from recorded values and label any train-plus-probe transfer estimate as provisional rather than observed for the current bank.

### plan_document — Produce one coherent, research-grounded method and falsifiable implementation and validation plan in CONVERGENT_SOLUTION.md.

- Signal: qualitative
- Depends on: prior_work, target_math
- Verify: Inspect the document for one named method, prior-work audit, supported arithmetic and citations, explicit data-use boundaries, stopping rules, and no claim of an unmeasured gain.

### external_check — Obtain the registered success signal for the plan-only deliverable.

- Signal: metric
- Depends on: plan_document
- Verify: Run the registered netflix-plan-only-check evaluator against the document and inspect its reported result without treating the plan as a verified model improvement.

## Strategy selection

Prefer correcting a material unsupported claim in the existing draft over adding another candidate method. Preserve one proposed mechanism and use recorded experiments to distinguish it from prior work.

## Feedback and learning

If the audit finds a prior implementation or unsupported numerical claim, revise the document before evaluation. If the external check rejects the document, use its specific findings to revise the affected checkpoint and reassess the method.

## Revision trigger

Revise this workflow if the proposed method is found in prior work, the available records cannot support the stated gap, or evaluator feedback shows that these checkpoints miss a requirement.

## Uncertainties

The existing document is untracked; its cited papers have not been independently checked in this workflow. The Phase 0 transfer estimate comes from a smaller prediction bank, and no hidden qualifying score or persona-model result exists.

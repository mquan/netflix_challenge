# Deterministic Metal reduction: full-device probe

The original-scope v2 producer completed full input admission (all63 array
schemas and91 bound files, both phases, known map and zero qualifying targets).
Its subsequent actual MPS preflight failed before training because Torch2.14
rejects `index_add_` with deterministic algorithms enabled. Retain that failed
attempt. No learned state exists. Do not disable determinism, use warn-only,
enable CPU numerical fallback or relax the original tolerances.

This separately versioned candidate replaces only duplicate-index floating
additions in the RBM kernel and predictor. All CD equations, tiles, visible
states, momentum, decay, dimensions and original full15+15 recipe remain the
same; a future production declaration must explicitly name this numerical
implementation. No frozen original source is edited and no old result is
promoted as a new result.

The Metal reduction assigns one GPU thread to each destination scalar. It
starts with that scalar's existing value and visits its source rows in stable
original order. There are no atomic floating additions or competing writers.
CPU sorting uses integer addresses only; learned floating values stay on MPS.
This uses the actual installed `torch.mps.compile_shader` API, documented at
https://docs.pytorch.org/docs/main/generated/torch.mps.compile_shader.html .

The owned probe has ONLY the mps_preflight phase, no production data read and
no fitting. It inherits original live owner/watchdog/two-lane gates, exact
original baseline bytes,8GiB worker/24GiB aggregate physical and RSS caps,
compressor and128MiB swap-growth limits and18.75GiB disk floor/reservation.
Resource pressure preserves the same worker/frame; the original supervisor
alone controls3stable-sample continuation. Source changes or terminal errors
preserve the attempt without automatic retry or baseline reset.

Keep strict deterministic algorithms, MPS fallback0 and fast-math0. Require
fresh Torch import after dispatch gates. The original small/cold-user check
and FULL required stress remain:500users,17770items,400hidden,50factors,
117570training rows,118069conditional rows,1000queries and17770rows for the
heavy user. Original tolerances2e-5/2e-5 (small) and5e-4/5e-5 (stress) remain.
Compare all gradients, momentum updates, expectations, visible draws and raw
queries with the independent float64 reference. Require bitwise repeated
gradients and three bitwise prediction repetitions on the actual device.

CPU/address and source-equivalence checks are preliminary. Only a source-bound
reviewed actual guarded MPS run passing BOTH complete cases admits this device
math candidate. It still does not establish full-data training, saved models,
paired replay or any score. Production assembly and the original known-only
own16/full recovered audits remain separate requirements. Both recovered
groups currently miss0.85725; no score gain is claimed here.

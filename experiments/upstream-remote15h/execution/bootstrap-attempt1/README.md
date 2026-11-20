# Bootstrap attempt 1 (root artifacts/upstream-remote15h) — build failed, preserved

- 2026-11-20 ~03:46Z `prepare` copied verified source, linked data, then
  `cargo build --release --locked --features blas --bin fit --bin split_3` failed
  at link: `Undefined symbols: _cblas_sdot, _cblas_sgemm` (see build.log).
- Cause: the helper's `environment()` whitelist drops `RUSTFLAGS`; the earlier
  upstream repro on this machine linked BLAS through
  `RUSTFLAGS='-C link-arg=-framework -C link-arg=Accelerate'`
  (netflix_upstream_repro scripts/upstream_repro.py:62).
- The split was never run; no preparation.json was written. The partial root is
  left untouched on disk per the handoff ("never rerun prepare over a partial").
- Repair: fresh owned root `artifacts/upstream-remote15h-r2`, with
  `artifacts/upstream-remote15h-r2/.cargo/config.toml` (copied here) adding the same
  Accelerate link args. Cargo discovers it from the parent of the native tree, so
  the hash-verified source files, the frozen plan and the helper script are unchanged.

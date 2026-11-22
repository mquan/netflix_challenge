# Complete context data replay with aligned source declarations

The original full data assembly is retained byte for byte. Its separate
verifier77143 failed at the query reader's final source-set comparison. The
verifier had copied the base source declaration and added the original data
and verifier code files to its records, but had not put that complete source
set back into the declaration passed to the reader. The reader rehashed a
strict subset and compared it to the larger record dictionary. This is a
source-set bookkeeping error; it is not evidence that data bytes changed.

The new verifier sets declaration['source_sha256'] to the exact fully bound
original manifest source set before capturing records or opening a query.
No source entry is removed and no identity, digest or numerical check is
relaxed. The original verifier is reproduced failing on a stricter temporary
full-width fixture, while the corrected verifier passes complete assembly,
independent statistics/solves/query replay and output-tamper rejection. The
original mock reader omitted this comparison and its initial fixture already
included the extra source files, which masked the production error.

Use a new owned single-phase replay. Read the original805KB manifest and all
seven original prepared outputs; do not rewrite or reassemble them. Bind the
old161-file CPU execution contract before taking the new snapshot. Preserve
its failure, release, manifest, assembly receipt and log. Independently visit
both full query sets, every raw scalar/context operator, all group statistics,
every eligible ridge equation and every stored held prediction. Publish the
new proof and completion only into the new namespace. All numerical helper
function bodies remain identical to the original independent verifier.

Keep the original3GiB worker cap,24GiB aggregate physical/RSS caps, both lanes,
18.75GiB disk floor/full remaining reservation, original byte-identical swap
baseline,128MiB growth limit, compressor thresholds, three stable admission
samples and same-process stack-preserving resource suspension. This is replay
only: no training, numerical score promotion, qualifying-label access or
model-budget change. The unchanged assembled outputs are not admitted until
the complete new independent proof is produced.

After completion, versioned downstream source contracts must explicitly adopt
this new proof. The old waiting neural/bag/own17 owners and original failure
remain historical records. Do not fabricate their predecessor completion or
edit their frozen plans. No new neural model has yet been fitted in this chain.

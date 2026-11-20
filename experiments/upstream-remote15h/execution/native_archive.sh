#!/bin/zsh
# Lossless archive of all generated native outputs of one root (preds_3, root logs/receipts,
# row map, cargo config, fit/split_3 binaries). Excludes linked raw datasets and the generated
# train_3/probe_3 rating arrays. Splits into <2 GiB parts, writes a membership/hash manifest,
# and uploads parts to the draft transfer release. Usage: native_archive.sh ROOT_NAME LABEL MANIFEST_DIR
set -euo pipefail
W=~/workspace/netflix_remote15h
ROOT=$W/artifacts/$1; LABEL=$2; MDIR=$W/$3
TAG=remote15h-own-20261120; REPO=mquan/netflix_challenge
OUT=$W/artifacts/native-archives/$LABEL
[[ -e $OUT ]] && { echo "exists: $OUT" >&2; exit 1; }
mkdir -p $OUT $MDIR
cd $ROOT
LIST=$OUT/members.txt
{ find source/preds_3 -type f; find . -maxdepth 1 -type f ! -name operation.lock; find .cargo -type f;
  echo source/target/release/fit; echo source/target/release/split_3; } | sed 's#^\./##' | sort > $LIST
tar -cf - -T $LIST | gzip -1 | split -b 1900m - $OUT/$LABEL.tar.gz.part-
~/workspace/netflix_upstream_repro/.venv/bin/python - $ROOT $LIST $OUT $MDIR/native-archive-$LABEL.json $TAG <<'EOF'
import hashlib, json, sys, datetime
from pathlib import Path
root, lst, out, dest, tag = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
members = {m: dict(sha256=sha(Path(root) / m), bytes=(Path(root) / m).stat().st_size)
           for m in Path(lst).read_text().split()}
parts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(Path(out).glob('*.part-*'))}
json.dump(dict(observed=datetime.datetime.now(datetime.timezone.utc).isoformat(), root=root,
  release_tag=tag, reassemble='cat PARTS(sorted) | gunzip | tar -x', parts=parts,
  members=members, excluded='linked raw datasets (data/train,probe,fulltrain,qual,probex) and generated data/train_3, data/probe_3 rating arrays',
  qualifying_labels_included=False), open(dest, 'w'), indent=2)
print(len(members), 'members', len(parts), 'parts')
EOF
for p in $OUT/*.part-*; do gh release upload $TAG $p --repo $REPO; done
echo "uploaded $LABEL"

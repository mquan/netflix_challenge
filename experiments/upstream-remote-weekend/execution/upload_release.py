"""Upload one producer's archive parts + prefixed index to the draft release.
Never overwrites: an existing same-name asset must match size and SHA-256."""
import hashlib, json, shutil, subprocess, sys, tempfile, datetime as dt
from pathlib import Path
REPO = 'mquan/netflix_challenge'; TAG = 'remote-weekend-20261120'
TARGET = 'research/remote-weekend-handoff'
job, arc, manifest = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()
def gh(*a, check=True): return subprocess.run(['gh', *a], capture_output=True, text=True, check=check)
index = json.loads((arc / 'index.json').read_text())
stage = Path(tempfile.mkdtemp(prefix=job + '-'))
named = stage / f'{job}-index.json'; shutil.copyfile(arc / 'index.json', named)
files = [arc / p['file'] for p in index['parts']] + [named]
if gh('release', 'view', TAG, '--repo', REPO, check=False).returncode:
    gh('release', 'create', TAG, '--repo', REPO, '--draft', '--target', TARGET,
       '--title', 'Remote weekend transfer (rbm400pf, tx300r, rbm800p)',
       '--notes', 'Transfer-only draft: own original-source training outputs. No raw ratings. Do not publish.')
rel = json.loads(gh('release', 'view', TAG, '--repo', REPO, '--json', 'assets').stdout)
existing = {a['name']: a for a in rel['assets']}
records = []
for f in files:
    rec = dict(name=f.name, bytes=f.stat().st_size, sha256=sha(f))
    if f.name in existing:
        if existing[f.name]['size'] != rec['bytes']:
            raise SystemExit(f'differing asset already exists, not overwriting: {f.name}')
        d = Path(tempfile.mkdtemp()); gh('release', 'download', TAG, '--repo', REPO, '-p', f.name, '-D', str(d))
        if sha(d / f.name) != rec['sha256']:
            raise SystemExit(f'differing asset already exists, not overwriting: {f.name}')
        shutil.rmtree(d); rec['state'] = 'identical_existing_verified'
    else:
        gh('release', 'upload', TAG, str(f), '--repo', REPO); rec['state'] = 'uploaded'
    records.append(rec)
rel = json.loads(gh('release', 'view', TAG, '--repo', REPO, '--json', 'assets,isDraft,url').stdout)
sizes = {a['name']: a['size'] for a in rel['assets']}
for r in records:
    if sizes.get(r['name']) != r['bytes']: raise SystemExit('post-upload size mismatch: ' + r['name'])
manifest.write_text(json.dumps(dict(observed=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    job=job, release_tag=TAG, release_draft=rel['isDraft'], release_url=rel['url'], target=TARGET,
    assets=records, numeric_arrays_in_git=False), indent=1) + '\n')
print('uploaded', job, len(records), 'assets')

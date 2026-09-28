#!/usr/bin/env python3
"""Broaden the re-encode verification of the 26 NEW sources to ~300 cells.

The 73-cell pass sampled one cell per source and every one reproduced bit-identically. That is
good evidence the PIPELINE was sound, but with 26 of 9,100 new cells checked (0.3%) it cannot
rule out a defect confined to particular cells. The original 47 sources are already at 100%
coverage via their `repro_err` column, so this targets the new 26 only.

SAMPLING - stratified, not just random:
  (a) one random CRF for every (source, preset) pair -> 26 x 10 = 260 cells.
      This guarantees complete source x preset coverage; a defect localised to one preset on one
      source cannot hide.
  (b) the remaining ~40 drawn at random over the whole new-26 grid, so CRF coverage is not
      limited to the 260 stratified picks.
  Seeded and reproducible. Cells already verified are skipped.

A cell passes if a fresh encode reproduces the stored VMAF for all three models and the stored
bitrate. SVT-AV1 at fixed settings is deterministic, so agreement should be exact.

Appends to verify73.csv, so re-running unify_dataset73.py folds every verified cell into
`repro_err` automatically.

    python3 verify300_new.py          # ~300 cells
    python3 verify300_new.py 150      # smaller
    python3 verify300_new.py 300 --plan
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import os, csv, json, shutil, subprocess, sys, time
import numpy as np, pandas as pd

SVT = _paths.tool('svt')
FF = _paths.tool('ffmpeg')
FFP = _paths.tool('ffprobe')
LOCAL = _paths.tool('sources')
WINF  = '/mnt/f/Pristine Videos'
TMP   = _paths.out('_verify'); STAGE = _paths.out('_verify_stage')
for d in (TMP, STAGE): os.makedirs(d, exist_ok=True)
OUT   = _paths.data('verify73.csv')
MODELS = [('v061', 'vmaf_v0.6.1'), ('v1', 'vmaf_v1.0.16_3d0h'), ('v1_hfr', 'vmaf_v1.0.16_hfr_3d0h')]
SEED  = 20260926
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
PLAN  = '--plan' in sys.argv
MIN_FREE_GB = 2.5

def free_gb(p='/mnt/c'):
    try: st = os.statvfs(p); return st.f_bavail * st.f_frsize / 1e9
    except Exception: return 999.0

d = pd.read_csv(_paths.out('new_dense_v1.csv'))
d['era'] = 'new26'

rng = np.random.default_rng(SEED)
picks = []
# (a) complete source x preset coverage
for (s, p), g in d.groupby(['source', 'preset'], sort=True):
    picks.append(g.index[rng.integers(0, len(g))])
strat = len(picks)
# (b) the remainder at random over the whole grid
pool = d.index.difference(picks)
extra = max(0, NSAMP - len(picks))
if extra and len(pool):
    picks += list(rng.choice(pool, size=min(extra, len(pool)), replace=False))
samp = d.loc[sorted(set(picks))].copy()
samp['why'] = np.where(np.arange(len(samp)) < strat, 'source_x_preset', 'random')

# drop anything already verified
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        done.add((r['source'], int(r['preset']), int(r['crf'])))
samp['k'] = list(zip(samp.source, samp.preset.astype(int), samp.crf.astype(int)))
todo = samp[~samp.k.isin(done)].copy()

est = (todo.enc_time_s + todo.score_s).sum()
print(f'sample {len(samp)} cells ({strat} stratified + {len(samp)-strat} random) | '
      f'{samp.source.nunique()} sources | already done {len(samp)-len(todo)} | to run {len(todo)}')
print(f'presets {sorted(samp.preset.unique())} | CRFs {samp.crf.nunique()} distinct')
print(f'recorded cost of the outstanding cells: {est/3600:.2f} h '
      f'(those timings were contended; treat as an upper bound)')
if PLAN:
    print(todo.groupby('preset').size().to_string())
    sys.exit(0)
if not os.path.isdir(WINF):
    sys.exit(f'F: not mounted ({WINF} absent). Nothing measured.')

def stage(name):
    dst = f'{STAGE}/{name}.y4m'
    if os.path.exists(dst): return dst
    src = f'{WINF}/{name}.y4m'
    if not os.path.exists(src): return None
    shutil.copyfile(src, dst)
    return dst if os.path.exists(dst) else None

def dur(path):
    o = subprocess.run([FFP, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                        path], capture_output=True, text=True).stdout.strip()
    try: return float(o)
    except ValueError: return 0.0

def score(ref, dist, model):
    log = f'{TMP}/s3.json'
    if os.path.exists(log): os.remove(log)
    r = subprocess.run([FF, '-hide_banner', '-nostdin', '-i', dist, '-i', ref, '-lavfi',
                        f'[0:v][1:v]libvmaf=model=version={model}:log_fmt=json:log_path={log}:n_threads=8',
                        '-f', 'null', '-'], capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.exists(log): return None
    return json.load(open(log))['pooled_metrics']['vmaf']['mean']

isnew = not os.path.exists(OUT)
fh = open(OUT, 'a', newline=''); w = csv.writer(fh)
if isnew:
    w.writerow(['source', 'era', 'preset', 'crf', 'why', 'stored_v061', 'fresh_v061',
                'stored_v1', 'fresh_v1', 'stored_v1_hfr', 'fresh_v1_hfr',
                'stored_kbps', 'fresh_kbps', 'd_v061', 'd_v1', 'd_v1_hfr', 'd_kbps_pct']); fh.flush()

t0 = time.monotonic(); nfail = 0; ndone = 0
for s, g in todo.groupby('source', sort=True):
    if free_gb() < MIN_FREE_GB:
        print(f'DISK GUARD {free_gb():.2f} GB - stopping cleanly', flush=True); break
    staged = None
    ref = f'{LOCAL}/{s}.y4m'
    if not os.path.exists(ref):
        staged = stage(s); ref = staged
    if not ref or not os.path.exists(ref):
        print(f'   MISSING SOURCE {s}', flush=True); continue
    D = dur(ref)
    px = int(g.iloc[0].width) * int(g.iloc[0].height)
    lp = ['--lp', '4'] if px >= 3840 * 2000 else []      # bitstream-identical; caps memory
    try:
        for _, r in g.sort_values(['preset', 'crf']).iterrows():
            dist = f'{TMP}/v3.ivf'
            e = subprocess.run([SVT, '-i', ref, '--preset', str(int(r.preset)),
                                '--crf', str(int(r.crf)), *lp, '-b', dist],
                               capture_output=True, stdin=subprocess.DEVNULL)
            if e.returncode != 0 or not os.path.exists(dist) or os.path.getsize(dist) == 0:
                print(f'   ENCFAIL {s} p{r.preset} c{r.crf} rc={e.returncode}', flush=True)
                if os.path.exists(dist): os.remove(dist)
                continue
            fresh = {k: score(ref, dist, m) for k, m in MODELS}
            kb = os.path.getsize(dist) * 8 / 1000 / D if D > 0 else None
            os.remove(dist)
            dv = [None if fresh[k] is None else fresh[k] - r[f'vmaf_{k}']
                  for k in ('v061', 'v1', 'v1_hfr')]
            dkb = None if not kb else (kb - r.bitrate_kbps) / r.bitrate_kbps * 100
            w.writerow([s, 'new26', int(r.preset), int(r.crf), r.why,
                        r.vmaf_v061, fresh['v061'], r.vmaf_v1, fresh['v1'],
                        r.vmaf_v1_hfr, fresh['v1_hfr'], r.bitrate_kbps,
                        round(kb, 3) if kb else '',
                        *[round(x, 6) if x is not None else '' for x in dv],
                        round(dkb, 4) if dkb is not None else '']); fh.flush()
            bad = any(x is not None and abs(x) > 1e-3 for x in dv) or (dkb is not None and abs(dkb) > 0.01)
            nfail += bad; ndone += 1
            if bad:
                print(f'   MISMATCH {s} p{int(r.preset)} c{int(r.crf)} '
                      f'dv061 {dv[0]:+.2e} dv1 {dv[1]:+.2e} dkbps {dkb:+.4f}%', flush=True)
    finally:
        if staged and os.path.exists(staged): os.remove(staged)
    print(f'[{s[:42]:42}] {ndone}/{len(todo)} cells | {nfail} mismatch | '
          f'{(time.monotonic()-t0)/60:.1f} min | C: {free_gb():.1f} GB', flush=True)

fh.close()
print(f'VERIFY300_COMPLETE {ndone} cells, {nfail} mismatch(es) in '
      f'{(time.monotonic()-t0)/60:.1f} min', flush=True)

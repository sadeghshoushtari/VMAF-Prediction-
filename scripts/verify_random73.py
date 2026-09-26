#!/usr/bin/env python3
"""Re-encode randomly sampled cells from source and compare against the stored dataset.

This is the only check that can catch a systematic measurement error. The internal-consistency
audit can prove the tables are well-formed and self-coherent; it cannot prove the numbers in them
came from the encodes they claim to. That requires redoing the measurement from the raw clip.

WHY IT MATTERS HERE: the original 47 sources carry a `repro_err` column - every one of their
16,450 rows was checked against a fresh re-score (max 2.6e-4). The 26 NEW sources have no such
column and have never been verified. This closes that gap.

SAMPLING: seeded and reproducible. Every one of the 73 sources gets at least one cell, then the
remaining draws are spread across preset bands so fast and slow presets are both covered. Sampling
is random within those constraints - not hand-picked, and not biased toward cheap cells, because
cheap cells are exactly where an error would be least likely to matter.

A cell passes if the fresh encode reproduces the stored VMAF for all three models and the stored
bitrate. SVT-AV1 at fixed settings is deterministic, so agreement should be near-exact; the
original run's tolerance was 1e-4 on v0.6.1 (which is stored rounded) and 0.0 on v1/v1_hfr.

    python3 verify_random73.py            # the default 50-cell sample
    python3 verify_random73.py 20         # smaller/faster
    python3 verify_random73.py 50 --plan  # print the sample and its cost estimate, measure nothing

-> verify73.csv
"""
import os, csv, json, shutil, subprocess, sys, time
import numpy as np, pandas as pd

V     = os.path.expanduser('~/vmaf')
SVT   = f'{V}/tools/SVT-AV1/Bin/Release/SvtAv1EncApp'
FF    = os.path.expanduser('~/ffmpeg-v1test/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg')
FFP   = FF.replace('bin/ffmpeg', 'bin/ffprobe')
LOCAL = f'{V}/sources'
WINF  = '/mnt/f/Pristine Videos'
TMP   = f'{V}/_verify'; STAGE = f'{V}/_verify_stage'
for d in (TMP, STAGE): os.makedirs(d, exist_ok=True)
OUT   = f'{V}/verify73.csv'
DATA  = next((p for p in [r'C:/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data',
                          '/mnt/c/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data']
              if os.path.exists(f'{p}/dense_vmaf_v1.csv')), None)
MODELS = [('v061', 'vmaf_v0.6.1'), ('v1', 'vmaf_v1.0.16_3d0h'), ('v1_hfr', 'vmaf_v1.0.16_hfr_3d0h')]
SEED   = 20260926
NSAMP  = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
PLAN   = '--plan' in sys.argv
MIN_FREE_GB = 2.5

def free_gb(p='/mnt/c'):
    try: st = os.statvfs(p); return st.f_bavail * st.f_frsize / 1e9
    except Exception: return 999.0

# ---------------------------------------------------------------- the dataset
old = pd.read_csv(f'{DATA}/dense_vmaf_v1.csv').rename(columns={'bitrate_kbps_carried': 'bitrate_kbps'})
old['era'] = 'old47'
new = pd.read_csv(f'{V}/new_dense_v1.csv'); new['era'] = 'new26'
C = ['source', 'width', 'height', 'preset', 'crf', 'vmaf_v061', 'vmaf_v1', 'vmaf_v1_hfr',
     'bitrate_kbps', 'enc_time_s', 'score_s', 'era']
d = pd.concat([old[C], new[C]], ignore_index=True)

# ---------------------------------------------------------------- the sample
rng = np.random.default_rng(SEED)
picks = []
# (a) one random cell per source, so no source goes unverified
for s, g in d.groupby('source', sort=True):
    picks.append(g.index[rng.integers(0, len(g))])
# (b) remaining draws spread over preset bands, random within band
bands = [(1, 3), (4, 7), (8, 10)]
rest = max(0, NSAMP - len(picks))
for k in range(rest):
    lo, hi = bands[k % len(bands)]
    pool = d.index[(d.preset >= lo) & (d.preset <= hi)].difference(picks)
    if len(pool): picks.append(pool[rng.integers(0, len(pool))])
samp = d.loc[sorted(set(picks))].copy()
samp['why'] = np.where(samp.index.isin(picks[:d.source.nunique()]), 'per_source', 'preset_band')

est = (samp.enc_time_s + samp.score_s).sum()
print(f'sample: {len(samp)} cells | {samp.source.nunique()} sources '
      f'| presets {sorted(samp.preset.unique())} | era {dict(samp.era.value_counts())}')
print(f'recorded cost of these cells: {est/60:.1f} min (timings were contended, treat as an upper bound)')
if PLAN:
    print(samp[['source', 'era', 'preset', 'crf', 'enc_time_s', 'score_s']].to_string(index=False))
    sys.exit(0)

HAVE_F = os.path.isdir(WINF)
if not HAVE_F:
    local_ok = sum(1 for s in samp.source.unique() if os.path.exists(f'{LOCAL}/{s}.y4m'))
    print(f'\nNOTE: F: is not mounted ({WINF} absent).')
    print(f'      {local_ok} of {samp.source.nunique()} sampled sources are on local disk and WILL be verified now.')
    print(f'      The other {samp.source.nunique()-local_ok} need the drive; re-run this script once it is back '
          f'and it will resume with only those.\n', flush=True)

# ---------------------------------------------------------------- primitives
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
    log = f'{TMP}/s.json'
    if os.path.exists(log): os.remove(log)
    r = subprocess.run([FF, '-hide_banner', '-nostdin', '-i', dist, '-i', ref, '-lavfi',
                        f'[0:v][1:v]libvmaf=model=version={model}:log_fmt=json:log_path={log}:n_threads=8',
                        '-f', 'null', '-'], capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.exists(log): return None
    return json.load(open(log))['pooled_metrics']['vmaf']['mean']

isnew = not os.path.exists(OUT)
fh = open(OUT, 'a', newline=''); w = csv.writer(fh)
if isnew:
    w.writerow(['source', 'era', 'preset', 'crf', 'why',
                'stored_v061', 'fresh_v061', 'stored_v1', 'fresh_v1',
                'stored_v1_hfr', 'fresh_v1_hfr', 'stored_kbps', 'fresh_kbps',
                'd_v061', 'd_v1', 'd_v1_hfr', 'd_kbps_pct']); fh.flush()
done = set()
if not isnew:
    for r in csv.DictReader(open(OUT)): done.add((r['source'], r['preset'], r['crf']))

t0 = time.monotonic(); nfail = 0
for s, g in samp.groupby('source', sort=True):
    g = g[~g.apply(lambda r: (r.source, str(r.preset), str(r.crf)) in done, axis=1)]
    if not len(g): continue
    if free_gb() < MIN_FREE_GB:
        print(f'DISK GUARD {free_gb():.2f} GB - stopping cleanly', flush=True); break
    staged = None
    ref = f'{LOCAL}/{s}.y4m'
    if not os.path.exists(ref):
        if not HAVE_F:
            print(f'   PENDING (needs F:) {s[:44]}', flush=True); continue
        staged = stage(s); ref = staged
    if not ref or not os.path.exists(ref):
        print(f'   MISSING SOURCE {s}', flush=True); continue
    D = dur(ref)
    r0 = g.iloc[0]
    try:
        # 4K at a slow preset exhausts this box's 7.8 GB and the encode gets OOM-killed
        # (it prints "Encoding" and leaves a 0-byte file). --lp caps the thread count and with it
        # the memory high-water mark. VERIFIED SAFE: the bitstream is byte-identical with and
        # without --lp (md5 match at --lp 2, --lp 4 and unset), so this cannot cause a false
        # mismatch. The wall protocol already used --lp 4 on 4K for the same reason.
        px = int(r0.width) * int(r0.height)
        lp = ['--lp', '4'] if px >= 3840 * 2000 else []
        for _, r in g.iterrows():
            dist = f'{TMP}/v.ivf'
            e = subprocess.run([SVT, '-i', ref, '--preset', str(int(r.preset)),
                                '--crf', str(int(r.crf)), *lp, '-b', dist],
                               capture_output=True, stdin=subprocess.DEVNULL)
            if e.returncode != 0 or not os.path.exists(dist) or os.path.getsize(dist) == 0:
                tail = (e.stdout + e.stderr).decode(errors='ignore').strip().splitlines()[-1:]
                print(f'   ENCFAIL {s} p{r.preset} c{r.crf} rc={e.returncode} '
                      f'size={os.path.getsize(dist) if os.path.exists(dist) else "none"} {tail}', flush=True)
                if os.path.exists(dist): os.remove(dist)
                continue
            fresh = {k: score(ref, dist, m) for k, m in MODELS}
            kb = os.path.getsize(dist) * 8 / 1000 / D if D > 0 else None
            os.remove(dist)
            dv = [None if fresh[k] is None else fresh[k] - r[f'vmaf_{k}'] for k in ('v061','v1','v1_hfr')]
            dkb = None if not kb else (kb - r.bitrate_kbps) / r.bitrate_kbps * 100
            w.writerow([s, r.era, int(r.preset), int(r.crf), r.why,
                        r.vmaf_v061, fresh['v061'], r.vmaf_v1, fresh['v1'],
                        r.vmaf_v1_hfr, fresh['v1_hfr'], r.bitrate_kbps,
                        round(kb, 3) if kb else '',
                        *[round(x, 6) if x is not None else '' for x in dv],
                        round(dkb, 4) if dkb is not None else '']); fh.flush()
            bad = any(x is not None and abs(x) > 1e-3 for x in dv) or (dkb is not None and abs(dkb) > 0.01)
            nfail += bad
            print(f'   {s[:30]:30} p{int(r.preset):>2} c{int(r.crf):>2} '
                  f'dv061 {dv[0]:+.2e} dv1 {dv[1]:+.2e} dkbps {dkb:+.4f}%'
                  f'{"   <-- MISMATCH" if bad else ""}', flush=True)
    finally:
        if staged and os.path.exists(staged): os.remove(staged)
    print(f'[{s[:40]}] done | {(time.monotonic()-t0)/60:.1f} min', flush=True)

fh.close()
print(f'VERIFY_COMPLETE {nfail} mismatch(es) in {(time.monotonic()-t0)/60:.1f} min', flush=True)

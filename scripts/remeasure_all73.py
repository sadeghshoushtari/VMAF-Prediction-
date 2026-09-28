#!/usr/bin/env python3
"""Quiet-machine re-measurement for the enlarged 73-source dataset.

Two jobs, done per source so each staged clip is copied from F: exactly once:

  A. GOOD-3 content features (luma_mean, luma_std, ti_mean) at full and half resolution,
     for the 26 NEW sources only - the original 47 already have them in halfres_feat.csv.
     These are required before the new sources can be modelled at all.

  B. The SYMMETRIC per-video wall, for ALL 73 sources in ONE session:
     one full-res p10/c55 encode + its own libvmaf pass (both targets), plus the matching
     symmetric cost of the 2-knot probe (2 half-res encodes + their own libvmaf passes).

WHAT IS DELIBERATELY DIFFERENT FROM THE 2026-09-05 RUN

  * time.monotonic() everywhere, not time.time(). This is MEASUREMENT TRAP #3: the system clock
    on this box resyncs BACKWARDS, and taking min-over-reps then SELECTS the backward jump, so
    every previously published second is understated. monotonic cannot go backwards.
  * enc time is recorded BOTH ways - SVT's self-reported "Total Encoding Time" (what the old run
    stored) and true wall clock. The old protocol kept only the self-reported figure, which
    understates real cost. Keeping both means old numbers stay reproducible and new ones honest.
  * All 73 in one session, so x-wall ratios are internally consistent. Seconds do not reproduce
    across sessions on this machine (same wall measured 8.005s and 6.975s); ratios reproduce to
    0.3-5%. Ratios are the quantity to report.

Everything else is held to the original protocol: 3 consecutive reps per source, MIN taken,
4K gets --lp 4, preset 10, crf 55 for the wall, knots at crf 33 and 51.

Sources are always read from LOCAL disk. Reading a reference off /mnt/f (drvfs, ~100 MB/s vs
~1.2 GB/s local) would inflate every encode time and make the new sources incomparable to the 47.

Resumable; writes incrementally. -> wall73.csv, newfeat73.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import os, re, csv, json, shutil, subprocess, sys, time
import numpy as np, pandas as pd

SVT = _paths.tool('svt')
FF = _paths.tool('ffmpeg')
FFP = _paths.tool('ffprobe')
LOCAL = _paths.tool('sources')                      # the original 47 live here
WINF  = '/mnt/f/Pristine Videos'            # all 73 live here
TMP   = _paths.out('_remeas'); STAGE = _paths.out('_remeas_stage')
for d in (TMP, STAGE): os.makedirs(d, exist_ok=True)

WALL = _paths.data('wall73.csv')
FEAT = _paths.out('newfeat73.csv')
N       = 3
TOTAL_T = re.compile(r'Total Encoding Time:\s*(\d+)\s*ms')
MODELS  = [('v061', 'vmaf_v0.6.1'), ('v1', 'vmaf_v1.0.16_3d0h')]
MIN_FREE_GB = 2.5

def free_gb(p='/mnt/c'):
    try: st = os.statvfs(p); return st.f_bavail * st.f_frsize / 1e9
    except Exception: return 999.0

# ------------------------------------------------------------------ inventory
old = pd.read_csv(_paths.data('dense_vmaf_73.csv')).groupby('source').agg(
        w=('width', 'first'), h=('height', 'first')).reset_index()
old['px'] = old.w * old.h; old['isnew'] = False
new = pd.read_csv(_paths.out('new_dense_v1.csv')).groupby('source').agg(
        w=('width', 'first'), h=('height', 'first')).reset_index()
new['px'] = new.w * new.h; new['isnew'] = True
meta = pd.concat([old, new], ignore_index=True).sort_values('px').reset_index(drop=True)
assert meta.source.nunique() == 73, f'expected 73 sources, got {meta.source.nunique()}'

def opener(path, header):
    isnew = not os.path.exists(path)
    fh = open(path, 'a', newline=''); w = csv.writer(fh)
    if isnew: w.writerow(header); fh.flush()
    return fh, w

wfh, wr = opener(WALL, ['source', 'px', 'isnew', 'stage', 'crf', 'run',
                        'enc_self_s', 'enc_wall_s', 'score_v061_s', 'score_v1_s', 'scale_s'])
ffh, fr = opener(FEAT, ['source', 'px', 'scale', 'luma_mean', 'luma_std', 'ti_mean', 'sec'])
wall_done = set(r['source'] for r in csv.DictReader(open(WALL))) if os.path.getsize(WALL) > 0 else set()
feat_done = set((r['source'], r['scale']) for r in csv.DictReader(open(FEAT))) if os.path.getsize(FEAT) > 0 else set()

# ------------------------------------------------------------------ primitives
def stage_from_F(name):
    """Copy one clip from F: to local disk. Timings must never read across drvfs."""
    dst = f'{STAGE}/{name}.y4m'
    if os.path.exists(dst): return dst
    src = f'{WINF}/{name}.y4m'
    if not os.path.exists(src): return None
    shutil.copyfile(src, dst)
    return dst if os.path.exists(dst) else None

def dims(path):
    o = subprocess.check_output([FFP, '-v', 'error', '-select_streams', 'v:0',
                                 '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path])
    return [int(x) for x in o.decode().strip().split(',')[:2]]

def enc(ref, out, crf, lp):
    """Returns (svt_self_reported_s, true_wall_s). monotonic clock."""
    t = time.monotonic()
    r = subprocess.run([SVT, '-i', ref, '--preset', '10', '--crf', str(crf), *lp, '-b', out],
                       capture_output=True, stdin=subprocess.DEVNULL)
    wc = time.monotonic() - t
    if r.returncode != 0 or not os.path.exists(out): return None, None
    m = TOTAL_T.search((r.stdout + r.stderr).decode(errors='ignore'))
    return (int(m.group(1)) / 1000 if m else wc), wc

def score(ref, dist, model):
    t = time.monotonic()
    r = subprocess.run([FF, '-hide_banner', '-nostdin', '-i', dist, '-i', ref, '-lavfi',
                        f'[0:v][1:v]libvmaf=model=version={model}:n_threads=8', '-f', 'null', '-'],
                       capture_output=True, stdin=subprocess.DEVNULL)
    return time.monotonic() - t if r.returncode == 0 else None

def good3(src, half):
    """luma-only decode + the three GOOD-3 statistics, as features.py defines them."""
    w, h = dims(src)
    cmd = [FF, '-v', 'error', '-nostdin', '-i', src]
    if half:
        cmd += ['-vf', 'scale=trunc(iw/4)*2:trunc(ih/4)*2']
        w, h = (w // 4) * 2, (h // 4) * 2
    cmd += ['-pix_fmt', 'gray', '-f', 'rawvideo', '-']
    ysz = w * h; t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=ysz, stdin=subprocess.DEVNULL)
    lm, ls, ti = [], [], []; prev = None
    while True:
        b = p.stdout.read(ysz)
        if len(b) < ysz: break
        Y = np.frombuffer(b, np.uint8).reshape(h, w).astype(np.float32)
        lm.append(Y.mean()); ls.append(Y.std())
        if prev is not None: ti.append((Y - prev).std())
        prev = Y
    p.stdout.close(); p.wait()
    return dict(luma_mean=float(np.mean(lm)), luma_std=float(np.mean(ls)),
                ti_mean=float(np.mean(ti)) if ti else 0.0), time.monotonic() - t0

# ------------------------------------------------------------------ main
t_start = time.monotonic()
print(f'{len(meta)} sources | wall done {len(wall_done)} | feature rows done {len(feat_done)}', flush=True)

for i, m in meta.iterrows():
    s = m.source
    need_wall = s not in wall_done
    need_feat = m.isnew and ((s, 'full') not in feat_done or (s, 'half') not in feat_done)
    if not need_wall and not need_feat:
        continue
    if free_gb() < MIN_FREE_GB:
        print(f'DISK GUARD {free_gb():.2f} GB - stopping cleanly', flush=True); break

    staged = None
    ref = f'{LOCAL}/{s}.y4m'
    if not os.path.exists(ref):
        staged = stage_from_F(s); ref = staged
    if not ref or not os.path.exists(ref):
        print(f'   MISSING {s}', flush=True); continue

    try:
        # ---- A. GOOD-3 features, new sources only -------------------------
        if need_feat:
            for tag, half in (('full', False), ('half', True)):
                if (s, tag) in feat_done: continue
                st, sec = good3(ref, half)
                fr.writerow([s, int(m.px), tag, round(st['luma_mean'], 5), round(st['luma_std'], 5),
                             round(st['ti_mean'], 5), round(sec, 4)]); ffh.flush()
            print(f'   feats {s[:40]}', flush=True)

        # ---- B. the symmetric wall ---------------------------------------
        if need_wall:
            is4k = m.px >= 3840 * 2000
            lp = ['--lp', '4'] if is4k else []
            halfref = f'{TMP}/h.y4m'
            t = time.monotonic()
            subprocess.run([FF, '-hide_banner', '-nostdin', '-y', '-i', ref, '-vf',
                            'scale=trunc(iw/4)*2:trunc(ih/4)*2', '-strict', '-1', halfref],
                           capture_output=True, stdin=subprocess.DEVNULL)
            scale_s = time.monotonic() - t

            for run in range(N):
                d = f'{TMP}/f.ivf'
                e_self, e_wall = enc(ref, d, 55, lp)
                if e_self is None:
                    print(f'   ENCFAIL {s}', flush=True); break
                s1 = score(ref, d, MODELS[0][1]); s2 = score(ref, d, MODELS[1][1])
                os.remove(d)
                wr.writerow([s, int(m.px), int(m.isnew), 'wall_full', 55, run,
                             round(e_self, 4), round(e_wall, 4),
                             round(s1, 4) if s1 else '', round(s2, 4) if s2 else '',
                             round(scale_s, 4)]); wfh.flush()
                for crf in (33, 51):
                    d = f'{TMP}/k.ivf'
                    k_self, k_wall = enc(halfref, d, crf, [])
                    if k_self is None: continue
                    k1 = score(halfref, d, MODELS[0][1]); k2 = score(halfref, d, MODELS[1][1])
                    os.remove(d)
                    wr.writerow([s, int(m.px), int(m.isnew), 'knot_half', crf, run,
                                 round(k_self, 4), round(k_wall, 4),
                                 round(k1, 4) if k1 else '', round(k2, 4) if k2 else '',
                                 '']); wfh.flush()
            if os.path.exists(halfref): os.remove(halfref)
    finally:
        if staged and os.path.exists(staged):
            os.remove(staged)                      # delete as you go

    el = time.monotonic() - t_start
    print(f'[{i+1}/{len(meta)}] {s[:40]:40} {"NEW" if m.isnew else "   "} '
          f'{m.px/1e6:5.2f} Mpx | {el/60:6.1f} min elapsed | C: {free_gb():.1f} GB', flush=True)

wfh.close(); ffh.close()
print(f'REMEASURE_COMPLETE in {(time.monotonic()-t_start)/60:.1f} min', flush=True)

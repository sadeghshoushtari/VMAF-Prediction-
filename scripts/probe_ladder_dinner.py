#!/usr/bin/env python3
"""Does a higher-resolution probe fix DinnerSceneCropped?

THE PROBLEM

DinnerSceneCropped is the worst-predicted source in the 73-source corpus: MAE 12-13 against a
corpus average near 1.7. The cause is not the model. Its HALF-RESOLUTION probe overestimates the
full-resolution quality by +10.20 VMAF, where the corpus mean is +0.97 (sd 1.49) - a z of +6.2.
Downscaling removes exactly the dark fine detail the full-res encode struggles with, so the probe
reports a much easier clip than the encoder actually sees.

WHAT THE EXISTING DATA ALREADY SAYS (45 sources with both half and 2/3 probes)

  * a low quality ceiling predicts a large half-res gap: spearman -0.405, p=0.006
  * but whether 2/3 REPAIRS the gap is not predictable from ceiling (p=0.13), luma (p=0.63)
    or ti (p=0.48); among the worst-gap sources the repair ranges from 0.41 to 4.02 VMAF
  * the worst gap among those 45 is +4.14 - DinnerScene at +10.20 is 2.5x beyond the observed
    range, so this cannot be settled by extrapolation

PRE-REGISTERED INTERPRETATION (fixed before measuring, so the result cannot be read to taste)

  Going half -> 2/3 shrinks |gap| by 0.51 on average across the 45 (34/45 improve). If
  DinnerScene is merely a typical source with a big gap, 2/3 lands it near +9.4.

    gap at 2/3 still > ~8      -> resolution is NOT the mechanism. The probe is wrong about this
                                  content for some other reason; report it as a named limitation.
    gap at 2/3 between 2 and 8 -> partial; higher rungs may help but the probe stays unreliable.
    gap at 2/3 < ~2            -> resolution IS the mechanism, and probe resolution needs to adapt
                                  to content. That would be a general finding, not a patch.

Three rungs are measured (0.5625, 2/3, 0.8) so the trend across resolution is visible rather than
a single point. Scoring is against the downscaled reference at each rung, matching the shipped
probe protocol exactly.

-> probe_ladder_dinner.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import os, csv, json, shutil, subprocess, sys, time
import numpy as np, pandas as pd

SVT = _paths.tool('svt')
FF = _paths.tool('ffmpeg')
FFP = _paths.tool('ffprobe')
WINF = '/mnt/f/Pristine Videos'
TMP  = _paths.out('_ladder'); STAGE = _paths.out('_ladder_stage')
for d in (TMP, STAGE): os.makedirs(d, exist_ok=True)
OUT = _paths.out('probe_ladder_dinner.csv')
MODELS = [('v061', 'vmaf_v0.6.1'), ('v1', 'vmaf_v1.0.16_3d0h'), ('v1_hfr', 'vmaf_v1.0.16_hfr_3d0h')]
KNOTS = [20, 33, 42, 51, 63]

SOURCES = ['DinnerSceneCropped_1920x1080_2997fps_10bit_420']
# two controls from the NEW 26 so the rung effect can be separated from this source's peculiarity:
# one ordinary camera clip, one that is also dark-ish but predicts normally.
SOURCES += ['Boat_1920x1080_5994_10bit_420', 'Motorcycle_1920x1080_30fps_8bit']

# scale filters: (tag, ffmpeg vf). Each keeps even dimensions.
RUNGS = [('f05625', 'scale=trunc(iw*0.5625/2)*2:trunc(ih*0.5625/2)*2'),
         ('twothird', 'scale=trunc(iw*2/6)*2:trunc(ih*2/6)*2'),
         ('f08', 'scale=trunc(iw*0.8/2)*2:trunc(ih*0.8/2)*2')]

if not os.path.isdir(WINF):
    sys.exit(f'F: not mounted ({WINF} absent). Nothing measured.')

def dims(p):
    o = subprocess.check_output([FFP, '-v', 'error', '-select_streams', 'v:0',
                                 '-show_entries', 'stream=width,height', '-of', 'csv=p=0', p])
    return [int(x) for x in o.decode().strip().split(',')[:2]]

def dur(p):
    o = subprocess.run([FFP, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p],
                       capture_output=True, text=True).stdout.strip()
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
    w.writerow(['source', 'rung', 'lr_w', 'lr_h', 'preset', 'crf',
                'vmaf_lr_v061', 'vmaf_lr_v1', 'vmaf_lr_v1_hfr', 'bitrate_kbps']); fh.flush()
done = set()
if not isnew:
    for r in csv.DictReader(open(OUT)): done.add((r['source'], r['rung'], int(r['crf'])))

t0 = time.monotonic()
for s in SOURCES:
    staged = f'{STAGE}/{s}.y4m'
    if not os.path.exists(staged):
        src = f'{WINF}/{s}.y4m'
        if not os.path.exists(src):
            print(f'MISSING {s}', flush=True); continue
        shutil.copyfile(src, staged)
    try:
        for tag, vf in RUNGS:
            if all((s, tag, c) in done for c in KNOTS): continue
            ref = f'{TMP}/r.y4m'
            subprocess.run([FF, '-hide_banner', '-nostdin', '-y', '-i', staged, '-vf', vf,
                            '-strict', '-1', ref], capture_output=True, stdin=subprocess.DEVNULL)
            lw, lh = dims(ref); D = dur(ref)
            for c in KNOTS:
                if (s, tag, c) in done: continue
                dist = f'{TMP}/d.ivf'
                e = subprocess.run([SVT, '-i', ref, '--preset', '10', '--crf', str(c), '-b', dist],
                                   capture_output=True, stdin=subprocess.DEVNULL)
                if e.returncode != 0 or not os.path.exists(dist) or os.path.getsize(dist) == 0:
                    print(f'   ENCFAIL {s} {tag} c{c}', flush=True); continue
                sc = {k: score(ref, dist, m) for k, m in MODELS}
                kb = os.path.getsize(dist) * 8 / 1000 / D if D > 0 else None
                os.remove(dist)
                w.writerow([s, tag, lw, lh, 10, c, sc['v061'], sc['v1'], sc['v1_hfr'],
                            round(kb, 3) if kb else '']); fh.flush()
                print(f'   {s[:28]:28} {tag:9} {lw}x{lh} c{c:>2}  v1 {sc["v1"]:.3f}', flush=True)
            if os.path.exists(ref): os.remove(ref)
    finally:
        if os.path.exists(staged): os.remove(staged)
    print(f'[{s[:40]}] done | {(time.monotonic()-t0)/60:.1f} min', flush=True)

fh.close()
print(f'LADDER_COMPLETE in {(time.monotonic()-t0)/60:.1f} min', flush=True)

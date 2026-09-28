#!/usr/bin/env python3
"""Complete the dataset: full grid + half-res probes for the 26 AOM-CTC sources added on
2026-09-22, reading the raw clips from F:\\Pristine Videos.

F: is not mounted in WSL, so each source is STAGED one at a time via a Windows-side copy, used,
and deleted. Peak extra disk is one clip (<=0.81 GB) plus one encode. Only the raw videos on F:
are permanent; everything this script creates on the WSL disk is removed as it goes.

Per source:
  A. half-resolution probes, preset 10, CRF 20/33/42/51/63  -> new_lowres_v1.csv
     (a source whose half-res side would fall under VMAF v1's 160 px floor is probed at 2/3)
  B. the full grid, 10 presets x 35 CRFs                    -> new_dense_v1.csv

Resumable: rows already present are skipped. Stops cleanly if C: runs low.
    python3 encode_new_sources.py            # everything
    python3 encode_new_sources.py pilot      # one small source, 2 presets, 2 CRFs
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import os, sys, subprocess, csv, json, time, re

SVT = _paths.tool('svt')
FF = _paths.tool('ffmpeg')
FP  = FF.replace('bin/ffmpeg', 'bin/ffprobe')
WIN = r'F:\Pristine Videos'
STAGE = _paths.out('_stage'); TMP = _paths.out('_newrun')
for d in (STAGE, TMP): os.makedirs(d, exist_ok=True)

DENSE = _paths.out('new_dense_v1.csv')
LOWRES = _paths.out('new_lowres_v1.csv')
MODELS = [('v061','vmaf_v0.6.1'), ('v1','vmaf_v1.0.16_3d0h'), ('v1_hfr','vmaf_v1.0.16_hfr_3d0h')]
CRFS = [20,24,27] + list(range(28,57)) + [58,60,63]
PRESETS = list(range(1,11))
KNOTS = [20,33,42,51,63]
PILOT = len(sys.argv) > 1 and sys.argv[1] == 'pilot'

NEW = """Aerial3200_1920x1080_5994_10bit_420 Boat_1920x1080_5994_10bit_420
DinnerSceneCropped_1920x1080_2997fps_10bit_420 FoodMarket_1920x1080_5994_10bit_420
Motorcycle_1920x1080_30fps_8bit Skater227_1920x1080_30fps TunnelFlag_1920x1080_5994_10bit_420
Vertical_Carnaby_1080x1920_5994 WorldCup_1920x1080_30p WorldCup_far_1920x1080_30p
BlueSky_360p25_v2 FourPeople_480x270_60 Vertical_Bayshore_270x480_2997
CosmosTreeTrunk_sdr_2048x858_25_8bit EuroTruckSimulator2_1920x1080p60_v2 Life_1080p30
Life_1080p30_v2 STARCRAFT_1080p60 Sniper_1920x1080P_30fps_8bit
SolLevanteDragons_sdr_1920x1080_24_10bit
BigBuckBunnyStudio1_1920x1080_60fps_10bit_420_020_0149
MissionControlClip1_1920x1080_60fps_10bit_420_0450_0579 MobileDeviceScreenSharing
SceneComposition_1 SceneComposition_2 Slides2r_1920x1080_30fps_8bit_420""".split()

MIN_FREE_GB = 3.0
def free_gb(p='/mnt/c'):
    try: st = os.statvfs(p); return st.f_bavail*st.f_frsize/1e9
    except Exception: return 999.0

def sh(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)

def stage(name):
    dst = f'{STAGE}/{name}.y4m'
    if os.path.exists(dst): return dst
    sh(['powershell.exe','-NoProfile','-Command',
        f"Copy-Item -LiteralPath '{WIN}\\{name}.y4m' -Destination "
        f"'\\\\wsl.localhost\\Ubuntu-22.04{dst.replace('/', chr(92))}' -Force"])
    return dst if os.path.exists(dst) else None

def probe_dims(path):
    o = sh([FP,'-v','error','-select_streams','v:0','-show_entries','stream=width,height',
            '-of','csv=p=0',path]).stdout.strip().split(',')
    return int(o[0]), int(o[1])

def dur(path):
    o = sh([FP,'-v','error','-show_entries','format=duration','-of','csv=p=0',path]).stdout.strip()
    try: return float(o)
    except ValueError: return 0.0

def score(ref, dist, model):
    log = f'{TMP}/s.json'
    r = sh([FF,'-hide_banner','-nostdin','-i',dist,'-i',ref,'-lavfi',
            f'[0:v][1:v]libvmaf=model=version={model}:log_fmt=json:log_path={log}:n_threads=8',
            '-f','null','-'])
    if r.returncode != 0 or not os.path.exists(log): return None
    return json.load(open(log))['pooled_metrics']['vmaf']['mean']

def opener(path, header):
    new = not os.path.exists(path)
    fh = open(path,'a',newline=''); w = csv.writer(fh)
    if new: w.writerow(header); fh.flush()
    return fh, w

def done_keys(path, cols):
    s = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            s.add(tuple(r[c] for c in cols))
    return s

dfh, dw = opener(DENSE, ['source','width','height','preset','crf','vmaf_v061','vmaf_v1',
                         'vmaf_v1_hfr','bitrate_kbps','enc_time_s','score_s'])
lfh, lw = opener(LOWRES, ['source','lr_w','lr_h','preset','crf','vmaf_lr_v061','vmaf_lr_v1',
                          'vmaf_lr_v1_hfr','bitrate_kbps','enc_s','score_s','scale'])
d_done = done_keys(DENSE, ['source','preset','crf'])
l_done = done_keys(LOWRES, ['source','crf'])

srcs = NEW[:1] if PILOT else NEW
presets = [10,6] if PILOT else sorted(PRESETS, reverse=True)
crfs = [33,51] if PILOT else CRFS
t0 = time.time(); ncell = 0; total = len(srcs)*len(presets)*len(crfs)
print(f'{len(srcs)} sources | {len(presets)}x{len(crfs)} = {len(presets)*len(crfs)} cells each '
      f'| {total} cells total | C: {free_gb():.1f} GB free', flush=True)

for si, s in enumerate(srcs, 1):
    need_d = [(p,c) for p in presets for c in crfs if (s,str(p),str(c)) not in d_done]
    need_l = [c for c in KNOTS if (s,str(c)) not in l_done]
    if not need_d and not need_l:
        print(f'[{si}/{len(srcs)}] {s} already complete', flush=True); continue
    if free_gb() < MIN_FREE_GB:
        print(f'DISK GUARD: {free_gb():.2f} GB free on C: - stopping cleanly', flush=True); break
    print(f'[{si}/{len(srcs)}] staging {s}', flush=True)
    ref = stage(s)
    if not ref: print(f'   STAGE FAILED {s}', flush=True); continue
    W, H = probe_dims(ref); D = dur(ref)

    # ---- A. probes -------------------------------------------------------------
    if need_l:
        half_w, half_h = (W//2)//2*2, (H//2)//2*2
        scl, tag = ('scale=trunc(iw/4)*2:trunc(ih/4)*2','half') if min(half_w,half_h) >= 160 \
                   else ('scale=trunc(iw*2/6)*2:trunc(ih*2/6)*2','twothird')
        lref = f'{TMP}/lr.y4m'
        sh([FF,'-hide_banner','-nostdin','-y','-i',ref,'-vf',scl,'-strict','-1',lref])
        lw_, lh_ = probe_dims(lref)
        for c in need_l:
            dist = f'{TMP}/lp.ivf'; t = time.time()
            r = sh([SVT,'-i',lref,'--preset','10','--crf',str(c),'-b',dist]); enc = time.time()-t
            if r.returncode != 0 or not os.path.exists(dist):
                print(f'   PROBE ENCFAIL {s} c{c}', flush=True); continue
            t = time.time(); sc = {k: score(lref,dist,m) for k,m in MODELS}; sco = time.time()-t
            dd = dur(dist); kb = os.path.getsize(dist)*8/1000/dd if dd > 0 else None
            os.remove(dist)
            lw.writerow([s,lw_,lh_,10,c,sc['v061'],sc['v1'],sc['v1_hfr'],
                         round(kb,3) if kb else None, round(enc,4), round(sco,4), tag]); lfh.flush()
        if os.path.exists(lref): os.remove(lref)
        print(f'   probes done ({tag} {lw_}x{lh_})', flush=True)

    # ---- B. the grid -----------------------------------------------------------
    for (p,c) in need_d:
        if free_gb() < MIN_FREE_GB:
            print(f'DISK GUARD mid-source: {free_gb():.2f} GB - stopping', flush=True); break
        dist = f'{TMP}/g.ivf'; t = time.time()
        r = sh([SVT,'-i',ref,'--preset',str(p),'--crf',str(c),'-b',dist]); enc = time.time()-t
        if r.returncode != 0 or not os.path.exists(dist):
            if os.path.exists(dist): os.remove(dist)
            print(f'   ENCFAIL {s} p{p} c{c}', flush=True); continue
        try:
            t = time.time(); sc = {k: score(ref,dist,m) for k,m in MODELS}; sco = time.time()-t
            kb = os.path.getsize(dist)*8/1000/D if D > 0 else None
        finally:
            if os.path.exists(dist): os.remove(dist)
        dw.writerow([s,W,H,p,c,sc['v061'],sc['v1'],sc['v1_hfr'],
                     round(kb,3) if kb else None, round(enc,3), round(sco,3)]); dfh.flush()
        ncell += 1
        if ncell % 25 == 0:
            el = time.time()-t0
            print(f'   {ncell} cells | last {s[:22]} p{p} c{c} enc {enc:.1f}s score {sco:.1f}s '
                  f'| C: {free_gb():.1f} GB | {el/3600:.2f} h elapsed', flush=True)
    os.remove(ref)                                  # DELETE AS YOU GO: the staged raw clip
    print(f'   {s} finished, staged copy removed | C: {free_gb():.1f} GB free', flush=True)

dfh.close(); lfh.close()
print(f'CAMPAIGN_STOPPED after {ncell} new cells in {(time.time()-t0)/3600:.2f} h', flush=True)

#!/usr/bin/env python3
"""Extend the three COMPETITOR feature sets to the 26 new sources.

The competitor comparison cannot run on 73 sources because all three of its feature tables cover
only the original 47:

    source_features.csv    SITI scalars   used by the VP9-DNN competitor
    vca_full_features.csv  16 VCA stats   used by VCA-RF and by LiteVPNet
    clip_embeddings.npz    CLIP ViT-B/16  used by LiteVPNet

Every definition below is ported verbatim from the script that produced the original table
(features.py, vca_full.py, extract_clip.py). Any drift would silently make the old and new
sources incomparable and invalidate the whole comparison, so the ported code is kept literal
rather than tidied, and a self-check at the end re-computes two ORIGINAL sources and compares
against the published values.

Sources are staged from F: one at a time and deleted immediately, so peak extra disk is one clip.
The CLIP weights are already cached locally; nothing here needs the network.

-> new_source_features.csv, new_vca_full_features.csv, new_clip_embeddings.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import csv, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import sobel, laplace

WINF  = Path('/mnt/f/Pristine Videos')
STAGE = Path(_paths.out('_featstage')); STAGE.mkdir(exist_ok=True)
VCAOUT = Path(_paths.out('vca_out_new')); VCAOUT.mkdir(exist_ok=True)
VCA = _paths.tool('vca')
FF = _paths.tool('ffmpeg')
FFP = _paths.tool('ffprobe')
FFDIR = _paths.tool('ffmpeg')        # the binary extract_clip.py used
SITI_COLS = ['si_mean','si_max','ti_mean','ti_max','luma_mean','luma_std','lap_var_mean','chroma_std','n_frames']
N_FRAMES, SIDE = 16, 224
SELFCHECK = ['BlueSky_360p25', 'Johnny_1280x720_60']     # two ORIGINAL sources, re-measured

import pandas as pd
NEW = sorted(set(pd.read_csv(_paths.data('dense_vmaf_73.csv')).source.unique()))
print(f'{len(NEW)} new sources to extend', flush=True)


def dims(path):
    o = subprocess.check_output([FFP, '-v', 'error', '-select_streams', 'v:0',
                                 '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)])
    w, h = o.decode().strip().split(',')[:2]
    return int(w), int(h)


def siti_for(path):
    """verbatim from features.py::features_for"""
    w, h = dims(path)
    ysz = w * h; csz = (w // 2) * (h // 2); fsz = ysz + 2 * csz
    p = subprocess.Popen([FF, '-v', 'error', '-nostdin', '-i', str(path),
                          '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-'],
                         stdout=subprocess.PIPE, bufsize=fsz)
    si, ti, lmean, lstd, lapv, cstd = [], [], [], [], [], []
    prevY = None; n = 0
    while True:
        buf = p.stdout.read(fsz)
        if len(buf) < fsz: break
        a = np.frombuffer(buf, np.uint8)
        Y = a[:ysz].reshape(h, w).astype(np.float32)
        U = a[ysz:ysz+csz].astype(np.float32); Vv = a[ysz+csz:].astype(np.float32)
        gx = sobel(Y, axis=1); gy = sobel(Y, axis=0)
        si.append(float(np.hypot(gx, gy).std()))
        lapv.append(float(laplace(Y).var()))
        lmean.append(float(Y.mean())); lstd.append(float(Y.std()))
        cstd.append(float(np.concatenate([U, Vv]).std()))
        if prevY is not None: ti.append(float((Y - prevY).std()))
        prevY = Y; n += 1
    p.stdout.close(); p.wait()
    return dict(si_mean=np.mean(si), si_max=np.max(si),
                ti_mean=np.mean(ti) if ti else 0.0, ti_max=np.max(ti) if ti else 0.0,
                luma_mean=np.mean(lmean), luma_std=np.mean(lstd),
                lap_var_mean=np.mean(lapv), chroma_std=np.mean(cstd), n_frames=n)


def _vca_parse(path):
    rows = list(csv.reader(open(path)))
    hdr = [x.strip().lower() for x in rows[0]]; ix = {x: i for i, x in enumerate(hdr)}
    d = np.array([[float(x) for x in r] for r in rows[1:] if r])
    return d, ix


def vca_for(path, stem):
    """verbatim from vca_full.py"""
    def run_block(b):
        out = VCAOUT / f'{stem}_B{b}.csv'
        if not out.exists():
            subprocess.run([VCA, '--input', str(path), '--complexity-csv', str(out),
                            '--block-size', str(b)], capture_output=True, text=True, check=True)
        return _vca_parse(out)
    f = {}
    for b in (8, 16, 32):
        d, ix = run_block(b)
        f[f'vE{b}'] = float(d[:, ix['e']].mean())
        f[f'vh{b}'] = float(d[1:, ix['h']].mean())
    d, ix = run_block(16)
    E, h = d[:, ix['e']], d[1:, ix['h']]
    f['vE16_std'] = float(E.std()); f['vE16_p90'] = float(np.percentile(E, 90))
    f['vh16_std'] = float(h.std()); f['vh16_p90'] = float(np.percentile(h, 90))
    for nm, key in [('vL','l'),('venU','energyu'),('venV','energyv'),
                    ('ventropy','entropy'),('vedge','edgedensity'),('veps','epsilon')]:
        f[nm] = float(d[:, ix[key]].mean())
    return f


import torch, clip
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
torch.set_num_threads(12)
NETS = {}
for key, name in [('vitb32','ViT-B/32'), ('vitb16','ViT-B/16')]:
    m, _ = clip.load(name, device='cpu'); m.eval(); NETS[key] = m
print('CLIP models loaded from cache', flush=True)


def clip_frames(path):
    vf = (f"select='not(mod(n\\,8))',scale=w='if(lt(iw,ih),{SIDE},-2)':h='if(lt(iw,ih),-2,{SIDE})',"
          f"crop={SIDE}:{SIDE}")
    p = subprocess.run([FFDIR, '-v', 'error', '-i', str(path), '-vf', vf, '-vsync', '0',
                        '-frames:v', str(N_FRAMES), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       capture_output=True, check=True)
    a = np.frombuffer(p.stdout, np.uint8)
    n = len(a) // (SIDE*SIDE*3)
    return a[:n*SIDE*SIDE*3].reshape(n, SIDE, SIDE, 3)


def clip_for(path):
    f = clip_frames(path)
    x = torch.from_numpy(f.copy()).permute(0,3,1,2).float().div_(255).sub_(MEAN).div_(STD)
    out = {}
    with torch.no_grad():
        for key, m in NETS.items():
            e = m.encode_image(x).float().numpy()
            out[key] = np.concatenate([e.mean(0), e.std(0)])
    return out


def resolve(name):
    """local copy if present (the original 47), else stage from F:"""
    loc = V / 'sources' / f'{name}.y4m'
    if loc.exists(): return loc, False
    dst = STAGE / f'{name}.y4m'
    if not dst.exists():
        src = WINF / f'{name}.y4m'
        if not src.exists(): return None, False
        shutil.copyfile(src, dst)
    return dst, True


# ---------------------------------------------------------------- run
SF, VF = V/'new_source_features.csv', V/'new_vca_full_features.csv'
CE = V/'new_clip_embeddings.npz'
done_sf = set(pd.read_csv(SF).source) if SF.exists() else set()
done_vf = set(pd.read_csv(VF).source) if VF.exists() else set()
emb = {}
if CE.exists():
    z = np.load(CE, allow_pickle=True)
    emb = {s: (z['vitb32'][i], z['vitb16'][i]) for i, s in enumerate(z['source'])}

sfh = open(SF, 'a', newline=''); sw = csv.writer(sfh)
if not done_sf: sw.writerow(['source'] + SITI_COLS); sfh.flush()
VCOLS = None
vfh = open(VF, 'a', newline=''); vw = csv.writer(vfh)

t0 = time.time()
for i, name in enumerate(NEW + SELFCHECK, 1):
    need = name not in done_sf or name not in done_vf or name not in emb
    if not need:
        print(f'[{i}/{len(NEW)+len(SELFCHECK)}] skip {name}', flush=True); continue
    path, staged = resolve(name)
    if path is None:
        print(f'   MISSING {name}', flush=True); continue
    try:
        if name not in done_sf:
            f = siti_for(path)
            sw.writerow([name] + [f'{f[c]:.4f}' if c != 'n_frames' else str(int(f[c])) for c in SITI_COLS])
            sfh.flush()
        if name not in done_vf:
            g = vca_for(path, name)
            if VCOLS is None:
                VCOLS = list(g.keys())
                if not done_vf: vw.writerow(['source'] + VCOLS); vfh.flush()
            vw.writerow([name] + [f'{g[c]:.5f}' for c in VCOLS]); vfh.flush()
        if name not in emb:
            e = clip_for(path); emb[name] = (e['vitb32'], e['vitb16'])
            np.savez(CE, source=np.array(list(emb.keys())),
                     vitb32=np.stack([emb[k][0] for k in emb]),
                     vitb16=np.stack([emb[k][1] for k in emb]))
    finally:
        if staged and path.exists(): path.unlink()
        for b in (8, 16, 32):
            p = VCAOUT / f'{name}_B{b}.csv'
            if p.exists(): p.unlink()
    print(f'[{i}/{len(NEW)+len(SELFCHECK)}] {name[:44]:44} {(time.time()-t0)/60:6.1f} min', flush=True)

sfh.close(); vfh.close()
print(f'FEATEXT_COMPLETE in {(time.time()-t0)/60:.1f} min', flush=True)

# ---------------------------------------------------------------- self-check
print('\nSELF-CHECK: two ORIGINAL sources re-measured against the published tables')
old_sf = pd.read_csv(_paths.data('source_features_73.csv')).set_index('source')
new_sf = pd.read_csv(SF).set_index('source')
for s in SELFCHECK:
    if s not in new_sf.index: continue
    d = {c: (float(new_sf.loc[s, c]) - float(old_sf.loc[s, c])) for c in SITI_COLS[:-1]}
    worst = max(abs(v) for v in d.values())
    print(f'  {s:28} max |delta| over SITI = {worst:.2e}  ' + ('OK' if worst < 1e-3 else 'MISMATCH'))

#!/usr/bin/env python3
"""Does the feature vector need more than GOOD-3, now that the corpus is 73 sources?

MOTIVATION

Adding the 26 new sources made the ORIGINAL 36 mainstream sources slightly WORSE under the
shipped method (1.6230 -> 1.7225). More training data should not hurt. The likely cause is that
the feature vector cannot tell the model what KIND of source it is looking at, so the 13 new
synthetic/screen sources pull the fit away from camera content without the model being able to
separate them.

Inspecting the shipped feature vector shows three things it does not contain:

    [luma_mean, ti_mean, luma_std, lv_at, lb_at, crf, preset]

  * no RESOLUTION - yet the corpus spans 129,600 to 8,294,400 px, a factor of 64
  * no CONTENT CLASS - camera vs synthetic vs screen content behave very differently under AV1
  * no SOURCE CEILING - although the probe knots already measure it. The source that hurts most
    (DinnerSceneCropped) is the darkest in the corpus and tops out at VMAF 88 where every other
    source reaches 95-100; its knot at the low CRF already says so.

Every feature tested here is FREE: derived from the clip dimensions, the AOM-CTC class map, or
the two knots that were already measured. No extra encoding, no extra scoring, so the cost
accounting is unchanged.

SCOPE

The primary metric is `all73_p1_10` - ALL 73 sources at ALL presets, including the sources that
score worst. The whole corpus is being treated as equally important, so nothing is excluded and
the mainstream subset is reported only for continuity with earlier numbers. A per-class and
per-era breakdown is written alongside, plus the error on DinnerSceneCropped specifically, so a
gain on the average cannot hide a loss on the hard cases.

The interpolant is held at pow_0.5_extrap (the variant established by curve_sweep73_family.py),
so this sweep varies features only.

-> feature_sweep73_results.csv, feature_sweep73_oof/*.npz
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

V    = os.path.expanduser('~/vmaf')
DOLD = next((p for p in [r'C:/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data',
                         '/mnt/c/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data']
             if os.path.exists(f'{p}/fallback_probe.csv')), None)
OUT    = f'{V}/feature_sweep73_results.csv'
OOFDIR = f'{V}/feature_sweep73_oof'; os.makedirs(OOFDIR, exist_ok=True)
N_JOBS, NTREES, SEED = 6, 300, 0
G3   = ['luma_mean', 'ti_mean', 'luma_std']
HARD = 'DinnerSceneCropped_1920x1080_2997fps_10bit_420'

dense  = pd.read_csv(f'{V}/dense_vmaf_73.csv').sort_values(['source','preset','crf']).reset_index(drop=True)
lowres = pd.read_csv(f'{V}/lowres_vmaf_73.csv')
featf  = pd.read_csv(f'{V}/feat_73.csv')
MAIN   = set(open(f'{V}/mainstream_sources_73.txt').read().split())
FB     = pd.read_csv(f'{DOLD}/fallback_probe.csv')

CLS = {}
for line in open(f'{V}/aomctc_class_map.tsv'):
    if '\t' in line and not line.startswith('class\t'):
        c, f = line.rstrip('\n').split('\t')
        CLS[os.path.splitext(f)[0]] = c


def probe_table(target):
    col = {'v061': 'vmaf_lr_v061', 'v1': 'vmaf_lr_v1'}[target]
    t = lowres[['source', 'crf', col, 'bitrate_kbps']].rename(columns={col: 'v'}).dropna(subset=['v'])
    fb = FB.rename(columns={col: 'v'})[['source', 'crf', 'v', 'bitrate_kbps']]
    return pd.concat([t, fb[~fb.source.isin(set(t.source))]], ignore_index=True)


FEATSETS = {
    'F0_base':  [],                       # the shipped vector
    'F1_res':   ['log_px'],
    'F2_class': ['is_syn', 'is_scc'],
    'F3_knot':  ['knot_lo', 'knot_drop'],
    'F4_all':   ['log_px', 'is_syn', 'is_scc', 'knot_lo', 'knot_drop'],
}


def run(target, knots, feats_scale, fset, interp='pow_0.5_extrap'):
    ks = tuple(sorted(knots))
    y_all = (dense.stored_v061 if target == 'v061' else dense.vmaf_v1).values.astype(float)
    f = featf[featf.scale == feats_scale][['source'] + G3]
    t = probe_table(target)
    t = t[t.crf.isin(ks)].copy(); t['llb'] = np.log10(t.bitrate_kbps)
    w_ = t.pivot_table(index='source', columns='crf', values=['v', 'llb'])
    w_.columns = [f'{x}_{z}' for x, z in w_.columns]; w_ = w_.dropna()

    df   = dense.merge(f, on='source').merge(w_.reset_index(), on='source')
    keep = dense.source.isin(set(w_.index)).values
    crf  = df.crf.values.astype(float)
    a, b = df[f'v_{ks[0]}'].values, df[f'v_{ks[1]}'].values
    ba, bb = df[f'llb_{ks[0]}'].values, df[f'llb_{ks[1]}'].values
    w = (crf - ks[0]) / (ks[1] - ks[0])
    lb_at = ba + (bb - ba) * np.clip(w, 0, 1)

    p = 0.5                                     # pow_0.5_extrap
    A = np.maximum(100.0 - a, 0.0) ** p; B = np.maximum(100.0 - b, 0.0) ** p
    lv = 100.0 - np.clip(A + (B - A) * w, 0.0, None) ** (1.0 / p)

    df['log_px']    = np.log10(df.width.values * df.height.values)
    cl              = df.source.map(CLS).fillna('?')
    df['is_syn']    = (cl == 'b1_syn').astype(float)
    df['is_scc']    = (cl == 'b2_scc').astype(float)
    df['knot_lo']   = a                          # the source's near-best achievable quality
    df['knot_drop'] = a - b                       # how fast it degrades across the knot span

    cols = G3 + ['lv_at_', 'lb_at_', 'crf_', 'preset_'] + FEATSETS[fset]
    df['lv_at_'] = lv; df['lb_at_'] = lb_at; df['crf_'] = crf; df['preset_'] = df.preset.values
    X = df[cols].values
    y = y_all[keep]; grp = dense.source.values[keep]

    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, grp):
        m = ExtraTreesRegressor(n_estimators=NTREES, min_samples_leaf=1, max_features=1.0,
                                n_jobs=N_JOBS, random_state=SEED).fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0, 100)

    idx = pd.DataFrame({'s': grp, 'p': df.preset.values, 'c': crf, 'i': np.arange(len(oof))})
    for _, g in idx.groupby(['s', 'p']):
        g = g.sort_values('c'); ix = g.i.values
        oof[ix] = IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(g.c.values, oof[ix])

    scored = ~((df.preset == 10) & (df.crf == 55)).values
    mains  = (df.source.isin(MAIN) & (df.preset >= 6)).values
    ae = np.abs(oof - y)
    res = {'MAE_all73_p1_10': float(ae[scored].mean()),
           'MAE_mainstream_p6_10': float(ae[scored & mains].mean()),
           'MAE_hard_source': float(ae[scored & (grp == HARD)].mean()) if (grp == HARD).any() else np.nan}
    for c in ['a1_4k', 'a2_2k', 'a3_720p', 'a4_360p', 'a5_270p', 'b1_syn', 'b2_scc']:
        msk = scored & (pd.Series(grp).map(CLS).fillna('?').values == c)
        res[f'MAE_{c}'] = float(ae[msk].mean()) if msk.any() else np.nan
    res['n_sources'] = int(pd.unique(grp).size)
    return res, oof, y, grp, scored, mains


PLAN = [(t, k, fs, fset) for t, k, fs in [('v1', (20, 51), 'half'), ('v061', (33, 51), 'full')]
        for fset in FEATSETS]
done = set(pd.read_csv(OUT).config) if os.path.exists(OUT) else set()
print(f'{len(PLAN)} configs | primary metric = all73_p1_10 (ALL 73 sources, nothing excluded)',
      flush=True)

for i, (tgt, kn, fs, fset) in enumerate(PLAN, 1):
    cfg = f'{tgt}|{fset}'
    if cfg in done:
        print(f'[{i}/{len(PLAN)}] skip {cfg}', flush=True); continue
    t0 = time.monotonic()
    try:
        res, oof, y, grp, sc, mn = run(tgt, kn, fs, fset)
    except Exception as e:
        print(f'[{i}/{len(PLAN)}] FAIL {cfg}: {type(e).__name__}: {e}', flush=True); continue
    el = time.monotonic() - t0
    np.savez(f'{OOFDIR}/{cfg.replace("|","__")}.npz', oof=oof, y=y, source=grp,
             scored=sc, mains=mn, config=cfg)
    pd.DataFrame([dict(config=cfg, target=tgt, featset=fset,
                       extra='+'.join(FEATSETS[fset]) or '(none)',
                       minutes=round(el/60, 2), **res)]).to_csv(
        OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(f'[{i}/{len(PLAN)}] {cfg:16} all73 {res["MAE_all73_p1_10"]:.4f} '
          f'main {res["MAE_mainstream_p6_10"]:.4f}  hard {res["MAE_hard_source"]:.2f} '
          f'({el/60:.1f} min)', flush=True)

print('FEATSWEEP_COMPLETE', flush=True)

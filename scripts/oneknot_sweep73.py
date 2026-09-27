#!/usr/bin/env python3
"""Can ONE knot replace two, now that the curve shape is known?

WHY

Cost on the 73-source corpus decomposes as knots 0.760x of the wall + downscale 0.225x +
features 0.091x = 1.010x at the median video. The knot term dominates, and halving it is the
only lever large enough to change the verdict:

    2 knots   aggregate 0.943x   median/video 1.010x   under wall 33/69
    1 knot    aggregate 0.624x   median/video 0.661x   under wall 66/69

WHY IT IS PLAUSIBLE ONLY NOW

With the shipped linear interpolant a single knot is useless: it yields a flat line, carrying no
CRF information at all. The curve-fitting sweep changed that. It established that the VMAF-vs-CRF
curve follows a power law in the distortion domain with a broad optimum near p=0.5, and that the
normalised shape is shared across sources - the canonical test found a spread of 0.07 against a
straight-line error of 0.20. If the SHAPE can be learned from the training sources for free, a
single anchor may be enough to place it, instead of two to define it.

THE THREE SHAPE MODELS (all learned inside the fold, never using the held-out source)

  k1_const   lv(c) = a                                   degenerate control: flat, no shape
  k1_offset  lv(c) = a + (Vbar(c) - Vbar(k))             shift the mean curve onto the knot
  k1_dscale  lv(c) = 100 - (100-a) * Dbar(c)/Dbar(k)     scale the mean DISTORTION curve onto it

`k1_dscale` is the one the curve-fitting result motivates: it assumes every source shares the
shape of distortion versus CRF and differs by a scale factor, which the single knot measures.
`k1_const` is included so that any gain can be attributed to the shape rather than to the anchor.

Knot placement is swept over 33, 42 and 51 because a lone anchor sits at the centre of its own
extrapolation, unlike a pair which brackets the span.

Accuracy only - the cost side is already measured in wall73.csv. Primary metric is all73_p1_10,
every source weighted equally.

-> oneknot_sweep73_results.csv, oneknot_sweep73_oof/*.npz
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
warnings.filterwarnings('ignore')

V    = os.path.expanduser('~/vmaf')
DOLD = next((p for p in [r'C:/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data',
                         '/mnt/c/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data']
             if os.path.exists(f'{p}/fallback_probe.csv')), None)
OUT    = f'{V}/oneknot_sweep73_results.csv'
OOFDIR = f'{V}/oneknot_sweep73_oof'; os.makedirs(OOFDIR, exist_ok=True)
N_JOBS, NTREES, SEED = 6, 300, 0
G3 = ['luma_mean', 'ti_mean', 'luma_std']
HARD = 'DinnerSceneCropped_1920x1080_2997fps_10bit_420'

dense  = pd.read_csv(f'{V}/dense_vmaf_73.csv').sort_values(['source','preset','crf']).reset_index(drop=True)
lowres = pd.read_csv(f'{V}/lowres_vmaf_73.csv')
featf  = pd.read_csv(f'{V}/feat_73.csv')
MAIN   = set(open(f'{V}/mainstream_sources_73.txt').read().split())
FB     = pd.read_csv(f'{DOLD}/fallback_probe.csv')


def probe_table(target):
    col = {'v061': 'vmaf_lr_v061', 'v1': 'vmaf_lr_v1'}[target]
    t = lowres[['source', 'crf', col, 'bitrate_kbps']].rename(columns={col: 'v'}).dropna(subset=['v'])
    fb = FB.rename(columns={col: 'v'})[['source', 'crf', 'v', 'bitrate_kbps']]
    return pd.concat([t, fb[~fb.source.isin(set(t.source))]], ignore_index=True)


def mean_curve(train_src, P):
    """Mean probe curve over TRAINING sources only, as a PCHIP over the measured knot CRFs."""
    p = P.loc[[s for s in P.index if s in train_src]]
    kc = np.array(sorted(p.columns), float)
    vb = p.values.astype(float).mean(0)
    o = np.argsort(kc)
    return PchipInterpolator(kc[o], vb[o], extrapolate=True)


def run(target, knot, feats_scale, kind):
    y_all = (dense.stored_v061 if target == 'v061' else dense.vmaf_v1).values.astype(float)
    f = featf[featf.scale == feats_scale][['source'] + G3]
    t = probe_table(target)
    P = t.pivot_table(index='source', columns='crf', values='v').dropna()      # 5 knots, for shape
    anchor = t[t.crf == knot].set_index('source').v
    br     = t[t.crf == knot].set_index('source').bitrate_kbps
    usable = [s for s in anchor.index if s in set(P.index)]
    df   = dense[dense.source.isin(usable)].merge(f, on='source').reset_index(drop=True)
    keep = dense.source.isin(usable).values
    crf  = df.crf.values.astype(float)
    a    = anchor.reindex(df.source).values
    lb_at = np.log10(br.reindex(df.source).values.astype(float))               # single-knot bitrate

    y   = y_all[keep]; grp = dense.source.values[keep]
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(np.zeros((len(y), 1)), y, grp):
        if kind == 'k1_const':
            lv = a.copy()
        else:
            g = mean_curve(set(grp[tr]), P)
            vb_c = g(crf); vb_k = float(g(knot))
            if kind == 'k1_offset':
                lv = a + (vb_c - vb_k)
            elif kind == 'k1_dscale':
                db_c = np.maximum(100.0 - vb_c, 1e-6); db_k = max(100.0 - vb_k, 1e-6)
                lv = 100.0 - np.maximum(100.0 - a, 0.0) * (db_c / db_k)
            else:
                raise ValueError(kind)
            lv = np.clip(lv, 0.0, 100.0)
        X = np.column_stack([df[G3].values, lv, lb_at, crf, df.preset.values])
        m = ExtraTreesRegressor(n_estimators=NTREES, min_samples_leaf=1, max_features=1.0,
                                n_jobs=N_JOBS, random_state=SEED).fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0, 100)

    idx = pd.DataFrame({'s': grp, 'p': df.preset.values, 'c': crf, 'i': np.arange(len(oof))})
    for _, g2 in idx.groupby(['s', 'p']):
        g2 = g2.sort_values('c'); ix = g2.i.values
        oof[ix] = IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(g2.c.values, oof[ix])

    scored = ~((df.preset == 10) & (df.crf == 55)).values
    mains  = (df.source.isin(MAIN) & (df.preset >= 6)).values
    ae = np.abs(oof - y)
    res = {'MAE_all73_p1_10': float(ae[scored].mean()),
           'MAE_mainstream_p6_10': float(ae[scored & mains].mean()),
           'MAE_hard_source': float(ae[scored & (grp == HARD)].mean()) if (grp == HARD).any() else np.nan,
           'n_sources': int(pd.unique(grp).size)}
    return res, oof, y, grp, scored, mains


PLAN = [(t, k, fs, kind)
        for t, fs in [('v1', 'half'), ('v061', 'full')]
        for k in (33, 42, 51)
        for kind in ('k1_const', 'k1_offset', 'k1_dscale')]
done = set(pd.read_csv(OUT).config) if os.path.exists(OUT) else set()
print(f'{len(PLAN)} one-knot configs | primary metric all73_p1_10 | reference: 2-knot '
      f'sqrt_extrap = 1.7225 (v1) / 1.6530 (v061)', flush=True)

for i, (tgt, knot, fs, kind) in enumerate(PLAN, 1):
    cfg = f'{tgt}|k{knot}|{kind}'
    if cfg in done:
        print(f'[{i}/{len(PLAN)}] skip {cfg}', flush=True); continue
    t0 = time.monotonic()
    try:
        res, oof, y, grp, sc, mn = run(tgt, knot, fs, kind)
    except Exception as e:
        print(f'[{i}/{len(PLAN)}] FAIL {cfg}: {type(e).__name__}: {e}', flush=True); continue
    el = time.monotonic() - t0
    np.savez(f'{OOFDIR}/{cfg.replace("|","__")}.npz', oof=oof, y=y, source=grp,
             scored=sc, mains=mn, config=cfg)
    pd.DataFrame([dict(config=cfg, target=tgt, knot=knot, shape=kind,
                       minutes=round(el/60, 2), **res)]).to_csv(
        OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(f'[{i}/{len(PLAN)}] {cfg:22} all73 {res["MAE_all73_p1_10"]:.4f} '
          f'main {res["MAE_mainstream_p6_10"]:.4f} hard {res["MAE_hard_source"]:.2f} '
          f'({el/60:.1f} min)', flush=True)

print('ONEKNOT_COMPLETE', flush=True)

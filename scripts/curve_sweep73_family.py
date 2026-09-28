#!/usr/bin/env python3
"""Close the curve-fitting question: sweep the whole power family, not three points of it.

WHAT THE EARLIER RUN LEFT OPEN

curve_sweep73.py compared six named interpolants and found sqrt_extrap best. That identified a
winner but did not close the question, because:
  * sqrt is one exponent. Nothing established it is the BEST exponent.
  * sqrt_clamp was never run, so for the winning family the SHAPE and the EXTRAPOLATION were
    never separated - the decomposition that was done for linear and logit.

THE FAMILY

Interpolating in the distortion domain raised to a power p:

    Vhat(c) = 100 - [ (100-a)^p + ((100-b)^p - (100-a)^p) * w ]^(1/p)

with w = (crf-k0)/(k1-k0), clipped to [0,1] for `clamp` and left free for `extrap`.

This is not an arbitrary parametrisation - it CONTAINS the known results exactly:
    p = 1.0  ->  linear     (verified identical to a + (b-a)w)
    p = 0.5  ->  sqrt       (verified identical to the sqrt_extrap of the previous run)
    p -> 0   ->  logdist    (the variant that lost worst)
So the sweep spans the current baseline, the current winner and the worst loser on one axis, and
the response curve over p says whether the winner sits at an optimum or on a slope.

Also included: a single x-axis variant (linear in V against log(CRF) rather than CRF) to check the
family being swept is the right one, since every candidate so far transforms the y axis only.

Everything else is held to the shipped pipeline exactly as in curve_sweep73.py. Results append to
the SAME csv and OOF directory, so one bootstrap can rank all variants together.

-> curve_sweep73_results.csv, curve_sweep73_oof/*.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

OUT    = _paths.out('curve_sweep73_results.csv')
OOFDIR = _paths.out('curve_sweep73_oof'); os.makedirs(OOFDIR, exist_ok=True)

N_JOBS, NTREES, SEED = 6, 300, 0
G3 = ['luma_mean', 'ti_mean', 'luma_std']

dense  = pd.read_csv(_paths.data('dense_vmaf_73.csv')).sort_values(['source','preset','crf']).reset_index(drop=True)
lowres = pd.read_csv(_paths.data('lowres_vmaf_73.csv'))
feat   = pd.read_csv(_paths.data('feat_73.csv'))
MAIN   = set(open(_paths.data('mainstream_sources_73.txt')).read().split())
FB     = pd.read_csv(_paths.data('fallback_probe.csv'))


def probe_table(target):
    col = {'v061': 'vmaf_lr_v061', 'v1': 'vmaf_lr_v1'}[target]
    t = lowres[['source', 'crf', col, 'bitrate_kbps']].rename(columns={col: 'v'}).dropna(subset=['v'])
    fb = FB.rename(columns={col: 'v'})[['source', 'crf', 'v', 'bitrate_kbps']]
    return pd.concat([t, fb[~fb.source.isin(set(t.source))]], ignore_index=True)


def lv_power(a, b, crf, ks, p, extrap):
    w = (crf - ks[0]) / (ks[1] - ks[0])
    if not extrap: w = np.clip(w, 0.0, 1.0)
    A = np.maximum(100.0 - a, 0.0) ** p
    B = np.maximum(100.0 - b, 0.0) ** p
    return 100.0 - np.clip(A + (B - A) * w, 0.0, None) ** (1.0 / p)


def lv_logcrf(a, b, crf, ks, extrap):
    """x-axis variant: straight line in V against log(CRF)."""
    w = (np.log(crf) - np.log(ks[0])) / (np.log(ks[1]) - np.log(ks[0]))
    if not extrap: w = np.clip(w, 0.0, 1.0)
    return a + (b - a) * w


def run(target, knots, feats, kind):
    ks = tuple(sorted(knots))
    y_all = (dense.stored_v061 if target == 'v061' else dense.vmaf_v1).values.astype(float)
    f = feat[feat.scale == feats][['source'] + G3]
    t = probe_table(target)
    t = t[t.crf.isin(ks)].copy(); t['llb'] = np.log10(t.bitrate_kbps)
    w_ = t.pivot_table(index='source', columns='crf', values=['v', 'llb'])
    w_.columns = [f'{x}_{z}' for x, z in w_.columns]; w_ = w_.dropna()

    df   = dense.merge(f, on='source').merge(w_.reset_index(), on='source')
    keep = dense.source.isin(set(w_.index)).values
    crf  = df.crf.values.astype(float)
    a, b = df[f'v_{ks[0]}'].values, df[f'v_{ks[1]}'].values
    ba, bb = df[f'llb_{ks[0]}'].values, df[f'llb_{ks[1]}'].values
    lb_at = ba + (bb - ba) * np.clip((crf - ks[0]) / (ks[1] - ks[0]), 0, 1)

    if kind.startswith('pow'):
        p = float(kind.split('_')[1]); extrap = kind.endswith('extrap')
        lv = lv_power(a, b, crf, ks, p, extrap)
    elif kind.startswith('logcrf'):
        lv = lv_logcrf(a, b, crf, ks, kind.endswith('extrap'))
    else:
        raise ValueError(kind)

    y   = y_all[keep]; grp = dense.source.values[keep]
    X   = np.column_stack([df[G3].values, lv, lb_at, crf, df.preset.values])
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
    res = {'MAE_mainstream_p6_10': float(np.abs(oof[scored & mains] - y[scored & mains]).mean()),
           'MAE_all73_p1_10': float(np.abs(oof[scored] - y[scored]).mean()),
           'MAE_all_rows': float(np.abs(oof[scored] - y[scored]).mean())}
    mm = scored & mains
    res['R2_main'] = float(1 - ((oof[mm]-y[mm])**2).sum() / ((y[mm]-y[mm].mean())**2).sum())
    res['n_sources'] = int(pd.unique(grp).size); res['n_rows'] = int(scored.sum())
    return res, oof, y, grp, scored, mains


# p=1.0 extrap == linear_extrap and p=0.5 extrap == sqrt_extrap and p=1.0 clamp == linear_clamp
# are already in the csv from the previous run; the grid below adds the rest of the curve.
PS   = [0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
PLAN = []
for tgt, kn, ft in [('v1', (20, 51), 'half'), ('v061', (33, 51), 'full')]:
    for p in PS:
        for ex in (True, False):
            PLAN.append((tgt, kn, ft, f'pow_{p}_{"extrap" if ex else "clamp"}'))
    PLAN.append((tgt, kn, ft, 'logcrf_extrap'))

done = set(pd.read_csv(OUT).config) if os.path.exists(OUT) else set()
todo = [x for x in PLAN if f'{x[0]}|{"+".join(map(str,x[1]))}|{x[2]}|{x[3]}' not in done]
print(f'{len(PLAN)} configs in the family grid, {len(PLAN)-len(todo)} already done, {len(todo)} to run',
      flush=True)

for i, (tgt, kn, ft, kind) in enumerate(todo, 1):
    cfg = f'{tgt}|{"+".join(map(str,kn))}|{ft}|{kind}'
    t0 = time.monotonic()
    try:
        res, oof, y, grp, sc, mn = run(tgt, kn, ft, kind)
    except Exception as e:
        print(f'[{i}/{len(todo)}] FAIL {cfg}: {type(e).__name__}: {e}', flush=True); continue
    el = time.monotonic() - t0
    np.savez(f'{OOFDIR}/{cfg.replace("|","__").replace("+","_")}.npz',
             oof=oof, y=y, source=grp, scored=sc, mains=mn, config=cfg)
    pd.DataFrame([dict(config=cfg, target=tgt, knots='+'.join(map(str, kn)), features=ft,
                       lv_interp=kind, minutes=round(el/60, 2), **res)]).to_csv(
        OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(f'[{i}/{len(todo)}] {cfg:40} main {res["MAE_mainstream_p6_10"]:.4f} '
          f'all73 {res["MAE_all73_p1_10"]:.4f} ({el/60:.1f} min)', flush=True)

print('FAMILY73_COMPLETE', flush=True)

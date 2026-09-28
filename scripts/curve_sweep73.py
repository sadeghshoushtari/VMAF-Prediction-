#!/usr/bin/env python3
"""The interpolant comparison, re-run on the full 73-source dataset.

WHY THIS RUN EXISTS

Two independent implementations already agree on the ordering, on the original 47 sources:
  * this session's sweep  : linear_extrap beat linear_clamp by 0.075 MAE, P(better)=0.974,
                            CI [-0.164, +0.001] - just touching zero. Every shape transform
                            (logit, logdist) lost decisively.
  * an earlier independent run on the same 47 sources: linear_extrap_anchored -0.0232
                            CI [-0.0497, +0.0025] (crosses zero), linear_extrap_probe_only
                            -0.0333 CI [-0.0589, -0.0095] (significant); its sqrt_clamp and
                            train_shape variants both lost.

That earlier run recorded its own outstanding gap as the lack of validation on an independent
corpus. The 26 new sources are exactly that missing corpus. n(mainstream) goes 36 -> 49, which is the only thing that can move a
0.03-0.08 MAE effect out of a +/-0.14 noise band.

WHAT IS HELD FIXED

Everything but the interpolant: same probe budget, same knots, same GOOD-3 content scalars, same
ExtraTrees(300), same LeaveOneGroupOut by source, same isotonic post-step. Only the function that
turns two measured knots into a value at an arbitrary CRF changes.

SCOPES REPORTED (they are not interchangeable - absolute MAEs differ by ~0.2 between them)
  mainstream_p6_10 : the headline scope. 49 natural-camera sources, presets 6-10.
  all73_p1_10      : all sources, all presets - the scope the earlier run used, for comparability.
  all_rows         : everything scored.

HONESTY
  * Paired bootstrap over sources for every delta; the established noise band is ~+/-0.14 MAE.
  * A fold-wise selection number is reported alongside the argmin, because picking the best of N
    variants on the full set is itself worth ~0.02 MAE of optimism (measured on the 47).

-> curve_sweep73_results.csv, curve_sweep73_oof/*.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
warnings.filterwarnings('ignore')

OUT    = _paths.out('curve_sweep73_results.csv')
OOFDIR = _paths.out('curve_sweep73_oof'); os.makedirs(OOFDIR, exist_ok=True)

N_JOBS = 6          # machine is quiet; nothing else is running
NTREES = 300
SEED   = 0
G3     = ['luma_mean', 'ti_mean', 'luma_std']
EPS    = 1e-6

dense  = pd.read_csv(_paths.data('dense_vmaf_73.csv')).sort_values(['source','preset','crf']).reset_index(drop=True)
lowres = pd.read_csv(_paths.data('lowres_vmaf_73.csv'))
feat   = pd.read_csv(_paths.data('feat_73.csv'))
MAIN   = set(open(_paths.data('mainstream_sources_73.txt')).read().split())
FB     = pd.read_csv(_paths.data('fallback_probe.csv'))

_c01   = lambda v: np.clip(np.asarray(v, float) / 100.0, EPS, 1 - EPS)
logit  = lambda v: np.log(_c01(v) / (1 - _c01(v)))
ilogit = lambda z: 100.0 / (1.0 + np.exp(-z))


def probe_table(target):
    """Knot measurements for all 73 sources. The two old 480x270 sources cannot be scored by v1
    at half-res (SpEED's 160px floor), so their knots come from the 2/3-res fallback - the same
    UPRES policy the shipped method uses. The two NEW 480x270 sources were already probed at 2/3
    by the campaign, so they need no substitution."""
    col = {'v061': 'vmaf_lr_v061', 'v1': 'vmaf_lr_v1'}[target]
    t = lowres[['source', 'crf', col, 'bitrate_kbps']].rename(columns={col: 'v'}).dropna(subset=['v'])
    fbc = {'v061': 'vmaf_lr_v061', 'v1': 'vmaf_lr_v1'}[target]
    fb = FB.rename(columns={fbc: 'v'})[['source', 'crf', 'v', 'bitrate_kbps']]
    t = pd.concat([t, fb[~fb.source.isin(set(t.source))]], ignore_index=True)
    return t


def canon_shape(train_src, target, ks):
    """Mean normalised probe shape over TRAINING sources only (u=1 at ks[0], 0 at ks[1])."""
    p = probe_table(target).pivot_table(index='source', columns='crf', values='v').dropna()
    p = p.loc[[s for s in p.index if s in train_src]]
    Vv = p.values.astype(float); kc = np.array(sorted(p.columns), float)
    a = Vv[:, list(p.columns).index(ks[0])]; b = Vv[:, list(p.columns).index(ks[1])]
    U = (Vv - b[:, None]) / (a - b)[:, None]
    return kc, U.mean(0)


def lv_for(kind, a, b, crf, ks, shape=None):
    w_raw = (crf - ks[0]) / (ks[1] - ks[0])
    w_cl  = np.clip(w_raw, 0.0, 1.0)
    if kind == 'linear_clamp':       return a + (b - a) * w_cl          # SHIPPED == np.interp
    if kind == 'linear_extrap':      return a + (b - a) * w_raw
    if kind == 'linear_extrap_clip': return np.clip(a + (b - a) * w_raw, 0.0, 100.0)
    if kind == 'logit_extrap':       return ilogit(logit(a) + (logit(b) - logit(a)) * w_raw)
    if kind == 'sqrt_extrap':        # linear in sqrt(100-V): milder than log-distortion.
        sa, sb = np.sqrt(np.maximum(100 - a, 0)), np.sqrt(np.maximum(100 - b, 0))
        return 100 - np.clip(sa + (sb - sa) * w_raw, 0, None) ** 2
    if kind == 'canonical_pchip':
        kc, ub = shape
        o = np.argsort(kc)
        u = PchipInterpolator(kc[o], ub[o], extrapolate=True)(crf)
        return b + u * (a - b)
    raise ValueError(kind)


def run(target, knots, feats, lv_kind):
    ks = tuple(sorted(knots))
    y_all = (dense.stored_v061 if target == 'v061' else dense.vmaf_v1).values.astype(float)
    f = feat[feat.scale == feats][['source'] + G3]

    t = probe_table(target)
    t = t[t.crf.isin(ks)].copy(); t['llb'] = np.log10(t.bitrate_kbps)
    w = t.pivot_table(index='source', columns='crf', values=['v', 'llb'])
    w.columns = [f'{x}_{z}' for x, z in w.columns]; w = w.dropna()

    df   = dense.merge(f, on='source').merge(w.reset_index(), on='source')
    keep = dense.source.isin(set(w.index)).values
    crf  = df.crf.values.astype(float)
    a, b = df[f'v_{ks[0]}'].values, df[f'v_{ks[1]}'].values
    ba, bbv = df[f'llb_{ks[0]}'].values, df[f'llb_{ks[1]}'].values
    lb_at = ba + (bbv - ba) * np.clip((crf - ks[0]) / (ks[1] - ks[0]), 0, 1)   # as shipped

    y   = y_all[keep]
    grp = dense.source.values[keep]
    folds = list(LeaveOneGroupOut().split(np.zeros((len(y), 1)), y, grp))

    oof = np.zeros(len(y))
    for tr, te in folds:
        shape = canon_shape(set(grp[tr]), target, ks) if lv_kind.startswith('canonical') else None
        lv = lv_for(lv_kind, a, b, crf, ks, shape)
        X  = np.column_stack([df[G3].values, lv, lb_at, crf, df.preset.values])
        m  = ExtraTreesRegressor(n_estimators=NTREES, min_samples_leaf=1, max_features=1.0,
                                 n_jobs=N_JOBS, random_state=SEED).fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0, 100)

    idx = pd.DataFrame({'s': grp, 'p': df.preset.values, 'c': crf, 'i': np.arange(len(oof))})
    for _, g in idx.groupby(['s', 'p']):
        g = g.sort_values('c'); ix = g.i.values
        oof[ix] = IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(g.c.values, oof[ix])

    scored = ~((df.preset == 10) & (df.crf == 55)).values
    mains  = (df.source.isin(MAIN) & (df.preset >= 6)).values
    S = {'mainstream_p6_10': scored & mains, 'all73_p1_10': scored, 'all_rows': scored}
    res = {}
    for k, m_ in S.items():
        e = oof[m_] - y[m_]
        res[f'MAE_{k}'] = float(np.abs(e).mean())
    res['R2_main'] = float(1 - ((oof[S['mainstream_p6_10']] - y[S['mainstream_p6_10']]) ** 2).sum()
                           / ((y[S['mainstream_p6_10']] - y[S['mainstream_p6_10']].mean()) ** 2).sum())
    res['n_sources'] = int(pd.unique(grp).size); res['n_rows'] = int(scored.sum())
    return res, oof, y, grp, scored, mains


KINDS = ['linear_clamp', 'linear_extrap', 'linear_extrap_clip',
         'logit_extrap', 'sqrt_extrap', 'canonical_pchip']
PLAN = [('v1', (20, 51), 'half', k) for k in KINDS] + \
       [('v061', (33, 51), 'full', k) for k in KINDS]

done = set(pd.read_csv(OUT).config) if os.path.exists(OUT) else set()
print(f'{len(PLAN)} configs planned, {len(done)} done, {len(PLAN)-len(done)} to run '
      f'| 73 sources, {len(dense)} cells, mainstream={len(MAIN)}', flush=True)

for i, (tgt, kn, ft, kind) in enumerate(PLAN, 1):
    cfg = f"{tgt}|{'+'.join(map(str,kn))}|{ft}|{kind}"
    if cfg in done:
        print(f'[{i}/{len(PLAN)}] skip {cfg}', flush=True); continue
    t0 = time.monotonic()
    try:
        res, oof, y, grp, sc, mn = run(tgt, kn, ft, kind)
    except Exception as e:
        print(f'[{i}/{len(PLAN)}] FAIL {cfg}: {type(e).__name__}: {e}', flush=True); continue
    el = time.monotonic() - t0
    np.savez(f'{OOFDIR}/{cfg.replace("|","__").replace("+","_")}.npz',
             oof=oof, y=y, source=grp, scored=sc, mains=mn, config=cfg)
    row = dict(config=cfg, target=tgt, knots='+'.join(map(str, kn)), features=ft,
               lv_interp=kind, minutes=round(el/60, 2), **res)
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(f'[{i}/{len(PLAN)}] {cfg:44} main {res["MAE_mainstream_p6_10"]:.4f} '
          f'all73 {res["MAE_all73_p1_10"]:.4f}  ({el/60:.1f} min)', flush=True)

print('SWEEP73_COMPLETE', flush=True)

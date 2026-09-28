#!/usr/bin/env python3
"""Three follow-ups on the 73-source corpus, in one comparable table.

A. KNOT PLACEMENT, RE-SWEPT UNDER THE NEW INTERPOLANT
   The knot-pair sweep was run with the OLD clamped-linear interpolant. On 73 sources only
   {20,51} (v1) and {33,51} (v0.6.1) have ever been tested. There is direct evidence the optimum
   moved: the gain from unclamping scales with how much of the CRF grid falls outside the knot
   span (0.0000 at {20,63} where nothing is outside, +0.0751 at {20,51} with 22.9% outside,
   +0.3680 at {20,42} with 48.6% outside). Under clamping, wide spans were the safe choice
   because everything outside got flattened. Extrapolation removes that penalty, so narrow
   well-placed knots that were previously punished may now win.
   -> all 10 pairs from the 5 measured knot CRFs, under BOTH interpolants, so the interaction
      (not just the new optimum) is visible.

B. MORE THAN TWO KNOTS, UNDER THE NEW INTERPOLANT
   Earlier work put 3 knots at 1.5615 @1.162x wall under the old interpolant. The question now is
   whether 2 knots plus a better interpolant already buys what the 3rd knot bought - a cost win
   rather than an accuracy win - or whether 3 still goes further.
   -> all 10 triples, plus the full 5-knot fit.

C. PRESET TRANSFER
   The probe is ALWAYS preset 10; the targets span presets 1-10. The model must learn that mapping
   from training sources with no explicit help, and no competitor uses the preset axis at all.
   Added feature `pshift` = mean over TRAINING sources of [V(preset,crf) - V(10,crf)], i.e. how
   much better preset p is than preset 10 at this CRF. Computed inside the fold, so it is
   leak-free, and free - it needs no new measurement.

Interpolants: `lin` = clamped linear (shipped), `p05` = p=0.5 distortion power, extrapolated.
With >2 knots both are applied piecewise between consecutive knots.
Primary metric all73_p1_10; every source weighted equally.

-> followup_sweep73_results.csv, followup_sweep73_oof/*.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import os, time, itertools, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

OUT  = _paths.out('followup_sweep73_results.csv')
OOFD = _paths.out('followup_sweep73_oof'); os.makedirs(OOFD, exist_ok=True)
N_JOBS, NTREES, SEED = 6, 300, 0
G3 = ['luma_mean', 'ti_mean', 'luma_std']
HARD = 'DinnerSceneCropped_1920x1080_2997fps_10bit_420'

dense  = pd.read_csv(_paths.data('dense_vmaf_73.csv')).sort_values(['source','preset','crf']).reset_index(drop=True)
lowres = pd.read_csv(_paths.data('lowres_vmaf_73.csv'))
featf  = pd.read_csv(_paths.data('feat_73.csv'))
MAIN   = set(open(_paths.data('mainstream_sources_73.txt')).read().split())
FB     = pd.read_csv(_paths.data('fallback_probe.csv'))


def probe_table(target):
    col = {'v061': 'vmaf_lr_v061', 'v1': 'vmaf_lr_v1'}[target]
    t = lowres[['source','crf',col,'bitrate_kbps']].rename(columns={col:'v'}).dropna(subset=['v'])
    fb = FB.rename(columns={col:'v'})[['source','crf','v','bitrate_kbps']]
    return pd.concat([t, fb[~fb.source.isin(set(t.source))]], ignore_index=True)


def interp_at(crf, ks, vals, kind):
    """Piecewise between consecutive knots; the outermost segment carries the extrapolation."""
    ks = np.asarray(ks, float)
    out = np.empty(len(crf))
    seg = np.clip(np.searchsorted(ks, crf, side='right') - 1, 0, len(ks) - 2)
    for j in range(len(ks) - 1):
        m = seg == j
        if not m.any(): continue
        a, b = vals[j][m], vals[j+1][m]
        w = (crf[m] - ks[j]) / (ks[j+1] - ks[j])
        if kind == 'lin':
            out[m] = a + (b - a) * np.clip(w, 0.0, 1.0)
        else:                                   # p = 0.5 in the distortion domain, extrapolated
            A = np.maximum(100.0 - a, 0.0) ** 0.5
            B = np.maximum(100.0 - b, 0.0) ** 0.5
            out[m] = 100.0 - np.clip(A + (B - A) * w, 0.0, None) ** 2
    return out


def run(target, knots, feats_scale, kind, pshift=False):
    ks = tuple(sorted(knots))
    y_all = (dense.stored_v061 if target == 'v061' else dense.vmaf_v1).values.astype(float)
    f = featf[featf.scale == feats_scale][['source'] + G3]
    t = probe_table(target)
    t = t[t.crf.isin(ks)].copy(); t['llb'] = np.log10(t.bitrate_kbps)
    w = t.pivot_table(index='source', columns='crf', values=['v','llb'])
    w.columns = [f'{x}_{z}' for x, z in w.columns]; w = w.dropna()
    if not all(f'v_{k}' in w.columns for k in ks):
        raise ValueError('knot missing')

    df   = dense.merge(f, on='source').merge(w.reset_index(), on='source')
    keep = dense.source.isin(set(w.index)).values
    crf  = df.crf.values.astype(float)
    vals = [df[f'v_{k}'].values for k in ks]
    bvals= [df[f'llb_{k}'].values for k in ks]
    lv = interp_at(crf, ks, vals, kind)
    lb = interp_at(crf, ks, bvals, 'lin')       # bitrate term unchanged, as shipped

    y   = y_all[keep]; grp = dense.source.values[keep]
    pres = df.preset.values
    base = [df[G3].values, lv[:,None], lb[:,None], crf[:,None], pres[:,None]]
    Xfix = np.hstack(base)

    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(Xfix, y, grp):
        X = Xfix
        if pshift:
            # mean (V(preset,crf) - V(10,crf)) over TRAINING sources only
            tf = pd.DataFrame({'s':grp[tr],'p':pres[tr],'c':crf[tr],'y':y[tr]})
            ref = tf[tf.p==10].groupby(['s','c']).y.mean()
            tf['r'] = list(map(lambda z: ref.get(z, np.nan), zip(tf.s, tf.c)))
            sh = (tf.dropna().assign(d=lambda q: q.y-q.r).groupby(['p','c']).d.mean())
            col = np.array([sh.get((p_, c_), 0.0) for p_, c_ in zip(pres, crf)])
            X = np.hstack(base + [col[:,None]])
        m = ExtraTreesRegressor(n_estimators=NTREES, min_samples_leaf=1, max_features=1.0,
                                n_jobs=N_JOBS, random_state=SEED).fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0, 100)

    idx = pd.DataFrame({'s':grp,'p':pres,'c':crf,'i':np.arange(len(oof))})
    for _, q in idx.groupby(['s','p']):
        q = q.sort_values('c'); ix = q.i.values
        oof[ix] = IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(q.c.values, oof[ix])

    scored = ~((df.preset == 10) & (df.crf == 55)).values
    mains  = (df.source.isin(MAIN) & (df.preset >= 6)).values
    ae = np.abs(oof - y)
    res = dict(MAE_all73_p1_10=float(ae[scored].mean()),
               MAE_mainstream_p6_10=float(ae[scored & mains].mean()),
               MAE_hard=float(ae[scored & (grp==HARD)].mean()) if (grp==HARD).any() else np.nan,
               n_sources=int(pd.unique(grp).size), n_knots=len(ks))
    return res, oof, y, grp, scored, mains


K5 = [20, 33, 42, 51, 63]
PLAN = []
# A. all pairs, both interpolants, v1 (the target where the interpolant result is significant)
for pr in itertools.combinations(K5, 2):
    for kind in ('lin', 'p05'):
        PLAN.append(('v1', pr, 'half', kind, False))
# B. all triples + the full 5-knot fit, new interpolant, v1
for tr_ in itertools.combinations(K5, 3):
    PLAN.append(('v1', tr_, 'half', 'p05', False))
PLAN.append(('v1', tuple(K5), 'half', 'p05', False))
# C. preset transfer, both targets, at each target's shipped knots, both interpolants
for tgt, pr, fs in [('v1', (20,51), 'half'), ('v061', (33,51), 'full')]:
    for kind in ('lin', 'p05'):
        PLAN.append((tgt, pr, fs, kind, True))

done = set(pd.read_csv(OUT).config) if os.path.exists(OUT) else set()
print(f'{len(PLAN)} configs | A: pairs x 2 interpolants | B: triples + 5-knot | C: preset transfer',
      flush=True)

for i, (tgt, ks, fs, kind, ps) in enumerate(PLAN, 1):
    cfg = f"{tgt}|{'+'.join(map(str,ks))}|{kind}{'|pshift' if ps else ''}"
    if cfg in done:
        print(f'[{i}/{len(PLAN)}] skip {cfg}', flush=True); continue
    t0 = time.monotonic()
    try:
        res, oof, y, grp, sc, mn = run(tgt, ks, fs, kind, ps)
    except Exception as e:
        print(f'[{i}/{len(PLAN)}] FAIL {cfg}: {type(e).__name__}: {e}', flush=True); continue
    el = time.monotonic() - t0
    np.savez(f'{OOFD}/{cfg.replace("|","__").replace("+","_")}.npz',
             oof=oof, y=y, source=grp, scored=sc, mains=mn, config=cfg)
    pd.DataFrame([dict(config=cfg, target=tgt, knots='+'.join(map(str,ks)), interp=kind,
                       pshift=ps, minutes=round(el/60,2), **res)]).to_csv(
        OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(f'[{i}/{len(PLAN)}] {cfg:26} all73 {res["MAE_all73_p1_10"]:.4f} '
          f'main {res["MAE_mainstream_p6_10"]:.4f} src {res["n_sources"]} ({el/60:.1f}m)', flush=True)

print('FOLLOWUP_COMPLETE', flush=True)

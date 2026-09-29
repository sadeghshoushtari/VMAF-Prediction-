#!/usr/bin/env python3
"""Is the content-feature extraction needed at all, once the probe measures the curve?

The earlier feature sweep only ever ADDED features (resolution, content class, knot-derived) and
found all of them null. It never removed the GOOD-3 scalars the shipped method already uses, so
"features add nothing" was never tested in the direction that would save anything.

This matters for cost, not just tidiness. The per-video cost decomposes as

    knots 0.760x wall  +  downscale 0.225x  +  GOOD-3 extraction 0.091x

and the method currently sits at a median 1.010x of each video's own wall - marginally OVER.
Removing feature extraction would take roughly 0.09x off every video, which is the difference
between being over the wall and under it. So the question is worth answering precisely: does the
method still work with no content features at all?

Four variants, everything else held to the shipped pipeline (p=0.5 extrapolated interpolant,
ExtraTrees(300), LeaveOneGroupOut by source, isotonic post-step):

    full        GOOD-3 + lv_at + lb_at + crf + preset      (shipped)
    no_g3       lv_at + lb_at + crf + preset               (drop content features entirely)
    no_lb       GOOD-3 + lv_at + crf + preset              (drop the bitrate term instead)
    probe_only  lv_at + crf + preset                       (the probe and nothing else)

-> results/feature_ablation73.csv
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

N_JOBS, NTREES, SEED = 6, 300, 0
G3 = ['luma_mean', 'ti_mean', 'luma_std']
HARD = 'DinnerSceneCropped_1920x1080_2997fps_10bit_420'

dense  = pd.read_csv(_paths.data('dense_vmaf_73.csv')).sort_values(
    ['source', 'preset', 'crf']).reset_index(drop=True)
lowres = pd.read_csv(_paths.data('lowres_vmaf_73.csv'))
featf  = pd.read_csv(_paths.data('feat_73.csv'))
MAIN   = set(open(_paths.data('mainstream_sources_73.txt')).read().split())
FB     = pd.read_csv(_paths.data('fallback_probe.csv'))

VARIANTS = {'full':       (True,  True),      # (use GOOD-3, use lb_at)
            'no_g3':      (False, True),
            'no_lb':      (True,  False),
            'probe_only': (False, False)}


def probe_table(target):
    col = {'v061': 'vmaf_lr_v061', 'v1': 'vmaf_lr_v1'}[target]
    t = lowres[['source', 'crf', col, 'bitrate_kbps']].rename(columns={col: 'v'}).dropna(subset=['v'])
    fb = FB.rename(columns={col: 'v'})[['source', 'crf', 'v', 'bitrate_kbps']]
    return pd.concat([t, fb[~fb.source.isin(set(t.source))]], ignore_index=True)


def run(target, knots, feats_scale, use_g3, use_lb):
    ks = tuple(sorted(knots))
    y_all = (dense.stored_v061 if target == 'v061' else dense.vmaf_v1).values.astype(float)
    f = featf[featf.scale == feats_scale][['source'] + G3]
    t = probe_table(target)
    t = t[t.crf.isin(ks)].copy(); t['llb'] = np.log10(t.bitrate_kbps)
    w = t.pivot_table(index='source', columns='crf', values=['v', 'llb'])
    w.columns = [f'{x}_{z}' for x, z in w.columns]; w = w.dropna()

    df   = dense.merge(f, on='source').merge(w.reset_index(), on='source')
    keep = dense.source.isin(set(w.index)).values
    crf  = df.crf.values.astype(float)
    a, b = df[f'v_{ks[0]}'].values, df[f'v_{ks[1]}'].values
    ba, bb = df[f'llb_{ks[0]}'].values, df[f'llb_{ks[1]}'].values
    wgt = (crf - ks[0]) / (ks[1] - ks[0])
    A = np.maximum(100.0 - a, 0.0) ** 0.5
    B = np.maximum(100.0 - b, 0.0) ** 0.5
    lv = 100.0 - np.clip(A + (B - A) * wgt, 0.0, None) ** 2          # p = 0.5, extrapolated
    lb = ba + (bb - ba) * np.clip(wgt, 0, 1)

    cols = ([df[G3].values] if use_g3 else []) + [lv[:, None]] + \
           ([lb[:, None]] if use_lb else []) + [crf[:, None], df.preset.values[:, None]]
    X = np.hstack(cols)
    y = y_all[keep]; grp = dense.source.values[keep]

    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, grp):
        m = ExtraTreesRegressor(n_estimators=NTREES, min_samples_leaf=1, max_features=1.0,
                                n_jobs=N_JOBS, random_state=SEED).fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0, 100)

    idx = pd.DataFrame({'s': grp, 'p': df.preset.values, 'c': crf, 'i': np.arange(len(oof))})
    for _, q in idx.groupby(['s', 'p']):
        q = q.sort_values('c'); ix = q.i.values
        oof[ix] = IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(q.c.values, oof[ix])

    scored = ~((df.preset == 10) & (df.crf == 55)).values
    mains  = (df.source.isin(MAIN) & (df.preset >= 6)).values
    ae = np.abs(oof - y)
    return dict(MAE_all73_p1_10=float(ae[scored].mean()),
                MAE_mainstream_p6_10=float(ae[scored & mains].mean()),
                MAE_hard=float(ae[scored & (grp == HARD)].mean()),
                n_features=X.shape[1]), oof, y, grp, scored, mains


OUTC = os.path.join(_paths.ROOT, 'results', 'feature_ablation73.csv')
OOFD = os.path.join(_paths.ROOT, 'results', 'oof', 'feature_ablation73_oof')
os.makedirs(OOFD, exist_ok=True)
done = set(pd.read_csv(OUTC).config) if os.path.exists(OUTC) else set()

PLAN = [(t, k, fs, v) for t, k, fs in [('v1', (20, 51), 'half'), ('v061', (33, 51), 'full')]
        for v in VARIANTS]
print(f'{len(PLAN)} configs | does the method still work without content features?', flush=True)

for i, (tgt, kn, fs, v) in enumerate(PLAN, 1):
    cfg = f'{tgt}|{v}'
    if cfg in done:
        print(f'[{i}/{len(PLAN)}] skip {cfg}', flush=True); continue
    g3, lb = VARIANTS[v]
    t0 = time.monotonic()
    res, oof, y, grp, sc, mn = run(tgt, kn, fs, g3, lb)
    np.savez(f'{OOFD}/{tgt}__{v}.npz', oof=oof, y=y, source=grp, scored=sc, mains=mn, config=cfg)
    pd.DataFrame([dict(config=cfg, target=tgt, variant=v,
                       minutes=round((time.monotonic() - t0) / 60, 2), **res)]).to_csv(
        OUTC, mode='a', header=not os.path.exists(OUTC), index=False)
    print(f'[{i}/{len(PLAN)}] {cfg:16} all73 {res["MAE_all73_p1_10"]:.4f} '
          f'main {res["MAE_mainstream_p6_10"]:.4f} hard {res["MAE_hard"]:.2f} '
          f'nfeat {res["n_features"]} ({(time.monotonic()-t0)/60:.1f}m)', flush=True)

print('ABLATION_COMPLETE', flush=True)

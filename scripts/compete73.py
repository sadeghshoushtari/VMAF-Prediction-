#!/usr/bin/env python3
"""Competitor comparison on the full 73-source corpus.

Every method is ported verbatim from the shipped compete.ipynb so that the four competitors are
represented exactly as they were when the published numbers were produced. Only the data changes:
73 sources instead of 47, using the feature tables extended by newsource_competitor_features.py.

METHODS
  vca_rf            RandomForest on 16 VCA features + crf + preset
  vp9_dnn           HistGB (monotone in crf) on SITI scalars + crf + preset
  litevpnet         HistGB on VCA + CLIP ViT-B/16 PCA(16), PCA fitted inside the fold
  content_adaptive  content prior + ONE full-res anchor at crf 30, level-shifted; the anchor
                    row itself is excluded from scoring because it is a real encode
  ours_cold         our feature-only model, no probe at all
  ours_2knot        the shipped 2-knot method (linear interpolation between the knots)
  ours_2knot_sqrt   the same, with the p=0.5 distortion-power interpolant established by
                    curve_sweep73_family.py - the one change this session established as real

SCOPE
  Primary is all73_p1_10: all 73 sources, all presets, nothing excluded. The mainstream subset is
  reported beside it for continuity with the older numbers, but the whole corpus is weighted
  equally, including the sources that predict worst.

THIS RUN MEASURES ACCURACY ONLY. Cost is a separate quiet-machine measurement: wall73.csv covers
our stages, but the competitors' feature-extraction times on the new 26 sources have not been
measured, so no x-wall number for them would be defensible yet.

-> compete73_metrics.csv, compete73_oof/*.npz
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

V    = os.path.expanduser('~/vmaf')
DOLD = '/mnt/c/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data'
OUT  = f'{V}/compete73_metrics.csv'
OOF  = f'{V}/compete73_oof'; os.makedirs(OOF, exist_ok=True)
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

# ---------------------------------------------------------------- data
d = pd.read_csv(f'{V}/dense_vmaf_73.csv').sort_values(['source','preset','crf']).reset_index(drop=True)
MAIN = set(open(f'{V}/mainstream_sources_73.txt').read().split())
g = d.source.values

# Unified 73-source competitor feature tables (old 47 + new 26, merged by
# make_competitor_tables_73; identical content to the two-table load this replaced).
sf  = pd.read_csv(f'{V}/source_features_73.csv')
vca = pd.read_csv(f'{V}/vca_full_features_73.csv')
_z  = np.load(f'{V}/clip_embeddings_73.npz', allow_pickle=True)
ES  = list(_z['source']); CLIPM = _z['vitb16']

missing = sorted(set(d.source) - set(sf.source)) + sorted(set(d.source) - set(vca.source)) + \
          sorted(set(d.source) - set(ES))
if missing:
    raise SystemExit(f'feature coverage incomplete, e.g. {missing[:4]} - run '
                     f'newsource_competitor_features.py first')
srow = np.array([ES.index(s) for s in d.source])
log(f'grid {len(d)} rows | {d.source.nunique()} sources | mainstream {len(MAIN)}')

TARGETS = {'v061': d.stored_v061.values.astype(float), 'v1': d.vmaf_v1.values.astype(float)}
GOOD3 = ['luma_mean','ti_mean','luma_std']
SITI  = ['si_mean','si_max','ti_mean','ti_max','luma_mean','luma_std','lap_var_mean','chroma_std']
VCAF  = ['vE8','vh8','vE16','vh16','vE32','vh32','vE16_std','vE16_p90','vh16_std','vh16_p90',
         'vL','venU','venV','ventropy','vedge','veps']
HGBKW = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=15,
             l2_regularization=1.0, random_state=0)
dg3 = d.merge(sf[['source']+GOOD3], on='source')
dsi = d.merge(sf[['source']+SITI], on='source')
dvc = d.merge(vca[['source']+VCAF], on='source')
lowres = pd.read_csv(f'{V}/lowres_vmaf_73.csv')
FB = pd.read_csv(f'{DOLD}/fallback_probe.csv')

def probe_table(tname):
    col = {'v061':'vmaf_lr_v061','v1':'vmaf_lr_v1'}[tname]
    t = lowres[['source','crf',col,'bitrate_kbps']].rename(columns={col:'v'}).dropna(subset=['v'])
    fb = FB.rename(columns={col:'v'})[['source','crf','v','bitrate_kbps']]
    return pd.concat([t, fb[~fb.source.isin(set(t.source))]], ignore_index=True)

# ---------------------------------------------------------------- methods
def m_vca_rf(y, tname):
    X = dvc[VCAF+['crf','preset']].values; oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        oof[te] = np.clip(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
                          .fit(X[tr], y[tr]).predict(X[te]), 0, 100)
    return oof, np.ones(len(y), bool)

def m_vp9(y, tname):
    X = dsi[SITI+['crf','preset']].values; mono = [0]*len(SITI)+[-1,0]; oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        oof[te] = np.clip(HistGradientBoostingRegressor(monotonic_cst=mono, **HGBKW)
                          .fit(X[tr], y[tr]).predict(X[te]), 0, 100)
    return oof, np.ones(len(y), bool)

def m_litevpnet(y, tname, npc=16):
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(dvc, y, g):
        ts = sorted(set(srow[tr]))
        ss = StandardScaler().fit(CLIPM[ts])
        P = PCA(npc, random_state=0).fit(ss.transform(CLIPM[ts])).transform(ss.transform(CLIPM))[srow]
        Xtr = np.column_stack([dvc[VCAF].values[tr], P[tr], dvc[['crf','preset']].values[tr]])
        Xte = np.column_stack([dvc[VCAF].values[te], P[te], dvc[['crf','preset']].values[te]])
        mono = [0]*(len(VCAF)+npc)+[-1,0]
        oof[te] = np.clip(HistGradientBoostingRegressor(monotonic_cst=mono, **HGBKW)
                          .fit(Xtr, y[tr]).predict(Xte), 0, 100)
    return oof, np.ones(len(y), bool)

def _cold_good3(y):
    X = dg3[GOOD3+['crf','preset']].values; oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        oof[te] = HistGradientBoostingRegressor(monotonic_cst=[0,0,0,-1,0], **HGBKW
                                                ).fit(X[tr], y[tr]).predict(X[te])
    return np.clip(oof, 0, 100)

def m_ours_cold(y, tname):
    return _cold_good3(y), np.ones(len(y), bool)

ANCHOR_CRF = 30
def m_content_adaptive(y, tname):
    cold = _cold_good3(y); oof = cold.copy(); test = np.ones(len(y), bool)
    tmp = pd.DataFrame({'s':g,'p':d.preset.values,'c':d.crf.values,'i':np.arange(len(y))})
    for _, grp in tmp.groupby(['s','p']):
        idx = grp.i.values; a = idx[np.argmin(np.abs(grp.c.values - ANCHOR_CRF))]
        oof[idx] = np.clip(cold[idx] + (y[a] - cold[a]), 0, 100)
        test[a] = False
    return oof, test

def _ours_knot(y, tname, knots, interp):
    t = probe_table(tname)
    lk = t[t.crf.isin(knots)].copy(); lk['llb'] = np.log10(lk.bitrate_kbps)
    w = lk.pivot_table(index='source', columns='crf', values=['v','llb'])
    w.columns = [f'{a}_{b}' for a,b in w.columns]; w = w.dropna()
    df = d.merge(sf[['source']+GOOD3], on='source').merge(w.reset_index(), on='source')
    keep = d.source.isin(set(w.index)).values
    cr = df.crf.values.astype(float)
    a_, b_ = df[f'v_{knots[0]}'].values, df[f'v_{knots[1]}'].values
    ba, bb = df[f'llb_{knots[0]}'].values, df[f'llb_{knots[1]}'].values
    fr_c = np.clip((cr-knots[0])/(knots[1]-knots[0]), 0, 1)
    if interp == 'linear':
        lv = a_ + (b_-a_)*fr_c
    else:                                            # p=0.5 distortion power, extrapolated
        fr = (cr-knots[0])/(knots[1]-knots[0]); p = 0.5
        A = np.maximum(100.0-a_,0.0)**p; B = np.maximum(100.0-b_,0.0)**p
        lv = 100.0 - np.clip(A+(B-A)*fr, 0.0, None)**(1.0/p)
    df['lv_at'] = lv; df['lb_at'] = ba + (bb-ba)*fr_c
    FEAT = GOOD3+['lv_at','lb_at','crf','preset']
    ysub, gsub, X = y[keep], g[keep], df[FEAT].values
    sub = np.zeros(len(ysub))
    for tr, te in LeaveOneGroupOut().split(X, ysub, gsub):
        sub[te] = np.clip(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=1,
                          max_features=1.0, n_jobs=-1, random_state=0).fit(X[tr], ysub[tr]).predict(X[te]), 0, 100)
    idx = pd.DataFrame({'s':gsub,'p':df.preset.values,'c':cr,'i':np.arange(len(sub))})
    for _, q in idx.groupby(['s','p']):
        q = q.sort_values('c'); ix = q.i.values
        sub[ix] = IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(q.c.values, sub[ix])
    oof = np.full(len(y), np.nan); oof[keep] = sub
    return oof, keep

def m_ours_2knot(y, tname):      return _ours_knot(y, tname, (33,51), 'linear')
def m_ours_2knot_sqrt(y, tname): return _ours_knot(y, tname, (33,51), 'sqrt')

METHODS = [('vca_rf', m_vca_rf), ('vp9_dnn', m_vp9), ('litevpnet', m_litevpnet),
           ('content_adaptive', m_content_adaptive), ('ours_cold', m_ours_cold),
           ('ours_2knot', m_ours_2knot), ('ours_2knot_sqrt', m_ours_2knot_sqrt)]

rows = []
for tname, y in TARGETS.items():
    for mname, fn in METHODS:
        st = time.time()
        oof, test = fn(y, tname)
        ok = test & ~np.isnan(oof)
        mains = (d.source.isin(MAIN) & (d.preset >= 6)).to_numpy()
        e = oof[ok]-y[ok]; ae = np.abs(e)
        em = oof[ok&mains]-y[ok&mains]
        r = dict(target=tname, method=mname,
                 MAE_all73=float(ae.mean()),
                 MAE_mainstream=float(np.abs(em).mean()),
                 R2_all73=float(1-(e**2).sum()/((y[ok]-y[ok].mean())**2).sum()),
                 within1=float((ae<=1).mean()*100), within2=float((ae<=2).mean()*100),
                 max_err=float(ae.max()), n_rows=int(ok.sum()),
                 n_sources=int(pd.unique(g[ok]).size), minutes=round((time.time()-st)/60,2))
        rows.append(r)
        np.savez(f'{OOF}/{tname}__{mname}.npz', oof=oof, y=y, source=g, ok=ok, mains=mains)
        pd.DataFrame([r]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        log(f'{tname:5} {mname:18} MAE_all73 {r["MAE_all73"]:.4f}  main {r["MAE_mainstream"]:.4f} '
            f'| {r["n_sources"]} src | {r["minutes"]:.1f} min')

print('COMPETE73_COMPLETE', flush=True)

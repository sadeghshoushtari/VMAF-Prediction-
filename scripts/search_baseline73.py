#!/usr/bin/env python3
"""Search baseline: why predict the curve at all, instead of measuring it cheaply?

Every method compared so far PREDICTS quality. Production tools mostly do not - they SEARCH,
bisecting over CRF with real encodes until the target quality is hit (dynamic-crf, viser,
per-title-analysis all work this way). That is the strongest challenge to the premise of this
work, and it was missing from the comparison.

THE COMMON TASK

Prediction and search do not answer the same question, so they are compared on the one they share:

    given a source, a preset and a target VMAF, choose the CRF that hits the target.

This is also the task the competing papers actually solve - LiteVPNet and the VP9 per-title work
both predict CRF for a target VMAF - so it is a fairer framing than asking a search method to
predict a whole surface.

  search      bisects over the CRF grid, paying one FULL-RESOLUTION encode at the target preset
              per probe, until the bracket closes. Exact, not approximated: every CRF it can
              query is already measured in the dense grid.
  ours        reads the CRF off the predicted curve, built from two HALF-RESOLUTION preset-10
              probe encodes, and pays nothing further per preset or per target.
  fixed       one CRF for everything, the per-corpus best in hindsight. The floor: what you get
              with no adaptation at all.

Error is |achieved VMAF - target|, where the achieved VMAF is the true measured value at the CRF
each method chose. Cost is counted in encodes, separated by resolution and preset because a
full-resolution preset-1 encode and a half-resolution preset-10 encode are not the same thing.

No raw video is needed: the dense grid contains every point any method could query.

-> results/search_baseline73.csv
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import numpy as np, pandas as pd

TARGET_COL = {'v1': 'vmaf_v1', 'v061': 'stored_v061'}
TARGETS = [80, 85, 90, 95]


def load(target):
    col = TARGET_COL[target]
    d = pd.read_csv(_paths.data('dense_vmaf_73.csv'),
                    usecols=['source', 'preset', 'crf', col]).rename(columns={col: 'v'})
    return d


def curve_map(d):
    """(source,preset) -> (crfs ascending, vmaf at those crfs)."""
    out = {}
    for (s, p), g in d.groupby(['source', 'preset']):
        g = g.sort_values('crf')
        out[(s, p)] = (g.crf.values.astype(float), g.v.values.astype(float))
    return out


def bisect_search(crfs, vals, target, max_probes=None):
    """Binary search for the CRF whose VMAF is closest to `target`.

    VMAF falls as CRF rises, so the array is descending. Each evaluation is one real encode at
    full resolution and the working preset. Returns (chosen_index, n_probes).
    """
    lo, hi = 0, len(crfs) - 1
    probed = {}

    def val(i):
        if i not in probed:
            probed[i] = vals[i]
        return probed[i]

    # the two ends are probed first, as any real implementation must to know the range
    val(lo); val(hi)
    while hi - lo > 1:
        if max_probes is not None and len(probed) >= max_probes:
            break
        mid = (lo + hi) // 2
        if val(mid) >= target:
            lo = mid
        else:
            hi = mid
    cand = sorted(probed)
    best = min(cand, key=lambda i: abs(vals[i] - target))
    return best, len(probed)


def predicted_pick(crfs, pred, target):
    """The CRF whose PREDICTED VMAF is closest to the target."""
    return int(np.argmin(np.abs(pred - target)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target-metric', choices=list(TARGET_COL), default='v1')
    ap.add_argument('--oof', default=None,
                    help='npz of out-of-fold predictions for "ours" '
                         '(default: results/oof/curve_sweep73_oof/<target>__..._pow_0.5_extrap.npz)')
    a = ap.parse_args()

    d = load(a.target_metric)
    cm = curve_map(d)
    print(f'{len(cm)} (source,preset) curves | {d.source.nunique()} sources')

    # our predicted surface, out of fold
    if a.oof is None:
        knots = '20_51__half' if a.target_metric == 'v1' else '33_51__full'
        a.oof = os.path.join(_paths.ROOT, 'results', 'oof', 'curve_sweep73_oof',
                             f'{a.target_metric}__{knots}__pow_0.5_extrap.npz')
    z = np.load(a.oof, allow_pickle=True)
    print(f'ours: {os.path.basename(a.oof)}')
    key = pd.DataFrame({'source': z['source'], 'oof': z['oof']})
    dd = pd.read_csv(_paths.data('dense_vmaf_73.csv'), usecols=['source', 'preset', 'crf'])
    dd = dd.sort_values(['source', 'preset', 'crf']).reset_index(drop=True)
    assert len(dd) == len(key) and (dd.source.values == key.source.values).all(), \
        'OOF row order does not match the dense grid'
    dd['pred'] = key.oof.values
    pm = {(s, p): g.sort_values('crf').pred.values.astype(float)
          for (s, p), g in dd.groupby(['source', 'preset'])}

    rows = []
    for (s, p), (crfs, vals) in cm.items():
        pred = pm[(s, p)]
        for t in TARGETS:
            if vals.max() < t:          # the target is unreachable for this source/preset
                continue
            i_s, nprobe = bisect_search(crfs, vals, t)
            i_o = predicted_pick(crfs, pred, t)
            i_best = int(np.argmin(np.abs(vals - t)))
            rows.append(dict(source=s, preset=int(p), target=t,
                             err_search=abs(vals[i_s] - t), probes_search=nprobe,
                             err_ours=abs(vals[i_o] - t),
                             err_oracle=abs(vals[i_best] - t),
                             crf_search=crfs[i_s], crf_ours=crfs[i_o], crf_oracle=crfs[i_best]))
    r = pd.DataFrame(rows)

    # the no-adaptation floor: one CRF for everything, best in hindsight per target
    fixed = {}
    for t in TARGETS:
        best, berr = None, 1e9
        for j, c in enumerate(cm[list(cm)[0]][0]):
            e = np.mean([abs(v[int(np.where(k == c)[0][0])] - t)
                         for k, v in cm.values() if v.max() >= t])
            if e < berr: best, berr = c, e
        fixed[t] = (best, berr)
    r['err_fixed'] = r.target.map({t: fixed[t][1] for t in TARGETS})

    out = _paths.out('search_baseline73.csv') if not os.path.isdir(
        os.path.join(_paths.ROOT, 'results')) else os.path.join(_paths.ROOT, 'results',
                                                                'search_baseline73.csv')
    r.to_csv(out, index=False)

    print(f'\n{len(r)} (source,preset,target) cases where the target is reachable\n')
    print(f'{"":12} {"mean |err|":>11} {"median":>8} {"within 1":>9} {"cost":>34}')
    print(f'{"fixed CRF":12} {r.err_fixed.mean():11.3f} {r.err_fixed.median():8.3f} '
          f'{(r.err_fixed<=1).mean()*100:8.1f}% {"0 encodes (no adaptation)":>34}')
    print(f'{"ours":12} {r.err_ours.mean():11.3f} {r.err_ours.median():8.3f} '
          f'{(r.err_ours<=1).mean()*100:8.1f}% {"2 half-res preset-10 encodes":>34}')
    print(f'{"search":12} {r.err_search.mean():11.3f} {r.err_search.median():8.3f} '
          f'{(r.err_search<=1).mean()*100:8.1f}% '
          f'{f"{r.probes_search.mean():.1f} FULL-res encodes/case":>34}')
    print(f'{"oracle":12} {r.err_oracle.mean():11.3f} {r.err_oracle.median():8.3f} '
          f'{(r.err_oracle<=1).mean()*100:8.1f}% {"the grid itself":>34}')

    print('\nper target:')
    for t in TARGETS:
        q = r[r.target == t]
        print(f'  VMAF {t}: ours {q.err_ours.mean():.3f} | search {q.err_search.mean():.3f} '
              f'({q.probes_search.mean():.1f} encodes) | fixed {q.err_fixed.mean():.3f} '
              f'| oracle {q.err_oracle.mean():.3f}  [n={len(q)}]')

    print('\ncost note: a search probe is a FULL-resolution encode at the working preset. Our two '
          'probes are\n  HALF-resolution and always preset 10, so they are far cheaper per encode '
          'AND are paid once\n  per source rather than once per (preset, target).')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

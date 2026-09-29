#!/usr/bin/env python3
"""VMAF against CRF for all 73 sources.

One panel per source, one line per preset. Panels are ordered by AOM-CTC class and then by
descending pixel count, so the resolution tiers and the synthetic/screen-content classes group
together and the structure across the corpus is visible rather than scattered alphabetically.

Preset is an ordered quantity, so it is encoded as a single-hue ramp from light (preset 1, the
slowest and highest quality) to dark (preset 10, the fastest), not as categorical colours. All
panels share both axes, so panel-to-panel comparison is direct.

    python scripts/plot_vmaf_curves.py                # v1, the default target
    python scripts/plot_vmaf_curves.py --target v061
    python scripts/plot_vmaf_curves.py --check        # verify against the CSV and exit

Writes figures/vmaf_vs_crf_73_<target>.png and .pdf, and prints a verification summary: every source drawn,
every curve complete, and a random sample of plotted points re-read from the source table.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TARGETS = {'v1': ('vmaf_v1', 'VMAF v1.0.16'),
           'v061': ('stored_v061', 'VMAF v0.6.1'),
           'v1_hfr': ('vmaf_v1_hfr', 'VMAF v1.0.16 HFR')}
CLASS_ORDER = ['a1_4k', 'a2_2k', 'a3_720p', 'a4_360p', 'a5_270p', 'b1_syn', 'b2_scc']
CLASS_LABEL = {'a1_4k': '4K', 'a2_2k': '2K', 'a3_720p': '720p', 'a4_360p': '360p',
               'a5_270p': '270p', 'b1_syn': 'synthetic', 'b2_scc': 'screen'}

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d8d4'
RAMP = LinearSegmentedColormap.from_list('preset', ['#cfe0f6', '#2a78d6', '#10305c'])


def load(target):
    col, _ = TARGETS[target]
    d = pd.read_csv(_paths.data('dense_vmaf_73.csv'),
                    usecols=['source', 'width', 'height', 'preset', 'crf', col])
    cls = {}
    for line in open(_paths.data('aomctc_class_map.tsv')):
        if '\t' in line and not line.startswith('class\t'):
            c, f = line.rstrip('\n').split('\t')
            cls[os.path.splitext(f)[0]] = c
    d['cls'] = d.source.map(cls).fillna('?')
    d['px'] = d.width * d.height
    return d, col


def order_sources(d):
    meta = d.groupby('source').agg(cls=('cls', 'first'), px=('px', 'first')).reset_index()
    meta['ci'] = meta.cls.map({c: i for i, c in enumerate(CLASS_ORDER)}).fillna(99)
    return meta.sort_values(['ci', 'px', 'source'], ascending=[True, False, True]).source.tolist()


def verify(d, col, order):
    """Checks that must hold before the figure is worth looking at."""
    ok = True
    n_src = d.source.nunique()
    print(f'  sources in table          : {n_src}')
    print(f'  sources to be drawn       : {len(order)}')
    if n_src != len(order) or n_src != 73:
        print('  FAIL: source count'); ok = False
    sizes = d.groupby(['source', 'preset']).size()
    print(f'  points per (source,preset): {sorted(sizes.unique())}  (expect [35])')
    if list(sizes.unique()) != [35]:
        print('  FAIL: incomplete curves'); ok = False
    npres = d.groupby('source').preset.nunique()
    print(f'  presets per source        : {sorted(npres.unique())}  (expect [10])')
    if list(npres.unique()) != [10]:
        print('  FAIL: missing presets'); ok = False
    nan = int(d[col].isna().sum())
    print(f'  missing {col:12}      : {nan}')
    if nan: print('  FAIL: nulls in the plotted column'); ok = False
    lo, hi = d[col].min(), d[col].max()
    print(f'  value range               : {lo:.3f} .. {hi:.3f}')
    if lo < 0 or hi > 100:
        print('  FAIL: outside [0,100]'); ok = False
    unclassed = sorted(set(d.source[d.cls == '?']))
    print(f'  sources without a class   : {len(unclassed)}')
    if unclassed: print('  FAIL:', unclassed[:3]); ok = False
    return ok


def spot_check(d, col, n=8, seed=0):
    """Re-read individual points straight from the CSV and compare with the merged frame."""
    raw = pd.read_csv(_paths.data('dense_vmaf_73.csv'), usecols=['source', 'preset', 'crf', col])
    rng = np.random.default_rng(seed)
    rows = d.sample(n, random_state=seed)
    worst = 0.0
    for _, r in rows.iterrows():
        m = raw[(raw.source == r.source) & (raw.preset == r.preset) & (raw.crf == r.crf)]
        worst = max(worst, abs(float(m[col].iloc[0]) - float(r[col])))
    print(f'  spot check {n} random points: max |delta| vs the CSV = {worst:.2e}')
    return worst == 0.0


def draw(d, col, label, order, out_png, out_pdf):
    ncol = 8
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.05 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, s in zip(axes, order):
        g = d[d.source == s]
        for p in range(1, 11):
            gp = g[g.preset == p].sort_values('crf')
            ax.plot(gp.crf, gp[col], lw=1.0, color=RAMP((p - 1) / 9), solid_capstyle='round')
        cl = g.cls.iloc[0]
        w, h = int(g.width.iloc[0]), int(g.height.iloc[0])
        ax.set_title(f'{s[:24]}\n{CLASS_LABEL.get(cl, cl)} · {w}x{h}',
                     fontsize=6.2, color=INK, linespacing=1.25, pad=3)
        ax.set_ylim(0, 100); ax.set_xlim(18, 65)
        ax.set_xticks([20, 35, 50, 63]); ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(labelsize=6, colors=MUTED, length=2)

    for ax in axes[len(order):]:
        ax.axis('off')

    # Axis labels go on the LAST OCCUPIED panel of each column and the first of each row,
    # not on the literal edges of the grid - the bottom row is nearly empty, so relying on
    # sharex would leave almost every panel without readable ticks.
    last_in_col = {}
    for i in range(len(order)):
        last_in_col[i % ncol] = i
    for c, i in last_in_col.items():
        axes[i].set_xlabel('CRF', fontsize=7, color=MUTED)
        axes[i].tick_params(labelbottom=True, labelsize=6, colors=MUTED)
    for i in range(0, len(order), ncol):
        axes[i].set_ylabel(label, fontsize=7, color=MUTED)

    fig.subplots_adjust(left=0.035, right=0.915, top=0.955, bottom=0.030,
                        wspace=0.18, hspace=0.62)
    cax = fig.add_axes([0.935, 0.35, 0.010, 0.30])
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(1, 10))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('preset  (1 = slowest, 10 = fastest)', color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_visible(False)

    fig.suptitle(f'{label} against CRF — all {len(order)} AOM-CTC sources, 10 presets each',
                 fontsize=13, color=INK, y=0.985)
    fig.savefig(out_png, dpi=150, facecolor='white')
    fig.savefig(out_pdf, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', choices=list(TARGETS), default='v1')
    ap.add_argument('--check', action='store_true', help='verify only, draw nothing')
    a = ap.parse_args()

    d, col = load(a.target)
    _, label = TARGETS[a.target]
    order = order_sources(d)

    print(f'VERIFICATION ({a.target}):')
    ok = verify(d, col, order)
    ok &= spot_check(d, col)
    per_class = d.groupby('cls').source.nunique().reindex(CLASS_ORDER).fillna(0).astype(int)
    print('  per class                 : ' + ', '.join(f'{CLASS_LABEL[c]} {per_class[c]}'
                                                       for c in CLASS_ORDER))
    print(f'  curves to draw            : {len(order) * 10} '
          f'({len(order)} sources x 10 presets, {len(d)} points)')
    if not ok:
        sys.exit('verification FAILED - figure not written')
    print('  all checks passed')
    if a.check:
        return

    png = _paths.figure(f'vmaf_vs_crf_73_{a.target}.png')
    pdf = _paths.figure(f'vmaf_vs_crf_73_{a.target}.pdf')
    draw(d, col, label, order, png, pdf)
    print(f'\nwrote {png}\n      {pdf}')


if __name__ == '__main__':
    main()

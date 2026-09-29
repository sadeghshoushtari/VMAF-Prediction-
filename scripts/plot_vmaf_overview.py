#!/usr/bin/env python3
"""All 73 sources on one axes: VMAF against CRF at a single preset.

WHICH PRESET, AND WHY

Preset 7, chosen by measurement rather than convention. For each preset, the mean absolute
deviation of its curve from the average across all ten presets is:

    preset   1     2     3     4     5     6     7     8     9    10
    dev    1.53  1.28  0.95  0.78  0.66  0.35  0.34  0.43  1.83  2.64

Preset 7 sits closest to the whole family (0.34), with preset 6 effectively tied. Presets 9 and 10
are the least representative - they are the outliers of the set, not the middle of it - so a single
figure drawn at preset 10 would misstate the corpus by roughly 2.6 VMAF.

Nothing important is lost by picking one preset: the ordering of sources is essentially
preset-invariant (Spearman >= 0.946 against preset 1 for every other preset), and the between-source
spread barely moves (sd 6.9 to 7.4 across all ten). Preset 7 also lies inside the presets 6-10 range
the headline metric is scored on.

DESIGN

73 thin grey lines give the corpus envelope; seven bold curves give the median of each AOM-CTC
class and are labelled directly, so class identity never depends on colour alone. The three
sources that define the extremes are named on the plot, since they are the ones that drive the
behaviour discussed in the results.

    python scripts/plot_vmaf_overview.py                  # preset 7, v1
    python scripts/plot_vmaf_overview.py --preset 10      # e.g. the probe preset
    python scripts/plot_vmaf_overview.py --check          # verify only

-> figures/vmaf_overview_73_<target>_p<preset>.png and .pdf
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS = {'v1': ('vmaf_v1', 'VMAF v1.0.16'),
           'v061': ('stored_v061', 'VMAF v0.6.1'),
           'v1_hfr': ('vmaf_v1_hfr', 'VMAF v1.0.16 HFR')}
CLASS_ORDER = ['a1_4k', 'a2_2k', 'a3_720p', 'a4_360p', 'a5_270p', 'b1_syn', 'b2_scc']
CLASS_LABEL = {'a1_4k': '4K', 'a2_2k': '2K', 'a3_720p': '720p', 'a4_360p': '360p',
               'a5_270p': '270p', 'b1_syn': 'synthetic', 'b2_scc': 'screen'}
# validated categorical palette: all hard gates pass at 7 slots
# (worst adjacent CVD dE 9.1, normal-vision dE 19.6); the contrast warning is met by
# direct-labelling every class curve rather than relying on the colour.
CLASS_COLOR = {'a1_4k': '#2a78d6', 'a2_2k': '#eb6834', 'a3_720p': '#1baf7a',
               'a4_360p': '#eda100', 'a5_270p': '#e87ba4', 'b1_syn': '#008300',
               'b2_scc': '#4a3aa7'}
INK, MUTED, GRID, FAINT = '#0b0b0b', '#52514e', '#d8d8d4', '#c9c9c4'
NAMED = {'DinnerSceneCropped_1920x1080_2997fps_10bit_420': 'DinnerScene — lowest ceiling',
         'RedKayak_360_2997': 'RedKayak — steepest fall',
         'MobileDeviceScreenSharing': 'MobileDeviceScreenSharing — flattest'}


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
    return d, col


def representativeness(d, col):
    """Mean |deviation| of each preset from the across-preset average. Lower is more typical."""
    avg = d.groupby(['source', 'crf'])[col].mean()
    out = {}
    for p, g in d.groupby('preset'):
        j = g.set_index(['source', 'crf'])[col]
        out[int(p)] = float((j - avg.reindex(j.index)).abs().mean())
    return out


def verify(d, col, preset):
    ok = True
    g = d[d.preset == preset]
    print(f'  sources at preset {preset:<2}       : {g.source.nunique()}  (expect 73)')
    if g.source.nunique() != 73: print('  FAIL: source count'); ok = False
    n = g.groupby('source').size()
    print(f'  points per source         : {sorted(n.unique())}  (expect [35])')
    if list(n.unique()) != [35]: print('  FAIL: incomplete curves'); ok = False
    if int(g[col].isna().sum()): print('  FAIL: nulls'); ok = False
    print(f'  value range               : {g[col].min():.3f} .. {g[col].max():.3f}')
    if g[col].min() < 0 or g[col].max() > 100: print('  FAIL: out of range'); ok = False
    miss = [s for s in NAMED if s not in set(g.source)]
    if miss: print('  FAIL: annotated source missing:', miss); ok = False
    per = g.groupby('cls').source.nunique().reindex(CLASS_ORDER).fillna(0).astype(int)
    print('  per class                 : ' + ', '.join(f'{CLASS_LABEL[c]} {per[c]}' for c in CLASS_ORDER))
    if per.sum() != 73: print('  FAIL: class coverage'); ok = False
    return ok


def draw(d, col, label, preset, dev, out_png, out_pdf):
    g = d[d.preset == preset]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))

    for s, gs in g.groupby('source'):
        gs = gs.sort_values('crf')
        ax.plot(gs.crf, gs[col], lw=0.7, color=FAINT, zorder=1, solid_capstyle='round')

    ends = []
    for c in CLASS_ORDER:
        gc = g[g.cls == c]
        if not len(gc): continue
        med = gc.groupby('crf')[col].median().sort_index()
        ax.plot(med.index, med.values, lw=2.4, color=CLASS_COLOR[c], zorder=3,
                solid_capstyle='round')
        ends.append((med.values[-1], c))

    # direct labels, nudged apart so none overlaps (the palette's contrast warning is met
    # by these labels, not by the colour alone)
    ends.sort(reverse=True)
    prev = None
    for v, c in ends:
        y = v if prev is None else min(v, prev - 3.4)
        ax.annotate(f'{CLASS_LABEL[c]}  ({(g.cls == c).groupby(g.source).any().sum()})',
                    xy=(63, v), xytext=(64.6, y), fontsize=9.5, color=CLASS_COLOR[c],
                    va='center', ha='left', weight='bold',
                    arrowprops=dict(arrowstyle='-', color=CLASS_COLOR[c], lw=0.8,
                                    shrinkA=0, shrinkB=2) if abs(y - v) > 1.2 else None)
        prev = y

    # Anchors are chosen from the actual curve positions so the three labels cannot collide:
    # DinnerScene is flat and low, RedKayak is steep, MobileDeviceScreenSharing hugs the ceiling.
    PLACE = {'MobileDeviceScreenSharing': ((34, 100), (22.5, 90.5), 'left'),
             'DinnerSceneCropped_1920x1080_2997fps_10bit_420': ((44, 74), (30.5, 58), 'left'),
             'RedKayak_360_2997': ((57, 42), (44.5, 24), 'left')}
    for s, note in NAMED.items():
        gs = g[g.source == s].sort_values('crf')
        ax.plot(gs.crf, gs[col], lw=1.6, color=INK, ls=(0, (4, 2)), zorder=4)
        (ax_, ay), (tx, ty), ha = PLACE[s]
        ax.annotate(note, xy=(ax_, ay), xytext=(tx, ty), fontsize=9, color=INK, ha=ha,
                    va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.30', fc='white', ec='none', alpha=0.88),
                    arrowprops=dict(arrowstyle='-', color=INK, lw=0.8, shrinkA=2, shrinkB=2,
                                    connectionstyle='arc3,rad=0.12'))

    ax.set_xlim(19, 72); ax.set_ylim(0, 102)
    ax.set_xticks([20, 25, 30, 35, 40, 45, 50, 55, 60, 63])
    ax.set_xlabel('CRF', fontsize=11, color=MUTED)
    ax.set_ylabel(label, fontsize=11, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5)

    ax.set_title(f'{label} against CRF — all 73 AOM-CTC sources at preset {preset}',
                 fontsize=13.5, color=INK, pad=16, loc='left')
    ax.text(0, 1.015, f'thin grey: every source · bold: median per class · '
                      f'preset {preset} is the most representative of the ten '
                      f'(mean deviation {dev[preset]:.2f} VMAF from the across-preset average)',
            transform=ax.transAxes, fontsize=9, color=MUTED, va='bottom')

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor='white')
    fig.savefig(out_pdf, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', choices=list(TARGETS), default='v1')
    ap.add_argument('--preset', type=int, default=7)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    d, col = load(a.target)
    _, label = TARGETS[a.target]
    dev = representativeness(d, col)

    print(f'VERIFICATION ({a.target}, preset {a.preset}):')
    ok = verify(d, col, a.preset)
    best = min(dev, key=dev.get)
    print('  preset representativeness : ' + ', '.join(f'p{p} {dev[p]:.2f}' for p in sorted(dev)))
    print(f'  most representative       : preset {best} ({dev[best]:.2f})'
          + ('' if best == a.preset else f'   <- NOTE: drawing preset {a.preset} instead'))
    if not ok:
        sys.exit('verification FAILED - figure not written')
    print('  all checks passed')
    if a.check:
        return

    png = _paths.figure(f'vmaf_overview_73_{a.target}_p{a.preset}.png')
    pdf = _paths.figure(f'vmaf_overview_73_{a.target}_p{a.preset}.pdf')
    draw(d, col, label, a.preset, dev, png, pdf)
    print(f'\nwrote {png}\n      {pdf}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Merge the original 47 and the new 26 into one 73-source dataset with a single schema.

The two halves were produced by different scripts and differ in three ways that are NOT free
renames. Each is handled explicitly rather than papered over:

1. `bitrate_kbps_carried` (old) vs `bitrate_kbps` (new).
   The old column name records that its bitrate was CARRIED from an earlier measurement rather
   than recomputed per cell; the new one was measured directly from each encode. The 2026-09-10
   verification showed the carried values reproduce to 0.000%, so they are numerically
   equivalent - but the provenance differs, which is why every row also gets an `era` column.
   The old name is kept because 6 downstream scripts read it.

2. `stored_v061` - MUST exist for the new rows.
   baseline.py line 74 uses `d.stored_v061` as the v0.6.1 TARGET, and 28 scripts reference it.
   Without it the new sources would silently produce NaN targets and drop out of every v0.6.1
   fit. For the old 47 it is the originally-stored (rounded) value, sitting ~3.3e-5 from the
   re-scored `vmaf_v061`. The new 26 had exactly one scoring pass, so their stored value IS
   `vmaf_v061`; it is copied across.

3. `repro_err` - deliberately left NaN where nothing was verified.
   For the old 47 it is |stored - fresh| from the 2026-09-10 full re-score (max 2.6e-4).
   The new 26 have never had a full re-score. Writing 0.0 there would assert a check that was
   never run, so unverified new rows get NaN, and only the cells actually re-encoded by
   verify_random73.py get a measured value. NaN means "not checked", not "perfect".

Also: lowres `enc_s` (new) -> `enc_time_s` (old name); `scale` is added to the old lowres rows,
where it is 'half' for all 47 (verified: pixel ratio 0.248-0.250 for every source).

mainstream_sources.txt is extended by the rule recovered from the existing file: mainstream =
natural camera classes a1-a5; b1_syn and b2_scc excluded. That rule reproduces the existing
36-source list exactly, with no counterexamples, so the 13 new camera sources join it and the
13 new synthetic/screen sources do not -> 49 of 73.

-> dense_vmaf_73.csv, lowres_vmaf_73.csv, feat_73.csv, mainstream_sources_73.txt, DATASET73.md
"""
import os, sys
import numpy as np, pandas as pd

V = os.path.expanduser('~/vmaf')
D = next((p for p in [r'C:/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data',
                      '/mnt/c/Users/Sadegh/Desktop/ML/VMAF_v1_Results/data']
          if os.path.exists(f'{p}/dense_vmaf_v1.csv')), None)
OUT = V

# ---------------------------------------------------------------- dense
o = pd.read_csv(f'{D}/dense_vmaf_v1.csv'); o['era'] = 'old47'
n = pd.read_csv(f'{V}/new_dense_v1.csv');  n['era'] = 'new26'
n = n.rename(columns={'bitrate_kbps': 'bitrate_kbps_carried'})
n['stored_v061'] = n['vmaf_v061']          # one scoring pass: the stored value IS this
n['repro_err']   = np.nan                  # never re-scored; NaN = not checked
COLS = ['source', 'width', 'height', 'preset', 'crf', 'vmaf_v061', 'vmaf_v1', 'vmaf_v1_hfr',
        'bitrate_kbps_carried', 'enc_time_s', 'score_s', 'stored_v061', 'repro_err', 'era']
dense = pd.concat([o[COLS], n[COLS]], ignore_index=True).sort_values(
    ['source', 'preset', 'crf']).reset_index(drop=True)

# fold in whatever verify_random73.py actually confirmed
vf = f'{V}/verify73.csv'
nver = 0
if os.path.exists(vf):
    v = pd.read_csv(vf)
    key = dense.set_index(['source', 'preset', 'crf']).index
    vk = v.set_index(['source', 'preset', 'crf'])
    err = vk[['d_v061', 'd_v1', 'd_v1_hfr']].abs().max(axis=1)
    m = pd.Series(err.reindex(key).values, index=dense.index)
    hit = m.notna() & (dense.era == 'new26')
    dense.loc[hit, 'repro_err'] = m[hit]
    nver = int(hit.sum())

# ---------------------------------------------------------------- lowres
lo = pd.read_csv(f'{D}/lowres_vmaf_v1.csv'); lo['era'] = 'old47'; lo['scale'] = 'half'
ln = pd.read_csv(f'{V}/new_lowres_v1.csv'); ln['era'] = 'new26'
ln = ln.rename(columns={'enc_s': 'enc_time_s'})
ln['stored_v061'] = ln['vmaf_lr_v061']; ln['repro_err'] = np.nan
LC = ['source', 'lr_w', 'lr_h', 'preset', 'crf', 'vmaf_lr_v061', 'vmaf_lr_v1', 'vmaf_lr_v1_hfr',
      'bitrate_kbps', 'enc_time_s', 'score_s', 'stored_v061', 'repro_err', 'scale', 'era']
lowres = pd.concat([lo[LC], ln[LC]], ignore_index=True).sort_values(
    ['source', 'crf']).reset_index(drop=True)

# ---------------------------------------------------------------- features
fo = pd.read_csv(f'{D}/halfres_feat.csv'); fo['era'] = 'old47'
fn = pd.read_csv(f'{V}/newfeat73.csv');    fn['era'] = 'new26'
feat = pd.concat([fo, fn], ignore_index=True).sort_values(['source', 'scale']).reset_index(drop=True)

# ---------------------------------------------------------------- mainstream
main_old = set(open(f'{D}/mainstream_sources.txt').read().split())
cls = {}
ac = f'{V}/aomctc_class_map.tsv'   # class -> filename, scraped from media.xiph.org/video/aomctc/test_set
if os.path.exists(ac):
    for line in open(ac):
        if '\t' in line and not line.startswith('class\t'):
            d_, f_ = line.rstrip('\n').split('\t')
            cls[os.path.splitext(f_)[0]] = d_
    # the rule must reproduce the existing list exactly, or we do not apply it
    bad = [s for s in o.source.unique()
           if (s in main_old) != cls.get(s, '').startswith('a')]
    if bad:
        sys.exit(f'mainstream rule does not reproduce the existing list: {bad[:3]} - not writing')
    main_new = sorted(s for s in n.source.unique() if cls.get(s, '').startswith('a'))
    mainstream = sorted(main_old | set(main_new))
else:
    main_new = []; mainstream = sorted(main_old)
    print('WARNING: class map missing, mainstream left at the original 36')

# ---------------------------------------------------------------- write
dense.to_csv(f'{OUT}/dense_vmaf_73.csv', index=False)
lowres.to_csv(f'{OUT}/lowres_vmaf_73.csv', index=False)
feat.to_csv(f'{OUT}/feat_73.csv', index=False)
open(f'{OUT}/mainstream_sources_73.txt', 'w').write('\n'.join(mainstream) + '\n')

print(f'dense_vmaf_73.csv   {len(dense):>6} rows | {dense.source.nunique()} sources'
      f' | repro_err known for {int(dense.repro_err.notna().sum())} rows'
      f' ({nver} newly verified)')
print(f'lowres_vmaf_73.csv  {len(lowres):>6} rows | {lowres.source.nunique()} sources'
      f' | scale {dict(lowres.scale.value_counts())}')
print(f'feat_73.csv         {len(feat):>6} rows | {feat.source.nunique()} sources'
      f' | scales {sorted(feat.scale.unique())}')
print(f'mainstream_sources_73.txt  {len(mainstream)} sources (+{len(main_new)} new)')

assert dense.source.nunique() == 73 and len(dense) == 73 * 350
assert lowres.source.nunique() == 73
assert feat.source.nunique() == 73 and len(feat) == 73 * 2
assert dense[COLS[:-1]].isna().sum().drop('repro_err').sum() == 0
print('\nassertions passed: 73 sources, 25550 dense cells, 146 feature rows, no unexpected nulls')

with open(f'{OUT}/DATASET73.md', 'w') as fh:
    fh.write(f"""# The 73-source dataset

{len(dense)} dense cells (73 sources x 10 presets x 35 CRFs), {len(lowres)} probe rows,
{len(feat)} feature rows. Built by `unify_dataset73.py`.

## Scope

This is the complete **pristine SDR video** portion of the AOM-CTC test set: classes
a1_4k (8), a2_2k (22), a3_720p (8), a4_360p (7), a5_270p (4), b1_syn (13), b2_scc (11).

It is NOT all 194 files under `media.xiph.org/video/aomctc/test_set/`. Deliberately excluded:

| group | files | why |
|---|---|---|
| `a1_4k_downscaled` | 40 | the same 8 4K clips at 5 lower resolutions - correlated copies would leak across leave-one-source-out folds |
| `f1_still_HiRes` + `f2_still_MidRes` | 58 | still images, not video; `ti_mean` undefined |
| `e_nonpristine` | 12 | already-compressed sources; measures recompression |
| `hdr1_4k` + `hdr2_2k` | 11 | HDR; the v0.6.1/v1.0.16 models are SDR-trained. **Open decision** |

## The `era` column

Every row carries `era` = `old47` or `new26`, because provenance differs:

- **bitrate**: `old47` carried it from an earlier measurement, `new26` measured it per encode.
  Verified equivalent to 0.000%, but the column name `bitrate_kbps_carried` reflects the older
  semantics.
- **`repro_err`**: `old47` has it for every row (full re-score, max 2.6e-4). For `new26` it is
  NaN except the {nver} cells re-encoded by `verify_random73.py`. **NaN means not checked.**
- **timings**: `enc_time_s`/`score_s` are contended wall-clock on `time.time()` for both eras and
  are NOT suitable for cost claims. Use `wall73.csv` (quiet machine, `time.monotonic()`,
  all 73 in one session) instead, and report x-wall ratios rather than seconds.

## mainstream

`mainstream_sources_73.txt` holds {len(mainstream)} sources. The rule, recovered from the original
36-source list and verified to reproduce it with no counterexamples: **natural camera content
(classes a1-a5) only**; synthetic (`b1_syn`) and screen content (`b2_scc`) are excluded.
""")
print(f'wrote {OUT}/DATASET73.md')

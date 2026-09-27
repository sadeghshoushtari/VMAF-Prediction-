# VMAF Prediction

Measured VMAF surfaces for the AOM-CTC test set, used to study how accurately the
quality/rate surface of an SVT-AV1 encode can be predicted without measuring it directly.

It holds the dataset and its provenance, the competitor comparison, and the method sweeps run
against them. Out-of-fold predictions are included for every configuration, so any reported
comparison can be re-checked by resampling alone, without refitting a model.

The four competitor implementations are reimplementations from their published papers, not the
original authors' code. No public implementation exists for three of them; the VCA feature
extractor is the only original code used. `DATASET.md` and the script docstrings record how each
reimplementation differs from its paper.

## Contents

```
data/
  dense_vmaf_73.csv        25,550 cells — 73 sources x 10 presets x 35 CRFs
  lowres_vmaf_73.csv          365 rows  — half-resolution probe knots
  feat_73.csv                 146 rows  — content features, full and half resolution
  mainstream_sources_73.txt    49 names — the natural-camera subset
  wall73.csv                  657 rows  — per-video cost measurement
  verify73.csv                372 rows  — re-encode verification record
  aomctc_class_map.tsv         96 rows  — source to AOM-CTC class
data/competitors/
  source_features_73.csv     SITI scalars, 73 sources
  vca_full_features_73.csv   16 VCA statistics, 73 sources
  clip_embeddings_73.npz     CLIP ViT-B/16 and B/32, mean+std pooled, 73 sources
results/
  compete73_metrics.csv      competitor comparison, 7 methods x 2 targets
  curve_sweep73_results.csv  interpolant sweep, 42 configurations
  feature_sweep73_results.csv, oneknot_sweep73_results.csv, probe_ladder_dinner.csv
  oof/                       out-of-fold predictions for every configuration above
scripts/
  compete73.py               the competitor comparison
  newsource_competitor_features.py   extend the competitor feature tables to new sources
  curve_sweep73.py, curve_sweep73_family.py, feature_sweep73.py, oneknot_sweep73.py
  probe_ladder_dinner.py     probe-resolution ladder on the worst-predicted source
  download_sources.sh        fetch the raw clips from media.xiph.org
  encode_new_sources.py      build the dense grid and the probe knots
  remeasure_all73.py         content features + per-video wall, one quiet session
  verify_random73.py         re-encode verification, one cell per source
  verify300_new.py           broader re-encode verification
  unify_dataset73.py         merge into the single 73-source schema
```

## The dataset

**73 sources x 10 presets x 35 CRFs = 25,550 cells.** Each cell is one SVT-AV1 encode scored
with three VMAF models (`v0.6.1`, `v1.0.16`, `v1.0.16_hfr`) plus its bitrate.

Scope is the complete **pristine SDR video** portion of the AOM-CTC test set — classes
`a1_4k`, `a2_2k`, `a3_720p`, `a4_360p`, `a5_270p`, `b1_syn`, `b2_scc`. It is not all 194 files
on the server; `DATASET.md` lists what is excluded and why.

The raw `.y4m` clips are ~37 GB and are not in this repository. They are public:
`scripts/download_sources.sh` fetches them from `media.xiph.org/video/aomctc/test_set` and
checks each against the server's content length.

## Verification

Two independent levels.

**Structural**, over all 25,550 cells: no duplicate `(source, preset, crf)`, no nulls, every
source exactly 350 cells, all VMAF values inside [0,100], 2 bitrate inversions out of 25,550
(0.008%, both under 0.71% of the mean).

**Re-encode**, 372 sampled cells: each re-encoded from the raw source and re-scored, then
compared against the stored values.

```
old47   n= 47   max |dVMAF| 0.000e+00   all-3-exact  47/47
new26   n=325   max |dVMAF| 0.000e+00   all-3-exact 325/325
ALL     n=372   max |dVMAF| 0.000e+00   all-3-exact 372/372
```

Every re-encoded cell reproduced bit-identically across all three models; maximum bitrate
deviation 0.0008%, which is the stored value's rounding.

Coverage is **not** uniform, and the `repro_err` column records this honestly: 100% of the
47 original sources (from an earlier full re-score, max 2.6e-4) and 3.6% of the 26 newer ones,
sampled to cover all 260 source x preset pairs and all 35 CRFs. `NaN` in `repro_err` means
*not checked*, not *checked and perfect*.

## Using the timing columns

`enc_time_s` and `score_s` in `dense_vmaf_73.csv` are wall-clock times taken during the
encoding campaign on a contended machine using a clock that is not monotonic on this host.
**They are not suitable for cost claims.**

Use `data/wall73.csv` instead. It was measured in one quiet session on a monotonic clock, all
73 sources together, 3 consecutive repetitions per source with the minimum taken. It records
the symmetric per-video wall (one full-resolution preset-10 CRF-55 encode plus its own libvmaf
pass) and the matching cost of the two-knot probe, and stores encode time both as reported by
the encoder and as true wall clock — the former understates the latter by 10-29%.

Report cost as a ratio to the wall, never in seconds: seconds do not reproduce across sessions
on this hardware (the same wall has measured 8.005 s and 5.766 s), while the ratios reproduce
to within about 1%.

## The `era` column

Every row carries `era` = `old47` or `new26`. The two halves were measured by different scripts
and their provenance differs in ways that are recorded rather than smoothed over — bitrate
carried versus measured per cell, and the `repro_err` coverage above. `DATASET.md` has the
detail.

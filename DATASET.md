# The 73-source dataset

25550 dense cells (73 sources x 10 presets x 35 CRFs), 365 probe rows,
146 feature rows. Built by `unify_dataset73.py`.

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
  NaN except the 325 cells re-encoded by `verify_random73.py`. **NaN means not checked.**
- **timings**: `enc_time_s`/`score_s` are contended wall-clock on `time.time()` for both eras and
  are NOT suitable for cost claims. Use `wall73.csv` (quiet machine, `time.monotonic()`,
  all 73 in one session) instead, and report x-wall ratios rather than seconds.

## mainstream

`mainstream_sources_73.txt` holds 49 sources. The rule, recovered from the original
36-source list and verified to reproduce it with no counterexamples: **natural camera content
(classes a1-a5) only**; synthetic (`b1_syn`) and screen content (`b2_scc`) are excluded.

# Channel count vs. electrode placement in glasses-form EEG seizure detection

When a wearable EEG seizure detector is built into a pair of glasses, it loses
channels *and* it loses freedom over where the electrodes go. Published
glasses-form results report the combined loss. This repository separates the
two.

Every channel count is evaluated twice:

| | electrode placement | channels |
|---|---|---|
| `Best-n` | unconstrained, ranked from the data | 7 / 4 / 2 |
| `Glass-n` | constrained to the glasses-frame chain | 7 / 4 / 2 |
| `Full-18` | standard double banana, the reference point | 18 |

`Best-n` channels are ranked by summed random-forest feature importance and
re-selected inside every fold. `Glass-n` electrodes are fixed to the chain
FP1–F7–T7–FT9–FT10–T8–F8–FP2, which traces the outline of a spectacle frame;
*n* electrodes in a chain give *n*−1 independent bipolar derivations.

The gap between the two curves at the same *n* is the cost of the placement
constraint alone. Three model families (random forest, logistic regression,
MLP) are run as a control: a curve shape that survives all three is a property
of the data rather than of one classifier.

## Status

Phase 0 of the Beniczky & Ryvlin (2018) validation framework — a retrospective
study on a public archive, not a clinical or prospective evaluation. Nothing
here has been validated on a device. The manuscript describing this work is
written outside this repository and is not included here.

## Results in short

Subject-wise cross-validation, event-level SzCORE scoring, random forest:

- Per-subject pass rates against the two pre-registered criteria range from
  4.35 % to 30.43 % depending on configuration and criterion.
- Pass rate does not fall monotonically with channel count.
- Window-level AUC-ROC stays in a narrow band (0.72–0.83) across all seven
  configurations, while event-level pass rates vary far more widely.

`output/06_results_szcore.csv` is the source of truth for every number above.
Per-subject detail is in `output/06a_per_subject_results_raw.csv`.

## Data

[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/) from
PhysioNet. The recordings are **not** redistributed here — download them into
a `CHB-MIT Dataset/` directory at the repository root. Everything is sampled at
256 Hz.

Three dataset decisions are applied before anything else runs, and each one
changes the denominator:

- **`chb21` is merged into `chb01`.** They are the same person recorded 1.5
  years apart. Keeping them separate leaks a subject across the train/test
  split. This leaves **23 subjects, not 24**.
- **The monopolar-montage files of `chb12` are excluded** (13 seizures, about 3
  hours), following Zanetti et al. 2022.
- **Duplicate derivations are dropped at load time**: `P7-T7` is `T7-P7` with
  inverted polarity, and the second `T8-P8` is an exact copy of the first. 21
  unique derivations remain. Without this, `Best-2` could silently select the
  same signal twice.

Two evaluation bases exist because three Glass-7 channels (`T7-FT9`,
`FT9-FT10`, `FT10-T8`) are missing from 28 files:

| | files | hours | seizures | used by |
|---|---|---|---|---|
| Tier A | 668 | 945.49 | 181 | the other six configurations |
| Tier B | 640 | 917.48 | 177 | `Glass-7` only |

Both tiers contain the same 23 subjects. The bias this introduces is measured,
not assumed — see `output/01b_tier_effect_glass7.md` and the
`known_comparison_limitations` block of `output/metadata.json`.

## Install

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Developed on Python 3.13.2 / Windows 11. `requirements.txt` is fully pinned,
including transitive dependencies; `output/metadata.json` records the exact
versions that produced the committed results.

## Reproduce

Every script runs standalone via `python src/<name>.py` and writes into
`output/`. Run them in this order.

**0 — audit.** Reads EDF headers only, no signal data.

```
python src/audit.py                        -> 00_channel_audit.csv, 00_channel_audit_summary.md
```

**1 — preprocess and extract features.** The long step: about 13 hours on a
desktop CPU. Checkpointed per file, so an interrupted run resumes where it
stopped.

```
python src/run_pipeline.py                 -> output/features/<subject>.parquet
python src/write_dataset_summary.py        -> 01_dataset_summary.md
python src/write_metadata.py               -> metadata.json
```

Three optional gates belong here, before any result is trusted:
`spotcheck_hybrid_windowing.py` (windowing scheme), `validate_features.py`
(fast implementation vs. the frozen reference), `scan_nonfinite_features.py`
(non-finite cells).

**2 — folds, configs, training.** 312 runs across configuration × model × fold.

```
python src/cv_folds.py                     -> 02_cv_folds.json
python src/model_configs.py                -> 03_model_configs.json
python src/train.py                        -> output/04_predictions/
```

`train.py` saves per-window probabilities and stops there. No threshold and no
post-processing decision is made during training, so every downstream choice
can be re-run without retraining.

**3 — operating points and scoring.** The sweep takes roughly 3 hours.

```
python src/evaluate_sweep.py                        -> output/05a_threshold_sweep/
python src/select_operating_points_percentile.py    -> 06_operating_points_percentile.csv
python src/rescore_per_subject.py                   -> 06_operating_points_percentile_per_subject.csv
python src/build_final_results.py                   -> 06_results_szcore.csv, 06_results_ali.csv,
                                                       06a_per_subject_results_raw.csv
```

**4 — figures.**

```
python src/build_figure_data.py             -> output/07_figures/data_*.csv
python src/build_tradeoff_curve.py          -> output/07_figures/data_tradeoff_curve.csv
python src/make_figures_plotnine.py         -> output/07_figures/*.svg
```

**Supporting analyses**, each independent of the others:

```
python src/postprocessing_sensitivity.py    -> 05_postprocessing_sensitivity.md
python src/measure_model_size.py            -> 06d_model_size_inference.md
python src/glass7_tier_effect.py            -> 01b_tier_effect_glass7.md
python src/entropy_pilot.py                 -> feature-cost pilot, see 00f_*.md
python src/entropy_pilot_analyze.py
python src/entropy_pilot_impact_test.py
```

`output/features/`, `output/04_predictions/` and `output/05a_threshold_sweep/`
are large and regenerable, so they are not committed. Everything else in
`output/` is.

## Where two scripts look like duplicates

Three pairs exist on purpose. Picking the wrong one silently changes the
results.

| use this | not this | why the other one is still here |
|---|---|---|
| `select_operating_points_percentile.py` | `select_operating_points.py` | The absolute-threshold version is a recorded negative finding: thresholds chosen on the tune set did not transfer to the test set. Its output (`06_operating_points.csv`) is the evidence. |
| `make_figures_plotnine.py` | `make_figures.py` | Same data, two renderers. The matplotlib version came first and is kept for comparison. Neither re-scores anything. |
| `features.py` | `features_reference.py` | `features_reference.py` is a frozen, deliberately slow implementation. `validate_features.py` asserts the fast one matches it on real EEG. |

`output/00b_tier_breakdown.md` is likewise kept although it is superseded: it
records the tier approach that was considered and rejected, and it says so in
its own first lines.

## Figures

`output/07_figures/` holds three generations of the same figures:

- `paper_fig1.svg`, `paper_fig2.svg`, `paper_fig3.svg` — the three that go in
  the manuscript. Rendered at IEEE single-column width (3.5 in) with no
  in-figure title, because the caption below the figure carries that text.
- `fig*_plotnine.svg` — the exploratory plotnine set, wider, with titles drawn
  inside the image.
- `fig*.svg` — the original matplotlib set.

All of them are generated from the same `data_*.csv` files. Producing a figure
never involves re-scoring.

## Locked parameters

Set before any result was seen and unchanged since. The full list with reasons
lives in `output/metadata.json`.

| | |
|---|---|
| filter | Butterworth order 4 (design order), zero-phase, 0.5–40 Hz bandpass |
| notch | 60 Hz, Q = 30 — the data was recorded in Boston, not Europe |
| window | 4 s, 0.5 s step (87.5 % overlap) |
| label | window counts as seizure at ≥ 50 % overlap |
| features | 89 per channel (83 DIHC-equivalent + 6 approximate zero-crossing) |
| CV | subject-wise 5-fold for the curves, leave-one-out for the main configurations |
| balancing | fit set only, undersampled to ~1:10 plus `class_weight='balanced'` |
| scoring | SzCORE (`timescoring` 0.0.7) primary, Ali et al. 2024 rule secondary |
| `random_state` | 42 everywhere |

Two sufficiency criteria were registered before any result was seen:

| use case | criterion |
|---|---|
| seizure counting for medication titration | sensitivity ≥ 60 %, ≤ 5 false alarms/day |
| real-time carer alerting | sensitivity ≥ 80 %, ≤ 2 false alarms/day |

Nothing is balanced or resampled on the test side. Balancing changes the
recording-hours denominator, which would make false alarms per day
meaningless.

## Known limitations

Recorded in `output/metadata.json` rather than summarised here, so that the
list cannot drift from the runs that produced it:

- `subjects.selection_deviation`, `windowing.deviation` and
  `features.feature_scope_deviation` — every departure from the locked
  parameters, each with its reason, approval date and who approved it.
- `known_comparison_limitations` — the Glass-7 versus Best-7 comparability
  problem, including the measured size of the Tier A/B bias.

The filter is zero-phase and therefore non-causal, so the pipeline as written
cannot run on a live signal. Operating points are calibrated per fold, not per
patient. Both are properties of this study design, not oversights.

## AI assistance

The author used Claude (Sonnet 5 and Opus 5 models) to write and test all
the Python code, AI-ASSISTANCE.md, and README.md, and to review the writing
for language consistency, logical consistency, grammar, and consistency
between the writing and the code. The author inspected and revised the code
before the final analysis was run, and reviewed the markdown files' content
as needed. The author takes full responsibility for the content of the
research. The repository contains the final reviewed code used to produce
the reported results.

`AI-ASSISTANCE.md` carries the same statement in Thai, plus a file-by-file
breakdown of what was written by whom and where each research decision and its
approval is recorded.

## Reuse

Code in `src/` may be reused with attribution. The CHB-MIT recordings are not
covered by anything in this repository — see
[PhysioNet's terms](https://physionet.org/content/chbmit/) for the data.

# SLEAP Social-Behavior Lab

An interactive, hands-on course in analyzing **animal social behavior from pose-tracking data**.
You start from raw [SLEAP](https://sleap.ai) pose output and work all the way to a trained behavior
classifier, learning the standard computational-ethology toolkit along the way:

**feature extraction → dimensionality reduction → clustering → statistics → labeling → machine learning.**

Everything runs in [**marimo**](https://marimo.io) — reactive Python notebooks where dragging a
slider instantly re-runs the analysis, so you can *watch* how each modeling choice changes the
result. The material uses a small bundled dataset of mouse social interactions (three mice per cage,
recorded continuously) from a "despotism" experiment with three phases (`pre`, `dep`, `post`), which
gives you both a **rank** axis (Dom / Mid / Sub) and an experimental **condition** axis to test.

## What you'll build

The course runs as **eight** Week-1 reactive notebooks (plus a Week-2 neural arm below). Each row links straight to a free
[molab](https://molab.marimo.io) cloud kernel — click **Run** to open that lesson in the browser
(nothing to install; each notebook self-bootstraps its data).

| # | Notebook | You learn | You produce | Run |
|---|----------|-----------|-------------|-----|
| 01 | `01_raw_signal` | What SLEAP outputs; the keypoint tensor `(frames, mice, nodes, xy)`; why a track slot ≠ an identity | Load & visualize a real `.slp`; scrub the skeleton; audit swaps at contact | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/01_raw_signal.py) |
| 02 | `02_body_eye_view` | Turning raw keypoints into **body-frame (egocentric)** social features (center + rotate into one animal's frame) | A per-event interpretable 19-feature vector, arena-invariant | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/02_body_eye_view.py) |
| 03 | `03_signal_in_time` | Reading the signal in value, **time & frequency** (Morlet wavelet), and **who-leads-whom** (lead-lag) — with honest nulls | A rhythm spectrogram + a shuffle-tested coordination estimate | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/03_signal_in_time.py) |
| 04 | `04_collapse_pca` | **PCA** and the dimensionality of behavior; covariate **residualization** as a *choice* | The ~6-axis behavioral manifold; the aggression cost of dropping PC1 | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/04_collapse_pca.py) |
| 05 | `05_collapse_map` | **UMAP** + **HDBSCAN**; the role of every hyperparameter; hierarchical subclustering | A 2-D behavioral map carved into data-driven **syllables** | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/05_collapse_map.py) |
| 06 | `06_reading_the_map` | Cluster **enrichment** for condition / sex / rank (χ², Bonferroni) — and the **pseudoreplication** reversal | Which effects survive when the *cage* is the unit | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/06_reading_the_map.py) |
| 07 | `07_behavior_in_time` | The **transition grammar** (Markov chain) and **activity clock** of a continuously-tracked cage, vs a shuffle null | A transition matrix + stationary distribution + a bootstrapped daily rhythm | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/07_behavior_in_time.py) |
| 08 | `08_decoder_graduates` | Training an **MLP** and evaluating it on a **held-out cage**; the same pipeline on a neural raster | Predicted-aggression clips + ROC/PR on unseen Cage 16 | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/08_decoder_graduates.py) |

Each notebook shows the **equations** behind the method (e.g. the UMAP objective, PCA
eigendecomposition) and a short **why-we-use-this** justification, not just code.

## Week 2 — The Neural Twin (notebooks 09–14)

Week 2 reuses Week 1's computational moves to read the brain (calcium imaging, demixing, tuning
curves, neural decoding). Beyond the eight Week-1 lessons above, a second week adds **six neural
notebooks**: each re-runs one Week-1 idea on neural data instead of pose — same math, new signal —
and ties itself explicitly to the Week-1 **twin** it mirrors. Notebooks 13 and 14 share the same
`SI3_2022` social-isolation dataset (behavior in 13, calcium in 14).

| # | Notebook | What it does | Week-1 twin | Run |
|---|----------|--------------|-------------|-----|
| 09 | `09_motion_correction` | Register a drifting miniscope movie (raw → rigid → piecewise-rigid) and prove it with a **motion index** | **01** `raw_signal` — align the signal across time before you read it | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/09_motion_correction.py) |
| 10 | `10_calcium_extraction` | Background-subtract a striatal movie and pull one cell's calcium trace from a hand-placed **ROI** | **02** `body_eye_view` — choosing an ROI *is* choosing a feature | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/10_calcium_extraction.py) |
| 11 | `11_demixing_sources` | **CNMF** demixes an optical mixture into per-cell footprints `A` + traces `C`; sort them into a neural sequence | **04** `collapse_pca` — decompose a mixture into its sources | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/11_demixing_sources.py) |
| 12 | `12_place_and_grid_cells` | Occupancy-normalized 2-D **rate maps** + Skaggs spatial information, validated against a shuffle null | **02** `body_eye_view` (tuning) — a rate map is a tuning curve over space | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/12_place_and_grid_cells.py) |
| 13 | `13_social_ethograms` | Build `(9, T)` social-contact **ethograms** for the SI3_2022 cohort; verify `is_social` and read the isolation effect honestly | **03/05** `signal_in_time` / `collapse_map` — discrete states stacked over time | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/13_social_ethograms.py) |
| 14 | `14_neural_social_decode` | Train a population **decoder** (LogReg + stratified CV + AUROC) to read social state off calcium on the same SI3_2022 mice | **08** `decoder_graduates` — the same estimator, now reading a brain | [Run](https://molab.marimo.io/github/Elmaestrotango/sleap-social-behavior-lab/blob/main/notebooks/14_neural_social_decode.py) |

## Quick start

Install [uv](https://docs.astral.sh/uv/) (a fast Python package manager), then:

```bash
uv sync                       # create the environment (Python 3.11)
uv run marimo edit notebooks/01_raw_signal.py
```

Work through `01 → 08` in order. In marimo, edit any cell or drag any slider and every dependent
cell updates automatically. To just *view* a notebook without editing:

```bash
uv run marimo run notebooks/05_collapse_map.py
```

## Run in the browser (no install for students)

Give students a browser experience with nothing to install. All the options below use a **real
Python kernel**, so `numba` / `umap-learn` / `hdbscan` work. The WebAssembly / GitHub-Pages
export does **not** — those libraries have no in-browser (Pyodide) build and lessons 05–08 need
them, so a static WASM site would break at the map notebook.

- **One link for the whole course (recommended).** `serve.py` publishes a landing page plus all
  eight lessons in order under a single URL, each with its own isolated kernel per visitor. Try it
  with `uv run python serve.py` (→ <http://localhost:7860>), then host it free on a Hugging Face
  Docker Space or self-host behind a tunnel. See [`DEPLOY.md`](DEPLOY.md).
- **One notebook at a time (molab).** [molab](https://molab.marimo.io) runs a single notebook on
  a free cloud kernel (the **Run** links in the lesson table above point straight to each one).
  Each notebook declares its dependencies inline (a PEP&nbsp;723 `# /// script` block pinned to
  `pyproject.toml`) and **self-bootstraps**: if it can't find a local checkout it downloads
  `course_utils.py` and the bundled data straight from this repo, so there's nothing to upload.
  It's one link *per lesson* rather than one course site — students open `01`…`08` in turn. See
  [`DEPLOY.md`](DEPLOY.md).

## What's in `data/`

Small, self-contained, no video required (everything renders skeletons on a blank canvas):

- `train_events.npz` — ~1.5k social-approach events: short keypoint windows
  `(N, T, 3, 15, 2)` (mice ordered *approacher, approachee, bystander*), per-mouse ranks, condition,
  and a registry label where one exists.
- `heldout_events.npz` — events from a **held-out cage** (camera 16) with ground-truth aggression
  labels, for honest evaluation in notebook 08.
- `answer_key.csv` — ground-truth categories for the training events (a grading aid / fallback for
  the labeling step in notebook 08).
- `cohort_meta.csv` — per-cage metadata (sex, rank order, condition).
- `raw_slp/example_*.slp` — a few short real SLEAP clips for notebook 01.

## Provenance (instructors)

The bundle is produced from a lab pipeline by `tools/build_dataset.py` (+ `tools/trim_slp.py`),
which require access to the source data and are **not** needed to run the course. See those files
for exactly how each field was derived. `tools/decode_example_slp.py` turns a raw `.slp` into the
small `example_slp_decoded.npz` that notebook 01 loads (so students need no `sleap-io`). These
tools need `sleap-io`, which is kept out of the default install — get it with `uv sync --extra
build`.

The Week-2 neural notebooks (09–14) were remade from the original EDGE Colab sources kept in
[`2025/`](2025/) (the NEU 457 lineage) — one legacy script per neural lesson, preserved for provenance.

## Skeleton

15 nodes, star topology with two hubs (`head`=1, `TTI`=11, the tail-torso intersection):

```
0 nose   1 head   2 L_ear   3 L_shoulder   4 neck   5 R_ear   6 R_shoulder
7 L_haunch   8 R_haunch   9 tail_1   10 tail_0   11 TTI   12 tail_2   13 tail_tip   14 trunk
```

Mice are colored by dominance **rank**: 🔴 Dom, 🔵 Mid, 🟢 Sub.

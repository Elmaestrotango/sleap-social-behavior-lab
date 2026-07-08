# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "plotly>=5.20",
#     "h5py>=3.10",
#     "gdown>=5.1",
#     "openpyxl>=3.1",
#     "imageio>=2.34",
#     "imageio-ffmpeg>=0.4",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os, sys, urllib.request
    import numpy as np
    import plotly.graph_objects as go
    _RAW = os.environ.get("COURSE_REPO_RAW",
        "https://raw.githubusercontent.com/Elmaestrotango/sleap-social-behavior-lab/main")
    def _find_root():
        p = os.getcwd()
        for _ in range(6):
            if os.path.isdir(os.path.join(p, "course")):
                return p
            p = os.path.dirname(p)
        return None
    ROOT = _find_root() or os.getcwd()
    _nu = os.path.join(ROOT, "course", "neural_utils.py")
    if not os.path.exists(_nu):
        os.makedirs(os.path.dirname(_nu), exist_ok=True)
        urllib.request.urlretrieve(_RAW + "/course/neural_utils.py", _nu)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import neural_utils as nu
    CACHE = nu.cache_dir(ROOT)
    return CACHE, ROOT, go, np, nu


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB11 · Demixing the Sources

        > **WEEK 2 — THE NEURAL TWIN**
        >
        > In Week 1 you learned to *read behavior* from pose. In **NB04** you took a wide,
        > redundant feature matrix and ran **PCA / source separation**: you turned an
        > entangled mixture into a small set of independent components, each one a clean
        > "direction" of variation you could name. That was the computational move —
        > **decompose a mixture into its sources.**
        >
        > This notebook is that move, run on the **brain instead of the body.** A one-photon
        > miniscope sees a 2-D image where the light from many neurons **overlaps in the same
        > pixels** — a spatial mixture. **CNMF** (constrained non-negative matrix
        > factorization) demixes it: it factors the movie into a set of **spatial footprints
        > `A`** (where each source lives) and **temporal traces `C`** (when each source
        > fires). Same idea as PCA — *separate a mixed signal into components* — but the
        > components here are literally single cells, and the mixing is optical, not
        > statistical.
        >
        > **The rig for today:** one striatal recording, `221007_4-0_D2` — **202 demixed
        > neurons**, ~16,800 frames at 30 fps. We'll look at the footprints, the traces, and
        > then find a moment where the population fires as a **sequence.**
        """
    )
    return


@app.cell
def _(nu):
    _d = nu.load_cnmf()
    A = _d["A"]
    C = _d["C"]
    Cn = _d["Cn"]
    S = _d["S"]
    Fs = _d["Fs"]
    img_shape = _d["img_shape"]
    n_neurons = _d["n_neurons"]
    n_frames = _d["n_frames"]
    # z-scored population raster, one row per neuron (source-script convention: per-neuron
    # mean/std across time). Computed once, reused everywhere below.
    C_z = nu.zscore(C.T, axis=1)
    # The 2025 script's behavior-clock "arena entry" frame, converted to the imaging clock
    # (behavior 25 fps -> imaging 30 fps). This anchors the sequence window + the exercise.
    ENTRY = int(7488 * (30 / 25))   # -> 8985
    WIN_LEN = 3 * 60 * 30           # 3 minutes at 30 fps -> 5400 frames
    return A, C, C_z, Cn, ENTRY, Fs, WIN_LEN, img_shape, n_frames, n_neurons


@app.cell(hide_code=True)
def _(Fs, mo, n_frames, n_neurons):
    mo.md(
        f"""
        ---
        ## 1. The correlation image — the "photo" the demixing starts from

        Before any demixing, a standard first look is the **local correlation image `Cn`**:
        each pixel is colored by how strongly its brightness fluctuates **together with its
        neighbors** over time. A lone bright pixel is noise; a *blob* of pixels that rise and
        fall in lockstep is a candidate cell body. `Cn` is where a human (or CNMF's
        initialization) first sees neurons emerge from the raw movie.

        This recording has **{n_neurons} neurons** across **{n_frames:,} frames** at
        **{Fs:.0f} fps** ({n_frames / Fs / 60:.1f} min). The bright rings and disks below are
        the sources CNMF will pull apart.
        """
    )
    return


@app.cell
def _(Cn, nu):
    nu.image_fig(Cn, title="Correlation image Cn — pixels that co-fluctuate with their neighbors",
                 colorscale="Viridis", colorbar_title="local corr", height=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. One source at a time — the footprint viewer

        CNMF's spatial output `A` is a matrix with **one row per neuron** and one column per
        pixel (`202 × 360000`). Reshape a single row back to the `600 × 600` image and you get
        that neuron's **spatial footprint** — its own private slice of the field of view, the
        set of pixels the demixing assigned to *that* cell. This is the imaging analog of a
        **PCA component's loading vector**: a spatial pattern that says "this source lives
        here."

        Drag the slider to walk through all 202 sources. On the **left** is the footprint
        (where the neuron is); on the **right** is its **calcium trace `C[:, k]`** (when it
        fires). Together they are one demixed source: a *where* and a *when*.
        """
    )
    return


@app.cell
def _(mo, n_neurons):
    neuron_ind = mo.ui.slider(0, n_neurons - 1, value=148, step=1,
                              label="neuron index (source)", debounce=True, full_width=True)
    return (neuron_ind,)


@app.cell
def _(A, C, img_shape, mo, neuron_ind, nu):
    _k = neuron_ind.value
    _fp = nu.footprint(A, _k, img_shape)
    _fig_fp = nu.image_fig(_fp, title=f"Footprint A[{_k}] — where source {_k} lives",
                           colorscale="Viridis", colorbar_title="weight", height=420)
    _fig_tr = nu.trace_fig(None, C[:, _k], title=f"Calcium trace C[:, {_k}] — when it fires",
                           xlabel="Time (frames)", ylabel="calcium (a.u.)", height=420)
    _fig_tr.update_traces(line=dict(color="#2ca02c", width=1))
    mo.vstack([neuron_ind, mo.hstack([_fig_fp, _fig_tr], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. All sources at once — the footprint montage

        Peak-normalize every footprint (so a dim cell and a bright cell count equally) and take
        the **maximum across all 202 sources** at each pixel. The result is a single image
        showing **every demixed neuron's territory** laid over the field of view — the full cast
        of cells CNMF found, and how they tile the tissue. Compare it back to `Cn` above: the
        bright blobs in the correlation image should reappear here as cleanly separated
        footprints.
        """
    )
    return


@app.cell
def _(A, img_shape, nu):
    nu.image_fig(nu.footprint_montage(A, img_shape),
                 title="Footprint montage — max projection of all 202 peak-normalized sources",
                 colorscale="Viridis", colorbar_title="peak-norm weight", height=520)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. The population raster — every source, all the time

        Stack all 202 traces into a matrix — one **row per neuron**, one column per frame — and
        z-score each row so a small quiet cell is on the same footing as a loud one. This is the
        **population raster**: the entire demixed recording in one image. Bright vertical smears
        are moments when many neurons fire together; horizontal streaks are individual cells that
        stay active.

        The **contrast** slider sets the color ceiling (`zmax`). Turn it down to bring out weak
        transients; turn it up to keep only the biggest calcium events.
        """
    )
    return


@app.cell
def _(mo):
    raster_zmax = mo.ui.slider(2.0, 12.0, value=6.0, step=0.5,
                               label="contrast ceiling (raster zmax, z-units)",
                               debounce=True, full_width=True)
    return (raster_zmax,)


@app.cell
def _(C_z, mo, np, nu, raster_zmax):
    # Downsample columns for a snappy display (compute stays full-res elsewhere).
    _T = C_z.shape[1]
    _step = max(1, _T // 1500)
    _disp = C_z[:, ::_step]
    _x = np.arange(0, _T, _step)
    _fig = nu.raster_fig(_disp, title="Population raster — z-scored C.T (all 202 sources)",
                         xlabel="Time (frames)", ylabel="Neuron", colorscale="Viridis",
                         zmin=0.0, zmax=float(raster_zmax.value), colorbar_title="z", height=460)
    _fig.data[0].x = _x
    mo.vstack([raster_zmax, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Stacked traces — reading sources like a physiologist

        A raster hides the *shape* of each transient. Line them up instead: take the first `N`
        sources, min-max normalize each, and **offset every trace vertically** so they don't
        overlap — the classic multi-unit stack a physiologist reads off an oscilloscope. Each
        line is one demixed neuron; the sharp asymmetric rises are calcium transients (fast up,
        slow decay).
        """
    )
    return


@app.cell
def _(mo):
    n_stack = mo.ui.slider(10, 50, value=30, step=5,
                           label="number of sources to stack", debounce=True, full_width=True)
    return (n_stack,)


@app.cell
def _(C, go, mo, n_frames, n_stack, np):
    _n = int(n_stack.value)
    _sub = C[:, :_n].astype(float)
    _mn = _sub.min(axis=0, keepdims=True)
    _mx = _sub.max(axis=0, keepdims=True)
    _norm = (_sub - _mn) / np.where(_mx - _mn == 0, 1.0, _mx - _mn)   # (T, n) in [0, 1]
    _step = max(1, n_frames // 3000)
    _t = np.arange(0, n_frames, _step)
    _fig = go.Figure()
    for _j in range(_n):
        _fig.add_scatter(x=_t, y=_norm[::_step, _j] + _j * 0.8, mode="lines",
                         line=dict(width=1), showlegend=False, hoverinfo="skip")
    _fig.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=40, b=10),
                       title=f"First {_n} sources — min-max normalized, offset by 0.8")
    _fig.update_xaxes(title="Time (frames)", range=[0, n_frames])
    _fig.update_yaxes(title="source (offset)", showticklabels=False)
    mo.vstack([n_stack, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. The neural sequence — sorting sources by *when* they fire

        Here is the payoff. Pick a window of the recording and, within it, order the neurons by
        the **time of their first big calcium event** (first frame each z-scored trace crosses a
        threshold). CNMF gave us the sources in an arbitrary order; re-sorting them by
        first-activation time reveals whether the population fires as a **sequence** — one cell
        after another, like a wave.

        On the **left** is the window in the raw source order (a speckle). On the **right** is the
        same window after `nu.sequence_sort` — if there's a sequence, the activity collapses onto
        a **diagonal**: early-firing neurons at the bottom, late-firing at the top. The default
        window is centered on the 2025 script's arena-entry moment, where the striatum lights up.
        """
    )
    return


@app.cell
def _(ENTRY, WIN_LEN, mo, n_frames):
    win_start = mo.ui.slider(0, n_frames - 600, value=ENTRY, step=30,
                             label="window start (frame)", debounce=True, full_width=True)
    win_len = mo.ui.slider(600, 8000, value=WIN_LEN, step=100,
                           label="window length (frames)", debounce=True, full_width=True)
    seq_thresh = mo.ui.slider(2.0, 8.0, value=5.0, step=0.5,
                              label="activation threshold (z)", debounce=True, full_width=True)
    return seq_thresh, win_len, win_start


@app.cell
def _(C_z, go, mo, n_frames, np, nu, seq_thresh, win_len, win_start):
    from scipy.stats import spearmanr

    def _seqness(raster, thr):
        # |Spearman| between row position and first-crossing time: 0 = no order, 1 = perfect diagonal
        _first = np.argmax(raster > thr, axis=1)
        _r, _ = spearmanr(np.arange(raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))

    _s = int(win_start.value)
    _e = min(_s + int(win_len.value), n_frames)
    _thr = float(seq_thresh.value)
    _win = C_z[:, _s:_e]
    _order = nu.sequence_sort(_win, thresh=_thr)
    _sorted = _win[_order]

    _q_un = _seqness(_win, _thr)
    _q_so = _seqness(_sorted, _thr)

    # downsample columns for display only
    _step = max(1, _win.shape[1] // 1200)
    _xd = np.arange(_s, _e, _step)
    _left = nu.raster_fig(_win[:, ::_step], title=f"unsorted  ·  sequenceness = {_q_un:.2f}",
                          xlabel="Time (frames)", ylabel="Neuron (CNMF order)",
                          colorscale="Viridis", zmin=0.0, zmax=6.0, colorbar_title="z", height=460)
    _left.data[0].x = _xd
    _right = nu.raster_fig(_sorted[:, ::_step],
                           title=f"sorted by first activation  ·  sequenceness = {_q_so:.2f}",
                           xlabel="Time (frames)", ylabel="Neuron (sequence order)",
                           colorscale="Viridis", zmin=0.0, zmax=6.0, colorbar_title="z", height=460)
    _right.data[0].x = _xd
    mo.vstack([mo.hstack([win_start, win_len, seq_thresh]),
               mo.hstack([_left, _right], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reference — CNMF, and where the analogy stops": mo.md(
            r"""
            **The method.** Pnevmatikakis et al. 2016, *Neuron* 89(2):285–299,
            "Simultaneous Denoising, Deconvolution, and Demixing of Calcium Imaging Data"
            (**CNMF**). The one-photon variant used here is **CNMF-E**
            (Zhou et al. 2018, *eLife* 7:e28728 — the striatal miniscope dataset this course
            also draws NB10 from). CNMF factors the movie `Y ≈ A · C + b` into non-negative
            **spatial footprints `A`** and **temporal traces `C`** (plus a background term `b`) —
            a constrained matrix factorization, the same family as the PCA/ICA decomposition you
            ran on behavior in NB04.

            **The shared mathematics.** Both PCA and CNMF write a data matrix as a **low-rank
            product of a spatial factor and a temporal factor**. PCA picks orthogonal directions
            of maximum variance; CNMF picks **non-negative, spatially-localized** factors so each
            component is a physically plausible cell. Non-negativity + a sparse deconvolution
            model is what turns "a component" into "a neuron."

            **Honest note on `S`.** The file also carries `S`, CNMF's **deconvolved spike
            estimate**. Deconvolution is calibrated for two-photon data; for **one-photon**
            miniscope recordings like this one, `S` is *not* validated — the spike times are a
            model output, not ground truth. We show `C` (the calcium) and treat `S` with
            suspicion. Do not report `S` as spike counts.

            **Where the analogy stops.** A PCA component is a *statistical* axis — abstract, can
            be negative, need not correspond to anything real. A CNMF footprint is a *physical
            claim*: "a cell is here." That is a stronger, falsifiable statement, and it can be
            **wrong** — two adjacent cells can be merged into one source, or one cell split into
            two, and no amount of variance-explained will tell you. Demixing is only as good as
            the footprints, and footprints are inferred, not observed.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 7. Exercise — does sorting actually make a sequence?

        > **Hypothesis.** Around arena entry, the striatal population fires as a **temporal
        > sequence** — so ordering neurons by their first activation time turns a formless raster
        > into a **diagonal** one. Sorting should raise a "sequenceness" score *far* above the
        > unsorted baseline.

        **Toolbox.**

        - `C_z` — the `(202, n_frames)` z-scored population raster (already built above).
        - `ENTRY` (= 8985), `WIN_LEN` (= 5400) — the arena-entry window on the imaging clock.
        - `nu.sequence_sort(raster, thresh=5.0)` — returns a permutation ordering neurons by
          first supra-threshold crossing.
        - `scipy.stats.spearmanr`, `np.argmax`.

        **Your job.** Build the fixed window `win = C_z[:, ENTRY:ENTRY+WIN_LEN]`. Define a
        **sequenceness** score = the *absolute Spearman correlation* between a neuron's **row
        position** and its **first-crossing time** (`np.argmax(win > 5.0, axis=1)`). Compute it
        for the **unsorted** window and for the window **after `sequence_sort`**. Fill in
        `seq_unsorted` and `seq_sorted`, then run the self-check.
        """
    )
    return


@app.cell
def _(C_z, ENTRY, WIN_LEN, np, nu):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    from scipy.stats import spearmanr as _spearmanr

    _thr = 5.0
    _win = C_z[:, ENTRY:ENTRY + WIN_LEN]

    def _sequenceness(_raster):
        _first = np.argmax(_raster > _thr, axis=1)
        _r, _ = _spearmanr(np.arange(_raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))

    _order = nu.sequence_sort(_win, thresh=_thr)

    seq_unsorted = _sequenceness(_win)
    seq_sorted = _sequenceness(_win[_order])
    # ---------------------------------------------------------------------------------------------
    return seq_sorted, seq_unsorted


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            from scipy.stats import spearmanr
            thr = 5.0
            win = C_z[:, ENTRY:ENTRY + WIN_LEN]

            def sequenceness(raster):
                first = np.argmax(raster > thr, axis=1)      # first activation frame per neuron
                r, _ = spearmanr(np.arange(raster.shape[0]), first)
                return abs(r)

            order = nu.sequence_sort(win, thresh=thr)         # order by first crossing
            seq_unsorted = sequenceness(win)                  # ~0.05  (no structure in CNMF order)
            seq_sorted   = sequenceness(win[order])           # ~0.79  (a clean diagonal)
            ```

            **What you should find:** the unsorted window has **near-zero** sequenceness
            (~0.05 — CNMF hands you sources in an arbitrary order), while after `sequence_sort` it
            jumps to **~0.79**. The jump is the whole point: the sort didn't *create* structure, it
            **revealed** a temporal sequence that was already latent in the population. (The sorted
            score is ~0.79 rather than a perfect 1.0 because ~144 of the 202 neurons never cross the
            threshold in this window; among the ~58 that do fire, the ordering is essentially
            perfect.)
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, seq_sorted, seq_unsorted):
    # Self-check with a tolerance band pinned from the real data:
    #   seq_unsorted ~ 0.05,  seq_sorted ~ 0.7938.
    # Pass = sorting lands in [0.65, 0.95] AND clears the unsorted baseline by a wide margin.
    _in_band = 0.65 <= seq_sorted <= 0.95
    _gain = seq_sorted - seq_unsorted > 0.4
    _ok = _in_band and _gain
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"✅ sorted sequenceness = {seq_sorted:.3f} — in the expected band [0.65, 0.95]"
           if _in_band else
           f"❌ sorted sequenceness = {seq_sorted:.3f} — outside [0.65, 0.95]; check window/threshold")
    _m2 = (f"✅ sorting beats the unsorted baseline ({seq_unsorted:.3f}) by "
           f"{seq_sorted - seq_unsorted:.3f} — a real sequence was revealed"
           if _gain else
           f"❌ gain over baseline = {seq_sorted - seq_unsorted:.3f} is too small — did you sort the raster?")
    _head = "PASS — the sort reveals a neural sequence" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band pinned from the real recording:
        sorted ≈ 0.79, unsorted ≈ 0.05. The score measures |Spearman(row, first-activation)| — how
        diagonal the raster is.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## The twin, closed

        You ran **NB04's move on the brain.** PCA took a redundant behavior matrix and split it
        into a few nameable components; **CNMF took an optical mixture — overlapping light in the
        same pixels — and split it into 202 nameable sources**, each with a footprint (`A`, the
        *where*) and a trace (`C`, the *when*). Decompose a mixture into its sources: the same
        operation, once on the body, once on the tissue. And then, by sorting those sources by
        *when* they fire, a shapeless raster resolved into a **sequence** — structure that was
        there all along, waiting for the right ordering.

        **Next (NB12): what do these sources *code*?** We'll hand a demixed population a **behavior
        variable** — the animal's position in space — and ask which neurons are **tuned** to it:
        place cells and grid cells, the tuning-curve twin of the behavioral detectors you built in
        Week 1.
        """
    )
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "plotly>=5.20",
#     "h5py>=3.9",
#     "gdown>=5.0",
#     "openpyxl>=3.1",
#     "imageio>=2.34",
#     "imageio-ffmpeg>=0.4.9",
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
        # NB10 · Extracting a Calcium Trace

        > **WEEK 2 — THE NEURAL TWIN**
        >
        > Week 1 you read *behavior* out of pose. This week you run the **same computational moves**
        > on the **brain**. Today's move mirrors **NB02 — The Body's-Eye View**.
        >
        > In NB02 you took a raw, high-dimensional signal (three mouse skeletons, ~11,700 raw numbers
        > per event) and, by **choosing a point of view**, collapsed it into a handful of interpretable
        > **features**. Choosing the body frame *was* the analysis: it decided what each number would
        > mean.
        >
        > A miniscope movie is the imaging version of that same raw signal — **250,000 pixels per frame,
        > over time**. Buried in it is a much simpler thing you actually want: **one cell's calcium
        > trace**, a single number per frame that rises when the neuron fires. Getting there is the exact
        > twin of NB02: strip the nuisance (here the static background, not the arena pose), then
        > **choose a region of interest**. *Choosing an ROI is choosing a feature.* Where you put the box
        > decides what the 1-D trace means.

        The rig today: a head-mounted **miniscope** watching GCaMP-labeled striatal neurons in a moving
        mouse (eLife article e28728). Raw, it is a shimmering 500×500 movie. By the end you will have
        turned it into a clean trace of a single cell lighting up.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. The raw signal — pixels over time

        We stream the striatum movie straight from eLife and **subsample** it: keep every 100th frame
        (`step=100`), so ~9,000 raw frames become **90**. That is enough to see the structure and keeps
        a bare cloud kernel honest — never load the whole movie into a Python loop.

        This is the high-dimensional input, the twin of NB02's raw skeletons: each frame is a
        **500 × 500 = 250,000-number** snapshot, and there are 90 of them. Scrub the slider and watch —
        by eye, individual cells are almost impossible to pick out of the fixed glow. That glow is the
        problem we solve next.
        """
    )
    return


@app.cell
def _(nu):
    # One sequential ffmpeg decode, subsampled to 90 frames (~7 s, cached video). This is the single
    # heavy beat of the notebook; everything downstream is cheap numpy on the 90x500x500 array.
    striatum_path = nu.fetch_url(nu.STRIATUM_URL, nu.STRIATUM_NAME)
    frames = nu.read_video(striatum_path, step=100)          # (90, 500, 500) float32
    _bs = nu.background_subtract(frames)                       # median bg removed + per-pixel z-score
    fg = _bs["fg"]                                             # (90, 500, 500) z-scored foreground
    bg = _bs["bg"]                                             # (500, 500) median background
    F = int(frames.shape[0])
    return F, bg, fg, frames


@app.cell
def _(F, mo):
    movie_t = mo.ui.slider(0, F - 1, value=0, step=1,
                           label="frame (subsampled movie, 0–89)", debounce=True, full_width=True)
    return (movie_t,)


@app.cell
def _(frames, mo, movie_t, nu):
    _raw = nu.image_fig(frames[movie_t.value], title=f"RAW miniscope frame {movie_t.value}",
                        colorscale="gray", colorbar_title="intensity", height=470)
    mo.vstack([movie_t, _raw])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Strip the nuisance — background subtraction + z-score

        In NB02 the nuisance was **arena pose**: where in the cage, which way facing. Subtracting it
        left only social geometry. Here the nuisance is the **static background** — the fixed pattern of
        bright and dark that every frame shares (uneven illumination, out-of-focus tissue, the lens).
        It carries no signal about *when a neuron fires*, so we remove it, exactly as `neural_utils`
        does it:

        $$
        \text{bg} = \operatorname{median}_t(\text{frames}), \qquad
        \text{fg} = \text{frames} - \text{bg}, \qquad
        \text{fg} \leftarrow \frac{\text{fg} - \mu_{\text{px}}}{\sigma_{\text{px}}}
        $$

        The **median over time** is the trick: a pixel sitting on inactive tissue looks the same in
        almost every frame, so its median *is* the background; a pixel that occasionally flares when a
        cell fires spends most frames dark, so the flare survives the subtraction. The per-pixel
        z-score then puts every pixel on the same footing (units of its own standard deviation), so a
        dim active cell isn't drowned out by a bright dead patch.

        Left is the background we removed. Right is one **foreground** frame on a symmetric ±3σ scale —
        the flat glow is gone and transient blobs (firing cells) pop out. Same slider drives both.
        """
    )
    return


@app.cell
def _(bg, fg, mo, movie_t, nu):
    _bg_fig = nu.image_fig(bg, title="background (median over time)", colorscale="gray",
                           colorbar_title="intensity", height=470)
    _fg_fig = nu.image_fig(fg[movie_t.value], title=f"foreground frame {movie_t.value}  (z-scored)",
                           colorscale="RdBu", zmin=-3, zmax=3, colorbar_title="σ", height=470)
    mo.vstack([movie_t, mo.hstack([_bg_fig, _fg_fig], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Where are the cells? — the max-projection

        A single foreground frame catches whichever cells happen to be firing *at that instant*. To see
        **every** cell that was ever active, collapse the whole movie to one image: take the
        **maximum of |foreground| over time** at each pixel.

        $$\text{active}(y,x) = \max_t \big|\text{fg}(t,y,x)\big|$$

        A pixel that ever flared shows up bright; a pixel that stayed at baseline stays dark. This is
        our **map of candidate cells** — the thing you will point an ROI at in the next section. It is
        the imaging analogue of NB02's feature table: a compact summary that tells you *where the
        signal lives* before you commit to reading any one number.
        """
    )
    return


@app.cell
def _(fg, np):
    maxproj = np.abs(fg).max(axis=0)                           # (500, 500) map of active pixels
    return (maxproj,)


@app.cell
def _(maxproj, nu):
    nu.image_fig(maxproj, title="max |foreground| over time — bright spots are active cells",
                 colorscale="Inferno", colorbar_title="peak σ", height=560)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Choose an ROI = choose a feature

        This is the beat that mirrors NB02 most directly. In NB02, choosing the body frame *defined*
        what the 19 features meant. Here, **choosing where to put a small box defines what your 1-D
        trace means.** Drop the box on a bright spot and the trace reads *that cell's* calcium; drop it
        on empty tissue and the trace reads *noise*.

        The ROI trace is nothing more than the box-average of the foreground at every frame:

        $$\text{trace}(t) = \operatorname{mean}_{y,x \in \text{ROI}} \text{fg}(t, y, x)$$

        Slide the ROI center over the max-projection (left; the red box shows where you are). The 1-D
        trace it extracts appears on the right, **live**. Aim the box at one of the bright blobs and
        the trace grows sharp calcium transients; slide onto the dark background and it flattens into a
        noisy line near zero. Remember image indexing is `[y, x]`, so `cx` moves the box horizontally
        and `cy` vertically.
        """
    )
    return


@app.cell
def _():
    # The ROI reader used by both the interactive panel and the exercise. A 2*r-wide box centered on
    # (cx, cy); image indexing is [y, x], so cy selects rows and cx selects columns.
    def roi_trace(fg_stack, cx, cy, r=10):
        y0, y1 = int(cy - r), int(cy + r)
        x0, x1 = int(cx - r), int(cx + r)
        return fg_stack[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return (roi_trace,)


@app.cell
def _(mo):
    roi_cx = mo.ui.slider(15, 485, value=243, step=1, label="ROI center x (cx)",
                          debounce=True, full_width=True)
    roi_cy = mo.ui.slider(15, 485, value=349, step=1, label="ROI center y (cy)",
                          debounce=True, full_width=True)
    return roi_cx, roi_cy


@app.cell
def _(maxproj, mo, nu, roi_cx, roi_cy, roi_trace, fg):
    _cx, _cy, _r = roi_cx.value, roi_cy.value, 10
    _img = nu.image_fig(maxproj, title="active-cell map — drag the ROI box",
                        colorscale="Inferno", colorbar_title="peak σ", height=460)
    _img.add_shape(type="rect", x0=_cx - _r, y0=_cy - _r, x1=_cx + _r, y1=_cy + _r,
                   line=dict(color="#00e5ff", width=3))
    _tr = roi_trace(fg, _cx, _cy, _r)
    _trace = nu.trace_fig(None, _tr, title=f"ROI trace at ({_cx}, {_cy}) — variance = {_tr.var():.3f}",
                          xlabel="frame", ylabel="mean foreground (σ)", height=460)
    mo.vstack([mo.hstack([roi_cx, roi_cy]), mo.hstack([_img, _trace], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The dataset & the paper — and where the analogy stops": mo.md(
            r"""
            **Provenance.** The movie is `elife-28728-video1.mp4`, distributed with the open-access
            eLife article **e28728** (DOI [10.7554/eLife.28728](https://doi.org/10.7554/eLife.28728)):
            a head-mounted **miniature microscope** (miniscope) recording GCaMP calcium fluorescence
            from **striatal** neurons in a freely moving mouse. We re-host nothing — the notebook
            streams it from eLife's server at runtime and caches it locally.

            **The method.** Median-background subtraction + per-pixel z-score + hand-placed ROI is the
            *pedagogical* skeleton of calcium extraction. Production pipelines (CNMF / CNMF-E,
            Pnevmatikakis et al. 2016; Zhou et al. 2018) replace the hand-drawn box with a learned,
            data-driven **spatial footprint** per neuron and demix overlapping cells — which is exactly
            **NB11**, the next notebook.

            **Where the analogy stops.** A rectangular ROI assumes one cell sits neatly inside the box
            and nothing else does. Real striatal fields are dense: footprints overlap, neuropil
            contaminates the average, and a single box will happily blend two cells or half a cell plus
            background. The clean trace you extract here is a *best case*. When cells crowd, hand-ROIs
            break and you need the demixing NB11 introduces — the same way NB02's 19 hand-built
            features eventually gave way to a learned representation.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Exercise — a cell fluctuates, background is flat

        **Hypothesis banner.** *A real cell's ROI trace carries transients (high variance); a
        background patch of the same size does not (low variance). If choosing an ROI really is choosing
        a feature, the cell-ROI variance should tower over the background-ROI variance.*

        **Toolbox.**

        - `roi_trace(fg, cx, cy, r=10)` — returns the `(90,)` box-average trace at center `(cx, cy)`.
        - `fg` — the `(90, 500, 500)` z-scored foreground you built in Section 2.
        - `maxproj` — the active-cell map; use it (or the slider above) to find a bright blob.
        - `numpy`: `trace.var()`.

        **Your job.** Pick **two** ROI centers: one on a bright cell (the slider defaults `(243, 349)`
        and `(154, 338)` both land on real cells), and one on dark background (e.g. `(30, 30)`).
        Extract both traces and fill in their variances below, then run the self-check.
        """
    )
    return


@app.cell
def _(fg, roi_trace):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    # A bright cell (use the ROI slider above to hunt for a blob, then read off cx, cy):
    _cell_trace = roi_trace(fg, cx=243, cy=349, r=10)
    cell_var = float(_cell_trace.var())

    # A background patch of the SAME size, on dark tissue:
    _bg_trace = roi_trace(fg, cx=30, cy=30, r=10)
    bg_var = float(_bg_trace.var())
    # ---------------------------------------------------------------------------------------------
    return bg_var, cell_var


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            cell_trace = roi_trace(fg, cx=243, cy=349, r=10)   # a bright blob on the max-projection
            cell_var   = float(cell_trace.var())               # ~0.918

            bg_trace   = roi_trace(fg, cx=30, cy=30, r=10)     # dark corner, no cell
            bg_var     = float(bg_trace.var())                 # ~0.149
            ```

            **What you should find:** the cell ROI has variance **≈ 0.92** and the background ROI
            **≈ 0.15** — a **~6×** gap. The cell trace visibly rises and falls (calcium transients);
            the background trace hovers near zero as flat noise. That gap *is* the payoff of choosing
            the right ROI: the same arithmetic (box-average of foreground) gives you signal or garbage
            depending only on **where** you point it — the imaging echo of NB02, where the same
            transform gave a useful feature only because you chose the right frame. (The second default,
            `(154, 338)`, lands on another real cell, variance ≈ 0.916 — try it too.)
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(bg_var, cell_var, mo):
    # Self-check with a tolerance band pinned from the real 90-frame movie:
    #   cell(243,349) var = 0.9181, cell(154,338) = 0.9155, background(30,30) = 0.1493  (ratio ~6.1).
    # Grade the HONEST claim: cell variance is well above background (ratio > 2.5) AND the background
    # patch really is flat (bg_var < 0.4). We do not grade the exact number, only the separation.
    _ratio = cell_var / bg_var if bg_var > 0 else float("inf")
    _p_cell = cell_var > 0.4
    _p_flat = bg_var < 0.4
    _p_sep = _ratio > 2.5
    _ok = _p_cell and _p_flat and _p_sep
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m_cell = (f"✅ cell ROI fluctuates (variance = {cell_var:.3f} > 0.4)" if _p_cell
               else f"❌ cell ROI variance = {cell_var:.3f} is low — is the box on a bright blob?")
    _m_flat = (f"✅ background ROI is flat (variance = {bg_var:.3f} < 0.4)" if _p_flat
               else f"❌ background ROI variance = {bg_var:.3f} is high — that patch has a cell in it")
    _m_sep = (f"✅ cell / background variance ratio = {_ratio:.1f}× (> 2.5) — the ROI choice made the feature"
              if _p_sep else
              f"❌ ratio = {_ratio:.1f}× is too small — the two ROIs are not clearly different")
    _head = "PASS — the cell ROI towers over background" if _ok else "Not yet — fix the flagged line"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m_cell}<br>{_m_flat}<br>{_m_sep}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band (pinned from the real movie):
        cell_var &gt; 0.4, bg_var &lt; 0.4, ratio &gt; 2.5. Pinned truth: cell ≈ 0.918, background ≈ 0.149,
        ratio ≈ 6.1×.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## The twin, closed — and what breaks

        **The move, both times.** *Raw high-D signal → strip the nuisance → choose a view → read one
        interpretable feature.* NB02 stripped arena pose and chose the body frame to read social
        geometry. NB10 stripped the static background and chose an ROI to read a calcium trace.
        **Choosing the ROI was choosing the feature** — the box's location decided what the number
        meant, exactly as the body frame did.

        **How it breaks.** A rectangular ROI is a blunt instrument. It assumes one cell fits the box
        and nothing else contaminates the average; in dense striatal tissue that assumption fails —
        overlapping cells blur together, neuropil leaks in, and a slightly-off box reads a mixture. The
        variance gap you measured is a *best case* on well-separated cells.

        **Next (NB11): let the data draw the ROIs.** Instead of a hand-placed box, CNMF-E learns a
        **spatial footprint** for every neuron and **demixes** overlapping cells automatically —
        turning today's one hand-chosen trace into a whole population at once. Same computational move
        (pixels → per-cell feature), now learned instead of hand-drawn — the imaging twin of trading
        NB02's hand-built features for a representation the data chooses.
        """
    )
    return


if __name__ == "__main__":
    app.run()

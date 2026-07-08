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

        **Week 2 — working with real neural recordings.**

        In Week 1 you measured behavior from pose keypoints: you started from a large raw signal
        (three mouse skeletons) and reduced it to a small set of interpretable numbers. This week you
        work with recordings of neural activity, and the overall approach is the same: start from a
        large raw signal and reduce it to a few numbers you can interpret. This notebook does that for
        an imaging movie of neurons in the brain.

        ## Why this matters

        To study how the brain produces social behavior, we need to measure what neurons are doing while
        an animal behaves. One common way to do this is **calcium imaging**. Before we can relate neural
        activity to behavior (later in Week 2), we first have to turn a raw imaging movie into a usable
        measurement of one neuron's activity over time. That extraction step is the subject of today's
        notebook.

        ## Definitions (read these first)

        - **Neuron firing.** When a neuron fires, calcium ions flow into the cell. So the amount of
          calcium inside a cell rises briefly each time the cell is active, then falls back down.
        - **GCaMP.** A protein sensor that scientists express in neurons. It glows brighter when it
          binds calcium. Because calcium tracks firing, the brightness of a GCaMP-labeled neuron is a
          **proxy for its activity**: brighter means more recently active.
        - **Miniscope.** A small head-mounted microscope that records this glowing tissue as a video
          while the mouse moves freely.
        - **Calcium trace.** One number per movie frame that describes how bright a single cell is over
          time. This is what we want to extract. Its rises are called calcium **transients**.
        - **Region of interest (ROI).** A small patch of pixels we select in the image, here a box
          placed over one cell. Averaging the pixels inside the box at each frame produces the trace.

        ## Today's data and goal

        We use a real miniscope movie of **striatal** neurons (from the open-access eLife article
        e28728). The raw movie is a 500 x 500 video: 250,000 pixel values per frame, changing over time.
        Our goal is to reduce it to a single clean calcium trace for one cell. The steps are: (1) view
        the raw movie, (2) remove the static background, (3) find where the active cells are, and
        (4) place an ROI to read out one cell's trace.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. The raw signal: pixels over time

        **Why.** Before processing anything, look at the input. The raw miniscope movie is the
        high-dimensional signal we will reduce, just as the raw skeletons were in Week 1.

        **Method.** We stream the striatum movie and **subsample** it: we keep every 100th frame
        (`step=100`), so about 9,000 raw frames become **90**. Subsampling keeps enough frames to see
        the structure while keeping memory and compute small. `nu.read_video(path, step=100)` reads the
        video file and returns a `(90, 500, 500)` array of frames (its input is the file path; its
        output is the stack of grayscale frames).

        Each frame is a **500 x 500 = 250,000-value** image, and there are 90 of them. Move the slider
        to scrub through the movie. Note that by eye it is hard to pick out individual cells: a fixed
        bright glow covers the whole field. Removing that glow is the next step.
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
        ## 2. Remove the static background (background subtraction + z-score)

        **Why.** Most of what you see in the raw movie is unchanging: uneven illumination, out-of-focus
        tissue, and the lens. This static pattern is the same in every frame, so it carries no
        information about *when a neuron fires*. Removing it leaves only the part of the signal that
        changes over time, which is the part we care about.

        **Definitions.**

        - **Background subtraction** means estimating the static part of the image and subtracting it,
          so that only the changing part remains.
        - **z-score** means rescaling each pixel to units of its own standard deviation, so pixels with
          different baseline brightness become directly comparable.

        **Method.** `nu.background_subtract(frames)` performs three operations:

        $$
        \text{bg} = \operatorname{median}_t(\text{frames}), \qquad
        \text{fg} = \text{frames} - \text{bg}, \qquad
        \text{fg} \leftarrow \frac{\text{fg} - \mu_{\text{px}}}{\sigma_{\text{px}}}
        $$

        Its input is the `(90, 500, 500)` frame stack; its outputs are the `(500, 500)` background image
        `bg` and the `(90, 500, 500)` z-scored foreground `fg`. Taking the **median over time** at each
        pixel is the key idea: a pixel on inactive tissue looks the same in almost every frame, so its
        median value *is* its background; a pixel that occasionally flares when a cell fires is dark in
        most frames, so the flare survives the subtraction. The per-pixel z-score then puts every pixel
        on the same scale, so a dim active cell is not overwhelmed by a bright inactive patch.

        Below, the left panel is the background that was removed. The right panel is one foreground frame
        on a symmetric ±3σ scale: the flat glow is gone, and short-lived bright spots (firing cells)
        stand out. The same slider controls both.
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
        ## 3. Where are the cells? The maximum projection

        **Why.** A single foreground frame only shows the cells that happen to be firing at that one
        instant. To decide where to place an ROI, we want a single image that shows **every** location
        that was active at any point in the movie.

        **Definition.** A **maximum projection** collapses a movie to one image by taking, at each
        pixel, the largest value that pixel ever reached. Here we use the maximum of the absolute
        foreground over time:

        $$\text{active}(y,x) = \max_t \big|\text{fg}(t,y,x)\big|$$

        **Method.** `np.abs(fg).max(axis=0)` takes the absolute value of the foreground (so both bright
        and dark deviations count) and then takes the maximum across the time axis. Its input is the
        `(90, 500, 500)` foreground; its output is a single `(500, 500)` image. A pixel that ever flared
        appears bright; a pixel that stayed at baseline stays dark. This image is our **map of candidate
        cells**, and it is where we will aim an ROI in the next section.
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
        ## 4. Place an ROI to read out one cell

        **Why.** We now have a map of where cells are, but we still want a single number per frame for
        one specific cell. Choosing where to place the ROI box is what determines which cell (or which
        piece of background) the trace describes.

        **Method.** The ROI trace is the average of the foreground inside the box at every frame:

        $$\text{trace}(t) = \operatorname{mean}_{y,x \in \text{ROI}} \text{fg}(t, y, x)$$

        The function `roi_trace(fg, cx, cy, r=10)` (defined in the next cell) does exactly this. Its
        inputs are the foreground stack and the box center `(cx, cy)` with half-width `r`; its output is
        a `(90,)` trace, one value per frame.

        Use the two sliders to move the ROI center over the active-cell map (left; the box marks the
        current position). The trace it extracts appears on the right and updates live. When the box
        sits on a bright spot, the trace shows sharp calcium transients; when it sits on dark
        background, the trace stays near zero and looks like flat noise. Remember that image indexing is
        `[y, x]`, so `cx` moves the box horizontally and `cy` moves it vertically.
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
        "Dataset, method, and limits": mo.md(
            r"""
            **Provenance.** The movie is `elife-28728-video1.mp4`, distributed with the open-access
            eLife article **e28728** (DOI [10.7554/eLife.28728](https://doi.org/10.7554/eLife.28728)):
            a head-mounted miniature microscope recording GCaMP calcium fluorescence from **striatal**
            neurons in a freely moving mouse. The notebook streams it from eLife's server at runtime and
            caches it locally.

            **What this method is.** Median-background subtraction, per-pixel z-score, and a hand-placed
            ROI form a simple, teachable version of calcium extraction. Production pipelines (CNMF /
            CNMF-E; Pnevmatikakis et al. 2016; Zhou et al. 2018) replace the hand-drawn box with a
            learned, data-driven **spatial footprint** for each neuron and separate overlapping cells.
            That is the subject of the next notebook, NB11.

            **Limits.** A rectangular ROI assumes that one cell sits inside the box and nothing else
            does. Real striatal tissue is dense: cells overlap, surrounding tissue (neuropil) leaks into
            the average, and a single box can blend two cells or half a cell plus background. The clean
            trace extracted here is a best case. When cells crowd together, hand-placed ROIs are no
            longer adequate and the demixing method in NB11 is needed.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Exercise: a cell fluctuates, background does not

        **What we expect.** A real cell's ROI trace should carry calcium transients, so it varies a lot
        over time (high variance). A background patch of the same size should stay near zero, so it
        varies little (low variance). If placing the ROI is what defines the measurement, then the
        cell-ROI variance should be clearly larger than the background-ROI variance.

        **Tools you will use.**

        - `roi_trace(fg, cx, cy, r=10)` returns the `(90,)` box-average trace at center `(cx, cy)`.
        - `fg` is the `(90, 500, 500)` z-scored foreground from Section 2.
        - `trace.var()` returns one number: how much the trace fluctuates over the 90 frames.
        - Use the active-cell map and the slider above to find a bright blob and a dark patch.

        **What to do.** In the next cell, edit the two marked lines so that `cell_var` measures a bright
        cell and `bg_var` measures dark background. The cell line is filled in for you as a worked
        example at `(243, 349)`; complete the background line with a dark patch such as `(30, 30)`. Then
        run the self-check below.

        **What you should see.** The cell trace visibly rises and falls, giving a variance near **0.9**.
        The background trace stays close to zero, giving a variance near **0.15**, roughly a **6x**
        difference. The self-check passes when the cell variance is well above the background variance.
        """
    )
    return


@app.cell
def _(fg, roi_trace):
    # -------------------------------------------------------------- YOUR CODE (edit the 2 marked lines)
    # roi_trace(fg, cx, cy, r=10) -> (90,) trace: the average foreground brightness inside a
    #     20x20-pixel box centered on pixel (cx, cy), one value per movie frame.
    # float(trace.var())          -> one number: how much that trace fluctuates over the 90 frames.

    # LINE 1 (worked example): a BRIGHT CELL at (243, 349), the slider's default position.
    _cell_trace = roi_trace(fg, cx=243, cy=349, r=10)
    cell_var = float(_cell_trace.var())

    # LINE 2 (you complete): a DARK BACKGROUND patch of the SAME size. Replace the cx and cy below with
    #     a spot on empty tissue. The corner (cx=30, cy=30) works; the interactive map above helps you
    #     confirm it is dark.
    _bg_trace = roi_trace(fg, cx=30, cy=30, r=10)
    bg_var = float(_bg_trace.var())
    # ------------------------------------------------------------------------------------------------
    return bg_var, cell_var


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            cell_trace = roi_trace(fg, cx=243, cy=349, r=10)   # a bright blob on the max-projection
            cell_var   = float(cell_trace.var())               # ~0.918

            bg_trace   = roi_trace(fg, cx=30, cy=30, r=10)      # dark corner, no cell
            bg_var     = float(bg_trace.var())                 # ~0.149
            ```

            **What you should find.** The cell ROI has variance about **0.92** and the background ROI
            about **0.15**, a difference of roughly **6x**. The cell trace clearly rises and falls
            (calcium transients); the background trace stays near zero. This shows that the same
            operation, averaging the foreground inside a box, gives useful signal or flat noise
            depending only on **where** the box is placed. Placing the ROI is what defines the
            measurement. (A second cell sits at `(154, 338)`, with variance about 0.916; try it as
            well.)
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(bg_var, cell_var, mo):
    # Self-check with a tolerance band pinned from the real 90-frame movie:
    #   cell(243,349) var = 0.9181, cell(154,338) = 0.9155, background(30,30) = 0.1493  (ratio ~6.1).
    # Grade the claim: cell variance is well above background (ratio > 2.5) AND the background
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
    _head = "PASS — the cell ROI variance is well above background" if _ok else "Not yet — fix the flagged line"
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
        ## Summary and what comes next

        **What we did.** We started from a raw imaging movie (250,000 pixel values per frame over time)
        and reduced it to one interpretable calcium trace. The steps were: remove the static background,
        build a map of active cells with a maximum projection, and place an ROI to average the
        foreground inside a box. This mirrors the Week 1 workflow of reducing a large raw signal to a
        few interpretable numbers; here the choice that defines the measurement is where the ROI box is
        placed.

        **Limits of this approach.** A rectangular ROI assumes one cell fits neatly in the box and
        nothing else contaminates the average. In dense striatal tissue that assumption often fails:
        overlapping cells blur together, surrounding tissue leaks into the average, and a slightly
        misplaced box reads a mixture. The variance separation measured here is a best case on
        well-separated cells.

        **Next (NB11): let the data define the ROIs.** Instead of a hand-placed box, CNMF-E learns a
        spatial footprint for every neuron and separates overlapping cells automatically, extracting a
        whole population of traces at once rather than one hand-chosen trace. It is the same operation
        (pixels to a per-cell measurement), now learned from the data rather than drawn by hand.
        """
    )
    return


if __name__ == "__main__":
    app.run()

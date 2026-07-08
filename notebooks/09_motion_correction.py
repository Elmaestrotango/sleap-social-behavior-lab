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
        # NB09 · Motion Correction

        Welcome to Week 2. Your longer-term goal as a behavioral neuroscientist is to study how the
        medial prefrontal cortex (mPFC) contributes to social behavior. In Week 1 you built an
        objective readout of *what a mouse does*, from tracked body points. In Week 2 you add the
        second half of that question: *what the brain is doing while the mouse does it.* To get there,
        you first have to be able to read the brain's signal cleanly. That is what this notebook is
        about.

        ## Why motion correction is needed

        The recordings in Week 2 come from a **miniature microscope** (a "miniscope"): a small camera
        mounted on the animal's head that films a patch of brain tissue while the mouse moves freely. A
        protein reporter makes a neuron glow brighter when it becomes active, so the recording is a
        movie in which bright spots switch on and off as neurons fire. To measure one neuron's activity
        over time, you look at the pixels where that neuron sits and track how their brightness changes
        frame by frame.

        This only works if a given pixel keeps pointing at the *same* piece of tissue. It does not. The
        animal walks, grooms, and turns, and the brain shifts slightly under the lens. The camera's view
        (the **field of view**, or FOV) drifts. When it drifts, pixel `(80, 40)` in frame 500 and pixel
        `(80, 40)` in frame 501 are no longer the same neuron. Read the brightness at a fixed pixel and
        you get a mixture of "the neuron fired" and "the neuron slid away," with no way to tell them
        apart. A neuron's trace is not meaningful until the frames are aligned.

        **Motion correction** (also called **registration**) fixes this. It estimates how much each
        frame has moved relative to a reference image, then shifts the frame back so that every pixel
        lines up with the same tissue across the whole recording.

        This is the same problem you already solved for behavior. In Week 1, a pose track was
        meaningless until you had stabilized it: fixed the identity of each mouse and made sure "mouse 2
        at frame 500" was the same animal as "mouse 2 at frame 501." Aligning a signal across time
        before you read it is a general step, and imaging needs it just as much as pose tracking does.

        ## What this notebook does

        You will look at one real miniscope movie that has been processed **three ways** and laid out
        side by side, `raw | rigid | pw-rigid`, so you can compare an uncorrected recording against two
        registration methods. You will read the motion two ways (by eye, and with a single number), and
        the closing exercise asks the concrete question: *does registration actually reduce
        frame-to-frame motion, and does the more flexible piecewise-rigid method beat plain rigid?*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. One movie, three registrations, side by side

        The file `mmc3.mp4` is a demonstration movie in which three versions of the *same* miniscope
        recording are placed left-to-right in a single frame. Comparing them side by side is the
        clearest way to see what each method does.

        **Definitions of the three panels:**

        - **left third — raw:** the recording straight off the sensor, with no correction applied.
        - **middle third — rigid registration:** the whole frame is shifted by one translation (one
          left/right and up/down offset) per frame, chosen to best match a reference template. This
          corrects *global* drift, where the entire FOV slides together. It cannot fix motion that
          differs across the frame.
        - **right third — piecewise-rigid registration:** the frame is divided into patches, each patch
          is shifted on its own, and the patch shifts are blended back together smoothly. Because
          different patches can move by different amounts, this corrects *non-uniform* motion, where one
          part of the tissue stretches or slides differently from another.

        **What the code does.** We split the movie's width into thirds and crop the top and bottom so
        the three panels line up. The frame slider picks which time point to show. Drag it and watch a
        single bright cell: in **raw** it wanders, in **rigid** it steadies, and in **pw-rigid** it holds
        most still. (The movie is subsampled in time so it loads quickly on a modest machine.)
        """
    )
    return


@app.cell
def _(CACHE, np, nu):
    # Load the side-by-side motion-correction movie (cached; downloaded once via gdown).
    # Subsample step=3 -> ~185 frames so a headless kernel loads it fast and keeps memory modest.
    _path = nu.fetch_gdrive(nu.MOCO_GDRIVE_ID, nu.MOCO_NAME, CACHE and nu.find_root())
    mov = nu.read_video(_path, step=3, gray=True)          # (F, H, W) grayscale float32
    raw, rigid, pwr = nu.split_thirds(mov)                  # each (F, H', w)
    F, H, W = raw.shape
    # global gray range for a shared colorscale across the three panels
    VMIN = float(np.percentile(mov, 1))
    VMAX = float(np.percentile(mov, 99))
    return F, H, VMAX, VMIN, W, pwr, raw, rigid


@app.cell
def _(F, mo):
    frame_ctrl = mo.ui.slider(0, F - 1, value=F // 2, step=1,
                              label="frame", debounce=True, full_width=True)
    return (frame_ctrl,)


@app.cell
def _(VMAX, VMIN, frame_ctrl, mo, pwr, raw, rigid):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as _pgo
    _t = frame_ctrl.value
    _panels = [("raw", raw), ("rigid", rigid), ("pw-rigid", pwr)]
    _fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.02,
                         subplot_titles=[n for n, _ in _panels])
    for _j, (_name, _p) in enumerate(_panels, start=1):
        _fig.add_trace(_pgo.Heatmap(z=_p[_t], colorscale="gray", zmin=VMIN, zmax=VMAX,
                                    showscale=(_j == 3), colorbar=dict(title="gray", len=0.9)),
                       row=1, col=_j)
        _ax = "" if _j == 1 else str(_j)
        _fig.update_yaxes(autorange="reversed", scaleanchor="x" + _ax, scaleratio=1,
                          visible=False, row=1, col=_j)
        _fig.update_xaxes(visible=False, row=1, col=_j)
    _fig.update_layout(template="plotly_white", height=340,
                       margin=dict(l=10, r=10, t=40, b=10),
                       title=f"frame {_t}  —  raw | rigid | pw-rigid")
    mo.vstack([frame_ctrl, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A single frame is a weak test of motion, because motion is a property of how the image changes
        *over time*, not of any one snapshot. Next we summarize the whole time axis in one picture.

        ---
        ## 2. Kymographs — turning motion into a picture

        **Definition.** A **kymograph** takes one horizontal line of the image (a single row of pixels,
        at height `y`) and stacks that line across every frame of the movie. The result is a 2-D image
        where the horizontal axis is **position along the row** and the vertical axis is **time**. Each
        row of the kymograph is what that one image line looked like at one moment.

        **How to read it.** A pixel feature that stays put over time traces a **straight vertical
        streak** down the kymograph. A feature that jitters left and right traces a **wiggly** streak.
        So "is the movie well registered?" becomes the simpler visual question "are the streaks
        straight?" This is the imaging counterpart of overlaying every frame's pose track and asking
        whether the line holds still, which is a sanity check you used in Week 1.

        **Controls.** Pick the row `y` (which image line to track) and a **time window** to highlight
        (the red band). Compare the three panels: raw streaks wander, rigid straightens them, and
        pw-rigid straightens them the most.
        """
    )
    return


@app.cell
def _(F, H, mo):
    row_ctrl = mo.ui.slider(0, H - 1, value=min(80, H - 1), step=1,
                            label="kymograph row y (which image line to track)",
                            debounce=True, full_width=True)
    win_ctrl = mo.ui.range_slider(0, F - 1, value=[int(0.55 * F), int(0.72 * F)], step=1,
                                  label="highlight time window [t0, t1] (frames)",
                                  debounce=True, full_width=True)
    return row_ctrl, win_ctrl


@app.cell
def _(mo, pwr, raw, rigid, row_ctrl, win_ctrl):
    from plotly.subplots import make_subplots as _mksub
    import plotly.graph_objects as _kgo
    _y = row_ctrl.value
    _t0, _t1 = win_ctrl.value
    _kymos = [("Raw", raw[:, _y, :]), ("Rigid", rigid[:, _y, :]), ("PW-Rigid", pwr[:, _y, :])]
    _fig = _mksub(rows=1, cols=3, horizontal_spacing=0.04,
                  subplot_titles=[n for n, _ in _kymos])
    for _j, (_name, _k) in enumerate(_kymos, start=1):
        _fig.add_trace(_kgo.Heatmap(z=_k, colorscale="gray", showscale=False), row=1, col=_j)
        _ax = "" if _j == 1 else str(_j)
        _fig.update_yaxes(autorange="reversed", title="time (frames)" if _j == 1 else None,
                          row=1, col=_j)
        _fig.update_xaxes(title="position (px)", row=1, col=_j)
        # red band marking the highlighted time window (spans the full subplot width)
        _fig.add_shape(type="rect", xref="x" + _ax + " domain", x0=0, x1=1,
                       yref="y" + _ax, y0=_t0, y1=_t1,
                       line=dict(color="red", width=1), fillcolor="red", opacity=0.12,
                       layer="above", row=1, col=_j)
    _fig.update_layout(template="plotly_white", height=460,
                       margin=dict(l=10, r=10, t=40, b=10),
                       title=f"kymographs at row y = {_y}   ·   window [{_t0}, {_t1}]")
    mo.vstack([row_ctrl, win_ctrl, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The kymograph turns "is it registered?" into "are the streaks straight?", but judging streaks by
        eye does not scale to thousands of movies, and it struggles to separate rigid from pw-rigid when
        both look reasonable. For that we need a single number.

        ---
        ## 3. The motion index — one number for how much jitter is left

        **Definition.** The **motion index** measures how much the picture changes from one frame to the
        next. The function `nu.motion_index(frames)` computes the mean absolute **frame-to-frame
        difference**,

        $$\text{MI} = \big\langle\,|f_{t+1} - f_{t}|\,\big\rangle,$$

        the average, over every pixel and every pair of neighboring frames, of how much a pixel's
        brightness changed. A perfectly still movie has a motion index near zero, because consecutive
        frames are nearly identical. A jittery movie has a large motion index, because every pixel keeps
        changing as the image slides around.

        **Purpose, inputs, outputs.**

        - `nu.motion_index(frames)` — *purpose:* score how much frame-to-frame motion a movie has.
          *Input:* one movie array of shape `(F, H, W)`. *Output:* a single number (the scalar motion
          index).
        - `nu.motion_index_trace(frames)` — *purpose:* show that same jitter over time instead of
          collapsing it to one number. *Input:* a movie `(F, H, W)`. *Output:* a 1-D array of length
          `F - 1`, one motion value per adjacent frame pair.

        If registration works, it should push the motion index **down**, and the more flexible method
        should push it lower still:

        $$\text{MI}(\text{raw}) \;>\; \text{MI}(\text{rigid}) \;>\; \text{MI}(\text{pw-rigid}).$$

        The per-frame trace below plots `nu.motion_index_trace` for all three versions. The pw-rigid
        line stays lowest across almost the whole recording. This is the same idea as a tracking-error
        trace in Week 1: a well-stabilized signal has a small frame-to-frame change.
        """
    )
    return


@app.cell
def _(go, mo, nu, pwr, raw, rigid):
    # Per-frame jitter trace for each registration (renders on load — this is a core beat).
    _tr_raw = nu.motion_index_trace(raw)
    _tr_rig = nu.motion_index_trace(rigid)
    _tr_pwr = nu.motion_index_trace(pwr)
    _fig = go.Figure()
    _fig.add_scatter(y=_tr_raw, mode="lines", name="raw", line=dict(color="#e45756", width=1))
    _fig.add_scatter(y=_tr_rig, mode="lines", name="rigid", line=dict(color="#4c78a8", width=1))
    _fig.add_scatter(y=_tr_pwr, mode="lines", name="pw-rigid", line=dict(color="#54a24b", width=1.4))
    _fig.update_layout(template="plotly_white", height=300,
                       margin=dict(l=10, r=10, t=40, b=10),
                       title="per-frame motion index (lower = more stable)",
                       xaxis_title="frame", yaxis_title="mean |Δ| to previous frame",
                       legend=dict(orientation="h", y=1.12))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Exercise — does registration reduce motion?

        **The question.** We want to confirm, with a number, that registration reduces frame-to-frame
        motion and that piecewise-rigid does better than plain rigid:
        $\text{MI}(\text{pw-rigid}) < \text{MI}(\text{rigid}) < \text{MI}(\text{raw})$.

        **What you already have.**

        - `nu.motion_index(frames)` — returns one number, the motion index of a movie of shape
          `(F, H, W)`; higher means more jitter.
        - `raw`, `rigid`, `pwr` — the three panels, already split out for you. No file loading is needed.

        **Your job.** Fill in the three lines in the next cell so that each variable holds the motion
        index of one panel. The first line, `mi_raw`, is done for you as a worked example: it calls
        `nu.motion_index` on the `raw` panel. Write the other two the same way, one for the `rigid`
        panel and one for the `pwr` panel.

        **What to expect.** After you run it, the self-check below reports the three numbers. You should
        see them fall in order, `mi_pwr < mi_rigid < mi_raw`, with pw-rigid removing roughly a tenth of
        the raw movie's motion. The self-check grades the *conclusion* (the ordering plus a tolerance
        band), not an exact decimal, because the absolute values change with how the movie is
        subsampled.
        """
    )
    return


@app.cell
def _(nu, pwr, raw, rigid):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    # Fill in the two blanks below. Each line should call nu.motion_index(...) on one panel.
    # PURPOSE: nu.motion_index(frames) takes a movie array (F, H, W) and returns ONE number,
    #          the average frame-to-frame change (bigger = more jitter).
    mi_raw = nu.motion_index(raw)      # worked example: motion index of the raw panel.
    mi_rigid = nu.motion_index(rigid)  # your turn: same call, but pass the `rigid` panel.
    mi_pwr = nu.motion_index(pwr)      # your turn: same call, but pass the `pwr` (pw-rigid) panel.
    # ---------------------------------------------------------------------------------------------
    return mi_pwr, mi_raw, mi_rigid


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            mi_raw   = nu.motion_index(raw)     # ~5.50 at step=3  (3.24 at full resolution)
            mi_rigid = nu.motion_index(rigid)   # ~5.10            (3.08)
            mi_pwr   = nu.motion_index(pwr)     # ~4.93            (2.97)
            ```

            **What you should find.** The three numbers fall in order:
            `mi_pwr < mi_rigid < mi_raw`. Rigid registration removes a few percent of the frame-to-frame
            motion in the raw movie; piecewise-rigid removes about 10 percent in total, because it also
            catches the *local* warping that a single whole-frame shift cannot. The absolute values
            depend on how the movie is subsampled (larger frame gaps make the apparent motion larger),
            which is why the **ordering** is the real result rather than any single decimal. This is the
            imaging-side version of the same point Week 1 made about pose tracks: alignment is not
            cosmetic. It measurably reduces the signal's jitter.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mi_pwr, mi_raw, mi_rigid, mo):
    # Honest self-check. The pinned conclusion (from the real mmc3 movie): registration MONOTONICALLY
    # reduces motion index (raw > rigid > pw-rigid) and pw-rigid removes ~10% of raw's motion.
    # Tolerance band on the reduction: 3%..25% (robust to the exact subsample step) so we grade the
    # honest finding, not an exact decimal.
    _order = (mi_pwr < mi_rigid < mi_raw)
    _red_pwr = (mi_raw - mi_pwr) / mi_raw if mi_raw else 0.0
    _band = 0.03 <= _red_pwr <= 0.25
    _ok = bool(_order and _band)
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = ("✅ ordering holds: MI(pw-rigid) &lt; MI(rigid) &lt; MI(raw)  "
           f"({mi_pwr:.3f} &lt; {mi_rigid:.3f} &lt; {mi_raw:.3f})") if _order else (
           f"❌ ordering broken: raw={mi_raw:.3f}, rigid={mi_rigid:.3f}, pw-rigid={mi_pwr:.3f} — "
           "registration should lower MI")
    _m2 = (f"✅ pw-rigid removes {100 * _red_pwr:.1f}% of raw motion — a real reduction, in band"
           if _band else
           f"⚠️ pw-rigid removes {100 * _red_pwr:.1f}% of raw motion — outside the 3–25% band; "
           "check you passed the right panels")
    _head = "PASS — registration measurably reduces motion" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Graded on the honest conclusion — the ordering plus a
        tolerance band (3–25% reduction), not an exact motion-index value, since that scales with
        subsampling.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reference: the algorithm and its limits": mo.md(
            r"""
            **Where these methods come from.** The rigid / piecewise-rigid split shown in this movie is
            the design of **NoRMCorre** (Pnevmatikakis & Giovannucci, *"NoRMCorre: An online algorithm
            for piecewise rigid motion correction of calcium imaging data,"* **J. Neurosci. Methods**
            291:83–94, 2017), the motion-correction stage of the widely used **CaImAn** pipeline
            (Giovannucci et al., *eLife* 2019). NoRMCorre estimates a reference template, aligns each
            frame to it by phase-correlation (rigid), then refines the result with **per-patch** shifts
            that are smoothly interpolated back together (piecewise-rigid) to handle non-uniform tissue
            motion. That is exactly the raw → rigid → pw-rigid progression in this movie.

            **The shared idea with Week 1.** Both are *registration*: estimate a transform (a single
            translation, or a field of local translations) that best maps one time sample onto a
            reference, then apply it so that later measurements are made in a stable frame. Pose
            stabilization and miniscope motion correction are two instances of the same alignment
            problem.

            **The limits of the analogy, and of the metric.** Pose tracking registers a handful of
            *labeled* keypoints whose identity is known. Motion correction registers *dense* pixel
            intensities with no labels, and has to build its own reference template, so it can be misled
            by real brightness changes (a neuron firing looks a little like the frame moving). Also, a
            lower motion index is necessary but not sufficient: overly aggressive piecewise warping can
            lower the motion index while smearing the real signal, so pipelines also check a correlation
            image and residual traces, not the motion index alone.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Summary and what comes next

        The point of this notebook is a single step: before you can read a signal, you must align it
        across time. Miniscope FOV drift is the imaging version of the pose-tracking drift you handled
        in Week 1. Rigid and piecewise-rigid registration are the tools that fix it, and the motion
        index is the number that confirms it worked (`pw-rigid < rigid < raw`).

        A registered movie is a stack of frames in which pixel `(y, x)` refers to the same neuron over
        time. That is the precondition for the rest of Week 2. **Next, in NB10, we pull a neuron's trace
        out of the pixels:** background-subtract a striatal miniscope movie, place a region of interest
        on a cell, and watch its calcium trace separate from the background. Registration is what makes
        "a neuron's trace" a well-defined thing to read; NB10 reads it.
        """
    )
    return


if __name__ == "__main__":
    app.run()

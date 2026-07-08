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
        # NB09 · Motion Correction — the neural twin of *registration*

        > **FROM: Circuit Team → TO: You (now on the imaging rig)**
        >
        > Welcome to Week 2. You spent Week 1 turning pixels into trustworthy pose. Now you point a
        > microscope at the brain — and hit the *same wall you hit on day one*. In **NB01** the very
        > first thing you had to do before reading a single behavior was **stabilize the tracks**:
        > fix identity, undo drift, make sure "mouse 2 at frame 500" is the same animal as "mouse 2 at
        > frame 501." Miniscope imaging has an identical disease. The animal moves, the tissue slides
        > under the lens, and **frame 500's pixel (80, 40) is no longer the same neuron as frame 501's
        > pixel (80, 40).** Before you can read *any* calcium signal, you must register the movie.
        >
        > **The deliverable:** a felt understanding that a "neuron's trace" is a *lie* until the frames
        > are aligned — and a number, the **motion index**, that says how well.
        > **It unblocks:** NB10 (pulling traces out of pixels) and everything after it.
        > **Today's lab-meeting question:** *does registration actually reduce frame-to-frame motion,
        > and does the fancier piecewise-rigid method beat plain rigid?*

        ## The twin, stated plainly

        | Week 1 — behavior (NB01) | Week 2 — brain (NB09) |
        |---|---|
        | pose tracks jitter / swap identity | imaging FOV drifts under the lens |
        | a "track" is meaningless until stabilized | a "neuron trace" is meaningless until registered |
        | fix identity → align across time | rigid / piecewise-rigid registration → align across time |

        **The shared computational move:** *before you can read a signal, you must align it across
        time.* Imaging drift is the pose-tracking drift problem wearing a lab coat. Registration is the
        imaging analog of stabilizing tracks.

        Below you'll look at one movie, **rendered three ways at once** — `raw | rigid | pw-rigid` —
        the way a motion-correction pipeline hands you its work so you can judge it by eye.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. One movie, three registrations, side by side

        The file `mmc3.mp4` is a demo output where three versions of the *same* miniscope movie are
        laid out left-to-right in one frame:

        - **left third — raw:** straight off the sensor, no correction.
        - **middle third — rigid:** the whole frame is shifted (one translation per frame) to best
          match a template. Corrects global drift; can't fix local warping.
        - **right third — pw-rigid:** *piecewise* rigid — the frame is split into patches, each shifted
          on its own, then blended. Corrects non-uniform motion (tissue that stretches, not just slides).

        We split the width into thirds and crop the top/bottom so the panels align. Drag the frame
        slider and stare at a bright cell: in **raw** it wanders; in **rigid** it steadies; in
        **pw-rigid** it locks. (We subsample the movie so a bare cloud kernel never chokes.)
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
        A single frame is a weak lie-detector — motion is a *time* property. So we flatten time into a
        picture next.

        ---
        ## 2. Kymographs — reading motion the way you read a stabilized track

        A **kymograph** fixes one horizontal line of the image (row `y`) and stacks that line across
        every frame: **x = position along the row, y = time.** A pixel column that stays put traces a
        **straight vertical streak**; a column that jitters left-right **wiggles**. It's the imaging
        twin of laying every frame's pose track on top of each other and asking *"is this line
        straight?"* — exactly the NB01 sanity check for a stabilized track.

        Pick the row `y` and a **time window** to highlight (the red band). Compare the three panels:
        raw streaks wander, rigid straightens them, pw-rigid straightens them most.
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
        The kymograph turns "is it registered?" into "are the streaks straight?" — but eyeballing
        doesn't scale, and it can't tell rigid from pw-rigid when both look decent. We need a **scalar**.

        ---
        ## 3. The motion index — one number for "how much jitter is left"

        `nu.motion_index(frames)` is the mean absolute **frame-to-frame difference**,
        $\;\text{MI} = \langle\,|f_{t+1} - f_{t}|\,\rangle$. A perfectly still movie has MI near zero
        (consecutive frames are identical); a jittery one has large MI because every pixel keeps
        changing. Registration should **push MI down**, and better registration should push it lower:

        $$\text{MI}(\text{raw}) \;>\; \text{MI}(\text{rigid}) \;>\; \text{MI}(\text{pw-rigid}).$$

        The per-frame version below (`nu.motion_index_trace`) shows the jitter over time for all three —
        the pw-rigid line rides lowest almost everywhere. This is the same logic as a *tracking-error
        trace* in NB01: a stabilized track has small frame-to-frame position change.
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
        ## 4. Exercise — does registration actually reduce motion?

        > **Hypothesis banner.** *Registration reduces frame-to-frame motion, and piecewise-rigid beats
        > plain rigid:* $\text{MI}(\text{pw-rigid}) < \text{MI}(\text{rigid}) < \text{MI}(\text{raw})$.

        **Toolbox.**

        - `nu.motion_index(frames)` — scalar mean |frame-to-frame Δ| for a `(F, H, W)` movie.
        - `raw`, `rigid`, `pwr` — the three `(F, H', w)` panels, already split for you.
        - No I/O needed; the movie is loaded.

        **Your job.** Compute the motion index for each of the three panels and store them as
        `mi_raw`, `mi_rigid`, `mi_pwr`. The self-check verifies the ordering **and** that pw-rigid
        removes a real chunk of motion (a tolerance band pinned from the real movie). Grade the
        *conclusion*, not a decimal: the honest result is that registration monotonically lowers MI.
        """
    )
    return


@app.cell
def _(nu, pwr, raw, rigid):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    mi_raw = nu.motion_index(raw)
    mi_rigid = nu.motion_index(rigid)
    mi_pwr = nu.motion_index(pwr)
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
            `mi_pwr < mi_rigid < mi_raw`. Rigid registration shaves a few percent of frame-to-frame
            motion off the raw movie; piecewise-rigid shaves off ~10% total — it catches the *local*
            warping that a single whole-frame shift can't. The absolute values scale with how you
            subsample (bigger frame gaps → bigger apparent motion), which is why the **ordering** is the
            real result, not the decimal. This is the imaging-side proof of the same claim NB01 made
            about tracks: alignment is not cosmetic, it measurably reduces the signal's jitter.
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
        "Deeper: the paper & where the analogy stops": mo.md(
            r"""
            **The motion-correction lineage.** The rigid / piecewise-rigid split you just looked at is
            the design of **NoRMCorre** — Pnevmatikakis & Giovannucci, *"NoRMCorre: An online algorithm
            for piecewise rigid motion correction of calcium imaging data,"* **J. Neurosci. Methods**
            291:83–94 (2017) — the motion-correction stage of the widely used **CaImAn** pipeline
            (Giovannucci et al., *eLife* 2019). NoRMCorre estimates a template, aligns each frame to it
            by phase-correlation (rigid), then refines with **per-patch** shifts that are smoothly
            interpolated back together (piecewise-rigid) to catch non-uniform tissue motion — exactly
            the raw → rigid → pw-rigid progression rendered in this movie.

            **The shared mathematics with NB01.** Both are *image/point registration*: estimate a
            transform (a translation, or a field of local translations) that best maps one time sample
            onto a reference, then apply it so downstream reads are made in a stable frame. Pose
            stabilization and miniscope motion correction are two instances of the same alignment
            problem.

            **Where the analogy stops.** Pose tracking registers a handful of *labeled keypoints* with
            known identity; motion correction registers *dense pixel intensities* with no labels, and
            must invent its own reference template — so it can be fooled by real brightness changes
            (a neuron firing looks a little like the frame moving) in a way keypoint tracking is not.
            And "better MI" is necessary but **not sufficient**: over-aggressive piecewise warping can
            lower the motion index while smearing real signal, so pipelines validate against a
            correlation image and residual traces, not MI alone.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## The twin, closed — and what we ship next

        You just relived NB01 on the imaging rig. **Before you can read a signal, you must align it
        across time.** The pose-tracking drift you stabilized in Week 1 is the same enemy as miniscope
        FOV drift; rigid and piecewise-rigid registration are the imaging analog of stabilizing tracks,
        and the **motion index** is the number that proves it worked (`pw-rigid < rigid < raw`).

        A registered movie is a stack of frames where pixel `(y, x)` means the same neuron over time.
        That is the *precondition* for the actual deliverable of Week 2. **Next (NB10): pull a neuron's
        trace out of the pixels** — background-subtract a striatal miniscope movie, drop an ROI on a
        cell, and watch its calcium trace separate from background. Registration made "a neuron's
        trace" a meaningful phrase; NB10 finally reads it.
        """
    )
    return


if __name__ == "__main__":
    app.run()

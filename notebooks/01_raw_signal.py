# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "hdbscan>=0.8.36",
#     "plotly>=5.20",
#     "imageio>=2.34",
#     "pillow>=10.0",
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
            if os.path.isdir(os.path.join(p, "course")) and os.path.isdir(os.path.join(p, "data")):
                return p
            p = os.path.dirname(p)
        return None
    ROOT = _find_root() or os.getcwd()
    _cu = os.path.join(ROOT, "course", "course_utils.py")
    if not os.path.exists(_cu):
        os.makedirs(os.path.dirname(_cu), exist_ok=True)
        urllib.request.urlretrieve(_RAW + "/course/course_utils.py", _cu)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import course_utils as cu
    ROOT, DATA, SCRATCH = cu.bootstrap()
    return ROOT, cu, go, np


@app.cell
def _(cu, np):
    import pandas as pd

    # --- the training corpus, and the held-out cage (Camera 16, sealed until NB08) ---
    ev = cu.load_events("data/train_events.npz")     # kp, ranks, condition, contact_rel, event_key, ...
    der = cu.load_derived("train")                   # X, pca_scores, cage, sex, tod_hour, ...
    ho = cu.load_events("data/heldout_events.npz")   # Camera 16 — kept sealed until NB08

    # --- our example approach event: one real approach we return to all week (Cage 15, male) ---
    # The three mice are ordered [approacher, approachee, bystander]. In this event the approacher
    # is the Dom mouse (red), the approachee is the Sub mouse (green), the bystander is Mid (blue).
    EX_IDX = 909
    ex_kp = ev["kp"][EX_IDX]                          # (130, 3, 15, 2) pose over time
    ex_ranks = ev["ranks"][EX_IDX]                    # (3,) rank of each ordered mouse
    ex_cr = int(ev["contact_rel"][EX_IDX])            # frame at which contact begins

    # Readout-Board benchmarks (committed); degrade gracefully if a value is missing.
    try:
        board = pd.read_csv(cu.data_path("data/readout_board.csv"))
    except Exception:
        board = None

    # mouse colors are ALWAYS by rank: Dom=red, Mid=blue, Sub=green, unknown=gray.
    ex_cols = tuple(cu.RANK_HEX.get(int(r), cu.RANK_HEX[0]) for r in ex_ranks)
    size_student = int(np.prod(ev["kp"].shape[1:]))   # 130 * 3 * 15 * 2 = 11,700 numbers per event
    return EX_IDX, board, der, ev, ex_cols, ex_cr, ex_kp, ex_ranks, ho, size_student


@app.cell
def _(go):
    # One reusable two-gauge Readout Board, shown at the top and bottom of the notebook.
    # Gauge A = size of the representation (how many numbers describe one event). Gauge B =
    # held-out readiness (a decoding score that stays at 0 until Phase 2). FIX: Gauge A uses
    # mode="number" only. An earlier version used mode="number+delta" against an 11,700 baseline,
    # which rendered a confusing negative ("-11,681") and crowded the title. NB01 IS the raw
    # baseline, so Gauge A simply shows 11,700 with no delta.
    def readout_fig(sizeA, sizeA_bench, readyB, readyB_bench):
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number", value=sizeA,
            number={"valueformat": ",.0f", "suffix": " numbers", "font": {"size": 44}},
            title={"text": "<b>Gauge A</b> · size of the representation<br>"
                           "<span style='font-size:0.8em;color:#888'>raw pose — the starting "
                           "point</span>"},
            domain={"row": 0, "column": 0}))
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=readyB,
            number={"valueformat": ".0%"},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#4c78a8"},
                   "threshold": {"line": {"color": "#e45756", "width": 3},
                                 "value": readyB_bench}},
            title={"text": "<b>Gauge B</b> · held-out readiness<br>"
                           "<span style='font-size:0.8em;color:#888'>rises through Phase 2</span>"},
            domain={"row": 0, "column": 1}))
        fig.update_layout(grid={"rows": 1, "columns": 2, "pattern": "independent"},
                          height=250, template="plotly_white",
                          margin=dict(l=30, r=30, t=95, b=20))
        return fig
    return (readout_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB01 · The Raw Signal
        ### *Reading the Social Brain — Week 1*

        Your job in this course is to study social behavior in mice with the same rigor a
        neuroscientist brings to neural recordings. To do that, we first need a clear, objective
        description of what each mouse is *doing* — not a hand-written note like "mouse looks
        aggressive," but a number we can measure, compare across animals, and check for mistakes.

        We start from **pose tracking**. The videos in this course were processed with **SLEAP**, a
        program that finds a small set of body landmarks (nose, ears, tail, and so on) on each mouse
        in every video frame. The output is not a picture; it is a table of coordinates. This week we
        build a pipeline that turns those coordinates into a trustworthy readout of social behavior.
        (Quantifying behavior this way is also how many neuroscience labs now measure what an animal
        does during an experiment.)

        **Today's goal.** Before building anything on top of the pose data, we check that it is
        reliable. Concretely: the data files label one mouse "the approacher." Is that label real, or
        just bookkeeping? And what happens downstream if the tracker accidentally swaps two mice? By
        the end of this notebook you will have inspected the raw pose, seen where it fails, and tested
        one of its central assumptions yourself.
        """
    )
    return


@app.cell(hide_code=True)
def _(board, mo, readout_fig, size_student):
    _benchA = 11700.0
    if board is not None:
        _r = board[(board["notebook"] == "NB01") & (board["gauge"] == "A")]
        if len(_r):
            _benchA = float(_r["value"].iloc[0])
    _fig = readout_fig(size_student, _benchA, 0.0, 0.86)
    mo.vstack([
        mo.md("## The Readout Board &nbsp;·&nbsp; *start of the week*"),
        _fig,
        mo.md("*The two gauges track two different things across the course. **Gauge A** counts how "
              "many numbers we use to describe a single event; it starts at its maximum "
              "(**11,700 raw numbers per event**) and shrinks as we build better representations. "
              "**Gauge B** is a decoding score on held-out data; it is **0** now because no decoder "
              "exists yet, and it rises in Week 2. The red tick marks our target (held-out "
              "AUROC ≈ 0.86).*"),
    ])
    return


@app.cell(hide_code=True)
def _(ho, mo):
    _n = len(ho["kp"])
    mo.md(
        f"""
        <div style="border:2px solid #999; border-radius:10px; padding:14px 18px;
        background:#f6f6f8; color:#333;">
        <div style="font-size:1.15em; font-weight:600; color:#555;">Held-out data · Camera 16
        (sealed)</div>
        <div style="margin-top:6px;">
        <b>{_n} events</b> recorded from a cage that no analysis this week will look at.
        </div>
        <div style="margin-top:8px; font-size:0.95em; color:#555;">
        A decoder is only convincing if it works on data it was never tuned on. We therefore set
        Camera 16 aside now and only open it in NB08, to test the finished pipeline on a cage it has
        never seen. <b>Notebooks until it is unsealed: 7.</b>
        </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1 · The pose tensor, axis by axis

        **Why this matters.** Everything we do rests on one array of numbers per event. If we do not
        understand its shape and its quirks, every later step inherits our confusion. So we start by
        reading the array carefully.

        **Definitions.**

        - A **keypoint** (or *node*) is one tracked body landmark — for example the nose or the base
          of the tail. Each keypoint has an (x, y) position in the video image, measured in pixels.
        - A **pose** is the full set of keypoints for one mouse in one frame. Our skeleton has
          **15 keypoints**.
        - A **frame** is one still image from the video. Our clips run at **50 frames per second (fps)**.
        - A **tensor** is just a multi-dimensional array of numbers (a table with more than two axes).

        SLEAP gives us, per event, one array `kp` with shape **`(T, mice, nodes, xy)`**. Read it left
        to right:

        | axis | size | meaning |
        |---|---|---|
        | `T` | 130 | frames (about 2.6 s at 50 fps) |
        | `mice` | 3 | the three mice, ordered `[approacher, approachee, bystander]` |
        | `nodes` | 15 | body keypoints (nose, head, tail base, tail tip, …) |
        | `xy` | 2 | pixel coordinates (note: y grows *downward* in image space) |

        Two properties matter downstream and are worth stating now:

        - **A missing keypoint is stored as `NaN`, not 0.** `NaN` means "not a number" — here it marks
          a keypoint the tracker could not place on that frame. Using 0 would falsely pin that body
          part to the top-left corner of the image; `NaN` honestly records "unknown," and our code
          skips it.
        - **A position in the array is not a guaranteed identity.** The mice are stored in fixed
          slots, and slot 0 is *labelled* "the approacher." But that label is only as good as the
          tracker's ability to keep each mouse in its own slot on every frame. When two mice are close
          together, the tracker can swap them — and then slot 0 silently holds the other mouse. We
          test this assumption later in the notebook.

        The 15 keypoints connect through two hubs — **head (node 1)** and **TTI (node 11**, the
        junction where the tail meets the torso) — in a star-shaped skeleton.
        """
    )
    return


@app.cell(hide_code=True)
def _(ev, ex_kp, mo):
    # Print the array's shape/axes in plain language, computed from the real data.
    _N, _T, _M, _Nn, _xy = ev["kp"].shape
    mo.md(
        f"""
        **Reading the shape directly from the data.** The full training array `ev["kp"]` has shape
        **`{tuple(ev["kp"].shape)}`** — that is **{_N} events**, each **{_T} frames** long, with
        **{_M} mice**, **{_Nn} keypoints** per mouse, and **{_xy} numbers (x, y)** per keypoint.

        One event, `ex_kp = ev["kp"][{{EX}}]`, has shape **`{tuple(ex_kp.shape)}`**. To pull out a
        single mouse's single keypoint over the whole event, we index the middle two axes and keep
        `:` on the first (time) and last (x, y) axes. For example, the approacher's nose track is
        `ex_kp[:, 0, cu.NOSE, :]`, an array of shape **`{tuple(ex_kp[:, 0, 0, :].shape)}`** — one
        (x, y) pair per frame. Multiplying the last four axes together gives
        **{_T} × {_M} × {_Nn} × {_xy} = {_T*_M*_Nn*_xy:,} numbers per event**, the value on Gauge A.
        """.replace("{EX}", "909")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise 1 — plot one keypoint's position over time

        **Goal.** Get comfortable indexing the pose tensor by pulling out a single keypoint and
        plotting how it moves. The output is a simple line plot you can sanity-check by eye.

        **What to edit.** In the cell below, exactly one line is blank. Replace `____` so that
        `nose_xy` becomes the approacher's nose track — mouse `0`, node `cu.NOSE`, all frames, both
        (x, y):

        ```python
        # ex_kp has shape (T, mice, nodes, xy) = (130, 3, 15, 2).
        # TODO: select mouse 0 (the approacher), node cu.NOSE, ALL frames, both (x, y) columns.
        # The result should have shape (130, 2): one (x, y) pair per frame.
        nose_xy = ____                      # e.g. ex_kp[:, 0, cu.NOSE, :]

        # The lines below already plot the two columns. Leave them as they are.
        # x = nose_xy[:, 0]   (horizontal pixel position)
        # y = nose_xy[:, 1]   (vertical pixel position)
        ```

        **What you should see.** Two red curves against frame number (the approacher is the Dom mouse,
        so we draw it red): a **solid** line for x and a **dashed** line for y. Both curves wander up
        and down as the mouse moves around the arena over the ~2.6 s. Short, steep segments are moments
        of fast movement; flat segments are moments the mouse holds still. If your curves are flat at a
        single value or you get a shape error, you indexed the wrong axis — check that you kept `:`
        on the first and last axes.
        """
    )
    return


@app.cell
def _(cu, ex_kp, go, np):
    # Reference solution (runs on load so the figure always renders).
    _nose_xy = ex_kp[:, 0, cu.NOSE, :]                 # (130, 2): approacher nose (x, y) per frame
    _t = np.arange(ex_kp.shape[0])
    _fig = go.Figure()
    _fig.add_scatter(x=_t, y=_nose_xy[:, 0], mode="lines", name="x (px)",
                     line=dict(color="#d62728", width=2))
    _fig.add_scatter(x=_t, y=_nose_xy[:, 1], mode="lines", name="y (px)",
                     line=dict(color="#d62728", width=2, dash="dash"))
    _fig.update_layout(template="plotly_white", height=320,
                       title="Approacher nose position over time (Dom mouse, red)",
                       xaxis_title="frame", yaxis_title="pixel coordinate",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(cu, ex_kp, mo, np):
    mo.accordion({
        "Reveal solution — Exercise 1": mo.md(
            r"""
            ```python
            nose_xy = ex_kp[:, 0, cu.NOSE, :]     # mouse 0, nose node, all frames, (x, y)
            #          |          |        |  |
            #        frames   approacher nose  x and y
            ```
            `ex_kp[:, 0, cu.NOSE, :]` keeps every frame (`:` on axis 0), selects mouse `0` on the
            mice axis, selects the nose on the nodes axis (`cu.NOSE` is 0), and keeps both `x` and `y`
            on the last axis. The result has shape """
            + f"`{tuple(ex_kp[:, 0, cu.NOSE, :].shape)}`" + r""" — 130 rows (frames), 2 columns
            (x, y).
            """)
    })
    return


@app.cell
def _(cu, ex_cols, ex_kp, np):
    # A labelled diagram of the skeleton so you know what the keypoints are. There are no raw video
    # frames in this teaching bundle, so instead of drawing the skeleton on a photo of a mouse we
    # draw one real pose on a blank canvas and print every keypoint's index and name. We use the
    # approacher's most-complete frame and color it by the approacher's rank (Dom = red).
    _nfin = np.isfinite(ex_kp[:, 0]).all(axis=2).sum(axis=1)   # tracked-node count per frame
    _best = int(np.argmax(_nfin))
    _pose = ex_kp[_best, 0]                                    # (15, 2) one mouse, one frame
    cu.labelled_skeleton_fig(_pose, color=ex_cols[0],
                             title="The 15-keypoint SLEAP skeleton (approacher, Dom = red)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Reading the diagram.** Each dot is one keypoint; each line is a skeleton *edge* connecting
        two keypoints. The layout is a **star**: most keypoints hang off one of two hubs. The
        **head (node 1)** anchors the front — nose, ears, shoulders, neck. The **TTI (node 11)**, the
        tail–torso junction, anchors the back — haunches, trunk, and the four tail keypoints
        (`tail_1`, `tail_0`, `tail_2`, `tail_tip`). Keep this picture in mind: when we talk about a
        keypoint dropping out or two mice being confused, this is the object it happens to.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    frame_slider = mo.ui.slider(0, 129, value=40, step=1, label="frame", debounce=True,
                                full_width=True)
    return (frame_slider,)


@app.cell
def _(cu, ex_cols, ex_cr, ex_kp, frame_slider, mo):
    # Scrub through the example event one frame at a time. Colors are by rank (Dom=red, Mid=blue,
    # Sub=green). As long as each color stays attached to one animal, identities are intact.
    _t = frame_slider.value
    _tag = "  ·  contact" if _t >= ex_cr else ""
    _fig = cu.skeleton_fig(ex_kp[_t], cu.SKELETON_EDGES, colors=ex_cols,
                           title=f"Example approach event — frame {_t}/129{_tag}", height=480)
    mo.vstack([
        mo.md(f"**Frame scrubber — the example approach event.** Contact begins at frame "
              f"**{ex_cr}**. Watch the three skeletons keep their colors as you drag: each rank "
              "color should stay glued to one animal. The white line connects the approacher to the "
              "approachee."),
        frame_slider,
        _fig,
    ])
    return


@app.cell
def _(cu, ex_cr, ex_kp, ex_ranks, mo):
    # The same event as a short looping GIF (embedded as a data-URI so it animates in the notebook).
    _gif = cu.event_gif_bytes(ex_kp, ex_ranks, contact_rel=ex_cr, cell=200, fps=20)
    mo.vstack([mo.md("*The whole 2.6 s at a glance: the approacher (Dom, red) closes on the "
                     "approachee (Sub, green); the small red dot marks the contact frames.*"),
               mo.Html(cu.gif_img_html(_gif, width=220))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2 · Checking the signal: what is missing, and where

        **Why this matters.** Pose tracking is not perfect. Some keypoints are found more reliably
        than others, and the reliable and unreliable ones are not random — they follow the anatomy and
        the camera view. Before trusting any measurement, we ask which keypoints we can rely on.

        **Definition.** *Tracked fraction* of a keypoint = the fraction of frames (across all events)
        on which that keypoint is present (finite, not `NaN`). A value of 1.0 means "found every
        frame"; 0.5 means "found half the time."
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Per-keypoint tracked fraction across the whole corpus, and the least-reliable keypoint.
    # cu.node_reliability(kp): input a (..., 15, 2) pose array; output a (15,) array giving the
    # fraction of frames each keypoint is finite.
    nr = cu.node_reliability(ev["kp"])
    least_node = cu.NODE_NAMES[int(np.argmin(nr))]
    return least_node, nr


@app.cell
def _(cu, go, nr):
    _tail = {9, 10, 12, 13}                              # the four tail keypoints
    _tip = cu.NODE_NAMES.index("tail_tip")
    _cols = ["#e45756" if i == _tip else ("#f2a25c" if i in _tail else "#4c78a8")
             for i in range(len(nr))]
    _fig = go.Figure(go.Bar(x=cu.NODE_NAMES, y=nr, marker_color=_cols))
    _fig.update_layout(template="plotly_white", height=340,
                       title="Per-keypoint tracked fraction — the tail keypoints drop out most",
                       yaxis_title="fraction of frames tracked", yaxis_range=[0, 1],
                       margin=dict(l=10, r=10, t=50, b=80))
    _fig.update_xaxes(tickangle=-45, showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(least_node, mo, nr):
    mo.md(
        f"""
        The body keypoints are tracked about **0.97–0.98** of frames. The **tail keypoints**
        (`tail_1`, `tail_0`, `tail_2`, `tail_tip`) drop to about **0.73–0.79**, and the single
        least-reliable keypoint is **`{least_node}`** (**{nr.min():.3f}**). This is not a cosmetic
        problem. The lab identifies and ranks each mouse from marks painted on the tail, so an
        unreliable tail directly weakens those identity and rank labels — a known source of about
        **16% labelling error** that we will have to keep in mind in NB06 and NB08.
        """
    )
    return


@app.cell
def _(cu, ev, mo):
    # Render a few events where the body is solid but the tail keypoints flicker in and out, so you
    # can SEE the dropout rather than just read a bar chart. These indices were chosen because the
    # body (nodes 0-8) and TTI stay tracked while the tail chain [9,10,12,13] appears ~half the
    # frames. Watch the tail dots blink on and off while the body stays put.
    _tail_events = [504, 954, 706]
    _events = [(ev["kp"][i], ev["ranks"][i], 40) for i in _tail_events]
    _gif = cu.grid_gif_bytes(_events, ncols=3, cell=170, fps=18)
    mo.vstack([
        mo.md("**Tail keypoints flickering (events 504, 954, 706).** The body skeleton stays intact "
              "while the tail dots repeatedly vanish and reappear — this is what the 0.73–0.79 tail "
              "tracked-fraction looks like frame by frame. Colors are by rank."),
        mo.Html(cu.gif_img_html(_gif, width=520)),
    ])
    return


@app.cell
def _(der, ev, go, np):
    # A quick look at the sample: how many events per condition and per cage, so you know what you
    # are working with.
    from plotly.subplots import make_subplots as _make_subplots
    _fig = _make_subplots(rows=1, cols=2, subplot_titles=("events per condition", "events per cage"))
    _conds, _cn = np.unique(ev["condition"], return_counts=True)
    _cpal = {"pre": "#54a24b", "dep": "#e45756", "post": "#4c78a8"}
    _fig.add_bar(x=list(_conds), y=_cn, marker_color=[_cpal.get(c, "#888") for c in _conds],
                 row=1, col=1, showlegend=False)
    _cages, _gn = np.unique(der["cage"], return_counts=True)
    _fig.add_bar(x=[f"cage {c}" for c in _cages], y=_gn, marker_color="#72539b",
                 row=1, col=2, showlegend=False)
    _fig.update_layout(template="plotly_white", height=320,
                       title="Your sample: 1500 training events, cages 9–15",
                       margin=dict(l=10, r=10, t=60, b=40))
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell
def _(cu, ex_cr, ex_kp, go, np):
    # How close do the two mice get? Plot the distance between the approacher's and approachee's
    # body centroids over the event. A "centroid" is the average position of a mouse's body
    # keypoints — a single point standing in for where the whole body is.
    _c0 = cu._centroids(ex_kp[:, 0])                    # approacher centroid per frame (T,2)
    _c1 = cu._centroids(ex_kp[:, 1])                    # approachee centroid per frame (T,2)
    _d01 = np.linalg.norm(_c0 - _c1, axis=1)            # centroid-to-centroid distance (px)
    _fig = go.Figure()
    _fig.add_scatter(y=_d01, mode="lines", line=dict(color="#333", width=2),
                     name="approacher–approachee distance")
    _fig.add_hline(y=150, line=dict(color="#e45756", dash="dash"),
                   annotation_text="≈ contact distance (150 px)", annotation_position="top left")
    _fig.add_vline(x=ex_cr, line=dict(color="#888", dash="dot"),
                   annotation_text="contact frame", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=320,
                       title="Example event — inter-mouse distance falls as the approach happens",
                       xaxis_title="frame", yaxis_title="pixels", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What the plot shows.** The black line is the distance (in pixels) between the two mice's
        body centroids at each frame; it falls as the approacher closes in. The dashed red line marks
        the distance at which we count the mice as being in contact.

        **Real numbers from this dataset.** These events were extracted as approaches: the two mice
        **start far apart** (centroid-to-centroid distance around **200–220 px**, median **221**) and
        **close** to about **154 px** by the contact frame (median over all 1,500 events), reaching a
        median closest approach of about **103 px**. At contact the approacher's nose comes within a
        median of **~37 px** of the approachee's tail base — that near-touch is what "contact"
        physically means. In this particular event the two centroids move from **178 px** at the start
        to **142 px** at contact (closest approach 78 px), a slightly tighter approach than the
        corpus median.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3 · A position in the array is not an identity: the swap problem

        **Why this matters.** Every label we build this week — who approached whom, who was
        aggressive — assumes the mouse in slot 0 stays the same animal for the whole event. If the
        tracker ever swaps two mice, that assumption breaks and the label can invert. So we need to
        understand when swaps happen and whether we can catch them.

        **Definition.** An **identity swap** is when the tracker exchanges two mice between slots at
        some frame. From that frame on, "the approacher" (slot 0) is really the other mouse.

        **How a swap detector works.** A mouse cannot teleport, so one way to flag a swap is to watch
        each track's **centroid velocity** — how far its body centroid moves from one frame to the
        next, measured in pixels per frame. (This is a *normalized* velocity: displacement divided by
        the time step, so it is a per-frame rate rather than a raw distance, and it can be compared
        across events.) A sudden large jump in centroid velocity is suspicious.

        **The catch.** If you swap two tracks at a frame, the jump this creates is exactly *how far
        apart the two mice were* at that frame. So a swap that happens **at contact** — where the mice
        are almost on top of each other — produces only a **tiny** jump that slips under any
        threshold. Swaps are therefore hardest to detect precisely where the mice are closest, which
        is exactly where aggression happens. Drag the threshold below to see this.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    thr_slider = mo.ui.slider(10, 300, value=80, step=5, label="swap-detector threshold (px)",
                              debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(cu, ex_cr, ex_kp, go, mo, np, thr_slider):
    # For the example event: at every frame, the jump a slot 0<->1 swap would create equals the
    # distance between those two tracks' centroids at that frame. Frames where that jump is below the
    # detector threshold are blind spots.
    _c0 = cu._centroids(ex_kp[:, 0]); _c1 = cu._centroids(ex_kp[:, 1])
    _induced = np.linalg.norm(_c0 - _c1, axis=1)          # jump a 0<->1 swap would create
    _thr = thr_slider.value
    _undetectable = _induced < _thr
    _fig = go.Figure()
    _fig.add_scatter(y=_induced, mode="lines", line=dict(color="#1f77b4", width=2),
                     name="jump a swap would create")
    _fig.add_hline(y=_thr, line=dict(color="#e45756", dash="dash"),
                   annotation_text=f"detector threshold = {_thr} px", annotation_position="top left")
    _frames = np.arange(len(_induced))
    _fig.add_scatter(x=_frames[_undetectable], y=_induced[_undetectable], mode="markers",
                     marker=dict(color="#e45756", size=6), name="swap here would be missed")
    _fig.add_vline(x=ex_cr, line=dict(color="#888", dash="dot"),
                   annotation_text="contact", annotation_position="bottom right")
    _fig.update_layout(template="plotly_white", height=360,
                       title="Where would a swap slip past the detector? (example event)",
                       xaxis_title="frame", yaxis_title="induced jump (px)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _n_blind = int(_undetectable.sum())
    mo.vstack([
        thr_slider,
        _fig,
        mo.md(f"At this threshold, **{_n_blind}/{len(_induced)} frames** are blind spots (red), and "
              "they cluster around **contact**. Lower the threshold and you catch more swaps but also "
              "flag honest fast movement as a swap (false alarms); raise it and swaps at contact "
              "disappear. No single threshold avoids both problems."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 2 — which events contain an identity swap?

        **Goal.** Use the centroid-velocity idea from Section 3 to find swaps yourself. You are given
        five events. Some are clean; in a couple of them we have deliberately introduced a swap (we
        exchanged two mice for a few frames, to simulate a tracking error). Your job is to plot the
        centroid velocities and decide which events show a swap.

        **The tool.** `cu.centroid_jumps(clip)` takes one event's pose array (shape `(T, 3, 15, 2)`)
        and returns an array of shape **`(3, T-1)`**: one row per mouse, each value the distance that
        mouse's centroid moved since the previous frame (pixels per frame). A clean event has three
        low, flat traces. A swap shows up as a **single tall spike shared by two mice at the same
        frame** — because both tracks jump by the inter-mouse distance at the instant they are
        exchanged.

        **What to edit.** In the cell below, one line is blank:

        ```python
        # clip is one event's pose array, shape (T, 3, 15, 2).
        # TODO: compute the per-mouse centroid velocity for this clip.
        # Replace ____ with cu.centroid_jumps(clip). Result shape: (3, 129).
        jumps = ____
        # The code below already plots the three rows of `jumps`, one small panel per event.
        ```

        **What you should see.** Five small panels, one per event. **Three** of them show three low,
        flat traces (natural movement, peaks around 20–25 px). **Two** of them show a single very tall
        spike (roughly 200–400 px) shared by two of the traces at the same frame — those are the
        swapped events. Note the two event numbers with the tall spike; then open the reveal to check
        your answer and watch the swap happen in the rendered GIFs.
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Build the five-event exercise set. We deliberately induce a swap in two of them by exchanging
    # track slots 0 and 1 for frames [40, 48). Reference computation runs on load so the plot and
    # self-check always work.
    swap_events = [1496, 95, 77, 1121, 110]

    def _induce_swap(kp_event, f0, f1):
        """Swap track slots 0<->1 for frames [f0, f1) to simulate an identity swap."""
        out = kp_event.copy()
        out[f0:f1, [0, 1]] = out[f0:f1, [1, 0]]
        return out

    swap_clips = []
    for _pos, _idx in enumerate(swap_events):
        _k = ev["kp"][_idx].copy()
        if _pos in (1, 3):                                 # events 95 and 1121 get a swap
            _k = _induce_swap(_k, 40, 48)
        swap_clips.append(_k)

    # cu.centroid_jumps(clip) -> (3, T-1) per-mouse centroid velocity (px/frame).
    swap_maxjump = np.array([float(np.nanmax(cu.centroid_jumps(c))) for c in swap_clips])
    SWAP_THR = 100.0
    swap_detected = [swap_events[i] for i in range(5) if swap_maxjump[i] > SWAP_THR]
    return SWAP_THR, swap_clips, swap_detected, swap_events, swap_maxjump


@app.cell
def _(cu, ev, go, swap_clips, swap_events):
    # Plot the three centroid-velocity traces for each of the five events. Lines colored by rank.
    from plotly.subplots import make_subplots as _make_subplots
    _fig = _make_subplots(rows=1, cols=5, shared_yaxes=True,
                          subplot_titles=[f"event {i}" for i in swap_events])
    for _col, (_idx, _clip) in enumerate(zip(swap_events, swap_clips), start=1):
        _jumps = cu.centroid_jumps(_clip)                 # (3, T-1)
        _ranks = ev["ranks"][_idx]
        for _m in range(3):
            _c = cu.RANK_HEX.get(int(_ranks[_m]), cu.RANK_HEX[0])
            _fig.add_scatter(y=_jumps[_m], mode="lines", line=dict(color=_c, width=1.5),
                             showlegend=False, row=1, col=_col)
    _fig.update_layout(template="plotly_white", height=300,
                       title="Centroid velocity per mouse — spot the two events with a tall spike",
                       margin=dict(l=10, r=10, t=60, b=30))
    _fig.update_yaxes(title_text="px / frame", row=1, col=1)
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(cu, ev, mo, swap_clips):
    # Reveal: render the two swapped events so the swap is visible. During frames 40-47 the red and
    # blue/green skeletons jump to each other's positions, then jump back.
    _g95 = cu.event_gif_bytes(swap_clips[1], ev["ranks"][95], contact_rel=40, cell=200, fps=18)
    _g1121 = cu.event_gif_bytes(swap_clips[3], ev["ranks"][1121], contact_rel=40, cell=200, fps=18)
    mo.accordion({
        "Reveal answer — Exercise 2": mo.vstack([
            mo.md(
                r"""
                **The swapped events are 95 and 1121.** The clean events (1496, 77, 110) have a
                maximum centroid velocity of only about **22–25 px/frame** (ordinary movement). The
                two swapped events spike far above that: event **95** jumps to about **385 px/frame**
                and event **1121** to about **192 px/frame** — each spike equal to how far apart the
                two mice were at the swapped frame. A threshold anywhere between ~30 and ~180 px
                separates them cleanly.

                In the GIFs below (the two swapped events), watch frames 40–47: the skeletons jump to
                each other's positions and then snap back. That jump is the swap.
                """),
            mo.Html(cu.gif_img_html(_g95, width=220) + cu.gif_img_html(_g1121, width=220)),
        ])
    })
    return


@app.cell(hide_code=True)
def _(SWAP_THR, mo, swap_detected, swap_events, swap_maxjump):
    # Self-check: the two events flagged by the threshold should be exactly {95, 1121}.
    _ok = set(swap_detected) == {95, 1121}
    _bg = "#e6f4ea" if _ok else "#fce8e6"
    _bd = "#34a853" if _ok else "#ea4335"
    _mark = "PASS" if _ok else "CHECK"
    _rows = "".join(f"<li>event {e}: max {m:.0f} px/frame</li>"
                    for e, m in zip(swap_events, swap_maxjump))
    mo.md(
        f"""
        <div style="background:{_bg}; border-left:6px solid {_bd}; padding:12px 16px;
        border-radius:6px;">
        <b>{_mark}</b><br>
        Maximum centroid velocity per event:
        <ul style="margin:6px 0;">{_rows}</ul>
        Events above the {SWAP_THR:.0f} px/frame threshold: <b>{sorted(swap_detected)}</b>
        &nbsp;→ {"matches the answer key {95, 1121}." if _ok else "expected {95, 1121}."}<br>
        <span style="color:#555;">The two swapped events stand out clearly, while the three clean
        events stay well below the threshold.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 3 — is "slot 0 = the approacher" a real label?

        **Goal.** The files *assert* that slot 0 is the approacher. Rather than trust the label, we
        test a prediction that follows from it: if slot 0 really is the mouse doing the approaching,
        it should **move more than slot 1 in the frames just before contact**. We check this across
        all 1,500 events.

        **Definitions.** *Speed* here is the distance a keypoint moves between consecutive frames
        (pixels per frame). We use the **TTI** keypoint (the tail–torso junction, node 11) as a stable
        stand-in for body position. `np.diff` takes frame-to-frame differences; `np.linalg.norm`
        turns an (x, y) difference into a single distance; `np.nanmean` averages while ignoring
        untracked (`NaN`) frames.

        **The prediction.** Across the corpus, in a clear majority of events, mouse 0's mean speed in
        the 50 frames before contact exceeds mouse 1's. Separately, `tail_tip` should be the
        least-reliable keypoint (from Section 2).

        **What to edit.** One line is blank in the cell below:

        ```python
        def mean_pre_speed(k, cr, m):
            t0 = max(0, cr - 50)                 # start 50 frames before contact
            tti = k[t0:cr, m, cu.TTI, :]         # that mouse's TTI track, (window, 2)
            if len(tti) < 2:
                return np.nan
            # TODO: per-frame speed = length of the frame-to-frame change in position.
            # Replace ____ with np.linalg.norm(np.diff(tti, axis=0), axis=1).
            step = ____
            return np.nanmean(step)
        ```

        **What you should see.** A green PASS box reporting the fraction of events where mouse 0 moves
        more than mouse 1. It should land near **0.69** (well above the 0.5 you would get by chance),
        with a very small p-value, and it should confirm `tail_tip` as the least-reliable keypoint.
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Reference solution (runs on load so the self-check can grade; ~1500-event loop, well under 1s).
    def _mean_pre_speed(k, cr, m):
        _t0 = max(0, cr - 50)
        _tti = k[_t0:cr, m, cu.TTI, :]                    # (window, 2)
        if len(_tti) < 2:
            return np.nan
        _step = np.linalg.norm(np.diff(_tti, axis=0), axis=1)   # px/frame
        return np.nanmean(_step) if np.isfinite(_step).any() else np.nan

    _kp = ev["kp"]; _cr = ev["contact_rel"].astype(int)
    _s0 = np.array([_mean_pre_speed(_kp[i], _cr[i], 0) for i in range(len(_kp))])
    _s1 = np.array([_mean_pre_speed(_kp[i], _cr[i], 1) for i in range(len(_kp))])
    _valid = np.isfinite(_s0) & np.isfinite(_s1)
    n_valid = int(_valid.sum())
    n_more = int(np.sum(_s0[_valid] > _s1[_valid]))
    frac_more = n_more / n_valid

    from scipy.stats import binomtest
    p_more = binomtest(n_more, n_valid, 0.5).pvalue
    return frac_more, n_more, n_valid, p_more


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution — Exercise 3": mo.md(
            r"""
            ```python
            def mean_pre_speed(k, cr, m):
                t0 = max(0, cr - 50)
                tti = k[t0:cr, m, cu.TTI, :]                    # (window, 2)
                if len(tti) < 2:
                    return np.nan
                step = np.linalg.norm(np.diff(tti, axis=0), axis=1)   # px/frame
                return np.nanmean(step)

            s0 = np.array([mean_pre_speed(kp[i], cr[i], 0) for i in range(len(kp))])
            s1 = np.array([mean_pre_speed(kp[i], cr[i], 1) for i in range(len(kp))])
            v  = np.isfinite(s0) & np.isfinite(s1)
            frac = np.mean(s0[v] > s1[v])                       # about 0.69
            p    = binomtest((s0[v] > s1[v]).sum(), v.sum(), 0.5).pvalue

            least = cu.NODE_NAMES[np.argmin(cu.node_reliability(kp))]   # 'tail_tip'
            ```
            The approacher out-moving the approachee about **69%** of the time confirms that slot 0 is
            a real role, not a coin flip. It is not 100%: passive co-approaches and the occasional
            swap live in the other 31%, which is the label noise we work to reduce over the week.
            """)
    })
    return


@app.cell(hide_code=True)
def _(frac_more, least_node, mo, n_valid, p_more):
    # Tolerance-band self-check: fraction is high and above 0.5, and tail_tip is least reliable.
    _ok_frac = (0.63 <= frac_more <= 0.75) and (p_more < 1e-6)
    _ok_node = (least_node == "tail_tip")
    _ok = _ok_frac and _ok_node
    _bg = "#e6f4ea" if _ok else "#fce8e6"
    _bd = "#34a853" if _ok else "#ea4335"
    _mark = "PASS" if _ok else "CHECK"
    mo.md(
        f"""
        <div style="background:{_bg}; border-left:6px solid {_bd}; padding:12px 16px;
        border-radius:6px;">
        <b>{_mark}</b><br>
        Fraction of events with <b>mouse 0 &gt; mouse 1</b> (pre-contact TTI speed):
        <b>{frac_more:.3f}</b> over {n_valid} events &nbsp;(sign test p = {p_more:.1e}).
        &nbsp;Target band: <b>0.63–0.75</b>, p &lt; 1e-6 → {"met." if _ok_frac else "not met."}<br>
        Least-reliable keypoint found: <b>{least_node}</b> →
        {"tail_tip, as predicted." if _ok_node else "expected tail_tip."}<br>
        <span style="color:#555;">The approacher really does move more, so the role label is honest
        at the population level, and the tail tip really is the weakest keypoint. Both results are
        graded against values pinned at build time.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Review questions": mo.md(
            r"""
            **1. Why does a single identity swap corrupt a "who approached whom" label more than a few
            dropped keypoints do?** A dropped keypoint removes one measurement on one frame — a small,
            local gap we can average over with `nanmean`. A swap relabels the whole event: every
            frame after the swap assigns the behavior to the wrong mouse, so the *direction* of the
            behavior flips. Missing data adds noise, which averages out; a swap adds a systematic
            error, which does not.

            **2. Why are swaps hardest to catch exactly when they matter most?** A velocity-based
            detector flags a swap by the size of the jump it creates, and that jump equals the
            distance between the two mice. During contact the mice are closest, so the jump is
            smallest — right where aggression happens, the swap is nearly invisible (Section 3).

            **3. Further reading.** The pose tracking used here is SLEAP (Pereira et al., 2022,
            *Nature Methods*); a related tool is DeepLabCut (Mathis et al., 2018, *Nature
            Neuroscience*). Both estimate body keypoints from video; neither, on its own, solves the
            identity-over-time problem we examined today.
            """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What we have committed to

        We have not discarded any numbers yet — we are still holding all **11,700** raw values per
        event. But we have committed to two assumptions that everything downstream depends on:

        1. **`NaN` frames can be safely skipped.** This is fine for small gaps but weakens the tail
           keypoints (tracked only ~73% of frames), which in turn weakens the tail-mark identity and
           rank labels (about 16% error). That limit is baked in now and caps rank results in NB06 and
           the decoder in NB08.
        2. **Each slot holds one identity for all 130 frames.** Section 3 showed this can fail at
           contact, exactly where aggression occurs, and Exercise 2 showed what a failure looks like.

        We also confirmed the useful news: the "approacher" label is real at the population level
        (Exercise 3, ~69%), so it is safe to build on — as long as we stay alert to swaps and to the
        weak tail.

        **Next → `02_body_eye_view.py`.** Before we can shrink those 11,700 numbers into something
        smaller, we have to choose a point of view. NB02 re-expresses every event from the
        approacher's own body frame (an *egocentric* transform) and reduces it to 19 interpretable
        features.
        """
    )
    return


@app.cell(hide_code=True)
def _(board, mo, readout_fig, size_student):
    _benchA = 11700.0
    if board is not None:
        _r = board[(board["notebook"] == "NB01") & (board["gauge"] == "A")]
        if len(_r):
            _benchA = float(_r["value"].iloc[0])
    _fig = readout_fig(size_student, _benchA, 0.0, 0.86)
    mo.vstack([
        mo.md("## The Readout Board &nbsp;·&nbsp; *end of NB01*"),
        _fig,
        mo.md(
            r"""
            We showed the raw signal is reliable enough to build on: the approacher role is real, the
            tail is the weak keypoint, and swaps are a contact-time risk we now know to watch for.
            **Gauge A** still reads its maximum — **11,700 raw numbers per event** — and **Gauge B**
            (the decoder) does not exist yet. Both change starting in NB02.
            """),
    ])
    return


if __name__ == "__main__":
    app.run()

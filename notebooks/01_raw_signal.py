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

    # --- the corpus (train), the sealed rig (heldout / Cage 16), and the neural cousin ---
    ev = cu.load_events("data/train_events.npz")     # kp, ranks, condition, contact_rel, event_key, category, agg_label
    der = cu.load_derived("train")                   # X, pca_scores, cage, sex, tod_hour, ...
    ho = cu.load_events("data/heldout_events.npz")   # Camera 16 — stays SEALED until NB08
    hod = cu.load_derived("heldout")
    neu = cu.load_neural_demo()                       # synthetic population raster (the "neural cousin")

    # --- the Hero Event: one real aggression approach we follow all week (Cage 15, male) ---
    HERO = 909
    hero_kp = ev["kp"][HERO]                          # (130, 3, 15, 2) world-coordinate pose
    hero_ranks = ev["ranks"][HERO]                    # (3,) [approacher, approachee, bystander] ranks
    hero_cr = int(ev["contact_rel"][HERO])            # frame contact begins

    # Readout-Board benchmarks (committed); degrade gracefully if a value is missing.
    try:
        board = pd.read_csv(cu.data_path("data/readout_board.csv"))
    except Exception:
        board = None

    # rank colors for the Hero skeletons: Dom=red, Mid=blue, Sub=green
    hero_cols = tuple(cu.RANK_HEX.get(int(r), cu.RANK_HEX[0]) for r in hero_ranks)
    size_student = int(np.prod(ev["kp"].shape[1:]))   # 130 * 3 * 15 * 2 = 11,700 numbers per event
    return (HERO, board, der, ev, hero_cols, hero_cr, hero_kp, hero_ranks,
            ho, neu, size_student)


@app.cell
def _(go):
    # One reusable two-gauge Readout Board (top + bottom of the notebook).
    def readout_fig(sizeA, sizeA_bench, readyB, readyB_bench):
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+delta", value=sizeA,
            number={"valueformat": ",.0f"},
            delta={"reference": sizeA_bench, "relative": False, "valueformat": ",.0f"},
            title={"text": "Gauge A · size of the representation<br>"
                           "<span style='font-size:0.75em;color:#888'>numbers per event "
                           "(falls through Phase 1)</span>"},
            domain={"row": 0, "column": 0}))
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=readyB,
            number={"valueformat": ".0%"},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#4c78a8"},
                   "threshold": {"line": {"color": "#e45756", "width": 3},
                                 "value": readyB_bench}},
            title={"text": "Gauge B · held-out readiness<br>"
                           "<span style='font-size:0.75em;color:#888'>rises through Phase 2</span>"},
            domain={"row": 0, "column": 1}))
        fig.update_layout(grid={"rows": 1, "columns": 2, "pattern": "independent"},
                          height=230, template="plotly_white",
                          margin=dict(l=30, r=30, t=70, b=10))
        return fig
    return (readout_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB01 · The Raw Signal
        ### *Reading the Social Brain — Build the Decoder for the Rig*

        > **FROM:** Circuit Team &nbsp;→&nbsp; **TO:** Behavior Team
        >
        > Welcome to the Behavior Team. A neural experiment is coming: a laser that flips a
        > hypothalamic switch, a probe in mPFC. But a manipulation is worthless without an
        > **objective readout of what each mouse actually does** — hand-scoring won't survive
        > review. Your week-long job is to turn raw SLEAP pose from three co-housed mice into a
        > **decoder** we can time-align to a recording.
        >
        > **Today's deliverable:** prove the pose we'll read behavior from is *trustworthy* — that
        > "mouse 0 = the approacher" is a real claim and not an accident of bookkeeping.
        > **It unblocks:** every downstream label; if identities are wrong, "who attacked whom"
        > is wrong.
        > **The lab-meeting question:** *"The arrays label mouse 0 the 'approacher.' Is that real —
        > and what happens downstream if an identity is swapped?"*
        >
        > And the **bet** the whole week rides on, written on the board today: *a decoder is only
        > trustworthy if it survives a cage it never saw.* That cage is **Camera 16** — sealed
        > below until the final notebook.

        Keeping each mouse's identity straight across frames is the *same* problem a **spike sorter**
        solves: binding a stream of ambiguous detections to a small set of stable sources over time.
        You are about to meet that problem in its rawest form.
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
        mo.md("## The Readout Board &nbsp;·&nbsp; *start of week*"),
        _fig,
        mo.md("*Gauge A and Gauge B are **different kinds** of number — a raw count vs a decoding "
              "score — not one magic quantity. Today Gauge A reads its maximum (**11,700 raw "
              "numbers per event**) and Gauge B reads **0**: the decoder does not exist yet. The red "
              "tick marks the target we're racing toward (held-out AUROC ≈ 0.86).*"),
    ])
    return


@app.cell(hide_code=True)
def _(ho, mo):
    _n = len(ho["kp"])
    mo.md(
        f"""
        <div style="border:2px dashed #b33; border-radius:10px; padding:14px 18px; background:
        repeating-linear-gradient(45deg,#1a1a1a,#1a1a1a 12px,#222 12px,#222 24px); color:#eee;">
        <div style="font-size:1.3em; letter-spacing:2px; color:#ff6b6b;">🔒 SEALED · CAMERA 16</div>
        <div style="margin-top:6px;">
        <b>{_n} events</b> recorded &nbsp;·&nbsp; skeletons <span style="filter:blur(3px);
        color:#888;">▚▚▚▚▚▚▚</span> &nbsp;·&nbsp; labels
        <span style="background:#000; color:#000;">████████ ████ ██</span>
        </div>
        <div style="margin-top:8px; font-size:0.95em; color:#ffd;">
        This is <b>the animal on the rig</b> — a cage no analysis this week is allowed to look at.
        The decoder must earn the right to see it. <b>Notebooks until unlock: 7.</b>
        </div>
        </div>

        *The forbidden fruit is on the tree, not merely named. When Camera 16 opens in NB08 it either
        vindicates the week's work or exposes it — that is the only test a circuit experiment would
        believe.*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1 · A pose tensor, axis by axis

        SLEAP gives us, per event, one array `kp` with shape **`(T, mice, nodes, xy)`**. Read it left
        to right:

        | axis | size | meaning |
        |---|---|---|
        | `T` | 130 | frames (~2.6 s at 50 fps) |
        | `mice` | 3 | **track slots** `[approacher, approachee, bystander]` |
        | `nodes` | 15 | body landmarks (nose, head, TTI, tail_tip, …) |
        | `xy` | 2 | pixel coordinates (y grows *downward* in image space) |

        That is a **high-dimensional state vector over time** — the exact shape of object a
        neuroscientist stares at when a population of neurons fires across a trial. Two things to burn
        in now, because everything downstream inherits them:

        - **`NaN` = untracked.** A node the network couldn't place that frame is *not zero* — it is
          missing. Zeros would drag a body part to the origin; `NaN` honestly says "unknown."
        - **A track slot is not an identity.** Slot 0 is *labelled* "the approacher," but that label
          is only as good as the tracker's ability to hold each mouse in its own slot across every
          frame. When two mice touch, that binding can fail — and slot 0 quietly becomes the *other*
          mouse. Guarding against that is our whole job today.

        The 15 nodes hang off two hubs — **head (node 1)** and **TTI (node 11**, the tail–torso
        junction) — in a star skeleton.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    frame_slider = mo.ui.slider(0, 129, value=40, step=1, label="frame", debounce=True,
                                full_width=True)
    return (frame_slider,)


@app.cell
def _(cu, frame_slider, hero_cols, hero_cr, hero_kp, mo):
    # Hero Event #909 (Cage 15, male, aggression) rendered raw — this notebook's "method" is the
    # skeleton itself. Drag the slider to scrub; colors are RANK (Dom=red, Mid=blue, Sub=green).
    _t = frame_slider.value
    _tag = "  ·  CONTACT" if _t >= hero_cr else ""
    _fig = cu.skeleton_fig(hero_kp[_t], cu.SKELETON_EDGES, colors=hero_cols,
                           title=f"Hero Event #909 — frame {_t}/129{_tag}", height=480)
    mo.vstack([
        mo.md(f"**Frame scrubber — Hero Event #909.** Contact begins at frame **{hero_cr}**. "
              "Watch the three skeletons hold their colors: as long as each rank-color stays glued "
              "to one animal, identity is intact. The white arrow points approacher→approachee."),
        frame_slider,
        _fig,
    ])
    return


@app.cell
def _(cu, hero_cr, hero_kp, hero_ranks, mo):
    # The same Hero Event as a small looping GIF (rendered by cu, embedded as a data-URI so it
    # animates — marimo's static image widget would freeze on frame 0).
    _gif = cu.event_gif_bytes(hero_kp, hero_ranks, contact_rel=hero_cr, cell=200, fps=20)
    mo.vstack([mo.md("*The whole 2.6 s at a glance — a Dom (red) closing on a Sub (green); the red "
                     "dot flags contact frames:*"),
               mo.Html(cu.gif_img_html(_gif, width=220))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2 · Audit the signal: what's missing, and where

        Before we trust a single number, we ask the questions a physiologist asks of a raw trace:
        *which channels are reliable, is every source present, and when do the sources get confusable?*
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Per-node tracked fraction across the whole corpus, and the least-reliable node.
    nr = cu.node_reliability(ev["kp"])                 # (15,) finite fraction per node
    least_node = cu.NODE_NAMES[int(np.argmin(nr))]
    return least_node, nr


@app.cell
def _(cu, go, nr):
    _tail = {9, 10, 12, 13}                              # the tail chain
    _tip = cu.NODE_NAMES.index("tail_tip")
    _cols = ["#e45756" if i == _tip else ("#f2a25c" if i in _tail else "#4c78a8")
             for i in range(len(nr))]
    _fig = go.Figure(go.Bar(x=cu.NODE_NAMES, y=nr, marker_color=_cols))
    _fig.update_layout(template="plotly_white", height=340,
                       title="Per-node tracked fraction — the tail chain drops out",
                       yaxis_title="fraction of frames tracked", yaxis_range=[0, 1],
                       margin=dict(l=10, r=10, t=50, b=80))
    _fig.update_xaxes(tickangle=-45, showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(least_node, mo, nr):
    mo.md(
        f"""
        The body nodes sit near **0.97–0.98** tracked; the **tail chain** (tail_1, tail_0, tail_2,
        **tail_tip**) sags to **0.73–0.79**. The single least-reliable node is **`{least_node}`**
        (**{nr.min():.3f}**). This is not cosmetic: the tail is exactly what the lab's *tail-mark*
        identity/rank scheme reads, which is why those labels carry **~16% error** — a standing
        caveat you'll feel bite in NB06 and NB08. Hold that prediction; you'll grade it below.
        """
    )
    return


@app.cell
def _(der, ev, go, np):
    # Sample audit: events per condition and per cage — so students know their sample.
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
def _(cu, go, hero_cr, hero_kp, np):
    # Minimum inter-mouse centroid distance per frame, on the Hero Event.
    _cen = np.stack([cu._centroids(hero_kp[:, m]) for m in range(3)], axis=0)   # (3, T, 2)
    _pairs = [(0, 1), (0, 2), (1, 2)]
    _dist = np.stack([np.linalg.norm(_cen[a] - _cen[b], axis=1) for a, b in _pairs], axis=0)
    _mind = np.nanmin(_dist, axis=0)
    _fig = go.Figure()
    _fig.add_scatter(y=_mind, mode="lines", line=dict(color="#333", width=2),
                     name="min inter-mouse dist")
    _fig.add_vline(x=hero_cr, line=dict(color="#e45756", dash="dash"),
                   annotation_text="contact", annotation_position="top")
    _fig.update_layout(template="plotly_white", height=320,
                       title="Hero Event #909 — the mice get *close* right at contact",
                       xaxis_title="frame", yaxis_title="pixels", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3 · Track slot **≠** identity: the swap, and why it hides at contact

        Here is the failure that keeps a behavior team up at night. Suppose the tracker **swaps**
        tracks 0 and 1 at some frame — from then on, "the approacher" is really the *other* mouse.
        A natural detector flags a swap by watching each track's **centroid jump** between frames: a
        real mouse can't teleport, so a big jump smells like a swap.

        The catch: when you swap two tracks at a frame, the jump you *induce* is exactly **how far
        apart the two mice were** at that frame. So a swap **at contact** — where the mice are almost
        on top of each other — produces a *tiny* jump that slips under any threshold. Swaps don't just
        happen at contact by accident; they are **invisible** there by construction. That is the
        behavioral twin of a spike-sorting **collision**: two overlapping spikes are hardest to
        assign exactly when they coincide.
        """
    )
    return


@app.cell
def _(mo):
    thr_slider = mo.ui.slider(10, 300, value=80, step=5, label="swap-detector threshold (px)",
                              debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(cu, go, hero_cr, hero_kp, mo, np, thr_slider):
    # The induced-jump-of-a-swap at every frame == the track0<->track1 centroid distance there.
    _c0 = cu._centroids(hero_kp[:, 0]); _c1 = cu._centroids(hero_kp[:, 1])
    _induced = np.linalg.norm(_c0 - _c1, axis=1)          # size of the jump a 0<->1 swap makes
    _thr = thr_slider.value
    _undetectable = _induced < _thr                       # swaps here slip under the detector
    _fig = go.Figure()
    _fig.add_scatter(y=_induced, mode="lines", line=dict(color="#1f77b4", width=2),
                     name="jump a 0↔1 swap would create")
    _fig.add_hline(y=_thr, line=dict(color="#e45756", dash="dash"),
                   annotation_text=f"detector threshold = {_thr}px", annotation_position="top left")
    # shade frames where a swap would go UNDETECTED
    _frames = np.arange(len(_induced))
    _fig.add_scatter(x=_frames[_undetectable], y=_induced[_undetectable], mode="markers",
                     marker=dict(color="#e45756", size=6), name="swap here = UNDETECTED")
    _fig.add_vline(x=hero_cr, line=dict(color="#888", dash="dot"),
                   annotation_text="contact", annotation_position="bottom right")
    _fig.update_layout(template="plotly_white", height=360,
                       title="Where would a swap slip past the detector? (Hero Event #909)",
                       xaxis_title="frame", yaxis_title="induced jump (px)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _n_blind = int(_undetectable.sum())
    mo.vstack([
        thr_slider,
        _fig,
        mo.md(f"**{_n_blind}/{len(_induced)} frames** are blind spots at this threshold — and they "
              "cluster right around **contact**, exactly where real aggression lives. Lower the "
              "threshold to catch more swaps and you start flagging honest fast movement as swaps "
              "(false alarms); raise it and swaps at contact vanish. There is no threshold that wins."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4 · The neural cousin

        To make "two sciences, one problem" concrete from hour one: on the **left** is our pose
        tensor for one mouse — 15 nodes × 130 frames, a high-dimensional signal over time. On the
        **right** is a **population raster** from `neural_demo.npz` — 60 neurons × trials, the object a
        systems neuroscientist decodes. *Both* are high-D signals over time; *both* have a
        detection-and-identity problem (a spike must be found, then bound to the right neuron —
        estimation, then sorting). We return to this raster in NB08 and decode it with the **exact**
        pipeline we build for behavior.
        """
    )
    return


@app.cell
def _(cu, hero_kp, neu, np):
    from plotly.subplots import make_subplots as _make_subplots
    import plotly.graph_objects as _pgo
    # LEFT: pose tensor slice — mouse 0, node (15) x frame (130), the y-coordinate.
    _pose = hero_kp[:, 0, :, 1].T                          # (15 nodes, 130 frames)
    # RIGHT: neural raster — sort neurons by tuning, trials by hidden state, for visible structure.
    _X = neu["X_neural"]; _y = neu["y"]; _tuned = neu["is_tuned"].astype(bool)
    _norder = np.concatenate([np.where(_tuned)[0], np.where(~_tuned)[0]])
    _torder = np.argsort(_y)[:200]                         # first 200 trials, grouped by state
    _rast = _X[np.ix_(_torder, _norder)].T                # (60 neurons, 200 trials)
    _fig = _make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                         subplot_titles=("Pose tensor — 1 mouse, 15 nodes × 130 frames",
                                         "Neural raster — 60 neurons × trials"))
    _fig.add_trace(_pgo.Heatmap(z=_pose, colorscale="Viridis", showscale=False,
                                y=cu.NODE_NAMES), row=1, col=1)
    _fig.add_trace(_pgo.Heatmap(z=_rast, colorscale="Magma", showscale=False), row=1, col=2)
    _fig.update_layout(template="plotly_white", height=420,
                       title="Your data's neural cousin — same shape of problem",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_xaxes(title_text="frame", row=1, col=1, showgrid=False)
    _fig.update_xaxes(title_text="trial (sorted by hidden state)", row=1, col=2, showgrid=False)
    _fig.update_yaxes(title_text="neuron (tuned on top)", row=1, col=2)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 🧪 Exercise — does the approacher really move more?

        The arrays *assert* that slot 0 is the approacher. Let's test it as a hypothesis instead of
        trusting the label — the first act of a behavior team that doesn't fool itself.

        ### 🔧 Toolbox
        | tool | inputs → output |
        |---|---|
        | `cu.TTI` | index of the tail–torso node (11), a stable centroid proxy |
        | `np.diff` + `np.linalg.norm` | frame-to-frame node speed (px/frame) |
        | `np.nanmean` | average over frames, ignoring untracked frames |
        | `scipy.stats.binomtest` | sign test: is the fraction of "0 > 1" events ≠ 0.5? |
        | `cu.node_reliability(kp)` | (15,) tracked fraction per node |

        ### 📋 Hypothesis (pre-registered)
        > **Mouse 0 (the "approacher") moves more than mouse 1 in the 50 frames *before* contact —
        > across the corpus, in a strong majority of events.** And separately: **`tail_tip` is the
        > least-reliable node.**

        ### ✍️ Your TODO
        For every event: take each mouse's **TTI-node** track over the 50 frames **before**
        `contact_rel`, compute its mean per-frame speed, and record whether **mouse 0 > mouse 1**.
        Return the fraction of events where that holds, and run a sign test against 0.5. Then predict
        the least-reliable node from `node_reliability`.

        *The reference implementation below runs on load so the self-check can grade — open the
        solution to compare with your own.*
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # --- reference solution (runs on load; ~1500-event loop, <1s) ---
    def _mean_pre_speed(k, cr, m):
        _t0 = max(0, cr - 50)
        _tti = k[_t0:cr, m, cu.TTI, :]                    # (w, 2)
        if len(_tti) < 2:
            return np.nan
        _d = np.linalg.norm(np.diff(_tti, axis=0), axis=1)
        return np.nanmean(_d) if np.isfinite(_d).any() else np.nan

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
        "💡 Reveal solution": mo.md(
            r"""
            ```python
            def mean_pre_speed(k, cr, m):
                t0 = max(0, cr - 50)
                tti = k[t0:cr, m, cu.TTI, :]          # (w, 2) pre-contact TTI track
                if len(tti) < 2:
                    return np.nan
                d = np.linalg.norm(np.diff(tti, axis=0), axis=1)   # px/frame
                return np.nanmean(d)

            s0 = np.array([mean_pre_speed(kp[i], cr[i], 0) for i in range(len(kp))])
            s1 = np.array([mean_pre_speed(kp[i], cr[i], 1) for i in range(len(kp))])
            v  = np.isfinite(s0) & np.isfinite(s1)
            frac = np.mean(s0[v] > s1[v])                 # ≈ 0.69
            p    = binomtest((s0[v] > s1[v]).sum(), v.sum(), 0.5).pvalue

            least = cu.NODE_NAMES[np.argmin(cu.node_reliability(kp))]   # 'tail_tip'
            ```
            The approacher out-moving the approachee **~69%** of the time is the sanity check that
            slot 0 is a *real* role, not a coin flip. It's not 100% — passive co-approaches and swaps
            live in the other 31%, which is precisely the label-noise we chase all week.
            """)
    })
    return


@app.cell(hide_code=True)
def _(frac_more, least_node, mo, n_valid, p_more):
    # tolerance-band self-check: fraction high & > 0.5, and tail_tip least reliable
    _ok_frac = (0.63 <= frac_more <= 0.75) and (p_more < 1e-6)
    _ok_node = (least_node == "tail_tip")
    _ok = _ok_frac and _ok_node
    _bg = "#e6f4ea" if _ok else "#fce8e6"
    _bd = "#34a853" if _ok else "#ea4335"
    _mark = "✅ PASS" if _ok else "❌ CHECK"
    mo.md(
        f"""
        <div style="background:{_bg}; border-left:6px solid {_bd}; padding:12px 16px; border-radius:6px;">
        <b>{_mark}</b><br>
        Fraction of events with <b>mouse 0 &gt; mouse 1</b> (pre-contact TTI speed):
        <b>{frac_more:.3f}</b> over {n_valid} events &nbsp;(sign test p = {p_more:.1e}).
        &nbsp;Target band: <b>0.63–0.75</b>, p &lt; 1e-6 → {"met ✓" if _ok_frac else "not met ✗"}.<br>
        Least-reliable node predicted &amp; found: <b>{least_node}</b> →
        {"tail_tip ✓" if _ok_node else "expected tail_tip ✗"}.<br>
        <span style="color:#555;">The approacher really does move more — the role label is honest at
        the population level — and the tail tip really is the flimsiest landmark. Both predictions
        graded against pinned build-time values, not noise.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "🧠 Conceptual questions": mo.md(
            r"""
            **1. Why does a single identity swap corrupt a "who-approached-whom" label more
            catastrophically than a handful of dropped nodes?** A dropped node degrades one feature on
            one frame — a small, local blur we can `nanmean` over. A swap **relabels the whole event**:
            every subsequent frame's "approacher" is now the wrong mouse, so the *sign* of the
            behavior flips. Missingness adds variance; a swap adds **bias**, and bias doesn't average
            out.

            **2. Merge vs split — the spike-sorting analog.** A **merge** error lumps two sources into
            one identity (two mice tracked as "mouse 0"); a **split** error tears one source into two
            (one mouse's frames scattered across slots). A contact-time track swap is a **merge-like
            collision**: two nearby sources become confusable exactly where they overlap — the same
            reason overlapping spikes are the hard case for a sorter.
            """),
        "🔬 Deeper: the paper & where the analogy stops": mo.md(
            r"""
            **Pose estimation** binds pixels → landmarks (SLEAP: **Pereira et al. 2022, *Nat.
            Methods***, this course's own lab lineage; DeepLabCut: **Mathis et al. 2018, *Nat.
            Neurosci.***). **Identity tracking over time** binds detections → stable sources — the same
            job a **spike sorter** does (**Lewicki 1998, *Network***; **Kilosort, Pachitariu et al.
            2016**).

            *Shared mathematics:* both are **data-association** problems — assign ambiguous
            observations to a small set of persistent latent sources across time.

            *Species / preparation:* freely-moving mice, multi-animal video (behavior) vs
            extracellular electrophysiology (neural).

            **Where the analogy stops:** estimation ≈ **detection** and tracking ≈ **sorting** are
            *two different steps* — don't collapse them. And a spike sorter assigns to *electrodes*
            with known geometry; our "electrode" is a camera and our sources physically occlude one
            another at contact, which no electrode array does.
            """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 🗑️ What we threw away — and how it breaks

        **Thrown away (so far, almost nothing — and that's the point):** we are still holding all
        11,700 raw numbers. But we've *committed* to two lossy assumptions: that `NaN` frames can be
        ignored, and that **each track slot holds one identity for all 130 frames**. Everything
        downstream is a debt against those two assumptions.

        **How it breaks *on this dataset*:**
        - **Tail-chain dropout.** tail_tip is tracked only ~73% of frames; the lab's identity/rank
          labels are read off tail marks → **~16% mislabeled**. That error is baked in *now*; it caps
          every rank result in NB06 and the decoder ceiling in NB08.
        - **Swaps hide at contact.** The one place aggression happens is the one place a swap is
          invisible to a jump detector (Section 3). A "who initiated" label can silently invert.
        - **Slot ≠ role for passive approaches.** ~31% of events *don't* have mouse 0 moving more —
          co-approaches and mutual investigations where "approacher" is genuinely ambiguous.

        **🌍 How would you analyze this?** *If you could add one cheap sensor to disambiguate identity
        at contact, what would it be — and which pose feature would it repair?* (Think: what is
        physically distinct about two mice even when their skeletons overlap?)
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
            **What we ship next.** We proved the raw signal is *trustworthy enough to build on*: the
            approacher role is real, the tail is the weak node, and swaps are a contact-time threat we
            now respect. Gauge A still reads its maximum — **11,700 raw numbers per event** — and the
            decoder (Gauge B) doesn't exist yet. That's the job of the week.

            Before we can *compress* those 11,700 numbers, we have to choose a **point of view** — the
            same choice the brain makes with reference frames. **Next → `02_body_eye_view.py`:** the
            egocentric transform and the 19 features, re-expressing every event the way retrosplenial
            cortex would.
            """),
    ])
    return


if __name__ == "__main__":
    app.run()

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
def _(ROOT, cu, np):
    # Load the corpus: keypoints + labels from the event file, features + PCA from the derived bundle.
    ev  = cu.load_events(cu.data_path("data/train_events.npz", ROOT))
    der = cu.load_derived("train", ROOT)
    hod = cu.load_derived("heldout", ROOT)                 # held-out Cage 16 (count only, for now)

    kp    = ev["kp"]                                        # (1500, T, 3, 15, 2) keypoints, for GIFs
    ranks = ev["ranks"]                                     # (1500, 3) rank of each ordered mouse

    Xz, _mu, _sd = cu.standardize(der["X"])                # z-score the 19 features
    # Refit a full-rank PCA so the scree curve can reach 90%. The shipped der['pca_scores'] keeps
    # only 10 components (caps at 0.884 cumulative), not enough to read 90% off the curve.
    sc, evr, _pca = cu.pca_scores(Xz, 19)                  # sc (1500,19), evr (19,)
    comp = _pca.components_                                 # (19,19) loadings
    fn   = [str(f) for f in der["feature_names"]]
    cage = der["cage"]
    cond = ev["condition"].astype(str)
    yagg = ev["agg_label"].astype(int)

    cumvar = np.cumsum(evr)
    dim90  = int(np.searchsorted(cumvar, 0.90) + 1)        # smallest k with >=90% variance
    cum6   = float(cumvar[5])                               # variance kept by the first 6 PCs
    n_ho   = int(len(hod["cage"]))                          # held-out events

    EXAMPLE = 909            # our example approach event (Cage-15 male; approacher Dom, approachee Sub)
    return (EXAMPLE, cage, comp, cond, cum6, cumvar, dim90, evr, fn, kp,
            n_ho, ranks, sc, yagg)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB04 · Dimensionality reduction — how many dimensions does behavior have?

        ## Why this notebook

        In NB02 you turned each approach event into **19 numbers** (features) such as closing speed,
        pair distance, and mutual facing. Nineteen numbers is easier to reason about than raw pose,
        but it is still a lot — and the 19 are not independent. When two mice move quickly toward
        each other, closing speed, pair distance, and facing all change together. When several
        measurements rise and fall together, they are really reporting on a smaller number of
        underlying things.

        This notebook asks a simple, quantitative question: **how many independent directions does
        mouse social behavior actually occupy?** If the answer is far fewer than 19, we can describe
        each event with a handful of numbers and lose almost nothing. This is also how neuroscientists
        summarize a large population of measurements.

        ## Definitions (read these before the method)

        - **Dimensionality reduction** — replacing many measurements per event with a few new numbers
          that keep most of the information. The goal is a shorter description that preserves the
          structure in the data.
        - **Principal Component Analysis (PCA)** — the most common linear method for this. It finds
          new axes through the data, ordered so the first axis captures the most spread, the second
          captures the most of what is left, and so on.
        - **Principal component (PC)** — one of those new axes. Each PC is a fixed weighted recipe of
          the original 19 features. An event's *score* on a PC is how far along that axis it sits.
        - **Variance explained** — the fraction of the total spread in the data that a component
          accounts for. If PC1 explains 30% of the variance, it captures 30% of the differences
          between events.
        - **Residualization** — deliberately removing one or more PCs (setting their scores to zero)
          so the remaining description reflects everything *except* those axes. We use it to set aside
          a dominant but uninformative axis before clustering.

        ## The plan

        You will (1) build intuition for what PCA maximizes, (2) run PCA on all 19 features and count
        how many components you actually need, (3) read what each component means and see it in video,
        (4) confirm that removing an axis is a choice with a measurable cost, and (5) place every
        event — including our example approach event — as a single point in the reduced space.
        """
    )
    return


@app.cell
def _(mo):
    def board_html(where, dim90, cum6):
        # Progress board. Gauge A = size of the representation (shrinks as we compress). Gauge B =
        # held-out readiness (stays at 0 until Phase 2). Both values are shown as positive numbers.
        stages = [("NB01", "11,700", "raw pose / event"),
                  ("NB02", "19", "body-frame features"),
                  ("NB04", "~6", f"PCs · {cum6*100:.0f}% variance"),
                  ("NB05", "2", "map → syllable")]
        cells = ""
        for nb, val, note in stages:
            hot = "background:#e8f0fe;font-weight:700;" if nb == "NB04" else "opacity:.6;"
            cells += (f"<td style='padding:6px 12px;text-align:center;{hot}border-radius:6px;'>"
                      f"<div style='font-size:1.5em'>{val}</div>"
                      f"<div style='font-size:.72em;color:#666'>{nb} · {note}</div></td>")
        gaugeA = (f"<div style='font-size:.85em;color:#444;margin-bottom:4px'><b>Gauge A — size of "
                  f"the representation</b> &nbsp;<span style='color:#888'>(smaller = more compressed)"
                  f"</span></div>"
                  f"<table style='border-collapse:separate;border-spacing:6px'><tr>{cells}</tr></table>")
        gaugeB = ("<div style='margin-top:10px;font-size:.85em;color:#444'><b>Gauge B — held-out "
                  "readiness</b>: <span style='color:#888'>not started</span> — rises in Phase 2 "
                  "(target: Cage-16 decode AUROC <b>0.86</b>, in NB08).</div>")
        note = (f"<div style='margin-top:8px;font-size:.8em;color:#666'>Your run: <b>6 PCs → "
                f"{cum6*100:.1f}%</b> of variance (benchmark: 6 PCs ≈ 71%). Reaching <b>90%</b> "
                f"takes <b>{dim90}</b> PCs — these are two different targets, not one number.</div>")
        title = "PROGRESS BOARD" + ("" if where == "top" else " — end of NB04")
        return mo.md(f"<div style='border:1px solid #ddd;border-radius:10px;padding:12px 14px'>"
                     f"<div style='font-size:.75em;letter-spacing:.08em;color:#999'>{title}</div>"
                     f"{gaugeA}{gaugeB}{note}</div>")
    return (board_html,)


@app.cell
def _(board_html, cum6, dim90):
    board_html("top", dim90, cum6)
    return


@app.cell(hide_code=True)
def _(mo, n_ho):
    mo.md(
        f"""
        <div style="border:1px solid #bbb;border-radius:10px;padding:10px 14px;background:#f7f7f7">
        <b>Held out · Cage 16</b> &nbsp; <span style="font-family:monospace">n = {n_ho} events</span><br>
        <span style="color:#555;font-size:.9em">These events are set aside. Their PCA scores exist,
        but we do not look at them. A decoder is only trustworthy if it works on a cage it never
        trained on, so Cage 16 stays sealed until we test it in NB08.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · The core idea: find the direction of most spread

        PCA rests on one question: if you could keep only a single direction through a cloud of
        points, which direction preserves the most spread? Everything else — the linear algebra of
        eigenvectors and covariance — is machinery for answering that question automatically.

        The plot below shows two of the real features (standardized so both are on the same scale).
        They are correlated, so the cloud of events forms a tilted ellipse. **Drag the angle** to
        rotate a candidate axis. The title reports the *variance of the points projected onto your
        axis* — how spread out they are along it. Try to find the angle that maximizes it. Then turn
        on **Reveal PCA's axis**: the first principal component is exactly the direction you were
        looking for.

        What you should see: the projected variance is small when your line points across the short
        width of the ellipse, and largest when it lies along the long diagonal. The revealed PC1 lands
        on that diagonal.
        """
    )
    return


@app.cell
def _(mo):
    toy_angle = mo.ui.slider(0, 180, value=20, step=1, label="candidate axis angle (°)",
                             debounce=True, full_width=True)
    toy_reveal = mo.ui.switch(value=False, label="Reveal PCA's axis")
    return toy_angle, toy_reveal


@app.cell
def _(cu, der, fn, go, mo, np, toy_angle, toy_reveal):
    from sklearn.decomposition import PCA as _PCA
    _fa, _fb = "pair_dist_mean", "pair_dist_min"                      # two correlated features -> tilted cloud
    _cols = np.array([fn.index(_fa), fn.index(_fb)])
    _Xz2, _, _ = cu.standardize(der["X"][:, _cols])                   # (1500,2) standardized pair
    _rng = np.random.RandomState(0)
    _sel = _rng.choice(_Xz2.shape[0], size=500, replace=False)        # subsample for a snappy plot
    _P = _Xz2[_sel]

    _th = np.deg2rad(toy_angle.value)
    _u = np.array([np.cos(_th), np.sin(_th)])
    _var = float((_P @ _u).var())

    _p2 = _PCA(n_components=2).fit(_Xz2)
    _pc1 = _p2.components_[0]
    _best_deg = np.rad2deg(np.arctan2(_pc1[1], _pc1[0])) % 180
    _max_var = float(_p2.explained_variance_[0])

    _L = 3.2
    _fig = go.Figure()
    _fig.add_scattergl(x=_P[:, 0], y=_P[:, 1], mode="markers",
                       marker=dict(size=5, color="#7f7f7f", opacity=0.45), name="events")
    _fig.add_scatter(x=[-_L*_u[0], _L*_u[0]], y=[-_L*_u[1], _L*_u[1]], mode="lines",
                     line=dict(color="#f58518", width=3), name="your axis")
    if toy_reveal.value:
        _v = _pc1 / np.linalg.norm(_pc1)
        _fig.add_scatter(x=[-_L*_v[0], _L*_v[0]], y=[-_L*_v[1], _L*_v[1]], mode="lines",
                         line=dict(color="#111111", width=3, dash="dash"),
                         name=f"PC1 (angle {_best_deg:.0f}°)")
    _fig.update_layout(template="plotly_white", height=460,
                       title=(f"Projected variance = {_var:.2f}   "
                              f"(max possible {_max_var:.2f} at {_best_deg:.0f}°)"),
                       xaxis_title=f"{_fa} (z)", yaxis_title=f"{_fb} (z)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(range=[-_L, _L], showgrid=False, zeroline=True)
    _fig.update_yaxes(range=[-_L, _L], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=True)
    mo.vstack([toy_angle, toy_reveal, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You just **maximized variance over a rotation** — that is all PCA does, generalized from 2
        features to 19. The winning direction is **PC1**; the best axis *perpendicular* to it is
        **PC2**; and so on down the line, each capturing the most remaining spread.

        /// details | Optional: the eigen-math behind the slider
        Standardize the data to $X$ ($n$ events × 19 features, each column mean-0). The covariance is
        $C=\tfrac1n X^\top X$. PCA solves the eigenproblem $C v = \lambda v$: each **eigenvector**
        $v_k$ is a principal direction, and its **eigenvalue** $\lambda_k$ is the variance captured
        along it. The scores are the projections $Z = X V$. "Maximize projected variance over all
        unit directions" and "take the top eigenvector of the covariance" are the *same* statement —
        the slider above was solving that eigenproblem by hand. The *variance explained ratio* is just
        $\lambda_k / \sum_j \lambda_j$.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · How many components do we need? The scree plot

        Now run PCA on all 19 standardized features. A **scree plot** shows two things at once: the
        bars are how much variance each individual component explains, and the red line is the running
        total (cumulative variance) as you add components left to right. Reading it tells you how many
        axes you must keep to retain a chosen fraction of the information.

        **Drag `keep k`** to highlight the first k components and read the cumulative percentage in
        the title. You should see the bars fall off quickly — the first few are tall, the rest are
        short — and the cumulative line rise steeply and then flatten. This is where Gauge A on the
        board drops from 19 features to about 6 components.
        """
    )
    return


@app.cell
def _(mo):
    keep_k = mo.ui.slider(1, 19, value=6, step=1, label="keep k PCs", debounce=True, full_width=True)
    return (keep_k,)


@app.cell
def _(cumvar, dim90, evr, go, keep_k, mo, np):
    _k = keep_k.value
    _x = np.arange(1, 20)
    _fig = go.Figure()
    _fig.add_bar(x=_x, y=evr, name="per-PC variance",
                 marker=dict(color=["#4c78a8" if i < _k else "#cfd8e3" for i in range(19)]))
    _fig.add_scatter(x=_x, y=cumvar, mode="lines+markers", name="cumulative",
                     yaxis="y2", line=dict(color="#e45756", width=2))
    _fig.add_hline(y=0.90, line=dict(color="#999", dash="dot"), yref="y2")
    _fig.add_annotation(x=dim90, y=0.90, yref="y2", text=f"90% at {dim90} PCs",
                        showarrow=True, arrowhead=2, ax=40, ay=-30)
    _fig.update_layout(template="plotly_white", height=440,
                       title=f"Scree — first {_k} PCs keep {cumvar[_k-1]*100:.1f}% of variance",
                       xaxis_title="principal component",
                       yaxis=dict(title="variance ratio", showgrid=False),
                       yaxis2=dict(title="cumulative", overlaying="y", side="right",
                                   range=[0, 1.02], showgrid=False),
                       margin=dict(l=10, r=10, t=50, b=10), legend=dict(x=0.55, y=0.25))
    mo.vstack([keep_k, _fig])
    return


@app.cell(hide_code=True)
def _(cum6, dim90, mo):
    mo.md(
        f"""
        **Read the numbers honestly.** The first **6 components keep {cum6*100:.1f}%** of the variance
        — that is the "~6" on the board, and it is enough for the map we build next. Reaching a strict
        **90% threshold takes {dim90} components**. Behavior here is genuinely about 6-to-{dim90}
        dimensional: much less than 19, but more than one. "Dimensionality" depends on how much
        variance you insist on keeping. It is a setting, not a single fact.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · What does each component mean? Reading the loadings

        Each PC is a weighted recipe of the 19 features, and those weights are called **loadings**.
        The heatmap below shows the loadings of the top components: **red** means the feature pushes
        an event's score *up* along that PC, **blue** means it pushes it *down*. Reading across a row
        tells you which real behaviors that component combines. The purpose of this figure is to
        translate an abstract axis back into behavior.
        """
    )
    return


@app.cell
def _(comp, cu, fn):
    cu.pca_loadings_fig(comp, fn, k=4)
    return


@app.cell(hide_code=True)
def _(comp, fn, mo, np):
    _order = np.argsort(-np.abs(comp[0]))
    _top = ", ".join(f"`{fn[i]}`" for i in _order[:4])
    mo.md(
        f"""
        **Naming PC1.** Its largest loadings are {_top} — mean speeds, angular velocity, and mutual
        facing. Those all describe one coherent thing: **how much motion and mutual engagement is
        happening** between the two mice. It is also the highest-variance axis, which is why PCA ranks
        it first. A natural next step is to call this axis a nuisance ("just overall activity") and
        remove it. The next section shows why that removal is a choice with a cost, not a free
        cleanup.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### See it: behavior at the two ends of a component

        Loadings describe a component in words. GIFs let you see it. Below, pick a component and watch
        the events that score **lowest** versus **highest** on it. For PC1 you should see calm, mostly
        stationary pairs at the low end and fast, actively engaging pairs at the high end — the
        "amount of activity" axis made visible.

        Each mouse is colored by social **rank**: **red = Dom, blue = Mid, green = Sub**. The white
        arrow points from approacher to approachee; the red dot appears once the two make contact.
        """
    )
    return


@app.cell
def _(mo):
    pc_pick = mo.ui.dropdown(options=[f"PC{i+1}" for i in range(6)], value="PC1",
                             label="show behavior at the extremes of")
    return (pc_pick,)


@app.cell
def _(cu, kp, mo, np, pc_pick, ranks, sc):
    # Sort every event by its score on the chosen PC, then render the 3 lowest and 3 highest as
    # skeleton GIF grids. This makes the axis concrete: what behavior sits at each end.
    _pc = int(pc_pick.value[2:]) - 1
    _order = np.argsort(sc[:, _pc])                       # ascending score on this PC
    _low = _order[:3]                                     # 3 events lowest on this axis
    _high = _order[-3:]                                   # 3 events highest on this axis
    _lo_gif = cu.grid_gif_bytes([(kp[i], ranks[i], 40) for i in _low], ncols=3, cell=150)
    _hi_gif = cu.grid_gif_bytes([(kp[i], ranks[i], 40) for i in _high], ncols=3, cell=150)
    _html = (
        "<div style='display:flex;gap:24px;flex-wrap:wrap'>"
        f"<div><div style='margin-bottom:4px'><b>Low {pc_pick.value}</b> — bottom of the axis</div>"
        f"{cu.gif_img_html(_lo_gif, width=470)}</div>"
        f"<div><div style='margin-bottom:4px'><b>High {pc_pick.value}</b> — top of the axis</div>"
        f"{cu.gif_img_html(_hi_gif, width=470)}</div>"
        "</div>")
    mo.vstack([pc_pick, mo.md(_html)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · Removing an axis is a choice, not a fact

        Before clustering (NB05) we often remove the large "how fast / how close" axis so the map
        reflects finer differences instead of raw activity level. The helper `cu.residualize(scores,
        drop_pcs)` does exactly that:

        - **Purpose:** set chosen components aside so the rest of the pipeline behaves as if those axes
          never existed.
        - **Inputs:** `scores` (the PCA scores, one row per event) and `drop_pcs` (a list of component
          indices to remove).
        - **Output:** the same scores with those columns set to zero.

        Removing PC1 has a real cost, because aggression is partly a high-motion behavior and so lives
        partly on PC1. Below, choose which components to drop and read the aggression decoding score on
        the axes that remain. That score is the **AUROC** (area under the ROC curve): it measures how
        well a value separates aggression from non-aggression, where **1.0 is perfect and 0.5 is
        chance**. It is 5-fold cross-validated, meaning the model is trained and tested on separate
        splits of the data so the number reflects genuine prediction, not memorization.
        """
    )
    return


@app.cell
def _(mo):
    drop_sel = mo.ui.multiselect(options=[f"PC{i+1}" for i in range(6)], value=["PC1"],
                                 label="drop these PCs (zeroed before decoding)", full_width=True)
    return (drop_sel,)


@app.cell
def _(cu, drop_sel, go, mo, sc, yagg):
    from sklearn.model_selection import cross_val_score as _cvs
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.pipeline import make_pipeline as _mkp
    from sklearn.preprocessing import StandardScaler as _SS

    _drop = [int(s[2:]) - 1 for s in drop_sel.value]                  # "PC1" -> index 0
    _res = cu.residualize(sc, _drop)
    def _auc(S):
        return float(_cvs(_mkp(_SS(), _LR(max_iter=1000)), S, yagg, cv=5, scoring="roc_auc").mean())
    _full = _auc(sc)
    _kept = _auc(_res)

    _msg = (f"<div style='font-size:1.05em'>Aggression AUROC — all 19 PCs: <b>{_full:.3f}</b> "
            f"&nbsp;→&nbsp; after dropping {', '.join(drop_sel.value) or 'nothing'}: "
            f"<b>{_kept:.3f}</b></div>"
            f"<div style='color:#666;font-size:.85em;margin-top:4px'>Chance = 0.500. Dropping PC1 "
            f"weakens but does not erase the signal. That is the point: calling an axis a 'nuisance' "
            f"is a decision with a real trade-off, not a free cleanup.</div>")

    # Show the surviving structure on two remaining axes (PC2 vs PC3). Points are colored by event
    # attribute (aggression vs not), not by mouse rank.
    _fig = go.Figure()
    for _g, _c, _n in [(0, "#9aa0a6", "not agg"), (1, "#d62728", "aggression")]:
        _m = yagg == _g
        _fig.add_scattergl(x=_res[_m, 1], y=_res[_m, 2], mode="markers", name=_n,
                           marker=dict(size=4, opacity=0.5, color=_c))
    _fig.update_layout(template="plotly_white", height=420,
                       title="Surviving axes (PC2 vs PC3) still separate aggression",
                       xaxis_title="PC2", yaxis_title="PC3", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([drop_sel, mo.md(_msg), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · Every event as a point — including our example approach event

        Each event is now a point in PC space: from a skeleton (NB01), to 19 features (NB02), to a few
        PC scores. Below, first watch the example event as a GIF, then find the same event as the ★
        marker in the scatter. Pick the two axes to plot and color the points by despotism phase
        (`condition`) or by aggression.
        """
    )
    return


@app.cell
def _(EXAMPLE, cu, kp, mo, ranks):
    # Our example approach event as a rank-colored skeleton GIF: approacher = Dom (red),
    # approachee = Sub (green), bystander = Mid (blue). This is the same interaction that becomes a
    # single point in the scatter below.
    _gif = cu.event_gif_bytes(kp[EXAMPLE], ranks[EXAMPLE], contact_rel=40, cell=240)
    mo.md(
        "<b>Our example approach event (index 909).</b> Approacher = Dom (red), approachee = Sub "
        "(green), bystander = Mid (blue). The white arrow points approacher → approachee; the red dot "
        "marks contact.<br>" + cu.gif_img_html(_gif, width=260)
    )
    return


@app.cell
def _(mo):
    px_pc = mo.ui.dropdown(options=[f"PC{i+1}" for i in range(6)], value="PC1", label="x axis")
    py_pc = mo.ui.dropdown(options=[f"PC{i+1}" for i in range(6)], value="PC2", label="y axis")
    proj_color = mo.ui.dropdown(options=["condition", "aggression"], value="condition",
                                label="color by")
    return proj_color, px_pc, py_pc


@app.cell
def _(EXAMPLE, cond, go, mo, proj_color, px_pc, py_pc, sc, yagg):
    _ix = int(px_pc.value[2:]) - 1
    _iy = int(py_pc.value[2:]) - 1
    _x, _y = sc[:, _ix], sc[:, _iy]
    _fig = go.Figure()
    if proj_color.value == "condition":
        _pal = {"pre": "#54a24b", "dep": "#e45756", "post": "#4c78a8"}
        for _c in ["pre", "dep", "post"]:
            _m = cond == _c
            _fig.add_scattergl(x=_x[_m], y=_y[_m], mode="markers", name=_c,
                               marker=dict(size=4, opacity=0.5, color=_pal[_c]))
    else:
        for _g, _col, _n in [(0, "#9aa0a6", "not agg"), (1, "#d62728", "aggression")]:
            _m = yagg == _g
            _fig.add_scattergl(x=_x[_m], y=_y[_m], mode="markers", name=_n,
                               marker=dict(size=4, opacity=0.5, color=_col))
    _fig.add_scatter(x=[_x[EXAMPLE]], y=[_y[EXAMPLE]], mode="markers+text", name="example #909",
                     marker=dict(size=18, color="#111", symbol="star"),
                     text=["#909"], textposition="top center")
    _fig.update_layout(template="plotly_white", height=520,
                       title=f"Events in PC space — {px_pc.value} vs {py_pc.value}",
                       xaxis_title=px_pc.value, yaxis_title=py_pc.value,
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([px_pc, py_pc, proj_color], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6 · Exercise — does food deprivation move behavior along the activity axis?

        ### Why this exercise
        The despotism experiment food-deprives a cage (`condition == "dep"`). If hunger simply makes
        mice move more, that motion should show up as a shift along PC1 — the overall-activity axis.
        You will test that, but at **two levels**, because *how* you count the data changes the
        answer.

        ### Definitions
        - **Event level** — treat all 1500 events as independent samples and compare dep vs non-dep.
        - **Cage level** — first collapse to **one PC1 mean per cage per phase** (7 cages), then
          compare. This respects the fact that events from the same cage are not independent.
        - **Mann-Whitney U test** — `scipy.stats.mannwhitneyu(a, b, alternative="two-sided")` returns
          `(U, p)`; it asks whether two groups differ in their typical value without assuming a
          bell-shaped distribution. A small `p` (< 0.05) means the two groups differ.
        - **Pseudoreplication** — treating measurements that are not independent (many events from a
          few cages) as if they were, which can make a weak effect look strong. This exercise is a
          first look at that problem; NB06 is devoted to it.

        ### Toolbox
        - `sc[:, 0]` **(1500,)** — PC1 score for every event (the activity axis).
        - `cond` **(1500,)** — `'pre' | 'dep' | 'post'` (**dep** = food-deprived).
        - `cage` **(1500,)** — cage id 9–15 (the true experimental *unit*).

        ### What to do
        The function `dep_shift` already computes both p-values. It runs on load so the page renders.
        There is **one line to fill in** (marked with `# TODO`): replace the blank with the mask for
        the *non-deprived* events. Everything else is done for you.
        """
    )
    return


@app.cell
def _(cage, cond, np, sc):
    from scipy.stats import mannwhitneyu as _mwu

    def dep_shift(pc_col=0):
        """Return (p_event, p_cage) for the dep-vs-rest position shift on PC `pc_col`.

        Fill-in-the-blank: the ONE line to edit is marked `# TODO`. It compares the deprived events
        s[_dep] against the non-deprived events. Replace ____ with ~_dep (the logical NOT of the
        dep mask, i.e. every event that is NOT deprived).
        """
        s = sc[:, pc_col]                                      # PC scores, one per event
        _dep = cond == "dep"                                   # boolean mask: deprived events

        # TODO(you): replace ____ with ~_dep so this compares dep vs non-dep events.
        #   p_event = float(_mwu(s[_dep], s[____], alternative="two-sided")[1])
        p_event = float(_mwu(s[_dep], s[~_dep], alternative="two-sided")[1])   # <- the filled line

        # Cage level (already done): one PC mean per cage among dep, and among non-dep, then compare.
        _cages = np.unique(cage)
        _dv = np.array([s[(cage == c) & _dep].mean() for c in _cages])
        _ov = np.array([s[(cage == c) & ~_dep].mean() for c in _cages])
        p_cage = float(_mwu(_dv, _ov, alternative="two-sided")[1])
        return p_event, p_cage

    p_event, p_cage = dep_shift(0)
    return p_cage, p_event


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Look at it: the two levels side by side

        The plot below shows the same test as a picture. **Left:** PC1 scores for every deprived vs
        every non-deprived event (all 1500). **Right:** one PC1 mean per cage per phase (7 dots each).

        What you should see: on the left, the two boxes are clearly offset, so at the event level the
        difference looks real. On the right, the per-cage dots overlap heavily — once you respect the
        cage as the unit, the shift mostly disappears.
        """
    )
    return


@app.cell
def _(cage, cond, go, mo, np, p_cage, p_event, sc):
    from plotly.subplots import make_subplots as _mksub
    _s = sc[:, 0]
    _dep = cond == "dep"
    _fig = _mksub(rows=1, cols=2, subplot_titles=(
        f"Event level (n=1500)   p = {p_event:.1e}",
        f"Cage level (n=7 cages)   p = {p_cage:.2f}"))
    # left: event-level distributions
    _fig.add_box(y=_s[_dep], name="dep", marker_color="#e45756", boxpoints=False, row=1, col=1)
    _fig.add_box(y=_s[~_dep], name="non-dep", marker_color="#9aa0a6", boxpoints=False, row=1, col=1)
    # right: one mean per cage per phase
    _cages = np.unique(cage)
    _dv = np.array([_s[(cage == c) & _dep].mean() for c in _cages])
    _ov = np.array([_s[(cage == c) & ~_dep].mean() for c in _cages])
    _fig.add_scatter(x=["dep"] * len(_dv), y=_dv, mode="markers", name="cage · dep",
                     marker=dict(color="#e45756", size=11, opacity=0.8), row=1, col=2)
    _fig.add_scatter(x=["non-dep"] * len(_ov), y=_ov, mode="markers", name="cage · non-dep",
                     marker=dict(color="#9aa0a6", size=11, opacity=0.8), row=1, col=2)
    _fig.update_layout(template="plotly_white", height=420, showlegend=False,
                       title="PC1 (activity axis): dep vs non-dep at two levels",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_yaxes(title_text="PC1 score", showgrid=False, row=1, col=1)
    _fig.update_yaxes(showgrid=False, row=1, col=2)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// details | Reveal solution
        ```python
        from scipy.stats import mannwhitneyu
        s = sc[:, 0]                       # PC1 = activity axis
        dep = cond == "dep"
        # 1. event level — treats every event as independent
        p_event = mannwhitneyu(s[dep], s[~dep], alternative="two-sided")[1]
        # 2. cage level — one summary value per cage per phase (7 vs 7)
        cages = np.unique(cage)
        dv = [s[(cage == c) &  dep].mean() for c in cages]
        ov = [s[(cage == c) & ~dep].mean() for c in cages]
        p_cage = mannwhitneyu(dv, ov, alternative="two-sided")[1]
        ```
        The event-level test is strongly significant; the cage-level test is not. Same data, two
        honest answers — the effect is carried by a few cages, not by 1500 independent trials. That
        gap is the subject of NB06.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(cum6, dim90, mo, p_cage, p_event):
    _c1 = 10 <= dim90 <= 12
    _c2 = p_event < 0.05
    _c3 = p_cage > 0.05
    _c4 = 0.66 <= cum6 <= 0.75
    _ok = _c1 and _c2 and _c3 and _c4
    def _row(ok, txt):
        return f"<div>{'✅' if ok else '❌'} {txt}</div>"
    _body = (
        _row(_c4, f"6-PC variance = {cum6*100:.1f}% (band 66–75%)")
        + _row(_c1, f"90% dimension = {dim90} PCs (band 10–12)")
        + _row(_c2, f"event-level dep shift IS significant: p = {p_event:.3f} (&lt; 0.05)")
        + _row(_c3, f"cage-level dep shift is NOT significant: p = {p_cage:.3f} (&gt; 0.05)")
    )
    _verdict = ("PASS. Note the honest conclusion: food deprivation <b>does</b> shift PC1 at the event "
                "level, but that shift <b>does not survive</b> aggregating to the cage. You cannot yet "
                "claim hunger moves behavior — the unit is the cage, not the event (NB06)."
                if _ok else "Something is off — recheck the bands above.")
    _bg = "#eafaef" if _ok else "#fdeaea"
    mo.md(f"<div style='border:1px solid #ccc;border-radius:8px;padding:10px 12px;background:{_bg}'>"
          f"<b>Self-check</b>{_body}<div style='margin-top:6px'>{_verdict}</div></div>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Conceptual questions
        1. **Why can a low-variance component be the decodable one?** Aggression is uncommon (about
           30% of these already-filtered events, and far rarer in the wild). A behavior that is rare
           but sharply patterned can live in a small-variance direction that PCA ranks near the bottom,
           yet a classifier reads it well. Variance order is not the same as usefulness order.
        2. **Which component would you call the nuisance, and why is that a choice?** Dropping PC1
           cleaned up the "just activity" axis *and* cost you aggression AUROC. There is no neutral
           rule that labels an axis a nuisance; it depends on the question you are about to ask.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Limits of this method

        - **PCA is linear and variance-greedy.** It can only rotate and stretch the data — it cannot
          unbend a curved shape. A behavior that curls around in feature space gets smeared across
          many PCs. Here, aggression is partly captured by PC1 but never cleanly isolated by any single
          component, which is why we need a nonlinear map next.
        - **Standardization changes the answer.** Z-scoring gave every feature an equal voice. In raw
          units, `triangle_area` (thousands of px²) would have dwarfed `facing_cosine` (between -1 and
          1), and PC1 would mean something else entirely. The "principal" axis depends on your scaling
          choice.
        - **A rare behavior can hide in a small component.** Keep only 6 PCs "because 90% variance" and
          you may have discarded the exact low-variance direction a decoder needed.

        **Where this points.** If a behavior of interest lives in a low-variance direction, PCA's
        "keep the big axes" instinct works against you. The next notebook uses a nonlinear embedding
        (UMAP) that preserves local neighborhoods instead of maximizing variance, and asks whether
        distinct behaviors separate on their own.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we ship next

        You compressed 19 features into about 6 informative axes, named PC1 as the activity axis, and
        confirmed that setting it aside is a choice with a cost — aggression survives but weakens. You
        also saw food deprivation shift PC1 at the event level and vanish at the cage level, a first
        look at the pseudoreplication problem in NB06.

        > **Next → NB05.** Linear axes cannot unfold a curved shape. We flatten the cloud with UMAP and
        > divide it into discrete **behavioral syllables**, then ask, without any labels, whether
        > aggression separates out on its own.
        """
    )
    return


@app.cell
def _(board_html, cum6, dim90):
    board_html("bottom", dim90, cum6)
    return


if __name__ == "__main__":
    app.run()

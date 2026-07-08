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
    # --- Load the corpus (features + metadata live in the derived bundle; labels in events) ---
    ev  = cu.load_events(cu.data_path("data/train_events.npz", ROOT))
    der = cu.load_derived("train", ROOT)
    hod = cu.load_derived("heldout", ROOT)                 # sealed Cage 16 (count only, for now)

    Xz, _mu, _sd = cu.standardize(der["X"])                # z-score the 19 features
    # Refit a FULL-rank PCA so the scree curve can reach the 90% mark. (The shipped der['pca_scores']
    # keeps only 10 components and caps at 0.884 cumulative variance — not enough to read 90% off.)
    sc, evr, _pca = cu.pca_scores(Xz, 19)                  # sc (1500,19), evr (19,)
    comp = _pca.components_                                 # (19,19) loadings
    fn   = [str(f) for f in der["feature_names"]]
    cage = der["cage"]
    cond = ev["condition"].astype(str)
    yagg = ev["agg_label"].astype(int)

    cumvar = np.cumsum(evr)
    dim90  = int(np.searchsorted(cumvar, 0.90) + 1)        # smallest k with >=90% variance
    cum6   = float(cumvar[5])                               # variance kept by the first 6 PCs
    n_ho   = int(len(hod["cage"]))                          # 470 sealed events

    HERO = 909            # cage-15 male aggression event (the clean hero; #742 in the design is cage-12/non-agg)
    return HERO, cage, comp, cond, cum6, cumvar, dim90, evr, fn, n_ho, sc, yagg


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB04 · The Collapse I — PCA and the dimensionality of behavior

        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > Before you show us clusters, answer one thing: **is a single boring axis — *how close, how
        > fast* — dominating everything you measure?** If our optogenetic readout is really just
        > "the mice moved more," we can't claim we flipped a *social* switch. Compress your 19
        > features down to their honest shape, tell us which axis is the nuisance, and prove that
        > setting it aside does **not** throw away the aggression signal we came for.
        >
        > **Today's lab-meeting question:** *How many dimensions does mouse social behavior actually
        > have — and does hunger move it along one of them?*

        You already turned 11,700 raw pose numbers into 19 interpretable features (NB02). But those
        19 are **not independent**: closing speed, pair distance, and mutual facing rise and fall
        together. Today you find the few underlying axes the 19 features are really tracing — the
        **behavioral manifold** — and you meet **residualization** not as a fact but as a *choice*.

        > **Neuroscience connection.** Finding that a 19-dimensional signal secretly lives on a
        > 6-dimensional shape is the exact discovery Stephens made for the worm's posture and
        > Cunningham & Yu formalized for cortex: population activity that *looks* high-dimensional
        > traces a low-dimensional manifold. You are about to do it for behavior.
        """
    )
    return


@app.cell
def _(mo):
    def board_html(where, dim90, cum6):
        # Gauge A = "size of the representation" (falls through Phase 1). Gauge B = "held-out
        # readiness" (rises through Phase 2 — still dormant here in Discover). Benchmarks are the
        # pinned readout_board.csv values; the student's freshly-computed number sits beside them.
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
                  f"the representation</b> &nbsp;<span style='color:#888'>(smaller = more distilled)</span></div>"
                  f"<table style='border-collapse:separate;border-spacing:6px'><tr>{cells}</tr></table>")
        gaugeB = ("<div style='margin-top:10px;font-size:.85em;color:#444'><b>Gauge B — held-out "
                  "readiness</b>: <span style='color:#b00'>dormant</span> — rises in Phase 2 "
                  "(target: Cage-16 decode AUROC <b>0.86</b>, unlocked in NB08).</div>")
        note = (f"<div style='margin-top:8px;font-size:.8em;color:#666'>Your run: <b>6 PCs → "
                f"{cum6*100:.1f}%</b> of variance (benchmark: 6 PCs ≈ 71%). Reaching <b>90%</b> "
                f"needs <b>{dim90}</b> PCs — these are <i>different kinds</i> of reduction, not one "
                f"magic number.</div>")
        title = "READOUT BOARD" + ("" if where == "top" else " — end of NB04")
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
        <div style="border:2px dashed #b00;border-radius:10px;padding:10px 14px;background:#fff6f6">
        <b>SEALED · Cage 16</b> &nbsp; <span style="color:#b00">🔒 redacted</span><br>
        <span style="font-family:monospace">n = {n_ho} events · skeletons ▓▓▓ greyed · labels ██████ blacked out</span><br>
        <span style="color:#666;font-size:.85em">The animal on the rig. Its PCA scores exist but stay
        hidden — a decoder is only trustworthy if it survives a cage it never trained on.
        <b>Notebooks until unlock: 4.</b></span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · The idea, by hand: find the axis of most spread

        PCA sounds like linear algebra. It is really one stubborn question: **if I could keep only a
        single direction through this cloud of points, which direction preserves the most of the
        spread?** Everything else — eigenvectors, covariance — is just the machine that answers it
        automatically.

        Below are two of the real features (standardized). They are correlated, so the cloud is a
        tilted ellipse. **Drag the angle** to rotate a candidate axis; the readout shows the
        *variance of the points projected onto it*. Find the maximum by hand. Then flip **Reveal
        PCA's axis** — the first principal component is exactly the angle you were hunting for.
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
                     line=dict(color="#e45756", width=3), name="your axis")
    if toy_reveal.value:
        _v = _pc1 / np.linalg.norm(_pc1)
        _fig.add_scatter(x=[-_L*_v[0], _L*_v[0]], y=[-_L*_v[1], _L*_v[1]], mode="lines",
                         line=dict(color="#2ca02c", width=3, dash="dash"),
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

        /// details | Deeper: the eigen-math you just skipped (optional)
        Standardize the data to $X$ ($n$ events × 19 features, each column mean-0). The covariance is
        $C=\tfrac1n X^\top X$. PCA solves the eigenproblem $C v = \lambda v$: each **eigenvector**
        $v_k$ is a principal direction, and its **eigenvalue** $\lambda_k$ is the variance captured
        along it. The scores are the projections $Z = X V$. "Maximize projected variance over all
        unit directions" and "take the top eigenvector of the covariance" are the *same* statement —
        the slider above was doing gradient-free eigen-decomposition by hand. `explained_variance_ratio`
        is just $\lambda_k / \sum_j \lambda_j$.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · Scree & cumulative variance — how many axes do we really need?

        Run PCA on all 19 standardized features and plot how much variance each component keeps
        (bars) and how it accumulates (line). **Drag `keep k`** to see how much you retain. The
        elbow is soft, but the story is clear: a handful of axes carry most of the signal. This is
        where **Gauge A falls to ~6**.
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
        **Read it honestly.** The first **6 PCs keep {cum6*100:.1f}%** of the variance — that is the
        "~6" on the board, and it is plenty for a *map*. But hitting a strict **90% needs {dim90}
        PCs**. Behavior here is genuinely ~6–{dim90}-dimensional, not 19 and not 1. "Dimensionality"
        is a slider, not a single truth.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · The loadings — reading PCs as "eigen-behaviors"

        A principal component is a *recipe*: a weighted blend of the 19 features. The heatmap below
        shows those weights for the top components (red = pushes the score up, blue = down). Read
        each row as a **behavioral syndrome**.
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
        **Name PC1.** Its heaviest loadings are {_top} — mean speeds, angular velocity, and mutual
        facing. That is one coherent thing: **overall motion & mutual-engagement magnitude** — *how
        much is happening* between the two mice. It is also, unsurprisingly, the **highest-variance**
        axis. The obvious move is to call it a nuisance ("just activity") and delete it. Hold that
        thought — the next section shows why that is a *choice*, not a free lunch.

        > This is the behavioral twin of a **tuning curve**: PC1 is a direction in feature space that
        > a population of "neurons" (features) covaries along, exactly as motor cortex covaries along
        > a preferred-direction axis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🔬 Deeper: the behavioral manifold — its papers, and where the analogy stops": mo.md(
                r"""
                **Shared mathematics.** "A high-dimensional signal secretly lives on a
                low-dimensional shape" is one of systems neuroscience's most reproduced findings.
                Stephens et al. (2008, *PLoS Comput. Biol.* 4:e1000028) showed the crawling
                **worm's** posture — a whole body's worth of angles — collapses onto ~4
                "eigenworms." Cunningham & Yu (2014, *Nat. Neurosci.* 17:1500) formalized the same
                move for **cortex**: population firing traces a low-dimensional **neural manifold**,
                and Gallego et al. (2017, *Neuron* 94:978; 2020, *Nat. Neurosci.* 23:260) showed
                those manifold axes are stable enough to decode across days. PC1 behaving as a
                covariation axis is the linear-algebra twin of a motor-cortex
                **preferred-direction / tuning-curve** axis (Churchland & Shenoy).

                **Same operation.** Their PCA and yours are the *identical* eigendecomposition of a
                covariance matrix — the top eigenvectors are the directions of most variance. The
                angle slider in §1 was doing that eigendecomposition by hand.

                **Where the analogy stops.** Shared **geometry is not shared biology**. Our axes are
                *designed* features of two mice; theirs are *measured* neurons. A matching
                low-dimensional shape is a **mathematical rhyme, not an identity** — and PCA is
                linear, so a genuinely *curved* behavioral manifold (the next notebook) will defeat
                it no matter how clean the scree plot looks.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · Residualization is a *choice*, not a fact

        For clustering (NB05) we often **drop** the big "how-fast/how-close" axis so the map reflects
        *finer* structure instead of raw activity. `cu.residualize` does this by zeroing chosen PC
        columns. But watch what it costs: **PC1 carries a lot of the aggression signal** (aggression
        *is* partly a high-motion behavior). Toggle which PCs to drop and read the 5-fold aggression
        AUROC on the surviving axes.
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
            f"<b>weakens but does not erase</b> the signal — that is the whole point: 'nuisance' is a "
            f"decision with a real trade-off, not a free cleanup.</div>")

    # visualize the residual structure on two surviving axes (PC2 vs PC3)
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
        ## 5 · The whole corpus, projected — and Hero Event #909

        Every event is now a point in PC space. Pick the axes, color by `condition` (the despotism
        phase) or `aggression`, and find the **★ hero** — the clean Cage-15 male aggression event we
        have followed since NB01 (index **909**; the design's "#742" is actually a non-aggression
        cage-12 event, so we use 909). PCA has re-rendered it one more way: from a skeleton, to 19
        features, to a single dot on the behavioral manifold.
        """
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
def _(HERO, cond, go, mo, proj_color, px_pc, py_pc, sc, yagg):
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
    _fig.add_scatter(x=[_x[HERO]], y=[_y[HERO]], mode="markers+text", name="Hero #909",
                     marker=dict(size=18, color="#111", symbol="star"),
                     text=["#909"], textposition="top center")
    _fig.update_layout(template="plotly_white", height=520,
                       title=f"Behavioral manifold — {px_pc.value} vs {py_pc.value}",
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
        ## 6 · Exercise — does hunger move behavior along the activity axis?

        ### Toolbox
        - `sc` **(1500, 19)** — PCA scores; column `0` is **PC1** (the motion-magnitude axis).
        - `cond` **(1500,)** — `'pre' | 'dep' | 'post'` (the despotism phase; **dep** = food-deprived).
        - `cage` **(1500,)** — cage id 9–15 (the true experimental *unit*).
        - `scipy.stats.mannwhitneyu(a, b, alternative="two-sided")` → `(U, p)`.

        > **Hypothesis (pre-registered):** *Food-deprived (dep) events sit at a different position on
        > PC1 — the overall-motion axis — than pre/post events.*

        Test it two ways and compare:
        1. **Event level** — all 1500 events, dep vs (pre+post), Mann-Whitney on `sc[:,0]`.
        2. **Cage level** — collapse to **one PC1 mean per cage per phase**, then dep vs non-dep
           across the 7 cages. This previews **pseudoreplication**: 1500 events from 7 cages are not
           1500 independent samples.

        Fill in the stub, then check yourself. (A reference implementation runs on load so the page
        renders; replace the body with your own.)
        """
    )
    return


@app.cell
def _(cage, cond, np, sc):
    from scipy.stats import mannwhitneyu as _mwu

    def dep_shift(pc_col=0):
        """Return (p_event, p_cage) for the dep-vs-rest position shift on PC `pc_col`.

        TODO(student): replace this reference body with your own.
          1. s = sc[:, pc_col]
          2. p_event: mannwhitneyu(s[dep], s[not dep], two-sided)
          3. p_cage : for each cage, mean(s) among dep and among non-dep -> two arrays of length 7,
                      then mannwhitneyu across cages.
        """
        s = sc[:, pc_col]
        _dep = cond == "dep"
        p_event = float(_mwu(s[_dep], s[~_dep], alternative="two-sided")[1])
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
        /// details | Reveal solution
        ```python
        from scipy.stats import mannwhitneyu
        s = sc[:, 0]                       # PC1 = motion-magnitude axis
        dep = cond == "dep"
        # 1. event level — treats every event as independent
        p_event = mannwhitneyu(s[dep], s[~dep], alternative="two-sided")[1]
        # 2. cage level — one summary value per cage per phase (7 vs 7)
        cages = np.unique(cage)
        dv = [s[(cage == c) &  dep].mean() for c in cages]
        ov = [s[(cage == c) & ~dep].mean() for c in cages]
        p_cage = mannwhitneyu(dv, ov, alternative="two-sided")[1]
        ```
        The event-level test screams significance; the cage-level test shrugs. Same data, honest
        answer — the effect is carried by a *few cages*, not 1500 independent trials. That gap is the
        whole subject of NB06.
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
    _verdict = ("PASS — and note the honest conclusion: dep <b>does</b> shift PC1 at the event level, "
                "but that shift <b>does not survive</b> aggregating to the cage. You cannot yet claim "
                "hunger moves behavior — the units are cages, not events (NB06)."
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
        1. **Why can a *low-variance* PC be the *decodable* one?** Aggression is rare (~30% of these
           already-filtered events, far rarer in the wild). A behavior that is uncommon but sharply
           patterned can live in a tiny-variance direction that PCA ranks near the bottom — yet a
           classifier reads it perfectly. Variance-order ≠ usefulness-order.
        2. **Which PC would *you* call "nuisance," and why is that a choice?** Dropping PC1 cleaned up
           the "just activity" axis *and* cost you aggression AUROC. There is no view-from-nowhere
           that labels an axis nuisance; it depends on the question you are about to ask.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we threw away · how it breaks

        - **PCA is linear and variance-greedy.** It can only rotate and stretch — it cannot *unbend*
          a curved manifold. A behavior that curls around in feature space gets smeared across many
          PCs. On this data, aggression is *partly* captured by PC1 but never cleanly isolated by any
          single component (that is why we need the nonlinear map next).
        - **Standardization changes the winner.** Z-scoring made every feature equal-voiced; in raw
          units, `triangle_area` (thousands of px²) would have swamped `facing_cosine` (∈[-1,1]) and
          PC1 would mean something else entirely. The "principal" axis is a modeling artifact of your
          scaling choice.
        - **A rare behavior hides in a small PC.** Reduce to 6 PCs "because 90% variance" and you may
          have discarded the exact low-variance direction your decoder needed.

        **How would you analyze this?** *If aggression lives in a low-variance direction, PCA's
        "keep the big axes" instinct is actively working against you. What would you do instead —
        and how would a **nonlinear** embedding that preserves local neighborhoods (next notebook)
        change the answer?*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we ship next

        You compressed 19 features into **~6 honest axes**, named PC1 as the activity/nuisance axis,
        and proved that setting it aside is a **choice with a cost** — aggression survives but pays.
        You also saw hunger nudge PC1 at the event level and vanish at the cage level: the first
        tremor of the pseudoreplication reckoning in NB06.

        > **Neuroscience connection (close).** What you drew is a **behavioral manifold** — the same
        > low-dimensional shape Cunningham & Yu and Gallego find when a cortical population's activity
        > collapses onto a few axes. *Where the analogy stops:* shared **geometry** is not shared
        > **biology** — our axes are designed features, theirs are measured neurons; a matching shape
        > is a mathematical rhyme, not an identity.

        > **Next → NB05, The Collapse II.** Linear axes can't unfold a curved manifold. We lay the
        > cloud flat with UMAP and carve it into discrete **behavioral syllables** — and ask, without
        > any labels, whether aggression falls out on its own.
        """
    )
    return


@app.cell
def _(board_html, cum6, dim90):
    board_html("bottom", dim90, cum6)
    return


if __name__ == "__main__":
    app.run()

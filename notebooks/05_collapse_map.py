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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB05 · The Collapse II — the Map

        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > Yesterday you compressed 19 features down to a handful of principal axes. Good — but PC1
        > and PC2 are still *our* axes. Today we want the **catalog of what these mice actually DO**,
        > discovered from the data, not from our assumptions. Lay the behavior out as a flat map,
        > carve it into recurring **syllables**, and hand us back the ethogram.
        >
        > **The one deliverable:** a 2-D behavioral map + a discrete set of data-driven syllables
        > (canonical cluster labels the rest of the week will agree on).
        >
        > **The circuit question it unblocks:** before we point a laser at VMHvl, we need to know
        > whether *aggression is even a natural kind* in these mice — a mode the animals fall into on
        > their own — or just a label we impose. If unsupervised clustering rediscovers it, the state
        > is real enough to gate an experiment on.
        >
        > **Today's lab-meeting question:** *Can we find aggression without ever being told which
        > events are aggressive?*

        This is Phase 1's climax: social life becomes a **landscape you can point at**. Embedding a
        cloud of behavior and carving it into modules is the exact computational-ethology move behind
        *behavioral syllables* — and it is step-for-step the move a neuroscientist makes when a
        population of neurons turns out to live on a low-dimensional manifold you can cluster into
        states.
        """
    )
    return


@app.cell
def _(ROOT, cu):
    # ---- canonical data loads (no raw np.load; the loaders fetch on a bare kernel) ----------------
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))   # kp, ranks, category, agg_label
    der = cu.load_derived("train", ROOT)                               # cage/sex/tod_hour/X/PCA live HERE
    sweep = cu.load_umap_sweep(ROOT)                                   # emb_grid + CANONICAL default_labels
    ho = cu.load_events(cu.data_path("data/heldout_events.npz", ROOT)) # Cage 16 — still sealed
    return der, ev, ho, sweep


@app.cell
def _(ROOT, cu, ev, sweep):
    # ---- the canonical map + syllables (every NB05-07 beat agrees on THESE) -----------------------
    di, dj = int(sweep["default_ij"][0]), int(sweep["default_ij"][1])
    emb0 = sweep["emb_grid"][di, dj]                     # the pinned default 2-D embedding (1500,2)
    clabels = sweep["default_labels"].astype(int)       # CANONICAL syllables: C0..C4 + noise(-1)
    agg = ev["agg_label"].astype(int)
    base_rate = float(agg.mean())                       # 0.300
    HERO = 909                                           # cage-15 / male / aggression (design's #742
    #                                                      is a cage-12 non-aggression event — replaced)

    # per-cluster aggression fraction -> the PUREST (highest-fraction) cluster and its lift
    _cs = sorted(c for c in set(clabels.tolist()) if c >= 0)
    _fr = {c: float(agg[clabels == c].mean()) for c in _cs}
    best_cluster = max(_fr, key=_fr.get)
    best_frac = _fr[best_cluster]
    best_lift = best_frac / base_rate
    # committed benchmark from the Readout Board
    board = {}
    try:
        import csv
        with open(cu.data_path("data/readout_board.csv", ROOT)) as _f:
            for _r in csv.DictReader(_f):
                board[(_r["gauge"], _r["stage"])] = _r["value"]
    except Exception:
        board = {}
    return HERO, agg, base_rate, best_cluster, best_frac, best_lift, board, clabels, di, dj, emb0


@app.cell
def _(mo):
    # ---- Readout Board renderer (used top AND bottom; degrades gracefully) ------------------------
    def make_board(board, base_rate, best_lift, best_frac):
        def _b(key, default):
            try:
                return f"{float(board.get(key, default)):g}"
            except Exception:
                return str(default)
        a_chain = [
            ("raw pose", _b(("A", "raw pose per event"), 11700)),
            ("features", _b(("A", "allocentric features"), 19)),
            ("PCs", _b(("A", "principal components"), 6)),
            ("2-D map", _b(("A", "behavioral map"), 2)),
            ("1 syllable", _b(("A", "one syllable"), 1)),
        ]
        steps = " &nbsp;→&nbsp; ".join(
            (f"<b style='color:#2563eb'>{v}</b>" if lbl in ("2-D map", "1 syllable") else f"{v}")
            + f"<br><span style='font-size:11px;color:#888'>{lbl}</span>"
            for lbl, v in a_chain)
        bench_lift = _b(("B", "best aggression cluster lift"), 1.4)
        return f"""
<div style="border:1px solid #d5d8de;border-radius:10px;padding:12px 16px;margin:6px 0;
            background:#fafbfc;font-family:'Liberation Sans',sans-serif">
  <div style="font-weight:700;font-size:15px;margin-bottom:6px">Readout Board</div>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="flex:2;min-width:280px">
      <div style="font-size:13px;color:#555;margin-bottom:4px">
        <b>Gauge A · size of the representation</b> (this notebook lands the map)</div>
      <div style="font-size:15px;line-height:1.9">{steps}</div>
      <div style="font-size:11px;color:#999;margin-top:4px">
        different <i>kinds</i> of reduction, not one magic number</div>
    </div>
    <div style="flex:1;min-width:200px;border-left:1px solid #e2e5ea;padding-left:16px">
      <div style="font-size:13px;color:#555;margin-bottom:4px">
        <b>Gauge B · held-out readiness</b></div>
      <div style="font-size:13px">best aggression-cluster lift</div>
      <div style="font-size:24px;font-weight:700;color:#2563eb">{best_lift:.2f}x
        <span style="font-size:13px;color:#888;font-weight:400">
          (you) &nbsp;vs&nbsp; {bench_lift}x benchmark</span></div>
      <div style="font-size:11px;color:#999">
        purest syllable is {best_frac:.0%} aggression vs {base_rate:.0%} base rate</div>
    </div>
  </div>
</div>"""
    return (make_board,)


@app.cell(hide_code=True)
def _(base_rate, best_frac, best_lift, board, make_board, mo):
    mo.md(make_board(board, base_rate, best_lift, best_frac))
    return


@app.cell(hide_code=True)
def _(ho, mo):
    # ---- Sealed Cage 16 (stays redacted until NB08; counter = 3) -----------------------------------
    _n = len(ho["kp"])
    mo.md(
        f"""
<div style="border:2px dashed #b02a37;border-radius:10px;padding:12px 16px;margin:6px 0;
            background:repeating-linear-gradient(45deg,#fbf0f1,#fbf0f1 12px,#f7e6e8 12px,#f7e6e8 24px);
            font-family:'Liberation Sans',sans-serif">
  <span style="font-weight:700;color:#b02a37">SEALED · Camera 16 — the animal on the rig</span>
  &nbsp;<span style="background:#b02a37;color:#b02a37;border-radius:3px">bnnn nnnn nnnnnn</span><br>
  <span style="font-size:13px;color:#555">Events on file: <b>{_n}</b> &nbsp;·&nbsp;
  skeletons <span style="filter:blur(3px);color:#999">greyed</span> &nbsp;·&nbsp;
  labels <span style="background:#333;color:#333">redacted--</span></span><br>
  <span style="font-size:12px;color:#b02a37">Unlocks in <b>3 notebooks</b> (NB08). The map you build
  today never sees Cage 16 — that is the whole point of the held-out bet.</span>
</div>"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Why the map is precomputed (and live UMAP is banned here)

        UMAP builds a fuzzy nearest-neighbor graph in feature space and lays it out in 2-D so that
        *neighbors stay neighbors*. It is exactly the right tool — and it is **poison on a cloud
        kernel**. A cold molab kernel spends **~28 s** compiling UMAP's numba kernels on the first
        call; the notebook's websocket times out and the app hangs before a single point is drawn.
        Dragging a slider that re-runs it would recompile over and over.

        So the engine **hard-guards `run_umap`** (it raises) and instead ships a **5x5 precomputed
        sweep** — the same 1500 events embedded across every `n_neighbors x min_dist` combination.
        We *select* a cell; we never recompute. The **one** live compute we allow is HDBSCAN, which
        is fast (~0.1 s).

        Below: the full sweep as small-multiples, each panel colored red where the event is
        aggression. Read rows (locality knob) and columns (packing knob) to feel what the two knobs
        do — **by selection, not by recompute.**
        """
    )
    return


@app.cell
def _(agg, cu, sweep):
    cu.sweep_grid_fig(
        sweep["emb_grid"], sweep["nn_values"], sweep["md_values"],
        color_key=agg, palette={0: "#b6bac1", 1: "#d62728"},
        names={0: "not agg", 1: "aggression"}, height=680)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · Select a cell from the sweep

        - **`n_neighbors`** — small = local detail, many islands; large = global shape, one continent.
        - **`min_dist`** — how tightly points may pack (small = tight, well-separated blobs).

        Pick any cell; the map redraws instantly from the precomputed grid, colored by the
        **canonical syllables** (`sweep["default_labels"]`) that every downstream notebook shares.
        The default cell (`n_neighbors=15, min_dist=0.0`) is the pinned canon.
        """
    )
    return


@app.cell
def _(mo, sweep):
    nn_pick = mo.ui.dropdown(options={f"{v}": i for i, v in enumerate(sweep["nn_values"])},
                             value=str(int(sweep["nn_values"][int(sweep["default_ij"][0])])),
                             label="n_neighbors")
    md_pick = mo.ui.dropdown(options={f"{v:g}": j for j, v in enumerate(sweep["md_values"])},
                             value=f"{float(sweep['md_values'][int(sweep['default_ij'][1])]):g}",
                             label="min_dist")
    return md_pick, nn_pick


@app.cell
def _(HERO, clabels, go, md_pick, mo, nn_pick, sweep):
    _i, _j = int(nn_pick.value), int(md_pick.value)
    _emb = sweep["emb_grid"][_i, _j]
    _fig = go.Figure()
    for _c in sorted(set(clabels.tolist())):
        _m = clabels == _c
        _nm = "noise" if _c < 0 else f"C{_c}"
        _col = "#cfd2d8" if _c < 0 else None
        _fig.add_scattergl(x=_emb[_m, 0], y=_emb[_m, 1], mode="markers", name=_nm,
                           marker=dict(size=5, opacity=0.7, color=_col))
    _fig.add_scatter(x=[_emb[HERO, 0]], y=[_emb[HERO, 1]], mode="markers", name="Hero #909",
                     marker=dict(symbol="star", size=17, color="#f5b400",
                                 line=dict(color="#333", width=1)))
    _fig.update_layout(template="plotly_white", height=520,
                       title=(f"Selected cell — n_neighbors={int(sweep['nn_values'][_i])}, "
                              f"min_dist={float(sweep['md_values'][_j]):g} — colored by canonical syllable"),
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([nn_pick, md_pick], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · Carve it — HDBSCAN (the one live compute)

        HDBSCAN groups dense regions and labels sparse points as **noise (-1)** instead of forcing
        every point into a cluster — no *k* to guess. **`min_cluster_size`** sets the smallest group
        you will accept. This runs live on the default embedding as you drag (it is fast). At the
        canonical `min_cluster_size = 15` it reproduces `default_labels` **exactly** — larger values
        merge syllables, smaller ones shatter the blob into noise.
        """
    )
    return


@app.cell
def _(mo):
    mcs = mo.ui.slider(8, 80, value=15, step=1, label="min_cluster_size (live)",
                       debounce=True, full_width=True)
    return (mcs,)


@app.cell
def _(cu, emb0, go, mcs, mo):
    _lab = cu.run_hdbscan(emb0, min_cluster_size=int(mcs.value))
    _nc = len([c for c in set(_lab.tolist()) if c >= 0])
    _noise = float((_lab == -1).mean())
    _fig = go.Figure()
    for _c in sorted(set(_lab.tolist())):
        _m = _lab == _c
        _fig.add_scattergl(x=emb0[_m, 0], y=emb0[_m, 1], mode="markers",
                           name=("noise" if _c < 0 else f"C{_c}"),
                           marker=dict(size=5, opacity=0.7,
                                       color=("#cfd2d8" if _c < 0 else None)))
    _fig.update_layout(template="plotly_white", height=470,
                       title=f"Live HDBSCAN — {_nc} clusters · {_noise:.0%} noise "
                             f"(min_cluster_size={int(mcs.value)})",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mcs, _fig,
               mo.md("*Drag to 15 to land the canon (5 clusters, ~15% noise). "
                     "This slider is **exploration only** — the shared syllables stay fixed at 15.*")])
    return


@app.cell(hide_code=True)
def _(best_cluster, clabels, mo):
    _sizes = {c: int((clabels == c).sum()) for c in sorted(set(clabels.tolist())) if c >= 0}
    mo.md(
        f"""
        ## 4 · The syllables, and where the Hero lives

        The canon carves out **{len(_sizes)} syllables** plus noise: sizes {_sizes}. The purest
        aggression pocket is **C{best_cluster}** — and **Hero Event #909** (a real cage-15 male
        aggression bout) lands right inside it. The single event we have followed since NB01 is now a
        *point on a data-driven map*, sitting in the cluster the data itself decided was its kind.
        Recolor by category below and render that cluster's exemplars to eyeball whether the machine's
        pocket really *looks* like fighting.
        """
    )
    return


@app.cell
def _(HERO, emb0, ev, go, np):
    # canonical map colored by REGISTRY category (agg / other-labeled / unlabeled) + Hero star
    _cat = ev["category"].astype(str)
    _is_agg = _cat == "aggression"
    _is_unl = _cat == ""
    _other = ~_is_agg & ~_is_unl
    _fig = go.Figure()
    _fig.add_scattergl(x=emb0[_is_unl, 0], y=emb0[_is_unl, 1], mode="markers", name="unlabeled",
                       marker=dict(size=4, opacity=0.5, color="#c9ccd1"))
    _fig.add_scattergl(x=emb0[_other, 0], y=emb0[_other, 1], mode="markers",
                       name="other labeled", marker=dict(size=5, opacity=0.75, color="#f59e0b"),
                       text=[_cat[i] for i in np.where(_other)[0]],
                       hovertemplate="%{text}<extra></extra>")
    _fig.add_scattergl(x=emb0[_is_agg, 0], y=emb0[_is_agg, 1], mode="markers", name="aggression",
                       marker=dict(size=6, opacity=0.8, color="#d62728"))
    _fig.add_scatter(x=[emb0[HERO, 0]], y=[emb0[HERO, 1]], mode="markers+text",
                     name="Hero #909", text=["#909"], textposition="top center",
                     marker=dict(symbol="star", size=18, color="#f5b400",
                                 line=dict(color="#333", width=1)))
    _fig.update_layout(template="plotly_white", height=560,
                       title="Canonical map — category tags overlaid (Hero #909 in its syllable)",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Render a syllable's exemplars

        Pick a cluster; we tile up to nine of its member events as skeleton GIFs (rank-colored:
        <span style="color:#d62728">Dom</span> / <span style="color:#1f77b4">Mid</span> /
        <span style="color:#2ca02c">Sub</span>). This is the *by-eye* half of cluster validation —
        the enrichment number in the exercise is the *by-stat* half. **C4** should read as clear
        fighting; the big **C0** blob should look like a grab-bag.
        """
    )
    return


@app.cell
def _(best_cluster, clabels, mo):
    _opts = {f"C{c}": c for c in sorted(set(clabels.tolist())) if c >= 0}
    clus_pick = mo.ui.dropdown(options=_opts, value=f"C{best_cluster}", label="syllable to render")
    return (clus_pick,)


@app.cell
def _(HERO, clabels, clus_pick, cu, ev, mo, np):
    _c = int(clus_pick.value)
    _idx = np.where(clabels == _c)[0]
    # put the Hero first if it belongs to this cluster, then fill up to 9
    _order = ([HERO] if HERO in _idx else []) + [i for i in _idx.tolist() if i != HERO]
    _pick = _order[:9]
    _events = [(ev["kp"][i], ev["ranks"][i], int(ev["contact_rel"][i])) for i in _pick]
    _gif = cu.grid_gif_bytes(_events, ncols=3, cell=120)
    _cap = (f"**C{_c}** · {len(_idx)} events · showing {len(_pick)}"
            + (" (Hero #909 top-left)" if HERO in _idx else ""))
    mo.vstack([clus_pick, mo.md(_cap), mo.Html(cu.gif_img_html(_gif, width=380))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · How UMAP lies

        The two panels below are the **same 1500 events, same canonical colors** — only the knobs
        differ. On the left (`n_neighbors=5, min_dist=0`) the map shatters into many tight islands;
        on the right (`n_neighbors=100, min_dist=0.8`) it melts into one soft continent. Nothing
        about the *behavior* changed. So, staring at any one map, you **may NOT conclude**:

        - **inter-cluster distance** — two blobs far apart are not "more different" than two blobs
          close together; UMAP does not preserve global distances.
        - **cluster size / density** — a big fluffy blob is not "more common" or "more spread out"
          than a tight one; `min_dist` alone sets apparent area.
        - **the number of clusters** — that is a knob (`n_neighbors`, `min_cluster_size`), not a fact.

        What you *may* trust: **who-is-near-whom** (local neighborhoods), and — only after a null
        test and a by-eye check — that a dense pocket is a recurring behavior.
        """
    )
    return


@app.cell
def _(clabels, go, sweep):
    from plotly.subplots import make_subplots as _msub
    _cells = [(0, 0, "n_neighbors=5, min_dist=0"), (4, 4, "n_neighbors=100, min_dist=0.8")]
    _fig = _msub(rows=1, cols=2, subplot_titles=[t for *_, t in _cells])
    for _col, (_i, _j, _t) in enumerate(_cells, start=1):
        _emb = sweep["emb_grid"][_i, _j]
        for _c in sorted(set(clabels.tolist())):
            _m = clabels == _c
            _fig.add_trace(go.Scattergl(
                x=_emb[_m, 0], y=_emb[_m, 1], mode="markers",
                name=("noise" if _c < 0 else f"C{_c}"),
                legendgroup=str(_c), showlegend=(_col == 1),
                marker=dict(size=4, opacity=0.7, color=("#cfd2d8" if _c < 0 else None))),
                row=1, col=_col)
    _fig.update_xaxes(showticklabels=False, showgrid=False)
    _fig.update_yaxes(showticklabels=False, showgrid=False)
    _fig.update_layout(template="plotly_white", height=430,
                       title="Same points, two knobs — do not read distance or size off either one",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6 · Where is subclustering? Hierarchical discovery (L1 -> L2)

        The real analysis pipeline this course is built on does not stop at one clustering pass. It
        runs a **coarse L1 pass**, then re-embeds and re-clusters *inside* an interesting L1 cluster
        to reveal **L2 sub-types**. That is exactly what a lab does with a giant undifferentiated
        blob: zoom in and ask whether it is one behavior or several wearing one coat.

        Take the largest syllable — **C0** (~930 events, aggression fraction near base rate, i.e. a
        grab-bag) — and re-run HDBSCAN *on just its members' coordinates* at a finer
        `min_cluster_size`. Watch a homogeneous-looking blob split into sub-types with **different
        aggression rates**: at the default setting one sub-type is nearly aggression-free (a clean
        "quiet co-presence" mode) while another carries the residual fights the coarse pass smeared
        in. This is discovery-by-refinement, and it is why "one cluster" is never a final answer.
        """
    )
    return


@app.cell
def _(clabels, mo):
    _opts = {f"C{c}": c for c in sorted(set(clabels.tolist())) if c >= 0}
    parent_pick = mo.ui.dropdown(options=_opts, value="C0", label="parent syllable to split")
    sub_mcs = mo.ui.slider(15, 60, value=25, step=1, label="sub min_cluster_size",
                           debounce=True)
    return parent_pick, sub_mcs


@app.cell
def _(agg, clabels, cu, emb0, go, mo, parent_pick, sub_mcs):
    _p = int(parent_pick.value)
    _mask = clabels == _p
    _sub = cu.run_hdbscan(emb0[_mask], min_cluster_size=int(sub_mcs.value))
    _e = emb0[_mask]
    _a = agg[_mask]
    _fig = go.Figure()
    _lines = []
    for _c in sorted(set(_sub.tolist())):
        _m = _sub == _c
        _nm = "noise" if _c < 0 else f"C{_p}.{_c}"
        _fig.add_scattergl(x=_e[_m, 0], y=_e[_m, 1], mode="markers", name=_nm,
                           marker=dict(size=6, opacity=0.75,
                                       color=("#cfd2d8" if _c < 0 else None)))
        if _c >= 0:
            _lines.append(f"**{_nm}**: n={int(_m.sum())}, aggression={_a[_m].mean():.0%}")
    _nsub = len([c for c in set(_sub.tolist()) if c >= 0])
    _fig.update_layout(template="plotly_white", height=460,
                       title=f"Sub-types inside C{_p} — {_nsub} L2 clusters "
                             f"(sub min_cluster_size={int(sub_mcs.value)})",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([parent_pick, sub_mcs], justify="start"), _fig,
               mo.md(f"C{_p} base aggression rate near {_a.mean():.0%}. Sub-types: "
                     + " · ".join(_lines))])
    return


@app.cell(hide_code=True)
def _(base_rate, mo):
    mo.md(
        f"""
        ## 7 · Exercise — did the map rediscover aggression?

        **Hypothesis (pre-registered):** *At least one data-driven syllable is enriched for
        aggression above the {base_rate:.0%} base rate.*

        **Toolbox.**
        - `clabels : (1500,) int` — canonical syllable of each event (-1 = noise).
        - `agg : (1500,) int` — 1 if the event is aggression, else 0.
        - `base_rate : float` — the corpus-wide aggression rate ({base_rate:.2f}).
        - You return `(cluster, frac, lift)` for the **purest** syllable: its id, its aggression
          fraction, and `lift = frac / base_rate`.

        **Your turn.** Fill in `purest_agg_cluster` below so it finds the syllable with the *highest*
        aggression fraction (ignore noise, `-1`). The stub ships a **deliberate trap** — it returns
        the *largest* cluster, which is not the purest. Fix it, then read the self-check.
        """
    )
    return


@app.cell
def _(np):
    def purest_agg_cluster(clabels, agg, base_rate):
        clabels = np.asarray(clabels); agg = np.asarray(agg)
        clusters = [c for c in sorted(set(clabels.tolist())) if c >= 0]
        # -------------------- TODO (student) --------------------
        # BUG: this returns the LARGEST cluster, not the PUREST. Replace `key` so it picks the
        # cluster with the highest aggression fraction, then compute frac and lift for THAT cluster.
        chosen = max(clusters, key=lambda c: (clabels == c).sum())     # <-- fix me
        # --------------------------------------------------------
        frac = float(agg[clabels == chosen].mean())
        lift = frac / base_rate
        return int(chosen), frac, lift
    return (purest_agg_cluster,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution": mo.md(
            r"""
            Rank clusters by **aggression fraction**, not size:

            ```python
            chosen = max(clusters, key=lambda c: agg[clabels == c].mean())
            frac   = float(agg[clabels == chosen].mean())
            lift   = frac / base_rate
            return int(chosen), frac, lift
            ```

            On the canonical labels this returns **C4**, `frac ~ 0.42`, `lift ~ 1.40x`. The largest
            cluster (C0, ~930 events) sits at ~0.27 — *below* base rate — which is why "biggest" is
            the wrong instinct: the interesting behavior hides in a small, dense pocket, not the blob.
            """)
    })
    return


@app.cell(hide_code=True)
def _(agg, base_rate, clabels, mo, purest_agg_cluster):
    # ---- self-check: tolerance band around the pinned build-time lift (1.40x) ----------------------
    _PIN, _TOL = 1.40, 0.12          # accept [1.28, 1.52]
    try:
        _c, _frac, _lift = purest_agg_cluster(clabels, agg, base_rate)
        _pass = abs(_lift - _PIN) <= _TOL
        if _pass:
            _bg, _fg, _icon = "#e7f6ec", "#166534", "PASS"
            _msg = (f"C{_c} is {_frac:.0%} aggression -> **{_lift:.2f}x lift** (within "
                    f"{_PIN:.2f}+/-{_TOL:.2f}). The map rediscovered aggression **without labels** — "
                    f"but read it honestly: 1.4x is a *modest* enrichment, and the pocket is still "
                    f"only ~42% aggression, not a pure fighting island. Unsupervised recovery is "
                    f"real here, not clean.")
        else:
            _bg, _fg, _icon = "#fdecec", "#9b1c1c", "NOT YET"
            _msg = (f"Got C{_c} at **{_lift:.2f}x lift** — outside {_PIN:.2f}+/-{_TOL:.2f}. If this is "
                    f"~0.9x, you returned the *largest* cluster (C0), which is below base rate. Rank "
                    f"by aggression fraction instead — see the solution.")
    except Exception as _e:
        _bg, _fg, _icon = "#fdecec", "#9b1c1c", "ERROR"
        _msg = f"`purest_agg_cluster` raised: `{_e}`. Return `(int, float, float)`."
    mo.md(f"""
<div style="background:{_bg};color:{_fg};border-radius:8px;padding:10px 14px;
            font-family:'Liberation Sans',sans-serif"><b>{_icon}</b> &nbsp; {_msg}</div>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Deeper: the paper & where the analogy stops": mo.md(
            r"""
            **Shared mathematics.** Nonlinear embedding of a high-D signal followed by density
            clustering into recurring states is *one method*, whether the signal is postural or
            neural — you build a neighbor graph, lay it flat, and carve dense pockets.

            **The lineage**
            - McInnes, Healy & Melville 2018 — **UMAP** (the embedding).
            - Berman et al. 2014, *J. R. Soc. Interface* — **MotionMapper** (t-SNE + watershed): the
              direct *method* ancestor of behavioral maps.
            - Hsu & Yttri 2021, *Nat. Commun.* — **B-SOiD** (UMAP + clustering on pose), essentially
              this pipeline.
            - Campello, Moulavi & Sander 2013 — **HDBSCAN** (the density carving).
            - Wiltschko et al. 2015, *Neuron* — the *concept* "behavioral syllable" only.

            **Species / preparation.** Freely-moving mice, markerless pose (our rig); the neural
            analog is population recordings embedded into low-D state maps.

            **Where the analogy stops.** UMAP coordinates and cluster *sizes* are **not metric** — you
            may not read inter-state distance or state prevalence off the picture. And MoSeq
            (Wiltschko 2015) is an **AR-HMM**, *not* embed-then-cluster — that temporal twin belongs
            to NB07, not here. A "syllable" on this map is a static pocket; a MoSeq syllable is a
            dynamical unit.
            """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8 · What we threw away / how it breaks

        **Discarded.** The map collapses each event to two coordinates — it throws away *time*
        (every frame-by-frame trajectory is gone; an event is a frozen dot), the residualized PC0/PC2
        "how-close/how-fast" axis (a modeling *choice* from NB04, not a fact), and any behavior too
        rare to reach `min_cluster_size` (mounting n=3, tail_bite n=7 dissolve into noise).

        **Three ways it breaks on THIS data.**
        1. **The blob swallows the majority.** C0 holds ~930 of 1500 events — density clustering will
           happily park most of behavior in one undifferentiated pocket, which is why we had to
           *subcluster* it to see anything.
        2. **Knob-sensitivity.** Nudge `n_neighbors` or `min_dist` and the syllable count and shapes
           move; "5 clusters" is a choice, not a discovery.
        3. **Modest, impure recovery.** The best aggression pocket is 1.4x base rate and only ~42%
           aggression — good enough to say "aggression is a natural kind here," *not* good enough to
           gate an experiment on the cluster alone. That is why NB08 trains a supervised decoder.

        **How would you analyze this?** The map merged two things you can tell apart by eye —
        aggression and vigorous non-aggressive chasing. *What single feature would you add to the
        19 to split them, and would you add it before UMAP (change the geometry) or use it only to
        re-color the map you already have?*
        """
    )
    return


@app.cell(hide_code=True)
def _(base_rate, best_frac, best_lift, board, make_board, mo):
    mo.md(make_board(board, base_rate, best_lift, best_frac))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we ship next

        **We shipped:** a 2-D behavioral map and a set of canonical syllables the whole week agrees
        on — and an honest verdict that unsupervised clustering *does* rediscover an aggression-tilted
        state (1.4x), so aggression is real enough to build on. Phase 1 (Discover) is complete: 11,700
        raw numbers -> a map you can point at.

        **The catch:** a map is only science if you can test claims on it. In **NB06** we do
        enrichment statistics *properly* — contingency tables, Bonferroni, and the pseudoreplication
        trap — and the beautiful sex/rank result you are about to get excited about **collapses** the
        moment cage becomes the unit. The map is built; next we find out how much of it we are allowed
        to believe.

        **Next -> `06_reading_the_map.py`.**
        """
    )
    return


if __name__ == "__main__":
    app.run()

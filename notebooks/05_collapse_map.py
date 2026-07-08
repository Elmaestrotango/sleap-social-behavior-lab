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
        # NB05 · Building a behavioral map

        ## Why we build a map

        In the previous notebook we took each interaction — originally 19 measured features — and
        compressed it down to a few **principal components** (the directions along which the events
        vary most). Those axes were useful, but they were chosen *by us*: PCA simply finds directions
        of greatest variance in the numbers we handed it. It does not know what a mouse is doing.

        In this notebook we take a different approach. Instead of imposing axes, we let the data show
        us which **kinds of behavior recur** across the 1,500 events, without deciding in advance what
        to look for. We will lay every event out as a single point on a two-dimensional map, group
        nearby points into behavioral types, and check whether one of those types corresponds to
        aggression — a category we never told the method about.

        The scientific question for this notebook is simple: **can an unsupervised method find
        aggression on its own, without ever being told which events are aggressive?** If it can, that
        is evidence that aggression is a genuine, recurring mode of behavior in these animals and not
        just a label we impose. (This is also how neuroscientists quantify behavior more generally.)

        ## Definitions you need first

        - **Unsupervised.** A method that looks only at the input features and groups the data by
          similarity. It is never shown the labels (here: which events are aggression). The opposite is
          *supervised*, where a model is trained on labelled examples (that comes later, in NB08).
        - **Embedding / 2-D map.** A procedure that places each high-dimensional event at an (x, y)
          position so that events which are similar in the original feature space end up near each
          other on the flat map. Here the tool is **UMAP**. One dot on the map is one whole interaction
          event, collapsed to a single location.
        - **Clustering.** Grouping the dots so that dense pockets of nearby points become named groups.
          Points in sparse regions can be left unassigned ("noise"). The tool here is **HDBSCAN**.
        - **Behavioral type (or "syllable").** One of the recurring groups the clustering finds — a
          pocket of the map where many events resemble one another. We call each group a syllable and
          give it a label (C0, C1, ...). "Syllable" is borrowed from the behavior literature; treat it
          here as a synonym for "a data-driven cluster of similar events."

        By the end you will have a 2-D behavioral map plus a fixed set of syllables that the rest of
        the week's notebooks all agree on.
        """
    )
    return


@app.cell
def _(ROOT, cu):
    # ---- canonical data loads (no raw np.load; the loaders fetch on a bare kernel) ----------------
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))   # kp, ranks, category, agg_label
    der = cu.load_derived("train", ROOT)                               # cage/sex/tod_hour/X/PCA live HERE
    sweep = cu.load_umap_sweep(ROOT)                                   # emb_grid + CANONICAL default_labels
    ho = cu.load_events(cu.data_path("data/heldout_events.npz", ROOT)) # Cage 16 — held out
    return der, ev, ho, sweep


@app.cell
def _(ROOT, cu, ev, sweep):
    # ---- the canonical map + syllables (every NB05-07 notebook agrees on THESE) -------------------
    di, dj = int(sweep["default_ij"][0]), int(sweep["default_ij"][1])
    emb0 = sweep["emb_grid"][di, dj]                     # the pinned default 2-D embedding (1500,2)
    clabels = sweep["default_labels"].astype(int)       # CANONICAL syllables: C0..C4 + noise(-1)
    agg = ev["agg_label"].astype(int)
    base_rate = float(agg.mean())                       # 0.300
    EXAMPLE = 909                                        # our running example approach event
    #                                                      (Cage 15, male; approacher = Dom, approachee = Sub)

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
    return EXAMPLE, agg, base_rate, best_cluster, best_frac, best_lift, board, clabels, di, dj, emb0


@app.cell
def _(mo):
    # ---- Readout Board renderer (used top AND bottom; degrades gracefully) ------------------------
    # Gauge A reports the SIZE of the current representation as a plain positive number at each step;
    # Gauge B reports held-out readiness. No negative deltas anywhere.
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
        <b>Gauge A · size of the representation</b> (this notebook produces the map)</div>
      <div style="font-size:15px;line-height:1.9">{steps}</div>
      <div style="font-size:11px;color:#999;margin-top:4px">
        different <i>kinds</i> of reduction, not one single number</div>
    </div>
    <div style="flex:1;min-width:200px;border-left:1px solid #e2e5ea;padding-left:16px">
      <div style="font-size:13px;color:#555;margin-bottom:4px">
        <b>Gauge B · held-out readiness</b></div>
      <div style="font-size:13px">best aggression-cluster lift</div>
      <div style="font-size:24px;font-weight:700;color:#2563eb">{best_lift:.2f}x
        <span style="font-size:13px;color:#888;font-weight:400">
          (yours) &nbsp;vs&nbsp; {bench_lift}x benchmark</span></div>
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
    # ---- Held-out Cage 16 (kept aside until NB08) --------------------------------------------------
    _n = len(ho["kp"])
    mo.md(
        f"""
<div style="border:2px dashed #b02a37;border-radius:10px;padding:12px 16px;margin:6px 0;
            background:repeating-linear-gradient(45deg,#fbf0f1,#fbf0f1 12px,#f7e6e8 12px,#f7e6e8 24px);
            font-family:'Liberation Sans',sans-serif">
  <span style="font-weight:700;color:#b02a37">HELD OUT · Camera 16</span>
  &nbsp;<span style="font-size:13px;color:#555">reserved for a fair final test</span><br>
  <span style="font-size:13px;color:#555">Events on file: <b>{_n}</b> &nbsp;·&nbsp;
  not used to build today's map</span><br>
  <span style="font-size:12px;color:#b02a37">These events are opened in <b>NB08</b>. The map you
  build today is fit only on the other cages, so we can later check it on data it has never seen.</span>
</div>"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Why the map is precomputed (live UMAP is disabled here)

        **What UMAP does.** UMAP builds a graph connecting each event to its nearest neighbors in the
        19-feature space, then lays that graph out in two dimensions so that neighbors in the original
        space stay neighbors on the flat map. Its inputs are the feature matrix and two settings
        (`n_neighbors`, `min_dist`); its output is an (x, y) coordinate for every event.

        **Why we do not run it live.** UMAP compiles specialized numerical code the first time it runs.
        On a fresh cloud kernel this first call takes roughly 28 seconds — long enough that the
        notebook's connection times out before any point appears, and every slider change would
        trigger another recompile. To avoid this, the course engine disables live UMAP (the
        `run_umap` helper raises if called here) and instead ships a **precomputed 5x5 sweep**: the
        same 1,500 events already embedded at every combination of the two settings. We *select* a
        precomputed map; we never recompute one. The only step we run live is HDBSCAN, which is fast
        (about 0.1 s).

        **The figure below** shows the full sweep as a grid of small maps. Each panel is the same
        1,500 events at a different `n_neighbors` (rows) and `min_dist` (columns); a point is drawn
        <span style="color:#d62728">red</span> if that event is aggression and gray otherwise. Scan
        the rows and columns to see how the two settings change the shape of the map.
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
        ## 2 · Select one map from the sweep

        The two settings control the shape of the map:

        - **`n_neighbors`** — how many neighbors each point is tied to. Small values emphasize local
          detail and tend to break the data into many small islands; large values emphasize global
          shape and tend to merge everything into one continent.
        - **`min_dist`** — how tightly points are allowed to pack. Small values produce tight,
          well-separated blobs; large values spread points out.

        Pick any cell below. The map redraws instantly from the precomputed grid, now colored by the
        **canonical syllables** (`sweep["default_labels"]`) that every later notebook uses. The gold
        star marks our example approach event (#909). The default cell
        (`n_neighbors=15, min_dist=0.0`) is the one the rest of the course is pinned to.
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
def _(EXAMPLE, clabels, go, md_pick, mo, nn_pick, sweep):
    _i, _j = int(nn_pick.value), int(md_pick.value)
    _emb = sweep["emb_grid"][_i, _j]
    _fig = go.Figure()
    for _c in sorted(set(clabels.tolist())):
        _m = clabels == _c
        _nm = "noise" if _c < 0 else f"C{_c}"
        _col = "#cfd2d8" if _c < 0 else None
        _fig.add_scattergl(x=_emb[_m, 0], y=_emb[_m, 1], mode="markers", name=_nm,
                           marker=dict(size=5, opacity=0.7, color=_col))
    _fig.add_scatter(x=[_emb[EXAMPLE, 0]], y=[_emb[EXAMPLE, 1]], mode="markers", name="example #909",
                     marker=dict(symbol="star", size=17, color="#f5b400",
                                 line=dict(color="#333", width=1)))
    _fig.update_layout(template="plotly_white", height=520,
                       title=(f"Selected map — n_neighbors={int(sweep['nn_values'][_i])}, "
                              f"min_dist={float(sweep['md_values'][_j]):g} — colored by canonical syllable"),
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([nn_pick, md_pick], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What one point represents

        Every dot on this map is **one entire interaction event** — 130 frames of two mice, reduced
        first to 19 features (NB02), then to a handful of principal components (NB04), and finally to
        one (x, y) location by UMAP. When two dots sit close together, the two events had similar
        posture and motion. The map has no units: the numbers on the axes are not distances in the
        cage, and we will return to what you may and may not read off it in section 5.

        Before we cluster, it helps to see a real event so the dots feel concrete. Below is our
        running example, event #909 — a single approach in Cage 15 rendered as a skeleton movie. Mice
        are colored **by rank**: <span style="color:#d62728"><b>Dom = red</b></span>,
        <span style="color:#1f77b4"><b>Mid = blue</b></span>,
        <span style="color:#2ca02c"><b>Sub = green</b></span>. The
        <span style="color:#d62728">red Dom</span> mouse is the **approacher**; the
        <span style="color:#2ca02c">green Sub</span> mouse is the **approachee**.
        """
    )
    return


@app.cell
def _(EXAMPLE, cu, ev, mo):
    # our running example approach event, rendered as a rank-colored skeleton GIF
    _kp = ev["kp"][EXAMPLE]
    _rk = ev["ranks"][EXAMPLE]
    _cr = int(ev["contact_rel"][EXAMPLE])
    _gif = cu.event_gif_bytes(_kp, _rk, contact_rel=_cr)
    mo.vstack([mo.md("**Example approach event #909** — the approacher (Dom, red) closes on the "
                     "approachee (Sub, green); contact is near the middle of the clip."),
               mo.Html(cu.gif_img_html(_gif, width=260))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · Group the points — HDBSCAN

        **What clustering does here.** HDBSCAN scans the map for dense pockets of points and calls
        each dense pocket a cluster. Points that sit in sparse regions are labelled **noise (-1)**
        rather than being forced into a group. Its input is the (x, y) map plus one setting,
        `min_cluster_size` (the smallest group it will accept); its output is a cluster label for
        every event. Unlike k-means, you do not tell it how many clusters to find.

        This is the one step we run live as you drag the slider (it is fast). At the canonical
        `min_cluster_size = 15` it reproduces the shared `default_labels` exactly. Larger values merge
        syllables together; smaller values break the map into more, smaller pieces and push more
        points into noise.
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
               mo.md("*Set the slider to 15 to reproduce the canonical result (5 clusters, about "
                     "15% noise). This slider is for exploration only — the shared syllables the rest "
                     "of the week uses stay fixed at 15.*")])
    return


@app.cell(hide_code=True)
def _(best_cluster, clabels, mo):
    _sizes = {c: int((clabels == c).sum()) for c in sorted(set(clabels.tolist())) if c >= 0}
    mo.md(
        f"""
        ## 4 · The syllables, and where the example event sits

        The canonical clustering produces **{len(_sizes)} syllables** plus noise, with sizes
        {_sizes}. The syllable with the highest aggression fraction is **C{best_cluster}**, and our
        example approach event (#909, a real Cage 15 aggression bout) lands inside it. The single event
        we have followed since NB01 is now one point on a data-driven map, sitting in the group the
        data itself placed it in.

        Below, the map is recolored by the **registry category** of each event (aggression,
        other-labeled, or unlabeled) so you can see how the labels we happen to have fall across the
        map. After that, you can render a chosen syllable's member events as skeleton GIFs and judge by
        eye whether the group really looks like one kind of behavior.
        """
    )
    return


@app.cell
def _(EXAMPLE, emb0, ev, go, np):
    # canonical map colored by REGISTRY category (agg / other-labeled / unlabeled) + example star
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
    _fig.add_scatter(x=[emb0[EXAMPLE, 0]], y=[emb0[EXAMPLE, 1]], mode="markers+text",
                     name="example #909", text=["#909"], textposition="top center",
                     marker=dict(symbol="star", size=18, color="#f5b400",
                                 line=dict(color="#333", width=1)))
    _fig.update_layout(template="plotly_white", height=560,
                       title="Canonical map — category tags overlaid (example event #909 in its syllable)",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Render a syllable's member events

        Pick a cluster below. We tile up to nine of its member events as skeleton GIFs, colored by
        rank (<span style="color:#d62728">Dom = red</span> /
        <span style="color:#1f77b4">Mid = blue</span> /
        <span style="color:#2ca02c">Sub = green</span>). Watching the members is the by-eye half of
        checking a cluster; the enrichment number in the section 7 exercise is the by-number half.
        **C4** should look like clear fighting; the large **C0** group should look like a mixture of
        many behaviors.
        """
    )
    return


@app.cell
def _(best_cluster, clabels, mo):
    _opts = {f"C{c}": c for c in sorted(set(clabels.tolist())) if c >= 0}
    clus_pick = mo.ui.dropdown(options=_opts, value=f"C{best_cluster}", label="syllable to render")
    return (clus_pick,)


@app.cell
def _(EXAMPLE, clabels, clus_pick, cu, ev, mo, np):
    _c = int(clus_pick.value)
    _idx = np.where(clabels == _c)[0]
    # put the example event first if it belongs to this cluster, then fill up to 9
    _order = ([EXAMPLE] if EXAMPLE in _idx else []) + [i for i in _idx.tolist() if i != EXAMPLE]
    _pick = _order[:9]
    _events = [(ev["kp"][i], ev["ranks"][i], int(ev["contact_rel"][i])) for i in _pick]
    _gif = cu.grid_gif_bytes(_events, ncols=3, cell=120)
    _cap = (f"**C{_c}** · {len(_idx)} events · showing {len(_pick)}"
            + (" (example event #909 top-left)" if EXAMPLE in _idx else ""))
    mo.vstack([clus_pick, mo.md(_cap), mo.Html(cu.gif_img_html(_gif, width=380))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · What the map does and does not tell you

        The two panels below are the **same 1,500 events with the same canonical colors** — only the
        two settings differ. On the left (`n_neighbors=5, min_dist=0`) the map breaks into many tight
        islands; on the right (`n_neighbors=100, min_dist=0.8`) it merges into one soft continent.
        Nothing about the behavior changed. Because the picture is so sensitive to the settings, there
        are three things you must **not** read off any single map:

        - **Distance between clusters.** Two groups far apart are not "more different" than two groups
          close together. UMAP does not preserve global distances.
        - **Cluster size or density.** A large, loose blob is not "more common" or "more variable" than
          a small tight one. `min_dist` alone changes the apparent area.
        - **The number of clusters.** That is set by the knobs (`n_neighbors`, `min_cluster_size`), not
          a fixed fact about the data.

        What you *can* trust is **which points are near which** (local neighborhoods), and — only after
        a statistical test and a look at the member events — that a dense pocket corresponds to a
        recurring behavior.
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
                       title="Same points, two settings — do not read distance or size off either one",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6 · Looking inside a cluster — subclustering (L1 -> L2)

        **Why do this.** A single clustering pass gives coarse groups. A large group may actually
        contain several distinct behaviors that happen to look similar at low resolution. The standard
        response is to zoom in: cluster once to get coarse groups (call this level 1), then take one
        large group and cluster *its members again* (level 2) to see whether it splits into sub-types.

        **The method.** Take the largest syllable — **C0** (about 930 events, with an aggression
        fraction close to the base rate, i.e. a mixed group) — and re-run HDBSCAN on just the
        coordinates of its members, at a finer `min_cluster_size`. Watch a group that looked
        homogeneous split into sub-types with **different aggression rates**: at the default setting
        one sub-type is nearly aggression-free (a quiet co-presence mode) while another carries the
        remaining fights the coarse pass had absorbed. This is why "one cluster" is rarely the final
        answer.
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
                       title=f"Sub-types inside C{_p} — {_nsub} level-2 clusters "
                             f"(sub min_cluster_size={int(sub_mcs.value)})",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([parent_pick, sub_mcs], justify="start"), _fig,
               mo.md(f"C{_p} overall aggression rate is about {_a.mean():.0%}. Sub-types: "
                     + " · ".join(_lines))])
    return


@app.cell(hide_code=True)
def _(base_rate, mo):
    mo.md(
        f"""
        ## 7 · Exercise — did the map rediscover aggression?

        **The question.** Is at least one data-driven syllable enriched for aggression above the
        overall {base_rate:.0%} base rate? If so, the unsupervised map found aggression on its own.

        **What you have to work with.**

        - `clabels : (1500,) int` — the canonical syllable of each event (-1 means noise).
        - `agg : (1500,) int` — 1 if the event is aggression, else 0.
        - `base_rate : float` — the corpus-wide aggression rate ({base_rate:.2f}).
        - You return `(cluster, frac, lift)` for the **purest** syllable: its id, its aggression
          fraction, and `lift = frac / base_rate` (how many times the base rate it reaches).

        **Your task.** The function `purest_agg_cluster` below is meant to return the syllable with the
        highest aggression fraction, but right now it ranks clusters by **size** instead. Edit the one
        marked line so it ranks by aggression fraction. The rest of the function already computes
        `frac` and `lift` for whichever cluster you choose — you do not need to touch it.

        **What you should see.** After the fix, the self-check box turns green and reports the purest
        syllable at roughly `1.40x` lift. Before the fix it reports the largest cluster (C0), which is
        *below* the base rate — a reminder that the interesting behavior lives in a small dense pocket,
        not the big blob.
        """
    )
    return


@app.cell
def _(agg, clabels, np):
    def purest_agg_cluster(clabels, agg, base_rate):
        clabels = np.asarray(clabels); agg = np.asarray(agg)
        clusters = [c for c in sorted(set(clabels.tolist())) if c >= 0]
        # -------------------- EDIT THE NEXT LINE ONLY --------------------
        # `max(clusters, key=...)` returns the cluster with the largest value of `key`.
        # Right now key = (clabels == c).sum(), which is the cluster's SIZE (number of events).
        # You want the cluster with the highest AGGRESSION FRACTION instead.
        # Replace  (clabels == c).sum()  with  agg[clabels == c].mean()
        #   -> agg[clabels == c].mean() is the fraction of events in cluster c that are aggression.
        chosen = max(clusters, key=lambda c: (clabels == c).sum())     # <-- EDIT THIS LINE
        # -----------------------------------------------------------------
        frac = float(agg[clabels == chosen].mean())
        lift = frac / base_rate
        return int(chosen), frac, lift
    return (purest_agg_cluster,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution": mo.md(
            r"""
            Rank the clusters by **aggression fraction**, not size:

            ```python
            chosen = max(clusters, key=lambda c: agg[clabels == c].mean())
            ```

            On the canonical labels this returns **C4**, with `frac ~ 0.42` and `lift ~ 1.40x`. The
            largest cluster (C0, about 930 events) sits near `0.27` — *below* the base rate — which is
            why ranking by size gives the wrong answer: the aggression signal is concentrated in a
            small, dense pocket, not the big group.
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
            _msg = (f"C{_c} is {_frac:.0%} aggression, giving a **{_lift:.2f}x lift** (within "
                    f"{_PIN:.2f} +/- {_TOL:.2f}). The map recovered aggression **without any labels**. "
                    f"Read the result honestly: 1.4x is a modest enrichment, and the pocket is still "
                    f"only about 42% aggression, not a pure fighting island. Unsupervised recovery is "
                    f"real here, but not clean.")
        else:
            _bg, _fg, _icon = "#fdecec", "#9b1c1c", "NOT YET"
            _msg = (f"Got C{_c} at **{_lift:.2f}x lift** — outside {_PIN:.2f} +/- {_TOL:.2f}. If this "
                    f"is about 0.9x, the function returned the *largest* cluster (C0), which is below "
                    f"the base rate. Rank by aggression fraction instead — see the solution.")
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
        "Further reading — the method lineage": mo.md(
            r"""
            The recipe used here — embed a high-dimensional signal into 2-D, then cluster the dense
            pockets — is a standard one in computational ethology. Some references:

            - McInnes, Healy & Melville 2018 — **UMAP** (the embedding used here).
            - Berman et al. 2014, *J. R. Soc. Interface* — **MotionMapper** (t-SNE + watershed): an
              early behavioral map.
            - Hsu & Yttri 2021, *Nat. Commun.* — **B-SOiD** (UMAP + clustering on pose), close to this
              pipeline.
            - Campello, Moulavi & Sander 2013 — **HDBSCAN** (the density clustering).
            - Wiltschko et al. 2015, *Neuron* — introduces the term "behavioral syllable."

            **One caution.** A syllable on this map is a *static* pocket: it groups events by overall
            posture and motion, with time collapsed away. Some methods (for example MoSeq, Wiltschko
            2015) instead model behavior as a sequence of states over time. That temporal view is the
            subject of NB07; here, a syllable is simply a cluster of similar events.
            """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8 · What the map discards, and how it can mislead

        **What is thrown away.** Collapsing each event to two coordinates discards *time* — every
        frame-by-frame trajectory is gone, and an event becomes a single frozen dot. It also discards
        any behavior too rare to reach `min_cluster_size` (for example mounting, n=3, and tail-bite,
        n=7, dissolve into noise). And the map inherits the modeling choices made in NB04 (which
        features were kept, how they were scaled).

        **Three ways it can mislead on this data.**

        1. **One group holds most events.** C0 contains about 930 of 1,500 events. Density clustering
           will happily park most of behavior in one undifferentiated group, which is why we had to
           subcluster it to see structure inside.
        2. **Sensitivity to settings.** Change `n_neighbors` or `min_dist` and the number and shapes
           of the clusters move. "5 clusters" is a choice, not a discovery.
        3. **Modest, impure recovery.** The best aggression pocket is 1.4x the base rate and only about
           42% aggression — enough to say aggression is a recurring mode here, but not enough to base
           an experiment on the cluster alone. That is why NB08 turns to a supervised decoder.

        **A question to think about.** The map merged two things you can often tell apart by eye:
        aggression and vigorous non-aggressive chasing. What single feature would you add to the 19 to
        separate them, and would you add it *before* UMAP (changing the geometry of the map) or use it
        only to recolor the map you already have?
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
        ## What comes next

        **What this notebook produced:** a 2-D behavioral map and a fixed set of canonical syllables
        that the rest of the week uses, plus an honest result — unsupervised clustering does recover an
        aggression-enriched state (about 1.4x the base rate), so aggression is a real, recurring mode
        of behavior in these animals. The first phase of the course is now complete: 11,700 raw numbers
        per event have become a map you can point at.

        **The limitation:** a map is only useful if you can test claims on it rigorously. In **NB06**
        we do the enrichment statistics carefully — contingency tables, correction for multiple
        comparisons, and the pseudoreplication problem — and see how an apparent sex or rank effect
        can weaken once **cage** is treated as the unit of analysis. The map is built; next we ask how
        much of it we are entitled to believe.

        **Next -> `06_reading_the_map.py`.**
        """
    )
    return


if __name__ == "__main__":
    app.run()

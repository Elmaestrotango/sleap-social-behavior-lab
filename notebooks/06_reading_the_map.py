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
        # NB06 · Reading the Map — statistics done honestly

        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > After we fire the laser at VMHvl we will ask one question: *did the manipulation shift the
        > distribution of behaviors on your map?* That is a **cluster-enrichment test**, and we need
        > you to prove the logic **now**, on variables we already have, before a single trial is run.
        >
        > **Deliverable:** an honest enrichment pipeline — which behavioral syllables are over-represented
        > in which condition — with the multiple-comparison and pseudoreplication traps handled correctly.
        >
        > **The question it unblocks:** *is a shift in the cluster distribution a real effect, or an
        > accident of which cages we happened to record?* Get this wrong and every opto result is wrong.
        >
        > **Today's lab-meeting question:** *Did hunger rewrite the ethogram — and is the sex difference
        > everyone is excited about real, or is it just cage identity wearing a costume?*

        Reading out a manipulation as *"does it shift the state distribution?"* is exactly how a circuit
        lab reads an opto or a lesion — the same cluster-level discipline that governs EEG/MEG statistics
        and inter-brain decoding of dominance. Today we learn to do it without fooling ourselves. **This
        is the notebook where a beautiful result dies.**
        """
    )
    return


@app.cell
def _(cu, np):
    ev = cu.load_events("data/train_events.npz")
    der = cu.load_derived("train")
    sweep = cu.load_umap_sweep()

    labels = sweep["default_labels"]          # canonical syllables C0..C4 (+ -1 noise)
    cage = der["cage"]                         # cage id 9..15
    sex = der["sex"]                           # 'M' / 'F'  (constant within a cage)
    cond = ev["condition"]                     # 'pre' / 'dep' / 'post'  (varies WITHIN a cage)
    ranks = ev["ranks"]                        # (N,3) approacher/approachee/bystander rank

    import pandas as pd
    hero_i = 909                               # Cage-15 male aggression event (the Hero); lives in C4
    return cage, cond, der, ev, hero_i, labels, pd, ranks, sex, sweep


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Deeper: the paper & where the analogy stops": mo.md(
                r"""
                **The neural twin.** Testing whether a manipulation moves the distribution of states
                across a *set of clusters* is the **cluster-based permutation** framework of
                **Maris & Oostenveld (2007, *J. Neurosci. Methods*)** — the standard for EEG/MEG that
                controls the family-wise error from testing many channels/time-points at once.
                **Kingsbury et al. (2019, *Cell*)** decoded dominance from *inter-brain* dmPFC coupling —
                a population read-out gated by the same "does this variable shift the state?" logic.

                **Shared mathematics.** Contingency-table chi-square + a permutation null that shuffles at
                the *correct exchangeable unit* — for us the **cage**, for them the **subject/trial block**.

                **Species / preparation tag.** Maris & Oostenveld: human M/EEG. Kingsbury: freely-moving
                **mice**, dmPFC electrophysiology. Our data: freely-moving mice, pose only.

                **Where the analogy stops.** Our "states" are clusters of *behavior*, not neurons; and our
                exchangeable unit is a **cage of three mice**, of which we have only seven. The math is
                identical; the power is not — that mismatch is the whole lesson below.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(cu, ev, der, np, go, pd, ROOT, labels):
    # ---- Readout Board builder (shared by the top + bottom panels) --------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score

    _y = ev["agg_label"].astype(int)
    _pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=1000))])
    student_auroc = float(cross_val_score(_pipe, der["X"], _y, cv=5, scoring="roc_auc").mean())
    n_syll = int(len({int(l) for l in labels if l >= 0}))

    try:
        _bdf = pd.read_csv(cu.data_path("data/readout_board.csv", ROOT))
    except Exception:
        _bdf = None

    def make_board(subtitle=""):
        _bench = None
        if _bdf is not None:
            _r = _bdf[(_bdf["gauge"] == "B") & (_bdf["notebook"] == "NB06")]
            if len(_r):
                _bench = float(_r["value"].iloc[0])
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number", value=n_syll,
            number={"font": {"size": 46}},
            title={"text": "Gauge A - representation<br>"
                           "<span style='font-size:0.75em;color:#888'>behavioral syllables on the map</span>"},
            domain={"x": [0.0, 0.46], "y": [0, 1]}))
        _ind = dict(
            mode=("gauge+number+delta" if _bench is not None else "gauge+number"),
            value=round(student_auroc, 3),
            number={"font": {"size": 40}, "valueformat": ".3f"},
            title={"text": "Gauge B - held-out readiness<br>"
                           "<span style='font-size:0.75em;color:#888'>features -> aggression (train CV AUROC)</span>"},
            gauge={"axis": {"range": [0.5, 1.0]}, "bar": {"color": "#4c78a8"}},
            domain={"x": [0.54, 1.0], "y": [0, 1]})
        if _bench is not None:
            _ind["delta"] = {"reference": _bench, "valueformat": ".3f"}
            _ind["gauge"]["threshold"] = {"line": {"color": "#e45756", "width": 3}, "value": _bench}
        fig.add_trace(go.Indicator(**_ind))
        fig.update_layout(template="plotly_white", height=230, margin=dict(l=30, r=30, t=78, b=10),
                          title=dict(text="READOUT BOARD" + (("  -  " + subtitle) if subtitle else ""),
                                     font=dict(size=17)))
        return fig
    return make_board, n_syll, student_auroc


@app.cell(hide_code=True)
def _(make_board, mo):
    mo.vstack([
        mo.md(r"""
        **Readout Board - top of NB06.** Gauge A held at **5 syllables** (Phase 1 finished carving the
        map). Gauge B is your own logistic decoder's cross-validated AUROC, shown beside the pinned
        benchmark (red line). Today we don't move the gauges - we ask whether the *claims* we make about
        the map survive honest statistics.
        """),
        make_board("do the effects on the map survive honest stats?"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sealed Cage 16 - the animal on the rig
        <div style="border:2px dashed #b33; border-radius:10px; padding:14px 18px; background:
        repeating-linear-gradient(45deg,#f6f6f6,#f6f6f6 12px,#efefef 12px,#efefef 24px);">
        <b>Camera 16 &middot; ~470 events &middot; ground-truth labels present but withheld</b><br>
        <span style="filter:blur(3px);opacity:0.45;letter-spacing:2px;">skeletons &#9619;&#9619;&#9619;&#9619; labels &#9608;&#9608;&#9608;&#9608;&#9608;&#9608; aggression &#9608;&#9608; non-agg &#9608;&#9608;&#9608;&#9608;</span><br>
        <span style="color:#b33;font-weight:600;">Unlocks in 2 notebooks (NB08).</span>
        <span style="color:#777;">We test the honest-statistics logic here so that when the seal breaks,
        the held-out numbers mean something.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 &middot; The Hero, on the map

        **Event #909** - a Cage-15 (male) aggression approach, the clip we have followed since NB01 - is
        now a single point on the behavioral map. It lands in **syllable C4**, the aggression-enriched
        cluster (agg fraction 0.42, 1.40x the 0.30 base rate). Keep an eye on C4: it is exactly the
        cluster whose "exciting sex effect" we are about to put on trial.
        """
    )
    return


@app.cell(hide_code=True)
def _(cu, sweep, labels, hero_i, ev, np, go, mo):
    _emb = sweep["emb_grid"][tuple(sweep["default_ij"])]
    _pal = {-1: "#d5d5d5", 0: "#8c9196", 1: "#f2b134", 2: "#4c78a8", 3: "#b279a2", 4: "#e45756"}
    _fig = go.Figure()
    for _c in [-1, 0, 1, 2, 3, 4]:
        _m = labels == _c
        _fig.add_trace(go.Scattergl(
            x=_emb[_m, 0], y=_emb[_m, 1], mode="markers",
            name=("noise" if _c < 0 else f"C{_c}"),
            marker=dict(size=4, opacity=0.55, color=_pal[_c])))
    _fig.add_trace(go.Scatter(
        x=[_emb[hero_i, 0]], y=[_emb[hero_i, 1]], mode="markers", name="Hero #909",
        marker=dict(size=20, color="black", symbol="star",
                    line=dict(color="white", width=1.5))))
    _fig.update_layout(template="plotly_white", height=430,
                       title="The behavioral map - Hero #909 sits in C4",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_xaxes(showgrid=False, showticklabels=False)
    _fig.update_yaxes(showgrid=False, showticklabels=False)

    _gif = cu.event_gif_bytes(ev["kp"][hero_i], ev["ranks"][hero_i],
                              int(ev["contact_rel"][hero_i]), cell=190)
    mo.vstack([
        mo.md(r"**Hero #909** (Cage 15 &middot; male &middot; aggression &middot; syllable **C4**):"),
        mo.Html(cu.gif_img_html(_gif, width=200)),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 &middot; Did hunger rewrite the ethogram?  (the *clean* headline)

        We test, per syllable, whether its **pre / dep / post** composition differs from the rest of the
        map (chi-square of cluster-vs-rest), then **Bonferroni-correct** across the syllables we tested.

        Why is this the *honest* headline? Because **condition varies *within* every cage** - each cage
        was recorded pre, deprived, and post. So the cage is not confounded with condition, and an
        event-level test is legitimate. This is the case where the naive math is actually valid.

        The bars below show -log10(p) for each syllable: the raw p (grey) versus the Bonferroni-corrected
        p (blue). The dashed line is the alpha = 0.05 threshold. Watch **C4**: its raw p clears the line,
        but multiplying by the 5 tests we ran drags it back under - a false positive we would have shipped
        without the correction. **C0** survives.
        """
    )
    return


@app.cell(hide_code=True)
def _(cu, labels, cond, np, go, mo):
    cond_results = cu.condition_enrichment(labels, cond)
    _clu = [f"C{r['cluster']}" for r in cond_results]
    _praw = np.array([r["p"] for r in cond_results])
    _pbon = np.array([r["p_bonf"] for r in cond_results])
    _enr = [r["enriched"] for r in cond_results]
    _fig = go.Figure()
    _fig.add_trace(go.Bar(x=_clu, y=-np.log10(_praw), name="raw p",
                          marker_color="#bcbcbc",
                          text=[f"enr:{e}" for e in _enr], textposition="outside"))
    _fig.add_trace(go.Bar(x=_clu, y=-np.log10(_pbon), name="Bonferroni p",
                          marker_color="#4c78a8"))
    _fig.add_hline(y=-np.log10(0.05), line=dict(color="#e45756", width=2, dash="dash"),
                   annotation_text="alpha = 0.05")
    _fig.update_layout(template="plotly_white", height=390, barmode="group",
                       title="Condition enrichment per syllable - raw vs Bonferroni",
                       yaxis_title="-log10(p)", margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_xaxes(showgrid=False)
    _min_bonf = float(min(r["p_bonf"] for r in cond_results))
    mo.vstack([
        mo.md(f"**Headline:** at least one syllable survives Bonferroni - smallest corrected "
              f"p = **{_min_bonf:.4f}** (syllable C{cond_results[0]['cluster']}, enriched in "
              f"*{cond_results[0]['enriched']}*). Hunger **does** shift the ethogram, honestly. "
              f"Note C4 (the aggression cluster) is 47% deprived-phase events, but its corrected "
              f"p is only ~0.06 - *not* significant once we pay for 5 tests."),
        _fig,
    ])
    return (cond_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 &middot; The result everyone loved  ->  the reversal

        Now the exciting one. Syllable **C4** - the aggression cluster - looks dramatically
        **male-skewed**. Run the naive event-level test and the number is breathtaking.
        """
    )
    return


@app.cell(hide_code=True)
def _(cu, labels, sex, np, go, mo):
    _ig = labels == 4
    _res = cu.covariate_enrichment(_ig, sex)          # EVENT-LEVEL (the seductive, wrong test)
    _inM = float((sex[_ig] == "M").mean()) * 100
    _outM = float((sex[~_ig] == "M").mean()) * 100
    _fig = go.Figure(go.Bar(
        x=["C4 (aggression)", "rest of map"], y=[_inM, _outM],
        marker_color=["#e45756", "#9aa0a6"],
        text=[f"{_inM:.0f}% male", f"{_outM:.0f}% male"], textposition="outside"))
    _fig.update_layout(template="plotly_white", height=340,
                       title=f"C4 is {_inM:.0f}% male vs {_outM:.0f}% elsewhere",
                       yaxis_title="% of events from male cages", yaxis_range=[0, 100],
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([
        mo.md(f"### Event-level test: **chi2 = {_res['chi2']:.1f},  p = {_res['p']:.2g}**\n\n"
              f"A p-value of ~3e-7. *Aggression is a male behavior* - write the paper, book the "
              f"seminar. **This is where most analyses stop.** Do not stop here."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### But look at the *cages*

        There are only **seven cages**: four male (9, 11, 13, 15) and three female (10, 12, 14). Sex is
        not measured on 1500 independent animals - it is measured on **7 cages**, and it is **100%
        confounded with cage identity**. The 1500 events are not 1500 samples of "sex"; they are
        *pseudoreplicates* of 7. Plot the C4 fraction **per cage** and the illusion breaks:
        """
    )
    return


@app.cell(hide_code=True)
def _(cage, sex, labels, np, go, mo):
    _units = np.unique(cage)
    _frac = np.array([(labels[cage == u] == 4).mean() for u in _units]) * 100
    _sx = np.array([sex[cage == u][0] for u in _units])
    _col = ["#4c78a8" if s == "M" else "#e45756" for s in _sx]
    _fig = go.Figure(go.Bar(
        x=[f"cage {u} ({s})" for u, s in zip(_units, _sx)], y=_frac,
        marker_color=_col, text=[f"{f:.0f}%" for f in _frac], textposition="outside"))
    _fig.update_layout(template="plotly_white", height=360,
                       title="C4 share per cage - the 'sex effect' is really cages 9 & 11",
                       yaxis_title="% of that cage's events in C4",
                       margin=dict(l=10, r=10, t=44, b=10), showlegend=False)
    _fig.update_xaxes(showgrid=False)
    mo.vstack([
        mo.md(r"Blue = male cages, red = female cages. The 'male aggression' signal is **two cages** "
              r"(9 and 11) towering over everyone. Male cages 13 and 15 sit right down with the females. "
              r"This is not a sex effect - it is a **cage effect** that happens to line up with sex."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The honest test: shuffle at the level of the *cage*

        The exchangeable unit is the cage, not the event. So we build the null by **permuting the
        sex label across the 7 cages** (4 M / 3 F kept fixed), recomputing the C4 male-share gap each
        time. If sex truly drove C4, the real gap should sit far in the tail. It does not.
        `cu.permutation_test(in_group, sex, unit=cage, n=5000)` returns the honest p.
        """
    )
    return


@app.cell(hide_code=True)
def _(cu, cage, sex, labels, np, go, mo):
    _ig = labels == 4
    _res = cu.permutation_test(_ig, sex, cage, n=5000, seed=0)   # cage-level (the honest test)
    sex_cage_p = float(_res["p_emp"])
    _obs = float(_res["stat"])

    # Reconstruct the null distribution for the histogram (same recipe as the helper).
    _rng = np.random.RandomState(1)
    _, _covnum = np.unique(sex, return_inverse=True)
    _units = np.unique(cage)
    _vals = np.array([_covnum[cage == u][0] for u in _units], float)
    def _stat(c):
        return abs(c[_ig].mean() - c[~_ig].mean())
    _null = np.empty(3000)
    for _k in range(3000):
        _m = dict(zip(_units, _rng.permutation(_vals)))
        _null[_k] = _stat(np.array([_m[u] for u in cage], float))

    _fig = go.Figure()
    _fig.add_trace(go.Histogram(x=_null, nbinsx=24, marker_color="#c7c7c7",
                                name="cage-shuffled null"))
    _fig.add_vline(x=_obs, line=dict(color="#e45756", width=3),
                   annotation_text="observed", annotation_position="top")
    _fig.update_layout(template="plotly_white", height=350,
                       title=f"Cage-level permutation null - observed gap is ordinary  (p = {sex_cage_p:.3f})",
                       xaxis_title="|male-share(C4) - male-share(rest)|",
                       margin=dict(l=10, r=10, t=44, b=10), showlegend=False)
    _fig.update_xaxes(showgrid=False)
    mo.vstack([
        mo.md(f"### Cage-level test: **p = {sex_cage_p:.3f}**\n\n"
              f"The observed gap sits comfortably *inside* the cloud of what random cage-relabelings "
              f"produce. The p-value went from **3e-7 to ~0.15** the moment we honored the real unit. "
              f"**The number you loved was pseudoreplication.** With 4 male vs 3 female cages we simply "
              f"do not have the power to say anything about sex - and no amount of events fixes that."),
        _fig,
    ])
    return (sex_cage_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Does the reversal hold for *any* cluster? (explore)

        Pick a syllable. We show its **event-level** sex p (naive) beside its **cage-level** permutation
        p (honest). Every cluster tells the same story: event-level significance evaporates at the cage
        level, because with 7 cages there is nothing there to find.
        """
    )
    return


@app.cell
def _(mo):
    cluster_pick = mo.ui.dropdown(
        options={"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4},
        value="C4", label="syllable")
    return (cluster_pick,)


@app.cell(hide_code=True)
def _(cluster_pick, cu, cage, sex, labels, go, mo):
    _c = cluster_pick.value
    _ig = labels == _c
    _event_p = cu.covariate_enrichment(_ig, sex)["p"]
    _cage_p = cu.permutation_test(_ig, sex, cage, n=2000, seed=0)["p_emp"]
    _fig = go.Figure(go.Bar(
        x=["event-level (naive)", "cage-level (honest)"],
        y=[_event_p, _cage_p],
        marker_color=["#9aa0a6", "#4c78a8"],
        text=[f"{_event_p:.2g}", f"{_cage_p:.3f}"], textposition="outside"))
    _fig.add_hline(y=0.05, line=dict(color="#e45756", width=2, dash="dash"),
                   annotation_text="alpha = 0.05")
    _fig.update_layout(template="plotly_white", height=330,
                       title=f"C{_c}: sex p - event vs cage level", yaxis_title="p-value",
                       yaxis_type="log", margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([cluster_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 &middot; Rank dyads - with a loud caveat

        Finally, the directed **rank dyad** composition of each syllable (who-approaches-whom by rank:
        Dom>Sub, Mid>Dom, ...). C4 shows a **Dom>Mid** enrichment that even survives Bonferroni. Before
        you believe it, read the caveat.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Rank labels are unreliable - treat every rank result as provisional.**

            Rank here comes from **tail-mark identity**, which carries roughly **16% ID error** (the tail
            chain drops out exactly when mice are in contact - i.e. during aggression, the very events in
            C4). Two independent problems compound:

            1. **Tube-test rank != homecage aggression.** They are correlated but *dissociable* axes; a
               dyad chi-square on tube-test rank is not a test of who fights whom.
            2. **Which way does 16% mislabeling bias the test?** *Random* misclassification usually
               **attenuates toward the null** (dilutes real structure -> makes a true effect *harder* to
               detect). But this error is **not random**: tail dropout is correlated with the contact
               posture of aggression, so it can *manufacture* apparent rank structure precisely inside
               C4. We cannot sign the bias - which is why this stays a flag, not a finding.
            """
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(cu, labels, ranks, np, go, mo):
    rank_results = cu.rank_dyad_enrichment(labels, ranks[:, 0], ranks[:, 1])
    _c4 = next((r for r in rank_results if r["cluster"] == 4), None)
    _dyads = list(cu.DYAD_LABELS)
    _fracs = [(_c4["dyad_fracs"][d] * 100 if _c4 else 0) for d in _dyads]
    # color each directed dyad by the APPROACHER's rank (RANK_HEX): Dom red / Mid blue / Sub green
    _appr_rank = [d[0] for d in cu.DYADS]
    _col = [cu.RANK_HEX[r] for r in _appr_rank]
    _fig = go.Figure(go.Bar(x=_dyads, y=_fracs, marker_color=_col,
                            text=[f"{f:.0f}%" for f in _fracs], textposition="outside"))
    _fig.update_layout(template="plotly_white", height=350,
                       title="C4 directed rank-dyad composition (bar color = approacher rank)",
                       yaxis_title="% of C4 events", margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_xaxes(showgrid=False)
    _msg = (f"C4 rank-dyad chi2 = {_c4['chi2']:.1f}, p = {_c4['p']:.4f}, "
            f"Bonferroni p = {_c4['p_bonf']:.3f} -> enriched dyad **{_c4['enriched_dyad']}**."
            if _c4 else "C4 not testable.")
    mo.vstack([
        mo.md(_msg + "  *Provisional - see the caveat above; the same 16% ID error that could dilute this"
                     " could also have fabricated it.*"),
        _fig,
    ])
    return (rank_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise - grade the two claims honestly

        **Toolbox** (inputs -> outputs):

        - `cu.condition_enrichment(labels, condition)` -> *list of per-cluster dicts* with `p`, `p_bonf`,
          `enriched`. Condition is **within-cage**, so this event-level test is legitimate.
        - `cu.covariate_enrichment(in_group_mask, covariate)` -> `{chi2, p, ...}`. **Event-level**
          chi-square; `in_group_mask` is a boolean `labels == c`, **not** a labels array.
        - `cu.permutation_test(in_group_mask, covariate, unit, n=5000)` -> `{stat, p_emp}`. Shuffles the
          covariate **at the `unit` level** (pass `cage`). This is the antidote to pseudoreplication.

        **Hypothesis banner (pre-registered):** *Food deprivation changes cluster composition and it
        survives when the cage - not the event - is the unit; the sex difference does not.*

        **Your task.** Fill in the one flagged line so `sex_test_p` is the **cage-level** p, not the
        event-level p, then read the verdict.
        """
    )
    return


@app.cell
def _(cu, labels, sex, cage, cond):
    def student_reversal(labels, sex, cage, cond):
        C = 4                                  # the aggression cluster
        in_group = labels == C

        # (1) CONDITION - within-cage, so the event-level Bonferroni test is honest:
        cond_res = cu.condition_enrichment(labels, cond)
        condition_min_bonf = min(r["p_bonf"] for r in cond_res)

        # (2) SEX - the naive event-level p (seductive but pseudoreplicated):
        sex_event_p = cu.covariate_enrichment(in_group, sex)["p"]

        # (3) TODO: replace the line below so sex_test_p is the CAGE-LEVEL permutation p.
        #     Hint:  cu.permutation_test(in_group, sex, unit=cage, n=5000)["p_emp"]
        sex_test_p = sex_event_p               # <-- naive placeholder; FIX ME

        return dict(condition_min_bonf=float(condition_min_bonf),
                    sex_event_p=float(sex_event_p),
                    sex_test_p=float(sex_test_p))

    student_verdict = student_reversal(labels, sex, cage, cond)
    return (student_verdict,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Reveal solution": mo.md(
                r"""
                Replace the flagged line with the cage-level permutation:

                ```python
                sex_test_p = cu.permutation_test(in_group, sex, unit=cage, n=5000)["p_emp"]
                ```

                Then `sex_test_p ~ 0.15` (was `3e-7` at the event level). The **graded correct verdict
                for sex is "cannot conclude - pseudoreplicated / underpowered at n = 4 vs 3 cages."**
                Condition, by contrast, is within-cage and survives Bonferroni (`min p_bonf ~ 0.003`).
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(student_verdict, mo):
    _cmb = student_verdict["condition_min_bonf"]
    _sxp = student_verdict["sex_test_p"]
    _cond_ok = _cmb < 0.05                              # condition survives Bonferroni  (~0.003)
    _sex_ok = 0.05 < _sxp < 0.6                         # honest: cannot conclude        (~0.15)
    _naive = _sxp < 1e-3                                # still the event-level placeholder?
    _pass = _cond_ok and _sex_ok
    if _pass:
        _kind, _head = "success", "PASS - you graded both claims honestly"
        _body = (f"Condition survives Bonferroni (min p_bonf = {_cmb:.4f} < 0.05) - hunger really does "
                 f"reshape the ethogram. Sex does **not** survive cage-level shuffling "
                 f"(p_emp = {_sxp:.3f}); the correct verdict is *cannot conclude - pseudoreplicated / "
                 f"underpowered at n = 4 vs 3 cages.*")
    elif _naive:
        _kind, _head = "danger", "Not yet - you are still using the event-level p for sex"
        _body = (f"`sex_test_p = {_sxp:.2g}` is the naive event-level number. It **looks** significant, "
                 f"but sex is 100% confounded with cage - that p is pseudoreplication. Swap in "
                 f"`cu.permutation_test(in_group, sex, unit=cage, n=5000)['p_emp']` and re-read the verdict.")
    else:
        _kind, _head = "warn", "Close - check the band"
        _body = (f"condition_min_bonf = {_cmb:.4f} (want < 0.05); sex_test_p = {_sxp:.3f} "
                 f"(want the honest cage-level value, roughly 0.05-0.6). The graded-correct sex verdict "
                 f"is *cannot conclude - pseudoreplicated.*")
    mo.callout(mo.md(f"**{_head}**\n\n{_body}"), kind=_kind)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Conceptual questions": mo.md(
                r"""
                1. **Why does testing many clusters inflate false positives?** With 5 tests at
                   alpha = 0.05, the chance of *at least one* false hit is ~ 1 - 0.95^5 ~ 0.23. Bonferroni
                   multiplies each p by the number of tests to hold the family-wise error at 0.05.
                2. **Why trust condition (within-cage) over sex (between-cage)?** Every cage supplies pre,
                   dep, and post events, so condition is not confounded with cage; sex is a fixed property
                   of the cage, so a "sex" test is really a 7-cage comparison in disguise.
                3. **Which way does 16% rank mislabeling bias a dyad chi-square?** *Random* error
                   attenuates toward the null (harder to detect a real effect); *systematic* error (tail
                   dropout during contact) can manufacture a spurious one. You cannot sign it without
                   characterizing the error - so rank stays a flag.
                4. **Where does chi-square break?** Small expected cell counts (rule of thumb < 5) make the
                   chi-square approximation unreliable - use an exact/permutation test there.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we threw away / how it breaks

        - **We discarded the event as the unit of inference.** Collapsing 1500 events to 7 cages is a
          brutal loss of apparent n - but it is the *honest* n. Everything that looked significant at the
          event level and vanishes at the cage level was never real.
        - **Failure mode 1 - pseudoreplication (the one that just bit us):** any between-cage variable
          (sex, genotype, a drug given per-cage) will look wildly significant at the event level and is
          almost always an artifact. Always ask *what is the exchangeable unit?*
        - **Failure mode 2 - multiple comparisons:** C4's condition effect (raw p = 0.012) evaporates
          under Bonferroni. Scan enough clusters x variables and something will always "reach p < 0.05."
        - **Failure mode 3 - small-cell chi-square:** the rarest syllables (C3, n = 32) have thin dyad
          cells; the chi-square p-value is only approximate there.
        - **Open-ended:** *With only 7 cages, what design or analysis would give you real power to test a
          sex effect?* (More cages? A within-animal manipulation? A mixed-effects model with cage as a
          random intercept? Sketch the one you would run and the n it needs.)
        """
    )
    return


@app.cell(hide_code=True)
def _(make_board, mo):
    mo.vstack([
        mo.md(r"""
        **Readout Board - bottom of NB06.** Gauge B unchanged in *value*, but its *meaning* is now
        earned: we validated **what the map encodes** - condition, yes (within-cage, Bonferroni-clean);
        sex, **not from this design** (7 cages, no power). An enrichment test you can trust is worth more
        than a decoder you cannot. This is precisely how a circuit lab will read out the opto trial:
        *did the manipulation move the state distribution, at the correct unit?*
        """),
        make_board("validated: condition YES - sex NOT from this design"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we ship next

        We now know how to read the map **without fooling ourselves**: enrichment with the right
        multiple-comparison correction and the right exchangeable unit. Condition genuinely reshapes the
        ethogram; the beloved sex effect was pseudoreplication.

        But so far every event has been a **frozen snapshot**. Behavior *moves* - a sniff becomes a chase
        becomes a fight. **NB07** models the **grammar** of how syllables follow one another (an observed
        Markov chain, the honest cousin of the HMMs that infer *hidden* brain states) and lays it over the
        **activity clock** of a single continuously-tracked cage. The baton passes from Hero #909 to the
        whole day it lived inside.
        """
    )
    return


if __name__ == "__main__":
    app.run()

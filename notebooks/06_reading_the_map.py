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
        # NB06 · Reading the Map — doing the statistics honestly

        You have built a behavioral map: every approach event is now a point on it, and nearby points
        share a posture-and-motion pattern we call a **syllable** (the clusters C0–C4). A natural next
        question is: *which syllables are over-represented under which condition?* For example, does
        food deprivation make aggression-like syllables more common? Does one syllable occur mostly in
        males?

        Answering questions like these requires a **statistical test**, and the goal of this notebook
        is to run that test **honestly**, so the conclusions hold up to scrutiny. (In plain terms, this
        is also how a neuroscientist would later check whether a manipulation changed an animal's
        behavior: by asking whether the distribution of behavioral states shifted.)

        We will meet two common traps that can turn a real-looking result into a false one —
        **multiple comparisons** and **pseudoreplication** — and learn the standard correction for
        each. Read the definitions below before starting.
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
    example_i = 909                            # our running example approach event (Cage 15); lives in C4
    return cage, cond, der, ev, example_i, labels, pd, ranks, sex, sweep


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Key terms (read first)": mo.md(
                r"""
                - **Enrichment test.** A test of whether one category is over- or under-represented
                  *inside* a group compared to *outside* it. Here: is a condition or sex
                  over-represented inside one syllable versus the rest of the map?

                - **Chi-square test.** Compares the counts you actually observe in a table
                  (for example, syllable × condition) with the counts you would expect if the two were
                  unrelated. A large chi-square means the observed composition differs from
                  independence more than chance would produce, which gives a small p-value.
                  *Input:* a contingency table of counts. *Output:* a chi-square statistic and a
                  p-value.

                - **Multiple comparisons.** Each single test at α = 0.05 has a 5% chance of a false
                  positive. Run 5 independent tests and the chance of *at least one* false positive
                  rises to about 1 − 0.95⁵ ≈ 23%. The **Bonferroni correction** controls this by
                  multiplying each p-value by the number of tests (equivalently, dividing the
                  threshold), keeping the overall false-positive rate near 5%.

                - **Pseudoreplication.** Treating measurements that are not independent as if they
                  were. Sex is a fixed property of a **cage**, and we have only 7 cages. So 1500 events
                  give us 7 independent observations of sex, not 1500. The correct **unit of analysis**
                  (the "exchangeable unit") is the cage.

                - **Permutation test.** Builds the comparison ("null") distribution by shuffling labels
                  many times *at the correct unit* — here, shuffling sex across the 7 cages — and
                  checking where the real value falls. *Input:* a group mask, a covariate, a unit id.
                  *Output:* an empirical p-value.
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
        # Gauge A = size of the representation (number of syllables on the map).
        # Gauge B = held-out readiness (cross-validated AUROC), with the pinned benchmark drawn as a
        # red threshold line. FIX: Gauge B uses mode="gauge+number" only. It previously added a
        # delta whose reference equalled the value, rendering a meaningless ~0.000; the pinned
        # benchmark is now shown as the red threshold line instead of a delta.
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
            mode="gauge+number",
            value=round(student_auroc, 3),
            number={"font": {"size": 40}, "valueformat": ".3f"},
            title={"text": "Gauge B - held-out readiness<br>"
                           "<span style='font-size:0.75em;color:#888'>features -> aggression (train CV AUROC)</span>"},
            gauge={"axis": {"range": [0.5, 1.0]}, "bar": {"color": "#4c78a8"}},
            domain={"x": [0.54, 1.0], "y": [0, 1]})
        if _bench is not None:
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
        **Readout Board — top of NB06.** Gauge A shows the size of our representation: **5 syllables**
        on the map (Phase 1 is finished carving it). Gauge B is your logistic decoder's
        cross-validated AUROC from NB05, with the pinned benchmark drawn as a red line. Today we do not
        move the gauges — we ask whether the **claims** we make about the map survive honest
        statistics.
        """),
        make_board("do the effects on the map survive honest stats?"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Held-out camera 16

        A separate set of about 470 events recorded on camera 16 has ground-truth labels that we keep
        hidden until **NB08**. We practice honest statistics here so that when we finally score the
        held-out set, the numbers mean something. Gauge B will only start moving in Phase 2.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Our example event, on the map

        **Event #909** — the Cage-15 approach we have followed since NB01 — is now a single point on
        the behavioral map. In this event the **approacher** is the Dom mouse (red) and the
        **approachee** is the Sub mouse (green). The event lands in syllable **C4**, the cluster with
        the highest fraction of aggression (aggression fraction 0.42, about 1.4× the 0.30 base rate).
        C4 is the syllable whose apparent sex difference we will test later in this notebook.
        """
    )
    return


@app.cell(hide_code=True)
def _(cu, sweep, labels, example_i, ev, np, go, mo):
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
        x=[_emb[example_i, 0]], y=[_emb[example_i, 1]], mode="markers", name="example #909",
        marker=dict(size=20, color="black", symbol="star",
                    line=dict(color="white", width=1.5))))
    _fig.update_layout(template="plotly_white", height=430,
                       title="The behavioral map - example event #909 sits in C4",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_xaxes(showgrid=False, showticklabels=False)
    _fig.update_yaxes(showgrid=False, showticklabels=False)

    _gif = cu.event_gif_bytes(ev["kp"][example_i], ev["ranks"][example_i],
                              int(ev["contact_rel"][example_i]), cell=190)
    mo.vstack([
        mo.md(r"**Example event #909** (Cage 15 · approacher = Dom/red, approachee = Sub/green · "
              r"syllable **C4**):"),
        mo.Html(cu.gif_img_html(_gif, width=200)),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(cu, ev, labels, np, mo):
    # A syllable is easier to understand by watching a few of its members than by reading about it.
    # Render six events drawn from C4 as a grid of rank-colored skeleton GIFs.
    _c4 = np.where(labels == 4)[0][:6]
    _events = [(ev["kp"][int(i)], ev["ranks"][int(i)], int(ev["contact_rel"][int(i)])) for i in _c4]
    _grid = cu.grid_gif_bytes(_events, ncols=3, cell=150)
    mo.vstack([
        mo.md(r"**Six example events from syllable C4** (mice colored by rank: Dom red, Int blue, "
              r"Sub green). C4 groups close, contact-heavy approaches. This is what the "
              r"aggression-enriched syllable actually looks like — useful to remember when we start "
              r"asking which conditions produce more of it."),
        mo.Html(cu.gif_img_html(_grid, width=470)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · Did food deprivation change the ethogram?  (a valid test)

        For each syllable we ask whether its **pre / dep / post** composition differs from the rest of
        the map. The tool is a **chi-square test** on a small table: (this syllable vs the rest) ×
        (pre / dep / post). Because we run one test per syllable — 5 tests — we then apply a
        **Bonferroni** correction so the chance of a false positive across all 5 stays near 5%.

        Why is this test valid at the event level? Because **condition varies within every cage** —
        each cage was recorded pre, deprived, and post. Condition is therefore not confounded with
        cage identity, so counting individual events is legitimate here.

        The bars below show −log10(p) for each syllable: the raw p (grey) versus the
        Bonferroni-corrected p (blue). The dashed line is the α = 0.05 threshold. Note **C4**: its raw
        p clears the line, but after paying for 5 tests the corrected p drops back below it — a false
        positive we would have reported without the correction. **C0** survives.
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
        mo.md(f"**Result:** at least one syllable survives Bonferroni — the smallest corrected "
              f"p = **{_min_bonf:.4f}** (syllable C{cond_results[0]['cluster']}, enriched in "
              f"*{cond_results[0]['enriched']}*). Food deprivation does shift the ethogram, and the "
              f"shift holds after correction. Note that C4 (the aggression syllable) is 47% "
              f"deprived-phase events, but its corrected p is only about 0.06 — not significant once "
              f"we account for the 5 tests."),
        _fig,
    ])
    return (cond_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · An apparent sex difference in C4

        Now a different comparison. Syllable **C4** — the aggression syllable — looks strongly
        **male-skewed**. Running the naive event-level chi-square gives a very small p-value. This is
        exactly the kind of number to be careful with, so we will examine it before accepting it.
        """
    )
    return


@app.cell(hide_code=True)
def _(cu, labels, sex, np, go, mo):
    _ig = labels == 4
    _res = cu.covariate_enrichment(_ig, sex)          # EVENT-LEVEL test (to be checked below)
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
        mo.md(f"### Event-level test: chi2 = {_res['chi2']:.1f}, p = {_res['p']:.2g}\n\n"
              f"Taken at face value, a p-value of about 3e-7 would say that aggression is a male "
              f"behavior. Many analyses would stop here. Before accepting the conclusion, we check "
              f"whether the event is the correct unit of analysis."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Look at the cages

        There are only **seven cages**: four male (9, 11, 13, 15) and three female (10, 12, 14). Sex
        is not measured on 1500 independent animals — it is a fixed property of each cage, so it is
        measured on **7 cages** and is **completely confounded with cage identity**. The 1500 events
        are not 1500 independent samples of "sex"; they are *pseudoreplicates* of 7. Plotting the C4
        fraction **per cage** shows this directly:
        """
    )
    return


@app.cell(hide_code=True)
def _(cage, sex, labels, np, go, mo):
    _units = np.unique(cage)
    _frac = np.array([(labels[cage == u] == 4).mean() for u in _units]) * 100
    _sx = np.array([sex[cage == u][0] for u in _units])
    _col = ["#762a83" if s == "M" else "#f1a340" for s in _sx]
    _fig = go.Figure(go.Bar(
        x=[f"cage {u} ({s})" for u, s in zip(_units, _sx)], y=_frac,
        marker_color=_col, text=[f"{f:.0f}%" for f in _frac], textposition="outside"))
    _fig.update_layout(template="plotly_white", height=360,
                       title="C4 share per cage - the apparent sex effect is two male cages (9 & 11)",
                       yaxis_title="% of that cage's events in C4",
                       margin=dict(l=10, r=10, t=44, b=10), showlegend=False)
    _fig.update_xaxes(showgrid=False)
    mo.vstack([
        mo.md(r"Purple = male cages, orange = female cages. The male-aggression signal comes from "
              r"**two cages** (9 and 11) sitting well above the rest. Male cages 13 and 15 sit down "
              r"among the female cages. This is not a sex effect; it is a **cage effect** that happens "
              r"to line up with sex."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The honest test: shuffle at the level of the cage

        The exchangeable unit is the cage, not the event. So we build the comparison distribution by
        **permuting the sex label across the 7 cages** (keeping 4 male / 3 female fixed) and
        recomputing the C4 male-share gap each time. If sex genuinely drove C4, the real gap should
        sit far out in the tail of this distribution. It does not.
        `cu.permutation_test(in_group, sex, unit=cage, n=5000)` returns this cage-level p-value.
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
        mo.md(f"### Cage-level test: p = {sex_cage_p:.3f}\n\n"
              f"The observed gap sits comfortably inside the distribution produced by random cage "
              f"relabelings. The p-value moved from about 3e-7 (event level) to about 0.15 (cage "
              f"level) once we used the correct unit. With 4 male versus 3 female cages there is not "
              f"enough power to draw any conclusion about sex, and adding more events does not change "
              f"that. This is the lesson of the notebook: choose the right unit, and report what the "
              f"data can actually support."),
        _fig,
    ])
    return (sex_cage_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Does the same pattern hold for any cluster?  (explore)

        Pick a syllable. We show its **event-level** sex p (naive) beside its **cage-level**
        permutation p (honest). For every cluster the story is the same: event-level significance
        disappears at the cage level, because with only 7 cages there is not enough power to detect a
        real sex effect if one existed.
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
        ## 4 · Rank dyads — with an important caveat

        Finally, we look at the directed **rank-dyad** composition of each syllable: who approaches
        whom, by rank (Dom→Sub, Mid→Dom, and so on). C4 shows a **Dom→Mid** enrichment that even
        survives Bonferroni. Before interpreting it, read the caveat below.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Rank labels are unreliable — treat every rank result as provisional.**

            Rank here comes from **tail-mark identity**, which carries roughly **16% identity error**.
            The tail keypoints tend to drop out exactly when mice are in contact — that is, during
            aggression, the very events that make up C4. Two separate problems compound:

            1. **Tube-test rank is not the same as homecage aggression.** The two are correlated but
               distinct measures; a dyad chi-square on tube-test rank is not directly a test of who
               fights whom.
            2. **The direction of the bias is unknown.** *Random* misclassification usually pushes a
               result *toward* the null (it dilutes real structure, making a true effect harder to
               detect). But this error is **not random**: tail dropout is tied to the contact posture
               of aggression, so it could instead *manufacture* apparent rank structure inside C4. We
               cannot sign the bias, so this stays a flag, not a finding.
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
            f"Bonferroni p = {_c4['p_bonf']:.3f}, enriched dyad **{_c4['enriched_dyad']}**."
            if _c4 else "C4 not testable.")
    mo.vstack([
        mo.md(_msg + "  *Provisional — see the caveat above; the same 16% identity error that could "
                     "dilute this effect could also have produced it.*"),
        _fig,
    ])
    return (rank_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise — grade the two claims honestly

        **Toolbox** (inputs → outputs):

        - `cu.condition_enrichment(labels, condition)` → a list of per-cluster dicts with `p`,
          `p_bonf`, `enriched`. Condition is **within-cage**, so this event-level test is legitimate.
        - `cu.covariate_enrichment(in_group_mask, covariate)` → `{chi2, p, ...}`. An **event-level**
          chi-square. `in_group_mask` is a boolean array (`labels == c`), **not** a labels array.
        - `cu.permutation_test(in_group_mask, covariate, unit, n=5000)` → `{stat, p_emp}`. Shuffles
          the covariate **at the `unit` level** (pass `cage`). This is the correction for
          pseudoreplication.

        **Hypothesis to grade:** *Food deprivation changes cluster composition and the change survives
        when the cage — not the event — is the unit; the sex difference does not survive.*

        **Your task.** Edit the **one flagged line** below so that `sex_test_p` holds the
        **cage-level** permutation p-value instead of the event-level p-value, then read the verdict.
        You are changing exactly one line; the two lines above it (condition and event-level sex) are
        already correct and give you the values to compare against. When you fix it, the self-check
        below should turn green: condition survives Bonferroni, and sex becomes "cannot conclude."
        """
    )
    return


@app.cell
def _(cu, labels, sex, cage, cond):
    def grade_claims(labels, sex, cage, cond):
        C = 4                                  # the aggression cluster
        in_group = labels == C

        # (1) CONDITION - within-cage, so the event-level Bonferroni test is honest:
        cond_res = cu.condition_enrichment(labels, cond)
        condition_min_bonf = min(r["p_bonf"] for r in cond_res)

        # (2) SEX - the naive event-level p (looks significant, but pseudoreplicated):
        sex_event_p = cu.covariate_enrichment(in_group, sex)["p"]

        # (3) TODO: replace the right-hand side below so sex_test_p is the CAGE-LEVEL permutation p.
        #     The line currently just copies the event-level p (the wrong answer) so the notebook
        #     still runs. Swap it for:
        #         cu.permutation_test(in_group, sex, unit=cage, n=5000)["p_emp"]
        #     After the fix, sex_test_p should be about 0.15 instead of about 3e-7.
        sex_test_p = sex_event_p               # <-- edit THIS line only

        return dict(condition_min_bonf=float(condition_min_bonf),
                    sex_event_p=float(sex_event_p),
                    sex_test_p=float(sex_test_p))

    student_verdict = grade_claims(labels, sex, cage, cond)
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

                Then `sex_test_p ≈ 0.15` (it was `3e-7` at the event level). The correct verdict for
                sex is "cannot conclude — pseudoreplicated and underpowered at n = 4 vs 3 cages."
                Condition, by contrast, is within-cage and survives Bonferroni (`min p_bonf ≈ 0.003`).
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
        _body = (f"Condition survives Bonferroni (min p_bonf = {_cmb:.4f} < 0.05), so food "
                 f"deprivation genuinely reshapes the ethogram. Sex does not survive cage-level "
                 f"shuffling (p_emp = {_sxp:.3f}); the correct verdict is *cannot conclude — "
                 f"pseudoreplicated and underpowered at n = 4 vs 3 cages.*")
    elif _naive:
        _kind, _head = "danger", "Not yet - you are still using the event-level p for sex"
        _body = (f"`sex_test_p = {_sxp:.2g}` is the naive event-level number. It looks significant, "
                 f"but sex is completely confounded with cage, so that p-value is pseudoreplication. "
                 f"Swap in `cu.permutation_test(in_group, sex, unit=cage, n=5000)['p_emp']` and "
                 f"re-read the verdict.")
    else:
        _kind, _head = "warn", "Close - check the band"
        _body = (f"condition_min_bonf = {_cmb:.4f} (want < 0.05); sex_test_p = {_sxp:.3f} "
                 f"(want the honest cage-level value, roughly 0.05-0.6). The graded-correct sex "
                 f"verdict is *cannot conclude — pseudoreplicated.*")
    mo.callout(mo.md(f"**{_head}**\n\n{_body}"), kind=_kind)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Conceptual questions": mo.md(
                r"""
                1. **Why does testing many clusters inflate false positives?** With 5 tests at
                   α = 0.05, the chance of *at least one* false hit is about 1 − 0.95⁵ ≈ 0.23.
                   Bonferroni multiplies each p by the number of tests to hold the family-wise error
                   at 0.05.
                2. **Why do we trust condition (within-cage) over sex (between-cage)?** Every cage
                   supplies pre, dep, and post events, so condition is not confounded with cage. Sex is
                   a fixed property of the cage, so a "sex" test is really a comparison of 7 cages in
                   disguise.
                3. **Which way does 16% rank mislabeling bias a dyad chi-square?** *Random* error
                   pushes toward the null (a real effect is harder to detect); *systematic* error (tail
                   dropout during contact) can create a spurious one. You cannot sign it without
                   characterizing the error, so rank stays a flag.
                4. **Where does chi-square break down?** When expected cell counts are small (a common
                   rule of thumb is fewer than 5), the chi-square approximation becomes unreliable; use
                   an exact or permutation test instead.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we set aside, and how these tests fail

        - **We gave up the event as the unit of inference for between-cage variables.** Collapsing
          1500 events to 7 cages is a large loss of apparent sample size, but 7 is the *honest*
          number. Anything that looked significant at the event level and vanished at the cage level
          was never a real effect.
        - **Failure mode 1 — pseudoreplication.** Any between-cage variable (sex, genotype, a drug
          given per cage) will look strongly significant at the event level and is usually an artifact.
          Always ask: *what is the exchangeable unit?*
        - **Failure mode 2 — multiple comparisons.** C4's condition effect (raw p = 0.012) does not
          survive Bonferroni. Scan enough clusters × variables and something will always reach
          p < 0.05 by chance.
        - **Failure mode 3 — small-cell chi-square.** The rarest syllables (for example C3, n = 32)
          have thin dyad cells, so the chi-square p-value there is only approximate.
        - **To think about:** with only 7 cages, what design or analysis would give real power to test
          a sex effect? (More cages? A within-animal manipulation? A mixed-effects model with cage as
          a random intercept?) Sketch the one you would run and the sample size it would need.
        """
    )
    return


@app.cell(hide_code=True)
def _(make_board, mo):
    mo.vstack([
        mo.md(r"""
        **Readout Board — bottom of NB06.** Gauge B is unchanged in value, but its meaning is now
        better supported: we checked **what the map encodes** and found that condition is real
        (within-cage, Bonferroni-clean) while sex cannot be concluded from this design (7 cages, no
        power). An enrichment test you can trust is more useful than a decoder you cannot interpret.
        This is also how a later manipulation would be read out: *did it move the state distribution,
        at the correct unit?*
        """),
        make_board("validated: condition YES - sex not from this design"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we do next

        We now know how to read the map without fooling ourselves: enrichment tests with the right
        multiple-comparison correction and the right exchangeable unit. Condition genuinely reshapes
        the ethogram; the apparent sex effect was pseudoreplication.

        So far, though, every event has been a frozen snapshot. Behavior unfolds over time — a sniff
        becomes a chase becomes a fight. **NB07** models the **grammar** of how syllables follow one
        another (an observed Markov chain: the probability that one syllable is followed by another)
        and lays it over the activity timeline of a single continuously tracked cage.
        """
    )
    return


if __name__ == "__main__":
    app.run()

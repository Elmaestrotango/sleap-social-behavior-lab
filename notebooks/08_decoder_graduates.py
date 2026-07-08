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


# ============================================================================ briefing
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 08 · The Decoder Graduates

        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > This is the one. For eight notebooks you collapsed 11,700 raw pose numbers per event down to
        > 19 features, ~6 components, a 2-D map, a syllable — and you learned, the hard way in NB06, which
        > results survive honest statistics. Today you **ship the readout**: hand-score ground truth, train
        > the decoder, and then meet **Cage 16 — the animal on the rig — for the first time.** If a decoder
        > trained on seven cages reads a cage it has *never seen*, the laser turns on.
        >
        > **Deliverable:** a validated aggression decoder + its honest held-out score on Cage 16.
        > **Circuit question it unblocks:** can we time-align an objective behavioral readout to a VMHvl
        > stim / mPFC recording without hand-scoring every frame?
        > **Today's lab-meeting question:** *Does the decoder generalize to a brand-new cage — and is it
        > trustworthy enough to gate a causal experiment?*

        The mission resolves in **three beats**: **(1) ground truth & the label-noise ceiling** (the final
        threat), **(2) train**, **(3) unlock Cage 16, validate honestly, and cash the neural check.**

        *Neuroscience connection (open):* leave-one-cage-out generalization is the behavioral face of a
        cross-session brain–machine-interface decoder that must survive a new recording — the same demand
        Padilla-Coreano and colleagues met when they decoded competitive rank from mPFC ensembles. You are
        building the behavioral half of a published neural-decoding pipeline, and by the end you will run the
        *identical code* on neurons.
        """
    )
    return


# ============================================================================ data
@app.cell
def _(ROOT, cu):
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))     # kp, ranks, agg_label, category...
    der = cu.load_derived("train", ROOT)                                 # X (1500,19), pca_scores, cage, sex
    ho = cu.load_events(cu.data_path("data/heldout_events.npz", ROOT))   # Cage 16 — the sealed rig (470)
    hod = cu.load_derived("heldout", ROOT)                               # X (470,19), sex all 'F'
    sweep = cu.load_umap_sweep(ROOT)                                     # default_labels = canonical clusters
    neu = cu.load_neural_demo(ROOT)                                      # X_neural (800,60), y, emb2d
    return der, ev, ho, hod, neu, sweep


@app.cell
def _(der, ev, ho, hod):
    X = der["X"]                          # (1500,19) train features
    y = ev["agg_label"].astype(int)       # (1500,) ground-truth aggression
    cage = der["cage"]                    # (1500,) cages 9..15
    Xh = hod["X"]                          # (470,19) Cage 16 features
    yh = ho["agg_label"].astype(int)      # (470,) Cage 16 ground truth
    return X, Xh, cage, y, yh


@app.cell
def _(ROOT, cu):
    # Committed benchmark values for the Readout Board (degrade gracefully if a row is missing).
    import csv as _csv
    _board = {}
    try:
        with open(cu.data_path("data/readout_board.csv", ROOT)) as _f:
            for _r in _csv.DictReader(_f):
                _board[(_r["gauge"], _r["notebook"], _r["stage"])] = _r
    except Exception:
        _board = {}
    def bench(gauge, notebook, stage, default=None):
        _row = _board.get((gauge, notebook, stage))
        try:
            return float(_row["value"])
        except Exception:
            return default
    return (bench,)


# --- CORE on-load decoder: train on cages 9–15, score Cage 16. ~1s fit, renders on load. ---
@app.cell
def _(X, Xh, cu, y):
    model = cu.make_mlp()          # impute → scale → MLP (sklearn only)
    model.fit(X, y)                # trained on the seven cages 9–15
    s_ho = model.predict_proba(Xh)[:, 1]   # P(aggression) on never-seen Cage 16
    s_tr = model.predict_proba(X)[:, 1]    # in-sample scores (for the FP-criterion demo)
    return model, s_ho, s_tr


@app.cell
def _(cu, s_ho, yh):
    res_ho = cu.eval_binary(yh, s_ho)      # {roc_auc, avg_precision, confusion, ...} on Cage 16
    return (res_ho,)


# ============================================================================ readout board (top)
@app.cell(hide_code=True)
def _(bench, go, mo, res_ho):
    _b = bench("B", "NB08", "held-out cage decode", 0.86)
    _auc = res_ho.get("roc_auc", float("nan"))
    _fig = go.Figure()
    _fig.add_trace(go.Indicator(
        mode="number", value=1,
        title={"text": "Gauge A · size of representation<br><sub>11,700 → 19 → 6 → 2-D → 1 decision</sub>"},
        number={"suffix": " decision"}, domain={"row": 0, "column": 0}))
    _fig.add_trace(go.Indicator(
        mode="gauge+number+delta", value=_auc,
        delta={"reference": _b, "valueformat": ".3f"},
        title={"text": f"Gauge B · held-out readiness<br><sub>Cage 16 AUC vs benchmark {_b:.2f}</sub>"},
        gauge={"axis": {"range": [0.5, 1.0]}, "bar": {"color": "#2ca02c"},
               "threshold": {"line": {"color": "#e45756", "width": 3}, "value": _b}},
        domain={"row": 0, "column": 1}))
    _fig.update_layout(grid={"rows": 1, "columns": 2}, height=260,
                       margin=dict(l=30, r=30, t=70, b=10), template="plotly_white")
    mo.vstack([mo.md("### Readout Board — the mission ledger"), _fig])
    return


# ============================================================================ sealed cage 16 → UNLOCKED
@app.cell(hide_code=True)
def _(Xh, mo, yh):
    _n, _npos = len(yh), int(yh.sum())
    mo.md(
        f"""
        <div style="border:2px solid #2ca02c;border-radius:10px;padding:14px 18px;
        background:linear-gradient(90deg,#f0fff4,#ffffff)">
        <b>🔓 CAGE 16 — UNLOCKED</b> &nbsp;·&nbsp; <b>notebooks until unlock: 0</b><br>
        The forbidden fruit is finally on the table. Camera 16 holds <b>{_n} events</b>
        ({_npos} aggression, base rate {_npos/_n:.3f}) with <b>{Xh.shape[1]} features</b> each — a
        <b>female</b> cage (the training set mixes M/F, so this doubles as a <b>cross-sex</b> test).
        It was greyed-out and blacked-out for seven notebooks precisely so today's number would be
        <i>honest</i>: nothing about Cage 16 ever touched the feature design, the PCA, the map, or the
        cluster labels. This is the only test a circuit experiment would believe.
        </div>
        """
    )
    return


# ============================================================================ ACT 1 — ground truth & ceiling
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Act 1 · Ground truth & the label-noise ceiling *(the final threat)*

        A decoder is only as honest as its labels. Before we train anything, **you** hand-score a handful of
        exemplar clips, then we measure your agreement with a reference key. A hard truth up front: comparing
        your labels to a single reference key measures **accuracy against that reference**, *not* inter-rater
        reliability — real inter-rater reliability needs a genuinely independent second labeler (offered as a
        stretch below). Cohen's κ corrects for chance agreement either way, but the *interpretation* changes.

        Whatever noise your labels carry does not average out — it **caps the ceiling** every downstream
        decoder can reach.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📋 Labeling guidelines (open me before you score)": mo.md(
            r"""
            Call an event **aggression** when you see a committed, high-velocity offensive act directed at the
            other mouse: a lunge, chase-to-contact, pin, or bite attempt. The white arrow points
            approacher → approachee; the red dot marks contact onset.

            **Do NOT** call it aggression for: passive co-resting, sniff/investigation, anogenital contact,
            grooming, or two mice simply on the ledge. The tricky ones are the **`mlp_fp`** clips — events an
            earlier model flagged as aggression that a human rejects. They sit right on the criterion, and how
            you score them is exactly where label noise enters.

            **Stretch — make "inter-rater" honest:** have a labmate score the same eight clips independently
            and compute κ between the *two of you*. That is inter-rater reliability; comparing to the key
            below is only accuracy-vs-reference.
            """)
    })
    return


@app.cell
def _(cu, ev, mo, np):
    # 8 exemplars: 3 clear aggression, 2 clear non-social, 3 ambiguous mlp_fp. GIFs build on load;
    # the *grading* is gated behind the form's submit (nothing expensive renders until you click).
    _cats = ev["category"]
    _agg = np.where(_cats == "aggression")[0][:3]
    _neg = np.where(_cats == "")[0][:2]
    _amb = np.where(_cats == "mlp_fp")[0][:3]
    label_idx = np.concatenate([_agg, _neg, _amb]).astype(int)

    _cells, _drops = [], {}
    for _j, _i in enumerate(label_idx):
        _gif = cu.event_gif_bytes(ev["kp"][_i], ev["ranks"][_i], int(ev["contact_rel"][_i]), cell=140)
        _img = cu.gif_img_html(_gif, width=140)
        _drops[f"d{_j}"] = mo.ui.dropdown(["aggression", "not aggression"],
                                          value="not aggression", label=f"clip #{int(_i)}")
        _cells.append(f'<div style="display:inline-block;text-align:center;margin:4px">'
                      f'{_img}<br>{{d{_j}}}</div>')
    _tpl = "<div>" + "".join(_cells) + "</div>"
    label_form = mo.md(_tpl).batch(**_drops).form(submit_button_label="Grade my labels")
    label_form
    return label_form, label_idx


@app.cell(hide_code=True)
def _(cu, ev, label_form, label_idx, mo, np):
    if label_form.value is None:
        _out = mo.md("*Score all eight clips above and click **Grade my labels** to measure your agreement "
                     "with the reference key.*")
    else:
        _v = label_form.value
        _student = np.array([1 if _v[f"d{_j}"] == "aggression" else 0 for _j in range(len(label_idx))])
        _ref = ev["agg_label"][label_idx].astype(int)     # reference key (answer_key-aligned)
        _acc = float((_student == _ref).mean())
        _kappa = cu.cohens_kappa(_student, _ref)
        _color = "#2ca02c" if _kappa >= 0.6 else ("#e6a100" if _kappa >= 0.3 else "#e45756")
        _out = mo.md(
            f"""
            <div style="border-left:5px solid {_color};padding:8px 14px;background:#fafafa">
            <b>Agreement with the reference key</b> (this is <i>accuracy vs a reference</i>, not inter-rater
            reliability): raw accuracy <b>{_acc:.2f}</b> over {len(label_idx)} clips · Cohen's κ
            <b>{_kappa:.2f}</b> (chance-corrected). Most disagreements land on the <code>mlp_fp</code> clips —
            that residual is the label noise that caps every decoder below.
            </div>
            """)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How the criterion moves the false-positive rate

        The `mlp_fp` events are the deliberately-ambiguous near-misses. Slide the decision criterion and watch
        the decoder's **false-positive rate on exactly these clips**: a loose criterion (low threshold) calls
        many of them aggression; tightening it trades those false alarms for missed real attacks. There is no
        free lunch — the label noise you just felt is the same ambiguity the decoder must price.
        """
    )
    return


@app.cell
def _(mo):
    fp_thr = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="decision criterion (threshold)",
                          debounce=True, full_width=True)
    return (fp_thr,)


@app.cell
def _(ev, fp_thr, go, mo, np, s_tr):
    _fp_mask = ev["category"] == "mlp_fp"
    _neg_mask = ev["agg_label"].astype(int) == 0
    _thr = np.linspace(0.05, 0.95, 19)
    _fp_amb = [float((s_tr[_fp_mask] >= t).mean()) for t in _thr]        # FP rate on ambiguous near-misses
    _fp_all = [float((s_tr[_neg_mask] >= t).mean()) for t in _thr]       # FP rate on all true negatives
    _here = float((s_tr[_fp_mask] >= fp_thr.value).mean())
    _fig = go.Figure()
    _fig.add_scatter(x=_thr, y=_fp_amb, mode="lines+markers", name="ambiguous mlp_fp clips",
                     line=dict(color="#e45756", width=3))
    _fig.add_scatter(x=_thr, y=_fp_all, mode="lines", name="all true-negative events",
                     line=dict(color="#9aa0a6", dash="dot"))
    _fig.add_vline(x=fp_thr.value, line=dict(color="#333", dash="dash"))
    _fig.update_layout(template="plotly_white", height=340,
                       title=f"FP rate at threshold {fp_thr.value:.2f}: {_here:.0%} of mlp_fp clips flagged",
                       xaxis_title="decision threshold", yaxis_title="false-positive rate",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([fp_thr, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Predict the ceiling.** If ~1 in 8 hand labels is wrong at the boundary (and NB01 already warned that
        tail-mark identity carries ~16% error), a *perfect* model still inherits that noise as an accuracy
        ceiling. Below we can simulate it directly: corrupt a fraction of the training labels and watch the
        held-out AUC fall. This refits several models, so it is gated.
        """
    )
    return


@app.cell
def _(mo):
    noise_btn = mo.ui.run_button(label="▶ Simulate the label-noise ceiling (refits ~5 models)")
    return (noise_btn,)


@app.cell
def _(X, Xh, cu, go, mo, noise_btn, np, y, yh):
    if not noise_btn.value:
        _out = mo.md("*Click to sweep training-label corruption and see the held-out ceiling fall.*")
    else:
        _levels = [0.0, 0.05, 0.10, 0.20, 0.30]
        _aucs = []
        _rng = np.random.RandomState(0)
        for _p in _levels:
            _yn = y.copy()
            _flip = _rng.rand(len(_yn)) < _p
            _yn[_flip] = 1 - _yn[_flip]
            _m = cu.make_mlp(); _m.fit(X, _yn)
            _aucs.append(cu.eval_binary(yh, _m.predict_proba(Xh)[:, 1])["roc_auc"])
        _fig = go.Figure(go.Scatter(x=_levels, y=_aucs, mode="lines+markers",
                                    line=dict(color="#e45756", width=3)))
        _fig.update_layout(template="plotly_white", height=320,
                           title="Held-out Cage-16 AUC vs fraction of corrupted training labels",
                           xaxis_title="fraction of labels flipped", yaxis_title="Cage-16 ROC-AUC",
                           margin=dict(l=10, r=10, t=50, b=10))
        _out = _fig
    _out
    return


# ============================================================================ ACT 2 — train
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Act 2 · Train the readout

        Now the decoder. `make_mlp` is a small sklearn pipeline (median-impute → standardize → MLP); a
        **logistic-regression** linear baseline keeps us honest about whether nonlinearity buys anything.

        **Every earlier stage should demonstrably feed this readout.** The feature-set comparison below pits
        the raw **19 features** against the **PCA scores** you built in NB04 and against **19 features +
        cluster-membership one-hot** from the canonical NB05 map — a *within-cage* 5-fold comparison. (The 19
        features already contain the NB03 coordination signals — `closing_speed`, the facing cosines,
        `heading_alignment` — so coordination is represented rather than bolted on.)
        """
    )
    return


@app.cell
def _(mo):
    featureset_btn = mo.ui.run_button(label="▶ Run the feature-set comparison (5-fold CV, several fits)")
    return (featureset_btn,)


@app.cell
def _(X, bench, cu, der, featureset_btn, go, mo, np, sweep, y):
    if not featureset_btn.value:
        _out = mo.md("*Click to compare feature sets by within-cage 5-fold cross-validated AUROC.*")
    else:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        _lab = sweep["default_labels"].astype(int)
        _oh = np.zeros((len(_lab), int(_lab.max()) + 2))
        _oh[np.arange(len(_lab)), _lab + 1] = 1.0                      # one-hot incl. noise (-1)
        _sets = {
            "PCA scores (NB04)": der["pca_scores"],
            "19 features": X,
            "19 feats + clusters (NB05)": np.hstack([X, _oh]),
        }
        _skf = StratifiedKFold(5, shuffle=True, random_state=0)
        _names, _mlp_auc, _lr_auc = [], [], []
        _lr = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=1000))])
        for _nm, _feat in _sets.items():
            _names.append(_nm)
            _mlp_auc.append(cross_val_score(cu.make_mlp(), _feat, y, cv=_skf, scoring="roc_auc").mean())
            _lr_auc.append(cross_val_score(_lr, _feat, y, cv=_skf, scoring="roc_auc").mean())
        _fig = go.Figure()
        _fig.add_bar(x=_names, y=_mlp_auc, name="MLP", marker_color="#4c78a8")
        _fig.add_bar(x=_names, y=_lr_auc, name="logistic (linear)", marker_color="#e6a100")
        _fig.add_hline(y=bench("B", "NB06", "features -> aggression (train CV)", 0.837),
                       line=dict(color="#888", dash="dot"),
                       annotation_text="NB06 benchmark 0.837")
        _fig.update_yaxes(range=[0.5, 0.9], title="5-fold CV AUROC")
        _fig.update_layout(template="plotly_white", height=380, barmode="group",
                           title="Feature-set comparison (within-cage CV) — earlier stages feed the readout",
                           margin=dict(l=10, r=10, t=50, b=10))
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two honest takeaways you will see: the **linear** baseline is nearly as good as the MLP (this problem
        is close to linearly separable in feature space), and **adding cluster one-hots barely helps** — the
        clusters are a *coarsening* of the same 19 features, not new information. That is the point of the
        whole Phase-1 collapse: the representation was already in the features.
        """
    )
    return


# ============================================================================ ACT 3 — Cage 16 unlocks
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Act 3 · Cage 16 — the honest held-out test

        The core test runs **on load, no button**: train on cages 9–15, evaluate on Cage 16. The ROC/PR
        curves, confusion matrix, and calibration below are the real thing.
        """
    )
    return


@app.cell
def _(cu, res_ho, s_ho, yh):
    roc_fig = cu.roc_pr_fig(yh, s_ho)
    roc_fig.update_layout(title=f"Cage 16 held-out · ROC-AUC = {res_ho.get('roc_auc', float('nan')):.3f}")
    roc_fig
    return


@app.cell
def _(cu, go, mo, np, res_ho, s_ho, yh):
    _cm = np.array(res_ho["confusion"])
    _fig1 = go.Figure(go.Heatmap(z=_cm, x=["pred: not", "pred: agg"], y=["true: not", "true: agg"],
                                 colorscale="Blues", showscale=False,
                                 text=_cm, texttemplate="%{text}", textfont={"size": 18}))
    _fig1.update_yaxes(autorange="reversed")
    _fig1.update_layout(template="plotly_white", height=320, title="Confusion @ 0.5",
                        margin=dict(l=10, r=10, t=40, b=10))
    _frac, _mean = cu.calibration_curve(yh, s_ho, n_bins=8)
    _fig2 = go.Figure()
    _fig2.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#bbb"),
                      showlegend=False)
    _fig2.add_scatter(x=_mean, y=_frac, mode="lines+markers", line=dict(color="#4c78a8"),
                      showlegend=False)
    _fig2.update_layout(template="plotly_white", height=320, title="Calibration (reliability curve)",
                        xaxis_title="mean predicted P", yaxis_title="observed fraction",
                        margin=dict(l=10, r=10, t=40, b=10))
    mo.hstack([_fig1, _fig2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The opto readout: pick a decision threshold

        For a causal experiment the threshold is a policy, not a default. A false "attack" fakes an effect; a
        missed one hides it. Slide the threshold and read **live precision/recall on Cage 16**.
        """
    )
    return


@app.cell
def _(mo):
    thr_slider = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="decision threshold (Cage 16)",
                              debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(go, mo, np, s_ho, thr_slider, yh):
    _grid = np.linspace(0.05, 0.95, 19)
    def _pr(t):
        _pred = (s_ho >= t).astype(int)
        _tp = int(((_pred == 1) & (yh == 1)).sum()); _fp = int(((_pred == 1) & (yh == 0)).sum())
        _fn = int(((_pred == 0) & (yh == 1)).sum())
        _p = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        _r = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        return _p, _r
    _P = [_pr(t)[0] for t in _grid]; _R = [_pr(t)[1] for t in _grid]
    _p_here, _r_here = _pr(thr_slider.value)
    _fig = go.Figure()
    _fig.add_scatter(x=_grid, y=_P, mode="lines", name="precision", line=dict(color="#4c78a8", width=3))
    _fig.add_scatter(x=_grid, y=_R, mode="lines", name="recall", line=dict(color="#e45756", width=3))
    _fig.add_vline(x=thr_slider.value, line=dict(color="#333", dash="dash"))
    _fig.update_layout(template="plotly_white", height=340,
                       title=f"@ threshold {thr_slider.value:.2f}:  precision {_p_here:.2f} · recall {_r_here:.2f}",
                       xaxis_title="threshold", yaxis_title="value", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([thr_slider, _fig])
    return


@app.cell(hide_code=True)
def _(X, cu, mo, np, res_ho, y):
    # Within-cage vs LOCO gap — computed live, reported honestly (the gap is INVERTED on this bundle).
    _rng = np.random.RandomState(0)
    _perm = _rng.permutation(len(y))
    _cut = int(0.7 * len(y))
    _mw = cu.make_mlp(); _mw.fit(X[_perm[:_cut]], y[_perm[:_cut]])
    _within = cu.eval_binary(y[_perm[_cut:]], _mw.predict_proba(X[_perm[_cut:]])[:, 1])["roc_auc"]
    _loco = res_ho.get("roc_auc", float("nan"))
    mo.md(
        f"""
        **Within-cage vs leave-one-cage-out (LOCO).** Within-cage 70/30 split AUC ≈ **{_within:.3f}**;
        held-out Cage-16 (LOCO) AUC ≈ **{_loco:.3f}**. On *this* bundle LOCO is actually the *easier* test —
        Cage 16 is a female control cage with a higher, cleaner aggression base rate (0.383), so held-out
        beats within-cage. That inverts the usual "generalization tax" and is worth stating out loud rather
        than assuming: the honest gap is whatever the data shows, not what you hoped for.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The decoder's receptive field

        `permutation_importance` shuffles one feature at a time and measures how much Cage-16 AUC drops — the
        features whose destruction hurts most are what the decoder actually reads. This re-scores many times,
        so it is gated.
        """
    )
    return


@app.cell
def _(mo):
    permimp_btn = mo.ui.run_button(label="▶ Compute permutation importance on Cage 16")
    return (permimp_btn,)


@app.cell
def _(Xh, cu, go, model, permimp_btn, np, yh):
    if not permimp_btn.value:
        _out = go.Figure().update_layout(
            template="plotly_white", height=120,
            title="Click the button above to reveal the decoder's receptive field",
            margin=dict(l=10, r=10, t=40, b=10))
    else:
        from sklearn.inspection import permutation_importance
        _r = permutation_importance(model, Xh, yh, n_repeats=10, random_state=0, scoring="roc_auc")
        _order = np.argsort(_r.importances_mean)
        _fig = go.Figure(go.Bar(x=_r.importances_mean[_order], y=[cu.FEATURE_NAMES[i] for i in _order],
                                orientation="h", marker_color="#4c78a8",
                                error_x=dict(type="data", array=_r.importances_std[_order])))
        _fig.update_layout(template="plotly_white", height=520,
                           title="Permutation importance on Cage 16 — the decoder's receptive field",
                           xaxis_title="drop in ROC-AUC when shuffled", margin=dict(l=10, r=10, t=50, b=10))
        _out = _fig
    _out
    return


# ---- Hero Event (this notebook's method = the decoder reads it) ----
@app.cell(hide_code=True)
def _(cu, ev, mo, s_tr):
    _HERO = 909      # cage 15, male, aggression, contact_rel=40, node reliability 0.998 (see engine note)
    _gif = cu.event_gif_bytes(ev["kp"][_HERO], ev["ranks"][_HERO], int(ev["contact_rel"][_HERO]), cell=200)
    mo.md(
        f"""
        **Hero Event #{_HERO}** (Cage 15, male — our canonical aggression event; index 742 in the design doc
        is a non-aggression Cage-12 clip in the shipped data, so we follow #909 throughout). We tracked it
        from raw skeleton → body-frame geometry → a point on the map → a syllable. The decoder now reads it:
        **P(aggression) = {s_tr[_HERO]:.2f}** — a confident hit. *(This is an in-sample sanity check — #909
        is a training event — so treat the number as a demonstration, not evidence; the honest evidence is
        the Cage-16 curve above.)*

        {cu.gif_img_html(_gif, width=200)}
        """
    )
    return


# ---- predicted-aggression Cage 16 clips (render on load) ----
@app.cell(hide_code=True)
def _(cu, ho, mo, np, s_ho):
    _top = np.argsort(s_ho)[::-1][:4]     # 4 highest-scored Cage-16 events
    _events = [(ho["kp"][i], ho["ranks"][i], int(ho["contact_rel"][i])) for i in _top]
    _grid = cu.grid_gif_bytes(_events, ncols=4, cell=140)
    _scores = ", ".join(f"{s_ho[i]:.2f}" for i in _top)
    mo.md(
        f"""
        **The decoder's confident calls on Cage 16** (top-4 predicted aggression, scores {_scores}). These
        are events it never trained on:

        {cu.gif_img_html(_grid, width=600)}
        """
    )
    return


# ============================================================================ opto simulation
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Simulate the opto readout

        The mission's whole point: gate a causal experiment. Imagine VMHvl stimulation **multiplies attack
        frequency**. We resample Cage 16 to inject that shift, then ask whether *this* decoder, at the current
        detection threshold, catches the change. Turn the knob and watch detected-aggression rate move.
        """
    )
    return


@app.cell
def _(mo):
    opto_mult = mo.ui.slider(1.0, 3.0, value=2.0, step=0.25, label="VMHvl-stim attack multiplier",
                             debounce=True, full_width=True)
    return (opto_mult,)


@app.cell
def _(go, mo, np, opto_mult, s_ho, yh):
    _rng = np.random.RandomState(0)
    _pos = np.where(yh == 1)[0]
    _n_extra = int(round((opto_mult.value - 1.0) * len(_pos)))
    _extra = _rng.choice(_pos, size=_n_extra, replace=True) if _n_extra > 0 else np.array([], int)
    _scores_stim = np.concatenate([s_ho, s_ho[_extra]])
    _thr = 0.5
    _base_rate = float((s_ho >= _thr).mean())
    _stim_rate = float((_scores_stim >= _thr).mean())
    _catches = "✅ the decoder catches it" if (_stim_rate - _base_rate) > 0.03 else "⚠️ shift too small to see"
    _fig = go.Figure(go.Bar(x=["baseline", f"VMHvl stim ×{opto_mult.value:g}"],
                            y=[_base_rate, _stim_rate], marker_color=["#9aa0a6", "#e45756"],
                            text=[f"{_base_rate:.2f}", f"{_stim_rate:.2f}"], textposition="outside"))
    _fig.update_layout(template="plotly_white", height=330,
                       title=f"Detected-aggression rate @ thr 0.5 — {_catches}",
                       yaxis_title="fraction flagged aggression", yaxis_range=[0, 1],
                       margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([opto_mult, _fig])
    return


# ============================================================================ cash the neural check
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Cash the neural check

        Here is the twist the whole week was built on. A synthetic **population raster** — 800 trials × 60
        neurons, a hidden binary brain state modulating firing — is decoded with the **identical**
        `make_mlp` / `eval_binary` you just used on behavior. Same code, different signal. You decode a brain
        with the exact pipeline you built to decode behavior.
        """
    )
    return


@app.cell
def _(cu, neu, np):
    _Xn = neu["X_neural"].astype(float); _yn = neu["y"].astype(int)
    _perm = np.random.RandomState(0).permutation(len(_yn)); _cut = 560
    _mn = cu.make_mlp(); _mn.fit(_Xn[_perm[:_cut]], _yn[_perm[:_cut]])
    _sn = _mn.predict_proba(_Xn[_perm[_cut:]])[:, 1]
    _res_neu = cu.eval_binary(_yn[_perm[_cut:]], _sn)
    neu_auc = _res_neu.get("roc_auc", float("nan"))
    return (neu_auc,)


@app.cell(hide_code=True)
def _(mo, neu_auc):
    mo.md(
        f"""
        **Held-out neural decode: ROC-AUC = {neu_auc:.3f}.** The identical pipeline that read aggression from
        pose reads the hidden state from spikes. The synthetic raster is cleanly separable (near-perfect
        here), which is the *point* of the demonstration, not a claim about real data — real ensembles are
        messier, and that mess is exactly why the honest held-out discipline you practiced on Cage 16 matters
        even more for neurons.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### CEBRA epilogue — a joint neural–behavior embedding

        The precomputed 2-D embedding below places every trial by its neural population activity, colored by
        the hidden state. This is the *aspiration* of methods like CEBRA: behavior and brain in one space.
        """
    )
    return


@app.cell
def _(go, mo, neu):
    _emb = neu["emb2d"]; _yn = neu["y"].astype(int)
    _fig = go.Figure()
    for _g, _c, _nm in [(0, "#4c78a8", "state 0"), (1, "#e45756", "state 1")]:
        _m = _yn == _g
        _fig.add_scattergl(x=_emb[_m, 0], y=_emb[_m, 1], mode="markers", name=_nm,
                           marker=dict(size=6, opacity=0.7, color=_c))
    _fig.update_layout(template="plotly_white", height=430,
                       title="Precomputed joint embedding — trials colored by hidden state",
                       xaxis_title="dim 1", yaxis_title="dim 2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([
        _fig,
        mo.accordion({"Where the analogy stops": mo.md(
            r"""
            CEBRA does **not** symmetrically merge brain and behavior. It uses behavior (or time, or a label)
            as a **contrastive** signal to *shape* a neural embedding — pulling together trials that share a
            behavioral context and pushing apart those that don't. The poetic "one space" is the goal;
            contrastive learning is the mechanism, and it is directional. *Schneider, Lee & Mathis 2023,
            Nature — synthetic demonstration here, not a fitted CEBRA model.*
            """)})
    ])
    return


# ============================================================================ neuroscience connection
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "🧠 Deeper: the paper & where the analogy stops": mo.md(
            r"""
            **Shared mathematics.** A leave-one-cage-out decoder is a population decoder that must survive a
            *held-out session* — the same demand a cross-session brain–machine interface faces. The math of
            fitting a classifier on one set of recordings and asking it to read a new one is identical whether
            the rows are events or trials and the columns are pose features or neurons.

            **Real references.** Georgopoulos et al. 1986 *Science* (population vector decoding); Glaser et
            al. 2020 *eNeuro* (machine-learning neural decoders); Gilja et al. 2012 *Nat. Neurosci.* (a BMI
            that stays stable across sessions); **Padilla-Coreano et al. 2022 *Nature*** — decoded competitive
            rank from mPFC ensembles using tracking + an HMM/GLM, i.e. the *mirror image* of what you built;
            Schneider, Lee & Mathis 2023 *Nature* (CEBRA).

            **Species / preparation tag.** Mouse homecage pose (behavior) ↔ rodent/primate electrophysiology
            (neural); the neural-demo raster is *synthetic*.

            **Where the analogy stops.** Held-out **cage** is the honest unit here because cage is the true
            source of non-independence — the neural analog is a held-out **session/subject**, not a held-out
            trial. And CEBRA shapes a neural embedding *using* behavior as a contrastive label; it is not a
            symmetric fusion. Tube-test rank ≠ homecage aggression either — correlated but dissociable axes
            (the standing rank caveat from NB06).
            """)
    })
    return


# ============================================================================ exercise scaffold
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 🧪 Exercise — graduate the decoder

        **Toolbox**
        - `cu.make_mlp()` → unfitted sklearn Pipeline; call `.fit(X, y)` then `.predict_proba(X)[:, 1]`.
        - `cu.eval_binary(y_true, y_score, thr=0.5)` → `{roc_auc, avg_precision, confusion, ...}` (takes the
          **score vector**, not the model).
        - `X, y` = train features/labels (cages 9–15); `Xh, yh` = Cage-16 features/labels.
        - `neu` → `X_neural (800,60)`, `y (800,)` for the neural check.

        **Hypothesis banner.** *A decoder trained on cages 9–15 detects aggression in never-seen Cage 16 at
        held-out ROC-AUC ≈ 0.86, and the same pipeline decodes the neural demo's hidden state well above
        chance.*

        **Your TODO** (write it in the next cell):
        1. Fit `cu.make_mlp()` on `(X, y)`; score `Xh`; compute `heldout_auc` with `cu.eval_binary`.
        2. Fit a fresh `cu.make_mlp()` on a split of `neu["X_neural"]`/`neu["y"]`; compute `neural_auc`.
        """
    )
    return


@app.cell
def _(X, Xh, cu, neu, np, y, yh):
    # ------------------------------------------------------------------ TODO: student writes this
    # heldout_auc = ...     # train on (X, y), score Xh, eval vs yh
    # neural_auc  = ...     # held-out split of neu["X_neural"] / neu["y"]
    #
    # Reference implementation (also revealed in the solution accordion):
    _m = cu.make_mlp(); _m.fit(X, y)
    heldout_auc = cu.eval_binary(yh, _m.predict_proba(Xh)[:, 1])["roc_auc"]

    _Xn = neu["X_neural"].astype(float); _yn = neu["y"].astype(int)
    _perm = np.random.RandomState(0).permutation(len(_yn))
    _mn = cu.make_mlp(); _mn.fit(_Xn[_perm[:560]], _yn[_perm[:560]])
    neural_auc = cu.eval_binary(_yn[_perm[560:]], _mn.predict_proba(_Xn[_perm[560:]])[:, 1])["roc_auc"]
    return heldout_auc, neural_auc


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"💡 Reveal solution": mo.md(
        r"""
        ```python
        m = cu.make_mlp(); m.fit(X, y)
        heldout_auc = cu.eval_binary(yh, m.predict_proba(Xh)[:, 1])["roc_auc"]

        Xn, yn = neu["X_neural"].astype(float), neu["y"].astype(int)
        perm = np.random.RandomState(0).permutation(len(yn))
        mn = cu.make_mlp(); mn.fit(Xn[perm[:560]], yn[perm[:560]])
        neural_auc = cu.eval_binary(yn[perm[560:]], mn.predict_proba(Xn[perm[560:]])[:, 1])["roc_auc"]
        ```
        Note we do **not** assert LOCO < within-cage here — on this bundle the gap is inverted (Cage 16 is a
        cleaner female control cage). The graded claim is the held-out AUC band and beating chance on neurons.
        """)})
    return


@app.cell(hide_code=True)
def _(heldout_auc, mo, neural_auc):
    _ok_ho = 0.80 <= heldout_auc <= 0.92          # band around pinned 0.860
    _ok_neu = neural_auc > 0.90                    # pinned ~1.000; assert well above chance
    _ok = _ok_ho and _ok_neu
    _color = "#2ca02c" if _ok else "#e45756"
    _msg = "PASS — the decoder graduates." if _ok else "check your fit/split"
    mo.md(
        f"""
        <div style="border:2px solid {_color};border-radius:8px;padding:10px 14px;background:#fafafa">
        <b>Self-check</b> ({_msg})<br>
        held-out Cage-16 AUC = <b>{heldout_auc:.3f}</b> — in band [0.80, 0.92]? <b>{_ok_ho}</b><br>
        neural-demo held-out AUC = <b>{neural_auc:.3f}</b> — beats chance (&gt;0.90)? <b>{_ok_neu}</b>
        </div>
        """
    )
    return


# ============================================================================ readout board (bottom)
@app.cell(hide_code=True)
def _(bench, go, mo, res_ho):
    _b = bench("B", "NB08", "held-out cage decode", 0.86)
    _auc = res_ho.get("roc_auc", float("nan"))
    _fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=_auc, delta={"reference": _b, "valueformat": ".3f"},
        title={"text": f"Gauge B · MISSION COMPLETE<br><sub>Cage 16 held-out AUC vs benchmark {_b:.2f}</sub>"},
        gauge={"axis": {"range": [0.5, 1.0]}, "bar": {"color": "#2ca02c"},
               "threshold": {"line": {"color": "#e45756", "width": 3}, "value": _b}}))
    _fig.update_layout(height=280, margin=dict(l=30, r=30, t=70, b=10), template="plotly_white")
    mo.vstack([mo.md("### Readout Board — final"), _fig])
    return


# ============================================================================ threw-away / ship-next
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What we threw away · how it breaks

        - **Time.** The decoder reads 19 window-summary features — it collapsed the whole 130-frame trajectory
          into means/maxes. A fast feint and a slow menace with the same summary look identical to it.
        - **Failure modes on this data.** (1) A decoder can secretly learn the *cage* instead of the behavior —
          Cage-16's tail-mark dropout could be a giveaway rather than the aggression; leave-one-cage-out is
          the guard, but only for the cages you held out. (2) The **label-noise ceiling** is real: ~16%
          identity error + boundary `mlp_fp` ambiguity caps accuracy no matter the model. (3) Under class
          imbalance a mis-set threshold silently wrecks precision or recall — the opto readout lives or dies
          on that one number.
        - **How would you analyze this?** You have a validated decoder. **Design the opto experiment it
          unblocks:** which cell type in VMHvl, what stim protocol, and — critically — *how would you
          time-align the decoder's per-event calls to the laser pulses* so a shift in detected aggression is
          causally attributable and not a threshold artifact?

        ## What we ship next

        The readout is validated: **11,700 raw numbers per event → one trustworthy decision on a cage you
        never trained on**, plus a pipeline that reads a population raster with the same code. Memo back from
        the circuit team: *"Readout validated — opto trial GREEN-LIT for Cage 16."*

        *Neuroscience connection (close):* every stage you built — detection, reference frames, features,
        rhythm, manifolds, syllables, grammar, decoding — has a neural twin, and you have now run the last one
        on actual neurons. Next week each stage gets pointed at brain data, and you already know how to run
        it.
        """
    )
    return


if __name__ == "__main__":
    app.run()

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
        # 08 · Training a decoder and testing it honestly

        **Why this notebook.** Over the previous notebooks you reduced each interaction from
        11,700 raw pose numbers to 19 features, then to a few principal components, a 2-D map, and a
        discrete syllable. The purpose of all that work is practical: to label behavior automatically,
        so a human does not have to score every frame by hand. In this notebook you build that
        automatic labeler — a **decoder** — and, just as importantly, you measure how well it works
        using a fair test.

        This is the same reason a behavioral neuroscientist quantifies behavior in the first place. If
        you later want to relate social behavior to the brain — say, to a recording in the medial
        prefrontal cortex (mPFC) — you first need an objective, reproducible readout of what each mouse
        is doing. A decoder provides that readout. Here you build one and check whether it can be
        trusted on data it has never seen.

        The notebook has three parts: **(1)** labels and the ceiling that label noise puts on any
        decoder, **(2)** training the decoder, and **(3)** testing it on a cage that was held out from
        the start. At the end we apply the *same* code to a small neural dataset, to show the method
        is not specific to pose.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Key terms (defined before we use them)

        - **Classifier / decoder.** A function that takes the measurements from one event (here, the
          19 features) and outputs a guess about a category — for us, *aggression* vs *not
          aggression*. "Decoder" is the same idea borrowed from neuroscience, where the input is neural
          activity and the output is a guess about behavior or a stimulus. We use the two words
          interchangeably.
        - **Training data vs held-out data.** We *train* (fit) the decoder on events whose correct
          answer we already know. To test it fairly, we then apply it to *held-out* events it never saw
          during training. Scoring a model on the same data it trained on flatters it; only held-out
          data shows whether it generalizes.
        - **Cross-validation.** A way to estimate held-out performance when data are limited: split the
          data into k parts, train on k−1 of them, test on the part left out, and repeat until every
          part has been tested once. Below we use 5-fold cross-validation (k = 5).
        - **Probability score.** Rather than a hard yes/no, the decoder outputs a number between 0 and
          1 — its estimated probability that the event is aggression. A **threshold** turns that score
          into a decision (for example, "call it aggression if the score is at least 0.5").
        - **ROC-AUC (also written AUROC).** A single number summarizing how well the scores separate
          the two classes, across every possible threshold. 1.0 is perfect ranking; 0.5 is chance (no
          better than a coin flip). Because it does not depend on the threshold you pick, it is a good
          overall summary.
        - **Precision and recall.** After you fix a threshold: *precision* is, of the events the
          decoder called aggression, the fraction that really were. *Recall* is, of the events that
          really were aggression, the fraction the decoder caught. Raising the threshold usually raises
          precision but lowers recall, and vice versa.
        """
    )
    return


# ============================================================================ data
@app.cell
def _(ROOT, cu):
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))     # kp, ranks, agg_label, category...
    der = cu.load_derived("train", ROOT)                                 # X (1500,19), pca_scores, cage, sex
    ho = cu.load_events(cu.data_path("data/heldout_events.npz", ROOT))   # Cage 16 — the held-out cage (470)
    hod = cu.load_derived("heldout", ROOT)                               # X (470,19), sex all 'F'
    sweep = cu.load_umap_sweep(ROOT)                                     # default_labels = canonical clusters
    neu = cu.load_neural_demo(ROOT)                                      # X_neural (800,60), y, emb2d
    return der, ev, ho, hod, neu, sweep


@app.cell
def _(der, ev, ho, hod):
    X = der["X"]                          # (1500,19) train features
    y = ev["agg_label"].astype(int)       # (1500,) ground-truth aggression
    cage = der["cage"]                    # (1500,) cages 9..15
    Xh = hod["X"]                         # (470,19) Cage 16 features
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
    # Readout Board. Gauge A = size of the representation (1 decision, the end of the Phase-1 collapse).
    # Gauge B = held-out ROC-AUC on Cage 16, with the committed benchmark marked as a threshold line.
    # FIX: Gauge B is mode="gauge+number" (delta removed). The delta compared the AUC to its own
    # benchmark (0.86 vs 0.86 ≈ 0.000), which added a meaningless "≈0.000" line; the benchmark is
    # already shown as the red threshold marker, so no delta is needed.
    _b = bench("B", "NB08", "held-out cage decode", 0.86)
    _auc = res_ho.get("roc_auc", float("nan"))
    _fig = go.Figure()
    _fig.add_trace(go.Indicator(
        mode="number", value=1,
        title={"text": "Gauge A · size of representation<br><sub>11,700 → 19 → 6 → 2-D → 1 decision</sub>"},
        number={"suffix": " decision"}, domain={"row": 0, "column": 0}))
    _fig.add_trace(go.Indicator(
        mode="gauge+number", value=_auc,
        title={"text": f"Gauge B · held-out ROC-AUC<br><sub>Cage 16, benchmark {_b:.2f} (red line)</sub>"},
        gauge={"axis": {"range": [0.5, 1.0]}, "bar": {"color": "#2ca02c"},
               "threshold": {"line": {"color": "#e45756", "width": 3}, "value": _b}},
        domain={"row": 0, "column": 1}))
    _fig.update_layout(grid={"rows": 1, "columns": 2}, height=260,
                       margin=dict(l=30, r=30, t=70, b=10), template="plotly_white")
    mo.vstack([mo.md("### Readout Board"), _fig])
    return


# ============================================================================ the held-out cage
@app.cell(hide_code=True)
def _(Xh, mo, yh):
    _n, _npos = len(yh), int(yh.sum())
    mo.md(
        f"""
        <div style="border:2px solid #2ca02c;border-radius:10px;padding:14px 18px;
        background:linear-gradient(90deg,#f0fff4,#ffffff)">
        <b>A cage held out for a fair test.</b><br>
        One cage — Camera 16 — was set aside from the very beginning. Nothing about it was used when we
        designed the features, ran the PCA, built the map, or labeled the clusters. That is
        deliberate: because the decoder never saw Cage 16 during any earlier step, its score here is an
        honest estimate of how the decoder would perform on genuinely new data. Cage 16 has
        <b>{_n} events</b> ({_npos} aggression, base rate {_npos/_n:.3f}) with <b>{Xh.shape[1]}
        features</b> each. It is a <b>female</b> cage, whereas the training cages mix male and female,
        so it also serves as a <b>cross-sex</b> test.
        </div>
        """
    )
    return


# ============================================================================ PART 1 — ground truth & ceiling
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 1 · Labels and the label-noise ceiling

        **Why this matters.** A decoder learns from labeled examples, so it can only be as reliable as
        those labels. If some labels are wrong, the decoder inherits that error: no model — however
        good — can exceed the quality of the answers it was trained and tested against. Before training
        anything, it is worth measuring how much the labels themselves disagree.

        **Definitions.** *Ground truth* means the labels we treat as correct (here, a human's judgment
        of whether an event is aggression). *Inter-rater reliability* measures how much two independent
        human labelers agree with each other. Below, **you** hand-score a few clips and we compare your
        labels to a single reference key. Note the distinction: comparing to one reference key measures
        your **accuracy against that key**, not true inter-rater reliability (which needs a second
        independent labeler — offered as a stretch below). We report *Cohen's κ*, a measure of
        agreement corrected for the amount you would expect by chance alone (0 = chance, 1 = perfect).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📋 Labeling guidelines (read before you score)": mo.md(
            r"""
            Call an event **aggression** when you see a committed, high-speed offensive act directed at
            the other mouse: a lunge, a chase to contact, a pin, or a bite attempt. In the clips, the
            white arrow points from the **approacher** toward the **approachee**, and the red dot marks
            the moment of contact.

            **Do not** call it aggression for: passive co-resting, sniffing or investigation,
            anogenital contact, grooming, or two mice simply resting on the ledge. The hardest cases
            are the clips tagged **`mlp_fp`** — events that an earlier model flagged as aggression but a
            human rejects. They sit right at the boundary of the definition, and how you score them is
            exactly where label noise enters.

            **Stretch (makes "inter-rater" honest):** have a labmate score the same eight clips
            independently, and compute κ between the two of you. That is genuine inter-rater
            reliability; comparing to the key below is only accuracy against a reference.
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
        _out = mo.md("*Score all eight clips above and click **Grade my labels** to measure your "
                     "agreement with the reference key.*")
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
            <b>Agreement with the reference key</b> (this is <i>accuracy against a reference</i>, not
            inter-rater reliability): raw accuracy <b>{_acc:.2f}</b> over {len(label_idx)} clips ·
            Cohen's κ <b>{_kappa:.2f}</b> (chance-corrected). Most disagreements fall on the
            <code>mlp_fp</code> clips — that leftover disagreement is the label noise that limits every
            decoder below.
            </div>
            """)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How the threshold changes the false-positive rate

        The `mlp_fp` events are the deliberately ambiguous near-misses. The slider below sets the
        decision **threshold** — the score above which the decoder calls an event aggression. Watch the
        decoder's **false-positive rate on exactly these ambiguous clips**: a low threshold calls many
        of them aggression (many false positives); raising it removes those false alarms but, on real
        data, would also start missing genuine attacks. There is no threshold that makes both errors
        zero — the ambiguity you felt while labeling is the same ambiguity the decoder must handle.
        """
    )
    return


@app.cell
def _(mo):
    fp_thr = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="decision threshold",
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
        **The label-noise ceiling.** If about 1 in 8 hand labels is wrong at the boundary (and NB01
        already noted that tail-mark identity carries roughly 16% error), then even a perfect model
        inherits that error as an upper limit on accuracy. We can show this directly: flip a fraction
        of the training labels on purpose and watch the held-out AUC fall. This refits several models,
        so it runs only when you click the button.
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
        _out = mo.md("*Click to corrupt a fraction of the training labels and see the held-out "
                     "AUC fall.*")
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


# ============================================================================ PART 2 — train
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 2 · Train the decoder

        **Why.** With labels in hand, we now fit the decoder and check that the earlier processing
        steps actually help it.

        **Method.** `cu.make_mlp()` builds a small scikit-learn pipeline. *Purpose:* turn features into
        an aggression probability. *Inputs:* a feature matrix `X` (rows = events, columns = features)
        and labels `y`. *Steps inside:* fill in any missing values with the column median, standardize
        each feature to comparable scale, then apply a small multi-layer perceptron (MLP) — a neural
        network with a couple of hidden layers. *Output:* an unfitted model; after `.fit(X, y)`,
        `.predict_proba(X)[:, 1]` returns one probability per event.

        As a check on whether the network's nonlinearity is doing real work, we also fit a **logistic
        regression** — a simpler, purely *linear* classifier — as a baseline. And we compare three
        feature sets, using 5-fold cross-validation *within the training cages*: the raw **19
        features**, the **PCA scores** from NB04, and the **19 features plus a one-hot code for cluster
        membership** from the NB05 map. (The 19 features already include the coordination signals from
        NB03 — `closing_speed`, the facing cosines, `heading_alignment` — so coordination is already
        represented.)
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
                           title="Feature-set comparison (within-cage cross-validation)",
                           margin=dict(l=10, r=10, t=50, b=10))
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two results are worth stating plainly. First, the **linear** baseline is nearly as good as the
        MLP, which tells you this problem is close to linearly separable in feature space — the
        nonlinearity buys little here. Second, **adding the cluster one-hots barely helps**: the
        clusters are a coarser summary of the same 19 features, not new information. That is the point
        of the whole feature-reduction process — the useful signal was already in the features.
        """
    )
    return


# ============================================================================ PART 3 — held-out test
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 3 · The held-out test on Cage 16

        **Why.** Cross-validation within the training cages is a good estimate, but the strongest test
        is a cage the decoder has never seen in any form. That is Cage 16.

        The test runs automatically, with no button: we train on cages 9–15 and evaluate on Cage 16.
        The ROC and precision–recall curves, the confusion matrix, and the calibration curve below are
        computed on that held-out cage.

        **Reading an ROC curve.** It plots the true-positive rate against the false-positive rate as
        the threshold varies. A curve that hugs the top-left corner is good; the diagonal is chance.
        The area under it is the ROC-AUC.
        """
    )
    return


@app.cell
def _(cu, res_ho, s_ho, yh):
    roc_fig = cu.roc_pr_fig(yh, s_ho)
    roc_fig.update_layout(title=f"Cage 16 held-out · ROC-AUC = {res_ho.get('roc_auc', float('nan')):.3f}")
    roc_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **The confusion matrix** (left) counts the four outcomes at a threshold of 0.5: correct
        rejections and detections on the diagonal, false positives and misses off it. **The
        calibration curve** (right) checks whether the probabilities mean what they say: for events the
        decoder scored around 0.7, did roughly 70% actually turn out to be aggression? A well-calibrated
        decoder follows the dotted diagonal.
        """
    )
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
        ### Choosing a decision threshold

        The threshold is a choice, not a fixed default, and it depends on what the readout is for. A
        false "attack" call inflates the apparent amount of aggression; a missed attack hides it. Slide
        the threshold and read **precision and recall on Cage 16** at each setting.
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
        **Within-cage versus leave-one-cage-out (LOCO).** A within-cage 70/30 split gives AUC ≈
        **{_within:.3f}**; the held-out Cage-16 (leave-one-cage-out) test gives AUC ≈ **{_loco:.3f}**.

        Usually held-out data is *harder*, because a model can pick up cage-specific quirks that do not
        transfer. Here the held-out test is actually a little *easier*: Cage 16 is a female control cage
        with a higher and cleaner aggression base rate (0.383), so the events are easier to separate.
        We report this rather than hide it. The honest gap is whatever the data show — the point of a
        held-out test is to measure that gap, not to assume its sign.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Which features the decoder relies on

        `permutation_importance` estimates how much each feature matters. *Purpose:* identify the
        features the decoder actually reads. *Method:* shuffle one feature's values across events
        (breaking its link to the label) and measure how far Cage-16 AUC drops; a large drop means the
        feature was important. *Inputs:* the fitted model, `Xh`, `yh`. *Output:* a mean drop per
        feature. It re-scores many times, so it runs on a button.
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
            title="Click the button above to see which features the decoder relies on",
            margin=dict(l=10, r=10, t=40, b=10))
    else:
        from sklearn.inspection import permutation_importance
        _r = permutation_importance(model, Xh, yh, n_repeats=10, random_state=0, scoring="roc_auc")
        _order = np.argsort(_r.importances_mean)
        _fig = go.Figure(go.Bar(x=_r.importances_mean[_order], y=[cu.FEATURE_NAMES[i] for i in _order],
                                orientation="h", marker_color="#4c78a8",
                                error_x=dict(type="data", array=_r.importances_std[_order])))
        _fig.update_layout(template="plotly_white", height=520,
                           title="Permutation importance on Cage 16",
                           xaxis_title="drop in ROC-AUC when shuffled", margin=dict(l=10, r=10, t=50, b=10))
        _out = _fig
    _out
    return


# ---- the example event, as read by the decoder ----
@app.cell(hide_code=True)
def _(cu, ev, mo, s_tr):
    _EX = 909      # our example event: Cage 15, male, aggression, contact_rel=40
    _gif = cu.event_gif_bytes(ev["kp"][_EX], ev["ranks"][_EX], int(ev["contact_rel"][_EX]), cell=200)
    mo.md(
        f"""
        **Our example approach event (#{_EX}, Cage 15, male).** This is the same interaction we have
        followed throughout the course: the **approacher** is the Dom mouse (red) and the
        **approachee** is the Sub mouse (green). We tracked it from the raw skeleton, through
        body-frame geometry, to a point on the map, to a syllable. The decoder now scores it:
        **P(aggression) = {s_tr[_EX]:.2f}**.

        Note this is an in-sample check — event #909 is a training event — so the number is a
        demonstration, not evidence. The honest evidence is the Cage-16 curve above.

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
        **The decoder's most confident aggression calls on Cage 16** (top four by score: {_scores}).
        These are events from the held-out cage — the decoder never trained on any of them. Watch the
        clips and judge for yourself whether they look like aggression; this is what the AUC number
        represents concretely.

        {cu.gif_img_html(_grid, width=600)}
        """
    )
    return


# ============================================================================ threshold + hypothetical experiment
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Using the decoder to detect a change in behavior

        A common use of a validated decoder is to detect whether some manipulation changed behavior.
        Suppose a manipulation increases the number of attacks. We can imitate that by resampling Cage
        16 to add more aggression events, then ask whether *this* decoder, at the current threshold,
        registers the increase. Move the slider to set how large the increase is and watch the detected
        rate change.
        """
    )
    return


@app.cell
def _(mo):
    opto_mult = mo.ui.slider(1.0, 3.0, value=2.0, step=0.25, label="attack-frequency multiplier",
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
    _catches = ("the decoder detects the change" if (_stim_rate - _base_rate) > 0.03
                else "the shift is too small for the decoder to register")
    _fig = go.Figure(go.Bar(x=["baseline", f"increased ×{opto_mult.value:g}"],
                            y=[_base_rate, _stim_rate], marker_color=["#9aa0a6", "#e45756"],
                            text=[f"{_base_rate:.2f}", f"{_stim_rate:.2f}"], textposition="outside"))
    _fig.update_layout(template="plotly_white", height=330,
                       title=f"Detected-aggression rate @ threshold 0.5 — {_catches}",
                       yaxis_title="fraction flagged aggression", yaxis_range=[0, 1],
                       margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([opto_mult, _fig])
    return


# ============================================================================ same pipeline on neural data
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## The same pipeline applied to neural data

        **Why this is here.** The word "decoder" comes from neuroscience, where it means a model that
        reads a category out of neural activity. The pipeline you just built is not specific to pose —
        it is a general classifier. To show that, we apply the **identical** `cu.make_mlp` and
        `cu.eval_binary` functions to a small population dataset: 800 trials × 60 neurons, where a
        hidden binary brain state modulates firing. The task is to decode that hidden state from the
        neural activity, using the same train/held-out split you used for behavior.

        (The neural dataset here is synthetic and cleanly separable, so it is easy on purpose. It
        demonstrates the method, not a claim about real recordings — which are messier, and where the
        honest held-out discipline you practiced on Cage 16 matters even more.)
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
        **Held-out neural decode: ROC-AUC = {neu_auc:.3f}.** The same pipeline that read aggression
        from pose reads the hidden brain state from spike counts. The high score reflects the fact that
        this synthetic dataset is cleanly separable, which is the point of the demonstration. Real
        neural populations are noisier, and the held-out test is exactly what you would use to check a
        real decoder.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### A joint neural–behavior embedding

        The precomputed 2-D embedding below places every trial by its neural population activity,
        colored by the hidden state. Methods such as CEBRA aim to put behavior and brain activity in a
        single low-dimensional space like this so they can be compared directly.
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
        mo.accordion({"How CEBRA actually uses behavior": mo.md(
            r"""
            CEBRA does not merge brain and behavior symmetrically. It uses behavior (or time, or a
            label) as a signal to *shape* a neural embedding — pulling together trials that share a
            behavioral context and pushing apart trials that do not. So the "single space" is a neural
            embedding organized by behavior, not a true fusion of the two. *Schneider, Lee & Mathis
            2023, Nature. The figure here is a synthetic demonstration, not a fitted CEBRA model.*
            """)})
    ])
    return


# ============================================================================ references
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "References — neural decoding": mo.md(
            r"""
            A leave-one-cage-out decoder faces the same demand as a population decoder that must work on
            a *held-out recording session*: fit a classifier on one set of data and ask it to read new
            data. The underlying method is the same whether the rows are events or trials and the
            columns are pose features or neurons.

            - Georgopoulos et al. 1986, *Science* — population-vector decoding of movement direction.
            - Glaser et al. 2020, *eNeuro* — machine-learning methods for neural decoding.
            - Gilja et al. 2012, *Nat. Neurosci.* — a brain–machine interface that stays stable across
              sessions.
            - Padilla-Coreano et al. 2022, *Nature* — decoded competitive rank from mPFC ensembles
              using tracking plus a statistical model; a neural counterpart to the behavior decoder here.
            - Schneider, Lee & Mathis 2023, *Nature* — CEBRA.

            One caution carried from NB06: tube-test rank and homecage aggression are related but
            distinct measures, so a decoder trained on one does not automatically read the other.
            """)
    })
    return


# ============================================================================ exercise
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise — build and check the decoder

        **Goal.** Reproduce the two core results of this notebook: (1) train the decoder on cages 9–15
        and score it on the held-out Cage 16, and (2) apply the same pipeline to the neural demo.

        **Toolbox**
        - `cu.make_mlp()` → an unfitted pipeline. Call `.fit(X, y)`, then `.predict_proba(X)[:, 1]` to
          get one aggression probability per event.
        - `cu.eval_binary(y_true, y_score)` → a dict including `"roc_auc"`. It takes the **score
          vector**, not the model.
        - `X, y` = training features/labels (cages 9–15); `Xh, yh` = Cage-16 features/labels;
          `neu["X_neural"]` (800×60) and `neu["y"]` (800,) = the neural demo.

        **Fill in the blanks.** In the next cell, two lines are marked `# <<< EDIT`. Each already
        contains the correct code so the notebook runs; the comment tells you what to write. Try
        rewriting each marked line yourself from the scaffold below, then re-run and compare against the
        self-check.

        ```python
        # 1) held-out behavior decode
        m = cu.make_mlp()
        m.fit(X, y)                                   # train on the seven training cages
        scores_ho = m.predict_proba(____)[:, 1]       # <<< score the HELD-OUT features (Xh, not X)
        heldout_auc = cu.eval_binary(yh, scores_ho)["roc_auc"]

        # 2) neural decode with the same pipeline
        neural_auc = cu.eval_binary(yn_test, m2.predict_proba(Xn_test)[:, 1])["____"]  # <<< "roc_auc"
        ```

        **What to expect.** The self-check should report a held-out Cage-16 AUC near **0.86** (in the
        band 0.80–0.92) and a neural-demo AUC above **0.90**. If your held-out AUC comes out
        suspiciously high (near 1.0), you probably scored `X` instead of `Xh` — that is testing on the
        training data.
        """
    )
    return


@app.cell
def _(X, Xh, cu, neu, np, y, yh):
    # ---- Fill in the two lines marked  # <<< EDIT.  Everything else is done for you. ----

    # (1) Held-out behavior decode: train on (X, y), then score the HELD-OUT cage.
    _m = cu.make_mlp(); _m.fit(X, y)
    _scores_ho = _m.predict_proba(Xh)[:, 1]                 # <<< EDIT: score Xh (the held-out features), not X
    heldout_auc = cu.eval_binary(yh, _scores_ho)["roc_auc"]

    # (2) Neural decode: same pipeline, a held-out split of the neural demo.
    _Xn = neu["X_neural"].astype(float); _yn = neu["y"].astype(int)
    _perm = np.random.RandomState(0).permutation(len(_yn))
    _mn = cu.make_mlp(); _mn.fit(_Xn[_perm[:560]], _yn[_perm[:560]])
    neural_auc = cu.eval_binary(_yn[_perm[560:]], _mn.predict_proba(_Xn[_perm[560:]])[:, 1])["roc_auc"]  # <<< EDIT: read "roc_auc"
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
        The two edits are: score `Xh` (not `X`) for the held-out behavior AUC, and read the
        `"roc_auc"` key for the neural AUC. We deliberately do **not** expect held-out AUC to be lower
        than within-cage here — on this dataset the held-out cage is a cleaner female control cage, so
        the gap is reversed. The graded claims are the held-out AUC band and beating chance on neurons.
        """)})
    return


@app.cell(hide_code=True)
def _(heldout_auc, mo, neural_auc):
    _ok_ho = 0.80 <= heldout_auc <= 0.92          # band around pinned 0.860
    _ok_neu = neural_auc > 0.90                    # pinned ~1.000; assert well above chance
    _ok = _ok_ho and _ok_neu
    _color = "#2ca02c" if _ok else "#e45756"
    _msg = "PASS" if _ok else "check your fit and split"
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
    # Final Readout Board. FIX: Gauge B is mode="gauge+number" (delta removed) for the same reason as
    # the top board — the benchmark is shown as the red threshold marker, so a delta against it is
    # redundant and rendered a meaningless "≈0.000".
    _b = bench("B", "NB08", "held-out cage decode", 0.86)
    _auc = res_ho.get("roc_auc", float("nan"))
    _fig = go.Figure(go.Indicator(
        mode="gauge+number", value=_auc,
        title={"text": f"Gauge B · held-out ROC-AUC<br><sub>Cage 16, benchmark {_b:.2f} (red line)</sub>"},
        gauge={"axis": {"range": [0.5, 1.0]}, "bar": {"color": "#2ca02c"},
               "threshold": {"line": {"color": "#e45756", "width": 3}, "value": _b}}))
    _fig.update_layout(height=280, margin=dict(l=30, r=30, t=70, b=10), template="plotly_white")
    mo.vstack([mo.md("### Readout Board — final"), _fig])
    return


# ============================================================================ limits / next
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What this decoder ignores, and how it can fail

        - **Time.** The decoder reads 19 window-summary features, which collapse the whole 130-frame
          trajectory into means and maxima. A fast feint and a slow approach with the same summary look
          identical to it.
        - **Failure modes on this data.**
          1. A decoder can accidentally learn the *cage* instead of the behavior — a cage-specific
             quirk such as tail-mark dropout can be a giveaway. Leave-one-cage-out is the guard against
             this, but only for the cages you actually held out.
          2. The **label-noise ceiling** is real: roughly 16% identity error plus the boundary
             `mlp_fp` ambiguity caps accuracy no matter how good the model is.
          3. Under class imbalance, a poorly chosen threshold quietly wrecks precision or recall.
             Whenever the readout is a single detected rate, that one number depends on the threshold.
        - **A question to think through.** You now have a validated decoder. If you were to use it to
          test a manipulation, how would you align the decoder's per-event calls in time to the
          manipulation, so that a change in detected aggression can be attributed to the manipulation
          rather than to a threshold artifact?

        ## What comes next

        The readout is validated: 11,700 raw numbers per event reduced to one trustworthy decision on a
        cage the decoder never trained on, plus a demonstration that the same code reads a neural
        population. In Week 2, the same processing steps — detection, reference frames, features,
        rhythm, dimensionality reduction, and decoding — are applied to real brain-imaging data.
        """
    )
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "plotly>=5.20",
#     "h5py>=3.8",
#     "gdown>=5.0",
#     "openpyxl>=3.1",
#     "imageio>=2.34",
#     "imageio-ffmpeg>=0.4",
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
            if os.path.isdir(os.path.join(p, "course")):
                return p
            p = os.path.dirname(p)
        return None
    ROOT = _find_root() or os.getcwd()
    _nu = os.path.join(ROOT, "course", "neural_utils.py")
    if not os.path.exists(_nu):
        os.makedirs(os.path.dirname(_nu), exist_ok=True)
        urllib.request.urlretrieve(_RAW + "/course/neural_utils.py", _nu)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import neural_utils as nu
    CACHE = nu.cache_dir(ROOT)
    return CACHE, ROOT, go, np, nu


# ============================================================================ briefing
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB14 · Decoding Social State from Neural Activity

        **Week 2 · using real calcium imaging data.**

        **Why this matters.** In Week 1 you built tools that turn raw pose into a compact description
        of behavior, and in NB08 you trained a classifier that reads a behavioral label (aggression)
        off pose features. In this notebook you apply the same kind of classifier to a different
        input: the activity of real neurons recorded while a mouse behaves socially. The scientific
        question is whether the recorded population of cells carries information about a social
        behavior, and whether social isolation changes how that information is distributed across
        cells.

        **Definitions used throughout.**

        - **Decode a behavior from neural activity.** Train a model that takes the neurons' activity
          at a moment in time and predicts a behavioral label for that moment (here, whether the
          focal mouse is socially engaging). If the model predicts above chance on frames it was not
          trained on, we say the behavior can be *decoded* from the population.
        - **Population vector.** The list of every recorded neuron's activity at one time frame. For a
          session with 218 neurons, each frame is a vector of 218 numbers. The decoder's input is one
          such vector per frame.
        - **Cross-validation.** To check that the model generalizes rather than memorizes, we split
          the frames into folds, train on some folds, and score on the held-out fold. Every frame is
          scored by a model that never saw it during training. This is the same idea used in NB08.
        - **AUROC** (area under the ROC curve). A single number summarizing how well the predicted
          probabilities separate the two classes. 0.5 is chance; 1.0 is perfect. This is the same
          yardstick used in NB08.

        **The plan.** (1) Put the calcium on the same clock as the behavior. (2) Look for individual
        cells whose activity differs during social vs non-social frames. (3) Count those cells across
        isolation conditions. (4) Train a population decoder and evaluate it with cross-validated
        AUROC.

        The decode target throughout is `is_social_sender`: a per-frame boolean that is True when the
        focal mouse is socially engaging the intruder.
        """
    )
    return


# ============================================================================ load
@app.cell
def _(nu):
    _d = nu.load_si()
    ent = _d["entrances"]
    beh = _d["behavior"]
    img = _d["imaging"]
    n_sessions = _d["n_sessions"]
    behavior_fps = _d["behavior_fps"]
    imaging_fps = _d["imaging_fps"]
    # condition label per session ("control" / "24hr" / "7d")
    cond_labels = [nu.si_condition_label(ent["Isolation Length"].iloc[s]) for s in range(n_sessions)]
    return beh, behavior_fps, cond_labels, ent, imaging_fps, img, n_sessions


@app.cell(hide_code=True)
def _(cond_labels, img, mo, n_sessions):
    _counts = {c: cond_labels.count(c) for c in ["control", "24hr", "7d"]}
    _neur = [im.shape[1] for im in img]
    mo.md(
        f"""
        **Dataset: SI3_2022 social-isolation cohort.** {n_sessions} sessions, each pairing miniscope
        calcium imaging with frame-by-frame social behavior scoring. In every session a focal mouse
        (group-housed **control**, or isolated for **24hr** or **7d**) meets an intruder; we record
        the focal mouse's neural activity and label each frame for whether it is socially engaging the
        intruder. Sessions per condition: **{_counts}**.

        Neuron counts differ across sessions (**{min(_neur)}–{max(_neur)}** cells) because each
        recording extracts its own set of cells from its own field of view, with no correspondence
        between sessions. This is the reason the decoder later is trained and tested *within* a single
        session: there is no shared neuron identity that would let a model trained on one session be
        applied to another.

        The two signals are sampled on different clocks: calcium at **30 fps**, behavior at
        **25 fps**. To line up a neuron's activity with a behavioral label, the calcium must first be
        resampled onto the behavior clock. That is the first step.
        """
    )
    return


# ============================================================================ interpolation sidebar
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Put the calcium on the behavior clock

        **Why.** A neuron's activity and a behavioral label must be indexed by the same time frames
        before we can ask whether they relate. Because calcium is sampled at 30 fps and behavior at
        25 fps, the two arrays have different lengths and do not line up frame for frame.

        **Method: `nu.interp_resample(C, n_out, axis=0)`.**

        - *Purpose:* resample a signal to a new number of time points by linear interpolation.
        - *Inputs:* `C`, an array of shape `(n_frames, n_neurons)`; `n_out`, the target number of
          frames (here, the length of the behavior array).
        - *Output:* an array of shape `(n_out, n_neurons)` on the behavior clock.

        Linear interpolation re-grids the samples that are already there; it does not create new
        peaks. The figure below shows one real neuron's trace at its original 30 fps sampling and
        after resampling to the 25 fps behavior length, so you can confirm the shape is preserved.

        After resampling we **z-score** each neuron (subtract its mean, divide by its standard
        deviation, so all cells are on a comparable scale) and **crop** to the first 3 minutes after
        the intruder enters (`Int_Entry`), the window where the social interaction occurs.
        """
    )
    return


@app.cell
def _(go, img, np, nu):
    # A real neuron, original 30 fps samples vs resampled onto a coarser behavior-length grid.
    _C = img[6][:, 40]                      # one neuron, session 6
    _seg = _C[:120]                         # first 120 frames for legibility
    _res = nu.interp_resample(_seg, int(len(_seg) * 25 / 30))
    _fig = go.Figure()
    _fig.add_scatter(x=np.linspace(0, 1, len(_seg)), y=_seg, mode="lines+markers",
                     line=dict(color="#4c78a8", width=1), marker=dict(size=4),
                     name="calcium @ 30 fps (original)")
    _fig.add_scatter(x=np.linspace(0, 1, len(_res)), y=_res, mode="markers",
                     marker=dict(color="#e45756", size=7, symbol="x"),
                     name="resampled → 25 fps behavior clock")
    _fig.update_layout(template="plotly_white", height=300,
                       margin=dict(l=10, r=10, t=40, b=10),
                       title="interp_resample: one neuron, 30 fps → 25 fps (shape preserved)",
                       xaxis_title="normalized time [0,1]", yaxis_title="calcium (a.u.)",
                       legend=dict(y=1.0))
    _fig
    return


# ============================================================================ session picker + pipeline
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. The population raster with the social overlay

        **Why.** Before decoding, it helps to look at the whole population at once and check whether
        activity visibly changes during social frames.

        **Definition.** A **raster** is a heatmap with one row per neuron and one column per time
        frame; color is that neuron's z-scored activity. It shows the entire population's activity
        over time in a single image.

        **Method.** Pick a session. We run the full pipeline for it —
        `interp_resample → zscore(axis=0) → crop [entry, entry + 3·60·25]` — and draw the z-scored
        population as a raster. The **green bands** mark frames where the focal mouse is socially
        engaging (`is_social_sender`); unmarked frames are non-social. If the population carries
        social information, the activity under the green bands should look different from the rest.
        """
    )
    return


@app.cell
def _(cond_labels, mo, n_sessions):
    _opts = {f"session {s}  ·  {cond_labels[s]}": s for s in range(n_sessions)}
    session_pick = mo.ui.dropdown(options=_opts, value="session 6  ·  7d",
                                  label="session (condition)")
    return (session_pick,)


@app.cell
def _(beh, behavior_fps, ent, img, np, nu, session_pick):
    # Full NB14 pipeline for the selected session: resample -> zscore -> crop.
    # Order matters: the pinned social-neuron counts use resample -> zscore -> crop (see contract #6).
    _s = int(session_pick.value)
    _iss = beh[_s]["is_social_sender"].astype(bool)
    _r = nu.zscore(nu.interp_resample(img[_s], len(_iss), axis=0), axis=0)
    _e = int(ent["Int_Entry"].iloc[_s])
    _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
    sess_neurons = _r[_t0:_t1]                 # (T, n_neurons) z-scored, cropped
    sess_social = _iss[_t0:_t1]                # (T,) bool
    sess_ncells = sess_neurons.shape[1]
    sess_frac = float(sess_social.mean())
    return sess_frac, sess_ncells, sess_neurons, sess_social


@app.cell
def _(go, mo, np, nu, session_pick, sess_frac, sess_ncells, sess_neurons, sess_social):
    # Raster (neurons x time) with green vrects over contiguous social bouts.
    _R = sess_neurons.T                          # (n_neurons, T)
    _fig = nu.raster_fig(_R, title=(f"{session_pick.selected_key}  ·  {sess_ncells} neurons  ·  "
                                    f"{sess_frac:.0%} of frames social"),
                         xlabel="time (frames, 25 fps)", ylabel="neuron",
                         zmin=-3, zmax=3, colorbar_title="z", height=460)
    # contiguous runs of social frames -> a handful of green shapes (cheap)
    _s = sess_social.astype(int)
    _edges = np.flatnonzero(np.diff(np.r_[0, _s, 0]))
    _starts, _ends = _edges[0::2], _edges[1::2]
    for _a, _b in zip(_starts, _ends):
        _fig.add_vrect(x0=_a, x1=_b, fillcolor="#2ca02c", opacity=0.18, line_width=0, layer="below")
    _fig.add_annotation(x=0.01, y=1.06, xref="paper", yref="paper", showarrow=False,
                        text="green = social (is_social_sender)", font=dict(color="#2ca02c", size=12))
    mo.vstack([session_pick, _fig])
    return


# ============================================================================ per-neuron histogram
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. One neuron at a time: social vs non-social

        **Why.** The raster shows the whole population. To build intuition, we now look at a single
        cell and ask whether its activity distribution differs between social and non-social frames.

        **Method.** For the chosen neuron we split its z-scored activity into **social** frames and
        **non-social** frames and overlay the two distributions. If the two distributions are shifted
        apart, that neuron's activity depends on social state. If they sit on top of each other, the
        neuron carries little social information. The title reports a simple summary statistic: the
        ratio of mean absolute activity in social vs non-social frames. A cell is flagged as a
        candidate social neuron when that ratio exceeds 1.5. Use the slider to move through the
        neurons and see which ones separate.
        """
    )
    return


@app.cell
def _(mo, sess_ncells):
    neuron_ind = mo.ui.slider(0, sess_ncells - 1, value=min(15, sess_ncells - 1), step=1,
                              label="neuron index", debounce=True, full_width=True)
    return (neuron_ind,)


@app.cell
def _(go, mo, neuron_ind, np, sess_neurons, sess_social):
    _i = int(neuron_ind.value)
    _x = sess_neurons[:, _i]
    _soc = _x[sess_social]
    _non = _x[~sess_social]
    _bins = dict(start=-5, end=5, size=0.2)
    _fig = go.Figure()
    _fig.add_histogram(x=_non, histnorm="probability density", name="non-social",
                       marker_color="#7f7f7f", opacity=0.55, xbins=_bins)
    _fig.add_histogram(x=_soc, histnorm="probability density", name="social",
                       marker_color="#2ca02c", opacity=0.55, xbins=_bins)
    # simple effect-size readout: |mean(soc)| / |mean(non)| ratio, the NB14 "ratio" statistic
    _ratio = float(np.abs(_soc).mean() / (np.abs(_non).mean() + 1e-12))
    _delta = float(_soc.mean() - _non.mean())
    _fig.update_layout(template="plotly_white", height=380, barmode="overlay",
                       margin=dict(l=10, r=10, t=50, b=10),
                       title=(f"neuron {_i}:  |soc|/|non| ratio = {_ratio:.2f}  "
                              f"(social-neuron if > 1.5)   ·   Δmean = {_delta:+.2f}"),
                       xaxis_title="activity (z-score)", yaxis_title="density", legend=dict(y=1.0))
    mo.vstack([neuron_ind, _fig])
    return


# ============================================================================ social-neuron mask
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Which cells are social neurons?

        **Why.** Rather than inspect cells one at a time, we apply the same test to every neuron at
        once and count how many pass.

        **Method: `nu.social_neuron_mask(neurons, is_social, method="ratio")`.**

        - *Purpose:* flag, for each neuron, whether its activity differs enough between social and
          non-social frames to count as a social neuron.
        - *Inputs:* `neurons`, the z-scored, cropped array `(T, n_neurons)`; `is_social`, the
          per-frame boolean `(T,)`; `method`, which statistic to use (`"ratio"` by default).
        - *Output:* a boolean array `(n_neurons,)`; `.sum()` gives the count of social neurons.

        The bar chart shows every neuron's ratio, sorted, with your threshold line. Drag the
        threshold and watch the count change. Green bars are cells currently flagged as social.
        """
    )
    return


@app.cell
def _(mo):
    ratio_thr = mo.ui.slider(1.0, 3.0, value=1.5, step=0.1,
                             label="social-neuron ratio threshold", debounce=True, full_width=True)
    return (ratio_thr,)


@app.cell
def _(go, mo, np, ratio_thr, sess_neurons, sess_social):
    _soc = np.abs(sess_neurons[sess_social]).mean(axis=0)
    _non = np.abs(sess_neurons[~sess_social]).mean(axis=0)
    _ratio = np.where(_non > 0, _soc / _non, 0.0)
    _order = np.argsort(_ratio)
    _rs = _ratio[_order]
    _thr = float(ratio_thr.value)
    _is_soc = _rs > _thr
    _n = int(_is_soc.sum())
    _fig = go.Figure()
    _fig.add_bar(x=np.arange(len(_rs)), y=_rs,
                 marker_color=np.where(_is_soc, "#2ca02c", "#c7c7c7"),
                 showlegend=False)
    _fig.add_hline(y=_thr, line=dict(color="#e45756", width=2, dash="dash"),
                   annotation_text=f"threshold {_thr:.1f}", annotation_position="top left")
    _fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=50, b=10),
                       title=f"{_n} social neurons of {len(_rs)}  (ratio > {_thr:.1f})",
                       xaxis_title="neuron (sorted by ratio)", yaxis_title="|soc| / |non| ratio")
    mo.vstack([ratio_thr, _fig])
    return


# ============================================================================ condition comparison
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Does isolation change the count?

        **Why.** The cohort-level question: does social isolation change how many cells carry the
        social signal?

        **Method.** For every session we run the pipeline, count social neurons, and group the counts
        by isolation condition. The dropdown offers three detection methods (`ratio`, `delta`,
        `percentile`); they disagree in the details, which is itself worth noting. With the default
        `ratio` method, controls tend to have the most social neurons and isolated animals fewer.

        **Read this honestly.** There are only **6 sessions per condition**, and the
        session-to-session spread is wide. We report the *direction* of the trend, not a significance
        claim. This is a descriptive observation, not a hypothesis test.
        """
    )
    return


@app.cell
def _(mo):
    method_pick = mo.ui.dropdown(options=["ratio", "delta", "percentile"], value="ratio",
                                 label="social-neuron detection method")
    return (method_pick,)


@app.cell
def _(beh, behavior_fps, cond_labels, ent, go, img, method_pick, mo, np, nu):
    # Sweep all sessions with the selected method; group counts by condition.
    _m = method_pick.value
    _per = []
    for _s in range(len(img)):
        _iss = beh[_s]["is_social_sender"].astype(bool)
        _r = nu.zscore(nu.interp_resample(img[_s], len(_iss), axis=0), axis=0)
        _e = int(ent["Int_Entry"].iloc[_s])
        _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
        _mask = nu.social_neuron_mask(_r[_t0:_t1], _iss[_t0:_t1], method=_m)
        _per.append(int(_mask.sum()))
    _per = np.array(_per)
    _conds = ["control", "24hr", "7d"]
    _fig = go.Figure()
    for _c in _conds:
        _vals = _per[[i for i, cc in enumerate(cond_labels) if cc == _c]]
        _fig.add_box(y=_vals, name=f"{_c}\n(mean {_vals.mean():.1f})", boxpoints="all",
                     pointpos=0, jitter=0.4, marker_color="#4c78a8", line_color="#4c78a8")
    _means = {c: float(_per[[i for i, cc in enumerate(cond_labels) if cc == c]].mean()) for c in _conds}
    _fig.update_layout(template="plotly_white", height=400, showlegend=False,
                       margin=dict(l=10, r=10, t=50, b=10),
                       title=(f"social-neuron count by isolation ({_m}) — "
                              f"control {_means['control']:.1f} · 24hr {_means['24hr']:.1f} · "
                              f"7d {_means['7d']:.1f}  (n=6 each)"),
                       yaxis_title="n social neurons per session")
    mo.vstack([method_pick, _fig])
    return


# ============================================================================ decoder
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. Decode social state from the population

        **Why.** The individual-neuron tests above look at one cell at a time. A behavior may be
        represented across many cells jointly, in a way no single neuron makes obvious. A population
        decoder tests whether the whole population, taken together, carries enough information to
        predict social state.

        **Method.** This is the same procedure as NB08, with a different input. There you fed 19 pose
        features into a `LogisticRegression` and cross-validated an aggression call. Here the input is
        the **population vector** (every neuron's z-scored activity at a frame) and the label is
        `is_social_sender`. We use 5-fold stratified cross-validation, so every frame is scored by a
        model that did not train on it, and report **AUROC** on the same 0.5-is-chance scale.

        Because each session has its own set of neurons, the decoder is trained and tested within a
        single session. A decoder that transferred across sessions would require matching neurons
        between recordings, which this dataset does not provide.
        """
    )
    return


@app.cell
def _(np, session_pick, sess_neurons, sess_social):
    # Population decoder: LogisticRegression, 5-fold stratified CV, AUROC (same procedure as NB08).
    from sklearn.linear_model import LogisticRegression as _LogReg
    from sklearn.preprocessing import StandardScaler as _Scaler
    from sklearn.pipeline import make_pipeline as _mkpipe
    from sklearn.model_selection import StratifiedKFold as _SKF, cross_val_predict as _cvp
    from sklearn.metrics import roc_auc_score as _auc_score, roc_curve as _roc_curve

    _X = sess_neurons
    _y = sess_social.astype(int)
    _clf = _mkpipe(_Scaler(), _LogReg(max_iter=1000, class_weight="balanced"))
    _skf = _SKF(5, shuffle=True, random_state=0)
    dec_proba = _cvp(_clf, _X, _y, cv=_skf, method="predict_proba")[:, 1]
    dec_y = _y
    dec_auc = float(_auc_score(dec_y, dec_proba))
    _fpr, _tpr, _thrs = _roc_curve(dec_y, dec_proba)
    dec_fpr, dec_tpr, dec_thrs = _fpr, _tpr, _thrs
    return dec_auc, dec_fpr, dec_proba, dec_thrs, dec_tpr, dec_y


@app.cell(hide_code=True)
def _(dec_auc, go, mo, session_pick):
    _fig = go.Figure(go.Indicator(
        mode="gauge+number", value=dec_auc,
        number={"valueformat": ".3f"},
        title={"text": f"population decoder AUROC · {session_pick.selected_key}"},
        gauge={"axis": {"range": [0.5, 1.0]},
               "bar": {"color": "#2ca02c" if dec_auc > 0.7 else "#e45756"},
               "steps": [{"range": [0.5, 0.6], "color": "#f2f2f2"},
                         {"range": [0.6, 0.7], "color": "#e6e6e6"}],
               "threshold": {"line": {"color": "#333", "width": 3}, "value": 0.5}}))
    _fig.update_layout(template="plotly_white", height=240, margin=dict(l=30, r=30, t=60, b=10))
    _fig
    return


# ============================================================================ decision-threshold slider
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Choosing a decision threshold

        **Why.** The decoder outputs a *probability*, not a yes/no call. A **decision threshold**
        converts the probability into a label: predict "social" when the probability is at or above
        the threshold. The choice is a trade-off. A low threshold labels more frames social, catching
        real bouts but also raising false positives; a high threshold is stricter, with fewer false
        positives but more missed bouts.

        **Definitions.** *Precision* is the fraction of frames the decoder called social that really
        are social. *Recall* is the fraction of truly social frames the decoder caught. The ROC curve
        plots the true-positive rate against the false-positive rate across all thresholds; the marker
        shows where the current threshold sits on that curve.

        **Method.** Slide the threshold and read the operating point on the ROC curve, together with
        precision, recall, and the confusion counts, all computed on the held-out
        (cross-validated) predictions.
        """
    )
    return


@app.cell
def _(mo):
    thr_slider = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05,
                              label="decision threshold", debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(dec_auc, dec_fpr, dec_proba, dec_tpr, dec_y, go, mo, np, thr_slider):
    _thr = float(thr_slider.value)
    _pred = (dec_proba >= _thr).astype(int)
    _tp = int(((_pred == 1) & (dec_y == 1)).sum())
    _fp = int(((_pred == 1) & (dec_y == 0)).sum())
    _fn = int(((_pred == 0) & (dec_y == 1)).sum())
    _tn = int(((_pred == 0) & (dec_y == 0)).sum())
    _prec = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
    _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
    # operating point on the ROC: this threshold's (fpr, tpr)
    _fpr_here = _fp / (_fp + _tn) if (_fp + _tn) else 0.0
    _tpr_here = _rec
    _roc = go.Figure()
    _roc.add_scatter(x=dec_fpr, y=dec_tpr, mode="lines", line=dict(color="#4c78a8", width=2),
                     name=f"ROC (AUC {dec_auc:.3f})")
    _roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                     line=dict(color="#bbb", width=1, dash="dash"), name="chance")
    _roc.add_scatter(x=[_fpr_here], y=[_tpr_here], mode="markers",
                     marker=dict(color="#e45756", size=13, symbol="x"),
                     name=f"threshold {_thr:.2f}")
    _roc.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10),
                       title=(f"@ threshold {_thr:.2f}:  precision {_prec:.2f} · recall {_rec:.2f}  "
                              f"(TP {_tp} · FP {_fp} · FN {_fn} · TN {_tn})"),
                       xaxis_title="false-positive rate", yaxis_title="true-positive rate",
                       legend=dict(y=0.05, x=0.55))
    _roc.update_xaxes(range=[-0.02, 1.02])
    _roc.update_yaxes(range=[-0.02, 1.02], scaleanchor="x", scaleratio=1)
    mo.vstack([thr_slider, _roc])
    return


# ============================================================================ citation
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Background — social isolation and neural decoding (and the limits of this analysis)": mo.md(
            r"""
            **Social isolation reshapes social circuits.** Matthews et al. 2016 *Cell* (dorsal raphe
            dopamine encodes a loneliness-like state after acute isolation); Zelikowsky et al. 2018
            *Cell* (chronic isolation, Tac2/neurokinin B, amygdala and hypothalamus). **Coding of
            conspecifics and social state:** Remedios et al. 2017 *Nature* (VMHvl mixed social
            representation); Kingsbury et al. 2019 *Cell* (interbrain neural coding of dominance).
            **Population decoding of social variables:** Padilla-Coreano et al. 2022 *Nature*
            (mPFC ensembles decode competitive rank and social behavior) — the closest methodological
            precedent for the decoder you just ran, and the same paper NB08 referenced from the
            behavioral side.

            **Shared method.** A population decoder is a supervised map from a high-dimensional state
            vector to a label, cross-validated so the score reflects generalization rather than
            memorization. It is the same estimator whether the vector is 19 pose features (NB08) or
            N neurons (here). Only the input changes.

            **Limits of this analysis.**

            - **Correlation, not cause.** A decodable social signal in the population does not mean
              these cells *drive* social behavior. Decoding is a read-out; establishing causation
              requires perturbation experiments.
            - **Small n, and two clocks aligned by interpolation.** Six sessions per condition, and
              the calcium was interpolated onto the behavior clock. Resampling can only re-grid
              existing structure, but it also smooths, and any mis-registration between the two clocks
              would distort the alignment. Treat the condition trend as descriptive.
            - **Within-session, not cross-session.** Each session extracts a different set of neurons
              (for example 202 vs 218 cells, with no identity correspondence), so this decoder is
              cross-validated within a single session's population. Applying a decoder across sessions
              would require matching neurons between recordings (for example cell registration or a
              shared latent space), which this dataset does not support.
            """
        )
    })
    return


# ============================================================================ exercise
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 7. Exercise — decode, then count

        **Hypothesis.** A linear readout of the population vector decodes social state above chance;
        and social isolation reduces the number of cells carrying the social signal, so controls have
        the most.

        **What is provided.** The cell below gives you a `_pipeline(s)` helper (it runs
        `interp_resample → zscore → crop` and returns the population vectors `X` and the per-frame
        `is_social_sender` labels `y` for session `s`), the cross-validation setup, and the
        per-session counting loop. Two lines are left for you to complete; both are marked `# TODO`.

        **Part 1 — decode.** The cross-validated probabilities `_proba` are already computed for
        session 6. On the line marked `# TODO (1)`, score them with AUROC and store the result in
        `decoder_auc`:

        ```python
        # TODO (1): score the cross-validated probabilities with AUROC
        decoder_auc = ____        # replace ____ with:  float(_auc_score2(_y, _proba))
        ```

        **Part 2 — count, then read the trend.** The dictionary `_means` already holds the mean
        social-neuron count for each condition. On the line marked `# TODO (2)`, pick the condition
        with the **highest** mean:

        ```python
        # TODO (2): pick the condition with the highest mean count
        most_social_condition = ____   # replace ____ with:  max(_means, key=_means.get)
        ```

        **What you should see.** After filling both lines and running, the self-check below should
        pass: `decoder_auc` lands near **0.95** (a linear population readout well above the 0.5
        chance line), and `most_social_condition` is **"control"** (means: control 11.2, 7d 7.8,
        24hr 5.8). Note this last point is a descriptive direction, not a significance test — n = 6
        per condition. The two lines are shown completed in the cell so the notebook runs; to
        practice, replace each right-hand side with `____` and reconstruct it yourself, then compare.
        """
    )
    return


@app.cell
def _(beh, behavior_fps, cond_labels, ent, img, np, nu):
    # ------------------------------------------------------------------ YOUR CODE (edit the TODO lines)
    # Provided helper: run resample -> zscore -> crop for one session and return
    #   X = population vectors (T, n_neurons),  y = per-frame is_social_sender label (T,).
    def _pipeline(s, key="is_social_sender"):
        _iss = beh[s][key].astype(bool)
        _r = nu.zscore(nu.interp_resample(img[s], len(_iss), axis=0), axis=0)
        _e = int(ent["Int_Entry"].iloc[s])
        _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
        return _r[_t0:_t1], _iss[_t0:_t1]

    from sklearn.linear_model import LogisticRegression as _LogReg2
    from sklearn.preprocessing import StandardScaler as _Scaler2
    from sklearn.pipeline import make_pipeline as _mkpipe2
    from sklearn.model_selection import StratifiedKFold as _SKF2, cross_val_predict as _cvp2
    from sklearn.metrics import roc_auc_score as _auc_score2

    # Part 1 — decode session 6. Everything up to the cross-validated probabilities is done for you.
    _X, _yb = _pipeline(6)
    _y = _yb.astype(int)
    _clf = _mkpipe2(_Scaler2(), _LogReg2(max_iter=1000, class_weight="balanced"))
    _skf = _SKF2(5, shuffle=True, random_state=0)
    _proba = _cvp2(_clf, _X, _y, cv=_skf, method="predict_proba")[:, 1]
    # TODO (1): score the cross-validated probabilities with AUROC.
    #   Replace the right-hand side with:  float(_auc_score2(_y, _proba))
    decoder_auc = float(_auc_score2(_y, _proba))          # <-- fill in

    # Part 2 — count social neurons per session, then average by condition. The loop is done for you.
    _counts = {}
    for _s in range(len(img)):
        _Xs, _ys = _pipeline(_s)
        _n = int(nu.social_neuron_mask(_Xs, _ys, method="ratio").sum())
        _counts.setdefault(cond_labels[_s], []).append(_n)
    _means = {c: float(np.mean(v)) for c, v in _counts.items()}
    # TODO (2): pick the condition with the HIGHEST mean count.
    #   Replace the right-hand side with:  max(_means, key=_means.get)
    most_social_condition = max(_means, key=_means.get)   # <-- fill in
    # ---------------------------------------------------------------------------------------------
    return decoder_auc, most_social_condition


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            def pipeline(s, key="is_social_sender"):
                iss = beh[s][key].astype(bool)
                r = nu.zscore(nu.interp_resample(img[s], len(iss), axis=0), axis=0)
                e = int(ent["Int_Entry"].iloc[s]); t0, t1 = e, int(e + 3*60*25)
                return r[t0:t1], iss[t0:t1]

            # Part 1 — decode
            X, y = pipeline(6); y = y.astype(int)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced"))
            proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                      method="predict_proba")[:, 1]
            decoder_auc = roc_auc_score(y, proba)          # ~ 0.95

            # Part 2 — count, then read the direction
            counts = {}
            for s in range(len(img)):
                Xs, ys = pipeline(s)
                counts.setdefault(cond_labels[s], []).append(
                    int(nu.social_neuron_mask(Xs, ys, method="ratio").sum()))
            means = {c: np.mean(v) for c, v in counts.items()}
            most_social_condition = max(means, key=means.get)   # "control"
            ```

            **What you should find.** The population decoder lands around **AUROC ≈ 0.95**: a linear
            readout of roughly 218 neurons predicts social state well above chance. The social-neuron
            count is highest in **controls** (control 11.2, 24hr 5.8, 7d 7.8), so isolation lowers the
            count. With n = 6 per condition and this much spread, that is a descriptive direction, not
            a p-value. The exercise grades the honest reading, not the noise.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(decoder_auc, most_social_condition, mo):
    # Self-check with tolerance bands pinned from the real data:
    #   decoder_auc ~ 0.95 (session 6); graded as "above chance" -> > 0.70.
    #   most_social_condition -> "control" (means control 11.17 > 7d 7.83 > 24hr 5.83).
    _p1 = float(decoder_auc) > 0.70
    _p2 = str(most_social_condition) == "control"
    _ok = _p1 and _p2
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"✅ decoder AUROC = {decoder_auc:.3f} — a linear population readout predicts social state "
           "above the 0.5 chance line" if _p1 else
           f"❌ decoder AUROC = {decoder_auc:.3f} is at or near chance — check the pipeline order "
           "(resample → zscore → crop) and that you fed the full population vector")
    _m2 = ("✅ controls carry the most social neurons — isolation lowers the count (descriptive; "
           "n=6/condition)" if _p2 else
           f"❌ most_social_condition = {most_social_condition!r}; the pinned means are "
           "control 11.2 > 7d 7.8 > 24hr 5.8 → 'control'")
    _head = "PASS — decoder above chance, and the condition direction read correctly" if _ok else \
            "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band: AUROC &gt; 0.70 (chance = 0.50;
        pinned ≈ 0.95). Part 2 is graded on the direction of the trend, not a significance test.</span>
        </div>
        """
    )
    return


# ============================================================================ close
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Summary

        You applied the NB08 decoding procedure to neural data. The classifier, the stratified
        cross-validation, and the AUROC yardstick are the same; only the input changed, from 19 pose
        features to a population of real calcium neurons. On session 6 the decoder read social vs
        non-social at **AUROC ≈ 0.95**, which means the population, taken jointly, carries clear
        information about social state. Alongside that, the number of social neurons trended down with
        isolation (controls highest), but with six sessions per condition this is reported as a
        direction, not a result.

        **The main caveats, stated once more.** n is small; the calcium was interpolated onto the
        behavior clock; and because each session extracts its own neurons, this decoder generalizes
        within a session, not across sessions. Making a decoder transfer to a new session would
        require matching neurons across recordings or learning a shared latent space, and then
        decoding. That is a natural next step beyond this course.

        Across the two weeks you have worked with pose, with time, with a low-dimensional map, with a
        behavior classifier, and now with neural population activity, using one consistent toolbox.
        """
    )
    return


if __name__ == "__main__":
    app.run()

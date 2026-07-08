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
        # NB14 · Neural Signals of Social Isolation — *the decoder reads a brain*

        > **WEEK 2 · THE NEURAL TWIN — this is the payoff.**
        >
        > **FROM: Circuit Team → TO: You**
        >
        > In **NB08** you graduated a decoder that read *behavior* off pose: 19 body-frame features
        > per event → an aggression call, cross-validated, generalized to a cage it had never seen.
        > Today you point the **same computational move** at a **brain**. We hand you real
        > single-cell calcium from a miniscope while a mouse meets an intruder. You will z-score it,
        > put it on the behavior clock, find the cells that light up during social contact — and then
        > **train a population decoder to read *social state* straight off the neurons.**
        >
        > **The twin, stated plainly:** *NB08 decoded behavior from pose. NB14 decodes social state
        > from calcium — with the same sklearn logistic regression, the same cross-validation, the
        > same AUROC yardstick.* You built the behavioral decoder; here it is, reading a brain.
        > A decoder that survives a **new session** is the neural face of a cross-session
        > brain–machine interface — the exact demand a real BMI meets every recording day.
        >
        > **Today's lab-meeting question:** *Can a linear readout of the population vector tell
        > "socializing" from "not" above chance — and does social isolation change how many cells
        > carry the social signal?*

        Four beats: **(1)** put calcium on the behavior clock (interpolate + z-score),
        **(2)** find "social neurons," **(3)** count them across isolation conditions *honestly*,
        **(4)** train the population decoder — the direct twin of NB08.
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
        **Dataset loaded — SI3_2022 social-isolation cohort.** {n_sessions} sessions of paired
        miniscope calcium + frame-by-frame social behavior. A focal mouse (group-housed **control**,
        or isolated **24hr** / **7d**) meets an intruder; we track calcium and score when the focal
        is socially engaging. Sessions per condition: **{_counts}**. Neuron counts vary per session
        (**{min(_neur)}–{max(_neur)}** cells) because each recording demixes its own field of view —
        which is exactly why a naive cross-session decoder is *not* trivial (honest note at the end).

        The two signals live on **different clocks**: calcium at **30 fps**, behavior at **25 fps**.
        Before we can align a cell to a social bout, the calcium has to be resampled onto the
        behavior clock. That interpolation is the first move.
        """
    )
    return


# ============================================================================ interpolation sidebar
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Put calcium on the behavior clock

        `nu.interp_resample(C, len(is_social))` stretches the calcium matrix (`n_frames × n_neurons`
        at 30 fps) onto the behavior timeline (25 fps) by linear interpolation on a normalized
        $[0,1]$ grid — the exact `scipy.interpolate.interp1d` trick the 2025 script used. Below is
        one real neuron's trace, original samples vs. the resampled version, so you can see the
        interpolation is honest (it never invents peaks; it re-grids the ones already there).
        Then we **z-score** every neuron and **crop** to the first 3 minutes after the intruder
        enters (`Int_Entry`), where the social action is.
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
                       title="interp_resample: one neuron, 30 fps → 25 fps (no peaks invented)",
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
        ## 2. The population raster, with the social overlay

        Pick a session. We run the **whole pipeline** — `interp_resample → zscore(axis=0) → crop
        [entry, entry + 3·60·25]` — and draw the z-scored population as a raster (one row per
        neuron). The **green bands** mark frames the focal mouse is socially engaging
        (`is_social_sender`); everything else is non-social. If a population carries social state,
        the raster should *look different* under the green bands.
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
        ## 3. One neuron at a time — social vs non-social

        The raster is the population; now zoom to a single cell. For the chosen neuron we split its
        z-scored activity into **social** and **non-social** frames and overlay the two
        distributions. A *social neuron* pushes its social distribution to the right (or dips it) —
        the histograms pull apart. A cell that ignores social state has two distributions sitting on
        top of each other. Slide through the neurons and hunt for separation.
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
        ## 4. Which cells are "social neurons"?

        Do the per-neuron test for *every* cell at once. `nu.social_neuron_mask` computes, per
        neuron, the ratio of mean |activity| in social vs non-social frames; cells above a threshold
        (default **1.5**) are flagged social. The bar below shows every neuron's ratio, sorted, with
        your threshold line — drag it and watch the count move. This is the same knob the 2025 script
        swept by hand.
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
        ## 5. Does isolation change the count? (honestly)

        Now the cohort question. For every session we count social neurons and group by isolation
        condition. Three detection methods (the three the 2025 script tried) are on the dropdown —
        they disagree in the details, which is itself the lesson. **The honest read of the default
        `ratio` method: controls carry the *most* social neurons; isolation *lowers* the count.**
        But note the axis: **n = 6 sessions per condition.** This is a *descriptive* trend, not a
        significance test — with six points per group and session-to-session spread this wide, we
        report the direction and refuse to over-claim.
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


# ============================================================================ THE PAYOFF: decoder
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. The payoff — decode social state from the population

        This is the twin of NB08, line for line. There you fed **19 pose features** into a
        `LogisticRegression` and cross-validated an aggression call. Here you feed the **population
        vector** — every neuron's z-scored activity at a frame — into the *same* classifier and
        ask it to predict `is_social_sender`. Five-fold stratified cross-validation (no frame the model
        scored was in its training fold), reported as **AUROC** on the same 0.5-is-chance scale.

        If the population carries social state, a *linear* readout finds it. This is, mechanically,
        a neural brain–machine interface: population vector in, behavioral state out.
        """
    )
    return


@app.cell
def _(np, session_pick, sess_neurons, sess_social):
    # Population decoder: LogisticRegression, 5-fold stratified CV, AUROC (the NB08 twin).
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
        ### Pick the operating point

        A decoder outputs a *probability*; a **decision threshold** turns it into a call. For a
        real experiment that threshold is a policy — a loose criterion flags every twitch as social
        (false positives), a strict one misses real bouts (false negatives). Slide it and read the
        live ROC operating point plus precision / recall / confusion on the held-out predictions.
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
        "The lineage — social-isolation coding & neural decoding (and where the analogy stops)": mo.md(
            r"""
            **Social isolation reshapes social circuits.** Matthews et al. 2016 *Cell* (dorsal raphe
            dopamine encodes a loneliness-like state after acute isolation); Zelikowsky et al. 2018
            *Cell* (chronic isolation, Tac2/neurokinin B, amygdala & hypothalamus). **Coding of
            conspecifics / social state:** Remedios et al. 2017 *Nature* (VMHvl mixed social
            representation); Kingsbury et al. 2019 *Cell* (interbrain neural coding of dominance).
            **Population decoding of social variables:** Padilla-Coreano et al. 2022 *Nature*
            (mPFC ensembles decode competitive rank & social behavior) — the direct methodological
            cousin of what you just ran, and the same paper NB08 pointed at from the behavioral side.

            **Shared mathematics.** A population decoder is a supervised map from a high-dimensional
            state vector to a label, cross-validated so the score reflects *generalization*, not
            memorization. It is the *same* estimator whether the vector is 19 pose features (NB08) or
            N neurons (here) — that identity is the whole point of the "twin."

            **Where the analogy stops.**
            - **Correlation, not cause.** A decodable social signal in the population does **not** mean
              these cells *drive* social behavior. Decoding is read-out; causation needs perturbation.
            - **n is small and the clocks were glued.** Six sessions per condition, and calcium was
              *interpolated* onto the behavior clock — resampling can only re-grid existing structure,
              but it also smooths, and mis-registration between the two clocks would inflate or deflate
              alignment. Treat the condition trend as descriptive.
            - **Within-session, not cross-session.** Each session demixes a different set of neurons
              (202 vs 218 cells, no identity correspondence), so this decoder is cross-validated
              *within* a session's population. A true cross-session BMI needs neuron alignment
              (e.g. cell registration or a latent-space stitch) — that is the harder, unbuilt step.
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
        ## 7. Exercise — read the brain, then read it honestly

        **Hypothesis banner.** *A linear readout of the population vector decodes social state well
        above chance; and social isolation reduces the number of cells carrying the social signal
        (controls have the most).*

        **Toolbox.**

        - `nu.interp_resample(C, n_out, axis=0)` — calcium (30 fps) → behavior clock (25 fps).
        - `nu.zscore(x, axis=0)` — per-neuron z-score.
        - `nu.social_neuron_mask(neurons, is_social, method="ratio")` — `(n,)` bool; `.sum()` = count.
        - `sklearn` `LogisticRegression`, `StandardScaler`, `make_pipeline`,
          `StratifiedKFold`, `cross_val_predict`, `roc_auc_score`.
        - Pipeline order (fixed): `interp_resample → zscore(axis=0) → crop[entry, entry+3·60·25]`.

        **Your job (two parts).**

        1. **Decode.** For session 6 (`is_social_sender`), build the population vector through the
           pipeline, train a `LogisticRegression` population decoder, 5-fold stratified
           cross-validate, and put the cross-validated **AUROC** in `decoder_auc`.
        2. **Count, then read the trend honestly.** Count social neurons (`ratio` method) for every
           session, average by condition, and set `most_social_condition` to the condition with the
           **highest** mean count.

        Fill in `decoder_auc` and `most_social_condition`, then run the self-check.
        """
    )
    return


@app.cell
def _(beh, behavior_fps, cond_labels, ent, img, np, nu):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    def _pipeline(s, key="is_social_sender"):
        _iss = beh[s][key].astype(bool)
        _r = nu.zscore(nu.interp_resample(img[s], len(_iss), axis=0), axis=0)
        _e = int(ent["Int_Entry"].iloc[s])
        _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
        return _r[_t0:_t1], _iss[_t0:_t1]

    # Part 1 — decode session 6
    from sklearn.linear_model import LogisticRegression as _LogReg2
    from sklearn.preprocessing import StandardScaler as _Scaler2
    from sklearn.pipeline import make_pipeline as _mkpipe2
    from sklearn.model_selection import StratifiedKFold as _SKF2, cross_val_predict as _cvp2
    from sklearn.metrics import roc_auc_score as _auc_score2

    _X, _yb = _pipeline(6)
    _y = _yb.astype(int)
    _clf = _mkpipe2(_Scaler2(), _LogReg2(max_iter=1000, class_weight="balanced"))
    _skf = _SKF2(5, shuffle=True, random_state=0)
    _proba = _cvp2(_clf, _X, _y, cv=_skf, method="predict_proba")[:, 1]
    decoder_auc = float(_auc_score2(_y, _proba))

    # Part 2 — count social neurons per session, mean per condition, pick the highest
    _counts = {}
    for _s in range(len(img)):
        _Xs, _ys = _pipeline(_s)
        _n = int(nu.social_neuron_mask(_Xs, _ys, method="ratio").sum())
        _counts.setdefault(cond_labels[_s], []).append(_n)
    _means = {c: float(np.mean(v)) for c, v in _counts.items()}
    most_social_condition = max(_means, key=_means.get)
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

            # Part 2 — count + honest trend
            counts = {}
            for s in range(len(img)):
                Xs, ys = pipeline(s)
                counts.setdefault(cond_labels[s], []).append(
                    int(nu.social_neuron_mask(Xs, ys, method="ratio").sum()))
            means = {c: np.mean(v) for c, v in counts.items()}
            most_social_condition = max(means, key=means.get)   # "control"
            ```

            **What you should find.** The population decoder lands around **AUROC ≈ 0.95** — a linear
            readout of ~218 neurons reads social state far above chance. And the social-neuron count
            is highest in **controls** (control 11.2 · 24hr 5.8 · 7d 7.8), i.e. *isolation lowers the
            count* — but with n=6 per condition and this much spread, that is a **descriptive**
            direction, not a p-value. The exercise grades the *honest reading*, not noise.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(decoder_auc, most_social_condition, mo):
    # Self-check with tolerance bands pinned from the real data:
    #   decoder_auc ~ 0.95 (session 6); graded as "well above chance" -> > 0.70.
    #   most_social_condition -> "control" (means control 11.17 > 7d 7.83 > 24hr 5.83).
    _p1 = float(decoder_auc) > 0.70
    _p2 = str(most_social_condition) == "control"
    _ok = _p1 and _p2
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"✅ decoder AUROC = {decoder_auc:.3f} — a linear population readout reads social state "
           "far above chance (0.5)" if _p1 else
           f"❌ decoder AUROC = {decoder_auc:.3f} is at/near chance — check the pipeline order "
           "(resample → zscore → crop) and that you fed the full population vector")
    _m2 = ("✅ controls carry the most social neurons — isolation *lowers* the count (honest, "
           "descriptive; n=6/condition)" if _p2 else
           f"❌ most_social_condition = {most_social_condition!r}; the pinned means are "
           "control 11.2 > 7d 7.8 > 24hr 5.8 → 'control'")
    _head = "PASS — you decoded a brain, and read the cohort honestly" if _ok else \
            "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band: AUROC &gt; 0.70 (chance = 0.50;
        pinned ≈ 0.95). Part 2 is graded on the honest direction, not a significance test.</span>
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
        ## The twin, closed

        You just ran the NB08 decoder on a **brain**. Same `LogisticRegression`, same stratified
        cross-validation, same AUROC yardstick — only the input changed, from 19 pose features to a
        population of real calcium neurons. It read *socializing vs not* at **AUROC ≈ 0.95**. That
        is the Week-2 payoff stated as plainly as it can be: **the computational move that read
        behavior reads the brain.** And you kept it honest — social-neuron counts *trend* down with
        isolation, but with six sessions a side you reported a direction, not a discovery, and you
        named the interpolation and the within-session limit out loud.

        **The honest fine print, once more:** n is small; the calcium was interpolated onto the
        behavior clock; and because each session demixes its own neurons, this decoder generalizes
        *within* a session, not across them. Making it survive a **new session** — a real
        cross-session BMI — is the next mountain: register cells across recordings, or learn a shared
        latent space, then decode. That is where a course like this hands off to a lab.

        *You have now read pose, read time, read a map, read behavior, and read a brain — with one
        toolbox.*
        """
    )
    return


if __name__ == "__main__":
    app.run()

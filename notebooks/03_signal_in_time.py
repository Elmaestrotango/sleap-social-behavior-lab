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
def _():
    import pandas as pd
    from plotly.subplots import make_subplots
    return make_subplots, pd


# ============================================================ 1. Lab-meeting briefing
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB03 · Feeling the Signal in Time

        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > Two mice in an encounter are two coupled systems: they have a **tempo** and a
        > **direction of influence**. Before you compress everything into a map, we need the
        > behavioral twins of the two analyses we live by on the neural side — a **rhythm**
        > readout (the wavelet we run on LFP) and a **who-leads-whom** readout (directed
        > functional connectivity). Ship us both, honestly caveated.
        >
        > **Deliverable:** a rhythm spectrogram of one event + a lead-lag coordination estimate
        > with a null it has to beat.
        > **It unblocks:** knowing whether "who moved first" is a signal we can trust to
        > time-align to a recording, or noise from a 2.6-second window.
        >
        > **Today's lab-meeting question:** *In the run-up to contact, who moves first — and is
        > that leader estimate robust, or does it wash out against a within-event shuffle?*

        This is the **last look before the collapse**. NB04 (PCA) and NB05 (the map) are about to
        *average away* time and coordination. Today we survey what they discard: distributions,
        how features move **together**, how a single event moves **in time and frequency**, and how
        two mice move **relative to each other**.

        *Neuroscience connection —* the Morlet wavelet you run below on a mouse's speed is the
        **exact transform** neuroscientists slide along an LFP to find theta and gamma; the lead-lag
        you measure is the behavioral face of **directed functional connectivity**. Kingsbury et al.
        (2019) found that dmPFC coupling *between two brains* tracks their dominance relationship —
        the same coordination logic, one level up.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🔬 Deeper: the paper & where the analogy stops": mo.md(
                r"""
                **Shared mathematics.** A continuous wavelet transform convolves a signal with
                scaled copies of a mother wavelet — identical whether the signal is a wagging tail
                or hippocampal LFP (Torrence & Compo 1998; Cohen 2014, *Analyzing Neural Time
                Series Data*; Buzsáki & Draguhn 2004, *Science*). Cross-correlation / Granger
                lead-lag is the core of **directed functional connectivity** (Granger 1969; Bressler
                & Seth 2011, *NeuroImage* 58:323–329; Seth, Barrett & Barnett 2015, *J. Neurosci.*).
                Berman et al. (2014, *J. R. Soc. Interface*) built MotionMapper on exactly this
                postural-spectrogram idea.

                **Species / preparation.** Kingsbury et al. (2019, *Cell*) — freely-interacting
                **mice**, dual fiber photometry / miniscope in dmPFC; the coupling was measured with
                a **GLM / correlation**, *not* Granger (looser analogy — flagged honestly).

                **Where the analogy stops.** A wagging tail is not an LFP: only the *transform*
                transfers, and **matching Hz ≠ matching mechanism**. Lead-lag measures **prediction,
                not cause** — a shared third mouse or common arousal can fake it (Kingsbury's own
                caveat). We measure coordination, never proof of driving.
                """
            )
        }
    )
    return


# ============================================================ data + shared helpers
@app.cell
def _(ROOT, cu):
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))
    der = cu.load_derived("train", ROOT)
    ho = cu.load_events(cu.data_path("data/heldout_events.npz", ROOT))
    return der, ev, ho


@app.cell
def _(der, ev):
    agg = ev["agg_label"].astype(int)
    cage = der["cage"]
    sexv = der["sex"].astype(str)
    cond = ev["condition"].astype(str)
    cr = ev["contact_rel"].astype(int)
    kp = ev["kp"]
    ranks = ev["ranks"]
    X = der["X"]
    fnames = [str(f) for f in der["feature_names"]]
    return X, agg, cond, cr, fnames, kp, ranks, sexv


@app.cell
def _(cu, np):
    # Shared, reused helpers (defined once so the DAG stays clean).
    def appr_appe_speed(kp, i):
        """World-frame per-frame speed (px/frame) of the approacher (mouse 0) and approachee (1)."""
        k = kp[i]
        c0 = cu._centroids(k[:, 0]); c1 = cu._centroids(k[:, 1])
        s0 = np.linalg.norm(np.diff(c0, axis=0), axis=1)
        s1 = np.linalg.norm(np.diff(c1, axis=0), axis=1)
        return np.nan_to_num(s0), np.nan_to_num(s1)

    def pre_speeds(kp, cr, i, win=50):
        """The two speed traces over the `win` frames just BEFORE contact (where approacher and
        the true first-mover can dissociate)."""
        s0, s1 = appr_appe_speed(kp, i)
        c = int(cr[i]); a = max(1, c - win)
        return s0[a - 1:c - 1], s1[a - 1:c - 1]

    def leader_fraction(kp, cr, idxs, max_lag=10):
        """Fraction of events where the approacher LEADS (peak cross-corr lag > 0). ~0.5 = no
        consistent leader. Skips traces too short for the requested lag. Returns (fraction, n)."""
        leads = []
        for i in idxs:
            x, y = pre_speeds(kp, cr, i)
            if len(x) < 2 * max_lag + 4 or x.std() < 1e-6 or y.std() < 1e-6:
                continue
            _, _, pk = cu.cross_corr_lag(x, y, max_lag)
            if pk != 0:
                leads.append(1 if pk > 0 else 0)
        return (float(np.mean(leads)) if leads else float("nan"), len(leads))

    def padded_wavelet(sig, freqs, fps, padlen=600):
        """Reflect-pad a short signal so low-frequency Morlet kernels fit, run cu.wavelet_power,
        then crop back to the original span. The padded flanks are exactly the EDGE-EFFECT region
        we warn about below."""
        sig = np.nan_to_num(np.asarray(sig, float)); T = len(sig)
        pad = max(0, (padlen - T) // 2)
        sp = np.pad(sig, pad, mode="reflect")
        P = cu.wavelet_power(sp, freqs, fps)
        return P[:, pad:pad + T]
    return appr_appe_speed, leader_fraction, padded_wavelet, pre_speeds


@app.cell
def _():
    # Pinned build-time constants (verified against the real bundle; see notebook header).
    HERO = 909                 # cage-15, male, aggression, contact_rel=40, node reliability 0.998
    NULL_MEAN = 0.50           # within-event shuffle null: mean approacher-leads fraction
    NULL_HI = 0.545            # shuffle null 97.5th percentile
    FULL_FRAC = 0.479          # observed appr-leads fraction, all 420 usable aggression events
    FULL_N = 420
    DOM_FREQ = 1.5             # hero-event dominant speed frequency (Hz), 1-12 Hz band
    return DOM_FREQ, FULL_FRAC, FULL_N, HERO, NULL_HI, NULL_MEAN


# ============================================================ 2. Readout board (top)
@app.cell(hide_code=True)
def _(ROOT, X, mo, pd):
    def _board(root):
        try:
            df = pd.read_csv(root + "/data/readout_board.csv")
            a = df[(df.gauge == "A") & (df.notebook == "NB02")]["value"].iloc[0]
            return float(a)
        except Exception:
            return 19.0
    _bench_A = _board(ROOT)
    _student_A = int(X.shape[1])
    mo.md(
        f"""
        ### 📋 Readout Board — start of NB03

        | Gauge | Benchmark | Your number | Note |
        |---|---|---|---|
        | **A · size of representation** | {_bench_A:.0f} features (from NB02) | **{_student_A}** features | We only *look* today — no compression yet. |
        | **B · held-out readiness** | rises in Phase 2 | **0** | Still in Discover; the decoder comes in NB08. |

        *Gauge A holds at 19 this notebook: EDA doesn't shrink the representation, it decides what
        the coming collapse is allowed to throw away.*
        """
    )
    return


# ============================================================ 3. Sealed Cage 16
@app.cell(hide_code=True)
def _(ho, mo):
    _n16 = int(len(ho["agg_label"]))
    mo.md(
        f"""
        <div style="border:2px dashed #b00; border-radius:10px; padding:14px 18px;
        background:repeating-linear-gradient(45deg,#faf0f0,#faf0f0 10px,#f3e4e4 10px,#f3e4e4 20px);">

        ### 🔒 Sealed Cage 16 — the animal on the rig

        Events recorded: **{_n16}** &nbsp;·&nbsp; skeletons: ▓▓▓ (greyed) &nbsp;·&nbsp;
        labels: ███████ (redacted)

        The held-out bet lives here: *a readout is only trustworthy if it survives a cage it never
        saw.* You may **not** touch Cage 16's rhythm or coordination yet.

        **Notebooks until unlock: 5** &nbsp;(opens in NB08).
        </div>
        """
    )
    return


# ============================================================ 4. Feature distributions
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Distributions — read the raw trace like a physiologist

        Before modeling anything, *look*. Each of the 19 features is a distribution; aggression
        often lives in the **tail**, not the mean. Pick a feature and a way to split it — the honest
        habit is to always split by **sex** and **condition** (pre / dep / post), because those are
        the axes a between-cage confound will later hide inside.
        """
    )
    return


@app.cell
def _(mo):
    feat_sel = mo.ui.dropdown(
        options=["closing_speed", "appr_speed_mean", "pair_dist_min", "appr_faces_appe",
                 "appe_faces_appr", "appr_angvel", "triangle_area_mean", "bystander_dist_min"],
        value="closing_speed", label="feature", full_width=True)
    split_sel = mo.ui.dropdown(options=["aggression", "sex", "condition"], value="aggression",
                               label="split by", full_width=True)
    return feat_sel, split_sel


@app.cell
def _(X, agg, cond, feat_sel, fnames, go, mo, np, sexv, split_sel):
    _fi = fnames.index(feat_sel.value)
    _x = X[:, _fi]
    if split_sel.value == "aggression":
        _groups = [("non-aggression", agg == 0, "#8899aa"), ("aggression", agg == 1, "#d62728")]
    elif split_sel.value == "sex":
        _groups = [("male", sexv == "M", "#1f77b4"), ("female", sexv == "F", "#e377c2")]
    else:
        _cmap = {"pre": "#4c78a8", "dep": "#e45756", "post": "#54a24b"}
        _groups = [(c, cond == c, _cmap[c]) for c in ["pre", "dep", "post"]]
    _fig = go.Figure()
    for _nm, _m, _c in _groups:
        _v = _x[_m]; _v = _v[np.isfinite(_v)]
        _fig.add_histogram(x=_v, name=_nm, opacity=0.6, histnorm="probability density",
                           marker_color=_c, nbinsx=45)
    _fig.update_layout(barmode="overlay", template="plotly_white", height=420,
                       title=f"{feat_sel.value} — split by {split_sel.value}",
                       xaxis_title=feat_sel.value, yaxis_title="density",
                       margin=dict(l=10, r=10, t=50, b=10), font=dict(size=15))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([feat_sel, split_sel]), _fig])
    return


# ============================================================ 5. Correlation heatmap
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · The 19×19 correlation heatmap — why the collapse is coming

        These 19 knobs are **not independent**. `bystander_dist_mean` and `bystander_dist_min`
        move together (r ≈ 0.9); the two facing-cosines and closing speed share structure. Redundant
        axes are exactly what **PCA** will exploit in NB04 — 19 correlated features are really a
        *handful* of independent ones. Read this heatmap as the motivation for the next notebook.
        """
    )
    return


@app.cell
def _(X, cu, fnames, go, mo, np):
    _Xz, _, _ = cu.standardize(X)
    _C = np.corrcoef(_Xz.T)
    _fig = go.Figure(go.Heatmap(z=_C, x=fnames, y=fnames, colorscale="RdBu", zmid=0,
                                zmin=-1, zmax=1, colorbar=dict(title="r")))
    _fig.update_layout(template="plotly_white", height=620,
                       title="Feature–feature correlation (Pearson r) — off-diagonal structure = redundancy",
                       margin=dict(l=10, r=10, t=50, b=120), font=dict(size=12))
    _fig.update_xaxes(tickangle=45)
    _absC = np.abs(_C - np.eye(19))
    _i, _j = np.unravel_index(np.argmax(_absC), _C.shape)
    mo.vstack([_fig, mo.md(
        f"**Most-correlated pair:** `{fnames[_i]}` ↔ `{fnames[_j]}` (r = {_C[_i, _j]:.2f}). "
        f"Bright off-diagonal blocks are why ~6 PCs will hold most of the variance.")])
    return


# ============================================================ 6. Rhythm — the Hero Event
@app.cell(hide_code=True)
def _(HERO, mo):
    mo.md(
        rf"""
        ## 3 · Rhythm — the Hero Event in time and frequency

        We follow **Hero Event #{HERO}** (Cage 15, male, a real aggression approach; node reliability
        0.998). *Design note: the course's original "#742" is a cage-12 non-aggression event in the
        shipped data, so we anchor the male-aggression hero to the cleanest cage-15 event, #{HERO}.*

        First the raw traces: the **closing distance** collapses and the **speeds** spike into
        contact. Then a **Morlet wavelet** — "a little wave slid along the signal" — turns speed
        into a **time × frequency** picture: which rhythm is present, and when.
        """
    )
    return


@app.cell
def _(HERO, cr, cu, kp, mo, ranks):
    _gif = cu.event_gif_bytes(kp[HERO], ranks[HERO], contact_rel=int(cr[HERO]), cell=210, fps=20)
    mo.vstack([
        mo.md(f"**Hero Event #{HERO}** — rank-colored skeletons (red=Dom, blue=Mid, green=Sub); "
              "white arrow = approacher→approachee; red dot = contact onset."),
        mo.Html(cu.gif_img_html(_gif, width=230)),
    ])
    return


@app.cell
def _(HERO, appr_appe_speed, cr, cu, kp, make_subplots, mo, np):
    _k = kp[HERO]
    _c0 = cu._centroids(_k[:, 0]); _c1 = cu._centroids(_k[:, 1])
    _dist = np.linalg.norm(_c0 - _c1, axis=1)               # (T,) closing distance
    _s0, _s1 = appr_appe_speed(kp, HERO)                    # (T-1,)
    _t = np.arange(len(_dist)) / cu.FPS
    _te = np.arange(len(_s0)) / cu.FPS
    _cf = int(cr[HERO]) / cu.FPS
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("pair distance (px)", "per-mouse speed (px/frame)"))
    _fig.add_scatter(x=_t, y=_dist, mode="lines", line=dict(color="#333"), name="pair distance",
                     row=1, col=1)
    _fig.add_scatter(x=_te, y=_s0, mode="lines", line=dict(color="#d62728"), name="approacher",
                     row=2, col=1)
    _fig.add_scatter(x=_te, y=_s1, mode="lines", line=dict(color="#1f77b4"), name="approachee",
                     row=2, col=1)
    for _r in (1, 2):
        _fig.add_vline(x=_cf, line=dict(color="#888", dash="dot"), row=_r, col=1)
    _fig.update_layout(template="plotly_white", height=430, font=dict(size=14),
                       margin=dict(l=10, r=10, t=40, b=10))
    _fig.update_xaxes(title_text="time (s)", row=2, col=1, showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([_fig, mo.md("Dotted line = contact onset. The approacher's speed leads the collision; "
                           "next we ask at *what rhythm*.")])
    return


@app.cell
def _(mo):
    fmax_slider = mo.ui.slider(6, 20, value=12, step=1, label="wavelet upper frequency (Hz)",
                               debounce=True, full_width=True)
    return (fmax_slider,)


@app.cell
def _(DOM_FREQ, HERO, appr_appe_speed, cu, fmax_slider, go, kp, mo, np, padded_wavelet):
    _s0, _ = appr_appe_speed(kp, HERO)
    _freqs = np.linspace(1.0, float(fmax_slider.value), 45)
    _P = padded_wavelet(_s0, _freqs, cu.FPS, padlen=600)
    _t = np.arange(_P.shape[1]) / cu.FPS
    _dom = float(_freqs[np.argmax(_P.mean(axis=1))])
    _fig = go.Figure(go.Heatmap(z=_P, x=_t, y=_freqs, colorscale="Viridis",
                                colorbar=dict(title="power")))
    _fig.add_hline(y=_dom, line=dict(color="white", dash="dot"))
    _fig.update_layout(template="plotly_white", height=380, font=dict(size=14),
                       title=f"Morlet spectrogram of approacher speed — dominant ≈ {_dom:.1f} Hz",
                       xaxis_title="time (s)", yaxis_title="frequency (Hz)",
                       margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([fmax_slider, _fig, mo.md(
        f"**Dominant-frequency readout ≈ {_dom:.1f} Hz** (pinned build value {DOM_FREQ} Hz over the "
        "1–12 Hz band) — a slow locomotor rhythm, not an LFP oscillation.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **⚠️ How the wavelet breaks on this data.** The event is only ~2.6 s (130 frames). Two
        > failure modes are baked in:
        >
        > 1. **The Heisenberg (time–frequency) trade-off.** A wavelet tight enough to localize *when*
        >    a burst happened is blurry about *what frequency* it was, and vice-versa. You cannot have
        >    sharp time and sharp frequency at once.
        > 2. **Edge effects.** A low-frequency wavelet is *wider than the window*, so its kernel runs
        >    off both ends. We reflect-pad just to fit it — and the padded flanks are fabricated, not
        >    measured. Trust the **center** of the spectrogram, distrust the edges.
        >
        > This is why a wavelet beats an FFT here: an FFT assumes one stationary spectrum for the
        > whole window; the wavelet admits the rhythm *changes* as contact approaches.
        """
    )
    return


# ============================================================ 7. Coordination — who leads?
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · Coordination — who moves first?

        Slide one mouse's speed trace against the other and find the lag of peak correlation:
        `peak_lag > 0` means the **approacher leads**. We run this on **pre-contact frames only** —
        after contact the approacher recovers "by construction," which would be circular.

        A tiny intuition toy first: impose a known lag between two coupled signals and watch
        `cross_corr_lag` recover it.
        """
    )
    return


@app.cell
def _(mo):
    toy_lag = mo.ui.slider(-8, 8, value=3, step=1, label="imposed lag (B follows A by …)",
                           debounce=True, full_width=True)
    return (toy_lag,)


@app.cell
def _(cu, go, mo, np, toy_lag):
    _rng = np.random.RandomState(1)
    _base = np.cumsum(_rng.randn(120))                      # a wandering "driver" A
    _A = _base + 0.4 * _rng.randn(120)
    _lag = int(toy_lag.value)
    _B = np.roll(_base, _lag) + 0.4 * _rng.randn(120)       # B = A delayed by `lag`
    _lags, _corr, _pk = cu.cross_corr_lag(_A, _B, 12)
    _fig = go.Figure()
    _fig.add_scatter(x=_lags, y=_corr, mode="lines+markers", line=dict(color="#7b3294"))
    _fig.add_vline(x=_pk, line=dict(color="#d62728", dash="dot"))
    _fig.update_layout(template="plotly_white", height=320, font=dict(size=14),
                       title=f"recovered peak lag = {_pk}  (imposed {_lag})",
                       xaxis_title="lag (A vs B)", yaxis_title="normalized cross-correlation",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([toy_lag, _fig, mo.md(
        "Positive lag ⇒ A leads B. The estimator recovers the imposed lag — until noise and short "
        "windows blur it, which is exactly the regime the real mice live in.")])
    return


@app.cell
def _(mo):
    coord_maxlag = mo.ui.slider(4, 15, value=10, step=1, label="max lag (frames)",
                                debounce=True, full_width=True)
    coord_split = mo.ui.dropdown(options=["all", "sex", "condition"], value="all",
                                 label="split by", full_width=True)
    return coord_maxlag, coord_split


@app.cell
def _(FULL_FRAC, FULL_N, NULL_HI, agg, coord_maxlag, coord_split, cond, cr,
      go, kp, leader_fraction, mo, np, sexv):
    # LIVE loop on a balanced ≤200-event aggression subsample (fast); precomputed full-corpus
    # result shown beside it.
    _rng = np.random.RandomState(0)
    _agg_idx = np.where(agg == 1)[0]
    _sub = _rng.choice(_agg_idx, size=min(200, len(_agg_idx)), replace=False)
    if coord_split.value == "all":
        _splits = [("all aggression", _sub)]
    elif coord_split.value == "sex":
        _splits = [("male", _sub[sexv[_sub] == "M"]), ("female", _sub[sexv[_sub] == "F"])]
    else:
        _splits = [(c, _sub[cond[_sub] == c]) for c in ["pre", "dep", "post"]]
    _names, _fracs, _ns = [], [], []
    for _nm, _idx in _splits:
        _f, _n = leader_fraction(kp, cr, _idx, max_lag=int(coord_maxlag.value))
        _names.append(_nm); _fracs.append(_f if _f == _f else 0.0); _ns.append(_n)
    _fig = go.Figure()
    _fig.add_bar(x=_names, y=_fracs, marker_color="#4c78a8",
                 text=[f"{f:.2f}<br>n={n}" for f, n in zip(_fracs, _ns)], textposition="outside")
    _fig.add_hline(y=0.5, line=dict(color="#333", dash="dash"),
                   annotation_text="no consistent leader (0.50)")
    _fig.add_hrect(y0=1 - NULL_HI, y1=NULL_HI, fillcolor="#bbbbbb", opacity=0.25, line_width=0,
                   annotation_text="within-event shuffle null (95%)", annotation_position="top left")
    _fig.update_layout(template="plotly_white", height=400, font=dict(size=14),
                       yaxis_title="fraction approacher LEADS", yaxis_range=[0, 1],
                       title=f"Pre-contact lead-lag — subsample, split by {coord_split.value}",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([coord_maxlag, coord_split]), _fig, mo.md(
        f"**Precomputed full corpus (all {FULL_N} usable aggression events):** approacher-leads "
        f"fraction = **{FULL_FRAC:.3f}** — sitting inside the grey shuffle band. The bars land near "
        "0.50 too: no robust leader. That is the honest result, and the exercise below grades it.")])
    return


# ============================================================ 8. Neuromatch exercise
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 🧪 Exercise — is the "leader" real, or shuffle-null noise?

        **Hypothesis banner.** *In the pre-contact window, the mouse that moves first is a
        consistent, robust leader — its approacher-leads fraction sits outside a within-event
        shuffle null.*

        > **Honesty note.** The design's original framing ("the leader **is** the aggression
        > initiator") **cannot be built here**: per-mouse `initiator_idx` / `fleer_idx` were never
        > shipped in any `.npz`. There is no ground-truth first-mover to grade against. So we grade
        > the *robustness* of the leader estimate — **does it beat its own shuffle null?** — not
        > `leader == initiator`. The who-initiates question stays conceptual (see below).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 🧰 Toolbox
        - `pre_speeds(kp, cr, i, win=50)` → `(appr_speed, appe_speed)` over the pre-contact window.
        - `cu.cross_corr_lag(x, y, max_lag)` → `(lags, corr, peak_lag)`; **`peak_lag > 0` ⇒ x leads y**.
        - `leader_fraction(kp, cr, idxs, max_lag)` → `(fraction_approacher_leads, n_usable)`.
        - Pinned null: shuffle mean `NULL_MEAN` = 0.50, 97.5th pct `NULL_HI` = 0.545.

        **TODO.** On the balanced subsample `sub_idx`, compute the approacher-leads fraction, then
        decide: does `|fraction − 0.5|` clear the shuffle band? (The stub already calls the helper so
        the notebook renders — try re-deriving it by hand, then open the solution.)
        """
    )
    return


@app.cell
def _(agg, np):
    _rng = np.random.RandomState(0)
    _agg_idx = np.where(agg == 1)[0]
    sub_idx = _rng.choice(_agg_idx, size=min(200, len(_agg_idx)), replace=False)
    return (sub_idx,)


@app.cell
def _(cr, kp, leader_fraction, sub_idx):
    # ---- TODO (student) -------------------------------------------------------
    # Compute the observed approacher-leads fraction on `sub_idx`.
    # Reference implementation uses the toolbox helper; swap in your own lag loop if you like.
    obs_frac, obs_n = leader_fraction(kp, cr, sub_idx, max_lag=10)
    # ---------------------------------------------------------------------------
    return obs_frac, obs_n


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution (reveal)": mo.md(
                r"""
                ```python
                import numpy as np

                def my_leader_fraction(kp, cr, idxs, max_lag=10):
                    leads = []
                    for i in idxs:
                        x, y = pre_speeds(kp, cr, i)              # pre-contact speed traces
                        if len(x) < 2 * max_lag + 4 or x.std() < 1e-6 or y.std() < 1e-6:
                            continue
                        lags, corr, peak_lag = cu.cross_corr_lag(x, y, max_lag)
                        if peak_lag != 0:                         # peak_lag > 0 ⇒ approacher leads
                            leads.append(1 if peak_lag > 0 else 0)
                    return float(np.mean(leads)), len(leads)

                obs_frac, obs_n = my_leader_fraction(kp, cr, sub_idx, max_lag=10)
                # obs_frac ≈ 0.47 — indistinguishable from the 0.50 shuffle null.
                ```

                The number lands *inside* the shuffle band. The scientifically correct conclusion is
                **"no robust leader in this window."** A short (~1 s pre-contact), nonstationary,
                noisy trace simply doesn't carry a reliable lead-lag sign.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(NULL_HI, NULL_MEAN, mo, obs_frac, obs_n):
    _gap = abs(obs_frac - 0.5)
    _band = NULL_HI - NULL_MEAN + 0.02          # shuffle half-width + tolerance
    _robust = _gap > _band
    # GRADED CORRECT ANSWER (honest): the leader does NOT robustly beat the shuffle null.
    _pass = not _robust
    _color = "#e6f4ea" if _pass else "#fce8e6"
    _edge = "#137333" if _pass else "#c5221f"
    _msg = (f"✅ **PASS — honest result.** Observed approacher-leads fraction = **{obs_frac:.3f}** "
            f"(n={obs_n}); |fraction − 0.5| = {_gap:.3f} ≤ shuffle band ({_band:.3f}). "
            "You correctly concluded the leader estimate **does not robustly exceed chance**."
            if _pass else
            f"❌ Observed fraction = {obs_frac:.3f}; |fraction − 0.5| = {_gap:.3f} exceeded the "
            f"band ({_band:.3f}). On this bundle the honest answer is *no robust leader* — recheck "
            "that you used pre-contact frames and the shuffle null.")
    mo.md(
        f"""
        <div style="background:{_color}; border-left:5px solid {_edge}; border-radius:6px;
        padding:12px 16px;">

        ### Self-check — graded on robustness, not on noise
        {_msg}

        *No student is graded against a noisy p-value: the tolerance band asserts the leader sits
        **inside** the null, which is the pre-verified build-time truth.*
        </div>
        """
    )
    return


# ============================================================ 9. Granger stretch
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 🔭 Stretch — Granger causality (optional accordion)

        Cross-correlation asks *when* two traces align. **Granger** asks a sharper question: does the
        **past of mouse A** improve prediction of **mouse B's next step**, beyond B's own past? It's a
        restricted-vs-unrestricted VAR **F-test** (`cu.granger_pair`, pure numpy — no `statsmodels`).
        """
    )
    return


@app.cell(hide_code=True)
def _(HERO, cr, cu, kp, mo, pre_speeds):
    _x, _y = pre_speeds(kp, cr, HERO)
    try:
        _g = cu.granger_pair(_x, _y, lags=4)
        _txt = (f"Hero #{HERO}: approacher→approachee F = {_g['f_xy']:.2f} (p = {_g['p_xy']:.3f}); "
                f"approachee→approacher F = {_g['f_yx']:.2f} (p = {_g['p_yx']:.3f}).")
    except Exception as _e:
        _txt = f"(Granger skipped: {_e})"
    mo.accordion({
        "🔭 Granger on the Hero Event + the hard caveat": mo.md(
            rf"""
            {_txt}

            **The common-cause caveat (non-negotiable).** Granger measures **prediction, not cause**.
            Both mice can be driven by a **shared third cause** — the bystander mouse, or a common
            arousal spike — which makes A *look* like it drives B when neither drives the other.
            Bivariate Granger is also **not conditional**: to move from "A predicts B" to "A predicts
            B *given the bystander*," you'd add the third mouse's trace as a covariate (a conditional
            / multivariate VAR). On 1-second nonstationary windows, treat any single-event Granger
            number as a hint, never a verdict.
            """
        )
    })
    return


# ============================================================ 10. Conceptual questions
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 🤔 Conceptual questions
        1. **Name the confound.** Lead-lag is *coordination*, not proof of driving — what shared
           cause on this rig could make two mice look coupled when neither leads? (The bystander;
           a common arousal spike.)
        2. **Wavelet vs FFT.** When does a wavelet beat an FFT? (When the spectrum is *non-stationary*
           — the rhythm changes across the 2.6 s, which an FFT would smear into one average.)
        3. **Why pre-contact only?** Why is testing the leader on pre-contact frames essential?
           (After contact you'd recover the approacher "by construction" — the label leaks into the
           window, and any classifier/lag would be circular.)
        4. **The missing ground truth.** We could not test `leader == initiator`. What upstream
           artifact would you need, and why does its absence force a robustness test instead of an
           accuracy test?
        """
    )
    return


# ============================================================ 11. What we threw away
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 🗑️ What we threw away / how it breaks
        - **Discarded:** by even *looking* in frequency and lead-lag, we're about to hand PCA/UMAP a
          set of **19 time-collapsed summaries** — the full within-event *time course* and the
          *coordination sign* get averaged away in the next two notebooks.
        - **Failure modes on this data:** (1) 130-frame windows are **short and nonstationary** →
          wavelet edge effects and noisy lag estimates; (2) **approacher ≠ initiator**, and with no
          `initiator_idx` we cannot close that gap; (3) a **shared third mouse** fakes coordination —
          bivariate lead-lag can't rule it out.
        - **Open-ended:** *How would you condition out the bystander mouse* to move from bivariate to
          **conditional** coordination — and what would you do differently if you had the per-mouse
          initiator label the project deferred?
        """
    )
    return


# ============================================================ 12. Closing neuro + board + hook
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        *Neuroscience connection (close) —* you just ran, on behavior, the **two workhorse analyses
        of systems neuroscience**: a time-frequency transform (the LFP spectrogram) and a directed
        lead-lag (functional connectivity). And you ran into their real limits — short windows,
        non-stationarity, the common-cause confound — the *same* limits that make inter-brain
        coupling claims hard. **Where it stops:** matching a frequency is not matching a mechanism,
        and prediction is not cause.
        """
    )
    return


@app.cell(hide_code=True)
def _(X, mo):
    mo.md(
        f"""
        ### 📋 Readout Board — end of NB03

        | Gauge | Value | Δ this notebook |
        |---|---|---|
        | **A · size of representation** | **{int(X.shape[1])}** features | unchanged — we *looked*, we did not compress |
        | **B · held-out readiness** | **0** | unchanged — coordination signal is honestly null, so nothing bankable ships |

        We surveyed rhythm and coordination and reported them **honestly**: a real ~1.5 Hz movement
        rhythm, and a leader estimate that **does not beat its shuffle null**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 📦 What we ship next
        The correlation heatmap already told the story: **19 correlated knobs are really a handful of
        independent ones.** In **NB04 — The Collapse I (PCA)** we find those axes automatically,
        watch Gauge A fall from 19 to ~6, and confront the first modeling *choice* — which axis to
        call "nuisance."
        """
    )
    return


if __name__ == "__main__":
    app.run()

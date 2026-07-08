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


# ============================================================ 1. Why this notebook
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB03 · The Signal in Time

        ## Why we are here

        You are a behavioral neuroscientist studying social behavior in mice. In the previous
        notebooks you turned raw video into **pose**: for each mouse, in each video frame, the
        pixel position of 15 body **keypoints** (nose, ears, shoulders, tail base, and so on). A
        keypoint is just an `(x, y)` coordinate — a labelled dot on the animal's body. Stacked over
        time, those dots become the movement of the animal.

        Two notebooks from now (NB04, NB05) we will **compress** each social interaction into a small
        set of numbers, and in doing so we will average away *time* — the moment-to-moment unfolding
        of the encounter. Before we discard it, this notebook looks carefully at what that time course
        contains. We ask three plain questions about an interaction:

        1. **What is the shape of each measurement?** (its distribution across many events)
        2. **How does one mouse move through time and at what rhythm?** (a time–frequency view)
        3. **Do two mice move together, and who moves first?** (coordination between animals)

        These are standard tools for quantifying behavior objectively, rather than by eye.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Definitions you need first

        **An approach event.** The dataset is a collection of short clips, each about 2.6 seconds
        long (130 frames at 50 frames per second). Each clip was extracted by the lab's tracking
        pipeline as an *approach*: a moment when two mice **start far apart** (their body centers
        roughly 200 pixels apart) and **close the distance** until they are in contact (centers about
        150 pixels apart, noses nearly touching). So every clip is centered on the same kind of
        moment — one animal coming up to another — with **contact** occurring at a fixed point,
        frame 40 (0.8 s in).

        **Three mice, named by role.** Each clip contains three animals, stored in a fixed order:

        - the **approacher** — the mouse that closes the distance (mouse 0),
        - the **approachee** — the mouse being approached (mouse 1),
        - the **bystander** — the third mouse, present but not the focus (mouse 2).

        **Rank, and how we color mice.** Each mouse has a social **rank** measured separately by the
        lab. Throughout every notebook, mice are colored **only by rank**: **Dom = red**,
        **Mid = blue**, **Sub = green** (gray = unknown). We never use color to mean anything else
        about a mouse.

        **The aggression label.** Some approaches escalate into aggression and some do not. A member
        of the lab **watched each clip and hand-scored it** as aggression or not — these are
        **human-labelled ground truth**, not something a model guessed. Roughly 450 of the 1,500
        clips are labelled aggression. We will often split plots by this label to see where
        aggression differs from ordinary approach.

        **Who interacts with whom.** With three animals in the frame, a natural question is: which two
        are actually engaged with each other? One simple clue is **coordination** — if two mice speed
        up and slow down *together*, their movements are **correlated**, and correlation is a hint
        that they are interacting rather than moving independently. We will make this precise in
        Section 4, and we will test how far the hint can be trusted.
        """
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
    EXAMPLE = 909              # example approach event: cage-15, male, aggression, contact at frame 40
    NULL_MEAN = 0.50           # within-event shuffle null: mean approacher-leads fraction
    NULL_HI = 0.545            # shuffle null 97.5th percentile
    FULL_FRAC = 0.479          # observed appr-leads fraction, all 420 usable aggression events
    FULL_N = 420
    DOM_FREQ = 1.5             # example-event dominant speed frequency (Hz), 1-12 Hz band
    HIFREQ = [87, 986, 1, 452, 575, 1052]   # well-tracked events with HIGH-frequency speed content
    CC_HI = 189                # example where approacher+approachee move together (peak corr ~0.89)
    CC_LO = 604                # example where a BYSTANDER pair is most correlated, not the interacting pair
    WHO_FRAC = 0.393           # frac of events where the interacting pair is the most-correlated of 3 pairs
    return (CC_HI, CC_LO, DOM_FREQ, FULL_FRAC, FULL_N, EXAMPLE, HIFREQ,
            NULL_HI, NULL_MEAN, WHO_FRAC)


# ============================================================ 2. Progress board (top)
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
        ### Progress board — start of NB03

        | Gauge | Reference | Your number | Note |
        |---|---|---|---|
        | **A · size of the representation** | {_bench_A:.0f} features (end of NB02) | **{_student_A}** features | We only *look* today; we do not compress yet. |
        | **B · held-out readiness** | rises in Phase 2 | **0** | Still in the Discover phase; the decoder is built in NB08. |

        Gauge A stays at 19 for this whole notebook. Exploratory analysis does not shrink the
        representation; its job is to decide **what** the later compression (NB04, NB05) is allowed to
        throw away.
        """
    )
    return


# ============================================================ 3. Held-out cage 16
@app.cell(hide_code=True)
def _(ho, mo):
    _n16 = int(len(ho["agg_label"]))
    mo.md(
        f"""
        <div style="border:2px dashed #b00; border-radius:10px; padding:14px 18px;
        background:repeating-linear-gradient(45deg,#faf0f0,#faf0f0 10px,#f3e4e4 10px,#f3e4e4 20px);">

        ### Held-out cage 16 — kept sealed

        Events recorded: **{_n16}** &nbsp;·&nbsp; skeletons hidden &nbsp;·&nbsp; labels hidden

        Cage 16 is set aside so we can later test whether our methods work on data they were never
        tuned on. A result is only trustworthy if it holds on a cage the analysis never saw. We do not
        touch Cage 16 in this notebook.

        **Unlocks in NB08.**
        </div>
        """
    )
    return


# ============================================================ 4. What an approach looks like
@app.cell(hide_code=True)
def _(EXAMPLE, mo):
    mo.md(
        rf"""
        ## The example approach event

        We will follow **one clip** throughout this notebook so every method has a concrete picture
        attached to it. Our example is event #{EXAMPLE} (Cage 15, male; a hand-scored aggression
        approach with clean tracking). The approacher is the **Dom (red)** mouse, the approachee is
        the **Sub (green)** mouse, and the **Mid (blue)** mouse is the bystander.

        The animation below shows the three skeletons over the 2.6-second clip. Watch the red mouse
        close in on the green one; the small red dot marks the moment of contact.
        """
    )
    return


@app.cell
def _(EXAMPLE, cr, cu, kp, mo, ranks):
    _gif = cu.event_gif_bytes(kp[EXAMPLE], ranks[EXAMPLE], contact_rel=int(cr[EXAMPLE]), cell=210, fps=20)
    mo.vstack([
        mo.md(f"**Example approach event #{EXAMPLE}** — skeletons colored by rank "
              "(red = Dom, blue = Mid, green = Sub); the white arrow points approacher → approachee; "
              "the red dot marks contact onset."),
        mo.Html(cu.gif_img_html(_gif, width=230)),
    ])
    return


# ============================================================ 5. Feature distributions
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Distributions — look at each measurement first

        **Why.** Before fitting any model, it is good practice to look at the raw numbers. From the
        pose, the pipeline computed **19 features** per event — single numbers that summarize the
        interaction, such as how fast the mice close the gap (`closing_speed`) or how directly one
        mouse faces the other (`appr_faces_appe`). Each feature has a **distribution**: the spread of
        its values across all 1,500 events. Aggression often shows up not as a shift in the *average*
        but as a heavier **tail** — a subset of events with unusually large values.

        **Method.** Pick a feature and a way to split the events into groups. A histogram shows, for
        each group, how often each value occurs. Splitting by **sex** and by **condition** (pre / dep
        / post) is a habit worth keeping, because those are the variables a hidden confound could ride
        in on later.

        - **Purpose:** compare a feature's distribution across groups.
        - **Inputs:** one feature column, and a grouping variable.
        - **Output:** overlaid histograms (probability density, so groups of different size are
          comparable).
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


# ============================================================ 6. Correlation heatmap
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · The 19×19 feature-correlation heatmap

        **Why.** The 19 features are **not independent** measurements. Many of them rise and fall
        together across events, which means the representation carries **redundant** information. This
        matters because the next notebook (NB04) uses a method called **PCA** to replace many
        redundant features with a few independent ones — and this heatmap is the picture that
        motivates it.

        **Definition — correlation.** The **Pearson correlation** `r` between two features measures
        how linearly they move together across events: `r = +1` (rise together), `r = -1` (one rises
        as the other falls), `r = 0` (unrelated). Here we compute `r` for every pair of the 19
        features and display it as a colored grid.

        **Method.** `np.corrcoef` takes the standardized feature matrix (events × features) and
        returns the 19×19 matrix of pairwise correlations. Bright off-diagonal blocks mark groups of
        features that are essentially measuring the same thing.
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
        f"Because several features are this redundant, about 6 combined axes will capture most of the "
        "variation in NB04.")])
    return


# ============================================================ 7. Rhythm in time & frequency
@app.cell(hide_code=True)
def _(EXAMPLE, mo):
    mo.md(
        rf"""
        ## 3 · Rhythm — one mouse in time and frequency

        **Why.** A single number like "average speed" hides *how* a mouse moved. Two animals can have
        the same average speed while one glides smoothly and the other darts in quick bursts. The
        **rhythm** of movement — how quickly speed rises and falls — is itself informative, and it is
        exactly the kind of structure the later compression discards.

        **Method, part 1 — raw traces.** First we simply plot, for example event #{EXAMPLE}, the
        distance between the two mice and each mouse's speed over time. We expect the distance to
        collapse and the speeds to rise as contact approaches.
        """
    )
    return


@app.cell
def _(EXAMPLE, appr_appe_speed, cr, cu, kp, make_subplots, mo, np, ranks):
    _k = kp[EXAMPLE]
    _c0 = cu._centroids(_k[:, 0]); _c1 = cu._centroids(_k[:, 1])
    _dist = np.linalg.norm(_c0 - _c1, axis=1)               # (T,) closing distance
    _s0, _s1 = appr_appe_speed(kp, EXAMPLE)                    # (T-1,)
    _t = np.arange(len(_dist)) / cu.FPS
    _te = np.arange(len(_s0)) / cu.FPS
    _cf = int(cr[EXAMPLE]) / cu.FPS
    _c_appr = cu.RANK_HEX[int(ranks[EXAMPLE][0])]              # approacher colored by ITS rank
    _c_appe = cu.RANK_HEX[int(ranks[EXAMPLE][1])]              # approachee colored by ITS rank
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("pair distance (px)", "per-mouse speed (px/frame)"))
    _fig.add_scatter(x=_t, y=_dist, mode="lines", line=dict(color="#333"), name="pair distance",
                     row=1, col=1)
    _fig.add_scatter(x=_te, y=_s0, mode="lines", line=dict(color=_c_appr), name="approacher (Dom)",
                     row=2, col=1)
    _fig.add_scatter(x=_te, y=_s1, mode="lines", line=dict(color=_c_appe), name="approachee (Sub)",
                     row=2, col=1)
    for _r in (1, 2):
        _fig.add_vline(x=_cf, line=dict(color="#888", dash="dot"), row=_r, col=1)
    _fig.update_layout(template="plotly_white", height=430, font=dict(size=14),
                       margin=dict(l=10, r=10, t=40, b=10))
    _fig.update_xaxes(title_text="time (s)", row=2, col=1, showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([_fig, mo.md("The dotted line marks contact. The distance falls and both speeds rise "
                           "into the collision. Next we ask *at what rhythm* the speed changes.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Method, part 2 — the wavelet spectrogram.** To read rhythm we use a **Morlet wavelet
        transform**. In plain terms: a wavelet is a short wave-shaped template. We slide a template of
        a given frequency along the speed signal and measure how strongly the signal matches it at
        each moment. Repeating this across many frequencies produces a **spectrogram** — a picture
        with **time** on the horizontal axis, **frequency** (rhythm, in cycles per second, Hz) on the
        vertical axis, and brightness showing how much of that rhythm is present at that moment.

        - **Function:** `cu.wavelet_power(signal, freqs, fps)` (wrapped by `padded_wavelet`, which
          pads the short clip so low-frequency templates fit).
        - **Inputs:** a 1-D signal (here the approacher's speed), a list of frequencies to test, and
          the sampling rate (50 fps).
        - **Output:** a `frequency × time` grid of power — bright where that rhythm is present.

        Slide the control to change the top frequency shown. The dotted white line marks the
        strongest (dominant) rhythm.
        """
    )
    return


@app.cell
def _(mo):
    fmax_slider = mo.ui.slider(6, 20, value=12, step=1, label="wavelet upper frequency (Hz)",
                               debounce=True, full_width=True)
    return (fmax_slider,)


@app.cell
def _(DOM_FREQ, EXAMPLE, appr_appe_speed, cu, fmax_slider, go, kp, mo, np, padded_wavelet):
    _s0, _ = appr_appe_speed(kp, EXAMPLE)
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
        f"**Dominant rhythm ≈ {_dom:.1f} Hz** (pinned build value {DOM_FREQ} Hz over the 1–12 Hz "
        "band). This is a slow rhythm — the pace of ordinary locomotion, a mouse taking a few steps "
        "per second — not a fast oscillation.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### A concrete use: finding events with high-frequency movement

        The example event's speed is dominated by a slow (~1.5 Hz) locomotor rhythm. But some events
        contain **high-frequency** speed content: the speed rises and falls several times per second,
        which corresponds to **quick, jerky movement** — rapid darting, scrambling, or repeated
        start-and-stop — rather than a smooth glide.

        We can use the wavelet to *find* those events: for every event we take the dominant frequency
        of the approacher's speed, and keep the ones with an unusually high value. Below are the
        skeleton animations of six such high-frequency events. Watch how their movement looks
        abrupt and stuttery compared with the smooth approach in the example event above. (Both
        aggression and non-aggression events appear here — high-frequency movement is a description of
        *how* the animal moved, not by itself a sign of aggression.)
        """
    )
    return


@app.cell
def _(HIFREQ, cr, cu, kp, mo, ranks):
    _events = [(kp[i], ranks[i], int(cr[i])) for i in HIFREQ]
    _gif = cu.grid_gif_bytes(_events, ncols=3, cell=150, fps=18)
    mo.vstack([
        mo.md(f"**High-frequency movement exemplars** (events {HIFREQ}) — skeletons colored by rank. "
              "Their speed changes several times per second: short darts and abrupt stops."),
        mo.Html(cu.gif_img_html(_gif, width=470)),
    ])
    return


@app.cell
def _(HIFREQ, appr_appe_speed, cu, go, kp, mo, np, padded_wavelet):
    # Spectrogram of one HIGH-frequency event, for contrast with the ~1.5 Hz example above.
    _i = HIFREQ[0]
    _s0, _ = appr_appe_speed(kp, _i)
    _freqs = np.linspace(1.0, 12.0, 45)
    _P = padded_wavelet(_s0, _freqs, cu.FPS, padlen=600)
    _t = np.arange(_P.shape[1]) / cu.FPS
    _dom = float(_freqs[np.argmax(_P.mean(axis=1))])
    _fig = go.Figure(go.Heatmap(z=_P, x=_t, y=_freqs, colorscale="Viridis",
                                colorbar=dict(title="power")))
    _fig.add_hline(y=_dom, line=dict(color="white", dash="dot"))
    _fig.update_layout(template="plotly_white", height=360, font=dict(size=14),
                       title=f"High-frequency event #{_i} — dominant ≈ {_dom:.1f} Hz",
                       xaxis_title="time (s)", yaxis_title="frequency (Hz)",
                       margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([_fig, mo.md(
        f"Here the power sits **higher** on the frequency axis (~{_dom:.1f} Hz) than in the example "
        "event's slow-locomotion spectrogram. The bright band moved up because the speed itself "
        "changes more times per second.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Two ways the wavelet is limited on this data

        These clips are only ~2.6 s long, and that creates two honest limitations to keep in mind:

        1. **Time–frequency trade-off.** A wavelet narrow enough to pin down *when* a burst happened
           is blurry about *what frequency* it was, and vice versa. You cannot have sharp timing and
           sharp frequency at the same time.
        2. **Edge effects.** A low-frequency template is wider than the clip, so it runs off both
           ends. We pad the signal just to fit the template, but the padded flanks are fabricated, not
           measured. Trust the **middle** of each spectrogram and distrust the extreme left and right
           edges.

        This is also why a wavelet is more appropriate here than a single Fourier transform (FFT): an
        FFT assumes one fixed spectrum for the whole clip, whereas the wavelet allows the rhythm to
        **change** as contact approaches — which it does.
        """
    )
    return


# ============================================================ 8. Coordination between two mice
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · Coordination — who is interacting with whom?

        **Why.** Each clip has three mice, and we would like to know which two are actually engaged.
        The pipeline *labelled* one pair as the approacher and approachee, but can we recover that
        from movement alone? The intuition from the top of the notebook: two mice that are interacting
        should move in a **coordinated** way — when one speeds up, so does the other.

        **Definition — cross-correlation.** To measure coordination we slide one mouse's speed trace
        against another's and, at each shift (**lag**), compute how well the two line up. The
        **peak** value (best alignment over small lags) is a single number between about -1 and +1:
        near +1 means the two speed traces rise and fall together (**highly coordinated**), near 0
        means they are unrelated.

        - **Function:** `cu.cross_corr_lag(x, y, max_lag)`.
        - **Inputs:** two speed traces and the largest lag to consider (in frames).
        - **Outputs:** the list of lags, the correlation at each lag, and the peak lag.

        Below, two examples make the idea concrete.
        """
    )
    return


@app.cell
def _(CC_HI, cr, cu, go, kp, make_subplots, mo, np, ranks):
    _i = CC_HI
    _k = kp[_i]
    _sp = [np.nan_to_num(np.linalg.norm(np.diff(cu._centroids(_k[:, m]), axis=0), axis=1))
           for m in range(3)]
    _t = np.arange(len(_sp[0])) / cu.FPS
    _lags, _corr, _pk = cu.cross_corr_lag(_sp[0], _sp[1], 10)
    _peak = float(_corr.max())
    _c0 = cu.RANK_HEX[int(ranks[_i][0])]; _c1 = cu.RANK_HEX[int(ranks[_i][1])]
    _fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                         subplot_titles=("approacher & approachee speed (px/frame)",
                                         "cross-correlation vs lag"))
    _fig.add_scatter(x=_t, y=_sp[0], mode="lines", line=dict(color=_c0), name="approacher",
                     row=1, col=1)
    _fig.add_scatter(x=_t, y=_sp[1], mode="lines", line=dict(color=_c1), name="approachee",
                     row=1, col=1)
    _fig.add_scatter(x=_lags, y=_corr, mode="lines+markers", line=dict(color="#7b3294"),
                     showlegend=False, row=1, col=2)
    _fig.update_layout(template="plotly_white", height=340, font=dict(size=13),
                       title=f"HIGH coordination — event #{_i}: peak correlation ≈ {_peak:.2f}",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    _gif = cu.event_gif_bytes(_k, ranks[_i], contact_rel=int(cr[_i]), cell=170, fps=18)
    mo.vstack([_fig, mo.Html(cu.gif_img_html(_gif, width=200)),
               mo.md("The two speed traces rise and fall together, so the peak correlation is high. "
                     "In the animation the two mice really are moving as a coordinated pair.")])
    return


@app.cell
def _(CC_LO, cr, cu, go, kp, make_subplots, mo, np, ranks):
    _i = CC_LO
    _k = kp[_i]
    _sp = [np.nan_to_num(np.linalg.norm(np.diff(cu._centroids(_k[:, m]), axis=0), axis=1))
           for m in range(3)]
    _t = np.arange(len(_sp[0])) / cu.FPS
    _pk01 = float(cu.cross_corr_lag(_sp[0], _sp[1], 10)[1].max())     # interacting pair
    _pk12 = float(cu.cross_corr_lag(_sp[1], _sp[2], 10)[1].max())     # a bystander pair
    _c0 = cu.RANK_HEX[int(ranks[_i][0])]; _c1 = cu.RANK_HEX[int(ranks[_i][1])]
    _c2 = cu.RANK_HEX[int(ranks[_i][2])]
    _fig = make_subplots(rows=1, cols=1)
    _fig.add_scatter(x=_t, y=_sp[0], mode="lines", line=dict(color=_c0), name="approacher")
    _fig.add_scatter(x=_t, y=_sp[1], mode="lines", line=dict(color=_c1), name="approachee")
    _fig.add_scatter(x=_t, y=_sp[2], mode="lines", line=dict(color=_c2, dash="dot"), name="bystander")
    _fig.update_layout(template="plotly_white", height=320, font=dict(size=13),
                       title=(f"LOW coordination for the labelled pair — event #{_i}: "
                              f"approacher–approachee ≈ {_pk01:.2f}, but approachee–bystander ≈ {_pk12:.2f}"),
                       xaxis_title="time (s)", yaxis_title="speed (px/frame)",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    _gif = cu.event_gif_bytes(_k, ranks[_i], contact_rel=int(cr[_i]), cell=170, fps=18)
    mo.vstack([_fig, mo.Html(cu.gif_img_html(_gif, width=200)),
               mo.md("Here the labelled approacher and approachee are **poorly** correlated, while the "
                     "approachee and the **bystander** happen to move together more closely. Correlation "
                     "alone would point at the wrong pair.")])
    return


@app.cell(hide_code=True)
def _(WHO_FRAC, mo):
    mo.md(
        rf"""
        ### Does the most-correlated pair identify the interacting pair?

        The two examples suggest correlation is a *hint*, not a proof. To check it properly, for every
        well-tracked event we computed the peak speed-correlation of all three possible pairs
        (approacher–approachee, approacher–bystander, approachee–bystander) and asked how often the
        **labelled interacting pair** was the **most-correlated** of the three.

        The answer is only **{WHO_FRAC*100:.0f}%** — barely better than the 33% you would get by
        chance among three pairs. Two mice can move together for reasons that have nothing to do with
        interacting: both happen to be resting still, both are walking across the cage in parallel, or
        all three share a common startle. This is the key caution for the rest of the section:
        **coordination is suggestive of who-interacts-with-whom, but it does not settle it.** The same
        caution returns when we ask *who moves first*, next.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Who moves first? A directed version of coordination

        **Why.** Cross-correlation also tells us about **order**: if the approacher's speed changes
        and the approachee's speed follows a moment later, the peak alignment occurs at a positive
        **lag**, and we say the approacher **leads**. Asking "who moves first before contact" is a
        natural behavioral question — but we will see it is hard to answer reliably on such short
        clips.

        We look only at the frames **before contact**. After contact, both mice are guaranteed to be
        moving together, so including those frames would answer the question trivially.

        First, a small demonstration that the estimator works when the answer is known: we build two
        signals where one is a delayed copy of the other, and check that `cross_corr_lag` recovers the
        delay we imposed.
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
        "A positive lag means A leads B. The estimator recovers the imposed lag cleanly on a long, "
        "clean signal — the regime the real, short, noisy mouse clips do **not** enjoy.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we run the same test on the real pre-contact traces. For each aggression event we ask
        whether the approacher leads, and we report the **fraction of events** in which it does. If
        there were no consistent leader, that fraction would sit near **0.50**. The gray band is a
        **shuffle null**: we scrambled the traces within each event to see how far from 0.50 the
        fraction wanders by chance alone. A real leader effect would have to poke **outside** that
        band.
        """
    )
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
        f"fraction = **{FULL_FRAC:.3f}** — inside the gray shuffle band. The bars land near 0.50 too: "
        "there is no robust leader. That is the honest result, and the exercise below asks you to "
        "confirm it.")])
    return


# ============================================================ 9. Exercise
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise — is the "leader" real, or shuffle-null noise?

        **The claim to test.** *Before contact, the approacher is a consistent leader — the fraction
        of events in which it leads sits outside the within-event shuffle null.*

        A note on scope. We would like to test the sharper claim "the leader is the aggression
        initiator," but the dataset does not ship a per-mouse initiator label, so there is no ground
        truth to grade that against. Instead we grade something we *can* check honestly: does the
        leader estimate beat its own shuffle null?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The tools you have
        - `pre_speeds(kp, cr, i)` → `(approacher_speed, approachee_speed)` over the pre-contact window.
        - `cu.cross_corr_lag(x, y, max_lag)` → `(lags, corr, peak_lag)`; **`peak_lag > 0` means x leads y**.
        - `leader_fraction(kp, cr, idxs, max_lag)` → `(fraction_approacher_leads, n_usable)`.
        - Pinned shuffle null: mean = 0.50, upper edge (97.5th pct) = 0.545.

        ### What to do
        You will edit **one line**. The cell below computes the approacher-leads fraction on the
        subsample `sub_idx`. The active line already calls `leader_fraction` so the notebook runs; the
        commented line just above it shows the same call with the helper name **blanked out** — try
        filling it in yourself, then run the cell. After it runs, the next cell draws your result as a
        single bar against the shuffle band.

        **Expected picture:** one bar sitting close to 0.50 and **inside** the gray band — i.e. no
        robust leader.
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
    # ---- TODO (student): edit ONE line -----------------------------------------
    # The function that returns (fraction_approacher_leads, n_usable) is `leader_fraction`.
    # Try replacing ____ with that function name, then run the cell:
    #     obs_frac, obs_n = ____(kp, cr, sub_idx, max_lag=10)
    #
    # The active line below already does it for you so the notebook renders:
    obs_frac, obs_n = leader_fraction(kp, cr, sub_idx, max_lag=10)
    # ---------------------------------------------------------------------------
    return obs_frac, obs_n


@app.cell
def _(NULL_HI, go, mo, obs_frac, obs_n):
    _fig = go.Figure()
    _fig.add_bar(x=["your subsample"], y=[obs_frac], marker_color="#4c78a8",
                 text=[f"{obs_frac:.2f}<br>n={obs_n}"], textposition="outside")
    _fig.add_hline(y=0.5, line=dict(color="#333", dash="dash"),
                   annotation_text="no consistent leader (0.50)")
    _fig.add_hrect(y0=1 - NULL_HI, y1=NULL_HI, fillcolor="#bbbbbb", opacity=0.25, line_width=0,
                   annotation_text="shuffle null (95%)", annotation_position="top left")
    _fig.update_layout(template="plotly_white", height=340, font=dict(size=14),
                       yaxis_title="fraction approacher LEADS", yaxis_range=[0, 1],
                       title="Your result vs the shuffle null", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Solution (reveal)": mo.md(
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
                        if peak_lag != 0:                         # peak_lag > 0 means approacher leads
                            leads.append(1 if peak_lag > 0 else 0)
                    return float(np.mean(leads)), len(leads)

                obs_frac, obs_n = my_leader_fraction(kp, cr, sub_idx, max_lag=10)
                # obs_frac is about 0.47 — indistinguishable from the 0.50 shuffle null.
                ```

                The number lands **inside** the shuffle band. The correct scientific conclusion is
                that there is **no robust leader in this window**: a short (~1 s), nonstationary, noisy
                pre-contact trace does not carry a reliable lead-or-follow sign.
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
    _msg = (f"**PASS.** Observed approacher-leads fraction = **{obs_frac:.3f}** "
            f"(n={obs_n}); |fraction − 0.5| = {_gap:.3f} ≤ shuffle band ({_band:.3f}). "
            "You correctly concluded the leader estimate **does not robustly exceed chance**."
            if _pass else
            f"Observed fraction = {obs_frac:.3f}; |fraction − 0.5| = {_gap:.3f} exceeded the "
            f"band ({_band:.3f}). On this bundle the honest answer is *no robust leader* — recheck "
            "that you used pre-contact frames and the shuffle null.")
    mo.md(
        f"""
        <div style="background:{_color}; border-left:5px solid {_edge}; border-radius:6px;
        padding:12px 16px;">

        ### Self-check
        {_msg}

        The grade is on the *robustness* conclusion, not on a noisy p-value: the tolerance band
        asserts the leader sits **inside** the null, which is the pre-verified build-time result.
        </div>
        """
    )
    return


# ============================================================ 10. Granger (optional)
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Optional deeper section — Granger causality

        **Why include it.** Cross-correlation asks *when* two traces align. **Granger causality** asks
        a sharper question: does knowing the **past of mouse A** improve our prediction of **mouse B's
        next step**, beyond what B's own past already tells us? If it does, we say A "Granger-causes"
        B. It is a statistical test comparing a model that uses only B's past to one that also uses
        A's past.

        - **Function:** `cu.granger_pair(x, y, lags=4)` (pure numpy, no `statsmodels`).
        - **Inputs:** two speed traces and how many past frames to use.
        - **Outputs:** an F-statistic and p-value for each direction (A→B and B→A). A small p-value
          suggests directed influence.
        """
    )
    return


@app.cell(hide_code=True)
def _(EXAMPLE, cr, cu, kp, mo, pre_speeds):
    _x, _y = pre_speeds(kp, cr, EXAMPLE)
    try:
        _g = cu.granger_pair(_x, _y, lags=4)
        _txt = (f"Example event #{EXAMPLE}: approacher→approachee F = {_g['f_xy']:.2f} "
                f"(p = {_g['p_xy']:.3f}); approachee→approacher F = {_g['f_yx']:.2f} "
                f"(p = {_g['p_yx']:.3f}).")
    except Exception as _e:
        _txt = f"(Granger skipped: {_e})"
    mo.accordion({
        "Granger on the example event, with the caveat": mo.md(
            rf"""
            {_txt}

            **The common-cause caveat.** Granger measures **prediction, not cause**. Both mice can be
            driven by a **shared third factor** — the bystander mouse, or a common startle — which
            makes A look like it drives B when neither actually does. Bivariate Granger is also not
            *conditional*: to move from "A predicts B" to "A predicts B *given the bystander*," you
            would add the third mouse's trace as an extra input (a conditional, multivariate model).
            On 1-second, nonstationary windows, treat any single-event Granger number as a hint, not
            a verdict.
            """
        )
    })
    return


# ============================================================ 11. Review questions
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Review questions
        1. **Name a confounder.** Coordination is not proof of interaction. What shared cause on this
           rig could make two mice look coupled when neither is driving the other? (The bystander; a
           common startle or arousal spike shared by all three.)
        2. **Wavelet vs FFT.** When is a wavelet more appropriate than a single FFT? (When the
           spectrum is *non-stationary* — the rhythm changes across the 2.6 s, which an FFT would
           average into one blurred spectrum.)
        3. **Why pre-contact only?** Why must the leader test use only pre-contact frames? (After
           contact both mice necessarily move together, so including those frames answers the question
           trivially.)
        4. **Missing ground truth.** We could not test "leader equals initiator." What label would the
           dataset need, and why does its absence force a robustness test rather than an accuracy
           test?
        """
    )
    return


# ============================================================ 12. What time-collapsing discards
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What the next notebooks will discard
        - **The time course.** By summarizing each event with 19 numbers, PCA and the map (NB04,
          NB05) drop the within-event *unfolding* — the rhythm and the moment-by-moment coordination
          we studied here.
        - **Where the methods break on this data.** (1) 130-frame clips are **short and
          nonstationary**, which causes wavelet edge effects and noisy lag estimates; (2) the
          approacher is **not necessarily the initiator**, and no initiator label exists to close that
          gap; (3) a **shared third mouse** can fake coordination that a two-animal measure cannot rule
          out.
        - **Something to think about.** How would you *condition out* the bystander to move from a
          two-animal coordination measure to a three-animal one — and what would change if the dataset
          did ship a per-mouse initiator label?
        """
    )
    return


# ============================================================ 13. Board + hook
@app.cell(hide_code=True)
def _(X, mo):
    mo.md(
        f"""
        ### Progress board — end of NB03

        | Gauge | Value | Change this notebook |
        |---|---|---|
        | **A · size of the representation** | **{int(X.shape[1])}** features | unchanged — we looked, we did not compress |
        | **B · held-out readiness** | **0** | unchanged — the coordination signal is honestly null, so nothing is bankable yet |

        We surveyed rhythm and coordination and reported both honestly: a real but slow (~1.5 Hz)
        movement rhythm, and a leader estimate that **does not beat its shuffle null**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What comes next
        The correlation heatmap already showed that the 19 features are really a **handful** of
        independent ones. In **NB04 — Compression I (PCA)** we find those combined axes automatically,
        watch Gauge A fall from 19 to about 6, and meet the first real modeling choice: which axis to
        treat as signal and which as nuisance.
        """
    )
    return


if __name__ == "__main__":
    app.run()

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


# ============================================================ 1. Briefing
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 07 · Behavior in Time — the grammar and the clock

        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > The rig runs for *hours*, not seconds. Before we can say a laser changed anything, we need
        > two temporal readouts of the *un-manipulated* animal. **(1) Does behavior have memory** —
        > does the state right now predict the state next, or is each moment an independent coin flip?
        > **(2) When are these mice active** across the day (ours run a *reversed* light cycle —
        > lights ON 21:00–09:00, so the **dark/active phase is ~09:00–21:00**)? Ship us a transition
        > **grammar** and an activity **clock**, both with an honest null and honest error bars.
        >
        > **Deliverable:** a first-order Markov transition matrix + a stationary distribution + a
        > shuffle-tested entropy, plus a bootstrapped activity-by-time-of-day curve.
        > **Unblocks:** any claim that opto *reorganized dynamics* — you can't detect a change in the
        > grammar until you've measured the baseline grammar.
        >
        > **Lab-meeting question:** *Does the behavioral grammar carry real memory that beats a
        > time-shuffled null — and can a 3-cage recording say anything honest about the daily rhythm?*

        Every frozen snapshot so far (NB01–06) treated an event as a still. Behavior **moves**. A
        transition matrix over labeled states is the *observed* cousin of the **hidden-Markov models**
        neuroscientists fit to latent brain states — and MoSeq's AR-HMM is exactly this idea applied
        to mouse behavior, its syllable transitions read out downstream by the striatum. The
        difference, which we will be honest about all notebook, is that **we labeled our states by
        hand**; the brain's are latent and must be *inferred* — and that inference *is* the HMM.
        """
    )
    return


# ============================================================ Readout board (top)
@app.cell
def _(ROOT, cu):
    import csv as _csv
    _board = []
    try:
        with open(cu.data_path("data/readout_board.csv", ROOT)) as _f:
            for _r in _csv.DictReader(_f):
                _board.append(_r)
    except Exception:
        _board = []
    board = _board
    return (board,)


@app.cell(hide_code=True)
def _(board, mo):
    def _find(gauge, nb):
        for r in board:
            if r.get("gauge") == gauge and r.get("notebook") == nb:
                return r
        return None
    _a = _find("A", "NB05")          # "one syllable" = 1 label
    _b06 = _find("B", "NB06")        # features -> aggression CV
    _b08 = _find("B", "NB08")        # held-out horizon
    _a_val = _a["value"] if _a else "1"
    _b06_val = _b06["value"] if _b06 else "0.837"
    _b08_val = _b08["value"] if _b08 else "0.86"
    mo.md(
        f"""
        ### 📋 Readout Board — *entering* NB07

        | Gauge | Where we are | Benchmark | Note |
        |---|---|---|---|
        | **A · size of representation** | **1 syllable** — a single state label per moment | `{_a_val}` | Phase-1 collapse bottomed out at NB05; NB07 does **not** shrink it further — it adds the **time axis**. |
        | **B · held-out readiness** | features→aggression CV **{_b06_val}** AUROC | `{_b06_val}` | Horizon: a decoder that survives an unseen cage (**{_b08_val}**, NB08). NB07 validates *dynamics*, not a decoder. |

        *Two different kinds of reduction — a compression (A) and a validation (B). They are not one
        magic number.*
        """
    )
    return


# ============================================================ Sealed Cage 16
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div style="border:2px dashed #b08; border-radius:10px; padding:14px 18px; background:
        rgba(180,0,140,0.05);">
        <b>🔒 SEALED — Camera 16 · "the animal on the rig"</b><br>
        Held-out cage · <b>470 events</b> · skeletons <span style="background:#333;color:#333;
        border-radius:3px;">████████</span> · labels <span style="background:#000;color:#000;
        border-radius:3px;">██████</span><br>
        <b>Notebooks until unlock: 1</b> — opens in <b>NB08</b>, where the decoder finally meets a cage
        it never trained on. The grammar you build here is on the <i>training</i> cages only.
        </div>
        """
    )
    return


# ============================================================ Data load
@app.cell
def _(cu):
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of all-NaN frames is expected
    ev = cu.load_events("data/train_events.npz")
    der = cu.load_derived("train")
    # Three continuous 24h cages at 2 fps: 15 = hero (M), 10 = context (F), 13 = context (M).
    t15 = cu.load_continuous_tracks("15")
    t10 = cu.load_continuous_tracks("10")
    t13 = cu.load_continuous_tracks("13")
    return ev, t10, t13, t15


@app.cell
def _(cu, ev):
    # Hero Event for NB07. Design's "#742" is actually cage-12 non-aggression in the shipped bundle,
    # so we use idx 508: cage 15, male, category='aggression', DEP phase (matching our continuous
    # dep-span cage 15), reliability 0.995, contact at frame 40, time-of-day ~15.3h.
    hero_idx = 508
    _kp = ev["kp"][hero_idx]
    _ranks = ev["ranks"][hero_idx]
    hero_key = str(ev["event_key"][hero_idx])
    hero_tod = float(cu.time_of_day(hero_key))
    hero_gif = cu.gif_img_html(
        cu.event_gif_bytes(_kp, _ranks, int(ev["contact_rel"][hero_idx]), cell=200), width=200)
    return hero_gif, hero_idx, hero_tod


# ============================================================ Baton hand-off
@app.cell(hide_code=True)
def _(hero_gif, hero_idx, hero_tod, mo):
    mo.md(
        f"""
        ## 1 · The baton hand-off — one event → the day it lived in

        All week we've followed one 2.6-second bout — our canonical **Hero Event #909** (Cage 15,
        male, *post* phase). This notebook's grammar and activity clock come from cage 15's single
        continuous 24-hour recording, which happens to be a *dep*-phase day; so to land the baton on
        this day's clock we follow #909's dep-phase sibling, **Hero Event #{hero_idx}** (Cage 15,
        male, dep phase — the cleanest cage-15 aggression on this day). Skeletons colored by rank
        (<span style="color:#d62728">Dom</span>/<span style="color:#1f77b4">Mid</span>/<span
        style="color:#2ca02c">Sub</span>):

        {hero_gif}

        This event happened at **{hero_tod:.1f} h** — deep in the dark/active phase. But 130 frames is
        a *snapshot*. To ask whether behavior has **memory** and a **daily rhythm**, we have to widen
        out to the whole day this event lived inside: a **continuous 24-hour recording** of the same
        cage 15, at 2 fps. Below, we'll drop a marker at **{hero_tod:.1f} h** on that day's activity
        clock — the baton passes from *one event* to *the session*.

        <div style="border-left:4px solid #e45756; padding:6px 12px; background:rgba(228,87,86,0.06);">
        <b>Why the sparse approach-events can't form a chain.</b> The 1500 event tensors are
        <i>disconnected</i> 130-frame clips ripped from different days and cages. A Markov chain needs a
        <b>contiguous</b> sequence — state<sub>t</sub> must actually be followed by state<sub>t+1</sub>
        in real time. So the grammar is built <b>only</b> from the continuous tracks, never from the
        event corpus.
        </div>
        """
    )
    return


# ============================================================ Discretize -> contiguous states
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · From kinematics to a contiguous state sequence

        `discretize_states(speed, centroids)` labels **every 0.5-second frame** of the continuous
        recording with one coarse cage-level state from two data-driven thresholds:

        - **rest** — mean mouse speed below the 40th percentile
        - **locomote** — moving, mice apart
        - **huddle** — the closest pair is nearer than the 25th-percentile distance (social proximity)

        This is the valid Markov substrate: a single unbroken ribbon of states, frame after frame.
        """
    )
    return


@app.cell
def _(cu, np, t15):
    # Recompute states live from raw kinematics and confirm they reproduce the shipped state_seq.
    _state_live, STATE_NAMES = cu.discretize_states(t15["speed"], t15["centroids"])
    states_match = bool(np.array_equal(_state_live, t15["state_seq"]))
    STATE_COLORS = ["#9e9e9e", "#ff7f0e", "#6a3d9a"]   # rest / locomote / huddle
    return STATE_COLORS, STATE_NAMES, states_match


@app.cell
def _(mo):
    # control for the state-ribbon scrubber (defined here, rendered next to its plot below)
    ribbon_start = mo.ui.slider(0, 172200, value=54000, step=600,
                                label="ribbon start frame (2 fps · window = 5 min)",
                                debounce=True, full_width=True)
    return (ribbon_start,)


@app.cell
def _(STATE_COLORS, STATE_NAMES, go, mo, ribbon_start, states_match, t15):
    _s0 = int(ribbon_start.value)
    _win = 600
    _seg = t15["state_seq"][_s0:_s0 + _win]
    _tod0 = float(t15["tod_hour"][_s0])
    _fig = go.Figure(go.Heatmap(
        z=[_seg], zmin=0, zmax=2,
        colorscale=[[0.0, STATE_COLORS[0]], [0.33, STATE_COLORS[0]],
                    [0.34, STATE_COLORS[1]], [0.66, STATE_COLORS[1]],
                    [0.67, STATE_COLORS[2]], [1.0, STATE_COLORS[2]]],
        showscale=False, hovertemplate="frame +%{x}<extra></extra>"))
    _fig.update_layout(template="plotly_white", height=170,
                       title=f"Cage-15 state ribbon — 5-min window starting ~{_tod0:.1f} h "
                             f"(gray=rest · orange=locomote · purple=huddle)",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_yaxes(showticklabels=False)
    _fig.update_xaxes(title="frames since window start (2 fps)")
    _check = ("✅ live `discretize_states` reproduces the shipped `state_seq` exactly"
              if states_match else "⚠️ mismatch — using shipped `state_seq`")
    mo.vstack([ribbon_start, _fig,
               mo.md(f"*{_check}. Notice the ribbon is **contiguous** — a real chain. "
                     f"States: {', '.join(STATE_NAMES)}.*")])
    return


# ============================================================ Per-cage grammar (compute)
@app.cell
def _(cu, np, t10, t13, t15):
    # Build the grammar for all three cages. Nulls at n=40: the shuffle-entropy of a 172,800-point
    # sequence is razor-tight (std ~1e-3), so 40 draws already pin it — bigger n is a gated extra.
    def _grammar(tr):
        s = tr["state_seq"]
        T = cu.transition_matrix(s, 3)
        return dict(
            sex=tr["sex"], T=T,
            H=float(cu.transition_entropy(T)),
            self=float(np.mean(np.diag(T))),
            frac=np.bincount(s, minlength=3) / len(s),
            null_H=cu.shuffle_transition_null(s, n=40, seed=0, stat="entropy"),
            null_self=cu.shuffle_transition_null(s, n=40, seed=0, stat="self"),
        )
    grammar = {"15": _grammar(t15), "10": _grammar(t10), "13": _grammar(t13)}
    CAGE_COLORS = {"15": "#1b9e77", "10": "#d95f02", "13": "#7570b3"}
    return CAGE_COLORS, grammar


# ============================================================ Transition matrix heatmap
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · The transition matrix — the grammar of the day

        `transition_matrix(state_seq)` counts every consecutive pair and row-normalizes, so
        **T[i, j] = P(next = j | now = i)**. The rows sum to 1: from any state, where does the animal
        go next? A near-diagonal matrix means behavior is **sticky** (long dwells); heavy off-diagonals
        mean it churns.
        """
    )
    return


@app.cell
def _(grammar, mo):
    cage_pick = mo.ui.dropdown(
        options={f"Cage {c} ({grammar[c]['sex']})" + (" · HERO" if c == "15" else ""): c
                 for c in ["15", "10", "13"]},
        value="Cage 15 (M) · HERO", label="cage")
    return (cage_pick,)


@app.cell
def _(STATE_NAMES, cage_pick, go, grammar, mo, np):
    _c = cage_pick.value
    _T = grammar[_c]["T"]
    _fig = go.Figure(go.Heatmap(
        z=_T, x=STATE_NAMES, y=STATE_NAMES, colorscale="Blues", zmin=0, zmax=1,
        text=np.round(_T, 2), texttemplate="%{text}", textfont=dict(size=15),
        colorbar=dict(title="P(next|now)")))
    _fig.update_layout(template="plotly_white", height=420,
                       title=f"Cage {_c} transition matrix — rows sum to 1",
                       xaxis_title="next state", yaxis_title="current state",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_yaxes(autorange="reversed")
    mo.vstack([cage_pick, _fig,
               mo.md(f"*The diagonal ({np.round(np.diag(_T),2).tolist()}) dominates — behavior "
                     f"**stays put** far more than chance (1/3 ≈ 0.33). That stickiness is the memory "
                     f"we are about to test against a null.*")])
    return


# ============================================================ Stationary distribution by simulation
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · The stationary distribution — release a walker

        Where does the animal spend its time *in the long run*? Forget eigenvectors. **Release a
        random walker** on the transition graph: start in `rest`, roll the dice `T[now]` to pick the
        next state, repeat thousands of times, and tally how often it lands in each state. That tally
        *is* the stationary distribution — the intuitive definition, no linear algebra. Drag the walk
        length and watch the estimate converge to the true long-run occupancy (the empirical state
        fractions).
        """
    )
    return


@app.cell
def _(mo):
    walk_steps = mo.ui.slider(500, 30000, value=8000, step=500,
                              label="walker steps", debounce=True, full_width=True)
    return (walk_steps,)


@app.cell
def _(STATE_COLORS, STATE_NAMES, cu, go, grammar, mo, np, walk_steps):
    _T = grammar["15"]["T"]
    _pi_sim = cu.stationary_dist(_T, method="simulate", steps=int(walk_steps.value), seed=0)
    _pi_true = grammar["15"]["frac"]
    _fig = go.Figure()
    _fig.add_bar(x=STATE_NAMES, y=_pi_sim, name=f"walker ({int(walk_steps.value)} steps)",
                 marker_color=STATE_COLORS)
    _fig.add_scatter(x=STATE_NAMES, y=_pi_true, name="true long-run fraction", mode="markers",
                     marker=dict(color="black", size=13, symbol="diamond-open", line=dict(width=3)))
    _fig.update_layout(template="plotly_white", height=380,
                       title="Cage-15 stationary distribution — walker vs. truth",
                       yaxis_title="fraction of time", margin=dict(l=10, r=10, t=44, b=10),
                       legend=dict(orientation="h", y=1.12))
    _err = float(np.abs(_pi_sim - _pi_true).max())
    mo.vstack([walk_steps, _fig,
               mo.md(f"*Max gap walker vs. truth: **{_err:.3f}**. More steps → the walker's tally "
                     f"snaps onto the true occupancy. The `eig` shortcut (leading left eigenvector) "
                     f"lives in the accordion below.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "🧮 Deeper: the eigen-form of the stationary distribution": mo.md(
            r"""
            The walker converges to the vector $\pi$ that is unchanged by one more step:
            $$\pi^\top T = \pi^\top,\qquad \textstyle\sum_i \pi_i = 1.$$
            That is: $\pi$ is the **left eigenvector of $T$ with eigenvalue 1**. `stationary_dist(T,
            method="eig")` returns exactly that (`np.linalg.eig(T.T)`, pick the eigenvector whose
            eigenvalue is closest to 1, normalize). The simulation and the eigenvector agree to a few
            parts in a thousand — we teach the walker because it *is* the definition; the eigenvector
            is the fast shortcut. (Requires the chain to be irreducible & aperiodic — ours is.)
            """
        )
    })
    return


# ============================================================ Entropy + stickiness vs null
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · Does the grammar beat chance? — the mandatory shuffle null

        A transition matrix always *looks* structured. To prove the structure is real, destroy the one
        thing that carries memory — **temporal order** — and recompute. `shuffle_transition_null`
        permutes the state sequence many times; each shuffle gives a matrix with the *same* state
        frequencies but *no* memory. Two statistics:

        - **Transition entropy** (bits): average uncertainty of the next state. Low = predictable
          grammar; `log₂(3) ≈ 1.58` = memoryless. The **real** entropy should fall far *below* the
          shuffle null.
        - **Self-transition ("stickiness")**: mean diagonal `P(stay)`. The real value should sit far
          *above* the shuffle null (which hovers near the chance ~0.33).

        Framed as an **n ≈ 3 case study** — the three cages are our units. (Our continuous span is a
        single *dep*-phase day, so we can **not** contrast pre/dep/post here; that honest limit is in
        the closing box.)
        """
    )
    return


@app.cell
def _(CAGE_COLORS, go, grammar, mo, np):
    from plotly.subplots import make_subplots as _msub
    _cages = ["15", "10", "13"]
    _fig = _msub(rows=1, cols=2, subplot_titles=(
        "Transition entropy (bits) — lower = more memory",
        'Self-transition "stickiness" — higher = stickier'))
    _fig.add_bar(x=[f"Cage {c}" for c in _cages], y=[grammar[c]["H"] for c in _cages],
                 marker_color=[CAGE_COLORS[c] for c in _cages], showlegend=False, row=1, col=1)
    _fig.add_bar(x=[f"Cage {c}" for c in _cages], y=[grammar[c]["self"] for c in _cages],
                 marker_color=[CAGE_COLORS[c] for c in _cages], showlegend=False, row=1, col=2)
    _nH = float(np.mean([grammar[c]["null_H"].mean() for c in _cages]))
    _nS = float(np.mean([grammar[c]["null_self"].mean() for c in _cages]))
    _fig.add_hline(y=_nH, line=dict(color="#e45756", dash="dash"), row=1, col=1,
                   annotation_text=f"shuffle null ≈ {_nH:.2f}", annotation_position="top left")
    _fig.add_hline(y=1.585, line=dict(color="#bbb", dash="dot"), row=1, col=1,
                   annotation_text="memoryless log₂3", annotation_position="bottom left")
    _fig.add_hline(y=_nS, line=dict(color="#e45756", dash="dash"), row=1, col=2,
                   annotation_text=f"shuffle null ≈ {_nS:.2f}", annotation_position="bottom left")
    _fig.update_layout(template="plotly_white", height=400, margin=dict(l=10, r=10, t=54, b=10))
    _fig.update_yaxes(range=[0, 1.7], row=1, col=1)
    _fig.update_yaxes(range=[0, 1.0], row=1, col=2)
    mo.vstack([_fig, mo.md(
        f"*All three cages: entropy **≈ {np.mean([grammar[c]['H'] for c in _cages]):.2f} bits** sits "
        f"far below the shuffle null **≈ {_nH:.2f}**, and stickiness **≈ "
        f"{np.mean([grammar[c]['self'] for c in _cages]):.2f}** sits far above **≈ {_nS:.2f}**. The "
        f"grammar carries real, robust memory — and the direction is identical across all three "
        f"cages.*")])
    return


# ============================================================ Activity clock
@app.cell
def _(cu, t10, t13, t15):
    # Bootstrapped activity-by-time-of-day for each cage (95% CI over 30-min bins).
    clocks = {c: cu.activity_by_tod(tr["speed"], tr["tod_hour"], bin_min=30, n_boot=200, seed=0)
              for c, tr in {"15": t15, "10": t10, "13": t13}.items()}
    return (clocks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6 · The activity clock — an honest n ≈ 3 case study

        `activity_by_tod` bins mean movement speed into 30-minute slots and bootstraps a 95% CI within
        each bin. The **shaded band is the dark/active phase (09:00–21:00)** under our reversed light
        cycle. The vertical marker is the **Hero Event's** time-of-day — the baton, landed on the day.

        <b>Read this as a case study, not a circadian law.</b> Three cages cannot support a population
        circadian claim; the bootstrap CI is *within-cage* sampling noise, not between-animal
        variability. We are describing *these* recordings, honestly.
        """
    )
    return


@app.cell
def _(CAGE_COLORS, clocks, go, grammar, hero_tod, mo):
    _fig = go.Figure()
    _fig.add_vrect(x0=9, x1=21, fillcolor="#000", opacity=0.06, line_width=0,
                   annotation_text="dark / active phase", annotation_position="top left")
    for _c in ["15", "10", "13"]:
        _ck = clocks[_c]; _col = CAGE_COLORS[_c]
        _rgba = "rgba(%d,%d,%d,0.15)" % tuple(int(_col[i:i + 2], 16) for i in (1, 3, 5))
        _fig.add_scatter(x=list(_ck["centers"]) + list(_ck["centers"])[::-1],
                         y=list(_ck["ci_high"]) + list(_ck["ci_low"])[::-1],
                         fill="toself", fillcolor=_rgba, line=dict(color="rgba(0,0,0,0)"),
                         hoverinfo="skip", showlegend=False)
        _fig.add_scatter(x=_ck["centers"], y=_ck["curve"], mode="lines", line=dict(color=_col, width=2),
                         name=f"Cage {_c} ({grammar[_c]['sex']})")
    _fig.add_vline(x=hero_tod, line=dict(color="#d62728", dash="dash"),
                   annotation_text=f"Hero @ {hero_tod:.1f}h", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=430,
                       title="Activity clock — mean speed by time of day (95% bootstrap CI)",
                       xaxis_title="time of day (h · reversed cycle)", yaxis_title="mean speed (px/s)",
                       margin=dict(l=10, r=10, t=44, b=10), legend=dict(orientation="h", y=1.12))
    _fig.update_xaxes(range=[0, 24], dtick=3)
    mo.vstack([_fig, mo.md(
        "*Activity concentrates in the dark/active window and the Hero Event lands squarely inside it "
        "— consistent across the three cages, but with n ≈ 3 we claim description, not inference.*")])
    return


# ============================================================ Exercise: Toolbox + Hypothesis
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 🧪 Exercise — build the grammar, then grade it against the null

        **Toolbox** (inputs → outputs)

        - `state_seq` — `(T,) int` contiguous cage-15 state sequence (0 rest · 1 locomote · 2 huddle).
        - `np.add.at(M, (rows, cols), 1)` — scatter-add: increment `M[rows[k], cols[k]]` for every `k`
          *without* a Python loop.
        - `cu.transition_matrix(state_seq, n_states)` → `(K,K)` row-stochastic — the reference answer.
        - `cu.transition_entropy(T)` → `float` bits.
        - `cu.shuffle_transition_null(state_seq, n, stat=...)` → `(n,)` null values
          (`stat="entropy"` or `"self"`).

        **Hypothesis banner** — *This cage's behavior has memory: the observed transition entropy falls
        well below a time-shuffled null, and the self-transition rate rises well above it.* (A
        pre-registered, falsifiable, direction-specified claim.)

        **Two-tier task.**
        **Tier 1 (you write it):** build the transition matrix **by counting** — use `np.add.at` to
        tally consecutive `(now, next)` pairs, then row-normalize.
        **Tier 2 (call the black boxes):** compute the observed entropy & stickiness and their shuffle
        nulls, and check the **gap**.
        """
    )
    return


@app.cell
def _(np, t15):
    # ---- YOUR TURN (Tier 1) ----------------------------------------------------------------
    # Build a row-stochastic transition matrix by COUNTING consecutive state pairs.
    # state_seq[t] -> state_seq[t+1]. Use np.add.at (no Python loop over frames).
    state_seq = t15["state_seq"]        # (T,) int, values in {0,1,2}
    K_states = 3

    def student_tmat(seq, K):
        M = np.zeros((K, K))
        # TODO: tally transitions with np.add.at(M, (seq[:-1], seq[1:]), 1)
        # TODO: row-normalize so every row sums to 1 (guard against empty rows)
        # --- replace the body below with your own ---
        np.add.at(M, (seq[:-1], seq[1:]), 1)
        row = M.sum(1, keepdims=True)
        return np.divide(M, row, out=np.zeros_like(M), where=row > 0)

    T_student = student_tmat(state_seq, K_states)
    return T_student, state_seq


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "💡 Reveal solution": mo.md(
            r"""
            ```python
            def student_tmat(seq, K):
                M = np.zeros((K, K))
                np.add.at(M, (seq[:-1], seq[1:]), 1)   # count now->next pairs, vectorized
                row = M.sum(1, keepdims=True)          # frames spent in each 'now' state
                return np.divide(M, row, out=np.zeros_like(M), where=row > 0)  # -> P(next|now)
            ```
            `np.add.at` is the un-buffered scatter-add: it adds 1 at every `(now, next)` coordinate,
            handling repeats correctly (a plain `M[rows, cols] += 1` would drop duplicates). Dividing
            each row by its total turns counts into conditional probabilities. This *is*
            `cu.transition_matrix`.

            **Tier 2:**
            ```python
            H_obs    = cu.transition_entropy(T_student)
            self_obs = np.mean(np.diag(T_student))
            null_H    = cu.shuffle_transition_null(state_seq, n=40, stat="entropy")
            null_self = cu.shuffle_transition_null(state_seq, n=40, stat="self")
            beats = (H_obs < null_H.min()) and (self_obs > null_self.max())   # robust gap, both ways
            ```
            """
        )
    })
    return


@app.cell
def _(T_student, cu, grammar, np, state_seq):
    # ---- graded self-check (Tier 1 exactness + Tier 2 robustness gap) ----
    _T_ref = cu.transition_matrix(state_seq, 3)
    tier1_ok = bool(np.allclose(T_student, _T_ref, atol=1e-9))

    H_obs = float(cu.transition_entropy(T_student))
    self_obs = float(np.mean(np.diag(T_student)))
    _null_H = grammar["15"]["null_H"]
    _null_self = grammar["15"]["null_self"]
    # robust gaps: entropy well BELOW null, stickiness well ABOVE null
    entropy_gap = float(_null_H.mean() - H_obs)     # expect ~0.74 bits
    self_gap = float(self_obs - _null_self.mean())  # expect ~0.50
    tier2_ok = bool((H_obs < _null_H.min()) and (self_obs > _null_self.max()))
    # tolerance bands pinned from the real bundle (cam15): H≈0.765, self≈0.832, null_H≈1.505
    bands_ok = bool((0.68 <= H_obs <= 0.86) and (0.79 <= self_obs <= 0.87)
                    and (entropy_gap >= 0.40) and (self_gap >= 0.40))
    return H_obs, bands_ok, entropy_gap, self_gap, self_obs, tier1_ok, tier2_ok


@app.cell(hide_code=True)
def _(H_obs, bands_ok, entropy_gap, mo, self_gap, self_obs, tier1_ok, tier2_ok):
    _pass = tier1_ok and tier2_ok and bands_ok
    _bg, _bd, _icon, _verdict = (
        ("rgba(40,170,80,0.12)", "#28aa50", "✅", "PASS")
        if _pass else ("rgba(228,87,86,0.12)", "#e45756", "❌", "CHECK YOUR CODE"))
    mo.md(
        f"""
        <div style="border:2px solid {_bd}; border-radius:10px; padding:12px 16px; background:{_bg};">
        <b>{_icon} Self-check — {_verdict}</b><br>
        <b>Tier 1</b> — your counted matrix matches <code>cu.transition_matrix</code>: <b>{tier1_ok}</b>.<br>
        <b>Tier 2</b> — observed entropy <b>{H_obs:.3f} bits</b> is below the shuffle null by
        <b>{entropy_gap:.2f} bits</b>; stickiness <b>{self_obs:.3f}</b> is above the null by
        <b>{self_gap:.2f}</b>. Robust both-ways gap: <b>{tier2_ok}</b>; within pinned bands:
        <b>{bands_ok}</b>.<br>
        <b>Graded conclusion:</b> the grammar carries <b>real memory</b> — the observed entropy and
        stickiness beat a time-shuffled null in a direction that holds across all three cages. This is
        the honest, robust signal (we grade the <i>gap vs. null</i>, never a single noisy number).
        </div>
        """
    )
    return


# ============================================================ Conceptual: observed vs hidden
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7 · Conceptual — observed Markov vs. *hidden* HMM/AR-HMM

        We just built a **fully-observed** first-order Markov chain: *we* assigned the labels
        (rest/locomote/huddle) with hand-picked thresholds, then measured transitions between them.
        A true **HMM / AR-HMM** flips the arrow — it treats the states as **hidden** and *infers* them
        from the emissions (the raw kinematics), discovering both the number of states and their
        boundaries from data. That is exactly what MoSeq's AR-HMM does to carve behavior into
        "syllables."

        **Questions to sit with:**

        1. A first-order chain assumes the next state depends only on the *current* one. What memory
           does that throw away, and how would you notice if a second-order dependency mattered?
        2. Why is the shuffle null *mandatory* rather than optional here?
        3. Our three states came from two thresholds we chose. What would a **hidden**-state model buy
           us that our observed labels can't — and how could it reveal a state we never named?
        4. Why can't the 1500 sparse 130-frame approach-events be strung into one valid chain?
        """
    )
    return


# ============================================================ Neuroscience accordion
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "🧠 Deeper: the paper & where the analogy stops": mo.md(
            r"""
            **Shared mathematics.** A transition matrix over behavioral states is the *observed* cousin
            of the **hidden-Markov / autoregressive-HMM** models neuroscientists fit to latent brain
            states — same Markov machinery (transition matrix, stationary distribution, entropy), one
            layer of inference apart.

            **The direct method ancestor.** *Wiltschko et al. 2015, Neuron* — **MoSeq** fits an AR-HMM
            to depth-video mouse pose and reads out sub-second behavioral "syllables"; *Markowitz et al.
            2018, Cell* shows the **dorsolateral striatum** encodes exactly these syllable *transitions*.
            Latent-state sequence models in neural data: *Jones et al. 2007, PNAS* (cortical state
            sequences during taste); *Mazzucato et al. 2015, J. Neurosci.* (metastable attractor states
            in cortex).

            **Species / preparation tag.** Freely-moving mice, home-cage social groups (ours); MoSeq is
            single-mouse open-field depth video; the striatal read-out is mouse electrophysiology.

            **Where the analogy stops.** Ours is a **fully-observed** chain — *we* labeled the states,
            so there is no inference, only counting. The brain's states are **latent**: they must be
            *inferred* from emissions, and **that inference is the HMM**. Matching the transition-matrix
            math does not mean our coarse kinematic states are the same objects as MoSeq syllables or as
            cortical attractor states — only that the same grammar tool reads all three.
            """
        )
    })
    return


# ============================================================ What we threw away
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8 · What we threw away / how it breaks

        **Discarded information.** The first-order assumption erases all memory beyond one step (a
        grooming→locomote→rest *bout* is invisible). Three coarse states collapse the whole 19-feature
        ethogram into rest/locomote/huddle — every fine behavior (aggression, mounting, sniffing) is
        crushed into one of three bins. Time-of-day binning throws away the exact sequence within a bin.

        **Concrete failure modes on *this* data.**

        1. **n ≈ 3, single condition.** Our continuous span is *one dep-phase day* per cage, so we
           **cannot** test whether deprivation reorganizes the grammar or flattens the clock — the
           headline lab-meeting question stays *open* until pre/post continuous spans exist. Three cages
           can't ground a circadian population claim.
        2. **Diagonal dominance.** Self-transitions (~0.83) swamp the matrix; the interesting
           off-diagonal *switches* are rare, so entropy is driven mostly by dwell length, not by
           genuine sequencing richness.
        3. **Threshold sensitivity.** The 40th/25th-percentile cutoffs *define* the states; nudge them
           and the whole grammar shifts. Our "huddle" is a distance heuristic, not verified social
           contact.

        **How would you analyze this?** *If you suspected a hidden bout structure the three observed
        labels miss — a recurring "prowl → lunge → retreat" motif — how would you detect it without
        pre-labeling it?* (Hint: this is precisely the job of an AR-HMM.)
        """
    )
    return


# ============================================================ Readout board (bottom)
@app.cell(hide_code=True)
def _(H_obs, grammar, mo, np, self_obs):
    _nH = float(np.mean([grammar[c]["null_H"].mean() for c in ["15", "10", "13"]]))
    mo.md(
        f"""
        ### 📋 Readout Board — *leaving* NB07

        | Gauge | Your fresh number | Benchmark | Verdict |
        |---|---|---|---|
        | **A · size of representation** | **3 states**, 1 label per moment — plus the **time axis** | `1` syllable (NB05) | Representation didn't shrink; we added **dynamics** on top of it. |
        | **B · held-out readiness** | grammar entropy **{H_obs:.2f} bits** vs shuffle **{_nH:.2f}**; stickiness **{self_obs:.2f}** vs **0.33** | memory beats null | ✅ **dynamics validated** — the baseline grammar is real and shuffle-robust across 3 cages. |

        The behavior team can now say what the *un-manipulated* grammar and rhythm look like — the
        prerequisite for ever claiming opto changed them.
        """
    )
    return


# ============================================================ What we ship next
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## → What we ship next

        We have the full representation now: a body-frame feature vector (NB02), a low-D manifold
        (NB04), a map carved into syllables (NB05), an honest account of what those syllables encode
        (NB06), and — today — the **grammar and clock** of how behavior moves through them, validated
        against a shuffle null.

        Everything so far *describes* behavior. The rig needs a **decision**. **Next → `08_decoder_graduates.py`:**
        the decoder gets a teacher (hand-labeled ground truth and its noise ceiling), gets trained —
        and then, for the first time, **Camera 16 unlocks** and we find out whether a readout built on
        seven cages survives a cage it has never seen. **Notebooks until unlock: 1.**
        """
    )
    return


if __name__ == "__main__":
    app.run()

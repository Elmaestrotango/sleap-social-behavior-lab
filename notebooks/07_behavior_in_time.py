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


# ============================================================ 1. Why this notebook
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 07 · Behavior in time — memory and daily rhythm

        ## Why this notebook

        Every notebook so far has treated a behavioral event as a still picture: we took one short
        clip and summarized it. But behavior unfolds *in time*. Two plain questions follow, and both
        matter before you could ever claim that some manipulation changed an animal's behavior:

        1. **Does behavior have memory?** If a mouse is resting right now, is it more likely to keep
           resting a moment later — or is each moment an independent coin flip, unaffected by what came
           just before?
        2. **Is there a daily rhythm?** When across the 24-hour day are these mice most active?

        Answering these for the **un-manipulated** animal gives you a baseline. Without a baseline you
        cannot detect a change. Measuring how behavior moves through time, and testing that structure
        against chance, is also how neuroscientists quantify behavior.

        ### Terms you will need first

        - **State** — a coarse label for what the cage is doing at one moment. We will use three:
          *rest*, *locomote* (moving, apart), and *huddle* (mice close together).
        - **Markov chain** — a sequence of states in which the next state depends only on the
          *current* state, not on the whole past. Everyday example: weather that is either "sunny" or
          "rainy", where tomorrow's odds depend only on today's weather.
        - **Transition matrix** — a table `T` where `T[i, j]` is the probability of moving to state
          `j` next, given you are in state `i` now. Each row sums to 1.
        - **Stationary distribution** — the long-run fraction of time spent in each state if the chain
          runs forever.
        - **Entropy** — a single number, in *bits*, measuring how unpredictable the next state is.
          Low entropy means the next state is easy to guess; high entropy means it is nearly random.

        The plan: label every frame of a continuous recording with a state, build the transition
        matrix, read off the stationary distribution and entropy, test them against a shuffled null,
        and finally look at the activity clock across the day.
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
        ### Readout board — entering NB07

        Two running gauges track our progress. Gauge A is the *size* of the representation (how many
        numbers describe a moment). Gauge B is *held-out readiness* (how well a decoder generalizes to
        a cage it never trained on). This is a markdown summary, not a plotly gauge.

        | Gauge | Where we are | Benchmark | Note |
        |---|---|---|---|
        | **A · size of representation** | **1 state label** per moment | `{_a_val}` | The representation stopped shrinking at NB05. NB07 does not shrink it further; it adds the **time axis**. |
        | **B · held-out readiness** | features → aggression cross-validation **{_b06_val}** AUROC | `{_b06_val}` | The target is a decoder that survives an unseen cage (**{_b08_val}**, NB08). NB07 validates *dynamics*, not a decoder. |
        """
    )
    return


# ============================================================ Held-out cage 16
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div style="border:1px solid #b08; border-radius:10px; padding:14px 18px; background:
        rgba(180,0,140,0.05);">
        <b>Held out — Camera 16</b><br>
        One cage (Camera 16, 470 events) is set aside and never used for training. It stays sealed
        until <b>NB08</b>, where the decoder is finally tested on it. The grammar we build in this
        notebook uses only the training cages.
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
    # Three continuous 24h cages at 2 fps: 15 = our example cage (M), 10 = context (F), 13 = context (M).
    t15 = cu.load_continuous_tracks("15")
    t10 = cu.load_continuous_tracks("10")
    t13 = cu.load_continuous_tracks("13")
    return ev, t10, t13, t15


@app.cell
def _(cu, ev):
    # Our example event for this notebook. Event #909 is the canonical example we have followed all
    # week (Cage 15). The grammar and activity clock below come from cage 15's continuous 24-hour
    # recording, which is a dep-phase day. Event #508 is a clean cage-15 approach event from that same
    # day, so it lands on this day's clock. contact is at frame 40; skeletons colored by rank
    # (approacher = Dom/red, approachee = Sub/green, bystander = Mid/blue).
    example_idx = 508
    _kp = ev["kp"][example_idx]
    _ranks = ev["ranks"][example_idx]
    example_key = str(ev["event_key"][example_idx])
    example_tod = float(cu.time_of_day(example_key))
    example_gif = cu.gif_img_html(
        cu.event_gif_bytes(_kp, _ranks, int(ev["contact_rel"][example_idx]), cell=200), width=200)
    return example_gif, example_idx, example_tod


# ============================================================ One event -> the day it lived in
@app.cell(hide_code=True)
def _(example_gif, example_tod, mo):
    mo.md(
        f"""
        ## 1 · From one event to the whole day

        All week we have followed one short approach event — our **example event** (Cage 15, male).
        The two interacting mice are the **approacher** and the **approachee**; a third mouse is a
        **bystander**. Skeletons are colored **only by social rank**
        (<span style="color:#d62728">Dom = red</span>,
        <span style="color:#1f77b4">Mid = blue</span>,
        <span style="color:#2ca02c">Sub = green</span>):

        {example_gif}

        This event occurred at about **{example_tod:.1f} h** (deep in the dark, active phase — see
        below). But 130 frames is a *snapshot*. To ask whether behavior has **memory** and a **daily
        rhythm**, we widen out to the whole day this event belongs to: a **continuous 24-hour
        recording of the same cage 15**, sampled at 2 fps. Later we will mark **{example_tod:.1f} h**
        on that day's activity clock, connecting the single event to its session.

        <div style="border-left:4px solid #888; padding:6px 12px; background:rgba(0,0,0,0.03);">
        <b>Why the short approach events cannot form a chain.</b> The 1500 event clips are
        <i>disconnected</i> 130-frame snippets taken from different days and cages. A Markov chain
        needs a <b>contiguous</b> sequence: the state at time <i>t</i> must be the real state that was
        actually followed by the state at <i>t+1</i>. So the grammar is built <b>only</b> from the
        continuous recording, never from the event corpus.
        </div>
        """
    )
    return


# ============================================================ Discretize -> contiguous states
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · Turning kinematics into a state sequence

        Before we can build a Markov chain we need a state label for every frame. The function
        `cu.discretize_states` does this.

        - **Purpose** — assign each moment one coarse, cage-level state.
        - **Inputs** — `speed` (T, 3) the movement speed of each mouse, and `centroids` (T, 3, 2) the
          body-center position of each mouse.
        - **Output** — `state_seq` (T,) an integer per frame, plus the list of state names.

        It applies two data-driven thresholds:

        - **rest (0)** — average mouse speed below the 40th percentile.
        - **locomote (1)** — moving, and the mice are apart.
        - **huddle (2)** — the closest pair of mice is nearer than the 25th-percentile distance
          (a proxy for social proximity).

        The result is the valid substrate for a Markov chain: a single unbroken ribbon of states,
        one frame after the next.
        """
    )
    return


@app.cell
def _(cu, np, t15):
    # Recompute states live from raw kinematics and confirm they reproduce the shipped state_seq.
    _state_live, STATE_NAMES = cu.discretize_states(t15["speed"], t15["centroids"])
    states_match = bool(np.array_equal(_state_live, t15["state_seq"]))
    STATE_COLORS = ["#9e9e9e", "#ff7f0e", "#6a3d9a"]   # rest / locomote / huddle (state colors, not mice)
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
    _check = ("The live `discretize_states` reproduces the shipped `state_seq` exactly"
              if states_match else "Mismatch — using the shipped `state_seq`")
    mo.vstack([ribbon_start, _fig,
               mo.md(f"*{_check}. The ribbon is **contiguous** — a real sequence in time, which is "
                     f"what a Markov chain requires. States: {', '.join(STATE_NAMES)}.*")])
    return


# ============================================================ Per-cage grammar (compute)
@app.cell
def _(cu, np, t10, t13, t15):
    # Build the grammar (transition matrix + summary statistics) for all three cages. Nulls at n=40:
    # the shuffle-entropy of a 172,800-point sequence is very tight (std ~1e-3), so 40 draws already
    # pin it; more draws is an optional extra.
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
    CAGE_COLORS = {"15": "#1b9e77", "10": "#d95f02", "13": "#7570b3"}   # per-cage, not per-mouse
    return CAGE_COLORS, grammar


# ============================================================ Transition matrix heatmap
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · The transition matrix

        A transition matrix summarizes the whole day's dynamics in one small table.

        **A tiny worked example first.** Suppose a cage has only two states, *rest* and *move*. If from
        *rest* the mouse stays resting 80% of the time and starts moving 20%, and from *move* it keeps
        moving 70% and settles to rest 30%, the transition matrix is:

        $$T = \begin{bmatrix} 0.8 & 0.2 \\ 0.3 & 0.7 \end{bmatrix}$$

        Row 1 reads "given resting now, next is 0.8 rest / 0.2 move." Every row sums to 1. A matrix
        with large numbers on the **diagonal** means behavior is *sticky* (long dwells in one state);
        large **off-diagonal** numbers mean it switches often.

        **The function.** `cu.transition_matrix(state_seq, n_states)`:

        - **Purpose** — estimate `T[i, j] = P(next = j | now = i)` from data.
        - **Input** — the contiguous `state_seq` and the number of states.
        - **Output** — a `(K, K)` matrix whose rows sum to 1.

        It works by counting every consecutive `(now, next)` pair and then dividing each row by its
        total. Pick a cage below to see its matrix.
        """
    )
    return


@app.cell
def _(grammar, mo):
    cage_pick = mo.ui.dropdown(
        options={f"Cage {c} ({grammar[c]['sex']})" + (" · example" if c == "15" else ""): c
                 for c in ["15", "10", "13"]},
        value="Cage 15 (M) · example", label="cage")
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
               mo.md(f"*The diagonal ({np.round(np.diag(_T),2).tolist()}) is the largest part of each "
                     f"row: behavior **stays in its current state** far more than chance (which for "
                     f"three equally likely states would be 1/3 ≈ 0.33). That stickiness is the memory "
                     f"we will test against a null.*")])
    return


# ============================================================ Stationary distribution by simulation
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · The stationary distribution

        The stationary distribution answers: *in the long run, what fraction of time is spent in each
        state?* There is a direct way to see it without any linear algebra. **Release a random
        walker** on the transition matrix: start in `rest`, use the current row `T[now]` as the
        probabilities for the next state, take a step, and repeat many thousands of times. Tally how
        often the walker lands in each state. That tally *is* the stationary distribution.

        `cu.stationary_dist(T, method="simulate", steps=...)`:

        - **Purpose** — estimate the long-run occupancy of each state.
        - **Input** — the transition matrix `T` and the number of walker steps.
        - **Output** — a vector of fractions, one per state, summing to 1.

        Drag the walk length and watch the estimate converge to the true long-run occupancy (the
        empirical state fractions, shown as open diamonds).
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
               mo.md(f"*Largest gap between walker and truth: **{_err:.3f}**. With more steps the "
                     f"walker's tally settles onto the true occupancy. A faster exact shortcut using "
                     f"an eigenvector is in the note below.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The eigenvector shortcut (optional math)": mo.md(
            r"""
            The walker converges to the vector $\pi$ that is unchanged by one more step:
            $$\pi^\top T = \pi^\top,\qquad \textstyle\sum_i \pi_i = 1.$$
            In words, $\pi$ is the **left eigenvector of $T$ with eigenvalue 1**.
            `cu.stationary_dist(T, method="eig")` returns exactly that: it takes `np.linalg.eig(T.T)`,
            picks the eigenvector whose eigenvalue is closest to 1, and normalizes it to sum to 1. The
            simulation and the eigenvector agree to within a few parts per thousand. We show the walker
            because it *is* the definition; the eigenvector is just the fast route to the same answer.
            (It requires the chain to be irreducible and aperiodic, which ours is.)
            """
        )
    })
    return


# ============================================================ Entropy + stickiness vs null
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · Does the grammar beat chance? The shuffle null

        A transition matrix always *looks* structured, even for random data. To show the structure is
        real, we destroy the one thing that carries memory — the **temporal order** — and recompute.
        `cu.shuffle_transition_null` permutes the state sequence many times; each shuffle keeps the
        same overall state frequencies but removes any memory. We compare two statistics to this null:

        - **Transition entropy** (bits): the average uncertainty of the next state. Lower means a more
          predictable grammar; `log₂(3) ≈ 1.58` bits is the memoryless case. The **real** entropy
          should fall well *below* the shuffle null.
        - **Self-transition, or "stickiness"**: the mean of the diagonal, `P(stay in current state)`.
          The real value should sit well *above* the shuffle null (which hovers near chance, ~0.33).

        `cu.shuffle_transition_null(state_seq, n, stat=...)`:

        - **Purpose** — build the distribution of a statistic under "no memory."
        - **Input** — the state sequence, the number of shuffles `n`, and which statistic
          (`"entropy"` or `"self"`).
        - **Output** — an array of `n` null values to compare the real number against.

        We treat this as a small **n ≈ 3 case study** — the three cages are our units. Our continuous
        recording is a single dep-phase day per cage, so we **cannot** compare pre/dep/post here; that
        honest limitation is stated in the closing section.
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
        f"*Across all three cages: entropy **≈ {np.mean([grammar[c]['H'] for c in _cages]):.2f} bits** "
        f"sits well below the shuffle null **≈ {_nH:.2f}**, and stickiness **≈ "
        f"{np.mean([grammar[c]['self'] for c in _cages]):.2f}** sits well above **≈ {_nS:.2f}**. The "
        f"grammar carries real memory, and the direction is the same in all three cages.*")])
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
        ## 6 · The activity clock

        The second question was the daily rhythm. `cu.activity_by_tod`:

        - **Purpose** — describe when across the day the animals move most.
        - **Input** — `speed` and `tod_hour` (the time of day, 0–24 h, for each frame).
        - **Output** — a curve of mean movement speed binned into 30-minute slots, with a bootstrap
          95% confidence interval within each bin.

        The shaded band marks the **dark, active phase (09:00–21:00)**: this colony runs a *reversed*
        light cycle (lights on 21:00–09:00), so the mice are active during our daytime. The vertical
        dashed line marks our example event's time of day.

        **Read this as a description, not a circadian law.** Three cages cannot support a population
        claim about circadian rhythm; the bootstrap interval reflects within-cage sampling noise, not
        variation between animals. We are describing *these* recordings, honestly.
        """
    )
    return


@app.cell
def _(CAGE_COLORS, clocks, example_tod, go, grammar, mo):
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
    _fig.add_vline(x=example_tod, line=dict(color="#d62728", dash="dash"),
                   annotation_text=f"example event @ {example_tod:.1f}h", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=430,
                       title="Activity clock — mean speed by time of day (95% bootstrap CI)",
                       xaxis_title="time of day (h · reversed cycle)", yaxis_title="mean speed (px/s)",
                       margin=dict(l=10, r=10, t=44, b=10), legend=dict(orientation="h", y=1.12))
    _fig.update_xaxes(range=[0, 24], dtick=3)
    mo.vstack([_fig, mo.md(
        "*Activity concentrates in the dark, active window, and the example event lands squarely "
        "inside it — consistent across the three cages. With n ≈ 3 this is a description, not an "
        "inference about the population.*")])
    return


# ============================================================ Exercise: build the transition matrix
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise — build the transition matrix by counting

        You will build the transition matrix yourself, the same way `cu.transition_matrix` does it,
        and then grade it against the shuffle null.

        **What you have**

        - `state_seq` — the `(T,)` contiguous cage-15 state sequence (0 rest · 1 locomote · 2 huddle).
        - `np.add.at(M, (rows, cols), 1)` — a "scatter-add": it adds 1 to `M[rows[k], cols[k]]` for
          every index `k`, all at once, with no Python loop. This is how you count many pairs quickly.

        **The idea.** A transition is a pair `(state now, state next)` = `(state_seq[t], state_seq[t+1])`.
        Count how many times each pair occurs, then divide each row by its total so the row gives
        probabilities.

        **Your task (one line to fill in).** In the cell below, replace the `____` with
        `np.add.at(M, (seq[:-1], seq[1:]), 1)`. The line above it makes the empty count table `M`, and
        the two lines below it already row-normalize for you. When you run it, the cell plots your
        matrix as a heatmap. **You should see a 3×3 grid with a strong diagonal (each diagonal cell
        around 0.8) and small off-diagonal values** — the same shape as the cage-15 matrix in Section
        3. If your diagonal is not dominant, your counting line is wrong.
        """
    )
    return


@app.cell
def _(np, t15):
    # ---- YOUR TURN -------------------------------------------------------------------------
    # Build a row-stochastic transition matrix by COUNTING consecutive state pairs.
    state_seq = t15["state_seq"]        # (T,) int, values in {0,1,2}
    K_states = 3

    def student_tmat(seq, K):
        M = np.zeros((K, K))                      # empty count table, one row/col per state
        # TODO: count every (now, next) pair into M.
        # Replace ____ with:  np.add.at(M, (seq[:-1], seq[1:]), 1)
        # seq[:-1] are the "now" states, seq[1:] are the "next" states, aligned frame by frame.
        np.add.at(M, (seq[:-1], seq[1:]), 1)      # <-- the ____ line (already filled with the answer)
        row = M.sum(1, keepdims=True)             # total frames spent in each "now" state
        return np.divide(M, row, out=np.zeros_like(M), where=row > 0)   # -> P(next | now)

    T_student = student_tmat(state_seq, K_states)
    return T_student, state_seq


@app.cell
def _(STATE_NAMES, T_student, go, mo, np):
    # Plot of YOUR matrix — compare against the described picture (strong diagonal ~0.8).
    _fig = go.Figure(go.Heatmap(
        z=T_student, x=STATE_NAMES, y=STATE_NAMES, colorscale="Blues", zmin=0, zmax=1,
        text=np.round(T_student, 2), texttemplate="%{text}", textfont=dict(size=15),
        colorbar=dict(title="P(next|now)")))
    _fig.update_layout(template="plotly_white", height=380,
                       title="Your counted transition matrix — expect a strong diagonal",
                       xaxis_title="next state", yaxis_title="current state",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_yaxes(autorange="reversed")
    mo.vstack([_fig, mo.md(
        f"*Your diagonal: {np.round(np.diag(T_student), 2).tolist()}. Each value should be near "
        f"0.8 — behavior stays put. Compare this shape to the cage-15 matrix in Section 3.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution": mo.md(
            r"""
            ```python
            def student_tmat(seq, K):
                M = np.zeros((K, K))
                np.add.at(M, (seq[:-1], seq[1:]), 1)   # count now->next pairs, vectorized
                row = M.sum(1, keepdims=True)          # frames spent in each 'now' state
                return np.divide(M, row, out=np.zeros_like(M), where=row > 0)  # -> P(next|now)
            ```
            `np.add.at` is an un-buffered scatter-add: it adds 1 at every `(now, next)` coordinate and
            handles repeats correctly (a plain `M[rows, cols] += 1` would drop duplicate coordinates).
            Dividing each row by its total turns counts into conditional probabilities. This is exactly
            what `cu.transition_matrix` computes.

            **Grade it against the null (already run for you in the self-check below):**
            ```python
            H_obs    = cu.transition_entropy(T_student)
            self_obs = np.mean(np.diag(T_student))
            null_H    = cu.shuffle_transition_null(state_seq, n=40, stat="entropy")
            null_self = cu.shuffle_transition_null(state_seq, n=40, stat="self")
            beats = (H_obs < null_H.min()) and (self_obs > null_self.max())   # gap in both directions
            ```
            """
        )
    })
    return


@app.cell
def _(T_student, cu, grammar, np, state_seq):
    # ---- graded self-check (exact match + robust gap vs the shuffle null) ----
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
        ("rgba(40,170,80,0.12)", "#28aa50", "PASS", "PASS")
        if _pass else ("rgba(228,87,86,0.12)", "#e45756", "CHECK", "CHECK YOUR CODE"))
    mo.md(
        f"""
        <div style="border:2px solid {_bd}; border-radius:10px; padding:12px 16px; background:{_bg};">
        <b>Self-check — {_verdict}</b><br>
        <b>Counting</b> — your matrix matches <code>cu.transition_matrix</code>: <b>{tier1_ok}</b>.<br>
        <b>Beats the null</b> — observed entropy <b>{H_obs:.3f} bits</b> is below the shuffle null by
        <b>{entropy_gap:.2f} bits</b>; stickiness <b>{self_obs:.3f}</b> is above the null by
        <b>{self_gap:.2f}</b>. Robust gap in both directions: <b>{tier2_ok}</b>; within the pinned
        tolerance bands: <b>{bands_ok}</b>.<br>
        <b>Conclusion:</b> the grammar carries <b>real memory</b> — the observed entropy and stickiness
        beat a time-shuffled null in a direction that holds across all three cages. We grade the
        <i>gap versus the null</i>, never a single noisy number.
        </div>
        """
    )
    return


# ============================================================ Conceptual: observed vs hidden
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7 · Observed states vs. hidden states

        We built a **fully-observed** first-order Markov chain: *we* assigned the labels
        (rest / locomote / huddle) using thresholds we chose, then counted transitions between them.
        There was no inference — only counting.

        A more powerful family of models, the **hidden Markov model (HMM)** and its autoregressive
        cousin (**AR-HMM**), turns the problem around. Instead of being told the states, the model
        treats them as **hidden** and *infers* them directly from the raw kinematics — discovering
        both how many states there are and where their boundaries fall. This is the idea behind
        behavior-segmentation tools that carve movement into recurring "syllables," and it is also how
        neuroscientists model latent states they cannot observe directly.

        **Questions to think about:**

        1. A first-order chain assumes the next state depends only on the *current* state. What memory
           does that throw away, and how might you notice if a two-step dependency mattered?
        2. Why is the shuffle null necessary here rather than optional?
        3. Our three states came from two thresholds we picked. What could a **hidden**-state model
           give us that our hand-chosen labels cannot — and how might it reveal a state we never named?
        4. Why can the 1500 short approach events not be strung into one valid Markov chain?
        """
    )
    return


# ============================================================ What we threw away
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8 · What we simplified, and how it can break

        **What we discarded.** The first-order assumption erases all memory beyond one step, so a
        multi-step *bout* (grooming → locomote → rest) is invisible. Three coarse states collapse a
        rich behavioral repertoire into rest / locomote / huddle, so fine behaviors (aggression,
        mounting, sniffing) all fall into one of three bins. Time-of-day binning discards the exact
        order of events within a bin.

        **Concrete limitations on this data.**

        1. **n ≈ 3, single condition.** The continuous recording is *one dep-phase day* per cage, so
           we **cannot** test whether deprivation reorganizes the grammar or flattens the clock. That
           question stays open until continuous pre/post recordings exist. Three cages also cannot
           ground a population circadian claim.
        2. **Diagonal dominance.** Self-transitions (~0.83) make up most of the matrix, so the
           interesting off-diagonal *switches* are rare, and entropy is driven mostly by how long each
           dwell lasts rather than by rich sequencing.
        3. **Threshold sensitivity.** The 40th / 25th-percentile cutoffs *define* the states; move them
           and the whole grammar shifts. "Huddle" is a distance heuristic, not verified social contact.

        **A question to carry forward.** If you suspected a repeating bout structure that the three
        observed labels miss — say a "approach → contact → retreat" motif — how would you detect it
        *without* labeling it by hand first? That is exactly the job a hidden-state model (AR-HMM) is
        built for.
        """
    )
    return


# ============================================================ Readout board (bottom)
@app.cell(hide_code=True)
def _(H_obs, grammar, mo, np, self_obs):
    _nH = float(np.mean([grammar[c]["null_H"].mean() for c in ["15", "10", "13"]]))
    mo.md(
        f"""
        ### Readout board — leaving NB07

        | Gauge | Your fresh number | Benchmark | Verdict |
        |---|---|---|---|
        | **A · size of representation** | **3 states**, 1 label per moment — plus the **time axis** | `1` state label (NB05) | The representation did not shrink; we added **dynamics** on top of it. |
        | **B · held-out readiness** | grammar entropy **{H_obs:.2f} bits** vs shuffle **{_nH:.2f}**; stickiness **{self_obs:.2f}** vs **0.33** | memory beats the null | Dynamics validated — the baseline grammar is real and shuffle-robust across 3 cages. |

        We can now describe what the *un-manipulated* grammar and daily rhythm look like — the
        prerequisite for ever claiming a manipulation changed them.
        """
    )
    return


# ============================================================ What we do next
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we do next

        We now have a full representation of behavior: a body-frame feature vector (NB02), a low-D
        manifold (NB04), a map carved into syllables (NB05), an account of what those syllables encode
        (NB06), and — today — the **grammar and daily clock** of how behavior moves through them,
        validated against a shuffle null.

        Everything so far *describes* behavior. Next we ask for a **decision**. In
        **`08_decoder_graduates.py`** the decoder gets a labeled training set (hand-scored ground truth
        and its noise ceiling), is trained, and is then tested on **Camera 16** — a cage it has never
        seen — to find out whether a readout built on seven cages generalizes to an eighth.
        """
    )
    return


if __name__ == "__main__":
    app.run()

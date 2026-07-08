# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "plotly>=5.20",
#     "h5py>=3.9",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB13 · Social Ethograms — the behavioral half of the neural dataset

        > **WEEK 2 — THE NEURAL TWIN** &nbsp;·&nbsp; *mirrors NB03/NB05 (reading behavioral states in time)*
        >
        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > We are about to decode calcium from a real social-isolation experiment (that's NB14). Before
        > we touch a single neuron, we need to read the **behavior** for the *same mice, same sessions* —
        > and we need it in exactly the form Week 1 taught you to produce: **discrete behavioral states,
        > laid out in time.** In Week 1 you built ethograms out of *pose* (NB03/NB05 turned the 19
        > features into per-frame states). Here the states are already hand/model-scored for you:
        > nine social-contact channels per session. **Same computational move — a stack of boolean
        > state-channels over a time axis — on the dataset whose brain we read next.**
        >
        > **The deliverable:** for each session, the `(9, T)` ethogram + its social-time fraction.
        > **It unblocks:** NB14 — every neuron there is aligned to *these* labels.
        > **Today's lab-meeting question:** *Is the `is_social` channel really the documented OR of
        > sender + receiver, and does time-spent-social change with isolation length?*

        Two readouts, one animal. Today you read the **behavioral** half; NB14 reads the **neural**
        half and asks whether individual cells track these very channels.
        """
    )
    return


@app.cell
def _(ROOT, np, nu):
    import pandas as pd
    import h5py

    # --- Behavior-only load (light + molab-safe) -----------------------------------------------
    # The task points at nu.load_si(). That helper is correct, but it EAGERLY loads the 250 MB
    # calcium.00.h5 (~280 MB into RAM, ~3 s) which NB13 never touches — NB14 is where the neural
    # half comes in. So here we fetch the SAME two files through nu's cached downloaders + constants
    # (entrances xlsx + social_bouts.00.h5) and parse them, staying light. NB14 calls the full
    # nu.load_si() to bring in calcium.
    _ent_path = nu.fetch_gdrive(nu.SI_ENTRANCES_ID, nu.SI_ENTRANCES_NAME, ROOT)
    _bouts_path = nu.fetch_gdrive(nu.SI_BOUTS_ID, nu.SI_BOUTS_NAME, ROOT)

    entrances = pd.read_excel(_ent_path)
    n_sessions = len(entrances)
    session_keys = list(nu.SI_SESSION_KEYS)
    behavior_fps = nu.BEHAVIOR_FPS

    behavior = []
    with h5py.File(_bouts_path, "r") as _f:
        for _s in range(n_sessions):
            _g = _f[f"session_{_s}"]
            behavior.append({_k: _g[_k][:].astype(bool) for _k in session_keys})

    # Condition label per session ("control" / "24hr" / "7d") via nu's canonical mapper.
    conditions = [nu.si_condition_label(v) for v in entrances["Isolation Length"]]
    lengths = np.array([len(behavior[s]["is_social"]) for s in range(n_sessions)])
    social_frac = np.array([behavior[s]["is_social"].mean() for s in range(n_sessions)])

    # Isolation-severity color ramp: group-housed control -> 24 h -> 7 d.
    COND_COLORS = {"control": "#4c78a8", "24hr": "#f58518", "7d": "#e45756"}
    COND_ORDER = ["control", "24hr", "7d"]
    return (COND_COLORS, COND_ORDER, behavior, behavior_fps, conditions,
            entrances, lengths, n_sessions, pd, session_keys, social_frac)


@app.cell(hide_code=True)
def _(mo, n_sessions):
    mo.md(
        f"""
        ---
        ## 1. The sessions — a reintroduction assay

        The **SI3_2022** dataset is a classic social-isolation design: a focal mouse is either
        **group-housed (control)** or isolated for **24 hours** or **7 days**, then reintroduced to a
        social partner while its striatal calcium is imaged. Each row below is one **session**
        ({n_sessions} total). `Int_Entry` is the frame the partner is introduced; `Isolation Length` is the
        experience the focal mouse carried in.

        The behavior lives on a **25 fps** clock (`behavior_fps`); the calcium NB14 reads is **30 fps** —
        keep that in mind, it's why NB14 has to resample. For now we just read the behavior.
        """
    )
    return


@app.cell
def _(COND_COLORS, conditions, entrances, go, lengths, np, social_frac):
    # Entrances table as a house-style plotly table, colored by condition.
    _sess = np.arange(len(entrances))
    _iso = list(entrances["Isolation Length"].astype(str))
    _entry = list(entrances["Int_Entry"].astype(int))
    _rowcol = [COND_COLORS[c] for c in conditions]
    # very light tint per row for the body cells
    def _tint(hexc):
        _h = hexc.lstrip("#")
        _r, _g, _b = int(_h[0:2], 16), int(_h[2:4], 16), int(_h[4:6], 16)
        _f = 0.85
        return f"rgb({int(_r+(255-_r)*_f)},{int(_g+(255-_g)*_f)},{int(_b+(255-_b)*_f)})"
    _fill = [[_tint(c) for c in _rowcol]]
    _tbl = go.Figure(go.Table(
        columnwidth=[0.8, 1.6, 1.2, 1.2, 1.1, 1.4],
        header=dict(values=["<b>session</b>", "<b>Isolation Length</b>", "<b>condition</b>",
                            "<b>Int_Entry (frame)</b>", "<b>length (frames)</b>",
                            "<b>social fraction</b>"],
                    fill_color="#2f3b52", font=dict(color="white", size=13), align="left",
                    height=30),
        cells=dict(values=[_sess, _iso, conditions, _entry, lengths,
                           [f"{v:.3f}" for v in social_frac]],
                   fill_color=_fill * 6, align="left", height=26,
                   font=dict(color="#222", size=12))))
    _tbl.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=30, b=10),
                       title="SI3_2022 sessions — 6 control · 6 × 24 h · 6 × 7 d")
    _tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. The nine channels — sender, receiver, and the `is_social` OR

        Each session ships **nine boolean channels**, one value per frame. They come in a
        **who-is-acting** structure — the focal mouse can be the one *doing* a social act (**sender**)
        or the one *receiving* it (**receiver**):

        | channel | meaning | side |
        |---|---|---|
        | `is_touching` | focal mouse's nose is on the partner's body | **sender** |
        | `is_ag_sniffing` | focal mouse is anogenital-sniffing the partner | **sender** |
        | `is_of_sniffing` | focal mouse is oro-facial (nose/face) sniffing the partner | **sender** |
        | `is_touched` | the partner's nose is on the focal mouse | **receiver** |
        | `is_ag_sniffed` | focal mouse is being anogenital-sniffed | **receiver** |
        | `is_of_sniffed` | focal mouse is being oro-facially sniffed | **receiver** |

        and three **derived** channels that pool those:

        $$
        \texttt{is\_social\_sender} = \texttt{is\_touching} \lor \texttt{is\_ag\_sniffing} \lor \texttt{is\_of\_sniffing}
        $$
        $$
        \texttt{is\_social\_receiver} = \texttt{is\_touched} \lor \texttt{is\_ag\_sniffed} \lor \texttt{is\_of\_sniffed}
        $$
        $$
        \boxed{\;\texttt{is\_social} = \texttt{is\_social\_sender} \lor \texttt{is\_social\_receiver}\;}
        $$

        That boxed identity is the definition we will **verify** in the exercise — it should hold
        exactly, frame-by-frame. Read the ethogram below like a Week-1 state plot: nine rows of
        on/off, stacked over time. Drag the session slider.
        """
    )
    return


@app.cell
def _(mo, n_sessions):
    session_sel = mo.ui.slider(0, n_sessions - 1, value=5, step=1,
                               label="session", debounce=True, full_width=True)
    return (session_sel,)


@app.cell
def _(COND_COLORS, behavior, behavior_fps, conditions, go, mo, np,
      session_sel, session_keys, social_frac):
    _s = int(session_sel.value)
    _d = behavior[_s]

    # Stack the nine channels into a (9, T) ethogram, in canonical key order.
    _etho = np.stack([_d[k] for k in session_keys], axis=0).astype(float)   # (9, T)
    _T = _etho.shape[1]

    # Max-pool the time axis for display so thin bouts survive downsampling (vectorized, no loop).
    _target = 1600
    _k = max(1, _T // _target)
    _Tt = (_T // _k) * _k
    _disp = _etho[:, :_Tt].reshape(9, _Tt // _k, _k).max(axis=2)
    _tsec = (np.arange(_disp.shape[1]) * _k) / behavior_fps

    _cond = conditions[_s]
    _eth = go.Figure(go.Heatmap(
        z=_disp, x=_tsec, y=session_keys,
        colorscale=[[0.0, "#f5f7fa"], [1.0, COND_COLORS[_cond]]],
        showscale=False, xgap=0, ygap=1,
        hovertemplate="%{y}<br>t=%{x:.1f}s<br>%{z:.0f}<extra></extra>"))
    _eth.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=50, b=40),
                       title=f"session {_s} · {_cond} · ethogram (max-pooled ×{_k})")
    _eth.update_xaxes(title="time (s)")
    # emphasize the is_social summary row
    _eth.add_shape(type="rect", xref="paper", yref="y",
                   x0=0, x1=1, y0=3.5, y1=4.5, line=dict(color="#222", width=1.5))

    # This session's per-channel occupancy (fraction of frames ON), horizontal bars.
    _frac = np.array([_d[k].mean() for k in session_keys])
    _bar = go.Figure(go.Bar(
        x=_frac, y=session_keys, orientation="h",
        marker_color=["#9aa7bd"] * len(session_keys),
        text=[f"{v:.3f}" for v in _frac], textposition="outside"))
    # recolor is_social with the condition color
    _cols = ["#9aa7bd"] * len(session_keys)
    _cols[session_keys.index("is_social")] = COND_COLORS[_cond]
    _bar.data[0].marker.color = _cols
    _bar.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=50, b=40),
                       title=f"session {_s} · fraction of time ON (is_social = {social_frac[_s]:.3f})",
                       xaxis_title="fraction of frames")
    _bar.update_xaxes(range=[0, max(0.25, float(_frac.max()) * 1.25)])

    mo.vstack([session_sel, mo.hstack([_eth, _bar], widths=[1.6, 1.0])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two things jump out of almost every session. **`is_touching` dominates** — nose-to-body
        contact is the bulk of social time — so `is_social_sender` (which pools it) tracks `is_social`
        closely. And the **sniff channels are sparse**, with anogenital-*sniffed* the rarest of all.
        The `is_social` row (boxed) is just the union lighting up whenever *any* channel below it is on.

        ---
        ## 3. One channel across all sessions

        Now flip the view: pick a single channel and see its occupancy in **every** session, colored by
        isolation condition. This is the cross-session structure NB14 will try to explain neurally.
        """
    )
    return


@app.cell
def _(mo, session_keys):
    chan_pick = mo.ui.dropdown(options=session_keys, value="is_social",
                               label="channel to compare across sessions")
    return (chan_pick,)


@app.cell
def _(COND_COLORS, COND_ORDER, behavior, chan_pick, conditions, go, mo,
      n_sessions, np):
    _key = chan_pick.value
    _frac = np.array([behavior[s][_key].mean() for s in range(n_sessions)])
    _sess = np.arange(n_sessions)
    _fig = go.Figure()
    for _c in COND_ORDER:
        _m = np.array([conditions[s] == _c for s in range(n_sessions)])
        _fig.add_bar(x=_sess[_m], y=_frac[_m], name=_c, marker_color=COND_COLORS[_c])
    _fig.update_layout(template="plotly_white", height=420, barmode="group",
                       margin=dict(l=10, r=10, t=50, b=40),
                       title=f"'{_key}' — fraction of time ON, by session",
                       xaxis_title="session", yaxis_title="fraction of frames",
                       legend=dict(orientation="h", y=1.08))
    _fig.update_xaxes(dtick=1)
    mo.vstack([chan_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Does isolation change social time?

        Group the per-session social fraction by condition. The **honest** expectation, and the neural
        twin, is subtle: the canonical circuit result is that isolation *raises* social **drive**
        (Matthews et al. 2016), and in NB14 you'll see isolation *lowers* the **count of social-tuned
        neurons**. Here we ask the plainest behavioral question — total time spent social — with only
        **six sessions per condition**. Pick a channel and read the boxes with a Kruskal–Wallis test
        annotated. Grade what the data actually says, not what you hope.
        """
    )
    return


@app.cell
def _(behavior, mo, session_keys):
    cond_chan = mo.ui.dropdown(options=session_keys, value="is_social",
                               label="channel for the condition comparison")
    return (cond_chan,)


@app.cell
def _(COND_COLORS, COND_ORDER, behavior, cond_chan, conditions, go, mo,
      n_sessions, np):
    from scipy.stats import kruskal
    _key = cond_chan.value
    _frac = np.array([behavior[s][_key].mean() for s in range(n_sessions)])
    _groups = {c: _frac[[conditions[s] == c for s in range(n_sessions)]] for c in COND_ORDER}
    _H, _p = kruskal(*[_groups[c] for c in COND_ORDER])
    _means = {c: float(_groups[c].mean()) for c in COND_ORDER}

    _fig = go.Figure()
    for _c in COND_ORDER:
        _fig.add_box(y=_groups[_c], name=f"{_c}", marker_color=COND_COLORS[_c],
                     boxpoints="all", pointpos=0, jitter=0.4, line=dict(width=1.5),
                     hovertemplate="%{y:.3f}<extra></extra>")
    _txt = (f"'{_key}' by condition   ·   means: "
            + "  ".join(f"{c}={_means[c]:.3f}" for c in COND_ORDER)
            + f"   ·   Kruskal–Wallis p = {_p:.3f}")
    _fig.update_layout(template="plotly_white", height=440, showlegend=False,
                       margin=dict(l=10, r=10, t=50, b=40), title=_txt,
                       yaxis_title="fraction of frames ON")
    mo.vstack([cond_chan, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For `is_social` the means fall **control ≈ 0.155 → 24 h ≈ 0.148 → 7 d ≈ 0.139** — a small,
        monotone decline — but the Kruskal–Wallis test is **not significant** (p ≈ 0.81, n = 6 per
        group). That is the honest read: with this many sessions the behavioral social-time signal is a *whisper*,
        not a shout. Hold that number. NB14 will show the isolation effect surfaces more clearly in the
        **neural** readout (fewer social-tuned cells) than in raw behavioral time — the twin's two
        halves disagree in strength, and noticing that is the point.

        ---
        ## 5. Exercise — verify the definition, then read the isolation effect honestly
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Hypothesis banner.** *The `is_social` channel is exactly the documented OR of the sender and
        receiver channels — frame-for-frame, in every session — and time-spent-social declines only
        weakly across isolation conditions.*

        **Toolbox.**

        - `behavior` — list of `n_sessions` dicts; each `behavior[s][key]` is a `(T,)` boolean array.
        - `session_keys` — the nine channel names; `conditions[s]` ∈ {`"control"`, `"24hr"`, `"7d"`}.
        - `social_frac` — per-session `is_social` fraction (already computed for you to check against).
        - `np.logical_or`, boolean array `.mean()`.

        **Your job (three numbers).**

        1. **`max_mismatch`** — across *all* sessions, the largest frame-fraction where
           `is_social` disagrees with `is_social_sender | is_social_receiver`. If the definition holds,
           this is **0**.
        2. **`mean_frac`** — the mean over sessions of each session's `is_social` fraction (sanity that
           you computed a *fraction*, not a frame *count*).
        3. **`control_minus_7d`** — mean `is_social` fraction of the **control** sessions minus that of
           the **7d** sessions (the honest isolation effect; expected small and positive).

        Fill in the three values in the next cell, then run the self-check.
        """
    )
    return


@app.cell
def _(behavior, conditions, n_sessions, np, session_keys):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    # 1. Elementwise verification of  is_social == is_social_sender | is_social_receiver
    _mism = []
    for _s in range(n_sessions):
        _d = behavior[_s]
        _defn = np.logical_or(_d["is_social_sender"], _d["is_social_receiver"])
        _mism.append(float((_d["is_social"] != _defn).mean()))
    max_mismatch = float(np.max(_mism))

    # 2. Mean over sessions of the per-session is_social fraction
    _fr = np.array([behavior[_s]["is_social"].mean() for _s in range(n_sessions)])
    mean_frac = float(_fr.mean())

    # 3. Honest isolation effect: control minus 7d social fraction
    _ctrl = _fr[[conditions[_s] == "control" for _s in range(n_sessions)]]
    _iso7 = _fr[[conditions[_s] == "7d" for _s in range(n_sessions)]]
    control_minus_7d = float(_ctrl.mean() - _iso7.mean())
    # ---------------------------------------------------------------------------------------------
    return control_minus_7d, max_mismatch, mean_frac


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            import numpy as np
            # 1. definition check
            mism = []
            for s in range(n_sessions):
                d = behavior[s]
                defn = np.logical_or(d["is_social_sender"], d["is_social_receiver"])
                mism.append((d["is_social"] != defn).mean())
            max_mismatch = float(np.max(mism))               # -> 0.0 exactly

            # 2. mean social fraction across sessions
            fr = np.array([behavior[s]["is_social"].mean() for s in range(n_sessions)])
            mean_frac = float(fr.mean())                     # -> ~0.147

            # 3. isolation effect
            ctrl = fr[[conditions[s] == "control" for s in range(n_sessions)]]
            iso7 = fr[[conditions[s] == "7d" for s in range(n_sessions)]]
            control_minus_7d = float(ctrl.mean() - iso7.mean())   # -> ~+0.016
            ```

            **What you should find.** The definition holds **exactly** (`max_mismatch == 0`): the
            file's `is_social` really is the OR of the sender and receiver channels, so you can trust it
            downstream. Mean social time is about **15% of frames**. And the isolation effect is a small
            positive **~+0.016** — control mice spend *slightly* more time social than 7-day-isolated
            mice, but with six sessions per group this is **not** significant (Section 4's Kruskal–Wallis
            p ≈ 0.81). The graded-correct answer is the *honest small effect*, not a dramatic one.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(control_minus_7d, max_mismatch, mean_frac, mo):
    # Honest self-check with tolerance bands pinned from the real data:
    #   max_mismatch  == 0            (definition is exact) -> band < 1e-6
    #   mean_frac     ~ 0.147         -> band [0.10, 0.20]  (a fraction, not a count)
    #   control_minus_7d ~ +0.016     -> honest SMALL effect, band [-0.05, 0.10]
    _p1 = float(max_mismatch) < 1e-6
    _p2 = 0.10 <= float(mean_frac) <= 0.20
    _p3 = -0.05 < float(control_minus_7d) < 0.10
    _ok = _p1 and _p2 and _p3
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = ("✅ is_social == sender | receiver exactly (max mismatch = "
           f"{max_mismatch:.1e})" if _p1 else
           f"❌ max_mismatch = {max_mismatch:.3e} — the OR definition should hold frame-for-frame")
    _m2 = (f"✅ mean social fraction = {mean_frac:.3f} (~15% of frames)" if _p2 else
           f"❌ mean_frac = {mean_frac:.3f} — expected ~0.147; did you compute a count instead of a fraction?")
    _m3 = (f"✅ control − 7d = {control_minus_7d:+.3f} — a small, honest decline; isolation barely moves "
           "behavioral social time (n.s. at n=6/group)"
           if _p3 else
           f"❌ control_minus_7d = {control_minus_7d:+.3f} is outside the honest band [−0.05, 0.10]")
    _head = "PASS — definition verified, effect read honestly" if _ok else "Not yet — fix the flagged line"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>{_m3}<br>
        <span style="font-size:0.9em;color:#555">The isolation effect is graded as a <i>small</i> effect
        on purpose — the exercise is scored against the honest null, not against noise. Tolerance bands:
        max_mismatch &lt; 1e-6 · mean_frac ∈ [0.10, 0.20] · control−7d ∈ [−0.05, 0.10].</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Deeper: the paradigm, the paper & where the analogy stops": mo.md(
            r"""
            **The social-isolation paradigm.** Isolate a social animal, then reintroduce a partner and
            watch what changes. The canonical circuit result is **Matthews et al. 2016, *Cell* 164:617**
            — dorsal-raphe dopamine neurons encode a *loneliness-like* state, and acute isolation
            **increases** social approach on reintroduction. Related: **Tomova et al. 2020, *Nat.
            Neurosci.*** (midbrain social-craving signal in humans). The SI3_2022 striatal-imaging
            dataset used here follows the same isolate-then-reintroduce logic, with the nine
            contact channels standing in for hand-scored ethogram states.

            **The shared computational move (the twin).** An **ethogram** is a stack of discrete
            behavioral-state channels over a time axis — exactly what Week 1's NB03/NB05 built out of
            pose (per-frame states from the 19 features). Here the states are pre-scored, but the object
            is identical: a `(n_states, T)` boolean matrix you read as a raster. NB14 aligns neural
            activity to *these same rows*.

            **Where the analogy stops.** (1) These channels are **scored labels** for a specific
            *dyadic reintroduction* assay, not the free homecage behavior of Week 1 — the vocabulary is
            narrower (touch + two sniff types), and `is_touching` swamps everything. (2) The behavioral
            isolation effect here is a **whisper** (control−7d ≈ +0.016, n.s. at n=6/group) — do **not**
            oversell it, and note the *direction* (control > 7d, slightly *less* social after isolation)
            runs opposite to Matthews' increased-approach result, likely because "total contact time in
            a fixed session" is a different measurement than "latency/probability of approach."
            (3) Striatal calcium (NB14) is one region in one imaging plane; a whole-brain loneliness
            circuit it is not.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## The twin, closed

        You read the **behavioral half** of the neural dataset in exactly the Week-1 currency:
        a `(9, T)` ethogram per session — discrete social states stacked over time — and you
        **verified its definition** (`is_social` = sender ∨ receiver, exactly) before trusting it
        downstream. You also measured the plain behavioral isolation effect and found it **honestly
        small**.

        **Next (NB14): read the other half.** Same mice, same sessions — now the calcium. We'll load the
        250 MB imaging file (the part `nu.load_si` pulls in), resample it from 30 fps onto *this* 25 fps
        behavior clock, and ask the question this notebook set up: **do individual striatal neurons track
        the `is_social` channel, and does isolation change how many of them do?** The behavioral labels
        you just validated are the ground truth every one of those neurons is scored against.
        """
    )
    return


if __name__ == "__main__":
    app.run()

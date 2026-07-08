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
        # NB02 · The Body's-Eye View

        > **FROM: Circuit Team → TO: Behavior Team**
        >
        > Arena coordinates are useless to us. A fight in the top-left corner and a fight in the
        > bottom-right corner are the *same behavior*, but their raw pixel coordinates share nothing.
        > Before we time-align a single opto trial, **re-express every event the way the brain
        > does — relative to the animal itself.** Ship us a compact, arena-invariant description of
        > the social geometry: one vector per event that a fight looks like no matter where, or which
        > way, it happens.
        >
        > **The deliverable:** the 19 body-frame features, `X (1500, 19)`.
        > **It unblocks:** every downstream analysis (PCA, the map, the decoder) — they all read `X`, not pixels.
        > **Today's lab-meeting question:** *After we strip out where-in-the-cage and which-way-facing,
        > what social geometry is left — and does aggression arrive from a different **direction**?*

        You just watched (NB01) that the pose is trustworthy. Now you choose a **point of view**. The
        same choice the brain makes: retrosplenial and parietal cortex convert a self-centered view
        into a stable frame, while place, grid, and head-direction cells hold the world-anchored map
        at the other end. Today you build the self-centered half by hand.
        """
    )
    return


@app.cell
def _(ROOT, cu, np):
    import pandas as pd
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))
    der = cu.load_derived("train", ROOT)
    kp = ev["kp"].astype(np.float32)
    X = der["X"]
    agg = ev["agg_label"].astype(int)
    ranks = ev["ranks"]
    contact = ev["contact_rel"].astype(int)
    cage = der["cage"]
    feat_names = [str(f) for f in cu.FEATURE_NAMES]

    # Hero Event: the design doc names #742, but in the SHIPPED bundle index 742 is a cage-12,
    # non-aggression event. We use the cleanest cage-15 (male) aggression event instead: idx 909.
    HERO = 909
    hero_hex = tuple(cu.RANK_HEX.get(int(r), cu.RANK_HEX[0]) for r in ranks[HERO])

    _board = pd.read_csv(cu.data_path("data/readout_board.csv", ROOT))
    board = _board
    return HERO, X, agg, board, cage, contact, ev, feat_names, hero_hex, kp, ranks


@app.cell(hide_code=True)
def _(board, go):
    # Readout Board helper — two gauges. Gauge A = size of the representation (falls through Phase 1);
    # Gauge B = held-out readiness (rises in Phase 2, still 0 here). Degrades gracefully if a row is
    # missing from readout_board.csv.
    def readout_fig(gauge_a_value, title):
        def _bench(gauge, nb):
            _m = board[(board["gauge"] == gauge) & (board["notebook"] == nb)]
            return float(_m["value"].iloc[0]) if len(_m) else None
        a_bench = _bench("A", "NB02")           # 19 features
        raw_bench = _bench("A", "NB01")          # 11,700 raw numbers
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+delta", value=gauge_a_value,
            number={"valueformat": ",.0f", "suffix": " numbers"},
            delta={"reference": raw_bench or 11700, "relative": False, "valueformat": ",.0f"},
            title={"text": "<b>Gauge A</b><br><span style='font-size:0.8em'>size of representation "
                           f"(was {int(raw_bench or 11700):,} raw)</span>"},
            domain={"row": 0, "column": 0}))
        fig.add_trace(go.Indicator(
            mode="number", value=0,
            number={"valueformat": ".0f", "suffix": "  (rises in Phase 2)"},
            title={"text": "<b>Gauge B</b><br><span style='font-size:0.8em'>held-out readiness — "
                           "not started</span>"},
            domain={"row": 0, "column": 1}))
        fig.update_layout(grid={"rows": 1, "columns": 2, "pattern": "independent"},
                          template="plotly_white", height=170, title=title,
                          margin=dict(l=20, r=20, t=60, b=10))
        return fig
    return (readout_fig,)


@app.cell(hide_code=True)
def _(X, readout_fig):
    # student's freshly-computed Gauge A number = the width of the feature matrix they will build.
    readout_fig(float(X.shape[1]), "Readout Board — start of NB02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. The rotation toy — face the mouse up, by hand

        The whole trick is two moves: **translate** so the focal mouse's tail-base (TTI) sits at the
        origin, then **rotate** so its heading (TTI → head) points straight up, at **+y**. Do the
        second move by hand first.

        Below is a little 5-point mouse pointing in some arbitrary arena direction. Drag the slider to
        **spin the paper** until it faces up. There's no algebra here — you're discovering the heading
        angle the way you'd turn a map to match the road. (The 2×2 rotation matrix that does this
        automatically is in the accordion.)
        """
    )
    return


@app.cell
def _(mo):
    toy_angle = mo.ui.slider(-180, 180, value=0, step=1,
                             label="your rotation β (degrees)", debounce=True, full_width=True)
    return (toy_angle,)


@app.cell
def _(go, mo, np, toy_angle):
    # A 5-node toy "mouse" in local coords (facing +x), placed at a hidden arena heading.
    _shape = np.array([[2.0, 0.0], [1.0, 0.0], [-1.2, 0.0],   # nose, head, TTI
                       [1.3, 0.55], [1.3, -0.55]])            # L_ear, R_ear
    _a0 = np.deg2rad(37.0)                                    # hidden arena heading
    _R0 = np.array([[np.cos(_a0), -np.sin(_a0)], [np.sin(_a0), np.cos(_a0)]])
    _arena = _shape @ _R0.T
    _b = np.deg2rad(toy_angle.value)
    _Rb = np.array([[np.cos(_b), -np.sin(_b)], [np.sin(_b), np.cos(_b)]])
    _pts = _arena @ _Rb.T
    # heading after the student's rotation (TTI[2] -> head[1]); target = +y (90 deg)
    _h = _pts[1] - _pts[2]
    _ang = np.rad2deg(np.arctan2(_h[1], _h[0]))
    _resid = ((_ang - 90.0 + 180) % 360) - 180
    _ok = abs(_resid) < 4.0
    _fig = go.Figure()
    _edges = [(0, 1), (1, 2), (1, 3), (1, 4)]
    for _u, _v in _edges:
        _fig.add_scatter(x=[_pts[_u, 0], _pts[_v, 0]], y=[_pts[_u, 1], _pts[_v, 1]],
                         mode="lines", line=dict(color="#d62728", width=3), showlegend=False)
    _fig.add_scatter(x=_pts[:, 0], y=_pts[:, 1], mode="markers",
                     marker=dict(color="#d62728", size=9), showlegend=False)
    _fig.add_annotation(x=0, y=2.6, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#2ca02c")
    _fig.add_annotation(x=0.15, y=2.7, text="+y (face here)", showarrow=False,
                        font=dict(color="#2ca02c", size=13))
    _fig.update_xaxes(range=[-3, 3], zeroline=True, showgrid=False)
    _fig.update_yaxes(range=[-3, 3], zeroline=True, showgrid=False, scaleanchor="x", scaleratio=1)
    _fig.update_layout(template="plotly_white", height=440, margin=dict(l=10, r=10, t=50, b=10),
                       title=("✅ facing +y! heading is now %.0f°" % _ang) if _ok
                             else ("heading = %.0f°  (need 90°, off by %.0f°)" % (_ang, _resid)))
    mo.vstack([toy_angle, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show me the matrix (what the slider is really doing)": mo.md(
            r"""
            A point $\mathbf{p}=(x,y)$ rotated by angle $\beta$ becomes $\mathbf{p}' = R(\beta)\,\mathbf{p}$ with

            $$R(\beta)=\begin{bmatrix}\cos\beta & -\sin\beta\\[2pt]\sin\beta & \cos\beta\end{bmatrix}.$$

            The engine doesn't make you guess $\beta$. It reads the heading angle
            $\varphi=\operatorname{atan2}(\text{head}_y-\text{TTI}_y,\ \text{head}_x-\text{TTI}_x)$
            straight off the skeleton, then rotates by $\alpha=\tfrac{\pi}{2}-\varphi$ so the heading
            lands on $+y$. That's exactly `cu._anchor_transform`: it also returns the **center** (the
            focal TTI) so the full move is *translate to origin, then rotate*:
            $\;\mathbf{p}' = R(\alpha)\,(\mathbf{p}-\mathbf{c})$. `cu.allocentricize` applies that same
            $(\mathbf{c}, R)$ — computed once from the approacher — to **all three** mice, so the whole
            social scene is re-expressed in the approacher's body frame.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(HERO, cage, mo, ranks):
    mo.md(
        f"""
        ---
        ## 2. Apply it to the Hero Event

        **Hero Event #{HERO}** — Cage {int(cage[HERO])}, male, a real aggression approach
        (approacher rank = **{['?','Dom','Mid','Sub'][int(ranks[HERO][0])]}**,
        approachee = **{['?','Dom','Mid','Sub'][int(ranks[HERO][1])]}**). On the **left** is the raw
        arena view — the mouse is wherever it happened to be in the cage. On the **right** is the same
        frame after `allocentricize`: the approacher's tail-base is pinned at the origin, facing up.
        Everything that moves on the right is *social geometry* — where the other two mice are
        **relative to the approacher**. Drag through the frames and watch the approachee close in.
        """
    )
    return


@app.cell
def _(contact, HERO, mo, kp):
    _T = kp[HERO].shape[0]
    hero_frame = mo.ui.slider(0, _T - 1, value=int(contact[HERO]), step=1,
                              label="frame (red dot = contact onset)", debounce=True, full_width=True)
    return (hero_frame,)


@app.cell
def _(HERO, cu, hero_frame, hero_hex, kp, mo, np):
    _raw = kp[HERO].astype(float)
    _body = cu.allocentricize(_raw)
    _t = hero_frame.value
    _fig_raw = cu.skeleton_fig(_raw[_t], cu.SKELETON_EDGES, colors=hero_hex,
                               title=f"RAW arena frame {_t}", height=460)
    _fig_body = cu.skeleton_fig(_body[_t], cu.SKELETON_EDGES, colors=hero_hex,
                                title=f"BODY-FRAME frame {_t} (approacher ↑)", height=460)
    # mark the approacher origin on the body-frame panel
    _fig_body.add_scatter(x=[0], y=[0], mode="markers",
                          marker=dict(symbol="x", size=12, color="black"), showlegend=False)
    mo.vstack([hero_frame, mo.hstack([_fig_raw, _fig_body], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. The honest terminology beat: *egocentric* vs *allocentric*

        The code you just called is named **`allocentricize`**, and the 19 features it produces are
        called "allocentric" throughout this codebase. **That name is a field misnomer, and we teach it
        openly rather than hide it.** Here is the real distinction, because it is exactly the distinction
        the brain implements:

        - **Egocentric** = *self-centered*. Positions are expressed relative to the animal's own body:
          "the other mouse is to **my** left, ahead of **me**." What you just built — center on the
          approacher, rotate its heading to +y — is a textbook **egocentric transform**. Retrosplenial
          and parietal cortex do this conversion continuously.
        - **Allocentric** = *world-centered*. Positions are expressed relative to a fixed external map:
          "the mouse is in the **northeast** corner." **Place cells** (O'Keefe 1971), **grid cells**
          (Hafting 2005), and **head-direction cells** (Taube 1990) hold this world-anchored map.

        So the field's label is backwards from the mechanism: our transform is **egocentric**, and the
        allocentric frame is the *other* endpoint — the world map the brain also maintains and constantly
        converts to and from. We keep the codebase name (`allocentricize`) so your code matches everyone
        else's, but when you reason about the science, call it what it is: **an egocentric, body-frame
        transform.** The reason it's the right move here is subtle and important: by removing the
        approacher's *own* arena pose (where it stands, which way it faces), you throw away everything
        that isn't social — and what survives is pure **relative** configuration, identical for a fight
        in any corner.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Deeper: the paper & where the analogy stops": mo.md(
            r"""
            **Reference frames in the brain.** O'Keefe & Dostrovsky 1971 *Brain Res.* (place cells);
            Hafting et al. 2005 *Nature* (grid cells); Taube, Muller & Ranck 1990 *J. Neurosci.*
            (head-direction cells); **Alexander et al. 2020 *Science Advances* 6:eaaz2322** (retrosplenial
            *egocentric* boundary-vector coding — the transform machinery itself). Social versions:
            Danjo et al. 2018 *Science* (**rats**, conspecific place-coding); Omer et al. 2018 *Science*
            (**bats**, coding the position of another individual).

            **The shared mathematics:** a rigid-body change of coordinates — one translation plus one
            2-D rotation — is the exact operation the retrosplenial↔hippocampal system runs to move
            between a self-centered view and a world map.

            **Species / preparation tag:** rodent hippocampal-entorhinal recordings (mouse & rat);
            the conspecific-coding results are **rat and bat**, *not* mouse.

            **Where the analogy stops:** your transform is *egocentric* and it **factors heading out**;
            place/grid/HD cells are the *allocentric endpoint* and they **encode** heading and position.
            They are opposite ends of one computation, not the same thing — and the social conspecific
            papers are rat/bat, so don't quietly assume a mouse homecage has an identical code.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. The 19 features — plain English

        `allocentricize` gives you a body-frame movie; `features_one` summarizes each event's movie
        into **19 numbers**. Every one is arena-invariant (Section 5 proves it). Here's what each means:

        | # | name | plain English |
        |---|------|----------------|
        | 0 | `appr_speed_mean` | approacher's average body speed (px/frame) |
        | 1 | `appr_speed_max` | approacher's **peak** speed — a lunge shows up here |
        | 2 | `appe_speed_mean` | approachee's average body speed |
        | 3 | `appe_speed_max` | approachee's peak speed — a flinch/flee spikes this |
        | 4 | `appr_body_len` | approacher nose→tail-base length (stretched vs hunched posture) |
        | 5 | `appe_body_len` | approachee body length |
        | 6 | `appr_angvel` | how fast the approacher **turns** (heading angular velocity) |
        | 7 | `appe_angvel` | how fast the approachee turns |
        | 8 | `pair_dist_mean` | average distance between the two mice |
        | 9 | `pair_dist_min` | their **closest** distance in the event |
        | 10 | `appr_nose_to_appe_tti_min` | closest approacher-nose → approachee-rump (anogenital sniff / rear attack) |
        | 11 | `appe_nose_to_appr_tti_min` | closest approachee-nose → approacher-rump |
        | 12 | `appr_faces_appe` | does the approacher **point at** the other? facing cosine, +1 = dead-on |
        | 13 | `appe_faces_appr` | does the approachee point back at the approacher? |
        | 14 | `closing_speed` | how fast the gap shrinks (+ = closing in) |
        | 15 | `heading_alignment` | are the two headings parallel (+1) or opposed (−1)? |
        | 16 | `bystander_dist_mean` | average distance to the **third** (bystander) mouse |
        | 17 | `bystander_dist_min` | closest the bystander gets |
        | 18 | `triangle_area_mean` | spread of the trio (area of the 3-centroid triangle) |

        Notice the mix: **kinematics** (speed, angular velocity), **posture** (body length),
        **relative geometry** (distances, facing cosines), and the **third mouse** — the whole social
        configuration, no arena coordinates anywhere.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. The invariance demo — pick up the whole cage and shake it

        Here's the payoff of choosing a body-frame point of view. Grab the Hero Event and apply a
        **random rigid motion** to the *entire scene*: rotate the whole cage by some angle and slide it
        somewhere else. The **raw coordinates swing wildly** (left) — but the **19 features don't move at
        all** (right), because every one is measured *between* the mice, not against the walls. Drag the
        angle and watch: the left panel spins, the right panel is frozen.
        """
    )
    return


@app.cell
def _(mo):
    inv_angle = mo.ui.slider(0, 350, value=0, step=10,
                             label="arena rotation applied to the whole event (degrees)",
                             debounce=True, full_width=True)
    return (inv_angle,)


@app.cell
def _(HERO, cu, feat_names, go, inv_angle, kp, mo, np):
    _hero = kp[HERO].astype(float)
    _f0 = cu.features_one(_hero)
    _th = np.deg2rad(inv_angle.value)
    _R = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
    _trans = np.array([600.0, -300.0])                      # a fixed, obvious translation
    _warp = np.einsum("ij,tmnj->tmni", _R, _hero) + _trans[None, None, None, :]
    _f1 = cu.features_one(_warp)
    _maxdiff = float(np.nanmax(np.abs(_f0 - _f1)))

    # left: raw node cloud at contact for original vs warped -> the coordinates clearly move
    _t = _hero.shape[0] // 2
    _p0 = _hero[_t].reshape(-1, 2); _p0 = _p0[np.isfinite(_p0).all(1)]
    _p1 = _warp[_t].reshape(-1, 2); _p1 = _p1[np.isfinite(_p1).all(1)]
    _left = go.Figure()
    _left.add_scatter(x=_p0[:, 0], y=_p0[:, 1], mode="markers",
                      marker=dict(color="#7f7f7f", size=6), name="original")
    _left.add_scatter(x=_p1[:, 0], y=_p1[:, 1], mode="markers",
                      marker=dict(color="#d62728", size=6), name="rotated + translated")
    _left.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False)
    _left.update_xaxes(showgrid=False)
    _left.update_layout(template="plotly_white", height=420, title="RAW pixel coordinates — they swing",
                        margin=dict(l=10, r=10, t=40, b=10), legend=dict(y=1.0))

    # right: the 19 features, both versions, overlaid -> identical
    _right = go.Figure()
    _right.add_bar(x=feat_names, y=_f0, marker_color="#7f7f7f", name="original")
    _right.add_bar(x=feat_names, y=_f1, marker_color="#d62728", name="warped", opacity=0.6)
    _right.update_layout(template="plotly_white", height=420, barmode="overlay",
                         title=f"19 features — frozen (max |Δ| = {_maxdiff:.2e})",
                         margin=dict(l=10, r=10, t=40, b=120), legend=dict(y=1.0))
    _right.update_xaxes(tickangle=-60)
    mo.vstack([inv_angle, mo.hstack([_left, _right], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The features are fixed to **~1e-4 or better** (numerically, exactly zero) no matter how you spin
        or shift the cage. That is *invariance*, and it is the reason the next five notebooks can treat
        one event as one point in a 19-D space without ever worrying about where in the arena it
        happened.

        ---
        ## 6. Which features carry aggression?

        Now split the corpus by `agg_label` (aggression vs not) and look at each feature's distribution.
        Pick a feature; the violins show the two groups with a Mann–Whitney U p-value annotated. Watch
        which features separate cleanly — and, foreshadowing today's lab-meeting answer, notice that the
        *geometry* features (facing, distance, angle) separate far **less** than the **kinematic** ones
        (speed, angular velocity).
        """
    )
    return


@app.cell
def _(feat_names, mo):
    feat_pick = mo.ui.dropdown(options=feat_names, value="appe_angvel",
                               label="feature to inspect")
    return (feat_pick,)


@app.cell
def _(X, agg, feat_names, feat_pick, go, mo, np):
    from scipy.stats import mannwhitneyu
    _i = feat_names.index(feat_pick.value)
    _a = X[agg == 1, _i]; _b = X[agg == 0, _i]
    _u, _p = mannwhitneyu(_a, _b)
    # Cohen's d for the header
    _pooled = np.sqrt(((len(_a) - 1) * _a.var(ddof=1) + (len(_b) - 1) * _b.var(ddof=1))
                      / (len(_a) + len(_b) - 2) + 1e-12)
    _d = (_a.mean() - _b.mean()) / _pooled
    _fig = go.Figure()
    _fig.add_violin(y=_b, name="not aggression", line_color="#7f7f7f", box_visible=True,
                    meanline_visible=True, points=False)
    _fig.add_violin(y=_a, name="aggression", line_color="#d62728", box_visible=True,
                    meanline_visible=True, points=False)
    _ptxt = f"p = {_p:.1e}" if _p >= 1e-300 else "p < 1e-300"
    _fig.update_layout(template="plotly_white", height=460,
                       title=f"{feat_pick.value}   ·   {_ptxt}   ·   Cohen's d = {_d:+.2f}",
                       yaxis_title=feat_pick.value, margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([feat_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A facing cosine (`appr_faces_appe`) is the behavioral cousin of **cosine directional tuning** in
        motor cortex (Georgopoulos 1986) — a neuron that fires most for one movement direction and falls
        off as the cosine of the angle away from it. The difference to keep straight: *our* detectors are
        **designed** by us; a tuning curve is **learned/measured** from the animal.

        ---
        ## 7. Exercise — does aggression arrive from a different *direction*?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Toolbox.**

        - `cu.allocentricize(kp_event)` — `(T,3,15,2)` world coords → `(T,3,15,2)` in the **approacher's**
          body frame (origin at approacher TTI, heading → +y). *Falls back to identity if head/TTI are
          missing.*
        - `cu.TTI` (= 11) — the tail-base node index; `cu.HEAD` (= 1) — the head node.
        - `contact` — per-event contact-onset frame; `agg` — per-event aggression label (0/1).
        - `np.histogram2d`, `np.nanmean`.

        **Hypothesis banner.** *Aggressive contacts arrive from a different angle than non-aggressive
        ones — the approachee sits in a different part of the approacher's body frame at contact.*

        **Your job (two parts).**

        1. **The transform, by hand.** For the Hero Event, apply `allocentricize`, take the first frame
           where the approacher's head and TTI are both finite, and confirm the approacher's **TTI lands
           at the origin** and its **heading points at +y**.
        2. **Front vs rear.** Pool all events. For each, take the **approachee's TTI** in the
           **approacher's body frame** at that event's `contact` frame. Call it *in front* if its
           `y > 0`. Compute the fraction *in front* for aggression vs non-aggression events, and report
           the difference.

        Fill in `front_diff` (part 2's aggression-minus-nonaggression fraction) and `hero_ok` (part 1,
        a bool). Then run the self-check.
        """
    )
    return


@app.cell
def _(HERO, agg, contact, cu, kp, np):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    # Part 1 — the transform, by hand, on the Hero Event:
    _bf_hero = cu.allocentricize(kp[HERO].astype(float))
    _ho = np.isfinite(kp[HERO].astype(float)[:, 0, cu.HEAD]).all(1)
    _to = np.isfinite(kp[HERO].astype(float)[:, 0, cu.TTI]).all(1)
    _anchor = np.where(_ho & _to)[0][0]
    _tti = _bf_hero[_anchor, 0, cu.TTI]
    _head = _bf_hero[_anchor, 0, cu.HEAD]
    _heading = (_head - _tti) / (np.linalg.norm(_head - _tti) + 1e-12)
    hero_ok = bool(np.allclose(_tti, 0.0, atol=1e-3) and np.allclose(_heading, [0.0, 1.0], atol=1e-2))

    # Part 2 — front vs rear, pooled. (A live loop over ~1500 events; allocentricize is a cheap
    # einsum so this runs in well under a second.)
    _front, _lab = [], []
    for _i in range(len(kp)):
        _bf = cu.allocentricize(kp[_i].astype(float))
        _t = min(int(contact[_i]), _bf.shape[0] - 1)
        _p = _bf[_t, 1, cu.TTI]                     # approachee TTI in approacher body frame
        if np.isfinite(_p).all():
            _front.append(_p[1] > 0)                # y>0 => in front of the approacher
            _lab.append(agg[_i])
    _front = np.array(_front); _lab = np.array(_lab)
    front_diff = float(_front[_lab == 1].mean() - _front[_lab == 0].mean())
    # ---------------------------------------------------------------------------------------------
    return front_diff, hero_ok


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            # Part 1
            bf_hero = cu.allocentricize(kp[HERO].astype(float))
            ho = np.isfinite(kp[HERO].astype(float)[:, 0, cu.HEAD]).all(1)
            to = np.isfinite(kp[HERO].astype(float)[:, 0, cu.TTI]).all(1)
            anchor = np.where(ho & to)[0][0]
            tti  = bf_hero[anchor, 0, cu.TTI]                       # ~ (0, 0)
            head = bf_hero[anchor, 0, cu.HEAD]
            heading = (head - tti) / np.linalg.norm(head - tti)    # ~ (0, 1)
            hero_ok = np.allclose(tti, 0, atol=1e-3) and np.allclose(heading, [0, 1], atol=1e-2)

            # Part 2
            front, lab = [], []
            for i in range(len(kp)):
                bf = cu.allocentricize(kp[i].astype(float))
                t = min(int(contact[i]), bf.shape[0] - 1)
                p = bf[t, 1, cu.TTI]           # approachee tail-base in approacher frame
                if np.isfinite(p).all():
                    front.append(p[1] > 0); lab.append(agg[i])
            front, lab = np.array(front), np.array(lab)
            front_diff = front[lab == 1].mean() - front[lab == 0].mean()
            ```

            **What you should find:** the transform check passes exactly, and `front_diff` is **tiny**
            (about +0.01). Almost every approach is *frontal* by construction — the approacher is, after
            all, approaching — so aggression does **not** arrive from a different direction. The honest
            answer to the lab-meeting question is *no*: the aggression signal lives in the **kinematic**
            features (speed, angular velocity; recall `appe_angvel` had Cohen's d ≈ 1.0), not in the
            approach angle. That is exactly why we keep all 19 features instead of just a geometry summary.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(front_diff, hero_ok, mo):
    # Honest self-check with a tolerance band. Part 1 is exact; part 2's GRADED-CORRECT answer is the
    # honest null: the front-vs-rear difference is SMALL (|diff| < 0.10; pinned full-corpus value ~0.01),
    # i.e. aggression does NOT arrive from a different direction.
    _p1 = bool(hero_ok)
    _p2 = abs(float(front_diff)) < 0.10
    _ok = _p1 and _p2
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _msg1 = "✅ approacher lands at origin facing +y" if _p1 else "❌ transform check failed — origin/heading off"
    _msg2 = (f"✅ front-vs-rear difference is small ({front_diff:+.3f}) — aggression does NOT arrive from a "
             "different direction; the signal is kinematic"
             if _p2 else
             f"❌ your front_diff = {front_diff:+.3f} is implausibly large — check the body-frame y-sign")
    _head = "PASS — both parts correct" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_msg1}<br>{_msg2}<br>
        <span style="font-size:0.9em;color:#555">Graded answer for part 2 is the honest null — the
        exercise is not scored against noise. Tolerance band: |front_diff| &lt; 0.10.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Sealed Cage 16 — visible but redacted; opens in NB08. Counter: 6 notebooks from now.
    mo.md(
        r"""
        ---
        <div style="background:#1a1a1a;color:#bbb;padding:16px 20px;border-radius:8px;
                    border:2px dashed #555">
        <b style="color:#ff5252;letter-spacing:1px">🔒 SEALED — CAMERA 16</b>
        &nbsp;·&nbsp; <span style="color:#888">the animal on the rig</span><br><br>
        <span style="font-family:monospace">
        events: <b style="color:#ddd">470</b> &nbsp;|&nbsp;
        aggression: <b>██████</b> &nbsp;|&nbsp;
        sex: <b>█</b> &nbsp;|&nbsp;
        rank labels: <b>████████</b><br>
        skeletons: <span style="color:#333;background:#333">▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓</span>
        </span><br><br>
        This is the cage the decoder must survive but has <i>never seen</i>. Its 19 features already
        exist — we just refuse to look. <b style="color:#ffb74d">Notebooks until unlock: 6.</b>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What we threw away / how it breaks

        **Discarded on purpose.** The body-frame transform deliberately throws away *where in the cage*
        the event happened and *which way the approacher faced in the arena* — an egocentric view keeps
        only relative geometry. That's the point, but it means you can **no longer ask arena questions**
        from `X` alone (does aggression cluster near a wall? at the food hopper?). Those need the raw
        coordinates back.

        **Concrete failure modes on this dataset.**

        1. **Silent identity fallback.** If the approacher's head or TTI is missing on *every* frame,
           `allocentricize` returns the event **unchanged** — the features are then computed in raw arena
           coordinates and are *not* invariant. This is invisible unless you audit for it (tail-chain and
           head dropout are exactly the nodes NB01 flagged as least reliable).
        2. **One bad heading rotates the whole scene.** The transform reads heading from a *single* anchor
           frame. If that frame's head/TTI is jittery, the entire event is rotated to the wrong angle and
           every geometry feature is corrupted — with no error raised.
        3. **Angle is nearly useless here.** As the exercise showed, approach *direction* barely separates
           aggression (front_diff ≈ 0.01). A pipeline that leaned on approach angle would find almost
           nothing; the real signal is kinematic.

        **How would you analyze this?** Head-direction cells *encode* heading; your transform *factors it
        out*. Sketch how a brain (or your pipeline) could use a head-direction signal to move **between**
        the egocentric and allocentric frames — and what you'd gain by keeping both representations
        instead of collapsing to one.
        """
    )
    return


@app.cell(hide_code=True)
def _(X, readout_fig):
    readout_fig(float(X.shape[1]), "Readout Board — end of NB02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What we ship next

        You shipped the deliverable: **`X (1500, 19)`** — every event as one arena-invariant vector of
        social geometry and kinematics, the same egocentric move retrosplenial cortex makes on every
        step. Gauge A fell from **11,700 raw numbers to 19**. And we already learned something honest:
        aggression is *not* a matter of direction, it's a matter of **motion**.

        But 19 numbers is still a lot to look at, and they are far from independent — speed, closing
        speed, and distance obviously move together. **Next (NB03): feel the signal in time.** Before we
        compress these 19, we'll *look* at them — in value, in time, and in frequency — the way a
        physiologist reads a raw trace, and measure who-moves-first between two coupled mice.
        """
    )
    return


if __name__ == "__main__":
    app.run()

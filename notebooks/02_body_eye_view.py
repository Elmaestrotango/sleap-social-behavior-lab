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
        # NB02 · The body-centered (egocentric) view

        ## Why this step matters

        You are studying social behavior. A behavior — an attack, a sniff, a chase — is the *same
        behavior* whether it happens in the top-left corner of the cage or the bottom-right, and
        whether the two mice face north or south. Raw pixel coordinates do not capture that: the same
        behavior in two locations produces completely different numbers, because the numbers describe
        *where in the arena* the mice are, not *what they are doing to each other*.

        Before we can compare events, describe them, or later train a classifier, we need to remove
        two things that are not part of the behavior: **where in the cage** the event happened, and
        **which way the animals happened to be facing in the arena**. What is left is the part that is
        genuinely social — the position and motion of each mouse *relative to the others*.

        ## Definitions (read these first)

        - **Keypoint** — a single tracked body point (nose, head, tail-base, …), stored as an
          `(x, y)` pixel location in the camera image. NB01 produced these.
        - **Body frame** (also called **egocentric coordinates**) — coordinates measured relative to
          one animal's own body: put the origin at that animal's tail-base, and rotate so the
          direction it faces points straight up (+y). In a body frame you describe the scene as
          "the other mouse is ahead of me and slightly to my left," instead of "the other mouse is at
          pixel (812, 344)."
        - **Invariant** — a number that does **not** change when you move or rotate the whole scene.
          A distance between two mice is invariant; a mouse's absolute pixel position is not.

        ## What we will do (the method)

        1. Take one event (all three mice, every frame) in raw arena coordinates.
        2. **Translate** so the approaching mouse's tail-base sits at the origin, then **rotate** so
           that mouse faces +y. This is the body-centered (egocentric) transform.
        3. Summarize the transformed event into **19 interpretable numbers** — speeds, distances, and
           facing angles — that are the same no matter where or which way the event happened.

        The reason we bother: behaviors are rotationally invariant. A mouse does not care about the
        arena's orientation; the social geometry that matters is relative to the animal. (This kind of
        relative, body-centered description is also how neuroscientists quantify social position.)

        **Deliverable of this notebook:** the feature matrix `X (1500, 19)` — one 19-number vector per
        event. Every later notebook reads `X`, not pixels.
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

    # Running example event. The design doc named #742, but in the shipped bundle index 742 is a
    # cage-12 non-aggression event; we use the cleanest cage-15 (male) aggression event instead.
    EX = 909
    ex_hex = tuple(cu.RANK_HEX.get(int(r), cu.RANK_HEX[0]) for r in ranks[EX])

    _board = pd.read_csv(cu.data_path("data/readout_board.csv", ROOT))
    board = _board
    return EX, X, agg, board, cage, contact, ev, ex_hex, feat_names, kp, ranks


@app.cell(hide_code=True)
def _(board, go):
    # Readout Board — two gauges. Gauge A = size of the representation (falls through Phase 1);
    # Gauge B = held-out readiness (rises in Phase 2, still 0 here). Degrades gracefully if a
    # board row is missing. FIX: mode="number" only (no delta). A delta against the 11,700 raw
    # baseline rendered a confusing NEGATIVE ("-11,681"); the "was 11,700 raw" context now lives
    # in the title text instead. Height/margin raised so the two-line titles do not overlap.
    def readout_fig(gauge_a_value, title):
        def _bench(gauge, nb):
            _m = board[(board["gauge"] == gauge) & (board["notebook"] == nb)]
            return float(_m["value"].iloc[0]) if len(_m) else None
        raw_bench = _bench("A", "NB01")          # 11,700 raw numbers, the Phase-1 starting point
        raw_txt = int(raw_bench) if raw_bench is not None else 11700
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number", value=gauge_a_value,
            number={"valueformat": ",.0f", "suffix": " numbers", "font": {"size": 44}},
            title={"text": "<b>Gauge A</b> · size of representation<br>"
                           "<span style='font-size:0.8em;color:#888'>"
                           f"now vs {raw_txt:,} raw (falls through Phase 1)</span>"},
            domain={"row": 0, "column": 0}))
        fig.add_trace(go.Indicator(
            mode="number", value=0,
            number={"valueformat": ".0f", "suffix": "  (rises in Phase 2)", "font": {"size": 44}},
            title={"text": "<b>Gauge B</b> · held-out readiness<br>"
                           "<span style='font-size:0.8em;color:#888'>not started</span>"},
            domain={"row": 0, "column": 1}))
        fig.update_layout(grid={"rows": 1, "columns": 2, "pattern": "independent"},
                          template="plotly_white", height=230, title=title,
                          margin=dict(l=20, r=20, t=95, b=20))
        return fig
    return (readout_fig,)


@app.cell(hide_code=True)
def _(X, readout_fig):
    # Gauge A now = the width of the feature matrix we are about to build (19).
    readout_fig(float(X.shape[1]), "Readout Board — start of NB02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Rotating a frame by hand

        The whole transform is two moves: **translate** so the focal mouse's tail-base (node `TTI`)
        sits at the origin, then **rotate** so its heading (tail-base → head) points straight up (+y).
        The second move — rotation — is worth doing by hand once so it is not a black box.

        Below is one real frame of our example event: all three mice, in raw arena coordinates,
        centered on the field for display. Drag the slider to rotate the **entire field** by an angle
        β. Watch two things:

        - The **numbers on the right** update live: the 2×2 rotation matrix `R(β)`, and one sample
          keypoint's coordinates before and after rotation.
        - The **shape never distorts.** Rotation is rigid: every distance and every angle *between*
          the mice stays exactly the same. Only the orientation of the whole picture on the page
          changes. That rigidity is the reason this operation is safe to apply to behavior — it moves
          the frame of reference without altering the behavior.

        Colors are by rank throughout the course: **Dom = red, Int = blue, Sub = green.**
        """
    )
    return


@app.cell
def _(mo):
    toy_angle = mo.ui.slider(-180, 180, value=0, step=5,
                             label="rotation β applied to the whole field (degrees)",
                             debounce=True, full_width=True)
    return (toy_angle,)


@app.cell
def _(EX, contact, cu, ex_hex, go, kp, mo, np, toy_angle):
    # Rotate ONE frame of the whole interaction (all three mice) by the slider angle β, so the
    # student sees the entire field turn as a rigid body.
    _t = int(contact[EX])
    _field = kp[EX][_t].astype(float)                       # (3, 15, 2) — one frame, three mice
    _pts = _field.reshape(-1, 2)
    _c = np.nanmean(_pts[np.isfinite(_pts).all(1)], axis=0)  # field center, for in-place rotation
    _centered = _field - _c
    _beta = np.deg2rad(toy_angle.value)
    _R = np.array([[np.cos(_beta), -np.sin(_beta)], [np.sin(_beta), np.cos(_beta)]])
    _rot = np.einsum("ij,mnj->mni", _R, _centered)

    _fig = go.Figure()
    for _m in range(3):
        _mk = _rot[_m]
        _ok = np.isfinite(_mk).all(1)
        _ex, _ey = [], []
        for _u, _v in cu.SKELETON_EDGES:
            if _ok[_u] and _ok[_v]:
                _ex += [_mk[_u, 0], _mk[_v, 0], None]
                _ey += [_mk[_u, 1], _mk[_v, 1], None]
        _fig.add_scatter(x=_ex, y=_ey, mode="lines",
                         line=dict(color=ex_hex[_m], width=2), showlegend=False, hoverinfo="skip")
        _fig.add_scatter(x=_mk[_ok, 0], y=_mk[_ok, 1], mode="markers",
                         marker=dict(color=ex_hex[_m], size=6), showlegend=False, hoverinfo="skip")
    _fig.update_xaxes(range=[-260, 260], showgrid=False, zeroline=True)
    _fig.update_yaxes(range=[260, -260], showgrid=False, zeroline=True, scaleanchor="x", scaleratio=1)
    _fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=44, b=10),
                       title=f"whole field rotated by β = {toy_angle.value}°")

    # live numeric readout beside the plot
    _in = _centered[0, cu.HEAD]                             # approacher head, centered (input)
    _out = _rot[0, cu.HEAD]                                 # approacher head, after rotation (output)
    _readout = mo.md(
        f"""
        **Rotation matrix `R(β)`, β = {toy_angle.value}°**

        |  |  |
        |---:|---:|
        | {_R[0, 0]:+.3f} | {_R[0, 1]:+.3f} |
        | {_R[1, 0]:+.3f} | {_R[1, 1]:+.3f} |

        Each keypoint `(x, y)` becomes `R(β)·(x, y)`.

        **Approacher head keypoint**

        before: ({_in[0]:+.1f}, {_in[1]:+.1f})
        after:  &nbsp;({_out[0]:+.1f}, {_out[1]:+.1f})

        The distance from this point to the origin is unchanged:
        {np.linalg.norm(_in):.1f} → {np.linalg.norm(_out):.1f} px. That is what "rigid" means — every
        point stays the same distance from the center; only the direction rotates.
        """
    )
    mo.vstack([toy_angle, mo.hstack([_fig, _readout], widths=[2, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "How the code picks β automatically": mo.md(
            r"""
            You do not have to guess β. The code reads the approacher's heading angle straight off the
            skeleton and rotates by exactly the amount that lands it on +y.

            - **Heading angle** of the approacher:
              $\varphi=\operatorname{atan2}(\text{head}_y-\text{TTI}_y,\ \text{head}_x-\text{TTI}_x)$.
            - **Rotation needed** to send that heading to straight up:
              $\alpha=\tfrac{\pi}{2}-\varphi$.
            - **Rotation matrix:**
              $R(\alpha)=\begin{bmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{bmatrix}$.

            `cu._anchor_transform` returns this $R$ together with the **center** (the approacher's
            tail-base), so the full move is *translate to origin, then rotate*:
            $\mathbf{p}' = R(\alpha)\,(\mathbf{p}-\mathbf{c})$. `cu.allocentricize` then applies that
            same $(\mathbf{c}, R)$ — computed once from the approacher — to **all three** mice, so the
            entire social scene is re-expressed in the approacher's body frame.

            **`cu.allocentricize`** · *purpose:* put an event in the approacher's body frame ·
            *input:* `kp_event` of shape `(T, 3, 15, 2)` in arena pixels ·
            *output:* the same-shaped array, translated and rotated. If the approacher's head or
            tail-base is missing on every frame it cannot find a heading and returns the event
            unchanged.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(EX, cage, contact, cu, kp, mo, ranks):
    _rn = ["unknown", "Dom", "Int", "Sub"]
    _gif = cu.event_gif_bytes(kp[EX], ranks[EX], contact_rel=int(contact[EX]), cell=200, fps=20)
    mo.vstack([
        mo.md(
            f"""
            ---
            ## 2. Applying the transform to the example event

            Our example is **event {EX}** — Cage {int(cage[EX])}, male, a real aggression approach.
            We label the two interacting mice plainly: the **approacher**
            (rank **{_rn[int(ranks[EX][0])]}**) and the **approachee**
            (rank **{_rn[int(ranks[EX][1])]}**). A third **bystander** mouse
            (rank **{_rn[int(ranks[EX][2])]}**) is also present.

            First, watch the behavior itself. The clip below is the raw event: the white arrow points
            approacher → approachee, and the red dot marks contact onset. Because we learn behavior by
            seeing it, most methods in this course will be checked against clips like this.
            """
        ),
        mo.Html('<div style="text-align:center">' + cu.gif_img_html(_gif, width=240) + "</div>"),
        mo.md(
            """
            Now compare the two coordinate frames side by side. On the **left** is the raw arena view —
            the mice are wherever they happened to be in the cage. On the **right** is the same frame
            after `allocentricize`: the approacher's tail-base is pinned at the origin (black ✕) and
            the approacher faces up. Everything that moves on the right is **social geometry** — where
            the other two mice sit *relative to the approacher*. Drag the frame slider and watch the
            approachee close in.
            """
        ),
    ])
    return


@app.cell
def _(contact, EX, mo, kp):
    _T = kp[EX].shape[0]
    ex_frame = mo.ui.slider(0, _T - 1, value=int(contact[EX]), step=1,
                            label="frame (red dot = contact onset)", debounce=True, full_width=True)
    return (ex_frame,)


@app.cell
def _(EX, cu, ex_frame, ex_hex, kp, mo, np):
    _raw = kp[EX].astype(float)
    _body = cu.allocentricize(_raw)
    _t = ex_frame.value
    _fig_raw = cu.skeleton_fig(_raw[_t], cu.SKELETON_EDGES, colors=ex_hex,
                               title=f"RAW arena frame {_t}", height=460)
    _fig_body = cu.skeleton_fig(_body[_t], cu.SKELETON_EDGES, colors=ex_hex,
                                title=f"BODY FRAME frame {_t} (approacher faces up)", height=460)
    # mark the approacher origin on the body-frame panel
    _fig_body.add_scatter(x=[0], y=[0], mode="markers",
                          marker=dict(symbol="x", size=12, color="black"), showlegend=False)
    mo.vstack([ex_frame, mo.hstack([_fig_raw, _fig_body], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Why the body frame is the right choice

        The function is named `allocentricize` in this codebase, and the 19 features are stored under
        that name, so your code matches everyone else's. When you reason about the science, though,
        describe it plainly for what it is: a **body-centered (egocentric) transform** — the scene is
        expressed relative to the approacher's own body.

        The point of the transform is what it *removes*. By centering on the approacher's tail-base and
        rotating its heading to +y, we throw away the approacher's own arena pose: where it stands and
        which way it faces in the cage. Those are not part of the behavior. What survives is the
        **relative configuration** — how far apart the mice are, who faces whom, how the trio is
        arranged — and that relative configuration is identical for the same behavior in any corner of
        the cage. The next section demonstrates that invariance directly.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. The 19 features, in plain English

        `allocentricize` gives you a body-frame movie of an event; `cu.features_one` summarizes that
        movie into **19 numbers**. Each is arena-invariant (Section 5 checks this). Here is what each
        one means:

        | # | name | plain meaning |
        |---|------|----------------|
        | 0 | `appr_speed_mean` | approacher's average body speed (px/frame) |
        | 1 | `appr_speed_max` | approacher's peak speed — a lunge shows up here |
        | 2 | `appe_speed_mean` | approachee's average body speed |
        | 3 | `appe_speed_max` | approachee's peak speed — a flinch or flee spikes this |
        | 4 | `appr_body_len` | approacher nose→tail-base length (stretched vs hunched posture) |
        | 5 | `appe_body_len` | approachee body length |
        | 6 | `appr_angvel` | how fast the approacher turns (heading angular velocity) |
        | 7 | `appe_angvel` | how fast the approachee turns |
        | 8 | `pair_dist_mean` | average distance between the two mice |
        | 9 | `pair_dist_min` | their closest distance during the event |
        | 10 | `appr_nose_to_appe_tti_min` | closest approacher-nose → approachee-rump distance (a rear sniff/attack) |
        | 11 | `appe_nose_to_appr_tti_min` | closest approachee-nose → approacher-rump distance |
        | 12 | `appr_faces_appe` | does the approacher point at the other? facing cosine, +1 = dead-on |
        | 13 | `appe_faces_appr` | does the approachee point back at the approacher? |
        | 14 | `closing_speed` | how fast the gap shrinks (positive = closing in) |
        | 15 | `heading_alignment` | are the two headings parallel (+1) or opposed (−1)? |
        | 16 | `bystander_dist_mean` | average distance to the third (bystander) mouse |
        | 17 | `bystander_dist_min` | closest the bystander gets |
        | 18 | `triangle_area_mean` | spread of the trio (area of the triangle of the three centroids) |

        `cu.features_one` · *purpose:* turn one event into these 19 numbers ·
        *input:* `kp_event` of shape `(T, 3, 15, 2)`, mice ordered [approacher, approachee, bystander]
        · *output:* a length-19 vector. Notice the mix: **kinematics** (speed, angular velocity),
        **posture** (body length), **relative geometry** (distances, facing cosines), and the
        **third mouse** — the whole social configuration, with no arena coordinates anywhere.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Checking invariance: rotate and move the whole cage

        This is the benefit of the body-centered choice. Take the example event and apply a **rigid
        motion** to the *entire scene*: rotate the whole cage by some angle and slide it somewhere
        else. Then compare two kinds of measurement:

        - **Arena-frame measurements** (bottom bars) — the approacher's absolute heading angle and its
          centroid position in the cage. These **change** with the warp, because they describe where
          the mouse is and which way it points in the arena.
        - **Body-frame features** (right panel) — the 19 numbers from `features_one`. These stay
          **frozen**, because every one is measured *between* the mice, not against the arena walls.

        Drag the angle. The raw node cloud on the left visibly swings, the arena measurements below it
        change, and the 19 features on the right do not move.
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
def _(EX, cu, feat_names, go, inv_angle, kp, mo, np):
    _ev = kp[EX].astype(float)
    _f0 = cu.features_one(_ev)
    _th = np.deg2rad(inv_angle.value)
    _R = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
    _trans = np.array([600.0, -300.0])                      # a fixed, obvious translation
    _warp = np.einsum("ij,tmnj->tmni", _R, _ev) + _trans[None, None, None, :]
    _f1 = cu.features_one(_warp)
    _maxdiff = float(np.nanmax(np.abs(_f0 - _f1)))

    # arena-frame quantities that DO change: approacher heading angle + centroid position (mid frame)
    def _arena_meas(evt):
        _mid = evt.shape[0] // 2
        _ap = evt[_mid, 0]
        _v = _ap[cu.HEAD] - _ap[cu.TTI]
        _hd = float(np.rad2deg(np.arctan2(_v[1], _v[0])))
        _cen = np.nanmean(_ap[cu.BODY_NODES], axis=0)
        return [_hd, float(_cen[0]), float(_cen[1])]
    _a0 = _arena_meas(_ev)
    _a1 = _arena_meas(_warp)
    _anames = ["heading angle (deg)", "centroid x (px)", "centroid y (px)"]

    # left: raw node cloud at contact, original vs warped -> the coordinates clearly move
    _t = _ev.shape[0] // 2
    _p0 = _ev[_t].reshape(-1, 2); _p0 = _p0[np.isfinite(_p0).all(1)]
    _p1 = _warp[_t].reshape(-1, 2); _p1 = _p1[np.isfinite(_p1).all(1)]
    _left = go.Figure()
    _left.add_scatter(x=_p0[:, 0], y=_p0[:, 1], mode="markers",
                      marker=dict(color="#7f7f7f", size=6), name="original")
    _left.add_scatter(x=_p1[:, 0], y=_p1[:, 1], mode="markers",
                      marker=dict(color="#d62728", size=6), name="rotated + moved")
    _left.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False)
    _left.update_xaxes(showgrid=False)
    _left.update_layout(template="plotly_white", height=420, title="RAW pixel coordinates — they swing",
                        margin=dict(l=10, r=10, t=40, b=10), legend=dict(y=1.0))

    # right: the 19 body-frame features, both versions, overlaid -> identical
    _right = go.Figure()
    _right.add_bar(x=feat_names, y=_f0, marker_color="#7f7f7f", name="original")
    _right.add_bar(x=feat_names, y=_f1, marker_color="#d62728", name="warped", opacity=0.6)
    _right.update_layout(template="plotly_white", height=420, barmode="overlay",
                         title=f"19 body-frame features — frozen (max |Δ| = {_maxdiff:.2e})",
                         margin=dict(l=10, r=10, t=40, b=120), legend=dict(y=1.0))
    _right.update_xaxes(tickangle=-60)

    # bottom: the arena-frame measurements that DO change
    _arena = go.Figure()
    _arena.add_bar(x=_anames, y=_a0, marker_color="#7f7f7f", name="original")
    _arena.add_bar(x=_anames, y=_a1, marker_color="#d62728", name="warped")
    _arena.update_layout(template="plotly_white", height=300, barmode="group",
                         title="ARENA-frame measurements — these DO change under the warp",
                         margin=dict(l=10, r=10, t=40, b=10), legend=dict(y=1.0))

    mo.vstack([inv_angle, mo.hstack([_left, _right], widths=[1, 1]), _arena])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The body-frame features stay fixed to about **1e-4 or better** (numerically, zero) no matter
        how you spin or shift the cage, while the arena heading and centroid change with every turn.
        That contrast is the definition of **invariance**, and it is why the next several notebooks can
        treat one event as one point in a 19-dimensional space without ever worrying about where in the
        arena it happened.

        ---
        ## 6. Which features carry aggression?

        Now use the aggression label (`agg`, 0/1) to ask which features actually differ between
        aggressive and non-aggressive events. For a chosen feature, the two **violin** plots show its
        distribution in each group. A violin is a smoothed histogram mirrored into a symmetric shape;
        the box inside marks the median and quartiles. The header reports a **Mann–Whitney U** p-value
        (a rank-based test of whether the two groups differ) and **Cohen's d** (the difference in
        means, in standard-deviation units — a plain effect size).

        Step through the features and notice the pattern: the **kinematic** features (speed, angular
        velocity) separate the two groups much more cleanly than the **geometry** features (facing,
        distance, angle). We will make that observation precise in the exercise.
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
        ---
        ## 7. Exercise — does aggression arrive from a different direction?

        ### Why ask this

        The kinematic features clearly separate aggression. A reasonable follow-up: does aggression
        also come from a different **direction** — does the approachee sit in a different part of the
        approacher's body frame at contact (in front vs behind)? This is exactly the kind of question
        the body frame makes easy to ask, because "in front" simply means the other mouse's tail-base
        has a positive y-coordinate once we are in the approacher's frame.

        ### Definitions you need

        - **Body frame** — from Section 1: approacher tail-base at the origin, heading pointing +y.
        - **"In front"** — in that frame, the approachee's tail-base (node `cu.TTI`) has **y > 0**.
        - `contact` — the per-event frame index where contact begins.

        ### What to do

        You will fill in **two blanks** (`____`). Everything else is written for you.

        - **Part 1** transforms the example event into the body frame and checks the approacher lands
          at the origin facing up. You fill the call to `cu.allocentricize`.
        - **Part 2** loops over all events, finds the approachee's tail-base in the approacher's body
          frame at contact, and marks it "in front" when its y-coordinate is positive. You fill the
          `y > 0` test.

        The cell returns `ex_ok` (Part 1, a bool), `front_diff` (Part 2, the aggression-minus-non
        fraction-in-front), and the raw y-values for the plot below.
        """
    )
    return


@app.cell
def _(EX, agg, contact, cu, kp, np):
    # ===================== EXERCISE — edit ONLY the two lines marked  # TODO , then run ============
    # The two TODO lines currently hold a deliberately-wrong placeholder, so the self-check below
    # reads "Not yet" until you fix them. Nothing else in this cell needs editing.
    #
    # Part 1 — transform the example event into the approacher's body frame.
    #   cu.allocentricize(event) · input: (T,3,15,2) arena coords · output: same shape, with the
    #   approacher's tail-base at the origin and its heading pointing +y.
    bf_ex = kp[EX].astype(float)         # TODO: wrap this in cu.allocentricize(...) so it is body-framed
    #
    # (Part 1 check — nothing to edit here. It reads the approacher's tail-base + head from the first
    #  fully-tracked frame and checks the tail-base is at (0,0) and the heading is (0,1).)
    _ho = np.isfinite(kp[EX].astype(float)[:, 0, cu.HEAD]).all(1)
    _to = np.isfinite(kp[EX].astype(float)[:, 0, cu.TTI]).all(1)
    _anchor = np.where(_ho & _to)[0][0]
    _tti = bf_ex[_anchor, 0, cu.TTI]
    _head = bf_ex[_anchor, 0, cu.HEAD]
    _heading = (_head - _tti) / (np.linalg.norm(_head - _tti) + 1e-12)
    ex_ok = bool(np.allclose(_tti, 0.0, atol=1e-3) and np.allclose(_heading, [0.0, 1.0], atol=1e-2))

    # Part 2 — front vs rear, over every event. (allocentricize is a cheap einsum, so looping over
    #   ~1500 events runs in well under a second.) This loop and the y-values are written for you.
    front_y, front_lab = [], []
    for _i in range(len(kp)):
        _bf = cu.allocentricize(kp[_i].astype(float))
        _t = min(int(contact[_i]), _bf.shape[0] - 1)
        _p = _bf[_t, 1, cu.TTI]                     # approachee tail-base in the approacher's frame
        if np.isfinite(_p).all():
            front_y.append(float(_p[1]))
            front_lab.append(int(agg[_i]))
    _y = np.array(front_y); _lab = np.array(front_lab)
    _in_front = _y > _y.max() + 1.0      # TODO: the approachee is in front when its y is positive: use  _y > 0
    front_diff = float(_in_front[_lab == 1].mean() - _in_front[_lab == 0].mean())
    # ==============================================================================================
    return ex_ok, front_diff, front_lab, front_y


@app.cell
def _(front_lab, front_y, go, np):
    # Expected picture: two histograms, both piled up at y > 0 (dashed line), heavily overlapping.
    # If aggression arrived from a different direction, the red distribution would shift off the gray
    # one — it does not.
    _y = np.array(front_y, float); _lab = np.array(front_lab, int)
    _fig = go.Figure()
    _fig.add_histogram(x=_y[_lab == 0], name="not aggression",
                       marker_color="#7f7f7f", opacity=0.6, nbinsx=40)
    _fig.add_histogram(x=_y[_lab == 1], name="aggression",
                       marker_color="#d62728", opacity=0.6, nbinsx=40)
    _fig.add_vline(x=0.0, line_dash="dash", line_color="#333")
    _fig.update_layout(barmode="overlay", template="plotly_white", height=380,
                       title="approachee tail-base y in the approacher's body frame at contact "
                             "(y > 0 = in front)",
                       xaxis_title="y (px, body frame)", yaxis_title="event count",
                       margin=dict(l=10, r=10, t=50, b=10), legend=dict(y=1.0))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            Fill the two blanks like this:

            ```python
            bf_ex = cu.allocentricize(kp[EX].astype(float))   # Part 1 blank
            _in_front = _y > 0                                 # Part 2 blank
            ```

            **What you should find:** the Part 1 check passes exactly (`ex_ok` is `True`), and
            `front_diff` is **tiny** (about +0.01). In the plot, both histograms sit almost entirely at
            y > 0 and overlap heavily. Almost every approach is frontal by construction — the approacher
            is, after all, approaching — so aggression does **not** arrive from a different direction.
            The honest answer is *no*: the aggression signal lives in the **kinematic** features (speed,
            angular velocity; recall `appe_angvel` had a large Cohen's d), not in the approach angle.
            That is exactly why we keep all 19 features instead of a geometry-only summary.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(ex_ok, front_diff, mo):
    # Self-check with a tolerance band. Part 1 is exact; the GRADED-CORRECT answer for Part 2 is the
    # honest null: the front-vs-rear difference is SMALL (|diff| < 0.10; pinned full-corpus value
    # ~0.01), i.e. aggression does NOT arrive from a different direction.
    _p1 = bool(ex_ok)
    _p2 = abs(float(front_diff)) < 0.10
    _ok = _p1 and _p2
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _msg1 = ("Part 1: approacher lands at the origin facing +y." if _p1
             else "Part 1: transform check failed — origin or heading is off. Fill bf_ex with "
                  "cu.allocentricize(kp[EX].astype(float)).")
    _msg2 = (f"Part 2: front-vs-rear difference is small ({front_diff:+.3f}) — aggression does NOT "
             "arrive from a different direction; the signal is kinematic."
             if _p2 else
             f"Part 2: your front_diff = {front_diff:+.3f} is implausibly large — check the y > 0 "
             "test and the body-frame y-sign.")
    _head = "PASS — both parts correct" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_msg1}<br>{_msg2}<br>
        <span style="font-size:0.9em;color:#555">The graded answer for Part 2 is the honest null, so
        the exercise is not scored against noise. Tolerance band: |front_diff| &lt; 0.10.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Held-out test cage — set aside now, evaluated in NB08. Stated plainly (no redaction theatrics).
    mo.md(
        r"""
        ---
        > **Held-out test cage (Camera 16).** One cage — 470 events — is set aside and never used for
        > training or for inspecting features. Its 19 features already exist; we simply do not look at
        > them until NB08, where the decoder is evaluated on this cage to measure how well it
        > generalizes to an animal it has never seen. Keeping it untouched is what makes that later
        > number trustworthy.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What the transform discards, and how it can break

        **Discarded on purpose.** The body-frame transform deliberately drops *where in the cage* the
        event happened and *which way the approacher faced in the arena*. That is the goal, but it
        means you can **no longer ask arena questions** from `X` alone (does aggression cluster near a
        wall? at the food hopper?). Those need the raw coordinates back.

        **Failure modes on this dataset.**

        1. **Silent fallback to raw coordinates.** If the approacher's head or tail-base is missing on
           *every* frame, `allocentricize` cannot find a heading and returns the event **unchanged** —
           the features are then computed in raw arena coordinates and are *not* invariant. This is
           invisible unless you audit for it, and head/tail-chain dropout are exactly the nodes NB01
           flagged as least reliable.
        2. **One bad frame rotates the whole scene.** The transform reads the heading from a *single*
           anchor frame. If that frame's head or tail-base is jittery, the entire event is rotated to
           the wrong angle and every geometry feature is corrupted, with no error raised.
        3. **Angle carries little here.** As the exercise showed, approach *direction* barely separates
           aggression (`front_diff` ≈ 0.01). A pipeline that leaned on approach angle would find almost
           nothing; the real signal is kinematic.

        **A question to sit with.** We factored heading *out* to get invariance. But heading is also
        information. What would you gain by keeping *both* representations — the body-frame features and
        the raw arena pose — instead of collapsing to one? (We keep the raw coordinates on disk for
        exactly this reason.)
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

        You built the deliverable: **`X (1500, 19)`** — every event as one arena-invariant vector of
        social geometry and kinematics. Gauge A fell from **11,700 raw numbers to 19**. And we learned
        something concrete: aggression is not a matter of *direction*, it is a matter of *motion*.

        Nineteen numbers is still a lot to look at, and they are far from independent — speed, closing
        speed, and distance clearly move together. **Next (NB03): reading the signal in time.** Before
        we compress these 19, we will look at them in value, in time, and in frequency — the way a
        physiologist reads a raw trace — and measure who moves first between two coupled mice.
        """
    )
    return


if __name__ == "__main__":
    app.run()

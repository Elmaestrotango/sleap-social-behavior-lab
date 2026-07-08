# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "plotly>=5.20",
#     "h5py>=3.10",
#     "gdown>=5.1",
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
        # NB12 · Place & Grid Cells — the world-map, closed

        > **WEEK 2 — THE NEURAL TWIN**
        >
        > Back in **NB02** you built the *egocentric* transform by hand: translate onto the mouse,
        > rotate its heading to +y, and re-express the social scene in the animal's own body frame.
        > At the end of that notebook we named the thing it *pointed at* but never built — the
        > **allocentric** world-map: the fixed, arena-anchored frame the brain also holds. We said
        > that **place cells** (O'Keefe & Dostrovsky 1971) and **grid cells** (Hafting 2005) are the
        > cells that *carry* that map, and left them sealed.
        >
        > **Today we open them.** Given a trajectory and spike trains, we draw the actual thing:
        > a **2-D spatial rate map**. And here is the twin, stated plainly —
        >
        > **A 2-D rate map is a tuning curve.** In NB02 a *tuning curve* was "how strongly does this
        > readout fire as a function of one variable" (facing cosine, Georgopoulos-style). A place
        > field is the exact same object with the variable being **(x, y) location** instead of angle.
        > Firing-as-a-function-of-a-variable — that is the shared computational move. Week-1 read
        > tuning to *social geometry*; today we read tuning to *space*, and it closes NB02's loop:
        > the allocentric endpoint the egocentric transform was reaching for.

        **The rig for today.** A rat runs while its two eyes are tracked; the centroid of the two eye
        positions is our position proxy, and simultaneously recorded neurons emit spikes. Our job:
        turn *(where was the animal)* + *(when did each neuron fire)* into *(where does each neuron
        like to be)*.
        """
    )
    return


@app.cell
def _(ROOT, nu):
    import os as _os
    # Download + unzip the NEU 457 rat place/grid data (Dropbox), then parse all three sessions.
    # fetch_zip_dropbox is cached: it only re-downloads if the .mat files are missing.
    _rat_dir = nu.fetch_zip_dropbox(root=ROOT)
    sessions = {name: nu.load_rat_mat(_os.path.join(_rat_dir, name)) for name in nu.RAT_FILES}
    session_names = list(nu.RAT_FILES)
    return session_names, sessions


@app.cell(hide_code=True)
def _(mo, sessions, session_names):
    _rows = "\n".join(
        f"| `{n}` | {sessions[n]['spikes'].shape[0]:,} | {sessions[n]['spikes'].shape[1]} |"
        for n in session_names
    )
    mo.md(
        f"""
        ---
        ## 1. The raw materials — a path and some spikes

        Three recording sessions loaded. Each `.mat` gives you `left` and `right` eye positions
        `(T, 2)`, a spike-count matrix `spikes` `(T, n_neurons)`, and the derived
        `centroid = (left + right) / 2` — the animal's position over time.

        | session | frames T | neurons |
        |---|---|---|
        {_rows}

        Pick a session below; it drives every plot in the notebook. Start with the first one
        (`20160609T194655`) — it has the most neurons and the cleanest fields.
        """
    )
    return


@app.cell
def _(mo, session_names):
    session_pick = mo.ui.dropdown(options=session_names, value=session_names[0],
                                  label="recording session (drives all plots)")
    return (session_pick,)


@app.cell
def _(go, mo, np, session_pick, sessions):
    # Trajectory: the animal's position over the whole session, colored by time.
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _ok = np.isfinite(_ctr).all(axis=1)
    _c = _ctr[_ok]
    _fig = go.Figure()
    _fig.add_scatter(x=_c[:, 0], y=_c[:, 1], mode="lines",
                     line=dict(color="#c9c9c9", width=1), name="path", showlegend=False)
    _fig.add_scatter(x=_c[:, 0], y=_c[:, 1], mode="markers",
                     marker=dict(color=np.arange(len(_c)), colorscale="Viridis", size=3,
                                 colorbar=dict(title="frame"), showscale=True),
                     name="time", showlegend=False, opacity=0.6)
    _fig.update_yaxes(scaleanchor="x", scaleratio=1, title="eye-centroid y (px)")
    _fig.update_xaxes(title="eye-centroid x (px)")
    _fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=50, b=10),
                       title=f"Trajectory · {session_pick.value} · {len(_c):,} tracked frames")
    mo.vstack([session_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Spikes on the path — the raw place-field percept

        The single most important picture in place-cell physiology, and the one O'Keefe drew by hand
        in 1971: **overlay the spikes onto the trajectory.** Every red dot is the animal's position
        at a frame where the chosen neuron fired. If the red dots pile up in one region and stay off
        everywhere else the animal went, you are *looking at a place field* — no statistics yet, just
        the eye.

        Slide `neuron` and watch the red cloud move (or not). Some neurons paint a tight patch;
        others sprinkle everywhere.
        """
    )
    return


@app.cell
def _(mo):
    neuron_ov = mo.ui.slider(0, 13, value=5, step=1, label="neuron",
                             debounce=True, full_width=True)
    return (neuron_ov,)


@app.cell
def _(mo, neuron_ov, np, nu, session_pick, sessions):
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _n = _spk.shape[1]
    _ni = min(neuron_ov.value, _n - 1)                 # clamp: sessions have 14 / 6 / 5 neurons
    _spiking = (_spk[:, _ni] > 0) & np.isfinite(_ctr).all(axis=1)
    _fig = nu.overlay_fig(
        _ctr[np.isfinite(_ctr).all(axis=1)], _ctr[_spiking],
        title=f"neuron {_ni} · {int(_spiking.sum())} spike-frames · {session_pick.value}",
        traj_name="path", pts_name="spikes", height=520)
    mo.vstack([neuron_ov, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. From dots to a field — kernel density estimate

        The dot cloud is suggestive but hard to compare between neurons. Smooth it into a continuous
        density with a **Gaussian KDE** (this is what seaborn's `kdeplot` did in the original problem
        set). Each spike drops a little Gaussian bump on the arena; sum them and you get a smooth
        "where does this neuron fire" surface.

        The **bandwidth** is the whole game — the same bias/variance knob as smoothing any histogram.
        Too small and the field shatters into individual spikes; too large and every field melts into
        one blob. Drag it and find the setting where a real field holds together but a diffuse neuron
        stays diffuse.
        """
    )
    return


@app.cell
def _(mo):
    kde_bw = mo.ui.slider(0.2, 2.0, value=0.6, step=0.1, label="bandwidth (× Scott default)",
                          debounce=True, full_width=True)
    return (kde_bw,)


@app.cell
def _(go, kde_bw, mo, neuron_ov, np, session_pick, sessions):
    from scipy.stats import gaussian_kde
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _ni = min(neuron_ov.value, _spk.shape[1] - 1)
    _spiking = (_spk[:, _ni] > 0) & np.isfinite(_ctr).all(axis=1)
    _pts = _ctr[_spiking]
    _finite = _ctr[np.isfinite(_ctr).all(axis=1)]
    _x0, _x1 = _finite[:, 0].min(), _finite[:, 0].max()
    _y0, _y1 = _finite[:, 1].min(), _finite[:, 1].max()
    _xg = np.linspace(_x0, _x1, 80)
    _yg = np.linspace(_y0, _y1, 80)
    _title = f"KDE place field · neuron {_ni} · bw ×{kde_bw.value:.1f}"
    if len(_pts) >= 8 and np.ptp(_pts[:, 0]) > 0 and np.ptp(_pts[:, 1]) > 0:
        _k = gaussian_kde(_pts.T)
        _k.set_bandwidth(_k.factor * kde_bw.value)      # seaborn-style bw_adjust multiplier
        _XX, _YY = np.meshgrid(_xg, _yg)
        _Z = _k(np.vstack([_XX.ravel(), _YY.ravel()])).reshape(_XX.shape)
        _fig = go.Figure(go.Contour(x=_xg, y=_yg, z=_Z, colorscale="Inferno",
                                    contours=dict(coloring="fill"),
                                    colorbar=dict(title="density")))
        _fig.add_scatter(x=_pts[:, 0], y=_pts[:, 1], mode="markers",
                         marker=dict(color="rgba(255,255,255,0.35)", size=3),
                         name="spikes", showlegend=False)
    else:
        _fig = go.Figure()
        _fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                            text=f"neuron {_ni} has too few spikes for a KDE ({len(_pts)})")
        _title += " — too sparse"
    _fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y (px)")
    _fig.update_xaxes(title="x (px)")
    _fig.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=50, b=10),
                       title=_title)
    mo.vstack([kde_bw, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. The honest version — an occupancy-normalized rate map

        The KDE has a bias the 2025 problem set flagged: it shows **where the neuron fired**, not
        **where the neuron fires *given* the animal was there.** A cell can look like it "prefers" a
        spot simply because the animal *spent all its time there*. The fix is the definition of a
        proper place field:

        $$\text{rate}(x,y) = \frac{\text{spikes in bin }(x,y)}{\text{frames spent in bin }(x,y)}$$

        `nu.rate_map` bins position, sums spikes per bin, and divides by **occupancy** — so a bin the
        animal barely visited but fired in twice reads as a genuinely high rate, and a bin it sat in
        forever without firing reads low. This is the neural twin's punchline again: it is a
        **tuning curve over 2-D space**, occupancy-corrected exactly the way you'd correct a
        stimulus-tuning curve for uneven stimulus sampling.

        The `bins` knob is your spatial resolution. Coarse bins are stable but blurry; fine bins are
        sharp but noisy (many bins get one or zero visits).
        """
    )
    return


@app.cell
def _(mo):
    rm_bins = mo.ui.slider(8, 40, value=20, step=2, label="spatial bins (per axis)",
                           debounce=True, full_width=True)
    return (rm_bins,)


@app.cell
def _(mo, neuron_ov, nu, rm_bins, session_pick, sessions):
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _ni = min(neuron_ov.value, _spk.shape[1] - 1)
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _ni], bins=int(rm_bins.value))
    _si = nu.spatial_information(_rm["rate"], _rm["occupancy"])
    # histogram2d indexes [x, y]; transpose so x is horizontal and y vertical in the image.
    _xc = 0.5 * (_rm["xedges"][:-1] + _rm["xedges"][1:])
    _yc = 0.5 * (_rm["yedges"][:-1] + _rm["yedges"][1:])
    _fig = nu.heatmap_fig(
        _rm["rate"].T, x=_xc, y=_yc,
        title=f"rate map · neuron {_ni} · {int(rm_bins.value)} bins · Skaggs SI = {_si:.3f} bits/spike",
        xlabel="x (px)", ylabel="y (px)", colorscale="Inferno",
        colorbar_title="spikes/frame", height=520)
    mo.vstack([rm_bins, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Which neurons are spatial? — Skaggs information

        A rate map gives one number per neuron that captures "how much does knowing the animal's
        position tell you about firing": the **Skaggs spatial information** (bits/spike),

        $$\text{SI} = \sum_i p_i \, \frac{r_i}{\bar r}\, \log_2\!\frac{r_i}{\bar r},$$

        with $p_i$ = occupancy probability of bin $i$, $r_i$ its rate, $\bar r$ the mean rate. A flat
        map $\to$ 0 bits; a sharp single field $\to$ large positive. Below, every neuron in the chosen
        session, ranked. This bar chart is your shopping list for the exercise — but read the honest
        caveat in the next section before you trust the tallest bar.
        """
    )
    return


@app.cell
def _(go, mo, np, nu, session_pick, sessions):
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _sis, _nsp = [], []
    for _i in range(_spk.shape[1]):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _i], bins=20)
        _sis.append(nu.spatial_information(_rm["rate"], _rm["occupancy"]))
        _nsp.append(int((_spk[:, _i] > 0).sum()))
    _sis = np.array(_sis)
    _fig = go.Figure(go.Bar(
        x=[f"n{_i}" for _i in range(len(_sis))], y=_sis, marker_color="#4c78a8",
        text=[f"{_n} spk" for _n in _nsp], textposition="outside",
        hovertext=[f"neuron {_i}: {_sis[_i]:.3f} bits/spike, {_nsp[_i]} spikes"
                   for _i in range(len(_sis))]))
    _fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=50, b=10),
                       title=f"Skaggs spatial information by neuron · {session_pick.value} · 20 bins",
                       yaxis_title="bits/spike", xaxis_title="neuron")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. Exercise — is there an *honest* place cell here?

        **Hypothesis banner.** *This recording contains at least one genuine place-like cell: a
        neuron whose spatial information beats what you'd get from the **same spikes** scattered at
        random positions.*

        Raw SI is not enough. A neuron that fires only 80 times can post a **huge** SI purely because
        a handful of spikes land in a handful of rarely-visited bins — pure sparsity, not tuning. The
        standard cure is a **shuffle null**: circularly shift the spike train (breaking the
        spike↔position pairing while preserving spike count and autocorrelation), recompute SI many
        times, and ask whether the *real* SI clears the shuffle's 95th percentile.

        **Toolbox.**

        - `sessions["20160609T194655.mat"]` → dict with `centroid (T,2)` and `spikes (T,n)`.
        - `nu.rate_map(x, y, spikes, bins=20)` → dict with `rate`, `occupancy`.
        - `nu.spatial_information(rate, occupancy)` → SI in bits/spike.
        - `np.roll(spikes_col, shift)` — circular shift; `np.random.default_rng(0)` for the shifts.

        **Your job.** For **neuron 5** of session `20160609T194655.mat` (a mid-count candidate):
        compute its observed SI, then build a 50-sample circular-shift null and take its 95th
        percentile as the chance band. Report `si_obs` and `si_band`, and whether the cell clears
        the band.

        Fill in `si_obs` and `si_band` below, then run the self-check.
        """
    )
    return


@app.cell
def _(np, nu, sessions):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    _d = sessions["20160609T194655.mat"]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _col = _spk[:, 5]                                   # candidate: neuron 5

    def _si_of(_s):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _s, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])

    si_obs = float(_si_of(_col))

    _rng = np.random.default_rng(0)
    _null = np.array([
        _si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
        for _ in range(50)
    ])
    si_band = float(np.percentile(_null, 95))          # 95th-percentile chance band
    # ---------------------------------------------------------------------------------------------
    return si_band, si_obs


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            d = sessions["20160609T194655.mat"]
            ctr, spk = d["centroid"], d["spikes"]
            col = spk[:, 5]

            def si_of(s):
                rm = nu.rate_map(ctr[:, 0], ctr[:, 1], s, bins=20)
                return nu.spatial_information(rm["rate"], rm["occupancy"])

            si_obs = si_of(col)                                    # ≈ 1.260 bits/spike
            rng = np.random.default_rng(0)
            null = np.array([si_of(np.roll(col, int(rng.integers(1000, len(col) - 1000))))
                             for _ in range(50)])
            si_band = np.percentile(null, 95)                      # ≈ 0.65–0.68
            place_like = si_obs > si_band                          # True
            ```

            **What you should find.** Neuron 5 posts `si_obs ≈ 1.26` bits/spike, and its shuffle band
            sits around `0.65–0.68`. The real cell clears the band by a wide margin (z ≈ +15), so
            **yes — this is an honest place-like cell.** The loop is closed: NB02's allocentric
            world-map is not a metaphor, it is *this neuron*, tuned to a patch of the arena.

            **The trap (why we shuffle).** If you had instead picked the neuron with the *highest raw*
            SI — **neuron 10, SI ≈ 2.14** — you would have been fooled. Neuron 10 fires only ~80
            times, and its *own* shuffle band is ≈ 2.45: it **fails** the test. High SI from few
            spikes is a sparsity artifact, not a place field. The honest conclusion of this dataset:
            place-like cells exist (neurons 4, 5, 6, …), but SI alone — without a spike-matched null —
            will hand you false positives.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, si_band, si_obs):
    # Honest self-check. Part A: the observed SI is pinned from real data (neuron 5, 20 bins) =
    # 1.2601. Part B: the graded-correct conclusion is that this cell CLEARS a spike-matched shuffle
    # band that lands well below it (pinned band ≈ 0.65–0.68 across seeds). We grade the honest
    # conclusion — a genuine place cell — not a raw SI leaderboard (which would crown the sparse
    # neuron 10).
    _a = abs(float(si_obs) - 1.2601) < 0.03
    _b = (0.45 < float(si_band) < 0.95) and (float(si_obs) > float(si_band) + 0.3)
    _ok = _a and _b
    _c = "#e8f5e9" if _ok else "#ffebee"
    _bd = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"✅ observed SI = {si_obs:.3f} bits/spike (neuron 5)" if _a
           else f"❌ si_obs = {si_obs:.3f} — expected ≈ 1.260 for neuron 5 at 20 bins")
    _m2 = (f"✅ chance band = {si_band:.3f}; the cell clears it by a wide margin — an honest "
           "place-like cell, NB02's allocentric map made concrete"
           if _b else
           f"❌ chance band = {si_band:.3f} looks off — did you circular-shift the SAME spike column "
           "and take the 95th percentile of 50 shuffles?")
    _head = "PASS — you found a real place cell" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_bd};padding:12px 16px;border-radius:6px">
        <b style="color:{_bd}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Graded on the honest conclusion: a spike-matched
        shuffle null, not a raw-SI leaderboard. Tolerance: |si_obs − 1.260| &lt; 0.03, band ∈
        (0.45, 0.95), and si_obs exceeds band by &gt; 0.3.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The papers, the credit, and where the analogy stops": mo.md(
            r"""
            **Place cells.** O'Keefe & Dostrovsky 1971, *Brain Res.* 34:171 — a hippocampal neuron
            that fires only when the animal occupies a particular location. The founding observation
            of the cognitive-map program (O'Keefe & Nadel 1978).

            **Grid cells.** Hafting, Fyhn, Molden, Moser & Moser 2005, *Nature* 436:801 — medial
            entorhinal neurons that fire on a periodic triangular lattice tiling the whole
            environment. Together, place + grid + head-direction cells are the **allocentric** world
            map NB02 pointed at (2014 Nobel Prize, O'Keefe / Moser / Moser).

            **Credit.** This analysis is adapted from the **NEU 457** (Princeton) problem set by
            **Talmo Pereira, Andrew Leifer, and David Tank.** The data and the KDE/rate-map framing
            are theirs; we rebuilt it interactive and added the occupancy normalization and the
            shuffle null.

            **Where the analogy stops — read this.** The "position" here is the **centroid of two
            tracked eye positions**, i.e. a *gaze/eye-in-head proxy*, not a clean allocentric
            body-position readout on a foraging arena the way the classic place-cell rigs measure it.
            So the fields you see are weaker and noisier than a textbook O'Keefe field, and some of
            what looks "spatial" may be gaze- or movement-coupled rather than pure allocentric place
            coding. That is exactly why the exercise grades the **shuffle-corrected** conclusion and
            not the prettiest map: with a proxy signal, honest nulls are the difference between a
            place cell and a sparsity artifact.
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

        NB02 built the **egocentric** transform and named — but sealed — its counterpart, the
        **allocentric** world-map. Today you unsealed it: a **2-D occupancy-normalized rate map is a
        tuning curve over space**, the same firing-as-a-function-of-a-variable move Week-1 used for
        social geometry, and at least one neuron in this rat genuinely tiles the arena. The loop that
        opened in NB02 is closed.

        You also met the honest failure mode that will follow us into the harder neural notebooks:
        **a big number is not a result until it beats a matched null.** Sparse spikes inflate SI;
        a proxy position variable muddies the field. Shuffle first, believe second.

        **Next (NB13): from one animal's map to a colony's state.** We leave the clean single-rat rig
        for the **social-isolation** dataset — behavior bouts + calcium across many sessions — and ask
        the messier question these tools were built for: *does the brain's read-out of another animal
        change when you take the group away?*
        """
    )
    return


if __name__ == "__main__":
    app.run()

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
        # NB12 · Place and grid cells

        **Why this matters.** In Week 1 you described social behavior in each mouse's own body
        frame: you translated the coordinate system onto the animal and rotated it so the animal's
        heading pointed a fixed direction (the *egocentric* view, NB02). That was a choice of
        reference frame. The brain also uses a second reference frame: a fixed, arena-anchored map
        of the outside world (the *allocentric* view). This notebook works with real neural
        recordings that show how that map is built, so the idea "space is represented in the brain"
        stops being an assertion and becomes something you can measure.

        **Definitions you will need.**

        - **Egocentric frame:** the world described relative to the animal's own body (left/right,
          front/back). This is what you built in Week 1.
        - **Allocentric frame:** the world described relative to fixed external landmarks (a corner
          of the arena, a wall). It does not move when the animal turns.
        - **Place cell** (O'Keefe & Dostrovsky, 1971): a neuron that fires only when the animal is
          in one particular location in the arena. The region it fires in is its *place field*.
        - **Grid cell** (Hafting et al., 2005): a neuron that fires at many locations arranged in a
          repeating triangular pattern that tiles the whole arena.

        Place cells and grid cells are the neurons that carry the allocentric map. Together with
        head-direction cells they earned the 2014 Nobel Prize (O'Keefe; May-Britt and Edvard Moser).

        **What we will do (the method).** We have, for each recording session, the animal's position
        over time and the spike times of several neurons. We will turn *(where the animal was)* plus
        *(when each neuron fired)* into *(where in the arena each neuron prefers to fire)*. The main
        picture we build is a **rate map**, defined in Section 4.

        **One honest caveat, stated up front.** In this dataset the animal's "position" is not a
        clean body-position readout. It is the average of two tracked **eye** positions, so it is a
        gaze / eye-in-head proxy rather than the body-on-arena position that classic place-cell rigs
        measure. The fields you see will be weaker and noisier than a textbook place field, and some
        apparent tuning may reflect gaze or movement rather than pure location. This is why the final
        exercise tests each cell against a control, not against the prettiest map.
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
        ## 1. The inputs — a path and some spikes

        **Why.** To ask "where does this neuron prefer to fire?" we need exactly two things aligned
        in time: the animal's location at every frame, and the number of spikes each neuron produced
        at every frame. Everything else in the notebook is built from these two arrays.

        **Definitions.**

        - **Session:** one continuous recording. We loaded three.
        - **Eye positions** `left`, `right`: the tracked (x, y) pixel positions of the two eyes,
          shape `(T, 2)` where `T` is the number of frames.
        - **Centroid:** `centroid = (left + right) / 2`, shape `(T, 2)`. This is our estimate of the
          animal's position at each frame (the gaze proxy described above).
        - **Spike-count matrix** `spikes`: shape `(T, n_neurons)`; entry `[t, j]` is how many times
          neuron `j` fired in frame `t`.

        | session | frames T | neurons |
        |---|---|---|
        {_rows}

        Use the dropdown below to choose which session drives every plot. Start with the first one
        (`20160609T194655`); it has the most neurons and the clearest fields.
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
        The plot above shows the whole path the animal took, with color running from early (dark)
        to late (yellow) frames. It tells you which parts of the arena were visited, and how often.
        A region the animal returned to many times will look densely painted; a region it rarely
        entered will be nearly empty. Keep this coverage in mind: a neuron cannot show a field in a
        place the animal never went, and Section 4 will correct for uneven coverage directly.

        The next cell replays the same path frame by frame. Press the play button. Watching the
        animal move through the arena makes it clearer that "position" is a value that changes
        continuously over the session.
        """
    )
    return


@app.cell
def _(go, np, session_pick, sessions):
    # Animated replay of the trajectory: a moving dot leaves a growing trail. Subsampled and capped
    # at ~40 animation frames so the exported HTML stays light. Static export shows the first frame.
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _c = _ctr[np.isfinite(_ctr).all(axis=1)]
    _step = max(1, len(_c) // 1200)
    _cs = _c[::_step]
    _cuts = np.linspace(2, len(_cs), 40).astype(int)
    _frames = [
        go.Frame(data=[
            go.Scatter(x=_cs[:k, 0], y=_cs[:k, 1], mode="lines",
                       line=dict(color="#9a9a9a", width=1)),
            go.Scatter(x=[_cs[k - 1, 0]], y=[_cs[k - 1, 1]], mode="markers",
                       marker=dict(color="#e45756", size=11)),
        ], name=str(int(k)))
        for k in _cuts
    ]
    _k0 = _cuts[0]
    _fig = go.Figure(
        data=[
            go.Scatter(x=_cs[:_k0, 0], y=_cs[:_k0, 1], mode="lines",
                       line=dict(color="#9a9a9a", width=1), name="path", showlegend=False),
            go.Scatter(x=[_cs[_k0 - 1, 0]], y=[_cs[_k0 - 1, 1]], mode="markers",
                       marker=dict(color="#e45756", size=11), name="animal", showlegend=False),
        ],
        frames=_frames,
    )
    _fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y (px)",
                      range=[_cs[:, 1].min(), _cs[:, 1].max()])
    _fig.update_xaxes(title="x (px)", range=[_cs[:, 0].min(), _cs[:, 0].max()])
    _fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=10, r=10, t=50, b=10),
        title=f"Trajectory replay · {session_pick.value}",
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.08, xanchor="left",
                          buttons=[dict(label="play", method="animate",
                                        args=[None, {"frame": {"duration": 90, "redraw": True},
                                                     "fromcurrent": True}])])])
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Spikes on the path

        **Why.** The simplest way to see whether a neuron is tuned to location is to mark every
        position at which it fired and look at where those marks land. This is the picture O'Keefe
        drew by hand in 1971, and it requires no statistics.

        **Definition.** In the plot below, the gray line is the full trajectory (everywhere the
        animal went). Each red dot is the animal's position at a frame in which the chosen neuron
        fired at least once. If the red dots concentrate in one region while the animal clearly
        visited the whole arena, that region is a candidate place field. If the red dots are spread
        evenly along the path, the neuron is probably not spatially tuned.

        **Method.** `nu.overlay_fig(trajectory, spike_positions, ...)` draws these two layers.
        Its inputs are the full set of tracked positions and the subset of positions where the
        neuron spiked; its output is the figure. Move the `neuron` slider to change which neuron's
        spikes are shown. Some neurons paint a tight patch; others sprinkle everywhere.
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
        title=f"Spikes on path · neuron {_ni} · {int(_spiking.sum())} spike-frames · {session_pick.value}",
        traj_name="path", pts_name="spikes", height=520)
    mo.vstack([neuron_ov, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. From dots to a smooth field — kernel density estimation

        **Why.** The dot cloud is suggestive but hard to compare across neurons: a cluster of dots
        and a slightly denser cluster look similar to the eye. Converting the dots into a smooth
        surface makes the shape and location of a field easier to read and to compare.

        **Definitions.**

        - **Kernel density estimate (KDE):** a way to turn scattered points into a smooth density.
          Place a small Gaussian bump on the arena at each spike position, then add all the bumps
          together. Where spikes are dense the bumps overlap and the surface is high; where spikes
          are sparse the surface is low.
        - **Bandwidth:** the width of each Gaussian bump. It is a smoothing control with a
          trade-off: too small and the surface breaks into one lump per spike (noisy); too large and
          separate fields blur into a single blob (over-smoothed).

        **Method.** `scipy.stats.gaussian_kde` takes the spike positions and returns a function you
        can evaluate on a grid to get the density surface. We start from its automatic bandwidth and
        multiply it by the slider value (the same `bw_adjust` idea used by seaborn's `kdeplot`).
        Drag `bandwidth` and find a setting where a real field holds together while a diffuse neuron
        stays diffuse.

        **Limitation (leads into Section 4).** This surface shows where the neuron *fired*, but not
        where it fires *given that the animal was there*. A neuron can look like it prefers a spot
        simply because the animal spent most of its time in that spot. Section 4 corrects for this.
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
    _title = f"KDE density · neuron {_ni} · bw ×{kde_bw.value:.1f} · {session_pick.value}"
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
        ## 4. The rate map — a spatial tuning curve

        **Why.** We want a picture of *firing rate as a function of location* that is not distorted
        by how the animal spent its time. The correct object is the **rate map**.

        **Definitions.**

        - **Occupancy:** for each small square region (bin) of the arena, the number of frames the
          animal spent in that bin. It measures coverage, not firing.
        - **Rate map:** for each bin, the firing rate

          $$\text{rate}(x,y) = \frac{\text{spikes fired in bin }(x,y)}{\text{frames spent in bin }(x,y)}.$$

          Dividing by occupancy removes the "spent all its time here" confound: a bin the animal
          barely visited but fired in twice reads as a genuinely high rate, and a bin it sat in for a
          long time without firing reads low.
        - **Spatial tuning curve:** the general idea that a neuron's firing rate is a function of
          some variable. In Week 1 that variable was social geometry (for example, facing angle). A
          rate map is the same idea with the variable being the animal's (x, y) location. That is the
          concrete link back to the body-frame work: location is a variable the brain represents, and
          the rate map is how we read that representation out.

        **Method.** `nu.rate_map(x, y, spikes, bins=N)` takes the x and y position arrays and a
        neuron's spike-count column, splits the arena into an `N × N` grid, sums spikes per bin,
        divides by occupancy, and returns a dict with `rate` and `occupancy`. The `bins` slider sets
        the spatial resolution: coarse bins are stable but blurry; fine bins are sharp but noisy
        because many bins get one or zero visits. The map's title also reports its Skaggs spatial
        information, defined next.
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
        title=f"Rate map · neuron {_ni} · {int(rm_bins.value)} bins · SI = {_si:.3f} bits/spike · {session_pick.value}",
        xlabel="x (px)", ylabel="y (px)", colorscale="Inferno",
        colorbar_title="spikes/frame", height=520)
    mo.vstack([rm_bins, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Which neurons are spatial? — Skaggs spatial information

        **Why.** A rate map is a picture; to compare many neurons we want one number per neuron that
        summarizes how spatial it is. That number lets us rank the neurons and pick candidates.

        **Definition.** The **Skaggs spatial information** measures how much knowing the animal's
        location tells you about whether the neuron will fire. It is reported in **bits per spike**:

        $$\text{SI} = \sum_i p_i \, \frac{r_i}{\bar r}\, \log_2\!\frac{r_i}{\bar r},$$

        where $p_i$ is the fraction of time spent in bin $i$ (occupancy probability), $r_i$ is that
        bin's firing rate, and $\bar r$ is the mean rate. A neuron whose rate is the same everywhere
        gives SI $= 0$ bits/spike (location tells you nothing). A neuron with one sharp field gives a
        large positive value.

        **Method.** `nu.spatial_information(rate, occupancy)` takes the two arrays from `nu.rate_map`
        and returns the SI as a single float. The bar chart below computes SI for every neuron in the
        chosen session and ranks them. Read the caveat in Section 6 before trusting the tallest bar:
        a high SI can be produced by very few spikes rather than by real tuning.
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
                       title=f"Spatial information by neuron · {session_pick.value} · 20 bins",
                       yaxis_title="bits/spike", xaxis_title="neuron")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. Exercise — is there a genuine place cell here?

        **Why a raw number is not enough.** A neuron that fires only 80 times can post a very high SI
        purely because a handful of spikes happen to land in a handful of rarely visited bins. That
        is sparsity, not spatial tuning. To tell a real field from a sparsity artifact we compare the
        neuron's SI against a control built from its own spikes.

        **Definitions.**

        - **Shuffle null:** a set of control values produced by breaking the link between spikes and
          position while keeping everything else the same. Here we **circularly shift** the spike
          train — slide it forward in time by a random amount and wrap the end around to the start
          (`np.roll`). This keeps the exact spike count and the train's temporal structure but pairs
          the spikes with the wrong positions, so any spatial information it produces is by chance.
        - **Chance band:** the 95th percentile of the shuffled SI values. A real cell should have an
          observed SI above this band.

        **Tools.**

        - `sessions["20160609T194655.mat"]` → dict with `centroid (T, 2)` and `spikes (T, n)`.
        - `nu.rate_map(x, y, spikes, bins=20)` → dict with `rate`, `occupancy`.
        - `nu.spatial_information(rate, occupancy)` → SI in bits/spike.
        - `np.roll(spike_column, shift)` → circular shift; `np.random.default_rng(0)` for the shifts.

        **Your task.** For **neuron 5** of session `20160609T194655.mat`, compute the observed SI and
        a 50-sample shuffle null, then take the null's 95th percentile as the chance band. The cell
        below is written for you except for **two blanks** marked `____`. Fill them in:

        1. `si_obs` — pass the real (unshuffled) spike column `_col` into `_si_of`.
        2. `si_band` — take the `95`th percentile of the shuffled values.

        **What you should see.** The plot underneath draws the 50 shuffled SI values as a gray
        histogram, the chance band as a dashed line, and your observed SI as a solid red line. If you
        filled the blanks correctly the gray histogram sits low (around 0.3–0.7 bits/spike), the
        dashed band sits near 0.65, and the red observed line is far to the right near 1.26 — clearly
        past the band. Then run the self-check below it.
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
        # PURPOSE: spatial information of one spike column. INPUT: a (T,) spike-count array.
        # OUTPUT: its Skaggs SI in bits/spike (a single float).
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _s, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])

    # TODO 1: compute the observed SI of neuron 5's REAL spikes.
    # Replace ____ with _col (the unshuffled spike column defined above).
    si_obs = float(_si_of(_col))          # <-- the ____ line (already filled with the answer)

    _rng = np.random.default_rng(0)
    _null = np.array([
        _si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
        for _ in range(50)
    ])

    # TODO 2: the chance band is the 95th percentile of the 50 shuffled SI values.
    # Replace ____ with 95.
    si_band = float(np.percentile(_null, 95))   # <-- the ____ line (already filled with the answer)
    # ---------------------------------------------------------------------------------------------
    return si_band, si_obs


@app.cell
def _(go, np, nu, sessions, si_band, si_obs):
    # Result plot for the exercise: the shuffle null (gray histogram) with your observed SI (red)
    # and chance band (dashed). The null is recomputed here with the canonical settings so the
    # picture is stable; the two vertical lines show the values YOU produced above. This plot always
    # uses the exercise session (20160609T194655), not the dropdown.
    _d = sessions["20160609T194655.mat"]
    _ctr = _d["centroid"]
    _col = _d["spikes"][:, 5]

    def _si_of(_s):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _s, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])

    _rng = np.random.default_rng(0)
    _null = np.array([
        _si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
        for _ in range(50)
    ])
    _fig = go.Figure()
    _fig.add_histogram(x=_null, nbinsx=18, marker_color="#c9c9c9", name="shuffled SI")
    _fig.add_vline(x=float(si_band), line=dict(color="#4c78a8", width=2, dash="dash"),
                   annotation_text="95% chance band", annotation_position="top")
    _fig.add_vline(x=float(si_obs), line=dict(color="#e45756", width=3),
                   annotation_text="observed", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10),
                       title="Neuron 5 spatial information vs shuffle null · 20160609T194655",
                       xaxis_title="spatial information (bits/spike)", yaxis_title="shuffles")
    _fig
    return


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

            si_obs = si_of(col)                                    # about 1.260 bits/spike
            rng = np.random.default_rng(0)
            null = np.array([si_of(np.roll(col, int(rng.integers(1000, len(col) - 1000))))
                             for _ in range(50)])
            si_band = np.percentile(null, 95)                      # about 0.65 to 0.68
            place_like = si_obs > si_band                          # True
            ```

            **Result.** Neuron 5 has `si_obs` about 1.26 bits/spike, and its shuffle band sits around
            0.65 to 0.68. The observed value is well above the band, so neuron 5 is a genuine
            place-like cell: a concrete example of the allocentric map this notebook set out to
            measure.

            **Why the shuffle matters.** If you had instead trusted the highest raw SI you would have
            been misled. In this session neuron 10 has the largest raw SI (about 2.14) but fires only
            about 80 times, and its own shuffle band is about 2.45 — so neuron 10 fails the test. High
            SI from very few spikes is a sparsity artifact, not a place field. The reliable
            conclusion is that place-like cells exist here (neurons 4, 5, 6, and others), but SI alone
            without a spike-matched control will produce false positives.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, si_band, si_obs):
    # Self-check. Part A: the observed SI is pinned from real data (neuron 5, 20 bins) = 1.2601.
    # Part B: the correct conclusion is that this cell clears a spike-matched shuffle band that lands
    # well below it (pinned band ~0.65-0.68 across seeds). We grade the shuffle-corrected conclusion
    # (a genuine place cell), not a raw-SI leaderboard (which would wrongly crown the sparse neuron
    # 10).
    _a = abs(float(si_obs) - 1.2601) < 0.03
    _b = (0.45 < float(si_band) < 0.95) and (float(si_obs) > float(si_band) + 0.3)
    _ok = _a and _b
    _c = "#e8f5e9" if _ok else "#ffebee"
    _bd = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"observed SI = {si_obs:.3f} bits/spike (neuron 5)" if _a
           else f"si_obs = {si_obs:.3f} — expected about 1.260 for neuron 5 at 20 bins")
    _m2 = (f"chance band = {si_band:.3f}; the observed SI is well above it, so this is a genuine "
           "place-like cell"
           if _b else
           f"chance band = {si_band:.3f} looks off — did you circular-shift the SAME spike column "
           "and take the 95th percentile of 50 shuffles?")
    _head = "Pass — this is a real place cell" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_bd};padding:12px 16px;border-radius:6px">
        <b style="color:{_bd}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Graded on the shuffle-corrected conclusion, not a
        raw-SI ranking. Tolerance: |si_obs − 1.260| &lt; 0.03, band ∈ (0.45, 0.95), and si_obs
        exceeds the band by &gt; 0.3.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "References, credit, and the limits of this dataset": mo.md(
            r"""
            **Place cells.** O'Keefe & Dostrovsky, 1971, *Brain Research* 34:171 — a hippocampal
            neuron that fires only when the animal occupies a particular location. This was the
            founding observation of the cognitive-map framework (O'Keefe & Nadel, 1978).

            **Grid cells.** Hafting, Fyhn, Molden, Moser & Moser, 2005, *Nature* 436:801 — medial
            entorhinal neurons that fire on a repeating triangular lattice tiling the environment.
            Place, grid, and head-direction cells together form the allocentric map introduced at the
            top of this notebook (2014 Nobel Prize: O'Keefe; May-Britt and Edvard Moser).

            **Credit.** This analysis is adapted from the **NEU 457** (Princeton) problem set by
            **Talmo Pereira, Andrew Leifer, and David Tank.** The data and the KDE / rate-map framing
            are theirs. We rebuilt it as an interactive notebook and added the occupancy
            normalization and the shuffle null.

            **Limits of this dataset.** The "position" used here is the centroid of two tracked eye
            positions — a gaze / eye-in-head proxy, not the clean body-on-arena position that the
            classic place-cell rigs measure. The fields are therefore weaker and noisier than a
            textbook place field, and some apparent tuning may be gaze- or movement-coupled rather
            than pure allocentric location coding. This is exactly why the exercise grades the
            shuffle-corrected conclusion rather than the prettiest map: with a proxy position signal,
            an honest control is what separates a real place cell from a sparsity artifact.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Summary

        In Week 1 you described behavior in each animal's egocentric body frame. This notebook worked
        in the complementary allocentric frame and showed how the brain represents the animal's
        location in the arena. The key object is the **occupancy-normalized rate map**: a spatial
        tuning curve that gives firing rate as a function of (x, y) location, the same
        firing-rate-as-a-function-of-a-variable idea used for social geometry in Week 1. At least one
        neuron in this recording tiles the arena with a genuine place field.

        You also met a failure mode that recurs in the neural notebooks: a large number is not a
        result until it beats a matched control. Sparse spikes inflate spatial information, and a
        proxy position variable adds noise, so the shuffle null comes first and belief comes second.

        **Next (NB13).** We leave the single-animal rig for the social-isolation dataset — behavior
        bouts plus calcium imaging across many sessions — and ask whether the brain's read-out of
        another animal changes when the group is removed.
        """
    )
    return


if __name__ == "__main__":
    app.run()

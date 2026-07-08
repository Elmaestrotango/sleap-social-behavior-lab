import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🐭 SLEAP Social-Behavior Lab

        An interactive course in analyzing **animal social behavior from pose-tracking data** —
        from raw [SLEAP](https://sleap.ai) output all the way to a trained behavior classifier.

        **Work through the lessons in order.** Each one opens as a live notebook: drag a slider or
        edit an input and the analysis re-runs instantly. Your session is private to you.

        **Week 1 — Behavior.**

        | # | Lesson | What you build |
        |---|--------|----------------|
        | 1 | [Pose & Identity](/01) | Read the pose tensor `(frames, mice, nodes, xy)`; why one identity error corrupts everything downstream |
        | 2 | [The Body-Centered View](/02) | Center + rotate into a body frame; the 19 social features and why behavior is rotationally invariant |
        | 3 | [Behavior in Time](/03) | Distributions, wavelet rhythm, and who-leads-whom coordination |
        | 4 | [PCA, Clustering & Statistics](/04) | The behavioral manifold and a UMAP map of syllables — and does sex or food deprivation really change behavior? |
        | 5 | [Dynamics & Decoding](/05) | The transition grammar in time, then a decoder tested across cohorts |

        **Week 2 — From behavior to the brain.** The same computational moves, now on neural
        recordings: calcium imaging, source separation, spatial tuning, and neural decoding.

        | # | Lesson | What you build |
        |---|--------|----------------|
        | 6 | [Motion Correction](/06) | Register a drifting miniscope movie so the signal stops moving |
        | 7 | [Extracting a Calcium Trace](/07) | Background-subtract + ROI → one cell's calcium trace |
        | 8 | [Separating the Neurons](/08) | CNMF footprints + traces from an optical mixture |
        | 9 | [Place & Grid Cells](/09) | Occupancy-normalized rate maps + spatial information, shuffle-tested |
        | 10 | [Decoding Social State](/10) | Social-contact ethograms, then a population decoder reads social state off calcium |

        ---
        Mice are colored by dominance **rank**: 🔴 Dom &nbsp;&nbsp; 🔵 Mid &nbsp;&nbsp; 🟢 Sub.
        """
    )
    return


if __name__ == "__main__":
    app.run()

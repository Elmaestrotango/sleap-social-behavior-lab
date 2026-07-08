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

        | # | Lesson | What you build |
        |---|--------|----------------|
        | 1 | [The Raw Signal](/01) | Load & scrub a real `.slp`; the keypoint tensor `(frames, mice, nodes, xy)`; why identity matters |
        | 2 | [The Body's-Eye View](/02) | Allocentric social feature vectors (center + rotate into a body frame) |
        | 3 | [The Signal in Time](/03) | Distributions, wavelet rhythm, and who-leads-whom coordination — with honest nulls |
        | 4 | [The Collapse I — PCA](/04) | The behavioral manifold; residualization as a *choice* |
        | 5 | [The Collapse II — the Map](/05) | A UMAP behavioral map carved into data-driven syllables |
        | 6 | [Reading the Map](/06) | Cluster enrichment done honestly (χ², Bonferroni, the pseudoreplication reversal) |
        | 7 | [Behavior in Time](/07) | The transition grammar and activity clock (Markov chain vs shuffle null) |
        | 8 | [The Decoder Graduates](/08) | Train an MLP; unlock held-out Cage 16; decode the neural cousin |

        **Week 2 — The Neural Twin.** The same computational moves, now on the brain:
        calcium imaging, demixing, tuning curves, and neural decoding. Each lesson mirrors a
        Week-1 twin.

        | # | Lesson | What you build |
        |---|--------|----------------|
        | 9 | [Motion Correction](/09) | Register a drifting miniscope movie; a motion index that proves it (twin of 01) |
        | 10 | [Extracting a Calcium Trace](/10) | Background-subtract + ROI → one cell's calcium trace (twin of 02) |
        | 11 | [Demixing the Sources](/11) | CNMF footprints + traces from an optical mixture; a neural sequence (twin of 04) |
        | 12 | [Place & Grid Cells](/12) | Occupancy-normalized rate maps + Skaggs information, shuffle-tested (twin of 02/tuning) |
        | 13 | [Social Ethograms](/13) | `(9, T)` social-contact ethograms for the SI3_2022 cohort (twin of 03/05) |
        | 14 | [Neural Social Decode](/14) | A population decoder reads social state off calcium (twin of 08) |

        ---
        Mice are colored by dominance **rank**: 🔴 Dom &nbsp;&nbsp; 🔵 Mid &nbsp;&nbsp; 🟢 Sub.
        """
    )
    return


if __name__ == "__main__":
    app.run()

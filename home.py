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
        | 1 | [Load SLEAP data](/01) | Load & scrub a real `.slp`; the keypoint tensor `(frames, mice, nodes, xy)` |
        | 2 | [Features](/02) | Allocentric social feature vectors (center + rotate into a body frame) |
        | 3 | [Clustering](/03) | A live PCA → residualize → UMAP → HDBSCAN behavioral map |
        | 4 | [Rank stats](/04) | Rank / condition-enriched clusters (χ², Bonferroni) |
        | 5 | [Label exemplars](/05) | Your own aggression / not-aggression labels via a click grid |
        | 6 | [MLP inference](/06) | Train an MLP; evaluate it on a held-out cage |

        ---
        Mice are colored by dominance **rank**: 🔴 Dom &nbsp;&nbsp; 🔵 Mid &nbsp;&nbsp; 🟢 Sub.
        """
    )
    return


if __name__ == "__main__":
    app.run()

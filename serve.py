"""Serve the entire SLEAP Social-Behavior Lab as ONE site with a landing page.

Each lesson is a live marimo app with its own isolated kernel per visitor, so
`numba` / `umap-learn` / `hdbscan` run normally — unlike the WASM / GitHub-Pages
export, which has no in-browser build for those. Share a single URL and students
click through the lessons in order.

Run locally:
    uv run python serve.py                       # -> http://localhost:7860

Or with uvicorn directly (same thing):
    uv run uvicorn serve:app --host 0.0.0.0 --port 7860

Deploy on anything that runs an ASGI app — a Hugging Face Docker Space, or a lab
VM / workstation behind a Cloudflare Tunnel or Tailscale. See DEPLOY.md.
"""
import os

import marimo

_HERE = os.path.dirname(os.path.abspath(__file__))

# The lessons' _find_root() walks up from the working directory looking for a
# folder that holds both course/ and data/. Anchor the process at the repo root
# so that resolves no matter how the server was launched.
os.chdir(_HERE)

_NB = os.path.join(_HERE, "notebooks")

# Landing page at "/", then the fourteen lessons in order at /01 .. /14
# (Week 1: pose/behavior 01–08; Week 2: the neural twin 09–14).
_LESSONS = [
    ("/01", "01_raw_signal.py"),
    ("/02", "02_body_eye_view.py"),
    ("/03", "03_signal_in_time.py"),
    ("/04", "04_collapse_pca.py"),
    ("/05", "05_collapse_map.py"),
    ("/06", "06_reading_the_map.py"),
    ("/07", "07_behavior_in_time.py"),
    ("/08", "08_decoder_graduates.py"),
    ("/09", "09_motion_correction.py"),
    ("/10", "10_calcium_extraction.py"),
    ("/11", "11_demixing_sources.py"),
    ("/12", "12_place_and_grid_cells.py"),
    ("/13", "13_social_ethograms.py"),
    ("/14", "14_neural_social_decode.py"),
]

_builder = marimo.create_asgi_app().with_app(path="", root=os.path.join(_HERE, "home.py"))
for _path, _fname in _LESSONS:
    _builder = _builder.with_app(path=_path, root=os.path.join(_NB, _fname))

app = _builder.build()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))

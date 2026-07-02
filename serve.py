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

# Landing page at "/", then the six lessons in order at /01 .. /06.
_LESSONS = [
    ("/01", "01_load_sleap.py"),
    ("/02", "02_features.py"),
    ("/03", "03_clustering.py"),
    ("/04", "04_rank_stats.py"),
    ("/05", "05_label_exemplars.py"),
    ("/06", "06_mlp_inference.py"),
]

_builder = marimo.create_asgi_app().with_app(path="", root=os.path.join(_HERE, "home.py"))
for _path, _fname in _LESSONS:
    _builder = _builder.with_app(path=_path, root=os.path.join(_NB, _fname))

app = _builder.build()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))

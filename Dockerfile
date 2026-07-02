# Serve the whole course (landing page + 6 lessons) from one container.
# Works on a Hugging Face Docker Space (port 7860) or any container host.
FROM python:3.11-slim

# Build deps in case any wheel falls back to source (numba/llvmlite, hdbscan).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Hugging Face Spaces convention: run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR /app
COPY --chown=user . /app

# Install the pinned environment from pyproject.toml (pulls marimo -> uvicorn).
RUN uv sync

EXPOSE 7860
CMD ["uv", "run", "python", "serve.py"]

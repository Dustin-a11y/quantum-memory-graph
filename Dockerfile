FROM python:3.15-slim

LABEL org.opencontainers.image.source="https://github.com/Dustin-a11y/quantum-memory-graph"
LABEL org.opencontainers.image.description="Knowledge graph + QAOA subgraph optimization for AI agent memory"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Build from THIS checkout. The image used to run `pip install quantum-memory-graph`, which installs
# whatever is on PyPI (1.3.0) — so no repository change was ever reflected in the published image,
# and this file could not be used to test a branch or a release commit.
COPY setup.cfg pyproject.toml README.md ./
COPY quantum_memory_graph ./quantum_memory_graph
COPY kong_graph ./kong_graph
RUN pip install --no-cache-dir ".[serve]"

ENV QMG_SIMILARITY_THRESHOLD=0.3
ENV QMG_DATA_DIR=/data

RUN mkdir -p /data

EXPOSE 8502

CMD ["python", "-m", "quantum_memory_graph", "--host", "0.0.0.0", "--port", "8502"]

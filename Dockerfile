FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/Dustin-a11y/quantum-memory-graph"
LABEL org.opencontainers.image.description="Knowledge graph + QAOA subgraph optimization for AI agent memory"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="1.2.1"

WORKDIR /app

RUN pip install --no-cache-dir quantum-memory-graph

ENV QMG_SIMILARITY_THRESHOLD=0.3
ENV QMG_DATA_DIR=/data

RUN mkdir -p /data

EXPOSE 8502

CMD ["python", "-m", "quantum_memory_graph", "--host", "0.0.0.0", "--port", "8502"]

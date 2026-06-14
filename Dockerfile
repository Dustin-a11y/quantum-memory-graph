FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir quantum-memory-graph

ENV QMG_SIMILARITY_THRESHOLD=0.3
ENV QMG_DATA_DIR=/data

RUN mkdir -p /data

EXPOSE 8502

CMD ["python", "-m", "quantum_memory_graph", "--host", "0.0.0.0", "--port", "8502"]

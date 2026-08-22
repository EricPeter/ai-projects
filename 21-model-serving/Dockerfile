FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
RUN mkdir -p /models
ENV MODEL_DIR=/models PORT=8021
EXPOSE 8021
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8021"]

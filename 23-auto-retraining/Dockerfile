FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data
EXPOSE 8023
CMD ["python", "-m", "src.pipeline"]

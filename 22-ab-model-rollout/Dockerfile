FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
COPY src/ ./src/
EXPOSE 8022
CMD ["uvicorn", "src.router:app", "--host", "0.0.0.0", "--port", "8022"]

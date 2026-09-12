# canary-1b-flash, cache-aware streaming FastConformer/Parakeet, GPU-PB word
# boosting and the Sortformer streaming diarizer all require a recent NeMo.
# 24.09 predates canary-1b-flash and GPU-PB, so pin a current container.
FROM nvcr.io/nvidia/nemo:25.07

WORKDIR /app

# Install project dependencies on top of NeMo base image
COPY pyproject.toml .
RUN pip install --no-cache-dir symspellpy soundfile httpx

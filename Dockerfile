# freelm serve — a local OpenAI-compatible endpoint over free LLM tiers.
#
#   docker run --rm -p 4000:4000 -e GEMINI_API_KEY=... -e FREELM_SERVER_KEY=change-me \
#     ghcr.io/shihabshahrier/freelm
#
# Then use base URL http://localhost:4000/v1 with API key "change-me".
# With no provider keys at all it serves the keyless endpoints (FREELM_KEYLESS=auto).
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . \
 && useradd --system --uid 10001 --home /home/freelm --create-home freelm

USER freelm
# Inside a container the server must listen on all interfaces; set
# FREELM_SERVER_KEY so only your clients can use it.
ENV FREELM_HOST=0.0.0.0 \
    FREELM_PORT=4000 \
    FREELM_CACHE_DIR=/home/freelm/.cache/freelm \
    PYTHONUNBUFFERED=1
EXPOSE 4000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/health', timeout=4)" || exit 1

ENTRYPOINT ["freelm"]
CMD ["serve"]

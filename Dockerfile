FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 && \
    sed '/^torch==/d' requirements.lock > /tmp/requirements-container.lock && \
    pip install --no-cache-dir --requirement /tmp/requirements-container.lock && \
    rm /tmp/requirements-container.lock
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scenarios ./scenarios
RUN mkdir -p /app/results/verified && chown -R app:app /app
USER app
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import torch, traci, vehicular_offloading; print(vehicular_offloading.__version__)"
ENTRYPOINT ["python", "-m", "vehicular_offloading"]
CMD ["--help"]

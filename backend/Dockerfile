# One image for two compose services: `backend` (API) and `worker` (runtime host).
# Debian trixie is pinned explicitly: its nix-bin (2.26) supports flakes well.
FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.3.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nix-bin \
    && rm -rf /var/lib/apt/lists/*

# Build users for single-user (root, daemon-less) Nix inside the container.
RUN groupadd --system nixbld \
    && for n in $(seq 1 10); do \
        useradd --system --no-create-home --home-dir /var/empty --shell /usr/sbin/nologin --groups nixbld "nixbld${n}"; \
    done

COPY docker/nix.conf /etc/nix/nix.conf

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev --no-root --no-ansi

COPY . .

EXPOSE 8000

CMD ["bash", "-lc", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

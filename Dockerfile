FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Create the app user WITH a home directory (-m)
RUN groupadd -r app && useradd -m -r -g app app

# Disable uv runtime cache inside the container
ENV UV_NO_CACHE=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "agent.py", "serve", "--host", "0.0.0.0"]
# The discovery service.
#
# 3.12 rather than the 3.14 the laptop venv uses: the code needs 3.10 for
# `X | None` and nothing newer, and 3.12 has slim images everywhere. Moving up
# is safe whenever the base image exists.
FROM python:3.12-slim

# The service makes outbound HTTPS calls to Gemini and fetches notices for the
# independent read, so it needs a CA bundle. Nothing else: no build toolchain,
# because none of these wheels compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so an edit to a prompt does not reinstall the SDK.
COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

# Only what the service imports. The CLI pipeline, its tests and the enum
# generator stay out of the image: generate_enums.py reads the backend's
# migrations, which are not here, and a file that cannot run is a file
# somebody will eventually try to run.
COPY enums.py vocab.py extract.py crosscheck.py llm.py service.py ./

# Runs unprivileged. The service writes nothing — state belongs to the backend
# — so it has no need of a writable directory either.
RUN useradd --create-home --uid 10001 discovery
USER discovery

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE 8080

# One worker on purpose. Every request is a 30-60 second wait on Gemini, so
# concurrency here buys nothing the backend does not already control: it calls
# one page at a time from one queue job, which is what makes a run's cost
# predictable. Raise it only alongside the worker count on the other side.
#
# Shell form so ${PORT} is expanded — Railway assigns the port and a JSON-form
# CMD would pass the literal string.
CMD uvicorn service:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75

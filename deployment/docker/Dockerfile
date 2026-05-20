FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY backend /app/backend

RUN python -m pip install --no-cache-dir -U pip \
    && python -m pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
CMD ["uvicorn", "backend.api_gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]

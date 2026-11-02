FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY .chainlit ./.chainlit
COPY chainlit.md ./
COPY src ./src

EXPOSE 8001
CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8001", "--headless"]

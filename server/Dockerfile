FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Context is assumed to be repository root: docker build -f server/Dockerfile .
COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy sdk and server sources
COPY sdk /app/sdk
COPY server /app/server

# Install local SDK
RUN pip install --no-cache-dir -e /app/sdk

WORKDIR /app/server

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

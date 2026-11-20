FROM python:3.11-slim
# LightGBM needs the OpenMP runtime, which the slim image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt
COPY features.py rules.py app.py ./
# Nothing here needs root: run as an unprivileged user
RUN useradd --system --no-create-home app
USER app
CMD ["python", "app.py"]

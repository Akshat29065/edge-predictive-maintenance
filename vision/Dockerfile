FROM python:3.11-slim
# onnxruntime needs the OpenMP runtime, which the slim image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY camera.py app.py ./
CMD ["python", "app.py"]
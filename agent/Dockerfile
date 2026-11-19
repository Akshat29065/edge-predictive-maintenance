FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt
COPY tools.py policy.py llm.py app.py ./
COPY prompts ./prompts
CMD ["python", "app.py"]
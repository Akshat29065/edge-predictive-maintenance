#!/bin/sh
# Starts the Ollama server, downloads the models on first run, then marks itself ready.
# A failed download (offline, flaky network) is retried every 30 s; the server keeps running and the
# agent uses its rule-based fallback until the models are there.
set -e
ollama serve &
SERVER=$!
until ollama list >/dev/null 2>&1; do sleep 1; done
for model in "$LLM_MODEL" "$LLM_FALLBACK_MODEL"; do
  if [ -n "$model" ]; then
    until ollama list | grep -q "^$model"; do
      ollama pull "$model" || { echo "pull of $model failed, retrying in 30 s"; sleep 30; }
    done
  fi
done
touch /tmp/ready
echo "models ready: $LLM_MODEL $LLM_FALLBACK_MODEL"
wait $SERVER

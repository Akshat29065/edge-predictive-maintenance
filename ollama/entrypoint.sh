#!/bin/sh
# Starts the Ollama server, downloads the models on first run, then marks itself ready.
set -e
ollama serve &
SERVER=$!
until ollama list >/dev/null 2>&1; do sleep 1; done
for model in "$LLM_MODEL" "$LLM_FALLBACK_MODEL"; do
  if [ -n "$model" ]; then
    ollama list | grep -q "^$model" || ollama pull "$model"
  fi
done
touch /tmp/ready
echo "models ready: $LLM_MODEL $LLM_FALLBACK_MODEL"
wait $SERVER
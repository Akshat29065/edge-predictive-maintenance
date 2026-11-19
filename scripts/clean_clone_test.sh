#!/usr/bin/env bash
# Proves the COMMITTED repo runs from nothing: clones it, starts it as a separate compose project
# (own volumes, so your working stack and its downloaded models are untouched), waits for data
# to flow through every stage, then tears the test project down.
#
#   git status                      # commit first; only committed files are tested
#   docker compose down             # free the ports (8086, 5020, 11434)
#   bash scripts/clean_clone_test.sh
#
# First run downloads ~3 GB of Qwen models: allow 10-20 minutes.
set -u
REPO="$(git rev-parse --show-toplevel)"
DIR="$(mktemp -d)/edge-clone"
P=cleanclone
fail=0
git clone -q "$REPO" "$DIR" || { echo "clone failed"; exit 1; }
cd "$DIR"
for f in data/processed/cwru_features.csv analytics/models/fault_model.txt vision/models/thermal.onnx vision/demo_images/index.csv; do
  [ -f "$f" ] && echo "PASS committed: $f" || { echo "FAIL missing in clone: $f"; fail=1; }
done
echo "starting test stack (no .env, defaults only) ..."
docker compose -p $P up -d --build || { echo "FAIL compose up"; exit 1; }
echo "waiting for Ollama to finish downloading models (up to 25 min) ..."
for i in $(seq 1 150); do
  [ "$(docker compose -p $P ps ollama --format '{{.Health}}')" = "healthy" ] && break
  sleep 10
done
echo "waiting 100 s for data to flow ..."; sleep 100
q() { docker compose -p $P exec -T influxdb influx query "from(bucket:\"machine\") |> range(start:-3m) |> filter(fn:(r)=> r._measurement==\"$1\" and r._field==\"$2\") |> count()" --org edge-lab --token dev-token-change-me 2>/dev/null | grep -E '^ *[0-9]{4}-' | awk '{s+=$NF} END{print s+0}'; }
check() { n=$(q "$1" "$2"); [ "$n" -gt 0 ] && echo "PASS $1: $n points" || { echo "FAIL $1: no points"; fail=1; }; }
check bearing_features heartbeat
check predictions condition
check vision_inspection label
check agent_decisions mode
modes=$(docker compose -p $P logs agent 2>/dev/null | grep -oE '\[(llm|rules)[^,]*' | sort | uniq -c)
echo "agent decision modes:"; echo "$modes"
[ -s logs/actions.log ] && echo "PASS logs/actions.log written" || { echo "FAIL logs/actions.log empty"; fail=1; }
docker compose -p $P down -v >/dev/null 2>&1
[ $fail = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
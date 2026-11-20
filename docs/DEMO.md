# Demo script (about 8 minutes)

## Before you start (once, so nothing downloads on stage)
```
docker compose up --build -d
docker compose ps                      # 7 services, ollama "healthy"
python scripts/check_pipeline.py       # all four sections should show data
```
Open three terminals in the repo root:
1. `docker compose logs -f agent`
2. `tail -f logs/alerts.log`
3. a spare one for `python scripts/check_pipeline.py` and `cat shared/state.json`

If you want to watch a loop from the start, run `docker compose restart simulator`. One scenario loop is 345 s.

## Timeline of one loop (seconds after the simulator starts)
| Time | What the "motor" is doing | What you should see |
|---|---|---|
| 0 to 60 | healthy | agent: `healthy -> no_action`; no alert |
| 60 to 105 | inner-race fault, 7 mil | `WARNING` alert + `MEDIUM` ticket ("within 2 weeks") |
| 105 to 150 | inner-race fault, 14 mil | `CRITICAL` alert + `HIGH` ticket |
| 150 to 195 | inner-race fault, 21 mil | `CRITICAL` alert + `HIGH` ticket |
| 195 to 225 | healthy | `no_action` |
| 225 to 270 | outer-race fault, 14 mil | `CRITICAL` alert, "outer-race" |
| 270 to 315 | ball fault, 21 mil | `CRITICAL` alert, "rolling-element (ball)" |
| 315 to 345 | healthy, loop restarts | `no_action` |
Alerts appear about 5 to 15 s after each change (analytics smoothing plus one agent cycle).

## Talking points, in order
1. **The data source (1 min).** "The simulator is a real Modbus TCP server. It replays recorded bearing vibration from the CWRU dataset, so everything downstream reads registers the way it would from a real device. Moving to a real machine means changing the host in `telegraf.conf`." Show `cat shared/state.json` as the ground truth that only the simulator and camera see.
2. **Ingest (30 s).** "Telegraf polls the registers every second into InfluxDB." Run `python scripts/check_pipeline.py` and point at the heartbeat counter increasing.
3. **Predictive + prescriptive (2 min).** Show a prediction line: condition, confidence, fault size, anomaly score, urgency, action. "LightGBM classifies the fault and its size, an anomaly score compares against a healthy baseline, and a rule table turns the diagnosis into an action with a time window."
4. **Vision (1 min).** "A MobileNetV3 model classifies a thermal image, healthy or faulty. The camera is simulated and shows the image that matches the motor's state." Point at the vision section of `check_pipeline.py`.
5. **The agent (2 min).** Watch terminal 1 and 2 through the 7 mil to 14 mil change. "The agent combines both sensors. If they agree it is a confirmed fault; if they disagree it raises a warning for manual inspection. Qwen decides which functions to call, and a rule policy checks the answer, so it cannot skip a required alert, change its severity, or invent an alarm."
6. **Resilience (1 min, optional but strong).**
   ```
   docker compose stop ollama      # agent log switches to [rules]; alerts keep coming
   docker compose start ollama     # back to [llm] within a minute
   docker compose stop vision      # after ~40 s: "Fresh sensor data missing" warning, repeated every 2 min
   docker compose start vision
   ```
7. **Honest limits (30 s).** Replayed data, easily separable thermal images, CPU-only LLM. See the README.

## Likely questions
- **Why Modbus?** Lightest common industrial protocol, one dependency, fits the replayed simulator; swapping to OPC UA means a different Telegraf input.
- **Why not let the LLM decide everything?** Small models are unreliable. The policy guarantees required alerts, fixes their severity and rejects invented ones, and the system keeps working with the LLM off.
- **How accurate is it?** Train on one load, test on the other: fault F1 0.96 and 0.999, fault size F1 0.97 and 0.87. Thermal vision scored 488 of 488 leave-one-load-out, which I treat as optimistic. See the README for caveats.
- **Why JSON output instead of native tool calling?** Qwen3 4B kept "thinking" until it ran out of tokens (0 of 9 valid calls); schema-constrained output gave 9 of 9 and cut latency to 2 to 12 s. `LLM_MODE=tools` switches back.
- **Does it run on the edge device?** The whole stack used about 4.6 GB of memory, no GPU, and starts with `docker compose up`.
- **What would you do with more time?** A real device or a live data source, a second dataset for the vision model, a remaining-useful-life estimate, and an evaluation on a machine the models have never seen.
"""Modbus TCP server that behaves like a motor monitor.

The values behind the registers are replayed CWRU features, so everything
downstream (Telegraf, InfluxDB, analytics) sees what a real device would send.

Register map (holding registers, function code 3, addresses start at 0):
    0-1    rms         float32, high word first (Telegraf byte order ABCD)
    2-3    peak
    4-5    p2p
    6-7    crest
    8-9    kurtosis
    10-11  skew
    12-13  shape
    14-15  centroid
    16     heartbeat   uint16 counter, +1 per update, wraps at 65535

The true condition (healthy or fault class) is NOT in the registers. It is
written to STATE_FILE so the camera simulator can stay in sync, and so the
analytics can be evaluated later. Models must never read it as an input.
"""
import asyncio
import json
import logging
import os
import struct
import time
from pathlib import Path

from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
)
from pymodbus.server import StartAsyncTcpServer

from replay import FEATURES, load_scenario, load_windows, replay, validate

log = logging.getLogger("simulator")

PORT = int(os.environ.get("MODBUS_PORT", "5020"))
INTERVAL = float(os.environ.get("REPLAY_INTERVAL_S", "1"))
CSV_PATH = os.environ.get("FEATURES_CSV", "/data/cwru_features.csv")
SCENARIO_PATH = os.environ.get("SCENARIO_FILE", "/app/scenario.yaml")
STATE_FILE = Path(os.environ.get("STATE_FILE", "/shared/state.json"))

HOLDING = 3                        # Modbus function code for holding registers
HEARTBEAT_ADDR = 2 * len(FEATURES)  # first register after the float block (16)


def float_to_registers(value):
    """Pack a float32 into two 16-bit registers, high word first."""
    return list(struct.unpack(">HH", struct.pack(">f", value)))


def write_state(sample, tick):
    """Atomically write the ground-truth condition for other simulators."""
    state = {
        "tick": tick,
        "timestamp": time.time(),
        "file": sample["file"],
        "cls": sample["cls"],
        "size_mil": sample["size_mil"],
        "load_hp": sample["load_hp"],
        "healthy": sample["cls"] == "normal",
    }
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(STATE_FILE)
    except OSError as exc:
        log.warning("could not write %s: %s", STATE_FILE, exc)


def publish(context, sample, tick, last_step):
    """Write one sample into the registers and the state file."""
    registers = []
    for value in sample["values"]:
        registers.extend(float_to_registers(value))
    registers.append(tick % 65536)
    context[0].setValues(HOLDING, 0, registers)
    write_state(sample, tick)
    if sample["step"] != last_step:
        log.info(
            "scenario step %d: %s (class=%s, fault size=%d mil, load=%d hp)",
            sample["step"], sample["file"], sample["cls"],
            sample["size_mil"], sample["load_hp"],
        )
    return sample["step"]


async def feeder(context, samples, last_step):
    tick = 1
    for sample in samples:
        await asyncio.sleep(INTERVAL)
        last_step = publish(context, sample, tick, last_step)
        tick += 1


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    by_file = load_windows(CSV_PATH)
    scenario = load_scenario(SCENARIO_PATH)
    validate(by_file, scenario)

    # pymodbus 3.9 shifts addresses by +1 internally, so a block that starts at 1
    # is what a client sees at address 0. All four blocks must be passed or the
    # holding-register block is silently replaced by an empty one.
    def block():
        return ModbusSequentialDataBlock(1, [0] * 32)

    device = ModbusSlaveContext(di=block(), co=block(), ir=block(), hr=block())
    context = ModbusServerContext(slaves=device, single=True)

    samples = replay(by_file, scenario)
    # Publish the first sample before the server accepts connections, so a
    # poller never reads an all-zero register block.
    last_step = publish(context, next(samples), 0, None)

    log.info(
        "Modbus TCP server on port %d, %.2f s per update, %d scenario steps",
        PORT, INTERVAL, len(scenario["steps"]),
    )
    await asyncio.gather(
        StartAsyncTcpServer(context=context, address=("0.0.0.0", PORT)),
        feeder(context, samples, last_step),
    )


if __name__ == "__main__":
    asyncio.run(main())
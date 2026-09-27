#!/usr/bin/env python3
"""Exercise real Ascend READ/WRITE without Controller or expert/model types.

Start listener and initiator on different NPUs. TCP exchanges only bounded
descriptors/acknowledgements; payloads use the forced Ascend Direct driver.
Successful results prove byte correctness, not a particular HCCS/RDMA route.
Use on a trusted test network. Results include warmup-free transfer timings;
they are a diagnostic, not an end-to-end model benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time

import torch
import torch_npu  # noqa: F401 -- register the NPU device module

from expertkit_transport.transports.transfer_engine.drivers.base import (
    MemoryRegion,
    MemorySlice,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    MooncakeDriverConfig,
    MooncakeMemoryTransport,
)

ALIGNMENT = 2 << 20
MAX_BYTES = 8 << 20
SIZES = (8192, 65536, 1 << 20, MAX_BYTES)
CONTROL_LIMIT = 4096


async def send(writer: asyncio.StreamWriter, message: dict) -> None:
    data = json.dumps(message).encode() + b"\n"
    if len(data) > CONTROL_LIMIT:
        raise ValueError("control frame too large")
    writer.write(data)
    await writer.drain()


async def receive(reader: asyncio.StreamReader) -> dict:
    data = await asyncio.wait_for(reader.readline(), 120)
    if not data.endswith(b"\n") or len(data) > CONTROL_LIMIT:
        raise ValueError("invalid control frame")
    message = json.loads(data)
    if not isinstance(message, dict):
        raise ValueError("control frame must be an object")
    return message


def pattern(size: int, seed: int) -> torch.Tensor:
    return (
        (torch.arange(size, dtype=torch.int32) + seed).remainder_(251).to(torch.uint8)
    )


async def fill(
    driver: MooncakeMemoryTransport, tensor: torch.Tensor, seed: int
) -> None:
    stream = torch.npu.Stream(device=driver.device)
    event = torch.npu.Event()
    with torch.npu.stream(stream):
        tensor.copy_(pattern(tensor.numel(), seed).pin_memory(), non_blocking=True)
        event.record(stream)
    await driver.wait_event(event, monotonic_deadline=math.inf)


async def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    owner = torch.empty(MAX_BYTES + ALIGNMENT - 1, dtype=torch.uint8, device=device)
    slab = owner.narrow(0, (-owner.data_ptr()) % ALIGNMENT, MAX_BYTES)
    torch.npu.synchronize(device)
    region = MemoryRegion(slab.data_ptr(), MAX_BYTES, str(device), owner)
    driver = MooncakeMemoryTransport(
        MooncakeDriverConfig(args.segment_host, "P2PHANDSHAKE", "ascend_direct", device)
    )
    reader = writer = server = None
    try:
        await driver.register_region(region)
        descriptor = {
            "session": driver.session_id,
            "address": region.address,
            "bytes": MAX_BYTES,
        }
        if args.role == "listener":
            connections = asyncio.Queue(maxsize=1)

            def connected(r, w):
                if connections.full():
                    w.close()
                else:
                    connections.put_nowait((r, w))

            server = await asyncio.start_server(
                connected, args.control_host, args.control_port, limit=CONTROL_LIMIT
            )
            print(json.dumps({"event": "ready", "backend": driver.backend}), flush=True)
            reader, writer = await asyncio.wait_for(connections.get(), 120)
            hello = await receive(reader)
            if hello != {"run": args.run_id, "operation": "read_write"}:
                raise ValueError("peer run does not match")
            await send(writer, descriptor)
            for size in SIZES:
                for iteration in range(args.iterations + 1):
                    seed = (iteration + size // 8192) % 251
                    await fill(driver, slab[:size], seed)
                    await send(
                        writer, {"size": size, "iteration": iteration, "seed": seed}
                    )
                    ack = await receive(reader)
                    if ack != {"size": size, "iteration": iteration, "written": True}:
                        raise ValueError("write completion mismatch")
                    await driver.acquire_remote_writes(monotonic_deadline=math.inf)
                    if not torch.equal(slab[:size].cpu(), pattern(size, seed + 31)):
                        raise ValueError("remote WRITE byte mismatch")
                    await send(writer, {"verified": True})
            # All peer operations have completed before deregistration.
            await send(writer, {"complete": True})
        else:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    args.control_host, args.control_port, limit=CONTROL_LIMIT
                ),
                30,
            )
            await send(writer, {"run": args.run_id, "operation": "read_write"})
            remote = await receive(reader)
            address = remote.get("address")
            if (
                type(address) is not int
                or not 0 < address < (1 << 64) - MAX_BYTES
                or remote.get("bytes") != MAX_BYTES
                or not isinstance(remote.get("session"), str)
                or not remote["session"]
            ):
                raise ValueError("invalid remote memory descriptor")
            results = []
            for size in SIZES:
                timings = {"read": [], "write": []}
                for iteration in range(args.iterations + 1):
                    ready = await receive(reader)
                    if ready.get("size") != size or ready.get("iteration") != iteration:
                        raise ValueError("peer iteration mismatch")
                    slices = [MemorySlice(region, 0, address, size)]
                    started = time.perf_counter()
                    await driver.read(
                        remote["session"],
                        slices,
                        monotonic_deadline=time.monotonic() + 120,
                    )
                    await driver.acquire_remote_writes(monotonic_deadline=math.inf)
                    read_ms = (time.perf_counter() - started) * 1000
                    if not torch.equal(slab[:size].cpu(), pattern(size, ready["seed"])):
                        raise ValueError("remote READ byte mismatch")
                    await fill(driver, slab[:size], ready["seed"] + 31)
                    started = time.perf_counter()
                    await driver.write(
                        remote["session"],
                        slices,
                        monotonic_deadline=time.monotonic() + 120,
                    )
                    write_ms = (time.perf_counter() - started) * 1000
                    await send(
                        writer, {"size": size, "iteration": iteration, "written": True}
                    )
                    if await receive(reader) != {"verified": True}:
                        raise ValueError("peer failed to verify WRITE")
                    if iteration:
                        timings["read"].append(read_ms)
                        timings["write"].append(write_ms)
                results.append(
                    {
                        "bytes": size,
                        **{
                            f"{op}_median_ms": statistics.median(values)
                            for op, values in timings.items()
                        },
                    }
                )
            if await receive(reader) != {"complete": True}:
                raise ValueError("peer completion missing")
            print(
                json.dumps(
                    {
                        "backend": driver.backend,
                        "device": str(device),
                        "verified": True,
                        "iterations": args.iterations,
                        "route": "not measured",
                        "results": results,
                    }
                ),
                flush=True,
            )
    except BaseException:
        # No close/deregister on ambiguous transfer or control failure.
        driver.quarantine("Ascend memory diagnostic failed; process must exit")
        raise
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        if server is not None:
            server.close()
            await server.wait_closed()
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("listener", "initiator"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--segment-host", required=True)
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100 or not 1 <= args.control_port <= 65535:
        parser.error("iterations must be 1..100 and port must be 1..65535")
    if not args.device.startswith("npu:") or not args.device[4:].isdigit():
        parser.error("an indexed NPU device is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

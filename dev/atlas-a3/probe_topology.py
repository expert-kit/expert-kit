"""Audit the Controller's ready-route topology without contacting Workers."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter
from dataclasses import dataclass

import grpc
from expertkit_proto.ek.control.v2 import lifecycle_pb2, lifecycle_pb2_grpc

type RouteKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Replica:
    worker_id: str
    start_id: str
    device: str


type Routes = dict[RouteKey, tuple[Replica, ...]]


class TopologyProtocolError(RuntimeError):
    """Report an invalid or discontinuous topology stream."""


class TopologyAssembler:
    """Assemble snapshots and updates without opening Worker connections."""

    def __init__(self, instance_id: int) -> None:
        self.instance_id = instance_id
        self.version = 0
        self.routes: Routes = {}
        self._pending_key: tuple[str, int, int | None, int] | None = None
        self._parts: dict[int, lifecycle_pb2.TopologyMessage] = {}

    @property
    def has_pending_parts(self) -> bool:
        return self._pending_key is not None

    def consume(self, message: lifecycle_pb2.TopologyMessage) -> bool:
        kind = message.WhichOneof("message")
        if kind == "snapshot":
            part = message.snapshot
            previous_version = None
            version = int(part.topology_version)
        elif kind == "update":
            part = message.update
            previous_version = int(part.previous_version)
            version = int(part.topology_version)
        else:
            raise TopologyProtocolError("message has neither snapshot nor update")

        if int(part.instance_id) != self.instance_id:
            raise TopologyProtocolError(
                "message instance ID does not match the request"
            )
        part_index = int(part.part_index)
        part_count = int(part.part_count)
        if part_count <= 0 or not 0 <= part_index < part_count:
            raise TopologyProtocolError("invalid multipart index or count")

        key = (kind, version, previous_version, part_count)
        if self._pending_key is None:
            self._pending_key = key
        elif self._pending_key != key:
            raise TopologyProtocolError("multipart topology messages were interleaved")
        if part_index in self._parts:
            raise TopologyProtocolError("multipart topology index was repeated")
        self._parts[part_index] = message
        if len(self._parts) != part_count:
            return False

        ordered = [self._parts[index] for index in range(part_count)]
        self._pending_key = None
        self._parts = {}

        if kind == "snapshot":
            routes = (
                route for completed in ordered for route in completed.snapshot.routes
            )
            decoded = _decode_routes(routes)
            if version < self.version:
                raise TopologyProtocolError("snapshot version moved backwards")
            self.routes = decoded
        else:
            if previous_version != self.version or version <= self.version:
                raise TopologyProtocolError(
                    "topology update does not join the installed version"
                )
            changes = (
                change for completed in ordered for change in completed.update.changes
            )
            seen: set[RouteKey] = set()
            for change in changes:
                route_key = (int(change.layer_id), int(change.expert_id))
                if route_key in seen:
                    raise TopologyProtocolError("topology update repeats a route")
                seen.add(route_key)
                replicas = _decode_replicas(change.replicas)
                if replicas:
                    self.routes[route_key] = replicas
                else:
                    self.routes.pop(route_key, None)

        self.version = version
        return True


def _decode_routes(routes: object) -> Routes:
    decoded: Routes = {}
    for route in routes:  # type: ignore[union-attr]
        key = (int(route.layer_id), int(route.expert_id))
        if key in decoded:
            raise TopologyProtocolError("topology snapshot repeats a route")
        replicas = _decode_replicas(route.replicas)
        if not replicas:
            raise TopologyProtocolError("topology snapshot contains an empty route")
        decoded[key] = replicas
    return decoded


def _decode_replicas(replicas: object) -> tuple[Replica, ...]:
    result: list[Replica] = []
    identities: set[tuple[str, str]] = set()
    for replica in replicas:  # type: ignore[union-attr]
        identity = (replica.worker_id, replica.start_id)
        if not all((*identity, replica.device)):
            raise TopologyProtocolError("topology contains incomplete Worker metadata")
        if identity in identities:
            raise TopologyProtocolError("topology route repeats a Worker process")
        identities.add(identity)
        result.append(
            Replica(
                worker_id=replica.worker_id,
                start_id=replica.start_id,
                device=replica.device,
            )
        )
    return tuple(result)


def _expected_routes(args: argparse.Namespace) -> set[RouteKey]:
    return {
        (layer_id, expert_id)
        for layer_id in range(args.moe_layer_start, args.moe_layer_end)
        for expert_id in range(args.experts_per_layer)
    }


def _report(assembler: TopologyAssembler, args: argparse.Namespace) -> bool:
    expected = _expected_routes(args)
    present = set(assembler.routes)
    missing = expected - present
    unexpected = present - expected

    print(f"topology_version={assembler.version}")
    print(f"ready_routes={len(expected - missing)}/{len(expected)}")
    print(f"unexpected_routes={len(unexpected)}")

    worker_routes: Counter[tuple[str, str, str]] = Counter()
    for replicas in assembler.routes.values():
        for replica in replicas:
            worker_routes[(replica.worker_id, replica.start_id, replica.device)] += 1
    print(f"worker_processes={len(worker_routes)}")
    for (worker_id, _start_id, device), route_count in sorted(worker_routes.items()):
        print(f"  {worker_id} device={device} routes={route_count}")

    if missing:
        print("missing:")
        for layer_id in range(args.moe_layer_start, args.moe_layer_end):
            expert_ids = sorted(
                expert_id
                for missing_layer, expert_id in missing
                if missing_layer == layer_id
            )
            if expert_ids:
                visible = expert_ids[: args.max_missing_ids]
                suffix = " ..." if len(expert_ids) > len(visible) else ""
                print(
                    f"  layer={layer_id} count={len(expert_ids)} "
                    f"expert_ids={visible}{suffix}"
                )

    if unexpected:
        visible = sorted(unexpected)[: args.max_missing_ids]
        suffix = " ..." if len(unexpected) > len(visible) else ""
        print(f"unexpected={visible}{suffix}")

    complete = not missing
    print(f"status={'COMPLETE' if complete else 'INCOMPLETE'}")
    return complete


async def _probe(args: argparse.Namespace) -> int:
    channel = grpc.aio.insecure_channel(args.controller)
    assembler: TopologyAssembler | None = None
    resolved = None
    installed = False
    overall_timeout = False
    try:
        instance_stub = lifecycle_pb2_grpc.InstanceServiceStub(channel)
        resolved = await instance_stub.ResolveDefaultInstance(
            lifecycle_pb2.ResolveDefaultInstanceRequest(),
            timeout=args.timeout,
            wait_for_ready=True,
        )
        assembler = TopologyAssembler(int(resolved.instance_id))
        topology_stub = lifecycle_pb2_grpc.TopologyServiceStub(channel)
        stream = topology_stub.WatchTopology(
            lifecycle_pb2.WatchTopologyRequest(
                instance_id=resolved.instance_id,
                current_version=0,
            ),
            wait_for_ready=True,
        )
        iterator = stream.__aiter__()
        deadline = asyncio.get_running_loop().time() + args.timeout
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    overall_timeout = True
                    break
                wait_seconds = remaining
                if installed and not assembler.has_pending_parts:
                    wait_seconds = min(wait_seconds, args.settle_seconds)
                try:
                    message = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=wait_seconds,
                    )
                except TimeoutError:
                    if installed and not assembler.has_pending_parts:
                        break
                    overall_timeout = True
                    break
                except StopAsyncIteration:
                    break
                installed = assembler.consume(message) or installed
        finally:
            stream.cancel()
    finally:
        await channel.close()

    if resolved is None or assembler is None:
        raise RuntimeError("Controller did not resolve the default instance")
    print(
        f"instance_id={resolved.instance_id} "
        f"model={resolved.model_name!r} instance={resolved.instance_name!r}"
    )
    complete = _report(assembler, args)
    if overall_timeout:
        print(f"wait_timeout_seconds={args.timeout}")
    return 0 if complete else 1


def _default_controller() -> str | None:
    if endpoint := os.getenv("EK_ADDR"):
        return endpoint
    if host := os.getenv("EK_NODE_A_IP"):
        port = os.getenv("EK_CONTROLLER_INTER_PORT", "15002")
        return f"{host}:{port}"
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ready expert routes published by the EK Controller."
    )
    parser.add_argument(
        "--controller",
        default=_default_controller(),
        help=(
            "Controller Frontend gRPC endpoint. Defaults to EK_ADDR or "
            "EK_NODE_A_IP:EK_CONTROLLER_INTER_PORT."
        ),
    )
    parser.add_argument("--moe-layer-start", type=int, default=1)
    parser.add_argument("--moe-layer-end", type=int, default=27)
    parser.add_argument("--experts-per-layer", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Report after this interval passes without another topology message.",
    )
    parser.add_argument("--max-missing-ids", type=int, default=16)
    args = parser.parse_args()
    if not args.controller:
        parser.error(
            "--controller is required when Controller environment variables are unset"
        )
    if not 0 <= args.moe_layer_start < args.moe_layer_end:
        parser.error("MoE layer range must be nonempty and nonnegative")
    if args.experts_per_layer <= 0:
        parser.error("--experts-per-layer must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.settle_seconds <= 0:
        parser.error("--settle-seconds must be positive")
    if args.max_missing_ids <= 0:
        parser.error("--max-missing-ids must be positive")
    return args


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_probe(_parse_args())))
    except (grpc.aio.AioRpcError, TopologyProtocolError, RuntimeError) as error:
        raise SystemExit(f"topology probe failed: {error}") from error


if __name__ == "__main__":
    main()

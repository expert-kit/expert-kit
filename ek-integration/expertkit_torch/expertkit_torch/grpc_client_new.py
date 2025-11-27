"""
New ExpertKit gRPC client with direct worker communication support.

This client implements the simplified v1 MVP design:
- Fetches routing table from controller (expert_id → worker_addr)
- Connects directly to workers using ConnectionPool
- Simple routing lookup (no complex client-side logic)
- Controller handles all scheduling decisions
"""

import asyncio
import io
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import grpc
import safetensors
import torch

from expertkit_torch.connection_pool import get_connection_pool
from expertkit_torch.pbpy.ek.control.v1 import routing_pb2, routing_pb2_grpc
from expertkit_torch.pbpy.ek.worker.v1 import expert_pb2, expert_pb2_grpc
from line_profiler import profile

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1 GB


class ExpertKitClient:
    """
    ExpertKit gRPC client with direct worker communication.

    Features:
    - Fetches routing table from controller
    - Direct worker connections via ConnectionPool
    - Automatic routing refresh on missing experts
    - Falls back to controller forwarding on failure
    """

    def __init__(
        self,
        controller_addr: str,
        timeout_sec: float = 2.0,
        enable_direct_path: bool = True,
        routing_refresh_interval: int = 30,
        decompose_threshold: int = 999999,  # Always decompose by default
    ):
        """
        Initialize ExpertKit client.

        Args:
            controller_addr: Controller address (host:port)
            timeout_sec: Request timeout in seconds
            enable_direct_path: Enable direct worker communication (default True)
            routing_refresh_interval: Routing table refresh interval in seconds
            decompose_threshold: Batch size threshold for using decomposition (default: always decompose)
        """
        self.controller_addr = controller_addr
        self.timeout = timeout_sec
        self.enable_direct_path = enable_direct_path
        self.decompose_threshold = decompose_threshold

        # Controller channel (will be created in start())
        self.controller_channel = None
        self.controller_stub = None
        self.routing_stub = None

        # Direct path state
        self.routing: Dict[str, str] = {}  # expert_id → worker_addr
        self.routing_version: int = 0
        self.routing_lock = None  # Will be created in start()

        # Connection pool for direct worker connections
        self.conn_pool = get_connection_pool(max_size=50)

        # Background refresh task
        self.refresh_interval = routing_refresh_interval
        self.refresh_task: Optional[asyncio.Task] = None

        # Event loop reference (set in start())
        self._event_loop = None

        logger.info(
            f"ExpertKitClient initialized: controller={controller_addr}, "
            f"direct_path={enable_direct_path}, timeout={timeout_sec}s"
        )

    async def start(self):
        """Start background tasks (routing refresh)."""
        # Store event loop reference
        self._event_loop = asyncio.get_event_loop()

        # Create gRPC channels in the current event loop
        self.controller_channel = grpc.aio.insecure_channel(
            self.controller_addr,
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )
        self.controller_stub = expert_pb2_grpc.ComputationServiceStub(
            self.controller_channel)
        self.routing_stub = routing_pb2_grpc.RoutingServiceStub(
            self.controller_channel)

        # Create async lock in current event loop
        self.routing_lock = asyncio.Lock()

        if self.enable_direct_path:
            # Initial routing fetch
            await self.refresh_routing()

            # Start background refresh
            self.refresh_task = asyncio.create_task(
                self._routing_refresh_loop())
            logger.info("Background routing refresh started")

    async def stop(self):
        """Stop background tasks and close connections."""
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass

        await self.conn_pool.close_all()
        if self.controller_channel:
            await self.controller_channel.close()
        logger.info("ExpertKitClient stopped")

    async def refresh_routing(self, expert_ids: Optional[List[str]] = None):
        """
        Fetch routing table from controller.

        Args:
            expert_ids: Optional list of specific expert IDs to fetch (None = all)
        """
        try:
            req = routing_pb2.GetRoutingReq(expert_ids=expert_ids or [])
            resp = await self.routing_stub.GetRouting(req, timeout=self.timeout)

            async with self.routing_lock:
                if expert_ids:
                    # Partial update
                    self.routing.update(resp.routing)
                else:
                    # Full replace
                    self.routing = dict(resp.routing)
                self.routing_version = resp.version

            logger.info(
                f"Routing table updated: {len(self.routing)} experts, version={resp.version}"
            )

            # Print sample for debugging
            if len(self.routing) > 0:
                sample = list(self.routing.items())[:3]
                print(f"[Debug] Sample routing entries: {sample}")

        except grpc.RpcError as e:
            logger.error(
                f"Failed to fetch routing: {e.code().name} - {e.details()}")
            raise RuntimeError(f"Routing fetch failed: {e.code().name}") from e

    async def _routing_refresh_loop(self):
        """Background task to periodically refresh routing table."""
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                await self.refresh_routing()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Routing refresh error: {e}")

    async def forward_expert_async(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward computation to experts (async version).

        Args:
            expert_ids: Expert IDs for each sequence [batch_size, n_routed_experts]
            hidden_state: Input tensor [batch_size, attn_dim]

        Returns:
            Output tensor [batch_size, n_routed_experts, expert_dim]
        """
        if not self.enable_direct_path:
            # Fallback to controller forwarding
            return await self._forward_via_controller(expert_ids, hidden_state)

        try:
            return await self._forward_direct(expert_ids, hidden_state)
        except Exception as e:
            logger.warning(
                f"Direct path failed, falling back to controller: {e}")
            return await self._forward_via_controller(expert_ids, hidden_state)

    @profile
    async def _forward_direct(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Forward via direct worker connections with request decomposition.

        Implements the controller's decomposition logic:
        - Decomposes multi-expert sequences into single-expert requests
        - Sends parallel requests to workers (one per unique expert)
        - Reconstructs output in correct format

        Note: Workers only accept single-expert-per-sequence format, so decomposition
        is always required regardless of batch size.
        """
        origin_device = hidden_state.device
        batch_size = len(expert_ids)
        n_experts_per_seq = len(expert_ids[0]) if expert_ids else 0

        # Note: Removed "simple" path as workers only accept single-expert-per-sequence format
        # All requests must be decomposed, regardless of batch size

        # Initialize result tensor: [batch_size, n_experts_per_seq, expert_dim]
        # We'll fill this as we receive responses
        result = [[None for _ in range(n_experts_per_seq)]
                  for _ in range(batch_size)]

        # Decompose request by expert (like controller's break_down_to_egress)
        # Map: expert_id -> [(seq_idx, expert_idx), ...]
        expert_to_sequences = defaultdict(list)

        for seq_idx, experts in enumerate(expert_ids):
            for expert_idx, expert_id in enumerate(experts):
                expert_to_sequences[expert_id].append((seq_idx, expert_idx))

        # Get routing for all needed experts
        needed_experts = list(expert_to_sequences.keys())
        async with self.routing_lock:
            missing_experts = [
                e for e in needed_experts if e not in self.routing]

        if missing_experts:
            logger.info(
                f"Refreshing routing for {len(missing_experts)} missing experts")
            await self.refresh_routing(missing_experts)

        # Send requests in parallel (one per expert)
        tasks = []
        for expert_id, seq_positions in expert_to_sequences.items():
            # Get worker address
            async with self.routing_lock:
                worker_addr = self.routing.get(expert_id)

            if not worker_addr:
                raise RuntimeError(
                    f"Expert {expert_id} not found in routing table")

            # Extract input tensors for sequences that need this expert
            seq_indices = [pos[0] for pos in seq_positions]
            # [n_seqs_for_expert, hidden_dim]
            expert_input = hidden_state[seq_indices]

            # Create task
            task = self._send_expert_request(
                worker_addr, expert_id, seq_positions, expert_input
            )
            tasks.append(task)

        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)

        # Reconstruct output (like controller's output reconstruction)
        for expert_idx, (expert_id, seq_positions) in enumerate(expert_to_sequences.items()):
            expert_output = responses[expert_idx]  # [n_seqs, expert_dim]

            for output_idx, (seq_idx, expert_pos) in enumerate(seq_positions):
                result[seq_idx][expert_pos] = expert_output[output_idx]

        # Convert result to tensor: [batch_size, n_experts_per_seq, expert_dim]
        output_tensors = []
        for seq_result in result:
            seq_tensors = [t for t in seq_result if t is not None]
            if len(seq_tensors) != n_experts_per_seq:
                raise RuntimeError(
                    f"Incomplete result: got {len(seq_tensors)}, expected {n_experts_per_seq}")
            output_tensors.append(torch.stack(seq_tensors, dim=0))

        final_output = torch.stack(output_tensors, dim=0)
        return final_output.to(origin_device)

    async def _send_expert_request(
        self,
        worker_addr: str,
        expert_id: str,
        seq_positions: List[tuple],
        expert_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Send single-expert request to a worker (decomposed format).

        Args:
            worker_addr: Worker address (with http:// prefix)
            expert_id: Expert ID to compute
            seq_positions: List of (seq_idx, expert_idx) tuples
            expert_input: Input tensor for sequences needing this expert

        Returns:
            Output tensor [n_sequences, expert_dim]
        """
        # Strip http:// prefix
        clean_addr = worker_addr.replace("http://", "").replace("https://", "")
        logger.debug(
            f"Sending {len(seq_positions)} sequences to worker {clean_addr} for expert {expert_id}")

        # Get channel from pool
        channel = await self.conn_pool.get_channel(clean_addr)
        stub = expert_pb2_grpc.ComputationServiceStub(channel)

        # Serialize tensor
        tensor_data = safetensors.torch.save({"data": expert_input})

        # Build request: each sequence has single expert (worker expects this format)
        seq_infos = [
            expert_pb2.ForwardReq.SequenceInfo(experts=[expert_id])
            for _ in seq_positions
        ]

        try:
            req = expert_pb2.ForwardReq(
                instance_id="test",  # TODO: Use actual instance ID
                sequences=seq_infos,
                tensor=tensor_data,
            )

            resp = await stub.Forward(req, timeout=self.timeout)

            # Deserialize output
            output = safetensors.torch.load(resp.output_tensor)["data"]
            return output

        except grpc.RpcError as e:
            logger.error(
                f"Worker {clean_addr} request failed: {e.code().name}")
            raise RuntimeError(
                f"Worker request failed: {e.code().name}") from e

    async def _forward_via_controller(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Fallback: Forward via controller (original path)."""
        origin_device = hidden_state.device

        # Serialize tensor
        tensor_data = safetensors.torch.save({"data": hidden_state})

        # Generate sequence info
        seq_infos = []
        for ids in expert_ids:
            seq_infos.append(expert_pb2.ForwardReq.SequenceInfo(experts=ids))

        try:
            req = expert_pb2.ForwardReq(
                instance_id="test", sequences=seq_infos, tensor=tensor_data
            )

            resp = await self.controller_stub.Forward(req, timeout=self.timeout)

            output = safetensors.torch.load(resp.output_tensor)["data"]
            return output.to(origin_device)

        except grpc.RpcError as e:
            logger.error(
                f"Controller forward failed: {e.code().name} - {e.details()}")
            raise RuntimeError(
                f"Controller forward failed: {e.code().name}") from e

    def forward_expert(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Synchronous wrapper for forward_expert_async.

        Args:
            expert_ids: Expert IDs for each sequence
            hidden_state: Input tensor

        Returns:
            Output tensor
        """
        # Use the event loop that was created in start()
        if self._event_loop is None:
            raise RuntimeError(
                "Client not started. Call await client.start() first.")

        # Run async function in the stored event loop
        return self._event_loop.run_until_complete(
            self.forward_expert_async(expert_ids, hidden_state)
        )

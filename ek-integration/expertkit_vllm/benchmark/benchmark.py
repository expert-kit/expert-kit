from expertkit_vllm.grpc_client import ExpertKitClient
from expertkit_vllm.pbpy import expert_pb2, expert_pb2_grpc
import argparse
import time
import torch
import numpy as np
import threading
import concurrent.futures
from concurrent import futures
import grpc
import io
import sys
import os
from tqdm import tqdm

# Add path to import our modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Import the modules


class BenchmarkExpertServer(expert_pb2_grpc.ExpertComputationServicer):
    """A simple benchmark server that returns predefined outputs."""

    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Create expert model weights
        self.gate_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, bias=False)

        # Move to device if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gate_proj.to(self.device)
        self.up_proj.to(self.device)
        self.down_proj.to(self.device)

        # Metrics
        self.num_requests = 0
        self.total_tokens = 0
        self.total_time = 0
        self.request_times = []
        self.lock = threading.Lock()

    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.num_requests = 0
            self.total_tokens = 0
            self.total_time = 0
            self.request_times = []

    def get_metrics(self):
        """Get the current metrics."""
        with self.lock:
            return {
                'num_requests': self.num_requests,
                'total_tokens': self.total_tokens,
                'total_time': self.total_time,
                'request_times': self.request_times.copy(),
                'avg_time': self.total_time / max(1, self.num_requests),
                'avg_tokens_per_sec': self.total_tokens / max(0.001, self.total_time)
            }

    def Forward(self, request, context):
        """Process a forward request, applying a real MLP computation."""
        start_time = time.time()

        try:
            # Deserialize input tensor
            tensor_bytes = request.tensor
            input_tensor = torch.load(io.BytesIO(tensor_bytes))

            # Move to device if needed
            if input_tensor.device != self.device:
                input_tensor = input_tensor.to(self.device)

            # Process with the MoE forward pass
            with torch.no_grad():
                gate_output = torch.nn.functional.silu(
                    self.gate_proj(input_tensor))
                up_output = self.up_proj(input_tensor)
                hidden = gate_output * up_output
                output = self.down_proj(hidden)

            # Serialize the output
            buf = io.BytesIO()
            torch.save(output.cpu(), buf)
            output_bytes = buf.getvalue()

            # Update metrics
            elapsed = time.time() - start_time
            with self.lock:
                self.num_requests += 1
                self.total_tokens += input_tensor.size(0)
                self.total_time += elapsed
                self.request_times.append(elapsed)

            return expert_pb2.ExpertForwardReply(output_tensor=output_bytes)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error: {str(e)}")
            raise


def run_server(port, hidden_size, intermediate_size):
    """Run the benchmark server."""
    servicer = BenchmarkExpertServer(hidden_size, intermediate_size)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    expert_pb2_grpc.add_ExpertComputationServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    return server, servicer


def run_client_benchmark(address, batch_size, hidden_size, num_requests, num_threads):
    """Run a benchmark as a client."""
    client = ExpertKitClient(address, timeout_sec=5.0)

    # Create a test tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor = torch.randn(batch_size, hidden_size, device=device)

    # Metrics
    total_time = 0
    request_times = []

    # Run a warmup
    print("Warming up...")
    for _ in range(5):
        client.forward_expert(layer=0, idx=0, hidden_state=test_tensor)

    # Run the benchmark in parallel
    print(f"Running benchmark with {num_threads} threads...")

    # Function to run a single request
    def run_request():
        start_time = time.time()
        client.forward_expert(layer=0, idx=0, hidden_state=test_tensor)
        return time.time() - start_time

    # Run requests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_request) for _ in range(num_requests)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_requests):
            request_time = future.result()
            total_time += request_time
            request_times.append(request_time)

    # Calculate metrics
    avg_time = total_time / num_requests
    p50 = np.percentile(request_times, 50)
    p90 = np.percentile(request_times, 90)
    p99 = np.percentile(request_times, 99)
    tokens_per_sec = (batch_size * num_requests) / total_time

    # Print results
    print("\nClient Benchmark Results:")
    print(f"Total requests: {num_requests}")
    print(f"Total tokens: {batch_size * num_requests}")
    print(f"Average request time: {avg_time*1000:.2f} ms")
    print(f"P50 latency: {p50*1000:.2f} ms")
    print(f"P90 latency: {p90*1000:.2f} ms")
    print(f"P99 latency: {p99*1000:.2f} ms")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")

    return {
        'avg_time': avg_time,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'tokens_per_sec': tokens_per_sec
    }


def main():
    parser = argparse.ArgumentParser(description="ExpertKit Benchmark")
    parser.add_argument("--mode", choices=["server", "client", "both"], default="both",
                        help="Benchmark mode: server, client, or both")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--host", type=str,
                        default="localhost", help="Server host")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--hidden-size", type=int,
                        default=4096, help="Hidden size")
    parser.add_argument("--intermediate-size", type=int,
                        default=11008, help="Intermediate size")
    parser.add_argument("--num-requests", type=int,
                        default=100, help="Number of requests")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Number of client threads")

    args = parser.parse_args()

    server = None
    servicer = None

    try:
        # Start server if needed
        if args.mode in ["server", "both"]:
            print(f"Starting server on port {args.port}...")
            server, servicer = run_server(
                args.port, args.hidden_size, args.intermediate_size
            )
            print(f"Server started on port {args.port}")

        # Run client benchmark if needed
        if args.mode in ["client", "both"]:
            print(
                f"Running client benchmark against {args.host}:{args.port}...")
            address = f"{args.host}:{args.port}"
            run_client_benchmark(
                address, args.batch_size, args.hidden_size,
                args.num_requests, args.num_threads
            )

        # If in server-only mode, keep running until interrupted
        if args.mode == "server":
            print("Server running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
                    metrics = servicer.get_metrics()
                    print(f"\rRequests: {metrics['num_requests']}, "
                          f"Tokens: {metrics['total_tokens']}, "
                          f"Tokens/sec: {metrics['avg_tokens_per_sec']:.2f}", end="")
            except KeyboardInterrupt:
                print("\nStopping server...")

        # If in both mode, print server metrics
        if args.mode == "both" and servicer is not None:
            metrics = servicer.get_metrics()
            print("\nServer Metrics:")
            print(f"Total requests: {metrics['num_requests']}")
            print(f"Total tokens: {metrics['total_tokens']}")
            print(f"Average request time: {metrics['avg_time']*1000:.2f} ms")
            print(
                f"Throughput: {metrics['avg_tokens_per_sec']:.2f} tokens/sec")

    finally:
        # Stop server if it was started
        if server is not None:
            print("Stopping server...")
            server.stop(0)


if __name__ == "__main__":
    main()

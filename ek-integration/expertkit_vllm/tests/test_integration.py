from expertkit_vllm.plugin import register
from expertkit_vllm.grpc_client import ExpertKitClient
from expertkit_vllm.expertkit_moe import ExpertKitMoE
import unittest
import threading
import time
import torch
import io
import os
import grpc
import tempfile
from concurrent import futures
from unittest.mock import patch, MagicMock
from transformers import PretrainedConfig

# Mock gRPC server and client
from expertkit_vllm.pbpy import expert_pb2, expert_pb2_grpc

# Add path to import our module
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Import module to test


# Create a mock expert service implementation
class MockExpertComputation(expert_pb2_grpc.ExpertComputationServicer):
    def __init__(self):
        # Store call history for verification
        self.call_history = []

        # Define expert behavior
        self.expert_behaviors = {}

        # Default behavior: multiply by 2
        self.default_behavior = lambda tensor: tensor * 2

    def register_expert_behavior(self, layer, idx, behavior_fn):
        """Register a custom behavior for a specific expert."""
        self.expert_behaviors[(layer, idx)] = behavior_fn

    def Forward(self, request, context):
        """Implementation of the Forward RPC method."""
        # Extract request details
        layer = request.layer
        idx = request.idx
        batch_size = request.batch_size

        # Deserialize input tensor
        input_tensor = torch.load(io.BytesIO(request.tensor))

        # Record this call
        self.call_history.append({
            'layer': layer,
            'idx': idx,
            'batch_size': batch_size,
            'input_tensor': input_tensor.clone()  # Clone to avoid modifications
        })

        # Apply the expert behavior
        if (layer, idx) in self.expert_behaviors:
            output_tensor = self.expert_behaviors[(layer, idx)](input_tensor)
        else:
            output_tensor = self.default_behavior(input_tensor)

        # Serialize the output tensor
        buf = io.BytesIO()
        torch.save(output_tensor, buf)
        output_bytes = buf.getvalue()

        return expert_pb2.ExpertForwardReply(output_tensor=output_bytes)


# Create a test server
class TestServer:
    def __init__(self, servicer, port=50051):
        self.servicer = servicer
        self.port = port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        expert_pb2_grpc.add_ExpertComputationServicer_to_server(
            servicer, self.server)
        self.server.add_insecure_port(f'[::]:{port}')

    def start(self):
        self.server.start()
        return self

    def stop(self):
        self.server.stop(0)


class TestExpertKitIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create and start the gRPC server
        cls.expert_servicer = MockExpertComputation()
        cls.server_port = 50051
        cls.server = TestServer(cls.expert_servicer, cls.server_port).start()
        time.sleep(1)  # Wait for server to start

    @classmethod
    def tearDownClass(cls):
        # Stop the server
        cls.server.stop()

    def setUp(self):
        # Create a configuration object
        self.config = MagicMock(spec=PretrainedConfig)
        self.config.expertkit_addr = f"localhost:{self.server_port}"
        self.config.expertkit_timeout_sec = 1.0
        self.config.hidden_size = 64
        self.config.n_routed_experts = 4
        self.config.num_experts_per_tok = 2
        self.config.norm_topk_prob = True
        self.config.routed_scaling_factor = 1.0
        self.config.n_shared_experts = None
        self.config.topk_method = ""

        # Patch tensor_model_parallel functions
        self.tp_patcher = patch('expertkit_vllm.expertkit_moe.tensor_model_parallel_all_reduce',
                                side_effect=lambda x: x)
        self.tp_patcher.start()

        self.tp_size_patcher = patch('expertkit_vllm.expertkit_moe.get_tensor_model_parallel_world_size',
                                     return_value=1)
        self.tp_size_patcher.start()

        # Create the MoE instance
        self.moe = ExpertKitMoE(
            config=self.config,
            quant_config=None,
            prefix="model.layers.10.mlp"
        )

        # Reset call history
        self.expert_servicer.call_history = []

        # Set up default expert behaviors
        for i in range(4):
            # Each expert multiplies by (i+1)
            self.expert_servicer.register_expert_behavior(
                10, i,
                lambda tensor, i=i: tensor * (i + 1)
            )

    def tearDown(self):
        self.tp_patcher.stop()
        self.tp_size_patcher.stop()

    def test_client_direct_call(self):
        """Test the ExpertKitClient directly."""
        # Create a client
        client = ExpertKitClient(
            self.config.expertkit_addr, self.config.expertkit_timeout_sec)

        # Create a test tensor
        test_tensor = torch.ones(2, self.config.hidden_size)

        # Call the expert
        output = client.forward_expert(
            layer=10,
            idx=2,
            hidden_state=test_tensor
        )

        # Expert 2 should multiply by 3
        expected_output = test_tensor * 3

        # Check output
        self.assertTrue(torch.allclose(output, expected_output))

        # Check call history
        self.assertEqual(len(self.expert_servicer.call_history), 1)
        self.assertEqual(self.expert_servicer.call_history[0]['layer'], 10)
        self.assertEqual(self.expert_servicer.call_history[0]['idx'], 2)
        self.assertEqual(self.expert_servicer.call_history[0]['batch_size'], 2)
        self.assertTrue(torch.allclose(
            self.expert_servicer.call_history[0]['input_tensor'],
            test_tensor
        ))

    def test_moe_forward_pass(self):
        """Test a complete forward pass through the MoE layer."""
        # Create a routing matrix that sends tokens to specific experts
        routing_logits = torch.tensor([
            [10.0, 5.0, 0.0, -5.0],  # Token 0 routes to experts 0 and 1
            [0.0, -5.0, 10.0, 5.0],  # Token 1 routes to experts 2 and 3
        ])

        # Mock the gate to return these routing logits
        self.moe.gate = MagicMock()
        self.moe.gate.return_value = (routing_logits, None)

        # Create an input tensor
        input_tensor = torch.ones(2, self.config.hidden_size)
        input_tensor[0] = 1.0  # Token 0 values
        input_tensor[1] = 2.0  # Token 1 values

        # Call the MoE forward
        output = self.moe(input_tensor)

        # Check that all experts were called
        self.assertEqual(len(self.expert_servicer.call_history), 4)

        # Verify each expert was called correctly
        expert_calls = {}
        for call in self.expert_servicer.call_history:
            expert_calls[(call['layer'], call['idx'])] = call

        # Check expert 0 was called with token 0
        self.assertIn((10, 0), expert_calls)
        self.assertTrue(torch.allclose(
            expert_calls[(10, 0)]['input_tensor'],
            input_tensor[0:1]
        ))

        # Check expert 1 was called with token 0
        self.assertIn((10, 1), expert_calls)
        self.assertTrue(torch.allclose(
            expert_calls[(10, 1)]['input_tensor'],
            input_tensor[0:1]
        ))

        # Check expert 2 was called with token 1
        self.assertIn((10, 2), expert_calls)
        self.assertTrue(torch.allclose(
            expert_calls[(10, 2)]['input_tensor'],
            input_tensor[1:2]
        ))

        # Check expert 3 was called with token 1
        self.assertIn((10, 3), expert_calls)
        self.assertTrue(torch.allclose(
            expert_calls[(10, 3)]['input_tensor'],
            input_tensor[1:2]
        ))

        # Check the output shape
        self.assertEqual(output.shape, input_tensor.shape)

    def test_error_handling(self):
        """Test that errors are properly propagated."""
        # Make expert 0 raise an error
        def raise_error(tensor):
            raise RuntimeError("Test error")

        self.expert_servicer.register_expert_behavior(10, 0, raise_error)

        # Create routing logits that send token 0 to expert 0
        routing_logits = torch.tensor([
            [10.0, 5.0, 0.0, -5.0],  # Token 0 routes to experts 0 and 1
            [0.0, -5.0, 10.0, 5.0],  # Token 1 routes to experts 2 and 3
        ])

        # Mock the gate to return these routing logits
        self.moe.gate = MagicMock()
        self.moe.gate.return_value = (routing_logits, None)

        # Create an input tensor
        input_tensor = torch.ones(2, self.config.hidden_size)

        # Expect an error
        with self.assertRaises(RuntimeError):
            output = self.moe(input_tensor)

    def test_plugin_registration(self):
        """Test the plugin registration function."""
        # Mock the ModelRegistry
        with patch('expertkit_vllm.plugin.ModelRegistry') as mock_registry:
            # Mock the environment variable
            with patch.dict(os.environ, {"EXPERTKIT_ENABLE": "1"}):
                # Call register
                register()

                # Check that register_hook was called
                mock_registry.register_hook.assert_called_once()

                # Check the target class
                target_class = mock_registry.register_hook.call_args[0][0]
                self.assertEqual(
                    target_class, "vllm.model_executor.models.deepseek_v2.DeepseekV2MoE")

    def test_plugin_not_registered_when_disabled(self):
        """Test that the plugin is not registered when disabled."""
        # Mock the ModelRegistry
        with patch('expertkit_vllm.plugin.ModelRegistry') as mock_registry:
            # Mock the environment variable to be unset
            with patch.dict(os.environ, {"EXPERTKIT_ENABLE": "0"}):
                # Call register
                register()

                # Check that register_hook was not called
                mock_registry.register_hook.assert_not_called()


if __name__ == '__main__':
    unittest.main()

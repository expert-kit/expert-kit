import unittest
from unittest.mock import Mock, patch, MagicMock
import copy
import torch
import torch.nn as nn
import io
import sys
from transformers import PretrainedConfig

# Add the parent directory to path so we can import our module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from expertkit_vllm.expertkit_moe import ExpertKitMoE
from expertkit_vllm.grpc_client import ExpertKitClient

MODEL_CONFIG_PATH = "./tests/test_model_config.json"

class TestExpertKitMoE(unittest.TestCase):
    def setUp(self):
        # Create a mock configuration
        # self.config = Mock(spec=PretrainedConfig)
        self.config = PretrainedConfig.from_json_file(MODEL_CONFIG_PATH)
        self.config.expertkit_addr = "localhost:50051"
        self.config.expertkit_timeout_sec = 1.0
        self.config.expertkit_timeout_sec = 1.0
        self.config.hidden_size = 64
        self.config.n_routed_experts = 4
        self.config.num_experts_per_tok = 2
        self.config.norm_topk_prob = True
        self.config.routed_scaling_factor = 1.0
        self.config.n_shared_experts = None
        self.config.topk_method = ""

        # Create mock for tensor_model_parallel_all_reduce
        self.tp_all_reduce_patcher = patch(
            'expertkit_vllm.expertkit_moe.tensor_model_parallel_all_reduce',
            side_effect=lambda x: x  # Just return the input tensor
        )
        self.mock_tp_all_reduce = self.tp_all_reduce_patcher.start()

        # Create mock for get_tensor_model_parallel_world_size
        self.tp_world_size_patcher = patch(
            'expertkit_vllm.expertkit_moe.get_tensor_model_parallel_world_size',
            return_value=1
        )
        self.mock_tp_world_size = self.tp_world_size_patcher.start()

        # Create mock for ExpertKitClient
        self.client_patcher = patch(
            'expertkit_vllm.expertkit_moe.ExpertKitClient')
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = Mock(spec=ExpertKitClient)
        self.mock_client_class.return_value = self.mock_client

        # Create the MoE layer
        self.moe = ExpertKitMoE(
            config=self.config,
            quant_config=None,
            prefix="model.layers.5.mlp"
        )

    def tearDown(self):
        # Stop all patchers
        self.tp_all_reduce_patcher.stop()
        self.tp_world_size_patcher.stop()
        self.client_patcher.stop()

    def test_initialization(self):
        """Test that the MoE layer initializes correctly."""
        # Check that the client was initialized with the correct parameters
        self.mock_client_class.assert_called_once_with(
            self.config.expertkit_addr,
            self.config.expertkit_timeout_sec
        )

        # Check that the layer_id was extracted correctly
        self.assertEqual(self.moe.layer_id, 5)

        # Check that the gate was initialized
        self.assertIsNotNone(self.moe.gate)

    def test_missing_config(self):
        """Test that initialization fails if expertkit_addr is missing."""
        # Create a config without expertkit_addr
        config = copy.deepcopy(self.config)
        config.n_routed_experts = 4
        config.routed_scaling_factor = 2.5
        del config.expertkit_addr

        # Expect a RuntimeError
        with self.assertRaises(RuntimeError):
            ExpertKitMoE(config=config, quant_config=None,
                         prefix="model.layers.0.mlp")
    
    def test_forward_simple_case(self):
        """Test forward pass with a simple input case."""
        # Create an input tensor
        batch_size = 2
        hidden_dim = self.config.hidden_size
        hidden_states = torch.randn(batch_size, hidden_dim)
        
        # Create routing logits that deterministically route to experts 0 and 1
        routing_logits = torch.tensor([
            [10.0, 9.0, -10.0, -10.0],  # Token 0 routes to experts 0 and 1
            [9.0, 10.0, -10.0, -10.0],  # Token 1 routes to experts 1 and 0
        ])
        
        # Create a PyTorch-compatible mock for the gate
        original_forward = self.moe.gate.forward
        self.moe.gate.forward = lambda x: (routing_logits, None)
        
        # Create mock expert outputs
        expert0_output = torch.ones(1, hidden_dim)  # For token 0
        expert1_output = torch.ones(1, hidden_dim) * 2  # For token 1
        expert0_output2 = torch.ones(1, hidden_dim) * 3  # For token 1
        expert1_output2 = torch.ones(1, hidden_dim) * 4  # For token 0
        
        # Set up the mock client to return these outputs
        def mock_forward_expert(layer, idx, hidden_state):
            # Return different values based on expert index and input
            if idx == 0:
                if torch.allclose(hidden_state, hidden_states[0:1]):
                    return expert0_output
                else:
                    return expert0_output2
            elif idx == 1:
                if torch.allclose(hidden_state, hidden_states[1:2]):
                    return expert1_output
                else:
                    return expert1_output2
            return torch.zeros_like(hidden_state)
        
        self.mock_client.forward_expert.side_effect = mock_forward_expert
        
        # Call forward
        output = self.moe(hidden_states)
        
        # Check that the output has the right shape
        self.assertEqual(output.shape, hidden_states.shape)
        
        # Check that forward_expert was called with the right arguments
        self.assertEqual(self.mock_client.forward_expert.call_count, 4)
    
    def test_forward_with_error(self):
        """Test that errors from the client are propagated."""
        # Create an input tensor
        batch_size = 2
        hidden_dim = self.config.hidden_size
        hidden_states = torch.randn(batch_size, hidden_dim)
        
        # Create routing logits
        routing_logits = torch.tensor([
            [10.0, 9.0, -10.0, -10.0],  # Token 0 routes to experts 0 and 1
            [9.0, 10.0, -10.0, -10.0],  # Token 1 routes to experts 1 and 0
        ])
        
        # Mock the gate to return these routing logits
        original_forward = self.moe.gate.forward
        self.moe.gate.forward = lambda x: (routing_logits, None)
        self.moe.gate.return_value = (routing_logits, None)
        
        # Make the client raise an error
        self.mock_client.forward_expert.side_effect = RuntimeError("gRPC failed: UNAVAILABLE")
        
        # Expect the error to be propagated
        with self.assertRaises(RuntimeError):
            self.moe(hidden_states)
    
    def test_forward_with_no_tokens_for_expert(self):
        """Test that experts with no routed tokens are skipped."""
        # Create an input tensor
        batch_size = 2
        hidden_dim = self.config.hidden_size
        hidden_states = torch.randn(batch_size, hidden_dim)
        
        # Create routing logits that route only to experts 0 and 1
        routing_logits = torch.tensor([
            [10.0, 9.0, -10.0, -10.0],  # Token 0 routes to experts 0 and 1
            [9.0, 10.0, -10.0, -10.0],  # Token 1 routes to experts 1 and 0
        ])
        
        # Mock the gate to return these routing logits
        original_forward = self.moe.gate.forward
        self.moe.gate.forward = lambda x: (routing_logits, None)
        self.moe.gate.return_value = (routing_logits, None)
        
        # Set up a simple mock response
        self.mock_client.forward_expert.return_value = torch.ones(1, hidden_dim)
        
        # Call forward
        output = self.moe(hidden_states)
        
        # Check that forward_expert was never called for experts 2 and 3
        for call in self.mock_client.forward_expert.call_args_list:
            args, kwargs = call
            expert_idx = kwargs["idx"]
            self.assertIn(expert_idx, [0, 1])
            self.assertNotIn(expert_idx, [2, 3])
    
    def test_integration_with_real_gate(self):
        """Test integration with a real gate implementation."""
        # Create a real gate implementation
        self.moe.gate = nn.Linear(self.config.hidden_size, self.config.n_routed_experts, bias=False)
        
        # Initialize weights to give predictable routing
        with torch.no_grad():
            weights = torch.zeros(self.config.n_routed_experts, self.config.hidden_size)
            # Expert 0 looks for positive values in the first dimension
            weights[0, 0] = 1.0
            # Expert 1 looks for positive values in the second dimension
            weights[1, 1] = 1.0
            # Expert 2 looks for positive values in the third dimension
            weights[2, 2] = 1.0
            # Expert 3 looks for positive values in the fourth dimension
            weights[3, 3] = 1.0
            self.moe.gate.weight.copy_(weights)
        
        # Create an input that will route to specific experts
        batch_size = 2
        hidden_dim = self.config.hidden_size
        hidden_states = torch.zeros(batch_size, hidden_dim)
        # Token 0 should route to experts 0 and 1
        hidden_states[0, 0] = 10.0
        hidden_states[0, 1] = 9.0
        # Token 1 should route to experts 2 and 3
        hidden_states[1, 2] = 10.0
        hidden_states[1, 3] = 9.0
        
        # Set up the mock client to return simple outputs
        def mock_forward_expert(layer, idx, hidden_state):
            # Return a tensor filled with the expert index
            return torch.ones_like(hidden_state) * idx
        
        self.mock_client.forward_expert.side_effect = mock_forward_expert
        
        # Call forward
        output = self.moe(hidden_states)
        
        # Check that the client was called for the right experts
        self.assertEqual(self.mock_client.forward_expert.call_count, 4)
        
        # Check expert 0 was called with token 0
        expert0_calls = [
            call for call in self.mock_client.forward_expert.call_args_list
            if call[0][1] == 0
        ]
        self.assertEqual(len(expert0_calls), 1)
        
        # Check expert 1 was called with token 0
        expert1_calls = [
            call for call in self.mock_client.forward_expert.call_args_list
            if call[0][1] == 1
        ]
        self.assertEqual(len(expert1_calls), 1)
        
        # Check expert 2 was called with token 1
        expert2_calls = [
            call for call in self.mock_client.forward_expert.call_args_list
            if call[0][1] == 2
        ]
        self.assertEqual(len(expert2_calls), 1)
        
        # Check expert 3 was called with token 1
        expert3_calls = [
            call for call in self.mock_client.forward_expert.call_args_list
            if call[0][1] == 3
        ]
        self.assertEqual(len(expert3_calls), 1)


if __name__ == '__main__':
    unittest.main()
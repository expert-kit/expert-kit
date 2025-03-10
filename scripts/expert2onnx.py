import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


class ExpertModule(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=7168):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate_output = F.silu(gate_output) * up_output
        output = self.down_proj(intermediate_output)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    args = parser.parse_args()
    dummy_batch = 1
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    model = ExpertModule()
    default_name = f"expert_{input_dim}_{hidden_dim}.onnx"
    dummy_input = torch.randn(dummy_batch, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        default_name,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

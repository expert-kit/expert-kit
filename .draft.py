# In[]
import safetensors
from safetensors.torch import save_file
import safetensors.torch
import torch
import triton
import triton.language as tl
from triton import Config


st = safetensors.safe_open("./data/model-00001-of-000163.safetensors", framework="pt")
st.keys()


# In[]
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# In[]
dev = torch.device("cuda:0")
layer = "model.layers.0.self_attn.q_a_proj.weight"
t1 = st.get_tensor(f"{layer}").to(dev)
t2 = st.get_tensor(f"{layer}_scale_inv").to(dev)
res = weight_dequant(t1, t2, block_size=128)

output = {
    f"src": t1,
    f"src_scale": t2,
    "triton_dequanted": res,
}

save_file(output, "./w8a16active-l0q_a_proj.safetensors")


# In[]

t = torch.arange(0, 120)
t = t.reshape(5, 4, 3, 2)
t2 = torch.arange(20) / 100
t2 = t2.reshape(5, 4)

t = t.reshape(5, 4, 6)


# print(t)
# In[]
t = torch.rand(3, 8)
p = torch.rand(1, 8)
c = torch.concat((t, p), dim=0)
c


# In[]
from torch import nn
import torch

t = torch.rand(3,4)
print(t.shape)
t = t.unsqueeze(2)
print(t.shape)

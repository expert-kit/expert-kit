from vllm import LLM, SamplingParams
import torch
import os

os.environ["EXPERTKIT_ENABLE"] = "1"
os.environ["VLLM_MLA_DISABLE"] = "1"
os.environ["EXPERTKIT_DEBUG_MODE"] = "1"
os.environ["EXPERTKIT_MODE"] = "moe_mode"

prompts = [
    "Hello, my name is",
    "The president of the United",
    # "The capital of France is",
    # "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
        model="/mnt/xact/kioxia/.cache/huggingface/hub/DeepSeek-R1",
        trust_remote_code=True,

        max_model_len=16,
        enforce_eager=True,
        cpu_offload_gb=64,
        max_num_batched_tokens=1024
    )

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
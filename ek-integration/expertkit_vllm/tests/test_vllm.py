from vllm import LLM, SamplingParams
import os

os.environ["VLLM_MLA_DISABLE"] = "1"

os.environ["EK_ENABLE"] = "0"
os.environ["EK_MODEL_NAME"] = "qwen3"
os.environ["EK_MODE"] = "expert_mode"
os.environ["EK_ADDR"] = "localhost:5002"
os.environ["EK_CLIENT_TIMEOUT"] = "2"
os.environ["EK_DEBUG_MODE"] = "0"

prompts = [
    "Hello, my name is",
    "The president of the United",
    # "The capital of France is",
    # "The future of AI is",
] * 32
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
        model="/home/liucp/Documents/gitRepos/expert-kit/expert-kit-deploy/data/qwen3/qwen3",
        trust_remote_code=True,

        # dtype=torch.float16,
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
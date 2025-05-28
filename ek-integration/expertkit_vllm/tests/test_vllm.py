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
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,

        enforce_eager=True,
    )

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
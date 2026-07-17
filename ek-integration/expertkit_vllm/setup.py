from setuptools import find_packages, setup

setup(
    name="expertkit-vllm",
    version="0.1.0",
    description="ExpertMesh plugin for vLLM",
    author="ExpertMesh Team",
    packages=find_packages(),
    install_requires=[
        "expertkit-transport==0.1.0",
        "vllm==0.25.1",
    ],
    entry_points={"vllm.general_plugins": ["register_expertkit = expertkit_vllm.plugin:register"]},
)

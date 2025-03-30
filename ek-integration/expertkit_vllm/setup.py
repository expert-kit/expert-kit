from setuptools import setup, find_packages

setup(
    name="vllm-expertmesh",
    version="0.1.0",
    description="ExpertMesh plugin for vLLM",
    author="ExpertMesh Team",
    packages=find_packages(),
    install_requires=[
        "vllm",
        "torch>=2.0.0",
        "grpcio>=1.44.0",
        "protobuf>=3.19.0",
    ],
    entry_points={
        "vllm.general_plugins": [
            "register_expertkit = expertkit_vllm.plugin:register"
        ]
    },
)

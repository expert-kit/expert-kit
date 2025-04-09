from setuptools import setup, find_packages

setup(
    name="expertkit-torch",
    version="0.1.0",
    description="ExpertMesh plugin for torch",
    author="ExpertMesh Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "grpcio>=1.44.0",
        "protobuf>=5.29.0",
    ],
    entry_points={
        "torch.general_plugins": [
            "register_expertkit = expertkit_torch.plugin:register"
        ]
    },
)

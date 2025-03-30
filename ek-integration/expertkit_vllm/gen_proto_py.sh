python -m grpc_tools.protoc \
    -Iexpertkit_vllm/pbpy:./proto \
    --python_out=./expertkit_vllm/pbpy \
    --pyi_out=./expertkit_vllm/pbpy \
    --grpc_python_out=./expertkit_vllm/pbpy \
    ./proto/expert.proto
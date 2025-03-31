#!/usr/bin/sh

PROTO_DIR_PATH="../../ek-proto"
OUTPUT_DIR="./expertkit_vllm/pbpy"
PACKAGE_PREFIX="expertkit_vllm.pbpy"

# generate protobuff and grpc code
python -m grpc_tools.protoc \
    -I$PROTO_DIR_PATH \
    --python_out=$OUTPUT_DIR \
    --pyi_out=$OUTPUT_DIR \
    --grpc_python_out=$OUTPUT_DIR \
    $(find $PROTO_DIR_PATH -name '*.proto')

# fix import path
find $OUTPUT_DIR -name "*_pb2*.py" -type f -exec sed -i "s/from ek\./from $PACKAGE_PREFIX.ek./g" {} \;
find $OUTPUT_DIR -name "*_pb2*.py" -type f -exec sed -i "s/import ek\./import $PACKAGE_PREFIX.ek./g" {} \;
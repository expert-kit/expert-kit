from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ExpertForwardRequest(_message.Message):
    __slots__ = ("layer", "idx", "batch_size", "tensor")
    LAYER_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    layer: int
    idx: int
    batch_size: int
    tensor: bytes
    def __init__(self, layer: _Optional[int] = ..., idx: _Optional[int] = ..., batch_size: _Optional[int] = ..., tensor: _Optional[bytes] = ...) -> None: ...

class ExpertForwardReply(_message.Message):
    __slots__ = ("output_tensor",)
    OUTPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    output_tensor: bytes
    def __init__(self, output_tensor: _Optional[bytes] = ...) -> None: ...

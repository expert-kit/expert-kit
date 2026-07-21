from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ActivationDType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVATION_DTYPE_UNSPECIFIED: _ClassVar[ActivationDType]
    ACTIVATION_DTYPE_FP16: _ClassVar[ActivationDType]
    ACTIVATION_DTYPE_BF16: _ClassVar[ActivationDType]
    ACTIVATION_DTYPE_FP32: _ClassVar[ActivationDType]
ACTIVATION_DTYPE_UNSPECIFIED: ActivationDType
ACTIVATION_DTYPE_FP16: ActivationDType
ACTIVATION_DTYPE_BF16: ActivationDType
ACTIVATION_DTYPE_FP32: ActivationDType

class ExpertKey(_message.Message):
    __slots__ = ("layer_id", "expert_id")
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ID_FIELD_NUMBER: _ClassVar[int]
    layer_id: int
    expert_id: int
    def __init__(self, layer_id: _Optional[int] = ..., expert_id: _Optional[int] = ...) -> None: ...

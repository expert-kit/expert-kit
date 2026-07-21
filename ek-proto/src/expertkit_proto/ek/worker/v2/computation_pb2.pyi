from expertkit_proto.ek.worker.v2 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ComputeErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPUTE_ERROR_CODE_UNSPECIFIED: _ClassVar[ComputeErrorCode]
    COMPUTE_ERROR_BUSY: _ClassVar[ComputeErrorCode]
    COMPUTE_ERROR_DRAINING: _ClassVar[ComputeErrorCode]
    COMPUTE_ERROR_STALE_TOPOLOGY: _ClassVar[ComputeErrorCode]
    COMPUTE_ERROR_EXPERT_NOT_READY: _ClassVar[ComputeErrorCode]
    COMPUTE_ERROR_INVALID_REQUEST: _ClassVar[ComputeErrorCode]
    COMPUTE_ERROR_UNSUPPORTED: _ClassVar[ComputeErrorCode]
COMPUTE_ERROR_CODE_UNSPECIFIED: ComputeErrorCode
COMPUTE_ERROR_BUSY: ComputeErrorCode
COMPUTE_ERROR_DRAINING: ComputeErrorCode
COMPUTE_ERROR_STALE_TOPOLOGY: ComputeErrorCode
COMPUTE_ERROR_EXPERT_NOT_READY: ComputeErrorCode
COMPUTE_ERROR_INVALID_REQUEST: ComputeErrorCode
COMPUTE_ERROR_UNSUPPORTED: ComputeErrorCode

class ExecuteRequest(_message.Message):
    __slots__ = ("instance_id", "layer_id", "topology_version", "token_count", "hidden_dim", "top_k", "dtype", "hidden_states", "expert_ids", "routing_weights")
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_DIM_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    EXPERT_IDS_FIELD_NUMBER: _ClassVar[int]
    ROUTING_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    layer_id: int
    topology_version: int
    token_count: int
    hidden_dim: int
    top_k: int
    dtype: _common_pb2.ActivationDType
    hidden_states: bytes
    expert_ids: bytes
    routing_weights: bytes
    def __init__(self, instance_id: _Optional[int] = ..., layer_id: _Optional[int] = ..., topology_version: _Optional[int] = ..., token_count: _Optional[int] = ..., hidden_dim: _Optional[int] = ..., top_k: _Optional[int] = ..., dtype: _Optional[_Union[_common_pb2.ActivationDType, str]] = ..., hidden_states: _Optional[bytes] = ..., expert_ids: _Optional[bytes] = ..., routing_weights: _Optional[bytes] = ...) -> None: ...

class ComputeError(_message.Message):
    __slots__ = ("code", "retryable", "observed_topology_version", "min_topology_version", "unavailable_expert_ids", "diagnostic")
    CODE_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    MIN_TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_EXPERT_IDS_FIELD_NUMBER: _ClassVar[int]
    DIAGNOSTIC_FIELD_NUMBER: _ClassVar[int]
    code: ComputeErrorCode
    retryable: bool
    observed_topology_version: int
    min_topology_version: int
    unavailable_expert_ids: _containers.RepeatedScalarFieldContainer[int]
    diagnostic: str
    def __init__(self, code: _Optional[_Union[ComputeErrorCode, str]] = ..., retryable: bool = ..., observed_topology_version: _Optional[int] = ..., min_topology_version: _Optional[int] = ..., unavailable_expert_ids: _Optional[_Iterable[int]] = ..., diagnostic: _Optional[str] = ...) -> None: ...

class ExecuteResponse(_message.Message):
    __slots__ = ("partial_output", "error")
    PARTIAL_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    partial_output: bytes
    error: ComputeError
    def __init__(self, partial_output: _Optional[bytes] = ..., error: _Optional[_Union[ComputeError, _Mapping]] = ...) -> None: ...

class OpenSharedMemoryRequest(_message.Message):
    __slots__ = ("instance_id", "session_id", "segment_name", "segment_size", "slot_count", "max_batch_tokens", "hidden_dim", "top_k", "dtype")
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_SIZE_FIELD_NUMBER: _ClassVar[int]
    SLOT_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_TOKENS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_DIM_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    session_id: str
    segment_name: str
    segment_size: int
    slot_count: int
    max_batch_tokens: int
    hidden_dim: int
    top_k: int
    dtype: _common_pb2.ActivationDType
    def __init__(self, instance_id: _Optional[int] = ..., session_id: _Optional[str] = ..., segment_name: _Optional[str] = ..., segment_size: _Optional[int] = ..., slot_count: _Optional[int] = ..., max_batch_tokens: _Optional[int] = ..., hidden_dim: _Optional[int] = ..., top_k: _Optional[int] = ..., dtype: _Optional[_Union[_common_pb2.ActivationDType, str]] = ...) -> None: ...

class OpenSharedMemoryResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ExecuteSharedMemoryRequest(_message.Message):
    __slots__ = ("session_id", "slot_index", "generation", "layer_id", "topology_version", "token_count", "timeout_micros")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SLOT_INDEX_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MICROS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    slot_index: int
    generation: int
    layer_id: int
    topology_version: int
    token_count: int
    timeout_micros: int
    def __init__(self, session_id: _Optional[str] = ..., slot_index: _Optional[int] = ..., generation: _Optional[int] = ..., layer_id: _Optional[int] = ..., topology_version: _Optional[int] = ..., token_count: _Optional[int] = ..., timeout_micros: _Optional[int] = ...) -> None: ...

class ExecuteSharedMemoryResponse(_message.Message):
    __slots__ = ("completed_generation", "error")
    COMPLETED_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    completed_generation: int
    error: ComputeError
    def __init__(self, completed_generation: _Optional[int] = ..., error: _Optional[_Union[ComputeError, _Mapping]] = ...) -> None: ...

class CloseSharedMemoryRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseSharedMemoryResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

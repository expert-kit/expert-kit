from ek.object.v1 import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ForwardReq(_message.Message):
    __slots__ = ("instance_id", "experts", "model_name", "layer_id", "tensor")
    class ExpertsInfo(_message.Message):
        __slots__ = ("activation",)
        ACTIVATION_FIELD_NUMBER: _ClassVar[int]
        activation: _containers.RepeatedScalarFieldContainer[bool]
        def __init__(self, activation: _Optional[_Iterable[bool]] = ...) -> None: ...
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    instance_id: str
    experts: _containers.RepeatedCompositeFieldContainer[ForwardReq.ExpertsInfo]
    model_name: str
    layer_id: str
    tensor: bytes
    def __init__(self, instance_id: _Optional[str] = ..., experts: _Optional[_Iterable[_Union[ForwardReq.ExpertsInfo, _Mapping]]] = ..., model_name: _Optional[str] = ..., layer_id: _Optional[str] = ..., tensor: _Optional[bytes] = ...) -> None: ...

class ForwardResp(_message.Message):
    __slots__ = ("output_tensor",)
    OUTPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    output_tensor: bytes
    def __init__(self, output_tensor: _Optional[bytes] = ...) -> None: ...

class ExpertState(_message.Message):
    __slots__ = ("stage",)
    class Stage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STAGE_UNSPECIFIED: _ClassVar[ExpertState.Stage]
        STAGE_ACTIVE: _ClassVar[ExpertState.Stage]
        STAGE_LOADING: _ClassVar[ExpertState.Stage]
        STAGE_EVICTING: _ClassVar[ExpertState.Stage]
    STAGE_UNSPECIFIED: ExpertState.Stage
    STAGE_ACTIVE: ExpertState.Stage
    STAGE_LOADING: ExpertState.Stage
    STAGE_EVICTING: ExpertState.Stage
    STAGE_FIELD_NUMBER: _ClassVar[int]
    stage: ExpertState.Stage
    def __init__(self, stage: _Optional[_Union[ExpertState.Stage, str]] = ...) -> None: ...

class RdmaEndpoint(_message.Message):
    __slots__ = ("qp_endpoint", "memory_region")
    QP_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_REGION_FIELD_NUMBER: _ClassVar[int]
    qp_endpoint: str
    memory_region: str
    def __init__(self, qp_endpoint: _Optional[str] = ..., memory_region: _Optional[str] = ...) -> None: ...

class RdmaEndpointPair(_message.Message):
    __slots__ = ("request_endpoint", "response_endpoint")
    REQUEST_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    request_endpoint: RdmaEndpoint
    response_endpoint: RdmaEndpoint
    def __init__(self, request_endpoint: _Optional[_Union[RdmaEndpoint, _Mapping]] = ..., response_endpoint: _Optional[_Union[RdmaEndpoint, _Mapping]] = ...) -> None: ...

class ExchangeReq(_message.Message):
    __slots__ = ("id", "addr", "channel", "device", "last_will", "rdma_endpoints")
    ID_FIELD_NUMBER: _ClassVar[int]
    ADDR_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    LAST_WILL_FIELD_NUMBER: _ClassVar[int]
    RDMA_ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    addr: str
    channel: str
    device: str
    last_will: bool
    rdma_endpoints: RdmaEndpointPair
    def __init__(self, id: _Optional[str] = ..., addr: _Optional[str] = ..., channel: _Optional[str] = ..., device: _Optional[str] = ..., last_will: bool = ..., rdma_endpoints: _Optional[_Union[RdmaEndpointPair, _Mapping]] = ...) -> None: ...

class ExchangeResp(_message.Message):
    __slots__ = ("state", "rdma_endpoints")
    class ExpertWithState(_message.Message):
        __slots__ = ("target",)
        TARGET_FIELD_NUMBER: _ClassVar[int]
        target: _object_pb2.ExpertSlice
        def __init__(self, target: _Optional[_Union[_object_pb2.ExpertSlice, _Mapping]] = ...) -> None: ...
    STATE_FIELD_NUMBER: _ClassVar[int]
    RDMA_ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    state: ExchangeResp.ExpertWithState
    rdma_endpoints: RdmaEndpointPair
    def __init__(self, state: _Optional[_Union[ExchangeResp.ExpertWithState, _Mapping]] = ..., rdma_endpoints: _Optional[_Union[RdmaEndpointPair, _Mapping]] = ...) -> None: ...

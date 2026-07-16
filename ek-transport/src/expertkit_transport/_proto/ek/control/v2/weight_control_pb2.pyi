from expertkit_transport._proto.ek.worker.v2 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExpertStateKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXPERT_STATE_UNSPECIFIED: _ClassVar[ExpertStateKind]
    EXPERT_READY: _ClassVar[ExpertStateKind]
    EXPERT_FAILED: _ClassVar[ExpertStateKind]
    EXPERT_REMOVED: _ClassVar[ExpertStateKind]

class WeightLoadStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHT_LOAD_STAGE_UNSPECIFIED: _ClassVar[WeightLoadStage]
    WEIGHT_LOAD_FETCH: _ClassVar[WeightLoadStage]
    WEIGHT_LOAD_READ: _ClassVar[WeightLoadStage]
    WEIGHT_LOAD_PARSE: _ClassVar[WeightLoadStage]
    WEIGHT_LOAD_VALIDATE: _ClassVar[WeightLoadStage]
    WEIGHT_LOAD_CONVERT: _ClassVar[WeightLoadStage]
    WEIGHT_LOAD_PLACE: _ClassVar[WeightLoadStage]

class WeightLoadErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHT_LOAD_ERROR_CODE_UNSPECIFIED: _ClassVar[WeightLoadErrorCode]
    WEIGHT_LOAD_ERROR_NOT_FOUND: _ClassVar[WeightLoadErrorCode]
    WEIGHT_LOAD_ERROR_IO: _ClassVar[WeightLoadErrorCode]
    WEIGHT_LOAD_ERROR_NETWORK: _ClassVar[WeightLoadErrorCode]
    WEIGHT_LOAD_ERROR_INVALID_FORMAT: _ClassVar[WeightLoadErrorCode]
    WEIGHT_LOAD_ERROR_UNEXPECTED_METADATA: _ClassVar[WeightLoadErrorCode]
    WEIGHT_LOAD_ERROR_UNSUPPORTED: _ClassVar[WeightLoadErrorCode]
    WEIGHT_LOAD_ERROR_INTERNAL: _ClassVar[WeightLoadErrorCode]
EXPERT_STATE_UNSPECIFIED: ExpertStateKind
EXPERT_READY: ExpertStateKind
EXPERT_FAILED: ExpertStateKind
EXPERT_REMOVED: ExpertStateKind
WEIGHT_LOAD_STAGE_UNSPECIFIED: WeightLoadStage
WEIGHT_LOAD_FETCH: WeightLoadStage
WEIGHT_LOAD_READ: WeightLoadStage
WEIGHT_LOAD_PARSE: WeightLoadStage
WEIGHT_LOAD_VALIDATE: WeightLoadStage
WEIGHT_LOAD_CONVERT: WeightLoadStage
WEIGHT_LOAD_PLACE: WeightLoadStage
WEIGHT_LOAD_ERROR_CODE_UNSPECIFIED: WeightLoadErrorCode
WEIGHT_LOAD_ERROR_NOT_FOUND: WeightLoadErrorCode
WEIGHT_LOAD_ERROR_IO: WeightLoadErrorCode
WEIGHT_LOAD_ERROR_NETWORK: WeightLoadErrorCode
WEIGHT_LOAD_ERROR_INVALID_FORMAT: WeightLoadErrorCode
WEIGHT_LOAD_ERROR_UNEXPECTED_METADATA: WeightLoadErrorCode
WEIGHT_LOAD_ERROR_UNSUPPORTED: WeightLoadErrorCode
WEIGHT_LOAD_ERROR_INTERNAL: WeightLoadErrorCode

class WorkerWeightMessage(_message.Message):
    __slots__ = ("open", "full_state", "state_updates", "drain_complete")
    OPEN_FIELD_NUMBER: _ClassVar[int]
    FULL_STATE_FIELD_NUMBER: _ClassVar[int]
    STATE_UPDATES_FIELD_NUMBER: _ClassVar[int]
    DRAIN_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    open: OpenWeightStream
    full_state: FullExpertStatePart
    state_updates: ExpertStateUpdates
    drain_complete: DrainComplete
    def __init__(self, open: _Optional[_Union[OpenWeightStream, _Mapping]] = ..., full_state: _Optional[_Union[FullExpertStatePart, _Mapping]] = ..., state_updates: _Optional[_Union[ExpertStateUpdates, _Mapping]] = ..., drain_complete: _Optional[_Union[DrainComplete, _Mapping]] = ...) -> None: ...

class OpenWeightStream(_message.Message):
    __slots__ = ("worker_id", "start_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    START_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    start_id: str
    def __init__(self, worker_id: _Optional[str] = ..., start_id: _Optional[str] = ...) -> None: ...

class FullExpertStatePart(_message.Message):
    __slots__ = ("placement_generation", "report_sequence", "part_index", "part_count", "experts")
    PLACEMENT_GENERATION_FIELD_NUMBER: _ClassVar[int]
    REPORT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    PART_INDEX_FIELD_NUMBER: _ClassVar[int]
    PART_COUNT_FIELD_NUMBER: _ClassVar[int]
    EXPERTS_FIELD_NUMBER: _ClassVar[int]
    placement_generation: int
    report_sequence: int
    part_index: int
    part_count: int
    experts: _containers.RepeatedCompositeFieldContainer[ExpertState]
    def __init__(self, placement_generation: _Optional[int] = ..., report_sequence: _Optional[int] = ..., part_index: _Optional[int] = ..., part_count: _Optional[int] = ..., experts: _Optional[_Iterable[_Union[ExpertState, _Mapping]]] = ...) -> None: ...

class ExpertStateUpdates(_message.Message):
    __slots__ = ("placement_generation", "report_sequence", "experts")
    PLACEMENT_GENERATION_FIELD_NUMBER: _ClassVar[int]
    REPORT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    EXPERTS_FIELD_NUMBER: _ClassVar[int]
    placement_generation: int
    report_sequence: int
    experts: _containers.RepeatedCompositeFieldContainer[ExpertState]
    def __init__(self, placement_generation: _Optional[int] = ..., report_sequence: _Optional[int] = ..., experts: _Optional[_Iterable[_Union[ExpertState, _Mapping]]] = ...) -> None: ...

class ExpertState(_message.Message):
    __slots__ = ("layer_id", "expert_id", "state", "failure")
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    FAILURE_FIELD_NUMBER: _ClassVar[int]
    layer_id: int
    expert_id: int
    state: ExpertStateKind
    failure: WeightLoadFailure
    def __init__(self, layer_id: _Optional[int] = ..., expert_id: _Optional[int] = ..., state: _Optional[_Union[ExpertStateKind, str]] = ..., failure: _Optional[_Union[WeightLoadFailure, _Mapping]] = ...) -> None: ...

class WeightLoadFailure(_message.Message):
    __slots__ = ("stage", "code", "retryable", "diagnostic")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    DIAGNOSTIC_FIELD_NUMBER: _ClassVar[int]
    stage: WeightLoadStage
    code: WeightLoadErrorCode
    retryable: bool
    diagnostic: str
    def __init__(self, stage: _Optional[_Union[WeightLoadStage, str]] = ..., code: _Optional[_Union[WeightLoadErrorCode, str]] = ..., retryable: bool = ..., diagnostic: _Optional[str] = ...) -> None: ...

class DrainComplete(_message.Message):
    __slots__ = ("drain_id",)
    DRAIN_ID_FIELD_NUMBER: _ClassVar[int]
    drain_id: int
    def __init__(self, drain_id: _Optional[int] = ...) -> None: ...

class ControllerWeightMessage(_message.Message):
    __slots__ = ("targets", "state_ack", "drain")
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    STATE_ACK_FIELD_NUMBER: _ClassVar[int]
    DRAIN_FIELD_NUMBER: _ClassVar[int]
    targets: TargetExpertListPart
    state_ack: StateReportAck
    drain: DrainAuthorizationPart
    def __init__(self, targets: _Optional[_Union[TargetExpertListPart, _Mapping]] = ..., state_ack: _Optional[_Union[StateReportAck, _Mapping]] = ..., drain: _Optional[_Union[DrainAuthorizationPart, _Mapping]] = ...) -> None: ...

class TargetExpertListPart(_message.Message):
    __slots__ = ("placement_generation", "part_index", "part_count", "experts")
    PLACEMENT_GENERATION_FIELD_NUMBER: _ClassVar[int]
    PART_INDEX_FIELD_NUMBER: _ClassVar[int]
    PART_COUNT_FIELD_NUMBER: _ClassVar[int]
    EXPERTS_FIELD_NUMBER: _ClassVar[int]
    placement_generation: int
    part_index: int
    part_count: int
    experts: _containers.RepeatedCompositeFieldContainer[TargetExpert]
    def __init__(self, placement_generation: _Optional[int] = ..., part_index: _Optional[int] = ..., part_count: _Optional[int] = ..., experts: _Optional[_Iterable[_Union[TargetExpert, _Mapping]]] = ...) -> None: ...

class TargetExpert(_message.Message):
    __slots__ = ("layer_id", "expert_id", "target_device", "peer_weight_endpoints")
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_DEVICE_FIELD_NUMBER: _ClassVar[int]
    PEER_WEIGHT_ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    layer_id: int
    expert_id: int
    target_device: str
    peer_weight_endpoints: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, layer_id: _Optional[int] = ..., expert_id: _Optional[int] = ..., target_device: _Optional[str] = ..., peer_weight_endpoints: _Optional[_Iterable[str]] = ...) -> None: ...

class StateReportAck(_message.Message):
    __slots__ = ("report_sequence",)
    REPORT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    report_sequence: int
    def __init__(self, report_sequence: _Optional[int] = ...) -> None: ...

class DrainAuthorizationPart(_message.Message):
    __slots__ = ("drain_id", "placement_generation", "min_topology_version", "stop_accepting_all_computation", "part_index", "part_count", "experts")
    DRAIN_ID_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_GENERATION_FIELD_NUMBER: _ClassVar[int]
    MIN_TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    STOP_ACCEPTING_ALL_COMPUTATION_FIELD_NUMBER: _ClassVar[int]
    PART_INDEX_FIELD_NUMBER: _ClassVar[int]
    PART_COUNT_FIELD_NUMBER: _ClassVar[int]
    EXPERTS_FIELD_NUMBER: _ClassVar[int]
    drain_id: int
    placement_generation: int
    min_topology_version: int
    stop_accepting_all_computation: bool
    part_index: int
    part_count: int
    experts: _containers.RepeatedCompositeFieldContainer[_common_pb2.ExpertKey]
    def __init__(self, drain_id: _Optional[int] = ..., placement_generation: _Optional[int] = ..., min_topology_version: _Optional[int] = ..., stop_accepting_all_computation: bool = ..., part_index: _Optional[int] = ..., part_count: _Optional[int] = ..., experts: _Optional[_Iterable[_Union[_common_pb2.ExpertKey, _Mapping]]] = ...) -> None: ...

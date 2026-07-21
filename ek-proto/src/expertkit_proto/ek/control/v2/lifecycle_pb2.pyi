from expertkit_proto.ek.worker.v2 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkerRunState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_RUN_STATE_UNSPECIFIED: _ClassVar[WorkerRunState]
    WORKER_RUNNING: _ClassVar[WorkerRunState]
    WORKER_SHUTTING_DOWN: _ClassVar[WorkerRunState]

class WorkerTransportType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_TRANSPORT_TYPE_UNSPECIFIED: _ClassVar[WorkerTransportType]
    WORKER_TRANSPORT_GRPC: _ClassVar[WorkerTransportType]
    WORKER_TRANSPORT_SHM: _ClassVar[WorkerTransportType]
WORKER_RUN_STATE_UNSPECIFIED: WorkerRunState
WORKER_RUNNING: WorkerRunState
WORKER_SHUTTING_DOWN: WorkerRunState
WORKER_TRANSPORT_TYPE_UNSPECIFIED: WorkerTransportType
WORKER_TRANSPORT_GRPC: WorkerTransportType
WORKER_TRANSPORT_SHM: WorkerTransportType

class RegisterWorkerRequest(_message.Message):
    __slots__ = ("worker_id", "start_id", "instance_id", "computation_endpoint", "peer_weight_endpoint", "backend", "activation_dtype", "device", "max_batch_tokens", "max_active_batches_per_device", "max_pending_batches_per_device", "transport_type")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    START_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    PEER_WEIGHT_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_DTYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_ACTIVE_BATCHES_PER_DEVICE_FIELD_NUMBER: _ClassVar[int]
    MAX_PENDING_BATCHES_PER_DEVICE_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_TYPE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    start_id: str
    instance_id: int
    computation_endpoint: str
    peer_weight_endpoint: str
    backend: str
    activation_dtype: _common_pb2.ActivationDType
    device: WorkerDevice
    max_batch_tokens: int
    max_active_batches_per_device: int
    max_pending_batches_per_device: int
    transport_type: WorkerTransportType
    def __init__(self, worker_id: _Optional[str] = ..., start_id: _Optional[str] = ..., instance_id: _Optional[int] = ..., computation_endpoint: _Optional[str] = ..., peer_weight_endpoint: _Optional[str] = ..., backend: _Optional[str] = ..., activation_dtype: _Optional[_Union[_common_pb2.ActivationDType, str]] = ..., device: _Optional[_Union[WorkerDevice, _Mapping]] = ..., max_batch_tokens: _Optional[int] = ..., max_active_batches_per_device: _Optional[int] = ..., max_pending_batches_per_device: _Optional[int] = ..., transport_type: _Optional[_Union[WorkerTransportType, str]] = ...) -> None: ...

class WorkerDevice(_message.Message):
    __slots__ = ("device", "max_experts")
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MAX_EXPERTS_FIELD_NUMBER: _ClassVar[int]
    device: str
    max_experts: int
    def __init__(self, device: _Optional[str] = ..., max_experts: _Optional[int] = ...) -> None: ...

class RegisterWorkerResponse(_message.Message):
    __slots__ = ("current_topology_version", "current_placement_generation")
    CURRENT_TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PLACEMENT_GENERATION_FIELD_NUMBER: _ClassVar[int]
    current_topology_version: int
    current_placement_generation: int
    def __init__(self, current_topology_version: _Optional[int] = ..., current_placement_generation: _Optional[int] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("worker_id", "start_id", "sequence", "state")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    START_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    start_id: str
    sequence: int
    state: WorkerRunState
    def __init__(self, worker_id: _Optional[str] = ..., start_id: _Optional[str] = ..., sequence: _Optional[int] = ..., state: _Optional[_Union[WorkerRunState, str]] = ...) -> None: ...

class HeartbeatSummary(_message.Message):
    __slots__ = ("last_sequence",)
    LAST_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    last_sequence: int
    def __init__(self, last_sequence: _Optional[int] = ...) -> None: ...

class WatchTopologyRequest(_message.Message):
    __slots__ = ("instance_id", "current_version")
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    current_version: int
    def __init__(self, instance_id: _Optional[int] = ..., current_version: _Optional[int] = ...) -> None: ...

class TopologyMessage(_message.Message):
    __slots__ = ("snapshot", "update")
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    snapshot: TopologySnapshotPart
    update: TopologyUpdatePart
    def __init__(self, snapshot: _Optional[_Union[TopologySnapshotPart, _Mapping]] = ..., update: _Optional[_Union[TopologyUpdatePart, _Mapping]] = ...) -> None: ...

class TopologySnapshotPart(_message.Message):
    __slots__ = ("instance_id", "topology_version", "part_index", "part_count", "routes")
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    PART_INDEX_FIELD_NUMBER: _ClassVar[int]
    PART_COUNT_FIELD_NUMBER: _ClassVar[int]
    ROUTES_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    topology_version: int
    part_index: int
    part_count: int
    routes: _containers.RepeatedCompositeFieldContainer[ExpertRoute]
    def __init__(self, instance_id: _Optional[int] = ..., topology_version: _Optional[int] = ..., part_index: _Optional[int] = ..., part_count: _Optional[int] = ..., routes: _Optional[_Iterable[_Union[ExpertRoute, _Mapping]]] = ...) -> None: ...

class TopologyUpdatePart(_message.Message):
    __slots__ = ("instance_id", "previous_version", "topology_version", "part_index", "part_count", "changes")
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_VERSION_FIELD_NUMBER: _ClassVar[int]
    PART_INDEX_FIELD_NUMBER: _ClassVar[int]
    PART_COUNT_FIELD_NUMBER: _ClassVar[int]
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    previous_version: int
    topology_version: int
    part_index: int
    part_count: int
    changes: _containers.RepeatedCompositeFieldContainer[RouteChange]
    def __init__(self, instance_id: _Optional[int] = ..., previous_version: _Optional[int] = ..., topology_version: _Optional[int] = ..., part_index: _Optional[int] = ..., part_count: _Optional[int] = ..., changes: _Optional[_Iterable[_Union[RouteChange, _Mapping]]] = ...) -> None: ...

class ExpertRoute(_message.Message):
    __slots__ = ("layer_id", "expert_id", "replicas")
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ID_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    layer_id: int
    expert_id: int
    replicas: _containers.RepeatedCompositeFieldContainer[WorkerRoute]
    def __init__(self, layer_id: _Optional[int] = ..., expert_id: _Optional[int] = ..., replicas: _Optional[_Iterable[_Union[WorkerRoute, _Mapping]]] = ...) -> None: ...

class WorkerRoute(_message.Message):
    __slots__ = ("worker_id", "start_id", "computation_endpoint", "device", "max_active_batches", "max_pending_batches", "max_batch_tokens", "transport_type")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    START_ID_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MAX_ACTIVE_BATCHES_FIELD_NUMBER: _ClassVar[int]
    MAX_PENDING_BATCHES_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_TYPE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    start_id: str
    computation_endpoint: str
    device: str
    max_active_batches: int
    max_pending_batches: int
    max_batch_tokens: int
    transport_type: WorkerTransportType
    def __init__(self, worker_id: _Optional[str] = ..., start_id: _Optional[str] = ..., computation_endpoint: _Optional[str] = ..., device: _Optional[str] = ..., max_active_batches: _Optional[int] = ..., max_pending_batches: _Optional[int] = ..., max_batch_tokens: _Optional[int] = ..., transport_type: _Optional[_Union[WorkerTransportType, str]] = ...) -> None: ...

class RouteChange(_message.Message):
    __slots__ = ("layer_id", "expert_id", "replicas")
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ID_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    layer_id: int
    expert_id: int
    replicas: _containers.RepeatedCompositeFieldContainer[WorkerRoute]
    def __init__(self, layer_id: _Optional[int] = ..., expert_id: _Optional[int] = ..., replicas: _Optional[_Iterable[_Union[WorkerRoute, _Mapping]]] = ...) -> None: ...

from ek.object.v1 import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreatePlanReq(_message.Message):
    __slots__ = ("plan",)
    PLAN_FIELD_NUMBER: _ClassVar[int]
    plan: _object_pb2.SchedulePlan
    def __init__(self, plan: _Optional[_Union[_object_pb2.SchedulePlan, _Mapping]] = ...) -> None: ...

class CreatePlanResp(_message.Message):
    __slots__ = ("plan",)
    PLAN_FIELD_NUMBER: _ClassVar[int]
    plan: _object_pb2.SchedulePlan
    def __init__(self, plan: _Optional[_Union[_object_pb2.SchedulePlan, _Mapping]] = ...) -> None: ...

class DeletePlanReq(_message.Message):
    __slots__ = ("plan_id",)
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    plan_id: str
    def __init__(self, plan_id: _Optional[str] = ...) -> None: ...

class DeletePlanResp(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InspectPlanReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InspectPlanResp(_message.Message):
    __slots__ = ("plan",)
    PLAN_FIELD_NUMBER: _ClassVar[int]
    plan: _containers.RepeatedCompositeFieldContainer[_object_pb2.SchedulePlan]
    def __init__(self, plan: _Optional[_Iterable[_Union[_object_pb2.SchedulePlan, _Mapping]]] = ...) -> None: ...

class ResolveRequest(_message.Message):
    __slots__ = ("node_id", "slice_id")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    slice_id: str
    def __init__(self, node_id: _Optional[str] = ..., slice_id: _Optional[str] = ...) -> None: ...

class ResolveReply(_message.Message):
    __slots__ = ("node_id", "slice_id")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    slice_id: str
    def __init__(self, node_id: _Optional[str] = ..., slice_id: _Optional[str] = ...) -> None: ...

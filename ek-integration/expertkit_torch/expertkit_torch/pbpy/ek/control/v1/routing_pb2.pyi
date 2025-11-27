from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetRoutingReq(_message.Message):
    __slots__ = ("expert_ids",)
    EXPERT_IDS_FIELD_NUMBER: _ClassVar[int]
    expert_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, expert_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class GetRoutingResp(_message.Message):
    __slots__ = ("routing", "version")
    class RoutingEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ROUTING_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    routing: _containers.ScalarMap[str, str]
    version: int
    def __init__(self, routing: _Optional[_Mapping[str, str]] = ..., version: _Optional[int] = ...) -> None: ...

class SubscribeRoutingReq(_message.Message):
    __slots__ = ("current_version",)
    CURRENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    current_version: int
    def __init__(self, current_version: _Optional[int] = ...) -> None: ...

class RoutingUpdate(_message.Message):
    __slots__ = ("type", "expert_id", "worker_addr", "version")
    class ChangeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ADDED: _ClassVar[RoutingUpdate.ChangeType]
        REMOVED: _ClassVar[RoutingUpdate.ChangeType]
        MODIFIED: _ClassVar[RoutingUpdate.ChangeType]
    ADDED: RoutingUpdate.ChangeType
    REMOVED: RoutingUpdate.ChangeType
    MODIFIED: RoutingUpdate.ChangeType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ADDR_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    type: RoutingUpdate.ChangeType
    expert_id: str
    worker_addr: str
    version: int
    def __init__(self, type: _Optional[_Union[RoutingUpdate.ChangeType, str]] = ..., expert_id: _Optional[str] = ..., worker_addr: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

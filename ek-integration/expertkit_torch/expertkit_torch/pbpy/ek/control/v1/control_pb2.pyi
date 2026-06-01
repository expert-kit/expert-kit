from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RebalanceReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RebalanceResp(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DuplicateReq(_message.Message):
    __slots__ = ("hostnames",)
    HOSTNAMES_FIELD_NUMBER: _ClassVar[int]
    hostnames: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, hostnames: _Optional[_Iterable[str]] = ...) -> None: ...

class DuplicateResp(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ManualReq(_message.Message):
    __slots__ = ("hostnames", "layers")
    HOSTNAMES_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    hostnames: _containers.RepeatedScalarFieldContainer[str]
    layers: str
    def __init__(self, hostnames: _Optional[_Iterable[str]] = ..., layers: _Optional[str] = ...) -> None: ...

class ManualResp(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

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

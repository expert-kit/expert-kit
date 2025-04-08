from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Metadata(_message.Message):
    __slots__ = ("id", "name", "tags")
    class TagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    tags: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., tags: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ExpertSlice(_message.Message):
    __slots__ = ("meta", "expert_meta", "replication")
    META_FIELD_NUMBER: _ClassVar[int]
    EXPERT_META_FIELD_NUMBER: _ClassVar[int]
    REPLICATION_FIELD_NUMBER: _ClassVar[int]
    meta: Metadata
    expert_meta: _containers.RepeatedCompositeFieldContainer[Metadata]
    replication: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, meta: _Optional[_Union[Metadata, _Mapping]] = ..., expert_meta: _Optional[_Iterable[_Union[Metadata, _Mapping]]] = ..., replication: _Optional[_Iterable[int]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("meta", "control_address", "data_address")
    META_FIELD_NUMBER: _ClassVar[int]
    CONTROL_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DATA_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    meta: Metadata
    control_address: str
    data_address: str
    def __init__(self, meta: _Optional[_Union[Metadata, _Mapping]] = ..., control_address: _Optional[str] = ..., data_address: _Optional[str] = ...) -> None: ...

class SliceAffinity(_message.Message):
    __slots__ = ("node_id", "slice_id")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    slice_id: str
    def __init__(self, node_id: _Optional[str] = ..., slice_id: _Optional[str] = ...) -> None: ...

class SchedulePlan(_message.Message):
    __slots__ = ("meta", "slices", "affinity")
    META_FIELD_NUMBER: _ClassVar[int]
    SLICES_FIELD_NUMBER: _ClassVar[int]
    AFFINITY_FIELD_NUMBER: _ClassVar[int]
    meta: Metadata
    slices: _containers.RepeatedCompositeFieldContainer[ExpertSlice]
    affinity: _containers.RepeatedCompositeFieldContainer[SliceAffinity]
    def __init__(self, meta: _Optional[_Union[Metadata, _Mapping]] = ..., slices: _Optional[_Iterable[_Union[ExpertSlice, _Mapping]]] = ..., affinity: _Optional[_Iterable[_Union[SliceAffinity, _Mapping]]] = ...) -> None: ...

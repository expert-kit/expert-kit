# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: ek/object/v1/object.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'ek/object/v1/object.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19\x65k/object/v1/object.proto\x12\x0c\x65k.object.v1\"\x81\x01\n\x08Metadata\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12.\n\x04tags\x18\x03 \x03(\x0b\x32 .ek.object.v1.Metadata.TagsEntry\x1a+\n\tTagsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"u\n\x0b\x45xpertSlice\x12$\n\x04meta\x18\x01 \x01(\x0b\x32\x16.ek.object.v1.Metadata\x12+\n\x0b\x65xpert_meta\x18\x02 \x03(\x0b\x32\x16.ek.object.v1.Metadata\x12\x13\n\x0breplication\x18\x03 \x03(\x03\"[\n\x04Node\x12$\n\x04meta\x18\x01 \x01(\x0b\x32\x16.ek.object.v1.Metadata\x12\x17\n\x0f\x63ontrol_address\x18\x02 \x01(\t\x12\x14\n\x0c\x64\x61ta_address\x18\x03 \x01(\t\"2\n\rSliceAffinity\x12\x0f\n\x07node_id\x18\x01 \x01(\t\x12\x10\n\x08slice_id\x18\x02 \x01(\t\"\x8e\x01\n\x0cSchedulePlan\x12$\n\x04meta\x18\x01 \x01(\x0b\x32\x16.ek.object.v1.Metadata\x12)\n\x06slices\x18\x02 \x03(\x0b\x32\x19.ek.object.v1.ExpertSlice\x12-\n\x08\x61\x66\x66inity\x18\x03 \x03(\x0b\x32\x1b.ek.object.v1.SliceAffinityb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ek.object.v1.object_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_METADATA_TAGSENTRY']._loaded_options = None
  _globals['_METADATA_TAGSENTRY']._serialized_options = b'8\001'
  _globals['_METADATA']._serialized_start=44
  _globals['_METADATA']._serialized_end=173
  _globals['_METADATA_TAGSENTRY']._serialized_start=130
  _globals['_METADATA_TAGSENTRY']._serialized_end=173
  _globals['_EXPERTSLICE']._serialized_start=175
  _globals['_EXPERTSLICE']._serialized_end=292
  _globals['_NODE']._serialized_start=294
  _globals['_NODE']._serialized_end=385
  _globals['_SLICEAFFINITY']._serialized_start=387
  _globals['_SLICEAFFINITY']._serialized_end=437
  _globals['_SCHEDULEPLAN']._serialized_start=440
  _globals['_SCHEDULEPLAN']._serialized_end=582
# @@protoc_insertion_point(module_scope)

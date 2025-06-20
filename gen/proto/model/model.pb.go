// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.36.6
// 	protoc        v5.28.2
// source: proto/model/model.proto

package model

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
	unsafe "unsafe"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type BertRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Text          string                 `protobuf:"bytes,1,opt,name=text,proto3" json:"text,omitempty"`
	ModelName     string                 `protobuf:"bytes,2,opt,name=modelName,proto3" json:"modelName,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *BertRequest) Reset() {
	*x = BertRequest{}
	mi := &file_proto_model_model_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *BertRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BertRequest) ProtoMessage() {}

func (x *BertRequest) ProtoReflect() protoreflect.Message {
	mi := &file_proto_model_model_proto_msgTypes[0]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use BertRequest.ProtoReflect.Descriptor instead.
func (*BertRequest) Descriptor() ([]byte, []int) {
	return file_proto_model_model_proto_rawDescGZIP(), []int{0}
}

func (x *BertRequest) GetText() string {
	if x != nil {
		return x.Text
	}
	return ""
}

func (x *BertRequest) GetModelName() string {
	if x != nil {
		return x.ModelName
	}
	return ""
}

type BertResponse struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Result        string                 `protobuf:"bytes,1,opt,name=result,proto3" json:"result,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *BertResponse) Reset() {
	*x = BertResponse{}
	mi := &file_proto_model_model_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *BertResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BertResponse) ProtoMessage() {}

func (x *BertResponse) ProtoReflect() protoreflect.Message {
	mi := &file_proto_model_model_proto_msgTypes[1]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use BertResponse.ProtoReflect.Descriptor instead.
func (*BertResponse) Descriptor() ([]byte, []int) {
	return file_proto_model_model_proto_rawDescGZIP(), []int{1}
}

func (x *BertResponse) GetResult() string {
	if x != nil {
		return x.Result
	}
	return ""
}

var File_proto_model_model_proto protoreflect.FileDescriptor

const file_proto_model_model_proto_rawDesc = "" +
	"\n" +
	"\x17proto/model/model.proto\x12\x05model\"?\n" +
	"\vBertRequest\x12\x12\n" +
	"\x04text\x18\x01 \x01(\tR\x04text\x12\x1c\n" +
	"\tmodelName\x18\x02 \x01(\tR\tmodelName\"&\n" +
	"\fBertResponse\x12\x16\n" +
	"\x06result\x18\x01 \x01(\tR\x06result2A\n" +
	"\vBertService\x122\n" +
	"\aPredict\x12\x12.model.BertRequest\x1a\x13.model.BertResponseB\x0fZ\r./proto/modelb\x06proto3"

var (
	file_proto_model_model_proto_rawDescOnce sync.Once
	file_proto_model_model_proto_rawDescData []byte
)

func file_proto_model_model_proto_rawDescGZIP() []byte {
	file_proto_model_model_proto_rawDescOnce.Do(func() {
		file_proto_model_model_proto_rawDescData = protoimpl.X.CompressGZIP(unsafe.Slice(unsafe.StringData(file_proto_model_model_proto_rawDesc), len(file_proto_model_model_proto_rawDesc)))
	})
	return file_proto_model_model_proto_rawDescData
}

var file_proto_model_model_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_proto_model_model_proto_goTypes = []any{
	(*BertRequest)(nil),  // 0: model.BertRequest
	(*BertResponse)(nil), // 1: model.BertResponse
}
var file_proto_model_model_proto_depIdxs = []int32{
	0, // 0: model.BertService.Predict:input_type -> model.BertRequest
	1, // 1: model.BertService.Predict:output_type -> model.BertResponse
	1, // [1:2] is the sub-list for method output_type
	0, // [0:1] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_proto_model_model_proto_init() }
func file_proto_model_model_proto_init() {
	if File_proto_model_model_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_proto_model_model_proto_rawDesc), len(file_proto_model_model_proto_rawDesc)),
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_proto_model_model_proto_goTypes,
		DependencyIndexes: file_proto_model_model_proto_depIdxs,
		MessageInfos:      file_proto_model_model_proto_msgTypes,
	}.Build()
	File_proto_model_model_proto = out.File
	file_proto_model_model_proto_goTypes = nil
	file_proto_model_model_proto_depIdxs = nil
}

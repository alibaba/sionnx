/*
 * Copyright (C) 2017-2019 Alibaba Group Holding Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Attribute type
class ValueType<int size, bit signed, bit vec, int value, string name, int vec_size=0> {
  int size_ = size;
  bit is_signed_ = signed;
  bit is_vector_ = vec;
  bit is_string_ = 0;
  bit is_ptr_ = 0;
  int value_ = value;
  string name_ = name;
  int vec_size_ = vec_size;
}

def i1   :  ValueType<1, 0, 0, 1, "bool_">;
def i8   :  ValueType<8, 0, 0, 2, "int8">;
def i32  :  ValueType<32, 1, 0, 3, "int32">;
def f32  :  ValueType<32, 1, 0, 4, "float32">;
def i64  :  ValueType<64, 1, 0, 5, "int64">;
def f64  :  ValueType<64, 1, 0, 6, "float64">;
def f16  :  ValueType<16, 1, 0, 7, "float16">;
def i16  :  ValueType<16, 1, 0, 8, "int16">;

def ui8  :  ValueType<8, 1, 0, 12, "uint8">;
def ui32 :  ValueType<32, 0, 0, 13, "uint32">;
def ui16 :  ValueType<16, 0, 0, 14, "uint16">;
def ui64 :  ValueType<64, 0, 0, 15, "uint64">;

def char :  ValueType<8, 1, 0, 20, "char">;
def str  :  ValueType<-1, 0, 0, 21, "str"> {
  let is_string_ = 1;
}

def i32_v1  :  ValueType<32, 1, 1, 25, "int32", 1>;
def f32_v1  :  ValueType<32, 1, 1, 26, "float32", 1>;

def i1_v2   :  ValueType<1, 0, 1, 31, "bool_", 2>;
def i8_v2   :  ValueType<8, 0, 1, 32, "int8>", 2>;
def i32_v2  :  ValueType<32, 1, 1, 33, "int32", 2>;
def f32_v2  :  ValueType<32, 1, 1, 34, "float32", 2>;
def i64_v2  :  ValueType<64, 1, 1, 35, "int64", 2>;
def f64_v2  :  ValueType<64, 1, 1, 36, "float64", 2>;
def ui32_v2 :  ValueType<32, 0, 1, 42, "uint32", 2>;
def ui64_v2 :  ValueType<64, 0, 1, 45, "uint64", 2>;
def str_v2 : ValueType<-1, 0, 1, 51, "str", 2> {
  let is_string_ = 1;
}

def i1_v3   :  ValueType<1, 0, 1, 61, "bool_", 3>;
def i8_v3   :  ValueType<8, 0, 1, 62, "int8", 3>;
def i32_v3  :  ValueType<32, 1, 1, 63, "int32", 3>;
def f32_v3  :  ValueType<32, 1, 1, 64, "float32", 3>;
def i64_v3  :  ValueType<64, 1, 1, 65, "int64", 3>;
def f64_v3  :  ValueType<64, 1, 1, 66, "float64", 3>;
def ui32_v3 :  ValueType<32, 0, 1, 72, "uint32", 3>;
def ui64_v3 :  ValueType<64, 0, 1, 75, "uint64", 3>;
def str_v3 : ValueType<-1, 0, 1, 81, "str", 3> {
  let is_string_ = 1;
}

def i1_v4   :  ValueType<1, 0, 1, 91, "bool_", 4>;
def i8_v4   :  ValueType<8, 0, 1, 92, "int8", 4>;
def i32_v4  :  ValueType<32, 1, 1, 93, "int32", 4>;
def f32_v4  :  ValueType<32, 1, 1, 94, "float32", 4>;
def i64_v4  :  ValueType<64, 1, 1, 95, "int64", 4>;
def f64_v4  :  ValueType<64, 1, 1, 96, "float64", 4>;
def ui32_v4 :  ValueType<32, 0, 1, 102, "uint32", 4>;
def ui64_v4 :  ValueType<64, 0, 1, 105, "uint64", 4>;
def str_v4 : ValueType<-1, 0, 1, 111, "str", 4> {
  let is_string_ = 1;
}

def i32_v6  :  ValueType<32, 1, 1, 123, "int32", 6>;
def tensor  :  ValueType<?, ?, ?, 124, "tensor", ?>;
def complex64 : ValueType<?,?,?,125, "complex64", 0>;
def complex128 : ValueType<?,?,?,126, "complex128", 0>;

//lock related
def mutex   : ValueType<40, 0, 0, 60, "std::shared_ptr<std::mutex>">;
def cv      : ValueType<40, 0, 0, 61, "std::shared_ptr<std::condition_variable>">;
def atomici32: ValueType<40, 0, 0, 62, "std::shared_ptr<std::atomic<int>>">;

//Tensor related
def pbuff   : ValueType<40, 0, 0, 70, "std::deque<Tensor*>">;
def shape_ptr_vec: ValueType<40, 0, 0, 71, "std::vector<TensorShape*>">;
def shape_ptr: ValueType<40, 0, 0, 72, "TensorShape*">;

//Instruction related
def instruction_ptr : ValueType<50, 0, 0, 80, "Instruction*">;

//Thread Pool related
//def tp     : ValueType<40, 0, 0, 90, "ThreadPool*">;

//Enum type defined in base.h
class EnumValueType<int value, string name> : ValueType<32, 0, 0, value, name>;

def dt      : EnumValueType<101, "SN_DataType">;
def df      : EnumValueType<102, "DataFormat">;
def padding : EnumValueType<103, "Padding">;
def padmode : EnumValueType<104, "PadMode">;
def codety  : EnumValueType<105, "CodeType">;
def resize  : EnumValueType<106, "ResizeMethod">;
def t_array_map : EnumValueType<107, "std::unordered_map<std::size_t, TensorArray>*">;
def tshape  : EnumValueType<108, "TensorShape">;
def device  : EnumValueType<107, "SN_DeviceType">;
def quantization_method : EnumValueType<109, "QuantizationMethod">;

//MultiThreading related
def thread_pool_ptr: ValueType<40, 0, 0, 120, "ThreadPool*">;

//Attributes in Instruction class
class Attr<string name, list<ValueType> type, string value_default="", list<string> value_min=[], list<string> value_max=[]> {
  string name_ = name;
  list<ValueType> type_ = type;
  string value_default_ = value_default;
  list<string> value_min_ = value_min;
  list<string> value_max_ = value_max;
}

class IOType<int id, list<ValueType> types, int min_dim = 0, int max_dim = 6, int normd = 1, int axis_bound = 0,
      list<string> min_val = [], list<string> max_val = [], int optional = 0> {
  int id_ = id;
  list<ValueType> types_ = types;
  int min_dim_ = min_dim;
  int max_dim_ = max_dim;
  int normd_ = normd;
  list<string> min_val_ = min_val;
  list<string> max_val_ = max_val;
  int axis_bound_ = axis_bound;
  int optional_ = optional;
}

// Instruction supported target
class Target;

def X86Interpret     : Target;
def X86Compile       : Target;
def MKLDNN           : Target;
def ARMInterpret     : Target;
def ARMCompile       : Target;
def NVGPUCompile     : Target;
def NVGPUInterpret   : Target;
def TensorRT         : Target;
def ALI_NPU          : Target;
def FPGA             : Target;
def FPGA_DLP         : Target;
def All              : Target;

// Custom function string
class Func<string sig, bit template=0, Target t=X86Interpret> {
 string sig_ = sig;
 bit is_template_ = template;
 Target target_ = t;
}
// Custom template function, e.g., template<typename T>
class TFunc<string sig, Target t=X86Interpret> : Func<sig, 1, t>;


// Instruction property
class Property;
def InferShape      : Property;
def broadcasting    : Property;
def nonzero         : Property;

// Instruction category
// Name: enum variable in OpCategory
class Category<string name> {
  string name_ = name;
}

// Instruction opcode
// Name: enum variable in OpCode
class OpCode<string name> {
  string name_ = name;
}

//Instruction class
class Instruction<list<OpCode> ops, list<Attr> attrs, list<IOType> in_types = [], list<IOType> out_types = [], list<Property> properties = []> {
  list<OpCode> ops_ = ops;
  list<Attr> attrs_ = attrs;
  //TODO: remove it, class name is deduced by record's name
  string class_name_ = "";
  bit arm_use_common_implementation = 1; // 1: forward the call to X86CPUInterpret
  int output_num_ = 1;
  int input_num_ = 1;
  list<IOType> in_types_ = in_types;
  list<IOType> out_types_ = out_types;
  list<Property> properties_ = properties;
}

def cat_activation :         Category<"OpCategoryActivation">;
def cat_array :              Category<"OpCategoryArray">;
def cat_norm:                Category<"OpCategoryNormalization">;
def cat_unary_reduction:     Category<"OpCategoryUnaryReduction">;
def cat_unary:               Category<"OpCategoryUnaryNoReduction">;
def cat_binary:              Category<"OpCategoryBinary">;
def cat_nn:                  Category<"OpCategoryNeuralNetwork">;
def cat_conv:                Category<"OpCategoryConvolutional">;
def cat_transform:           Category<"OpCategoryTransform">;
def cat_tensor:              Category<"OpCategoryTensor">;
def cat_control:             Category<"OpCategoryControlFlow">;
def cat_regional:            Category<"OpCategoryRegional">;
def cat_fusion:              Category<"OpCategoryFusion">;
def cat_quantization:        Category<"OpCategoryQuantization">;
def cat_sparse:              Category<"OpCategorySparse">;
def cat_sparse_seg_reduction:Category<"OpCategorySparseSegmentReduction">;
def cat_string:              Category<"OpCategoryString">;
def cat_pooling:             Category<"OpCategoryPooling">;
def cat_image:               Category<"OpCategoryImageProcess">;
def cat_custom_call:         Category<"OpCategoryCustomCall">;
def cat_none:                Category<"OpCategoryNone">;
def cat_tensorarray:         Category<"OpCategoryTensorArray">;
def cat_data_preprocess:     Category<"OpCategoryDataPreprocess">;


//Define opcode
// Binary
def op_add		: OpCode<"OpAdd">;
def op_sub		: OpCode<"OpSub">;
def op_mul		: OpCode<"OpMul">;
def op_div		: OpCode<"OpDiv">;
def op_maximum		: OpCode<"OpMaximum">;
def op_minimum		: OpCode<"OpMinimum">;
def op_shift		: OpCode<"OpShift">;
def op_pow		: OpCode<"OpPow">;
def op_dot		: OpCode<"OpDot">;
def op_and		: OpCode<"OpAnd">;
def op_or		: OpCode<"OpOr">;
def op_not		: OpCode<"OpNot">;
def op_xor		: OpCode<"OpXor">;
def op_greater		: OpCode<"OpGreater">;
def op_gt		: OpCode<"OpGt">;
def op_lt		: OpCode<"OpLt">;
def op_less		: OpCode<"OpLess">;
def op_equal		: OpCode<"OpEqual">;
def op_ne		: OpCode<"OpNe">;
def op_real_div		: OpCode<"OpRealDiv">;
def op_mat_mul		: OpCode<"OpMatMul">;
def op_quantized_mat_mul : OpCode<"OpQuantizedMatMul">;
def op_inner_product		: OpCode<"OpInnerProduct">;

// Unary with reduction
def op_sum		: OpCode<"OpSum">;
def op_prod		: OpCode<"OpProd">;
def op_mean		: OpCode<"OpMean">;
def op_max		: OpCode<"OpMax">;
def op_min		: OpCode<"OpMin">;
def op_all		: OpCode<"OpAll">;
def op_any		: OpCode<"OpAny">;
def op_arg_max		: OpCode<"OpArgMax">;
def op_arg_min		: OpCode<"OpArgMin">;
def op_l1               : OpCode<"OpL1">;
def op_l2               : OpCode<"OpL2">;
def op_log_sum          : OpCode<"OpLogSum">;
def op_log_sum_exp      : OpCode<"OpLogSumExp">;
def op_sum_square       : OpCode<"OpSumSquare">;

// Unary
def op_erf              : OpCode<"OpErf">;
def op_neg		: OpCode<"OpNeg">;
def op_abs		: OpCode<"OpAbs">;
def op_sign		: OpCode<"OpSign">;
def op_exp		: OpCode<"OpExp">;
def op_log		: OpCode<"OpLog">;
def op_tanh		: OpCode<"OpTanh">;
def op_rcp		: OpCode<"OpReciprocal">;
def op_sqr		: OpCode<"OpSqr">;
def op_rsqrt		: OpCode<"OpRsqrt">;
def op_sqrt		: OpCode<"OpSqrt">;
def op_rsq		: OpCode<"OpRsq">;
def op_ceil             : OpCode<"OpCeil">;
def op_floor            : OpCode<"OpFloor">;
def op_clip             : OpCode<"OpClip">;
def op_acos             : OpCode<"OpAcos">;
def op_asin             : OpCode<"OpAsin">;
def op_atan             : OpCode<"OpAtan">;
def op_sin              : OpCode<"OpSin">;
def op_tan              : OpCode<"OpTan">;
def op_cos              : OpCode<"OpCos">;
def op_sinh             : OpCode<"OpSinh">;
def op_cosh             : OpCode<"OpCosh">;
def op_asinh            : OpCode<"OpAsinh">;
def op_acosh            : OpCode<"OpAcosh">;
def op_atanh            : OpCode<"OpAtanh">;

// Convolution
def op_conv		: OpCode<"OpConv">;
def op_conv2d		: OpCode<"OpConv2D">;
def op_conv2d_filter_grad: OpCode<"OpConv2DFilterGrad">;
def op_conv2d_input_grad: OpCode<"OpConv2DInputGrad">;
def op_conv3d		: OpCode<"OpConv3D">;
def op_deconv		: OpCode<"OpDeconv">;
def op_depthwise_conv2d : OpCode<"OpDepthwiseConv2D">;
def op_quantized_conv2d : OpCode<"OpQuantizedConv2D">;
def op_quantized_deconv : OpCode<"OpQuantizedDeconv">;

// Transformation
def op_transpose	: OpCode<"OpTranspose">;
def op_InvertPermutation : OpCode<"OpInvertPermutation">;
def op_concat		: OpCode<"OpConcat">;
def op_slice		: OpCode<"OpSlice">;
def op_strided_slice	: OpCode<"OpStridedSlice">;
def op_reverse          : OpCode<"OpReverse">;

// Tensor
def op_copy		: OpCode<"OpCopy">;
def op_gemm		: OpCode<"OpGemm">;
def op_topk		: OpCode<"OpTopK">;
def op_reduce		: OpCode<"OpReduce">;
def op_select		: OpCode<"OpSelect">;
def op_random_uniform	: OpCode<"OpRandomUniform">;
def op_truncated_normal	: OpCode<"OpTruncatedNormal">;
def op_range		: OpCode<"OpRange">;
def op_cast		: OpCode<"OpCast">;
def op_fill             : OpCode<"OpFill">;
def op_embed		: OpCode<"OpEmbed">;

// Placeholder for execution
def op_variable		: OpCode<"OpVariable">;
def op_read_variable		: OpCode<"OpReadVariable">;
def op_no_op		: OpCode<"OpNoOp">;
def op_placeholder	: OpCode<"OpPlaceholder">;
def op_none		: OpCode<"OpNone">;
def op_assign		: OpCode<"OpAssign">;
def op_assign_add : OpCode<"OpAssignAdd">;
def op_assign_sub : OpCode<"OpAssignSub">;
def op_constant		: OpCode<"OpConstant">;
def op_apply_momentum : OpCode<"OpApplyMomentum">;
def op_constant_like    : OpCode<"OpConstantLike">;

// Activatation
def op_softmax		: OpCode<"OpSoftmax">;
def op_hardmax          : OpCode<"OpHardmax">;
def op_softplus         : OpCode<"OpSoftplus">;
def op_softsign         : OpCode<"OpSoftsign">;
def op_logsoftmax       : OpCode<"OpLogSoftmax">;
def op_logsoftmaxcrossentropy       : OpCode<"OpLogSoftmaxCrossEntropy">;
def op_relu		: OpCode<"OpRelu">;
def op_sigmoid		: OpCode<"OpSigmoid">;
def op_hard_sigmoid     : OpCode<"OpHardSigmoid">;
def op_dropout		: OpCode<"OpDropout">;
def op_relu6            : OpCode<"OpRelu6">;
def op_leaky_relu       : OpCode<"OpLeakyRelu">;
def op_relu_grad		: OpCode<"OpReluGrad">;
def op_selu             : OpCode<"OpSelu">;
def op_elu              : OpCode<"OpElu">;
def op_prelu            : OpCode<"OpPRelu">;

// Neural network
def op_bias_add		: OpCode<"OpBiasAdd">;
def op_bias_add_grad		: OpCode<"OpBiasAddGrad">;
def op_lrn		: OpCode<"OpLRN">;
def op_batch_norm	: OpCode<"OpBatchNormalization">;
def op_fused_batch_norm	: OpCode<"OpFusedBatchNorm">;
def op_fused_batch_norm_grad : OpCode<"OpFusedBatchNormGrad">;
def op_scale     	: OpCode<"OpScale">;
def op_quantized_scale      	: OpCode<"QuantizedScaleOp">;
def op_proposal		: OpCode<"OpProposal">;
def op_max_pooling	: OpCode<"OpMaxPooling">;
def op_avg_pool 	: OpCode<"OpAveragePool">;
def op_global_avg_pool  : OpCode<"OpGlobalAveragePool">;
def op_global_max_pool  : OpCode<"OpGlobalMaxPool">;
def op_lp_pooling       : OpCode<"OpLpPooling">;
def op_max_pooling_grad	: OpCode<"OpMaxPoolingGrad">;
def op_avg_pooling_grad	: OpCode<"OpAvgPoolingGrad">;
def op_roipooling	: OpCode<"OpRoipooling">;
def op_max_pooling3d    : OpCode<"OpMaxPooling3D">;
def op_avg_pooling3d    : OpCode<"OpAvgPooling3D">;
def op_max_unpool       : OpCode<"OpMaxUnpool">;
def op_max_roi_pooling  : OpCode<"OpMaxRoiPooling">;
def op_quantized_max_pooling : OpCode<"OpQuantizedMaxPooling">;
def op_quantized_avg_pooling : OpCode<"OpQuantizedAvgPooling">;


// Tensorflow extension
def op_broadcast	: OpCode<"OpBroadcast">;
def op_parameter        : OpCode<"OpParameter">;
def op_tuple            : OpCode<"OpTuple">;

// Array
def op_rank		: OpCode<"OpRank">;
def op_reshape		: OpCode<"OpReshape">;
def op_lstm		: OpCode<"OpLstm">;
def op_bucketize	: OpCode<"OpBucketize">;
def op_fusion		: OpCode<"OpFusion">;
def op_expand           : OpCode<"OpExpand">;
def op_expand_dims	: OpCode<"OpExpandDims">;
def op_gather		: OpCode<"OpGather">;
def op_gathernd         : OpCode<"OpGatherNd">;
def op_identity		: OpCode<"OpIdentity">;
def op_one_hot		: OpCode<"OpOneHot">;
def op_shape		: OpCode<"OpShape">;
def op_shapeN		: OpCode<"OpShapeN">;
def op_stack		: OpCode<"OpStack">;
def op_unpack		: OpCode<"OpUnpack">;
def op_tile		: OpCode<"OpTile">;
def op_unique		: OpCode<"OpUnique">;
def op_where		: OpCode<"OpWhere">;
def op_ones_like	: OpCode<"OpOnesLike">;
def op_zeros_like	: OpCode<"OpZerosLike">;
def op_squeeze		: OpCode<"OpSqueeze">;
def op_pad		: OpCode<"OpPad">;
def op_split		: OpCode<"OpSplit">;
def op_add_n		: OpCode<"OpAddN">;
def op_size             : OpCode<"OpSize">;
def op_eyelike          : OpCode<"OpEyeLike">;
def op_flatten          : OpCode<"OpFlatten">;
def op_depth_to_space   : OpCode<"OpDepthToSpace">;
def op_space_to_depth   : OpCode<"OpSpaceToDepth">;
def op_compress         : OpCode<"OpCompress">;

// Sparse
def op_sparse_to_dense	: OpCode<"OpSparseToDense">;
def op_sparse_reshape	: OpCode<"OpSparseReshape">;
def op_sparse_fill_empty_rows	: OpCode<"OpSparseFillEmptyRows">;
def op_sparse_cross		: OpCode<"OpSparseCross">;
def op_sparse_segment_sum	: OpCode<"OpSparseSegmentSum">;
def op_sparse_segment_mean	: OpCode<"OpSparseSegmentMean">;

// String
def op_string_to_hash_bucket_fast	: OpCode<"OpStringToHashBucketFast">;

// Image process
def op_resize_bilinear		: OpCode<"OpResizeBilinear">;
def op_prior_box		: OpCode<"OpPriorBox">;
def op_non_max_suppression	: OpCode<"OpNonMaxSuppression">;
def op_detection_output		: OpCode<"OpDetectionOutput">;
def op_crop_and_resize          : OpCode<"OpCropAndResize">;

// Custom
def op_custom_call		: OpCode<"OpCustomCall">;
def op_rpn_and_roi_align		: OpCode<"OpRpnAndRoiAlign">;
def op_frcnn_post_process		: OpCode<"OpFRCNNPostProcess">;

// Control flow
def op_while		        : OpCode<"OpWhile">;
def op_enter                    : OpCode<"OpEnter">;
def op_merge                    : OpCode<"OpMerge">;
def op_switch                   : OpCode<"OpSwitch">;
def op_loop_cond                : OpCode<"OpLoopCond">;
def op_exit                     : OpCode<"OpExit">;
def op_next_iteration           : OpCode<"OpNextIteration">;

// Tensor array
def op_tensor_array             : OpCode<"OpTensorArray">;
def op_tensor_array_read        : OpCode<"OpTensorArrayRead">;
def op_tensor_array_write       : OpCode<"OpTensorArrayWrite">;
def op_tensor_array_scatter     : OpCode<"OpTensorArrayScatter">;
def op_tensor_array_gather      : OpCode<"OpTensorArrayGather">;
def op_tensor_array_size        : OpCode<"OpTensorArraySize">;

// data preprocess
def op_one_shot_iterator        : OpCode<"OpOneShotIterator">;
def op_iterator_get_next        : OpCode<"OpIteratorGetNext">;

// Quantize inst
def op_quantization             : OpCode<"OpQuantization">;
def op_dequantization           : OpCode<"OpDeQuantization">;

// Data Transfer
def op_transfer                 : OpCode<"OpTransfer">;

// FPGA_DLP
def op_dlp_fused_conv2d         : OpCode<"OpDLPFusedConv2D">;
def op_dlp_fused_conv3d         : OpCode<"OpDLPFusedConv3D">;
def op_dlp_fused_dw_conv2d      : OpCode<"OpDLPFusedDepthWiseConv2D">;
def op_dlp_fc                   : OpCode<"OpDLPFC">;
def op_dlp_deconv               : OpCode<"OpDLPDeconv">;

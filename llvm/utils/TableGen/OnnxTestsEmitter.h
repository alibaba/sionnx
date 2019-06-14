//===- OnnxTestsEmiiter.h - Helper for the Onnx Tests Emiiter--------------===//
//
// Copyright (C) 2017-2019 Alibaba Group Holding Limited
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_ONNX_TESTS_H
#define LLVM_UTILS_TABLEGEN_ONNX_TESTS_H

#include <cstdint>
#include <stdio.h>
// #define WINDOWS
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#define INVALID_AXIS 11

using namespace std;
using namespace llvm;

namespace onnx {

/*! 
 * Get current working directory.
 */
std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}

/*!
 * Enum of Onnx Data Types
 */
enum OnnxDataType {
    UNDEFINED = 0,
    // Basic types.
    FLOAT = 1,   // float
    UINT8 = 2,   // uint8_t
    INT8 = 3,    // int8_t
    UINT16 = 4,  // uint16_t
    INT16 = 5,   // int16_t
    INT32 = 6,   // int32_t
    INT64 = 7,   // int64_t
    STRING = 8,  // string
    BOOL = 9,    // bool

    // IEEE754 half-precision floating-point format (16 bits wide).
    // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10,

    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,     // complex with float32 real and imaginary components
    COMPLEX128 = 15,    // complex with float64 real and imaginary components

    // Non-IEEE floating-point format based on IEEE754 single-precision
    // floating-point number truncated to 16 bits.
    // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16,
    ONNX_TYPE_COUNT,
    // Future extensions go here.
};    

/*!
 * String of Onnx types, indexed by OnnxDataType enum.
 */
static const std::string onnx_type_names[ONNX_TYPE_COUNT] = {
  "onnx.TensorProto.UNDEFINED",
  "onnx.TensorProto.FLOAT",
  "onnx.TensorProto.UINT8",
  "onnx.TensorProto.INT8",
  "onnx.TensorProto.UINT16",
  "onnx.TensorProto.INT16",
  "onnx.TensorProto.INT32",
  "onnx.TensorProto.INT64",
  "onnx.TensorProto.STRING",
  "onnx.TensorProto.BOOL",
  "onnx.TensorProto.FLOAT16",
  "onnx.TensorProto.DOUBLE",
  "onnx.TensorProto.UINT32",
  "onnx.TensorProto.UINT64",
  "onnx.TensorProto.COMPLEX64",
  "onnx.TensorProto.COMPLEX128"
};

/*!
 * String of numpy types, indexed by OnnxDataType enum.
 */
static const std::string np_type_names[ONNX_TYPE_COUNT] = {
  "INVALID",
  "np.float32",
  "np.uint8",
  "np.int8",
  "np.uint16",
  "np.int16",
  "np.int32",
  "np.int64",
  "np.string_",
  "np.bool_",
  "np.float16",
  "np.float64",
  "np.uint32",
  "np.uint64",
  "np.complex64",
  "np.complex128"
};

/*!
 * A struct for pooling ops' attributes
 */
struct PoolAttr {
  std::vector<int> strides;
  std::vector<int> pads;
  std::vector<int> kernels;
  bool init = false;
};

/*!
 * A struct to store operand's properties and attributes.
 */
struct OperandAttr {
  string name = "";
  bool handle_broadcast = false;
  int i = 0; //operand index
  PoolAttr pool_attr;
  bool isBatchNorm = false;
  bool isCompress = false;
  bool isConcat = false;
  bool isDepthToSpace = false;
  int axis = INVALID_AXIS;
  std::vector<string> attr_vals;
};

/*!
 * Reshape types for OpReshape
 */
enum RESHAPE_TYPE {
  REORDER = 1,
  REDUCE = 2,
  EXTEND = 3,
  ONE_DIM = 4,
  NEGATIVE_DIM = 5,
};

/*!
 * Helper function to reshape a tensor.
 */
vector<int> reshape(vector<int>& shape, RESHAPE_TYPE type) {
  vector<int> ret;
  switch (type) {
    case REORDER:
      ret = shape;
      reverse(ret.begin(), ret.end());
      break;

    case REDUCE: {
      if (shape.size() < 2) {
        return shape;
      }
      ret.push_back(shape[0] * shape[1]);
      for (unsigned i = 2; i < shape.size(); ++i) {
        ret.push_back(shape[i]);
      }
      }
      break;

    case EXTEND: {
      //find a extendable dim
      for (unsigned i = 0; i < shape.size(); ++i) {
        if (shape[i]%2 == 0) {
          ret.push_back(2);
          ret.push_back(shape[i]/2);
        } else if (shape[i]%3 == 0) {
          ret.push_back(3);
          ret.push_back(shape[i]/3);
        } else if (shape[i]%5 == 0) {
          ret.push_back(5);
          ret.push_back(shape[i]/5);
        } else if (shape[i]%7 == 0) {
          ret.push_back(7);
          ret.push_back(shape[i]/7);
        } else {
          ret.push_back(shape[i]);
        }
      }
      if (ret.size() == shape.size()) {
        ret.push_back(1);
      }
      }
      break;

    case ONE_DIM: {
      int size = 1;
      for (unsigned i = 0; i < shape.size(); ++i) {
        size *= shape[i];
      }
      ret.push_back(size);
      }
      break;

    case NEGATIVE_DIM: {
      if (shape.size() < 2) {
        ret = shape;
        ret.push_back(-1);
      } else {
        ret.push_back(shape[0]);
        for (unsigned i = 2; i < shape.size(); ++i) {
          ret.push_back(shape[i]);
        }
        ret.push_back(-1);
      }
      }
      break;

    default: break;
  }
  return ret;
}

/*!
 * Numpy random function selector based on data type.
 */
static StringRef GetRandomFn(StringRef type, bool& is_int, bool& is_unsign) {
  switch (type[0]) {
    case 'f':
      is_int = false;
      return "np.random.uniform"; //Return random floats in the half-open interval [0.0, 1.0).

    case 'u':
      is_unsign = true;
    case 'i':
      is_int = true;
      return "np.random.randint"; //Return random integers from low (inclusive) to high (exclusive).

    case 's':
      return "np.random.bytes";

    default:
      break;
  }
  return "";
}

/*!
 * Helper function to decode element types.
 */
static StringRef GetOnnxElementType(StringRef np_typeName) {
  StringRef type_name = np_typeName;
  std::size_t found = np_typeName.find(".");
  if (found!=std::string::npos) {
    type_name = np_typeName.substr(found+1);
  }

  assert(type_name.size());
  switch (type_name[0]) {
    case 'f':
      if (!type_name.compare("float32")) {
        return "FLOAT";
      } else if (!type_name.compare("float64")) {
        return "DOUBLE";
      } else if (!type_name.compare("float16")) {
        return "FLOAT";
      }
      break;

    case 'i':
      if (!type_name.compare("int32")) {
        return "INT32";
      } else if (!type_name.compare("int64")) {
        return "INT64";
      } else if (!type_name.compare("int8")) {
        return "INT8";
      } else if(!type_name.compare("int16")) {
        return "INT16";
      }
      break;

    case 'b':
      return "BOOL";

    case 'u':
      if (!type_name.compare("uint32")) {
        return "UINT32";
      } else if (!type_name.compare("uint64")) {
        return "UINT64";
      } else if (!type_name.compare("uint8")) {
        return "UINT8";
      } else if(!type_name.compare("uint16")) {
        return "UINT16";
      }
      break;

    case 'o':
      return "STRING";

    case 'c':
      if (!type_name.compare("complex64")) {
        return "COMPLEX64";
      } else if (!type_name.compare("complex128")) {
        return "COMPLEX128";
      }
      break;

    default:
      break;
  }
  return "";
}

/*!
 * Decode td record to numpy types.
 */
static StringRef GetNPType(StringRef typeName, int& dim, bool& isInt) {
  dim = 0;
  isInt = false;
  StringRef type_name = typeName;

  std::size_t found = typeName.find("_");
  if (found!=std::string::npos) {
    dim = 1;
    type_name = typeName.substr(0, found);
  }

  assert(type_name.size());
  switch (type_name[0]) {
    case 'f':
      if (!type_name.compare("f32")) {
        return "float32";
      } else if (!type_name.compare("f64")) {
        return "float64";
      } else if (!type_name.compare("f16")) {
        return "float16";
      }
      break;

    case 'i':
      isInt = true;
      if (!type_name.compare("i32")) {
        return "int32";
      } else if (!type_name.compare("i64")) {
        return "int64";
      } else if (!type_name.compare("i8")) {
        return "int8";
      } else if(!type_name.compare("i16")) {
        return "int16";
      } else if(!type_name.compare("i1")) {
        return "bool_";
      }
      break;

    case 'u':
      if (!type_name.compare("ui32")) {
        return "uint32";
      } else if (!type_name.compare("ui64")) {
        return "uint64";
      } else if (!type_name.compare("ui8")) {
        return "uint8";
      } else if(!type_name.compare("ui16")) {
        return "uint16";
      }
      break;

    case 's':
      if (!type_name.compare("str")) {
        return "object";
      }
      break;

    case 'c':
      if (!type_name.compare("complex64")) {
        return "complex64";
      } else if (!type_name.compare("complex128")) {
        return "complex128";
      }
      break;

    default:
      break;
  }
  return "";
}

/*!
 * Pre-defined maximum size in one dimension.
 * Ususally the max size in one dimension decreases as dimension size increases.
 */
int GetMaxSizeInOneDim(int dim_size) {
  int max_size = 200;
  if (dim_size > 2)
    max_size = 40;
  if (dim_size > 3)
    max_size = 25;
  if (dim_size > 4)
    max_size = 10;
  return max_size;
}

/*!
 * Helper function to generate a random integer, in range [min, max).
 * if exclusive_list is not nullptr, intergers occurred in exclusive_list will be discarded.
 */
int GetRandomInt(int min = 0, int max = 0, vector<int>* exclusive_list=nullptr) {
  int ret;
  if (exclusive_list) {
    do {
      ret =  min + ( std::rand() % ( max - min + 1 ) );
    } while (find(exclusive_list->begin(), exclusive_list->end(), ret) != exclusive_list->end());
  } else {
    ret =  min + ( std::rand() % ( max - min + 1 ) );
  }
  return ret;
}

/*!
 * Helper function to generate a random string from a string list.
 */
string GetRandomString(std::string& s) {
  std::vector<std::string> strs;
  size_t pos = 0;
  while ((pos = s.find(",")) != std::string::npos) {
    std::string item = s.substr(0, pos);
    strs.push_back(item);
    s.erase(0, pos + 1);
  }
  strs.push_back(s);
  int id = GetRandomInt(0, strs.size()-1);
  return "'" + strs[id] + "'";
}

/*!
 * Helper function to generate a random boolean value(true or false).
 */
int GetRandomBool() {
  int val = GetRandomInt(0, 10);
  return (val > 5) ? 1 : 0;
}

/*!
 * Helper function to generate a random float value, in range [min, max).
 */
float GetRandomFloat(float min = 0, float max = 0) {
  return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}

/*!
 * Defines some magic numbers to control the max/min value generated by random functions.
 */
struct MIN_MAX_LIMIT {
  static constexpr int val_min_i = -128;
  static constexpr int val_max_i = 127;
  static constexpr float val_min_f = -500;
  static constexpr float val_max_f = 500;
};

/*!
 * A class to decode llvm table-gen files and emit onnx tests.
 */
class CodeEmitterGen {
  RecordKeeper &Records;

 public:
  CodeEmitterGen(RecordKeeper &record) : Records(record) { }
  void gen_onnx_test(raw_ostream &o, bool smoke = false);
  void EmitLicense(raw_ostream &o);
  void EmitPrefix(raw_ostream &o);
  void EmitOneClassOnnx(Record* oneclass, raw_ostream &o, bool smoke = false);
  void EmitOneTypeOnnx(Record* oneclass, raw_ostream &o, std::string& name, std::vector<StringRef>& in_type);

 private:
  int test_count_ = 0;
};

}

#endif

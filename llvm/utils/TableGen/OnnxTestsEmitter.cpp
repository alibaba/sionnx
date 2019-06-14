//===- OnnxInstrEmitterGen.cpp - Conformance Tests Generator for Onnx -----===//
//
// Copyright (C) 2017-2019 Alibaba Group Holding Limited
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "OnnxTestsEmitter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
#include <vector>
#include <fstream>
#include<iostream>

using namespace llvm;
using namespace std;
using namespace onnx;

namespace onnx {
/*!
 * Controls data dimension generated in tests. All the supported dimensions should
 * be tests inside a single test file.
 */  
static vector<int> inputs_dim_mask;

/*!
 * Main function to generate a single test file.
 * Each test file may contain multiple tests.
 */
void CodeEmitterGen::gen_onnx_test(raw_ostream &o, bool smoke) {
  EmitLicense(o);
  EmitPrefix(o);
  std::vector<Record *> Insts = Records.getAllDerivedDefinitions("Instruction");
  std::srand (static_cast <unsigned> (time(0)));
  for (std::vector<Record *>::iterator IC = Insts.begin(), EC = Insts.end();
       IC != EC; ++IC) {
    Record *R = *IC;
    //Emit tests for one op code.
    EmitOneClassOnnx(R, o, smoke);
  }

  return;
}

/*!
 * Emit license information of current test file.
 */
void CodeEmitterGen::EmitLicense(raw_ostream &o) {
  o << "#/*\n";
  o << "# * Copyright (C) 2017-2019 Alibaba Group Holding Limited\n";
  o << "# *\n";
  o << "# * Licensed under the Apache License, Version 2.0 (the \"License\");\n";
  o << "# * you may not use this file except in compliance with the License.\n";
  o << "# * You may obtain a copy of the License at\n";
  o << "# *\n";
  o << "# *      http://www.apache.org/licenses/LICENSE-2.0\n";
  o << "# *\n";
  o << "# * Unless required by applicable law or agreed to in writing, software\n";
  o << "# * distributed under the License is distributed on an \"AS IS\" BASIS,\n";
  o << "# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n";
  o << "# * See the License for the specific language governing permissions and\n";
  o << "# * limitations under the License.\n";
  o << "# */\n\n";
}

/*!
 * Prefix comments including importing necessary modules.
 */
void CodeEmitterGen::EmitPrefix(raw_ostream &o) {
  o << "\n";
  o << "#/===- TableGen'erated file -----------------------------------------------===\\\n";
  o << "#|                                                                            |\n";
  o << "#| Onnx Conformance Tests Source Fragment                                     |\n";
  o << "#|                                                                            |\n";
  o << "#| Automatically generated file, do not edit!                                 |\n";
  o << "#|                                                                            |\n";
  o << "#\\===----------------------------------------------------------------------===/\n";
  o << "\n";
  o << "from __future__ import absolute_import\n";
  o << "from __future__ import division\n";
  o << "from __future__ import print_function\n";
  o << "from __future__ import unicode_literals\n";

  o << "import numpy as np\n";
  o << "import math\n";
  o << "import onnx\n";
  o << "from ..base import Base\n";
  o << "from . import expect\n";  
  o << "\n";
}

namespace {
/* !
 * Helper function to select a dimension to test.
 * All the dimensions supported should be tested in a single file.
 */
int GetDimension(int input_index, int min = 0, int max = 0) {
  int dim;
  // A mask representing all the valid dimensions to be tested.
  // For example, the mask will be Ox3F if dimension within [0, 5] need to be tested.
  int full_dim_mask = ((1 << min) - 1) ^ ((1 << (1+max)) - 1);
  if ((inputs_dim_mask[input_index] & full_dim_mask) != full_dim_mask) {
    //some dims are not tested yet
    /*!
     * If dim is generated before, get a new one unless
     * all possible dims are generated.
     */
    do {
      dim = GetRandomInt(min, max);
    } while ((1 << dim) & inputs_dim_mask[input_index]);
  } else {
    if (inputs_dim_mask[input_index] == full_dim_mask) {
      //clear the inputs_dim_mask for better uniform dim distribution
      inputs_dim_mask[input_index] = 0;
    }
    dim = GetRandomInt(min, max);
  }
  /*!
   * Update the dim mask after the dimension value is generated.
   */
  inputs_dim_mask[input_index] |= (1 << dim);
  return dim;
}
}

/*!
 * Emit one node definition for current operator.
 * Multiple tests may be generated for each node definition.
 */
void CodeEmitterGen::EmitOneClassOnnx(Record* oneclass, raw_ostream &o, bool smoke) {
    inputs_dim_mask.clear();
    std::string name = oneclass->getValueAsString("class_name_").str();
    if (name == "") {
      name = oneclass->getName();
      name.resize(name.size()-2);
    }
    assert(name.size() && "class name expected ");
    o << "print(\"Exporting Tests of " << name << "\")";

    name[0] = toupper(name[0]);
    o << "\n\n\nclass " + name + "(Base):\n\n";
    o << "    @staticmethod\n";
    o << "    def export():\n";

    //Get algorithm for op
    auto gName = name;
    unsigned int count = 0;
    while (count < gName.size()) {
      char c = gName[count];
      if (isupper(c))
        gName[count] = tolower(c);
      count++;
    }
    //Get the algorithm file
    std::ifstream infile(onnx::GetCurrentWorkingDir()+"/"+ gName + ".algorithm");
    auto list_in = oneclass->getValueAsListOfDefs("in_types_");
    auto list_out = oneclass->getValueAsListOfDefs("out_types_");
    if (infile.good()) {
      std::string line;    
      while (std::getline(infile, line)) {
	if ((line[0] == '#') && (line[1] == '*')) continue;
        o << "        " << line << "\n";;
      }
    } else {
      o << "\n\n\n ==========No Algorithm File! Tests Generation Aborted!=============\n";
      std::cout << "ERROR(" << gName << "): No algorithm file found!\n";
      return;
    }
    o << "\n";

    unsigned int type_combination_count = 0;
    for (unsigned int i = 0; i < list_in.size(); ++i) {
      unsigned int num_types = list_in[i]->getValueAsListInit("types_")->size();
      type_combination_count = num_types > type_combination_count ? num_types : type_combination_count;
    }
    if (!type_combination_count) {
      for (unsigned int i = 0; i < list_out.size(); ++i) {
        unsigned int num_types = list_out[i]->getValueAsListInit("types_")->size();
        type_combination_count = num_types > type_combination_count ? num_types : type_combination_count;
      }
    }
    std::vector<StringRef> in_types;
    int max_test_num = 200;

    std::vector<std::vector<StringRef>> candidate_types;
    std::vector<int> candidate_input_indexes;
    for (unsigned int idx = 0; idx < type_combination_count; ++idx) {
      auto list = list_in.size() ? list_in : list_out;
      for (unsigned int i = 0; i < list.size(); ++i) {
        ListInit* types_in = list[i]->getValueAsListInit("types_");
        bool inputs_match = (types_in->size() == type_combination_count);
        unsigned id = idx;
        if (!inputs_match) {
          //If the input operand's type doesn't match others, test them respectively
          if (!candidate_input_indexes.size()) {
            //Only need initialize once
            candidate_input_indexes.push_back(i);
            vector<StringRef> type_v;
            //start from index 1, save the types to be tested
            for (unsigned m = 1; m < types_in->size(); ++m) {
              StringRef t_name = types_in->getElementAsRecord(m)->getName();
              type_v.push_back(t_name);
            }
            candidate_types.push_back(type_v);
          }
          // Test the first type
          id = 0;
        }
        auto r = types_in->getElementAsRecord(id);
        StringRef type_name = r->getName();
        int dim;
        bool isInt;
        StringRef np_type = GetNPType(type_name, dim, isInt);        
        in_types.push_back(np_type);
      }
      int iter_size = max_test_num/(10 * type_combination_count);
      while (iter_size--) {
        EmitOneTypeOnnx(oneclass, o, name, in_types);
      }
     
      // For smoke tests, only test one type combination
      if (smoke)
        break;

      // If there is more types need to be tested, test them here
      if (candidate_types.size()) {
        for (unsigned k = 0; k < candidate_types.size(); ++k) {
          for (unsigned l = 0; l < candidate_types[k].size(); ++l) {
            int index = candidate_input_indexes[k];
            int dim;
            bool isInt;
            StringRef np_type = GetNPType(candidate_types[k][l], dim, isInt);
            in_types[index] = np_type;
            iter_size = std::max<int>(1, max_test_num/(10 * type_combination_count));
            while (iter_size--) {
              EmitOneTypeOnnx(oneclass, o, name, in_types);
            }
          }
        }
      }
      in_types.clear();
    }
    o << "###Total number of tests generated: " << test_count_;
    return;
        
}

/*!
 * Helper function to handle attributes of current operator.
 */
string handleAttributes(Record* r, int index, PoolAttr* pool_attr = nullptr) {
  StringRef TypeName = r->getValueAsListOfDefs("type_")[index]->getName();
  StringRef AttrName = r->getValueAsString("name_");
  auto val_min_s = r->getValueAsListOfStrings("value_min_");
  auto val_max_s = r->getValueAsListOfStrings("value_max_");
  int dim;
  bool isInt = false;
  bool isString = !TypeName.substr(0, 2).compare("st");
  bool isTensor = !TypeName.substr(0, 6).compare("tensor");
  bool isBool = !TypeName.substr(0, 2).compare("i1");

  int val_min_i, val_max_i = 0;
  float val_min_f, val_max_f = 0.0;
  GetNPType(TypeName, dim, isInt);

  int random_range_selector = 0;
  if (val_min_s.size()) {
    random_range_selector = GetRandomInt(0, val_min_s.size()-1);
  }
  if (isInt) {
    val_min_i = val_min_s.size() ? stoi(val_min_s[random_range_selector]) : MIN_MAX_LIMIT::val_min_i;
    val_max_i = val_max_s.size() ? stoi(val_max_s[random_range_selector]) : MIN_MAX_LIMIT::val_max_i;
  } else if (!isString && !isTensor) {
    val_min_f = val_min_s.size() ? stof(val_min_s[random_range_selector]) : MIN_MAX_LIMIT::val_min_f;
    val_max_f = val_max_s.size() ? stof(val_max_s[random_range_selector]) : MIN_MAX_LIMIT::val_max_i;
  }
  if (dim == 0) {
    if (isString) { 
      std::string val = r->getValueAsListOfStrings("value_min_")[0];
      return GetRandomString(val);
    } else if (isTensor) {
      return "value_tensor";
    }
    if (isInt) {
      return std::to_string(isBool ? GetRandomBool() : GetRandomInt(val_min_i, val_max_i));
    } else {
      return std::to_string(GetRandomFloat(val_min_f, val_max_f));
    }
  } else if (dim == 1) {
    //Handle vector attributes
    int vec_len = r->getValueAsListOfDefs("type_")[index]->getValueAsInt("vec_size_");
    //Axes vector should not contain any duplicated axis value.
    bool axes_list = !AttrName.compare("axes");
    std::string str_v = "[";
    if (isInt) {
      std::vector<int> vals;
      while (vec_len--) {
	//Generate elements with random function.
        int tmp = isBool ? GetRandomBool()
                         :  (axes_list ? GetRandomInt(val_min_i, val_max_i, &vals)
                                       : GetRandomInt(val_min_i, val_max_i));
        vals.push_back(tmp);
        str_v += std::to_string(tmp);
        if (vec_len) {
          str_v += ",";
        }
      }
      str_v += "]";
      if (pool_attr && pool_attr->init) {
	//Special handling for convolution attributes.
        if (!AttrName.compare("pads")) {
          pool_attr->pads = vals;
        } else if (!AttrName.compare("strides")) {
          pool_attr->strides = vals;
        } else if (!AttrName.compare("kernel_shape")) {
          pool_attr->kernels = vals;
        }
      }
    } else {
      while (vec_len--) {
        str_v += std::to_string(GetRandomFloat(val_min_f, val_max_f));
        if (vec_len) {
          str_v += ",";
        }
      }
      str_v += "]";
    }
    return str_v;
  } else {
    assert(false && "Attr type not supported yet.");
    return "";
  }
}

/*!
 * Helper function to generate operands for pooling ops.
 */
int GetPoolOperands(PoolAttr& pool_attr, int idx, int min_size, int max_size) {
  if (idx == 0)
    return 1; //First dimension has length 1.
  else if (idx == 1)
    return GetRandomInt(1, 10); //Second dimesion has length [1 - 10]
  else {
    //For other dimensions, the length should be no less than kernel_size + padding_size along each dimension.
    int pad_sum = pool_attr.pads[idx-2] + pool_attr.pads[pool_attr.pads.size()-idx+1];
    min_size = pool_attr.kernels[idx-2] - pad_sum;
    min_size = min_size > 0 ? min_size : 1;
    max_size = max_size > min_size ? max_size : (min_size + 5);
    return GetRandomInt(min_size, max_size);
  }  
}

/*!
 * Helper function to generate operands BatchNormalization ops.
 */
int GetBatchNormOperands(int operand_idx, int dim_idx, int min_size, int max_size,
      std::vector<std::vector<int>>& shapes) {
  if (operand_idx==0) {
    if (dim_idx == 0)
      return 1;
    else if (dim_idx == 1)
      return GetRandomInt(1, 10);
    else {
      return GetRandomInt(min_size, max_size);
    }
  } else {
    //If the first operand has shape NCD1D2...Dn,
    //all the other dimensions should have length of C.
    return shapes[0][1];
  }
}

/*!
 * Helper function to generate operands for OpCompress.
 */
int GetCompressOperands(int axis, int min_size, std::vector<std::vector<int>>& shapes) {
  //The second operand's length should be the size of the first operand's dimension [axis] 
  //axis value could be negative.
  int max_size = 1;
  if (axis == INVALID_AXIS) {
    for (unsigned i = 0; i < shapes[0].size(); ++i) {
      max_size *= shapes[0][i];
    }
  } else {
    if (axis < 0) 
      axis = shapes[0].size() + axis;
    max_size = shapes[0][axis];
  }
  return GetRandomInt(min_size, max_size);
}

/*!
 * Generate operands for current operator, given the information of dimension size,
 * value range, value shape, attributes value, and operand index value.
 */
std::vector<int> GenerateOneOperand(int dim, int max_size, 
  std::vector<std::vector<int>> shapes, OperandAttr& op_attr, int input_index) {
  std::vector<int> shape;
  for (int k = 0; k < dim; ++k) {
    int size;
    if ((op_attr.i > 0) && op_attr.handle_broadcast && (k >= ( dim - (int)shapes[0].size()))) {
      bool selector = GetRandomBool();
      size = selector ? 1 : shapes[0][k - dim + shapes[0].size()];
    } else {
      int min_size = 1;
      if (op_attr.pool_attr.init) {
	//Special handling of pooling operands.
        size = GetPoolOperands(op_attr.pool_attr, k, min_size, max_size);
      } else if (op_attr.isBatchNorm) {
	//Special handling of batchnorm operands.
        size = GetBatchNormOperands(input_index, k, min_size, max_size, shapes);
      } else if (op_attr.isCompress && (input_index == 1)) {
	//Special handling of compress operands.
        //Handle limitations on operand 1 of OpCompress
        size = GetCompressOperands(op_attr.axis, min_size, shapes);
      } else if (op_attr.isConcat && (op_attr.i > 0)) {
	//Special handling of concat operands.
        int axis_val = op_attr.axis < 0 ? op_attr.axis + shapes[0].size() : op_attr.axis;
        size = (k == axis_val) ? GetRandomInt(min_size, max_size) : shapes[0][k];
      } else if (op_attr.isDepthToSpace && (k==1)) {
	//Special handling of depth_to_space operands.
        size = stoi(op_attr.attr_vals[0]);
        size = size * size * GetRandomInt(1, 5);
      } else if (!op_attr.name.compare("Gemm") && (op_attr.i != 0)) {
	 //Special handling of gemm operands.
	 //Dimension 1 of input A and dimension 0 of input B should match
	 //after transpose if applicable. 
        int transA = stoi(op_attr.attr_vals[2]);
        int transB = stoi(op_attr.attr_vals[3]);
        if (op_attr.i == 1) {
          if (transA == transB) {
            size = (k == 0) ? shapes[0][1] : shapes[0][0];
          } else {
            size = (k == 0) ? shapes[0][0] : shapes[0][1];
          }
        } else if (op_attr.i == 2) {
          if (transA == transB) {
            if (transA) {
              size = (k == 0) ? shapes[0][1] : shapes[1][0];
            } else {
              size = (k == 0) ? shapes[0][0] : shapes[1][1];
            }
          } else if (transA) {
            size = (k == 0) ? shapes[0][1] : shapes[1][1];
          } else if (transB) {
            size = (k == 0) ? shapes[0][0] : shapes[1][0];
          }
        } 
      } else if (!op_attr.name.compare("MatMul")
                 && (op_attr.i != 0)) {
	//Special handling of MatMul operands.
        if (k >= dim - 2 && (dim > 1)) {
          size = (k == (dim - 1)) ? shapes[0][dim - 2] : shapes[0][dim - 1]; 
        } else {
          size = shapes[0][k];
        }
      } else if (!op_attr.name.compare("Conv")) {
	//Special handling of Conv operands.
        auto kernel_shape = op_attr.attr_vals[3];
        auto pos = kernel_shape.find(",");
	//Get minimum h and w value according to kernel shape.
        int h = stoi(kernel_shape.substr(1, pos));
        int w = stoi(kernel_shape.substr(pos+1, kernel_shape.size() - pos));
        if (op_attr.i == 0) {
          size = (k < 2) ? 1 : GetRandomInt((k == 2) ? h : w, max_size);
        } else if (op_attr.i == 1) {
          size = (k < 2) ? 1 : ((k == 2) ? h : w);
        } else {
          size = shapes[1][0];
        }
      } else if (!op_attr.name.compare("OneHot") && (op_attr.i == 2)) {
	//Special handling of OneHot operands.
        size = 2;
      } else if (!op_attr.name.compare("Squeeze")) {
	//Special handling of Squeeze operands.
        bool need_squeeze = (op_attr.attr_vals[0].find(","+to_string(k)+",") != string::npos) ||
                            (op_attr.attr_vals[0].find("["+to_string(k)+",") != string::npos) ||
                            (op_attr.attr_vals[0].find("["+to_string(k)+"]") != string::npos) ||
                            (op_attr.attr_vals[0].find(","+to_string(k)+"]") != string::npos);
        size = need_squeeze ? 1 : GetRandomInt(min_size, max_size);
      } else {
        if (!op_attr.name.compare("LRN")) { 
          //Special handling of LRN operands.
          min_size = stoi(op_attr.attr_vals.back());
        }
	//Otherwise, generate the input operands shape with random function.
        size = GetRandomInt(min_size, max_size);
      }
    }
    shape.push_back(size);
  }
  return shape;
}

/*!
 * Generate tests for a given data type.
 * Multiple tests may be generated for a specific data type.
 */
void CodeEmitterGen::EmitOneTypeOnnx(Record* oneclass, raw_ostream &o, std::string& name,
       std::vector<StringRef>& in_type) {
    //Reset inputs_dim_mask for each type combination.
    inputs_dim_mask.clear();

    auto list_in = oneclass->getValueAsListOfDefs("in_types_");
    auto list_out = oneclass->getValueAsListOfDefs("out_types_");
    bool no_input = (list_in.size() == 0);
    if (no_input) {
      //Some ops have no inputs, such as OpConstant.
      //In this case, we check the output info instead.
      inputs_dim_mask.push_back(0);
      int min_dim = list_out[0]->getValueAsInt("min_dim_");
      int max_dim = list_out[0]->getValueAsInt("max_dim_");
      int dim = GetDimension(0, min_dim, max_dim);
      if (dim == 0) { //scalar
        o << "        values = np." << in_type[0] << "(np.random.rand())\n";
      } else {
        o << "        values = np.random.randn(";
	//Get the max length in one dimension based on the dimension size.
	//In general, the larger the dimension is, the smaller each dimension length will be.
        int max_size = GetMaxSizeInOneDim(dim);
        for (int k = 0; k < dim; ++k) {
          int size = GetRandomInt(1, max_size);
          o << std::to_string(size);
          if (k != dim - 1) {
            o << ",";
          }
        }
        o << ").astype(np." << in_type[0] << ")\n";
      }
      //For now only OpConstant has no input.
      //Generate Constant node in this case.
      o << "        value_tensor = onnx.helper.make_tensor(\n";
      o << "            name='',\n";
      o << "            data_type=onnx.TensorProto." << GetOnnxElementType(in_type[0]) << ",\n";
      o << "            dims=values.shape,\n";
      o << "            vals=values.flatten().astype(np." << in_type[0] << ")\n";
      o << "        )\n";
    }

    //Otherwise, generate a regular node.
    o << "        node = onnx.helper.make_node(\n";
    o << "            \'" << name << "\',\n";
    o << "            inputs=[";    
    //For dynamic sized inputs, the index value is -1
    bool dynamic_size_input = list_in.size() && (list_in[0]->getValueAsInt("id_") == -1);
    //Generate a random size if the input's size is dynamic.
    unsigned int input_size = dynamic_size_input ? GetRandomInt(1, 10) : list_in.size();
    //Declare inputs.
    for (unsigned int i = 0; i < input_size; ++i) {
      inputs_dim_mask.push_back(0);
      o << "\'x_" << std::to_string(i) << "\'";
      if (i != input_size - 1) {
        o << ",";
      }
    }
    o << "],\n";
    //Declare outputs.
    o << "            outputs=[";
    for (unsigned int j = 0; j < list_out.size(); ++j) {
      o << "\'y_" << std::to_string(j) << "\'";
      if (j != list_out.size() - 1) {
        o << ",";
      }    
    }
    o << "],\n";
    //Generate Attributes
    int axis = INVALID_AXIS;
    PoolAttr pool_attr;
    pool_attr.init = !name.compare("AveragePool");
    std::vector<std::string> attr_vals;
    std::vector<Record*> Attrs = oneclass->getValueAsListOfDefs("attrs_");
    // Check if any attr has multiple types
    // Pick a valid type for current attribute randomly.
    if (Attrs.size()) {
      unsigned int type_num = 1;
      for (std::vector<Record *>::iterator IC = Attrs.begin(), EC = Attrs.end();
           IC != EC; ++IC) {
          Record *R = *IC;
          type_num = R->getValueAsListOfDefs("type_").size() > type_num 
                     ? R->getValueAsListOfDefs("type_").size() 
                     : type_num;
      }     
      unsigned int type_id = GetRandomInt(0, type_num-1); 
      for (std::vector<Record *>::iterator IC = Attrs.begin(), EC = Attrs.end();
           IC != EC; ++IC) {
          Record *R = *IC;
          StringRef AttrName = R->getValueAsString("name_");
          StringRef default_val = R->getValueAsString("value_default_");
	  //Enable default attribute value testing randomly.
          bool test_default_attr = GetRandomBool();
          if (test_default_attr && default_val.size()) {
	    //When testing the default attribute value, the test should 
            //not set the attribute value explicitly in node.
            if (R->getValueAsListOfDefs("type_")[0]->getName() == "str") {
              attr_vals.emplace_back("'"+default_val.str()+"'");
            } else if (default_val == "?") {
              //Special handling of OpEyeLike, the default value is input's types.
              attr_vals.push_back("np."+in_type[0].str());
            } else {
              attr_vals.push_back(default_val);
              if (!AttrName.compare("axis")) {
                axis = std::stoi(default_val);
              }
            }
          } else {
            //In other cases, generate explicit attribute values.
            int index = R->getValueAsListOfDefs("type_").size() > (type_id + 1)
                        ? type_id 
                        : 0;
            string val = handleAttributes(R, index, &pool_attr);
	    //Some special handling of some particular ops.
            if (!name.compare("LRN") && !AttrName.compare("size")) {
               int size = std::stoi(val);
              if (size % 2 == 0) {
                size = size + 1;
              }
              val = to_string(size);
            } else if (!AttrName.compare("axis")) {	    
              axis = std::stoi(val);
            } else if (!AttrName.compare("axes")) {
              auto list = val.substr(1); //skip "["
              auto tmp = list.find_first_of(",");
              int max = 0;
              while (tmp != string::npos) {
                int cur = stoi(list.substr(0, 1));
                max = max > cur ? max : cur;
                list = list.substr(tmp+1);
                tmp = list.find_first_of(","); 
              }
              int last = stoi(list.substr(0, 1));
              max = max > last ? max : last;
              axis = max;
            } else if (!AttrName.compare("dtype") && !name.compare("EyeLike")) {
              int type_index = stoi(val);
              o << "            " << AttrName << "=" << onnx::onnx_type_names[type_index] << ",\n";
              attr_vals.push_back(onnx::np_type_names[type_index]);              
              continue;
            }
          
            o << "            " << AttrName << "=" << val << ",\n";
            attr_vals.push_back(val);         
          }
      }
    }

    o << "        )\n";

    int count = 0;
    //For each constructed node, there will be multiple unit tests.
    //max_test_count is the maximum tests number for each node.
    int max_test_count = no_input ? 1 : 10;
    auto gName = name;
    gName[0] = std::tolower(gName[0]);
    
    bool handle_broadcast = false;
    bool handle_nonzero = false;
    //Handle properties.
    ListInit *PropertyList = oneclass->getValueAsListInit("properties_");
    for ( unsigned i = 0, e = PropertyList->size(); i!=e; ++i ) {
      Record *Property = PropertyList->getElementAsRecord(i);
      if ( Property->getName() == "broadcasting" )
        handle_broadcast = true;
      else if ( Property->getName() == "nonzero" ) 
        handle_nonzero = true;
      else
        assert(false && "unsupported property");
    }
    
    bool isCompress = !name.compare("Compress");
    bool isBatchNorm = !name.compare("BatchNormalization");
    bool isConcat = !name.compare("Concat");
    while (count < max_test_count) { 
    std::vector<std::vector<int>> shapes;
    //Generate input operands for each unit test.
    for (unsigned int i = 0; i < input_size; ++i) {
      int input_index = dynamic_size_input ? 0 : i;
      if (!name.compare("Reshape") && i > 0) {
	//Special handling of reshape. The second operand is a vector representing the target shape.
	//The target shape is calculated with reshape() helper function.
	//Reshape type is selected randomly.
        vector<int> result = reshape(shapes[0], (RESHAPE_TYPE)GetRandomInt(REORDER, NEGATIVE_DIM));
        o << "        x_" << std::to_string(i) << " = np.array([";
        for (unsigned m = 0; m < result.size(); ++m) {
          o << result[m];
          if (m < result.size() - 1) {
            o << ",";
          }
        }
        o << "], dtype = np." << in_type[i] << ")\n";
        continue;
      }
      //Generate an input according to the value range defined in the td file.
      //Some ops may have speical properties such as broadcasting, and some may
      //have restrictions by axis value.
      std::vector<int> shape;
      int min_dim = list_in[input_index]->getValueAsInt("min_dim_");
      int axis_bound = list_in[input_index]->getValueAsInt("axis_bound_");
      //min_dim should be no less than axis value
      if (axis_bound && (axis != INVALID_AXIS)) {
	//If there is axis attribute, the input's rank should be larger than the axis value.
        int rank = (axis < 0) ? (axis * -1) - 1 : axis;
        min_dim = min_dim < (rank+1) ? (rank+1) : min_dim;
      }
      int max_dim = list_in[input_index]->getValueAsInt("max_dim_");
      if (pool_attr.init) {
	//For pooling operands, the rank is always pads.size()/2 +2.
        max_dim = min_dim = pool_attr.pads.size()/2 + 2;
      }
      if (handle_broadcast && ((input_size > 2) || !name.compare("PRelu")/* unidirectional broadcast*/) && (i > 0)) {
	//When broadcasting is supported, and there are more than two inputs,
	//Or in the case of OpPRelu,
	//the max dimension for all other inputs except the first one should
	//be the first input's rank.
        max_dim = shapes[0].size();
      }
      //Get current input's dimension.
      int dim = GetDimension(i, min_dim, max_dim);
      if ((i > 0) && (isConcat || (!name.compare("MatMul")))) {
	//For matmul/concat, other inputs; rank should match the first one.
        dim = shapes[0].size();
      }
        
      bool is_string = (in_type[input_index][0] == 'o');
      int normal_dis = list_in[input_index]->getValueAsInt("normd_");
      if (!normal_dis || handle_nonzero) { 
	//For non-normal distruction specified in the td file,
	//user-defined low and high value should be respected.
	//Select appropriate random functions based on the input element type.
        auto min_val_v = list_in[input_index]->getValueAsListOfStrings("min_val_");
        auto max_val_v = list_in[input_index]->getValueAsListOfStrings("max_val_");
        bool is_int, is_unsign = false;
        auto fn = GetRandomFn(in_type[input_index], is_int, is_unsign);
        string max_val_s;
        string min_val_s;
        int min_max_index = 0;
        if (!min_val_v.size()) {
          if (is_int) {
            min_val_s = to_string(MIN_MAX_LIMIT::val_min_i);
          } else {
            min_val_s = to_string(MIN_MAX_LIMIT::val_min_f);
          }
        } else {
          //There could be more than 1 pair of min/max range defined in the td file.
	  //Select one pair randomly.
          min_max_index = GetRandomInt(0, min_val_v.size() - 1);
          min_val_s = min_val_v[min_max_index];
        }
        if (!max_val_v.size()) {
          if (is_int) {
            max_val_s = to_string(MIN_MAX_LIMIT::val_max_i);
          } else {
            max_val_s = to_string(MIN_MAX_LIMIT::val_max_f); 
          }
        } else {
          max_val_s = max_val_v[min_max_index];
        }
        if (max_val_s == "?") {
	  //Some op's max value could be undefined in the td file(marked by "?")
	  //This means the actua value is determined by some restrictions such
	  //as the axis value, other input's shape.
          if (!name.compare("Gather") && (i > 0)) {
            int axis_val = axis < 0 ? axis + shapes[0].size() : axis;
            max_val_s = to_string(shapes[0][axis_val] - 1);
          }
        }
        if (max_val_s == min_val_s) {
          o << "        x_" << std::to_string(i) << " = np.full((";
        } else {
          o << "        x_" << std::to_string(i) << " = ";
          if (handle_nonzero) {
            //For some inputs that can't be zero
            if (!is_unsign) {
              //For signed case, limit range should be [min, 0) and -[min, 0)
              max_val_s = "0";
              min_val_s = "-100";
              bool modify_range = GetRandomBool();
              if (modify_range)
                o << "-1 * ";
            } else {
              //For unsigned case, limit range should be max - [0, max)
              min_val_s = "0";
              max_val_s = "100";
              o << max_val_s << " - ";
            }
          } 
          o << fn << "(" << min_val_s << ", " << max_val_s << ", size=(";
        }
        int max_size = GetMaxSizeInOneDim(dim);
        for (int k = 0; k < dim; ++k) {
          int size;
          if ((i > 0) && handle_broadcast && (k >= (dim - (int)shapes[i-1].size()))) {
            bool selector = GetRandomBool();
            size = selector ? 1 : shapes[i-1][k - dim + shapes[i-1].size()];
          } else if (!name.compare("Tile") && (i == 1)) {
            size = shapes[0].size();
          } else {
            size = GetRandomInt(1, max_size);
          }
          o << std::to_string(size);
          if (k != dim - 1) {
            o << ",";
          }
          shape.push_back(size);
        }
        if (max_val_s == min_val_s) {
          o << "), " << max_val_s << ", np." << in_type[input_index] << ")\n";
        } else {
          o << ")).astype(np." << in_type[input_index] << ")\n";
        }
      } else {
        if (is_string) {
          o << "        ss = []\n";
        }
        if (dim == 0 && !is_string) { //scalar 
          o << "        x_" << std::to_string(i) << " = np." << in_type[input_index] << "(";
          if (!name.compare("Expand") && (i==1)) {
            o << "1)\n";
          } else {
            o << "np.random.rand())\n";
          }
        } else {
          OperandAttr op_attr;
          op_attr.handle_broadcast = handle_broadcast;
          op_attr.i = i;
          op_attr.pool_attr = pool_attr;
          op_attr.isBatchNorm = isBatchNorm;
          op_attr.isCompress = isCompress;
          op_attr.isConcat = isConcat;
          op_attr.isDepthToSpace = !name.compare("DepthToSpace");
          op_attr.attr_vals = attr_vals;
          op_attr.axis = axis;
          op_attr.name = name;
          int max_size = GetMaxSizeInOneDim(dim);
          o << "        x_" << std::to_string(i) << " = ";
          if (is_string) {
	    //Handle string type inputs.
            shape = GenerateOneOperand(dim, max_size, shapes, op_attr, input_index);
            int sum = 1;
            for (unsigned j = 0; j < shape.size(); ++j) {
              sum *= shape[j];
            }
            o << "np.chararray((" << sum << "), itemsize=1, unicode=True)\n";
            o << "        for i in x_" << std::to_string(i) << ":\n";
            o << "            s = str(i).encode(\'utf-8\').decode(\'utf-8\')\n";
            o << "            ss.append(s)\n";
            o << "        x_" << std::to_string(i) << " = np.array(ss).astype(np.object";
            if (shape.size()) {
              o << ").reshape(";
              for (unsigned j = 0; j < shape.size(); ++j) {
                o << shape[j];
                if (j != shape.size() - 1) 
                  o << ",";
              }
            }
            o << ")\n";
            shapes.push_back(shape);
            continue;
          }
	  //Special handling.
          if (!name.compare("Expand") && (i==1)) {
            o << "np.asarray(";
          }
	  //Generate one operand value.
          o << "np.random.randn(";
          shape = GenerateOneOperand(dim, max_size, shapes, op_attr, input_index);
          for (unsigned int k = 0; k < shape.size(); ++k) {
            o << to_string(shape[k]);
            if (k != shape.size() - 1) {
              o << ",";
            }
          }
          o << ").astype(np." << in_type[input_index] << ")";
          if (!name.compare("Expand") && (i==1)) {
            o << ".shape, dtype = np." << in_type[input_index] << ")";
          }
          o << "\n";
        }
      }
      shapes.push_back(shape);
    }

    // handle dynamic_size_input
    if (dynamic_size_input) {
      o << "        x = [np.array(a) for a in [";
      for (unsigned int i = 0; i < input_size; ++i) {
        o << "x_" << std::to_string(i);
        if (i != input_size - 1) {
          o << ",";
        }
      }
      o << "]]\n";
    } 
    o << "        ";
    for (unsigned int i = 0; i < list_out.size(); ++i) {
      o << "y_" << std::to_string(i);
      if (i != list_out.size() - 1) {
        o << ",";
      }
    }
    //Generate outpus.
    if (no_input) {
      //No input case.
      o << " = values\n";
    } else {
      o << " = " << name << "_compute(";
      if (dynamic_size_input) {
	//Dynamic input case.
        o << "x";
      } else {
        for (unsigned int i = 0; i < input_size; ++i) {
          o << "x_" << std::to_string(i);
          if (i != input_size - 1) {
            o << ",";
          }
        }
      }

      if (Attrs.size()) {
	//Append attribute's value if exists.
        int i = 0;
        for (std::vector<Record *>::iterator IC = Attrs.begin(), EC = Attrs.end();
             IC != EC; ++IC) {
            o << ", " << attr_vals[i++]; 
        }
      }
      o << ")\n";
    }
    //Generate "expect(....)".
    o << "        expect(node, inputs=[";
    for (unsigned int i = 0; i < input_size; ++i) {
      o << "x_" << std::to_string(i);
      if (i != input_size - 1) {      
        o << ",";
      }
    }
    o << "], outputs=[";
    for (unsigned int j = 0; j < list_out.size(); ++j) {
      o << "y_" << std::to_string(j);
      if (j != list_out.size() - 1) {
        o << ",";
      }
    }    
    o << "], name=\'test_" << gName << "_" << std::to_string(test_count_) <<"\')\n\n";    
    count++;
    test_count_++;
    }
}
} // end onnx namespace

namespace llvm {
/*!
 * Interface function for llvm TableGen to generate onnx compliance tests.
 */
void EmitOnnxTests(RecordKeeper &RK, raw_ostream &OS, bool smoke) {
  onnx::CodeEmitterGen(RK).gen_onnx_test(OS, smoke);
}
} // end namespace llvm

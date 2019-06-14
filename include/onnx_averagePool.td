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

include "onnx_instr.td"

def AveragePoolOp : Instruction<[op_avg_pool],
                            [Attr<"pads", [i32_v2, i32_v4, i32_v6], "", ["0"], ["10"]>,
                             Attr<"strides", [i32_v1, i32_v2, i32_v3], "", ["0"], ["10"]>,
                             Attr<"kernel_shape", [i32_v1, i32_v2, i32_v3], "", ["1"], ["10"]>,
                             Attr<"auto_pad", [str],"NOTSET",["NOTSET,SAME_UPPER,SAME_LOWER,VALID"]>,
                             Attr<"count_include_pad", [i32], "0", ["0"], ["1"]>],
                            [IOType<0, [f32, f16, f64], 3, 5>],
                            [IOType<0, [f32, f16, f64]>]>;



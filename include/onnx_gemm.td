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

def GemmOp : Instruction<[op_gemm],
                        [Attr<"alpha", [f32], "1.0">,
                         Attr<"beta", [f32], "1.0">,
                         Attr<"transA", [i1], "0">,
                         Attr<"transB", [i1], "0">],
                        [IOType<0, [f32, f16, f64, i32, i64, ui32, ui64], 2, 2>,
                         IOType<1, [f32, f16, f64, i32, i64, ui32, ui64], 2, 2>,
                         IOType<2, [f32, f16, f64, i32, i64, ui32, ui64], 2, 2>],
                        [IOType<0, [f32, f16, f64, i32, i64, ui32, ui64]>]>;



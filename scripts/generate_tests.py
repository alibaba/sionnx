#*
#* Copyright (C) 2017-2019 Alibaba Group Holding Limited
#*
#* Licensed under the Apache License, Version 2.0 (the "License");
#* you may not use this file except in compliance with the License.
#* You may obtain a copy of the License at
#*
#*      http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing, software
#* distributed under the License is distributed on an "AS IS" BASIS,
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#* See the License for the specific language governing permissions and
#* limitations under the License.

import sys
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='Generate conformanc tests')
parser.add_argument("-profile_level", help="Specify the profile level: 0=smoke tests; 1=full tests", type=int)
parser.parse_args()
args = parser.parse_args()

option = "-gen-onnx-smoke-tests"
if args.profile_level:
    option = "-gen-onnx-smoke-tests" if args.profile_level==0 else "-gen-onnx-tests"
print("======Generating tests with option " + option + "========")

if not os.path.exists("tests"):
    os.makedirs("tests")

os.system("cp ../include/onnx_*.td -r . | cp ../include/*.algorithm -r .")
dir_path = os.path.dirname(os.path.realpath(__file__))
td_files = glob.glob(os.path.join(dir_path, '*.td'))

lens = len(td_files)
for k in range(lens):
    base = os.path.basename(td_files[k])
    out_file_name = os.path.splitext(base)[0]
    os.system("../llvm/build/bin/llvm-tblgen " + option + " " + td_files[k] + " -I ./ -o ./tests/" + out_file_name + ".py") 
    print(out_file_name + ".py generated.")

os.system("rm onnx_*.td | rm *.algorithm")

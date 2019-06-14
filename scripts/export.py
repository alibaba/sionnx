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
import argparse

parser = argparse.ArgumentParser(description='Export conformanc tests to protobuf files')
parser.add_argument("onnx_path", help="Specify the absolute root path of onnx source code", type=str)
parser.parse_args()
args = parser.parse_args()

path = args.onnx_path
dir_path = os.path.dirname(os.path.realpath(__file__))
os.system("rm " + path + "/onnx/backend/test/case/node/onnx_*.py")
os.system("cp ./tests/onnx_*.py " + path + "/onnx/backend/test/case/node/.")
os.chdir(path)
os.system("python setup.py install")
os.chdir(dir_path)
os.system("backend-test-tools generate-data -o ./exported/")

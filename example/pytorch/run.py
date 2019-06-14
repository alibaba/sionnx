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

import numpy as np
import onnx
import os
import glob
import caffe2.python.onnx.backend
from caffe2.python import core, workspace

from onnx import numpy_helper
import os

fail_sum = 0
dir_path = os.path.dirname(os.path.realpath(__file__))
test_dir = glob.glob(os.path.join(dir_path, 'test_*'))
model_paths = glob.glob(os.path.join(os.path.join(dir_path, 'test_*'), '*.onnx'))
m_len = len(model_paths)
for k in range(m_len):
  model = onnx.load(model_paths[k])
  test_data_dir = os.path.join(test_dir[k], 'test_data_set_0')

  # Load inputs
  inputs = []
  inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
  for i in range(inputs_num):
      input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
      tensor = onnx.TensorProto()
      with open(input_file, 'rb') as f:
          tensor.ParseFromString(f.read())
      inputs.append(numpy_helper.to_array(tensor))

  # Load reference outputs
  ref_outputs = []
  ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
  for j in range(ref_outputs_num):
      output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(j))
      tensor = onnx.TensorProto()
      with open(output_file, 'rb') as f:
          tensor.ParseFromString(f.read())
      ref_outputs.append(numpy_helper.to_array(tensor))


  # Run the model on the backend
  try:
      outputs = list(caffe2.python.onnx.backend.run_model(model, inputs))
  except RuntimeError:
      print("!!Error: Model execution of " + test_dir[k] + " failed.")
      fail_sum = fail_sum + 1
      continue

  idx = 0
  # Results verification with golden data.
  for ref_o, o in zip(ref_outputs, outputs):
      try:
          np.testing.assert_almost_equal(ref_o, o, decimal=5, err_msg="Failed test: " + test_dir[k])
      except AssertionError:
          print("!!Error: Output " + str(idx) + " of test: " + test_dir[k] + " failed")
          fail_sum = fail_sum + 1
      idx = idx + 1

print("============Summary:=============")
print(str(m_len) + " tests in total.")
print(str(m_len - fail_sum) + " tests passed.")
print(str(fail_sum) + " tests failed.")
print("=================================")

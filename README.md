# SIONNX
[![License](https://img.shields.io/badge/license-Apache%202-4EB1BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

## Introduction
   Sionnx is a tool to automatically generate tests for ONNX compliance. It's originated from Sinian project and can generate compliance tests with configurable settings for ONNX runtime. The generated tests can be exported to protobuf format to be compatible with many ONNX runtime frameworks.

   Sionnx includes a DSL to describe ONNX instrucitons. It leverages LLVM TableGen toolchain to parse the DSL files and generate tests in Python. The TableGen is customized to handle the DSL syntax.

   Sinian is Alibaba’s heterogeneous hardware acceleration and optimization platform, targeting extreme performance and high execution efficiency for machine learning and data-intensive applications. Sinian is a unified platform to support both machine learning training and inferencing, but fully tailorable statically for cloud computing, edge computing, and IoT devices. Sinian makes it seamless to build, train, and deploy machine learning models without suffering the loss of performance portability.

![sinian](logo.png "Sinian") 
![sionnx](logo-sionnx.png "Sionnx")

## Getting Started

### Build From Source

- git clone https://github.com/alibaba/sionnx.git
- cd sionnx && mkdir build
- cd build && cmake ../
- make 

### Generate unit ONNX tests:

- cd sionnx/scripts
- python generate_tests.py (-profile_level: 0=smoke tests; 1=full tests. Default is 0)
- Generated tests are under folder sionnx/scripts/tests.

### Export Tests as Model File + Data Files:

- Prerequisite: Download ONNX(https://github.com/onnx/onnx) source code.
- cd sionnx/scripts
- python export.py #onnx_path#(the absolute root path of onnx source code).

### Support A New Op:

1. Add/modify .td file in include/.
2. Add .algorithm file(written in numpy) in include/.
3. Update llvm/utils/TableGen/OnnxTestsEmitter.cpp if necessary.
4. Build.

## Documentation
Sionnx: Automatic Unit Test Generator for ONNX Conformance: https://arxiv.org/abs/1906.05676

## License
```
Copyright (C) 2017-2019 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


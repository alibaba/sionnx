# SIONNX
[![License](https://img.shields.io/badge/license-Apache%202-4EB1BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

## Introduction
   SIONNX is an auto-gen tests tool for ONNX compliance. It's part of Sinian heterogeneous computing framework and can generate compliance tests automatically for ONNX runtime with configurable settings. The generated tests can be exported to protobuf format to be compatible with many ONNX runtime frameworks.

![sinian](logo.png "Sinian" | width=400)

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

1. Add .td file in include/.
2. Add .golden file(written in numpy) in include/.
3. Update llvm/utils/TableGen/OnnxTestsEmitter.cpp if necessary.
4. Build.

## Documentation


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


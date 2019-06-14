include(ExternalProject)

set(tblgen_build_dir ${CMAKE_BINARY_DIR}/tblgen)
ExternalProject_Add(onnx_ir_tblgen
  PREFIX ${CMAKE_BINARY_DIR}/tblgen
  LIST_SEPARATOR ,
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/llvm
  BINARY_DIR ${tblgen_build_dir}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_BUILD_TOOLS=OFF -DLLVM_BUILD_UTILS=OFF
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} llvm-tblgen
  INSTALL_COMMAND ""
  USES_TERMINAL_CONFIGURE 0
  USES_TERMINAL_BUILD 1
  BUILD_BYPRODUCTS ${tblgen_build_dir}/bin/llvm-tblgen
)

set(tblgen_generated_scripts)
set(onnx_instr_td ${CMAKE_CURRENT_SOURCE_DIR}/onnx_instr.td)
set(onnx_instr_info_gen ${CMAKE_CURRENT_BINARY_DIR}/onnx_instr_info.gen)

file(GLOB goldens "*.algorithm")
foreach(g ${goldens})
  get_filename_component(g_name ${g} NAME_WE)
  set(ogn "${g_name}.algorithm")
  configure_file(${g} ${ogn} COPYONLY)
endforeach()

add_custom_command(OUTPUT ${onnx_instr_info_gen}
  COMMAND ${tblgen_build_dir}/bin/llvm-tblgen
  -gen-onnx-op-info
  ${onnx_instr_td} -o ${onnx_instr_info_gen}
  DEPENDS onnx_ir_tblgen ${onnx_instr_td} ${${tblgen_build_dir}/bin/llvm-tblgen}
    COMMENT "Building onnx_instr_info.gen..."
)

file(GLOB local_onnx_tds "onnx_*.td")

#generate onnx_xxx.py
foreach(t ${local_onnx_tds})
  get_filename_component(inst_name ${t} NAME_WE)
  set(ofn "${inst_name}.py")
  string(TOUPPER "${inst_name}" def_name)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND ${tblgen_build_dir}/bin/llvm-tblgen
    -gen-onnx-tests -I${CMAKE_CURRENT_SOURCE_DIR}
    ${t} -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
#    -target ${onnx_target}
    -ifdef ${def_name}
    DEPENDS onnx_ir_tblgen ${t} ${onnx_instr_td} ${${tblgen_build_dir}/bin/llvm-tblgen}
    COMMENT "Building ${ofn}..."
  )

  set(tblgen_generated_scripts ${tblgen_generated_scripts} ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/tests/${ofn} PROPERTIES GENERATED 1)
endforeach()

add_custom_target(OnnxTests
    DEPENDS ${onnx_instr_info_gen} ${tblgen_generated_scripts})
add_dependencies(OnnxTests onnx_ir_tblgen)



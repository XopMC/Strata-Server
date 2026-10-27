cmake_minimum_required(VERSION 3.24)
project(strata_deepseek_classic_source LANGUAGES NONE)

# This source island deliberately keeps its old matching C++ classes/headers.
# It does not consume new upstream engine classes or historical object archives.
option(STRATA_DS_ENABLE_SERVING_FACTORY "Unqualified source factory (not admitted)" OFF)
if(STRATA_DS_ENABLE_SERVING_FACTORY)
  message(FATAL_ERROR "DeepSeek serving factory is disabled pending clean-build binding/reseal and lifecycle qualification")
endif()

set(STRATA_DS_CONTEXT "262144" CACHE STRING "Selected native558 context: 262144 or 524288")
set_property(CACHE STRATA_DS_CONTEXT PROPERTY STRINGS 262144 524288)
if(NOT STRATA_DS_CONTEXT STREQUAL "262144" AND NOT STRATA_DS_CONTEXT STREQUAL "524288")
  message(FATAL_ERROR "Only the two selected native558 context closures are admitted")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit 13.4 EXACT REQUIRED)
find_program(STRATA_DS_CXX NAMES clang++-18 REQUIRED)
find_program(STRATA_DS_AR NAMES llvm-ar-18 llvm-ar REQUIRED)
find_program(STRATA_DS_MOLD NAMES mold REQUIRED)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/build-manifest.json" manifest)
string(JSON unresolved LENGTH "${manifest}" unresolved)
string(JSON source_problems LENGTH "${manifest}" source_problems)
if(NOT unresolved EQUAL 0 OR NOT source_problems EQUAL 0)
  message(FATAL_ERROR "DeepSeek source recovery is incomplete; see build-manifest.json and STATUS.md")
endif()

set(run "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/build_from_manifest.py"
  --source "${CMAKE_CURRENT_SOURCE_DIR}" --binary "${CMAKE_CURRENT_BINARY_DIR}"
  --cuda-root "${CUDAToolkit_TARGET_DIR}"
  --cuda-library-dir "${CUDAToolkit_LIBRARY_DIR}"
  --cxx "${STRATA_DS_CXX}" --nvcc "${CUDAToolkit_NVCC_EXECUTABLE}" --ar "${STRATA_DS_AR}")

function(expand_output input output)
  string(REPLACE "@BINARY_ROOT@" "${CMAKE_CURRENT_BINARY_DIR}" result "${input}")
  set("${output}" "${result}" PARENT_SCOPE)
endfunction()

string(JSON count LENGTH "${manifest}" units)
math(EXPR last "${count} - 1")
set(unit_outputs)
foreach(i RANGE 0 ${last})
  string(JSON key MEMBER "${manifest}" units ${i})
  string(JSON output GET "${manifest}" units "${key}" output)
  expand_output("${output}" output)
  set("unit_${key}" "${output}")
  list(APPEND unit_outputs "${output}")
endforeach()

# Direct DSO links (currently recompute) depend on freshly compiled object inputs.
foreach(i RANGE 0 ${last})
  string(JSON key MEMBER "${manifest}" units ${i})
  set(dependencies "${CMAKE_CURRENT_SOURCE_DIR}/build-manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/build_from_manifest.py")
  string(JSON n LENGTH "${manifest}" units "${key}" dependencies)
  if(n GREATER 0)
    math(EXPR end "${n} - 1")
    foreach(j RANGE 0 ${end})
      string(JSON dep GET "${manifest}" units "${key}" dependencies ${j})
      list(APPEND dependencies "${CMAKE_CURRENT_SOURCE_DIR}/${dep}")
    endforeach()
  endif()
  string(JSON n LENGTH "${manifest}" units "${key}" argv)
  math(EXPR end "${n} - 1")
  foreach(j RANGE 0 ${end})
    string(JSON arg GET "${manifest}" units "${key}" argv ${j})
    if(arg MATCHES "^@BINARY_ROOT@/objects/")
      expand_output("${arg}" dep)
      if(NOT dep STREQUAL "${unit_${key}}")
        list(APPEND dependencies "${dep}")
      endif()
    endif()
  endforeach()
  add_custom_command(OUTPUT "${unit_${key}}" COMMAND ${run} unit "${key}"
    DEPENDS ${dependencies} VERBATIM COMMAND_EXPAND_LISTS)
endforeach()

string(JSON archive_count LENGTH "${manifest}" archives)
math(EXPR archive_last "${archive_count} - 1")
set(archive_outputs)
foreach(i RANGE 0 ${archive_last})
  string(JSON key MEMBER "${manifest}" archives ${i})
  string(JSON output GET "${manifest}" archives "${key}" output)
  expand_output("${output}" output)
  string(JSON n LENGTH "${manifest}" archives "${key}" members)
  math(EXPR end "${n} - 1")
  set(dependencies)
  foreach(j RANGE 0 ${end})
    string(JSON sha GET "${manifest}" archives "${key}" members ${j} object)
    list(APPEND dependencies "${unit_${sha}}")
  endforeach()
  add_custom_command(OUTPUT "${output}" COMMAND ${run} archive "${key}"
    DEPENDS ${dependencies} "${CMAKE_CURRENT_SOURCE_DIR}/build-manifest.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/build_from_manifest.py" VERBATIM COMMAND_EXPAND_LISTS)
  list(APPEND archive_outputs "${output}")
endforeach()
string(JSON output GET "${manifest}" engines "${STRATA_DS_CONTEXT}" output)
expand_output("${output}" output)
add_custom_command(OUTPUT "${output}" COMMAND ${run} engine "${STRATA_DS_CONTEXT}"
  DEPENDS ${unit_outputs} ${archive_outputs} "${CMAKE_CURRENT_SOURCE_DIR}/build-manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/build_from_manifest.py" VERBATIM COMMAND_EXPAND_LISTS)
add_custom_target(strata_deepseek_classic ALL DEPENDS "${output}")
# Never load both context DSOs into one interposable namespace. The existing C ABI
# and SONAME remain unchanged; one selected context is built/loaded per owner.

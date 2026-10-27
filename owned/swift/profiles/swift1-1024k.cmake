# Imported selected source closure; Root chooses which profile to include.
if(NOT DEFINED STRATA_PROFILE_HEADERS)
  set(STRATA_PROFILE_HEADERS ${STRATA_CLASSIC_HEADERS})
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/objects" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k")
# Source-only profile swift1-1024k; no build/model validation claimed.
if(NOT TARGET obj_66c329dca7782a1d)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66c329dca7782a1d.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-O3" "-DNDEBUG" "-std=c++20" "-I${CUDAToolkit_INCLUDE_DIR}" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v113/qsa_capture.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66c329dca7782a1d.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v113/qsa_capture.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_66c329dca7782a1d DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66c329dca7782a1d.o")
endif()
if(NOT TARGET obj_d2e9d16459e0b99e)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d2e9d16459e0b99e.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin" "/usr/bin/clang++-18" "-forward-unknown-to-host-compiler" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-Xptxas=-v" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v112/qsa_register.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d2e9d16459e0b99e.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v112/qsa_register.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_d2e9d16459e0b99e DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d2e9d16459e0b99e.o")
endif()
if(NOT TARGET obj_73a65df92ca17d32)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73a65df92ca17d32.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-O3" "-DNDEBUG" "-std=c++20" "-I${CUDAToolkit_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v168-r3/prefill_hook.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73a65df92ca17d32.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v168-r3/prefill_hook.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_73a65df92ca17d32 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73a65df92ca17d32.o")
endif()
if(NOT TARGET obj_4d3bedb7534cca4b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4d3bedb7534cca4b.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin" "/usr/bin/clang++-18" "-forward-unknown-to-host-compiler" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-Xptxas=-v" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v142/selector.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4d3bedb7534cca4b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v142/selector.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_4d3bedb7534cca4b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4d3bedb7534cca4b.o")
endif()
if(NOT TARGET obj_1875396d27a5c9e8)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1875396d27a5c9e8.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin" "/usr/bin/clang++-18" "-forward-unknown-to-host-compiler" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-Xptxas=-v" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v167/selector16.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1875396d27a5c9e8.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v167/selector16.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_1875396d27a5c9e8 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1875396d27a5c9e8.o")
endif()
if(NOT TARGET obj_f8f8cf29d7aeaee5)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f8f8cf29d7aeaee5.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin" "/usr/bin/clang++-18" "-forward-unknown-to-host-compiler" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-Xptxas=-v" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v168/check_ids.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f8f8cf29d7aeaee5.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v168/check_ids.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_f8f8cf29d7aeaee5 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f8f8cf29d7aeaee5.o")
endif()
if(NOT TARGET obj_fcd7cf0176c16fed)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fcd7cf0176c16fed.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-O3" "-DNDEBUG" "-std=c++20" "-DSTRATA_VERSION=\"0.1.39\"" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_PREFILL_FUSED=1" "-DSTRATA_PREFILL_MMQ=1" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v18/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v142/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v168/generate.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fcd7cf0176c16fed.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v168/generate.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_fcd7cf0176c16fed DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fcd7cf0176c16fed.o")
endif()
if(NOT TARGET obj_fb2c2cb14cbbfa2c)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb2c2cb14cbbfa2c.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin" "/usr/bin/clang++-18" "-forward-unknown-to-host-compiler" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-x" "cu" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-own-heretic-q4km/research/rvn-bf16-ssm-compat-candidate/src/kernels/cuda/bf16_shared_down_compat.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb2c2cb14cbbfa2c.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-own-heretic-q4km/research/rvn-bf16-ssm-compat-candidate/src/kernels/cuda/bf16_shared_down_compat.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_fb2c2cb14cbbfa2c DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb2c2cb14cbbfa2c.o")
endif()
if(NOT TARGET obj_ca8a98bc3224bdfd)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ca8a98bc3224bdfd.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin" "/usr/bin/clang++-18" "-O3" "-std=c++20" "--generate-code=arch=compute_89,code=sm_89" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v2/bf16_projection.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ca8a98bc3224bdfd.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v2/bf16_projection.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_ca8a98bc3224bdfd DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ca8a98bc3224bdfd.o")
endif()
if(NOT TARGET obj_e4dc16ac4ee5c890)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e4dc16ac4ee5c890.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/layer.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e4dc16ac4ee5c890.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/layer.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_e4dc16ac4ee5c890 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e4dc16ac4ee5c890.o")
endif()
if(NOT TARGET obj_d09bd2860b0f4f86)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d09bd2860b0f4f86.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/session.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d09bd2860b0f4f86.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/session.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_d09bd2860b0f4f86 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d09bd2860b0f4f86.o")
endif()
if(NOT TARGET obj_1667155e707512ca)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1667155e707512ca.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/src/core/expert_source.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1667155e707512ca.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/src/core/expert_source.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_1667155e707512ca DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1667155e707512ca.o")
endif()
if(NOT TARGET obj_34a95da22d3f5a1b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_34a95da22d3f5a1b.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/remote_experts.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_34a95da22d3f5a1b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/remote_experts.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_34a95da22d3f5a1b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_34a95da22d3f5a1b.o")
endif()
if(NOT TARGET obj_6566a7367dccc966)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6566a7367dccc966.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/include" "-forward-unknown-to-host-compiler" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-x" "cu" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/remote_expert_opt.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6566a7367dccc966.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/remote_expert_opt.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6566a7367dccc966 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6566a7367dccc966.o")
endif()
if(NOT TARGET obj_d1161d8c0474dc1a)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d1161d8c0474dc1a.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/expert_cache.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d1161d8c0474dc1a.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/expert_cache.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_d1161d8c0474dc1a DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d1161d8c0474dc1a.o")
endif()
if(NOT TARGET obj_7947b0c4a524fbba)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7947b0c4a524fbba.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/native_head.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7947b0c4a524fbba.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/native_head.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_7947b0c4a524fbba DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7947b0c4a524fbba.o")
endif()
if(NOT TARGET obj_2890fdcafb327260)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2890fdcafb327260.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/native_dense.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2890fdcafb327260.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/native_dense.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_2890fdcafb327260 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2890fdcafb327260.o")
endif()
if(NOT TARGET obj_17de87d99a19f000)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_17de87d99a19f000.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v142/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-own-heretic-q4km/research/rvn-bf16-ssm-compat-candidate/src/core/verify.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_17de87d99a19f000.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-own-heretic-q4km/research/rvn-bf16-ssm-compat-candidate/src/core/verify.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_17de87d99a19f000 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_17de87d99a19f000.o")
endif()
if(NOT TARGET obj_f611040249752996)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f611040249752996.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-O3" "-DNDEBUG" "-std=c++20" "-DSTRATA_VERSION=\"0.1.39\"" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_PREFILL_FUSED=1" "-DSTRATA_PREFILL_MMQ=1" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v18/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v142/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v78/mtp.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f611040249752996.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v78/mtp.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_f611040249752996 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f611040249752996.o")
endif()
if(NOT TARGET obj_556ba9dd8db975fc)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_556ba9dd8db975fc.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/conversation_snapshot.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_556ba9dd8db975fc.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/conversation_snapshot.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_556ba9dd8db975fc DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_556ba9dd8db975fc.o")
endif()
if(NOT TARGET obj_29eb60798b053a70)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_29eb60798b053a70.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/conversation_state.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_29eb60798b053a70.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/conversation_state.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_29eb60798b053a70 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_29eb60798b053a70.o")
endif()
if(NOT TARGET obj_670c4a216496a57f)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_670c4a216496a57f.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/conversation_memory.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_670c4a216496a57f.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/conversation_memory.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_670c4a216496a57f DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_670c4a216496a57f.o")
endif()
if(NOT TARGET obj_a4cb6ae88a859449)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4cb6ae88a859449.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/peer_experts.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4cb6ae88a859449.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/peer_experts.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_a4cb6ae88a859449 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4cb6ae88a859449.o")
endif()
if(NOT TARGET obj_810d0f3de71b0550)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_810d0f3de71b0550.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_PREFILL_FUSED=1" "-DSTRATA_PREFILL_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/gemm.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_810d0f3de71b0550.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/gemm.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_810d0f3de71b0550 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_810d0f3de71b0550.o")
endif()
if(NOT TARGET obj_76e8079cf3b53f61)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_76e8079cf3b53f61.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_PREFILL_FUSED=1" "-DSTRATA_PREFILL_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/kernels.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_76e8079cf3b53f61.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/kernels.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_76e8079cf3b53f61 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_76e8079cf3b53f61.o")
endif()
if(NOT TARGET obj_5d560c571fa7876c)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5d560c571fa7876c.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/native/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v18/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v69/src/prefill" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v41/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_PREFILL_FUSED=1" "-DSTRATA_PREFILL_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/swiglu-quantize-integration-r60/prefill-swiglu-quant-r60.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5d560c571fa7876c.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/swiglu-quantize-integration-r60/prefill-swiglu-quant-r60.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_5d560c571fa7876c DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5d560c571fa7876c.o")
endif()
if(NOT TARGET obj_b348dedbbe0acb3f)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b348dedbbe0acb3f.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_PREFILL_FUSED=1" "-DSTRATA_PREFILL_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-x" "cu" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v63/src/prefill/wire_bf16_v63.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b348dedbbe0acb3f.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v63/src/prefill/wire_bf16_v63.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_b348dedbbe0acb3f DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b348dedbbe0acb3f.o")
endif()
if(NOT TARGET obj_1575e44ccc7babe5)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1575e44ccc7babe5.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_PREFILL_FUSED=1" "-DSTRATA_PREFILL_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-x" "cu" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v69/src/prefill/hc_upmix_v69.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1575e44ccc7babe5.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v69/src/prefill/hc_upmix_v69.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_1575e44ccc7babe5 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1575e44ccc7babe5.o")
endif()
if(NOT TARGET obj_7ce37f4a71ca1a5b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ce37f4a71ca1a5b.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-Xcompiler=-fuse-ld=mold" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/prefill-fused-scale-r34/fused_scale_combine_r34.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ce37f4a71ca1a5b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/prefill-fused-scale-r34/fused_scale_combine_r34.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_7ce37f4a71ca1a5b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ce37f4a71ca1a5b.o")
endif()
if(NOT TARGET obj_224ba431e03b42c6)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_224ba431e03b42c6.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-Xcompiler=-fuse-ld=mold" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/swiglu-quantize-fusion-r56/swiglu_quantize_r56.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_224ba431e03b42c6.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/swiglu-quantize-fusion-r56/swiglu_quantize_r56.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_224ba431e03b42c6 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_224ba431e03b42c6.o")
endif()
if(NOT TARGET obj_a525907265eb4cc7)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a525907265eb4cc7.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-O3" "-DNDEBUG" "-std=c++20" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/spec/suffix_drafter.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a525907265eb4cc7.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/spec/suffix_drafter.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_a525907265eb4cc7 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a525907265eb4cc7.o")
endif()
if(NOT TARGET obj_fd855d452b9de7f1)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fd855d452b9de7f1.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-O3" "-DNDEBUG" "-std=c++20" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/spec/controller.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fd855d452b9de7f1.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/spec/controller.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_fd855d452b9de7f1 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fd855d452b9de7f1.o")
endif()
if(NOT TARGET obj_adf96e3540107f42)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adf96e3540107f42.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-O3" "-DNDEBUG" "-std=c++20" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/spec/draft_policy.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adf96e3540107f42.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/spec/draft_policy.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_adf96e3540107f42 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adf96e3540107f42.o")
endif()
if(NOT TARGET obj_bae05c1ffd46bff2)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bae05c1ffd46bff2.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/dequant_s2.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bae05c1ffd46bff2.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/dequant_s2.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_bae05c1ffd46bff2 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bae05c1ffd46bff2.o")
endif()
if(NOT TARGET obj_784c0a0915e62797)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_784c0a0915e62797.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_784c0a0915e62797.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_784c0a0915e62797 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_784c0a0915e62797.o")
endif()
if(NOT TARGET obj_a5805b76b0e5245a)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a5805b76b0e5245a.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s_gemv.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a5805b76b0e5245a.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s_gemv.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_a5805b76b0e5245a DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a5805b76b0e5245a.o")
endif()
if(NOT TARGET obj_6e56f7f4dac80ad3)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6e56f7f4dac80ad3.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv_quads.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6e56f7f4dac80ad3.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv_quads.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6e56f7f4dac80ad3 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6e56f7f4dac80ad3.o")
endif()
if(NOT TARGET obj_004fa4cb94ac00e1)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_004fa4cb94ac00e1.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv_fast.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_004fa4cb94ac00e1.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv_fast.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_004fa4cb94ac00e1 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_004fa4cb94ac00e1.o")
endif()
if(NOT TARGET obj_8ea263582e98e249)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ea263582e98e249.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/router_top10.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ea263582e98e249.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/router_top10.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_8ea263582e98e249 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ea263582e98e249.o")
endif()
if(NOT TARGET obj_8840cfb59c0debd6)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8840cfb59c0debd6.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/quantize_act.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8840cfb59c0debd6.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/quantize_act.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_8840cfb59c0debd6 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8840cfb59c0debd6.o")
endif()
if(NOT TARGET obj_6feb227a7f656f71)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6feb227a7f656f71.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/rope.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6feb227a7f656f71.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/rope.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6feb227a7f656f71 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6feb227a7f656f71.o")
endif()
if(NOT TARGET obj_65aa00ea4810d568)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_65aa00ea4810d568.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/sampler.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_65aa00ea4810d568.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/sampler.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_65aa00ea4810d568 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_65aa00ea4810d568.o")
endif()
if(NOT TARGET obj_4589f81c2da502b6)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4589f81c2da502b6.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv_q8.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4589f81c2da502b6.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_gemv_q8.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_4589f81c2da502b6 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4589f81c2da502b6.o")
endif()
if(NOT TARGET obj_ab28f9930d6304ed)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ab28f9930d6304ed.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_expert_grouped.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ab28f9930d6304ed.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/s2_expert_grouped.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_ab28f9930d6304ed DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ab28f9930d6304ed.o")
endif()
if(NOT TARGET obj_2645c4eeda3695e0)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2645c4eeda3695e0.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/shared_expert.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2645c4eeda3695e0.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/shared_expert.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_2645c4eeda3695e0 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2645c4eeda3695e0.o")
endif()
if(NOT TARGET obj_73055b364d017cdf)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73055b364d017cdf.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/gr.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73055b364d017cdf.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/gr.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_73055b364d017cdf DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73055b364d017cdf.o")
endif()
if(NOT TARGET obj_33b1d88f55a67c96)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_33b1d88f55a67c96.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/gdn.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_33b1d88f55a67c96.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/gdn.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_33b1d88f55a67c96 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_33b1d88f55a67c96.o")
endif()
if(NOT TARGET obj_a40c8c3c0e1b4d1b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a40c8c3c0e1b4d1b.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/qsa.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a40c8c3c0e1b4d1b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/qsa.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_a40c8c3c0e1b4d1b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a40c8c3c0e1b4d1b.o")
endif()
if(NOT TARGET obj_adce9a8e9b1f4d28)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adce9a8e9b1f4d28.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/elementwise.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adce9a8e9b1f4d28.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/elementwise.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_adce9a8e9b1f4d28 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adce9a8e9b1f4d28.o")
endif()
if(NOT TARGET obj_777595a91b27ef5c)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_777595a91b27ef5c.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/bf16_gemv.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_777595a91b27ef5c.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/bf16_gemv.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_777595a91b27ef5c DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_777595a91b27ef5c.o")
endif()
if(NOT TARGET obj_a032d3f342144e7d)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a032d3f342144e7d.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_mmvq.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a032d3f342144e7d.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_mmvq.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_a032d3f342144e7d DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a032d3f342144e7d.o")
endif()
if(NOT TARGET obj_7b079f8a435f7b74)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7b079f8a435f7b74.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_bf16.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7b079f8a435f7b74.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_bf16.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_7b079f8a435f7b74 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7b079f8a435f7b74.o")
endif()
if(NOT TARGET obj_66a1ada8c8b5576a)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66a1ada8c8b5576a.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gr_norm.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66a1ada8c8b5576a.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gr_norm.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_66a1ada8c8b5576a DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66a1ada8c8b5576a.o")
endif()
if(NOT TARGET obj_690802c12108b2ad)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_690802c12108b2ad.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_moe.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_690802c12108b2ad.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_moe.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_690802c12108b2ad DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_690802c12108b2ad.o")
endif()
if(NOT TARGET obj_229273f0b958b189)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_229273f0b958b189.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gdn.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_229273f0b958b189.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gdn.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_229273f0b958b189 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_229273f0b958b189.o")
endif()
if(NOT TARGET obj_c4d7ee7d1a7aec65)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c4d7ee7d1a7aec65.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gr_postops.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c4d7ee7d1a7aec65.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gr_postops.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_c4d7ee7d1a7aec65 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c4d7ee7d1a7aec65.o")
endif()
if(NOT TARGET obj_5fcbdf03642567b7)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5fcbdf03642567b7.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_router.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5fcbdf03642567b7.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_router.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_5fcbdf03642567b7 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5fcbdf03642567b7.o")
endif()
if(NOT TARGET obj_f93f4af381d91290)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f93f4af381d91290.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gdn_preprocess.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f93f4af381d91290.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_gdn_preprocess.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_f93f4af381d91290 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f93f4af381d91290.o")
endif()
if(NOT TARGET obj_12cb75b0cbfee851)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_12cb75b0cbfee851.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_qsa.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_12cb75b0cbfee851.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_qsa.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_12cb75b0cbfee851 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_12cb75b0cbfee851.o")
endif()
if(NOT TARGET obj_b6c3fba60e9cfa9d)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b6c3fba60e9cfa9d.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_rope.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b6c3fba60e9cfa9d.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_rope.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_b6c3fba60e9cfa9d DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b6c3fba60e9cfa9d.o")
endif()
if(NOT TARGET obj_6be8c4776846a045)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6be8c4776846a045.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_ple_postops.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6be8c4776846a045.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_ple_postops.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6be8c4776846a045 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6be8c4776846a045.o")
endif()
if(NOT TARGET obj_6b4a215742520f42)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6b4a215742520f42.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_flash_attn.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6b4a215742520f42.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_flash_attn.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6b4a215742520f42 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6b4a215742520f42.o")
endif()
if(NOT TARGET obj_2888171f54901fb1)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2888171f54901fb1.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_qsa_indexer.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2888171f54901fb1.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_qsa_indexer.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_2888171f54901fb1 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2888171f54901fb1.o")
endif()
if(NOT TARGET obj_cb9762599067a6d7)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_cb9762599067a6d7.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_qsa_score.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_cb9762599067a6d7.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/native_qsa_score.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_cb9762599067a6d7 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_cb9762599067a6d7.o")
endif()
if(NOT TARGET obj_c1c4280d7b6f4493)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1c4280d7b6f4493.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/ple.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1c4280d7b6f4493.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/ple.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_c1c4280d7b6f4493 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1c4280d7b6f4493.o")
endif()
if(NOT TARGET obj_250470595f2c3664)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_250470595f2c3664.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/kv_q8.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_250470595f2c3664.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/kv_q8.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_250470595f2c3664 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_250470595f2c3664.o")
endif()
if(NOT TARGET obj_80753c4723890d8c)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80753c4723890d8c.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/kv_stream.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80753c4723890d8c.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/kv_stream.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_80753c4723890d8c DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80753c4723890d8c.o")
endif()
if(NOT TARGET obj_8fb0d11e872f936b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8fb0d11e872f936b.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/kv_q4.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8fb0d11e872f936b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/kv_q4.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_8fb0d11e872f936b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8fb0d11e872f936b.o")
endif()
if(NOT TARGET obj_c1fc16f0c00fb3c7)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1fc16f0c00fb3c7.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin" "/usr/bin/clang++-18" "-forward-unknown-to-host-compiler" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-Xptxas=-v" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v36/fused_gr.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1fc16f0c00fb3c7.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-flash-next-speed/source-v36/fused_gr.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_c1fc16f0c00fb3c7 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1fc16f0c00fb3c7.o")
endif()
if(NOT TARGET obj_1781b08aa1fdca2d)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1781b08aa1fdca2d.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/cvec.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1781b08aa1fdca2d.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/cvec.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_1781b08aa1fdca2d DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1781b08aa1fdca2d.o")
endif()
if(NOT TARGET obj_8c767f783a3822fa)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8c767f783a3822fa.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-forward-unknown-to-host-compiler" "--threads=1" "-DSTRATA_VERSION=\"0.1.39\"" "-USTRATA_QSA_COMPONENT_V328" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v179/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-Xptxas=-v" "-x" "cu" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v331/src/kernels/cuda/qsa_decode_attn.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8c767f783a3822fa.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v331/src/kernels/cuda/qsa_decode_attn.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_8c767f783a3822fa DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8c767f783a3822fa.o")
endif()
if(NOT TARGET obj_75b513c7b6d785b0)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_75b513c7b6d785b0.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/qsa_prompt_attn.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_75b513c7b6d785b0.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/qsa_prompt_attn.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_75b513c7b6d785b0 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_75b513c7b6d785b0.o")
endif()
if(NOT TARGET obj_1928fadaf19168e9)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1928fadaf19168e9.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/fused_gdn.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1928fadaf19168e9.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/fused_gdn.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_1928fadaf19168e9 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1928fadaf19168e9.o")
endif()
if(NOT TARGET obj_afb7a8be1c825eb4)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afb7a8be1c825eb4.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/dequant_bf16.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afb7a8be1c825eb4.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/dequant_bf16.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_afb7a8be1c825eb4 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afb7a8be1c825eb4.o")
endif()
if(NOT TARGET obj_d43c8cc37918e3e7)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d43c8cc37918e3e7.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/qsa_select.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d43c8cc37918e3e7.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/qsa_select.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_d43c8cc37918e3e7 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d43c8cc37918e3e7.o")
endif()
if(NOT TARGET obj_c2ab3119f48870cb)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2ab3119f48870cb.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/verify_kernels.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2ab3119f48870cb.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cuda/verify_kernels.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_c2ab3119f48870cb DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2ab3119f48870cb.o")
endif()
if(NOT TARGET obj_82aefbdaa7e88856)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_82aefbdaa7e88856.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/native" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/src/kernels/cuda/iq_kernels.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_82aefbdaa7e88856.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/src/kernels/cuda/iq_kernels.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_82aefbdaa7e88856 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_82aefbdaa7e88856.o")
endif()
if(NOT TARGET obj_8015c230a3ae090d)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8015c230a3ae090d.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/ngram.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8015c230a3ae090d.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/ngram.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_8015c230a3ae090d DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8015c230a3ae090d.o")
endif()
if(NOT TARGET obj_62bc032e59e531fa)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_62bc032e59e531fa.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/ngram/ple_reader.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_62bc032e59e531fa.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/ngram/ple_reader.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_62bc032e59e531fa DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_62bc032e59e531fa.o")
endif()
if(NOT TARGET obj_0508bc3f1c7e7c8c)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0508bc3f1c7e7c8c.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/platform/direct_file.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0508bc3f1c7e7c8c.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/platform/direct_file.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_0508bc3f1c7e7c8c DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0508bc3f1c7e7c8c.o")
endif()
if(NOT TARGET obj_0b5ebdc168571270)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0b5ebdc168571270.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-std=c++20" "-O3" "-arch=sm_89" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/native" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-Xptxas=-v" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/prefill-audit-r15/generated-r24b/swift_nvfp4_gpu_nc_r24b.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0b5ebdc168571270.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/prefill-audit-r15/generated-r24b/swift_nvfp4_gpu_nc_r24b.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_0b5ebdc168571270 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0b5ebdc168571270.o")
endif()
if(NOT TARGET obj_df8c4aa08f86d0eb)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_df8c4aa08f86d0eb.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/device.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_df8c4aa08f86d0eb.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/device.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_df8c4aa08f86d0eb DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_df8c4aa08f86d0eb.o")
endif()
if(NOT TARGET obj_e6133a3c0262c615)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6133a3c0262c615.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/pinned.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6133a3c0262c615.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/pinned.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_e6133a3c0262c615 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6133a3c0262c615.o")
endif()
if(NOT TARGET obj_afd5eb269545c2f3)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afd5eb269545c2f3.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/platform/memory.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afd5eb269545c2f3.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/platform/memory.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_afd5eb269545c2f3 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afd5eb269545c2f3.o")
endif()
if(NOT TARGET obj_883f1e9190415cb0)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_883f1e9190415cb0.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/graph.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_883f1e9190415cb0.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/graph.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_883f1e9190415cb0 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_883f1e9190415cb0.o")
endif()
if(NOT TARGET obj_227567953600b900)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_227567953600b900.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/weights.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_227567953600b900.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/weights.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_227567953600b900 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_227567953600b900.o")
endif()
if(NOT TARGET obj_54a87d1e65493190)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_54a87d1e65493190.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "-Wall" "-Wextra" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/layout.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_54a87d1e65493190.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/core/layout.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_54a87d1e65493190 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_54a87d1e65493190.o")
endif()
if(NOT TARGET obj_ed82e761982d1916)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ed82e761982d1916.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++20" "-mavx512f" "-mavx512bw" "-mavx512vl" "-mavx512dq" "-mavx512vnni" "-mavx512vbmi" "-mf16c" "-mfma" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/expert.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ed82e761982d1916.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/expert.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_ed82e761982d1916 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ed82e761982d1916.o")
endif()
if(NOT TARGET obj_97bed4c20f0e2089)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_97bed4c20f0e2089.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-O3" "-DNDEBUG" "-std=c++20" "-DSTRATA_NATIVE_EXPERTS=1" "-fPIC" "-fopenmp=libgomp" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/source-v293/src/kernels/cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/native/optimized/pool-nvfp4-r10.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_97bed4c20f0e2089.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/native/optimized/pool-nvfp4-r10.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_97bed4c20f0e2089 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_97bed4c20f0e2089.o")
endif()
if(NOT TARGET obj_427f67ab524b0629)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_427f67ab524b0629.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++20" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/expert_layout.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_427f67ab524b0629.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/expert_layout.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_427f67ab524b0629 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_427f67ab524b0629.o")
endif()
if(NOT TARGET obj_90ec546a97009b60)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90ec546a97009b60.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++20" "-mavx2" "-mfma" "-mf16c" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/q2_avx2.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90ec546a97009b60.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/q2_avx2.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_90ec546a97009b60 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90ec546a97009b60.o")
endif()
if(NOT TARGET obj_eaa8613e523371cd)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_eaa8613e523371cd.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-std=c++20" "-O3" "-fPIC" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/src/kernels/cpu/native_expert.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_eaa8613e523371cd.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/src/kernels/cpu/native_expert.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_eaa8613e523371cd DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_eaa8613e523371cd.o")
endif()
if(NOT TARGET obj_624b57f60dad7406)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_624b57f60dad7406.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++20" "-mavx512f" "-mavx512bw" "-mavx512vl" "-mavx512dq" "-mavx512vnni" "-mavx512vbmi" "-mf16c" "-mfma" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/iq_avx512.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_624b57f60dad7406.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/iq_avx512.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_624b57f60dad7406 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_624b57f60dad7406.o")
endif()
if(NOT TARGET obj_69bb32a6f5395c3b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_69bb32a6f5395c3b.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++20" "-mavx2" "-mfma" "-mf16c" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/iq_avx2.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_69bb32a6f5395c3b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/iq_avx2.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_69bb32a6f5395c3b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_69bb32a6f5395c3b.o")
endif()
if(NOT TARGET obj_930b14dc3adcb6e7)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_930b14dc3adcb6e7.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++20" "-mavx2" "-mfma" "-mf16c" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/kq_avx2.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_930b14dc3adcb6e7.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/kq_avx2.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_930b14dc3adcb6e7 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_930b14dc3adcb6e7.o")
endif()
if(NOT TARGET obj_b7f9ba0565a7b7c3)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b7f9ba0565a7b7c3.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DSTRATA_NATIVE_EXPERTS=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/third_party/ggml" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++20" "-mavx" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/kq_avx1.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b7f9ba0565a7b7c3.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/kernels/cpu/kq_avx1.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_b7f9ba0565a7b7c3 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b7f9ba0565a7b7c3.o")
endif()
if(NOT TARGET obj_434f672df1a7c922)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_434f672df1a7c922.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-std=c++20" "-O3" "-DSTRATA_NATIVE_EXPERTS=1" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-mavx512f" "-mavx512bw" "-mavx512vl" "-mavx512dq" "-mavx512vnni" "-mfma" "-fPIC" "-mavx512vbmi" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/build/optimized-nvfp4-vbmi-r8/vbmi-r6.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_434f672df1a7c922.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/build/optimized-nvfp4-vbmi-r8/vbmi-r6.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_434f672df1a7c922 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_434f672df1a7c922.o")
endif()
if(NOT TARGET obj_c2aa030eda8b6355)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2aa030eda8b6355.o"
  COMMAND "${CMAKE_C_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=gnu11" "-Wshadow" "-Wstrict-prototypes" "-Wpointer-arith" "-Wmissing-prototypes" "-Werror=implicit-int" "-Werror=implicit-function-declaration" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wdouble-promotion" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/ggml-cpu.c" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2aa030eda8b6355.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/ggml-cpu.c" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_c2aa030eda8b6355 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2aa030eda8b6355.o")
endif()
if(NOT TARGET obj_f31ce91afa54b227)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f31ce91afa54b227.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/ggml-cpu.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f31ce91afa54b227.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/ggml-cpu.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_f31ce91afa54b227 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f31ce91afa54b227.o")
endif()
if(NOT TARGET obj_b984c183ecb61c3e)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b984c183ecb61c3e.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/repack.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b984c183ecb61c3e.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/repack.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_b984c183ecb61c3e DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b984c183ecb61c3e.o")
endif()
if(NOT TARGET obj_fb5e64586bfe298f)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb5e64586bfe298f.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/iqp.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb5e64586bfe298f.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/iqp.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_fb5e64586bfe298f DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb5e64586bfe298f.o")
endif()
if(NOT TARGET obj_3624b1d208cbc3a2)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3624b1d208cbc3a2.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/hbm.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3624b1d208cbc3a2.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/hbm.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_3624b1d208cbc3a2 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3624b1d208cbc3a2.o")
endif()
if(NOT TARGET obj_6ee4364dae50380f)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6ee4364dae50380f.o"
  COMMAND "${CMAKE_C_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=gnu11" "-Wshadow" "-Wstrict-prototypes" "-Wpointer-arith" "-Wmissing-prototypes" "-Werror=implicit-int" "-Werror=implicit-function-declaration" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wdouble-promotion" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/quants.c" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6ee4364dae50380f.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/quants.c" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6ee4364dae50380f DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6ee4364dae50380f.o")
endif()
if(NOT TARGET obj_e6c94be1edf5d9d1)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6c94be1edf5d9d1.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/traits.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6c94be1edf5d9d1.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/traits.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_e6c94be1edf5d9d1 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6c94be1edf5d9d1.o")
endif()
if(NOT TARGET obj_a4a02ea980ca353f)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4a02ea980ca353f.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/amx/amx.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4a02ea980ca353f.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/amx/amx.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_a4a02ea980ca353f DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4a02ea980ca353f.o")
endif()
if(NOT TARGET obj_63f28da62cef2fb5)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_63f28da62cef2fb5.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/amx/mmq.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_63f28da62cef2fb5.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/amx/mmq.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_63f28da62cef2fb5 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_63f28da62cef2fb5.o")
endif()
if(NOT TARGET obj_80ffe1014a4e7c8b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80ffe1014a4e7c8b.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/binary-ops.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80ffe1014a4e7c8b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/binary-ops.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_80ffe1014a4e7c8b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80ffe1014a4e7c8b.o")
endif()
if(NOT TARGET obj_52c5a1fd758a390d)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_52c5a1fd758a390d.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/unary-ops.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_52c5a1fd758a390d.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/unary-ops.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_52c5a1fd758a390d DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_52c5a1fd758a390d.o")
endif()
if(NOT TARGET obj_7ea71bed76624554)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ea71bed76624554.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/vec.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ea71bed76624554.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/vec.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_7ea71bed76624554 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ea71bed76624554.o")
endif()
if(NOT TARGET obj_b17a1c7bc66a205f)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b17a1c7bc66a205f.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/ops.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b17a1c7bc66a205f.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/ops.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_b17a1c7bc66a205f DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b17a1c7bc66a205f.o")
endif()
if(NOT TARGET obj_966e9b237c674fc6)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_966e9b237c674fc6.o"
  COMMAND "${CMAKE_C_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=gnu11" "-Wshadow" "-Wstrict-prototypes" "-Wpointer-arith" "-Wmissing-prototypes" "-Werror=implicit-int" "-Werror=implicit-function-declaration" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wdouble-promotion" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/arch/x86/quants.c" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_966e9b237c674fc6.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/arch/x86/quants.c" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_966e9b237c674fc6 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_966e9b237c674fc6.o")
endif()
if(NOT TARGET obj_ef29e4a45bdc3d16)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ef29e4a45bdc3d16.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DGGML_USE_CPU_REPACK" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/.." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" "-march=native" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/arch/x86/repack.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ef29e4a45bdc3d16.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cpu/arch/x86/repack.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_ef29e4a45bdc3d16 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ef29e4a45bdc3d16.o")
endif()
if(NOT TARGET obj_2d966953a446d201)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2d966953a446d201.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/prefill-audit-r15/generated-r20/moe-mmq-dm-workspace-r20.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2d966953a446d201.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/research/prefill-audit-r15/generated-r20/moe-mmq-dm-workspace-r20.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_2d966953a446d201 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2d966953a446d201.o")
endif()
if(NOT TARGET obj_07b58b10606e8a49)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07b58b10606e8a49.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/moe_fused.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07b58b10606e8a49.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/moe_fused.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_07b58b10606e8a49 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07b58b10606e8a49.o")
endif()
if(NOT TARGET obj_7111a609c9d0e81c)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7111a609c9d0e81c.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/moe_fused_iq.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7111a609c9d0e81c.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/moe_fused_iq.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_7111a609c9d0e81c DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7111a609c9d0e81c.o")
endif()
if(NOT TARGET obj_d56b8f5d77c0ef8b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d56b8f5d77c0ef8b.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/ggml_cuda_host.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d56b8f5d77c0ef8b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/src/prefill/ggml_cuda_host.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_d56b8f5d77c0ef8b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d56b8f5d77c0ef8b.o")
endif()
if(NOT TARGET obj_871ba4fee76088c5)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_871ba4fee76088c5.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/quantize.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_871ba4fee76088c5.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/quantize.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_871ba4fee76088c5 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_871ba4fee76088c5.o")
endif()
if(NOT TARGET obj_4f5232d89facda14)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4f5232d89facda14.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4f5232d89facda14.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_4f5232d89facda14 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4f5232d89facda14.o")
endif()
if(NOT TARGET obj_90bac3c2c6b922b4)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90bac3c2c6b922b4.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90bac3c2c6b922b4.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_90bac3c2c6b922b4 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90bac3c2c6b922b4.o")
endif()
if(NOT TARGET obj_7d9ca4b7a42369d1)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7d9ca4b7a42369d1.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7d9ca4b7a42369d1.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_7d9ca4b7a42369d1 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7d9ca4b7a42369d1.o")
endif()
if(NOT TARGET obj_691d625fcb464184)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_691d625fcb464184.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_691d625fcb464184.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_691d625fcb464184 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_691d625fcb464184.o")
endif()
if(NOT TARGET obj_6a557dea7e6f0a2a)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6a557dea7e6f0a2a.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6a557dea7e6f0a2a.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6a557dea7e6f0a2a DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6a557dea7e6f0a2a.o")
endif()
if(NOT TARGET obj_99658bcbf6712024)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_99658bcbf6712024.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_99658bcbf6712024.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_99658bcbf6712024 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_99658bcbf6712024.o")
endif()
if(NOT TARGET obj_bf22084487eb01b9)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bf22084487eb01b9.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bf22084487eb01b9.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_bf22084487eb01b9 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bf22084487eb01b9.o")
endif()
if(NOT TARGET obj_875c6fe7c738d04b)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_875c6fe7c738d04b.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_875c6fe7c738d04b.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_875c6fe7c738d04b DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_875c6fe7c738d04b.o")
endif()
if(NOT TARGET obj_8ab0cbc76c6a04a2)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ab0cbc76c6a04a2.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ab0cbc76c6a04a2.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_8ab0cbc76c6a04a2 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ab0cbc76c6a04a2.o")
endif()
if(NOT TARGET obj_b4384bcc63b5c762)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b4384bcc63b5c762.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b4384bcc63b5c762.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_b4384bcc63b5c762 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b4384bcc63b5c762.o")
endif()
if(NOT TARGET obj_b89eca6252589efd)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b89eca6252589efd.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b89eca6252589efd.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_b89eca6252589efd DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b89eca6252589efd.o")
endif()
if(NOT TARGET obj_07f1470b6e90e8bc)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07f1470b6e90e8bc.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07f1470b6e90e8bc.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_07f1470b6e90e8bc DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07f1470b6e90e8bc.o")
endif()
if(NOT TARGET obj_9f8a07cb6ca90d6f)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_9f8a07cb6ca90d6f.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-DSTRATA_VERSION=\"0.1.39\"" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/Strata/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-isystem" "${CUDAToolkit_INCLUDE_DIR}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_9f8a07cb6ca90d6f.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_9f8a07cb6ca90d6f DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_9f8a07cb6ca90d6f.o")
endif()
if(NOT TARGET obj_896cfcdc94e0f18e)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_896cfcdc94e0f18e.o"
  COMMAND "${CMAKE_CUDA_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-ccbin=${CMAKE_CUDA_HOST_COMPILER}" "-O3" "-DNDEBUG" "-std=c++20" "--generate-code=arch=compute_89,code=[compute_89,sm_89]" "--extended-lambda" "-use_fast_math" "-DSTRATA_MMQ_KQUANTS=1" "-DSTRATA_ORCA_Q4KS_MMQ=1" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Swift 1.5 Flash Next/work/swift-nvfp4-compare/backend/Strata-NVFP4/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_896cfcdc94e0f18e.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_896cfcdc94e0f18e DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_896cfcdc94e0f18e.o")
endif()
if(NOT TARGET obj_6259e313a9c096e3)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6259e313a9c096e3.o"
  COMMAND "${CMAKE_C_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=gnu11" "-Wshadow" "-Wstrict-prototypes" "-Wpointer-arith" "-Wmissing-prototypes" "-Werror=implicit-int" "-Werror=implicit-function-declaration" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wdouble-promotion" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml.c" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6259e313a9c096e3.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml.c" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_6259e313a9c096e3 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6259e313a9c096e3.o")
endif()
if(NOT TARGET obj_3f5a66c7b09e927e)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3f5a66c7b09e927e.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3f5a66c7b09e927e.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_3f5a66c7b09e927e DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3f5a66c7b09e927e.o")
endif()
if(NOT TARGET obj_f480bbf376df6289)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f480bbf376df6289.o"
  COMMAND "${CMAKE_C_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=gnu11" "-Wshadow" "-Wstrict-prototypes" "-Wpointer-arith" "-Wmissing-prototypes" "-Werror=implicit-int" "-Werror=implicit-function-declaration" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wdouble-promotion" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-alloc.c" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f480bbf376df6289.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-alloc.c" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_f480bbf376df6289 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f480bbf376df6289.o")
endif()
if(NOT TARGET obj_be4326618e1c0628)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_be4326618e1c0628.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-backend.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_be4326618e1c0628.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-backend.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_be4326618e1c0628 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_be4326618e1c0628.o")
endif()
if(NOT TARGET obj_dd4d839c447ad47c)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_dd4d839c447ad47c.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-backend-meta.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_dd4d839c447ad47c.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-backend-meta.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_dd4d839c447ad47c DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_dd4d839c447ad47c.o")
endif()
if(NOT TARGET obj_13e970f2e8e1a802)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_13e970f2e8e1a802.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-opt.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_13e970f2e8e1a802.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-opt.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_13e970f2e8e1a802 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_13e970f2e8e1a802.o")
endif()
if(NOT TARGET obj_d71acff813915c17)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d71acff813915c17.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-threading.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d71acff813915c17.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-threading.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_d71acff813915c17 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d71acff813915c17.o")
endif()
if(NOT TARGET obj_ae0c4079936ffcad)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ae0c4079936ffcad.o"
  COMMAND "${CMAKE_C_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=gnu11" "-Wshadow" "-Wstrict-prototypes" "-Wpointer-arith" "-Wmissing-prototypes" "-Werror=implicit-int" "-Werror=implicit-function-declaration" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wdouble-promotion" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-quants.c" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ae0c4079936ffcad.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/ggml-quants.c" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_ae0c4079936ffcad DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ae0c4079936ffcad.o")
endif()
if(NOT TARGET obj_c5cefc9ef260ef77)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c5cefc9ef260ef77.o"
  COMMAND "${CMAKE_CXX_COMPILER}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-DGGML_SCHED_MAX_COPIES=4" "-DSTRATA_VERSION=\"0.1.39\"" "-D_GNU_SOURCE" "-D_XOPEN_SOURCE=600" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/." "-I${CMAKE_CURRENT_SOURCE_DIR}/source/RVN Qwen3.8 Flash Next/work/qwenflash-next-speed/build-v35-mtp-q4/ggml/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/../include" "-O3" "-DNDEBUG" "-std=c++17" "-Wmissing-declarations" "-Wmissing-noreturn" "-Wall" "-Wextra" "-Wpedantic" "-Wcast-qual" "-Wno-unused-function" "-Wno-array-bounds" "-Wextra-semi" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/gguf.cpp" -o "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c5cefc9ef260ef77.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/Shared/work/strata-llama/ggml/src/gguf.cpp" ${STRATA_PROFILE_HEADERS}
  VERBATIM)
add_custom_target(obj_c5cefc9ef260ef77 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c5cefc9ef260ef77.o")
endif()
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_0.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_0.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_0.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e4dc16ac4ee5c890.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d09bd2860b0f4f86.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1667155e707512ca.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_34a95da22d3f5a1b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6566a7367dccc966.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d1161d8c0474dc1a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7947b0c4a524fbba.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2890fdcafb327260.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_17de87d99a19f000.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f611040249752996.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_556ba9dd8db975fc.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_29eb60798b053a70.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_670c4a216496a57f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4cb6ae88a859449.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_0.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e4dc16ac4ee5c890.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d09bd2860b0f4f86.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1667155e707512ca.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_34a95da22d3f5a1b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6566a7367dccc966.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d1161d8c0474dc1a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7947b0c4a524fbba.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2890fdcafb327260.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_17de87d99a19f000.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f611040249752996.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_556ba9dd8db975fc.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_29eb60798b053a70.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_670c4a216496a57f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4cb6ae88a859449.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_1.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_1.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_1.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_810d0f3de71b0550.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_76e8079cf3b53f61.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5d560c571fa7876c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b348dedbbe0acb3f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1575e44ccc7babe5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ce37f4a71ca1a5b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_224ba431e03b42c6.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_1.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_810d0f3de71b0550.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_76e8079cf3b53f61.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5d560c571fa7876c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b348dedbbe0acb3f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1575e44ccc7babe5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ce37f4a71ca1a5b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_224ba431e03b42c6.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_2.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_2.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_2.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a525907265eb4cc7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fd855d452b9de7f1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adf96e3540107f42.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_2.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a525907265eb4cc7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fd855d452b9de7f1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adf96e3540107f42.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_3.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_3.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_3.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bae05c1ffd46bff2.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_784c0a0915e62797.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a5805b76b0e5245a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6e56f7f4dac80ad3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_004fa4cb94ac00e1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ea263582e98e249.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8840cfb59c0debd6.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6feb227a7f656f71.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_65aa00ea4810d568.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4589f81c2da502b6.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ab28f9930d6304ed.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2645c4eeda3695e0.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73055b364d017cdf.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_33b1d88f55a67c96.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a40c8c3c0e1b4d1b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adce9a8e9b1f4d28.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_777595a91b27ef5c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a032d3f342144e7d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7b079f8a435f7b74.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66a1ada8c8b5576a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_690802c12108b2ad.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_229273f0b958b189.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c4d7ee7d1a7aec65.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5fcbdf03642567b7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f93f4af381d91290.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_12cb75b0cbfee851.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b6c3fba60e9cfa9d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6be8c4776846a045.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6b4a215742520f42.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2888171f54901fb1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_cb9762599067a6d7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1c4280d7b6f4493.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_250470595f2c3664.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80753c4723890d8c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8fb0d11e872f936b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1fc16f0c00fb3c7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1781b08aa1fdca2d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8c767f783a3822fa.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_75b513c7b6d785b0.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1928fadaf19168e9.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afb7a8be1c825eb4.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d43c8cc37918e3e7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2ab3119f48870cb.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_82aefbdaa7e88856.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8015c230a3ae090d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_62bc032e59e531fa.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0508bc3f1c7e7c8c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0b5ebdc168571270.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_3.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bae05c1ffd46bff2.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_784c0a0915e62797.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a5805b76b0e5245a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6e56f7f4dac80ad3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_004fa4cb94ac00e1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ea263582e98e249.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8840cfb59c0debd6.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6feb227a7f656f71.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_65aa00ea4810d568.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4589f81c2da502b6.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ab28f9930d6304ed.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2645c4eeda3695e0.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73055b364d017cdf.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_33b1d88f55a67c96.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a40c8c3c0e1b4d1b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_adce9a8e9b1f4d28.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_777595a91b27ef5c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a032d3f342144e7d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7b079f8a435f7b74.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66a1ada8c8b5576a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_690802c12108b2ad.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_229273f0b958b189.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c4d7ee7d1a7aec65.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_5fcbdf03642567b7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f93f4af381d91290.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_12cb75b0cbfee851.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b6c3fba60e9cfa9d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6be8c4776846a045.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6b4a215742520f42.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2888171f54901fb1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_cb9762599067a6d7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1c4280d7b6f4493.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_250470595f2c3664.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80753c4723890d8c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8fb0d11e872f936b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c1fc16f0c00fb3c7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1781b08aa1fdca2d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8c767f783a3822fa.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_75b513c7b6d785b0.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1928fadaf19168e9.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afb7a8be1c825eb4.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d43c8cc37918e3e7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2ab3119f48870cb.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_82aefbdaa7e88856.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8015c230a3ae090d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_62bc032e59e531fa.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0508bc3f1c7e7c8c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_0b5ebdc168571270.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_4.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_4.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_4.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_df8c4aa08f86d0eb.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6133a3c0262c615.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afd5eb269545c2f3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_883f1e9190415cb0.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_227567953600b900.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_54a87d1e65493190.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_4.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_df8c4aa08f86d0eb.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6133a3c0262c615.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_afd5eb269545c2f3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_883f1e9190415cb0.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_227567953600b900.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_54a87d1e65493190.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_5.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_5.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_5.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ed82e761982d1916.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_97bed4c20f0e2089.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_427f67ab524b0629.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90ec546a97009b60.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_eaa8613e523371cd.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_624b57f60dad7406.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_69bb32a6f5395c3b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_930b14dc3adcb6e7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b7f9ba0565a7b7c3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_434f672df1a7c922.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_5.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ed82e761982d1916.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_97bed4c20f0e2089.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_427f67ab524b0629.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90ec546a97009b60.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_eaa8613e523371cd.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_624b57f60dad7406.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_69bb32a6f5395c3b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_930b14dc3adcb6e7.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b7f9ba0565a7b7c3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_434f672df1a7c922.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_6.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_6.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_6.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2aa030eda8b6355.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f31ce91afa54b227.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b984c183ecb61c3e.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb5e64586bfe298f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3624b1d208cbc3a2.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6ee4364dae50380f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6c94be1edf5d9d1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4a02ea980ca353f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_63f28da62cef2fb5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80ffe1014a4e7c8b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_52c5a1fd758a390d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ea71bed76624554.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b17a1c7bc66a205f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_966e9b237c674fc6.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ef29e4a45bdc3d16.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_6.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c2aa030eda8b6355.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f31ce91afa54b227.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b984c183ecb61c3e.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb5e64586bfe298f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3624b1d208cbc3a2.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6ee4364dae50380f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_e6c94be1edf5d9d1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_a4a02ea980ca353f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_63f28da62cef2fb5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_80ffe1014a4e7c8b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_52c5a1fd758a390d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7ea71bed76624554.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b17a1c7bc66a205f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_966e9b237c674fc6.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ef29e4a45bdc3d16.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_7.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_7.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_7.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2d966953a446d201.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07b58b10606e8a49.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7111a609c9d0e81c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d56b8f5d77c0ef8b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_871ba4fee76088c5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4f5232d89facda14.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90bac3c2c6b922b4.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7d9ca4b7a42369d1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_691d625fcb464184.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6a557dea7e6f0a2a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_99658bcbf6712024.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bf22084487eb01b9.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_875c6fe7c738d04b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ab0cbc76c6a04a2.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b4384bcc63b5c762.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b89eca6252589efd.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07f1470b6e90e8bc.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_9f8a07cb6ca90d6f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_896cfcdc94e0f18e.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_7.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_2d966953a446d201.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07b58b10606e8a49.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7111a609c9d0e81c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d56b8f5d77c0ef8b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_871ba4fee76088c5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4f5232d89facda14.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_90bac3c2c6b922b4.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_7d9ca4b7a42369d1.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_691d625fcb464184.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6a557dea7e6f0a2a.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_99658bcbf6712024.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_bf22084487eb01b9.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_875c6fe7c738d04b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_8ab0cbc76c6a04a2.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b4384bcc63b5c762.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_b89eca6252589efd.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_07f1470b6e90e8bc.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_9f8a07cb6ca90d6f.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_896cfcdc94e0f18e.o"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_8.a"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_8.a"
  COMMAND "${STRATA_AR}" qc "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_8.a" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6259e313a9c096e3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3f5a66c7b09e927e.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f480bbf376df6289.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_be4326618e1c0628.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_dd4d839c447ad47c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_13e970f2e8e1a802.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d71acff813915c17.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ae0c4079936ffcad.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c5cefc9ef260ef77.o"
  COMMAND "${STRATA_AR}" s "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_8.a"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_6259e313a9c096e3.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_3f5a66c7b09e927e.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f480bbf376df6289.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_be4326618e1c0628.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_dd4d839c447ad47c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_13e970f2e8e1a802.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d71acff813915c17.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ae0c4079936ffcad.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_c5cefc9ef260ef77.o"
  VERBATIM)
set(STRATA_PROFILE_BINARY "${CMAKE_CURRENT_BINARY_DIR}/strata-swift1-1024k")
add_custom_command(OUTPUT "${STRATA_PROFILE_BINARY}"
  COMMAND "${CMAKE_CXX_COMPILER}" "-std=c++20" "-I${CUDAToolkit_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include" "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-Wl,--wrap=_ZN6strata7kernels23qsa_block_topk_parallelEPKfPKilllRKNS0_9QsaShapesEPiPvmS9_" "-fuse-ld=mold" "-O3" "-DNDEBUG" "-Wl,--wrap=cudaGraphInstantiate" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66c329dca7782a1d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d2e9d16459e0b99e.o" "-Wl,--wrap=_ZN6strata7kernels14qsa_block_topkEPKfPKilllRKNS0_9QsaShapesEPiPvl" "-Wl,--wrap=_ZN6strata7prefill7Prefill3runEPKlllRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73a65df92ca17d32.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4d3bedb7534cca4b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1875396d27a5c9e8.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f8f8cf29d7aeaee5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fcd7cf0176c16fed.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb2c2cb14cbbfa2c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ca8a98bc3224bdfd.o" "-L${CUDAToolkit_LIBRARY_DIR}/stubs" "-L${CUDAToolkit_LIBRARY_DIR}" "-Wl,-rpath,${CUDAToolkit_LIBRARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_0.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_1.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_2.a" "${CUDAToolkit_LIBRARY_DIR}/libcudart.so" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_0.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_3.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_4.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_5.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_6.a" "${CUDAToolkit_LIBRARY_DIR}/libcublas.so" "${CUDAToolkit_LIBRARY_DIR}/libcublasLt.so" "${CUDAToolkit_LIBRARY_DIR}/libculibos.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_7.a" "${CUDAToolkit_LIBRARY_DIR}/libcudart.so" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_8.a" "-lm" "-lcudadevrt" "-lcudart_static" "-lrt" "-lpthread" "-ldl" "${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/required_native_peer.cpp" -o "${STRATA_PROFILE_BINARY}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/required_native_peer.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../../include/strata/core/required_native_peer.hpp" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_66c329dca7782a1d.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_d2e9d16459e0b99e.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_73a65df92ca17d32.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_4d3bedb7534cca4b.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_1875396d27a5c9e8.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_f8f8cf29d7aeaee5.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fcd7cf0176c16fed.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_fb2c2cb14cbbfa2c.o" "${CMAKE_CURRENT_BINARY_DIR}/objects/obj_ca8a98bc3224bdfd.o" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_0.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_1.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_2.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_3.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_4.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_5.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_6.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_7.a" "${CMAKE_CURRENT_BINARY_DIR}/archives/swift1-1024k/libprofile_8.a"
  VERBATIM)
add_custom_target(strata_swift1_1024k ALL DEPENDS "${STRATA_PROFILE_BINARY}")

set(STRATA_SWIFT_SWIFT1_1024K_BINARY "${STRATA_PROFILE_BINARY}")

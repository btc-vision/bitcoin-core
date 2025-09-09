# Copyright (c) 2024-present The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://opensource.org/license/mit/.

if(ENABLE_GPU_ACCELERATION)
  add_library(bitcoin_kernel_gpu STATIC
    gpu_test.cu
    gpu_test.h
    gpu_logging.cpp
    gpu_mining.cu
    gpu_mining.h
  )
  
  target_include_directories(bitcoin_kernel_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  
  set_target_properties(bitcoin_kernel_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )
  
  target_compile_options(bitcoin_kernel_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  
  target_link_libraries(bitcoin_kernel_gpu PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
  )
endif()
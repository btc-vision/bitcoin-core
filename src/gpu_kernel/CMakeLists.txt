# Copyright (c) 2024-present The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://opensource.org/license/mit/.

if(ENABLE_GPU_ACCELERATION)
  add_library(bitcoin_kernel_gpu STATIC
    # Core GPU utilities
    gpu_logging.cpp
    gpu_hash.cuh
    gpu_types.h
    gpu_utils.h

    # UTXO set management
    gpu_utxo.cu
    gpu_utxo.h
    gpu_cuckoo.cu
    gpu_utxo_loader.cu
    gpu_utxo_loader_wrapper.cpp
    gpu_utxo_compact.cu
    gpu_utxo_wrapper.cpp

    # GPUDirect Storage (direct GPU-to-disk I/O)
    gpu_direct_storage.h
    gpu_direct_storage.cu

    # Script execution (Phase 1+)
    gpu_script_types.cuh
    gpu_script_stack.cuh
    gpu_script_num.cuh

    # Phase 2: Opcodes and interpreter
    gpu_opcodes.cuh
    gpu_opcodes_crypto.cuh
    gpu_eval_script.cuh

    # Phase 3: secp256k1 elliptic curve cryptography
    gpu_secp256k1_field.cuh
    gpu_secp256k1_scalar.cuh
    gpu_secp256k1_group.cuh
    gpu_secp256k1_ecmult.cuh
    gpu_ecdsa_verify.cuh
    gpu_schnorr_verify.cuh

    # Phase 4: Sighash computation
    gpu_sighash.cuh

    # Phase 5: Signature operations integration
    gpu_opcodes_sig.cuh

    # Phase 6: Batch validation
    gpu_batch_validator.h
    gpu_batch_validator.cu

    # Phase 7: Transaction dependency graph for parallel batching
    gpu_tx_graph.h
    gpu_tx_graph.cpp

    # Test helpers
    gpu_hash_test_kernels.cu
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
  
  set_target_properties(bitcoin_kernel_gpu PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
  
  target_compile_options(bitcoin_kernel_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--threads 0 --expt-relaxed-constexpr -UNDEBUG --std=c++20>
  )
  
  # Find cuFile library for GPUDirect Storage
  find_library(CUFILE_LIBRARY cufile
    HINTS ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
  )

  target_link_libraries(bitcoin_kernel_gpu PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
  )

  # Optionally link cuFile if available
  if(CUFILE_LIBRARY)
    message(STATUS "Found cuFile library: ${CUFILE_LIBRARY}")
    target_link_libraries(bitcoin_kernel_gpu PUBLIC ${CUFILE_LIBRARY})
    target_compile_definitions(bitcoin_kernel_gpu PUBLIC HAVE_CUFILE=1)
  else()
    message(STATUS "cuFile library not found - GPUDirect Storage will use fallback mode")
  endif()
endif()
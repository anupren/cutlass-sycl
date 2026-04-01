# Copyright (c) 2025 - 2026 INTEL CORPORATION. All rights reserved.
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if(CUTLASS_ENABLE_SYCL)
  if(SYCL_INTEL_TARGET)
    cutlass_test_unit_add_executable(
      cutlass_test_unit_gemm_device_tensorop_xe
      xe_gemm_bf16_bf16_bf16_tensor_op_bf16.cpp
      xe_gemm_fp16_fp16_fp16_tensor_op_fp16.cpp
      xe_gemm_bf16_bf16_bf16_tensor_op_fp32.cpp
      xe_gemm_bf16_bf16_fp32_tensor_op_bf16.cpp
      xe_gemm_bf16_bf16_fp32_tensor_op_fp32.cpp
      xe_gemm_fp16_fp16_fp16_tensor_op_fp32.cpp
      xe_gemm_fp16_fp16_fp32_tensor_op_fp32.cpp
      xe_gemm_s8_s8_s32_tensor_op_s32.cpp
      xe_gemm_tf32_tf32_fp32_tensor_op_fp32.cpp
      xe_gemm_f8_f8_fp32_tensor_op_fp32.cpp
      xe_gemm_fp16_s8_fp32_tensor_op_fp32.cpp
      gemm_universal_bf16n_bf16t_f32n_tensor_op_f32_xe.cpp
      gemm_universal_fp8_fp8_fp32_tensor_op_f32_xe_models.cpp
    )

    cutlass_test_unit_add_executable(
      cutlass_test_unit_gemm_device_tensorop_cooperative_xe
      xe_gemm_bf16_bf16_fp32_tensor_op_fp32_cooperative.cpp
      xe_gemm_fp16_fp16_fp32_tensor_op_fp32_cooperative.cpp
      # xe_gemm_fp16_fp16_f32_ptr_array_cooperative.cpp
      # TODO (Codeplay): fix gemm cooperative tests for s8 and tf32
      # xe_gemm_s8_s8_s32_tensor_op_s32_cooperative.cpp
      # xe_gemm_tf32_tf32_fp32_tensor_op_fp32_cooperative.cpp
    )

    cutlass_test_unit_add_executable(
      cutlass_test_unit_gemm_device_tensorop_epilogue_fusion_xe
      xe_gemm_bf16_bf16_fp32_tensor_op_fp32_evt.cpp
    )

    cutlass_test_unit_add_executable(
      cutlass_test_unit_gemm_device_mixed_input_tensorop_xe
      gemm_universal_s8t_bf16n_f32t_mixed_input_tensor_op_f32_xe.cpp
      gemm_universal_f16t_s4t_f32t_mixed_input_tensor_op_f32_xe.cpp
      gemm_universal_f16t_s4n_f32t_mixed_input_tensor_op_f32_xe.cpp
    )

    # Group Gemm test
    cutlass_test_unit_add_executable(
      cutlass_test_unit_gemm_device_tensorop_xe_group_gemm
      xe_gemm_bf16_bf16_fp32_tensor_op_fp32_group_gemm.cpp
      xe_gemm_bf16_bf16_bf16_tensor_op_fp32_group_gemm.cpp
      xe_gemm_fp16_fp16_fp32_tensor_op_fp32_group_gemm.cpp
      xe_gemm_fp16_fp16_fp16_tensor_op_fp32_group_gemm.cpp
    )

    cutlass_test_unit_add_executable(
      cutlass_test_unit_gemm_device_mixed_dtype_tensorop_xe_group_gemm
      xe_gemm_bf16_s8_fp32_tensor_op_fp32_group_gemm.cpp
      xe_gemm_fp16_s8_fp32_tensor_op_fp32_group_gemm.cpp
      xe_gemm_bf16_u4_fp32_tensor_op_fp32_group_gemm.cpp
      xe_gemm_fp16_u4_fp32_tensor_op_fp32_group_gemm.cpp
      xe_gemm_fp16_fp8_fp32_tensor_op_fp32_group_gemm.cpp
    )

    # Single unified test executable combining all new test files
    cutlass_test_unit_add_executable(
      cutlass_test_unit_gemm_device_unified_tests
      gemm_universal_lincomb_per_rowbias_eltact.cpp
      gemm_universal_mainloopintelxexmx16group.cpp
      gemm_universal_mainloopintelxexmx16_group_fp8.cpp
      gemm_universal_mainloopintelxexmx16group_precision.cpp
      gemm_universal_fp_scaling.cpp
    )

    add_custom_target(
      cutlass_test_unit_gemm_device
      DEPENDS
      cutlass_test_unit_gemm_device_tensorop_xe
      cutlass_test_unit_gemm_device_tensorop_cooperative_xe
      cutlass_test_unit_gemm_device_tensorop_epilogue_fusion_xe
      cutlass_test_unit_gemm_device_mixed_input_tensorop_xe
      cutlass_test_unit_gemm_device_tensorop_xe_group_gemm
      cutlass_test_unit_gemm_device_mixed_dtype_tensorop_xe_group_gemm
      cutlass_test_unit_gemm_device_tensorop_xe
      cutlass_test_unit_gemm_device_unified_tests
    )

    add_custom_target(
      test_unit_gemm_device
      DEPENDS
      test_unit_gemm_device_tensorop_xe
      test_unit_gemm_device_tensorop_cooperative_xe
      test_unit_gemm_device_tensorop_epilogue_fusion_xe
      test_unit_gemm_device_mixed_input_tensorop_xe
      test_unit_gemm_device_tensorop_xe_group_gemm
      test_unit_gemm_device_mixed_dtype_tensorop_xe_group_gemm
    )
  else()
    # Dummy targets if not building for Intel
    add_custom_target(cutlass_test_unit_gemm_device)
    add_custom_target(test_unit_gemm_device)
  endif()
endif()
cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_cluster_multicast_sm90

  BATCH_SOURCES ON
  BATCH_SIZE 4

  sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_unspecialized.cu
  sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized.cu
  sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_pingpong.cu
  sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative.cu
  sm90_gemm_f8_f8_f32_tensor_op_f32_cluster_warpspecialized_cooperative.cu
)
cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_gmma_rs_warpspecialized_sm90

  BATCH_SOURCES ON
  BATCH_SIZE 4

  sm90_gemm_tf32_tf32_f32_tensor_op_f32_gmma_rs_cluster_warpspecialized.cu
  sm90_gemm_f8_f8_f32_tensor_op_f32_rs_cluster_warpspecialized_cooperative.cu
  sm90_gemm_f16_f16_f32_tensor_op_f32_rs_cluster_warpspecialized_cooperative.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_f32_tf32_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_tf32t_tf32n_f32t_tensor_op_f32_sm80.cu
  gemm_tf32n_tf32t_f32t_tensor_op_f32_sm80.cu
  gemm_tf32n_tf32n_f32t_tensor_op_f32_sm80.cu
  gemm_tf32t_tf32t_f32t_tensor_op_f32_sm80.cu
  gemm_universal_cf32n_cf32n_cf32n_tensor_op_f32_sm80.cu
  gemm_cf32n_cf32t_cf32t_tensor_op_tf32_f32_sm80.cu
  gemm_cf32t_cf32n_cf32t_tensor_op_tf32_f32_sm80.cu

  gemm_f32n_f32n_f32t_tensor_op_f32_sm80.cu
  gemm_f32n_f32n_f32t_tensor_op_bf16_f32_sm80.cu

  sm80_gemm_f16_f16_f32_tensor_op_f32.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_f64

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f64n_f64t_f64t_tensor_op_f64_sm80.cu
  gemm_f64t_f64n_f64t_tensor_op_f64_sm80.cu

  gemm_universal_cf64n_cf64t_cf64t_tensor_op_f64_sm80.cu
  gemm_universal_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm80.cu
  gemm_cf64n_cf64t_cf64t_tensor_op_f64_sm80.cu
  gemm_cf64t_cf64n_cf64t_tensor_op_f64_sm80.cu
  gemm_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm80.cu
  gemm_cf64t_cf64n_cf64t_tensor_op_f64_gaussian_sm80.cu

  sm80_gemm_f64_f64_f64_tensor_op_f64.cu
  
  # SM90 device level tests
  gemm_f64n_f64t_f64t_tensor_op_f64_sm90.cu
  gemm_f64t_f64n_f64t_tensor_op_f64_sm90.cu

  gemm_cf64n_cf64t_cf64t_tensor_op_f64_sm90.cu
  gemm_cf64t_cf64n_cf64t_tensor_op_f64_sm90.cu
  gemm_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm90.cu
  gemm_cf64t_cf64n_cf64t_tensor_op_f64_gaussian_sm90.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_s32_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_s8t_s8n_s32t_tensor_op_s32_sm80.cu
  gemm_s8t_s8n_s32n_tensor_op_s32_sm80.cu
  gemm_s8t_s8n_s8n_tensor_op_s32_sm80.cu
  gemm_s8t_s8n_s8t_tensor_op_s32_sm80.cu
  gemm_s8t_s8n_f16t_tensor_op_s32_sm80.cu
  gemm_s4t_s4n_s32n_tensor_op_s32_sm80.cu
  gemm_s4t_s4n_s32t_tensor_op_s32_sm80.cu
  gemm_s4t_s4n_s4n_tensor_op_s32_sm80.cu
  gemm_s4t_s4n_s4t_tensor_op_s32_sm80.cu
  gemm_b1t_b1n_s32n_tensor_op_s32_sm80.cu
  gemm_b1t_b1n_s32t_tensor_op_s32_sm80.cu

  gemm_s8n_s8t_s8n_tensor_op_s32_sm80.cu
  gemm_s4n_s4t_s4n_tensor_op_s32_sm80.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_wmma

  BATCH_SOURCES ON
  BATCH_SIZE 4

  # wmma floating point tests
  gemm_f16t_f16n_f16t_wmma_tensor_op_f16_sm70.cu
  gemm_f16n_f16t_f16t_wmma_tensor_op_f16_sm70.cu
  gemm_f16t_f16t_f16t_wmma_tensor_op_f16_sm70.cu
  gemm_f16n_f16n_f16t_wmma_tensor_op_f16_sm70.cu
  gemm_f16t_f16n_f16n_wmma_tensor_op_f16_sm70.cu
  gemm_f16n_f16t_f16n_wmma_tensor_op_f16_sm70.cu
  gemm_f16t_f16t_f16n_wmma_tensor_op_f16_sm70.cu
  gemm_f16n_f16n_f16n_wmma_tensor_op_f16_sm70.cu

  gemm_f16t_f16n_f32t_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16t_f32t_wmma_tensor_op_f32_sm70.cu
  gemm_f16t_f16t_f32t_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16n_f32t_wmma_tensor_op_f32_sm70.cu
  gemm_f16t_f16n_f32n_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16t_f32n_wmma_tensor_op_f32_sm70.cu
  gemm_f16t_f16t_f32n_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16n_f32n_wmma_tensor_op_f32_sm70.cu

  gemm_f16t_f16n_f16t_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16t_f16t_wmma_tensor_op_f32_sm70.cu
  gemm_f16t_f16t_f16t_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16n_f16t_wmma_tensor_op_f32_sm70.cu
  gemm_f16t_f16n_f16n_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16t_f16n_wmma_tensor_op_f32_sm70.cu
  gemm_f16t_f16t_f16n_wmma_tensor_op_f32_sm70.cu
  gemm_f16n_f16n_f16n_wmma_tensor_op_f32_sm70.cu

  # wmma int8 tests
  gemm_s8t_s8n_s32t_wmma_tensor_op_s32_sm72.cu
  gemm_s8t_s8n_s32n_wmma_tensor_op_s32_sm72.cu

  gemm_s8t_s8n_s8t_wmma_tensor_op_s32_sm72.cu
  gemm_s8t_s8n_s8n_wmma_tensor_op_s32_sm72.cu

  # wmma uint8 tests
  gemm_u8t_u8n_s32t_wmma_tensor_op_s32_sm72.cu

  # wmma sub byptes (s4 and b1) tests
  gemm_s4t_s4n_s32n_wmma_tensor_op_s32_sm75.cu
  gemm_s4t_s4n_s32t_wmma_tensor_op_s32_sm75.cu

  gemm_b1t_b1n_s32n_wmma_tensor_op_s32_sm75.cu
  gemm_b1t_b1n_s32t_wmma_tensor_op_s32_sm75.cu

  # wmma floating point tests (using singestage pipeline)
  gemm_f16t_f16n_f16t_singlestage_wmma_tensor_op_f16_sm70.cu
  gemm_f16t_f16n_f16n_singlestage_wmma_tensor_op_f16_sm70.cu

  gemm_f16t_f16n_f32t_singlestage_wmma_tensor_op_f32_sm70.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_planar_complex

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_planar_complex_f16_f16_f32_tensor_op_sm70.cu
  gemm_planar_complex_f16_f16_f32_tensor_op_sm75.cu
  gemm_planar_complex_f16_f16_f32_tensor_op_sm80.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm89

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f8t_f8n_f32t_tensor_op_f32_sm89.cu
  gemm_f8t_f8n_f32t_tensor_op_f32_sparse_sm89.cu
  gemm_f8t_f8n_f16t_tensor_op_f16_sm89.cu
  gemm_f8t_f8n_f8t_tensor_op_f32_sm89.cu
#  gemm_f8t_f8n_f8t_tensor_op_f32_sparse_sm89.cu
  gemm_f8t_f8n_f8t_tensor_op_f16_sm89.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_grouped

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_grouped_sm80.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_grouped_scheduler

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_grouped_scheduler_sm80.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_grouped_rank_2k_scheduler

  BATCH_SOURCES ON
  BATCH_SIZE 4

  rank_2k_grouped_scheduler_sm80.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_sparse_tensorop_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f16n_f16n_f16t_tensor_op_f32_sparse_sm80.cu
  gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_f16n_f16t_f16t_tensor_op_f16_sparse_sm80.cu
  gemm_f16n_f16t_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_f16t_f16n_f16t_tensor_op_f16_sparse_sm80.cu
  gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_f16t_f16t_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_f32t_f32n_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_f32n_f32t_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_f32t_f32t_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_f32n_f32n_f32t_tensor_op_f32_sparse_sm80.cu
  gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cu
  gemm_s4t_s4n_s32t_tensor_op_s32_sparse_sm80.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemv_device

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemv.cu
)

if (CUTLASS_NVCC_DEVICE_COMPILE)

  cutlass_test_unit_gemm_device_add_executable(
    cutlass_test_unit_gemm_device_gemm_with_fused_epilogue_tensorop
  
    gemm_with_reduction_f16n_f16n_f16n_tensorop_f32_sm75.cu
    gemm_with_broadcast_f16n_f16n_f16n_tensorop_f32_sm75.cu
  
    gemm_with_reduction_f16t_f16n_f16n_tensorop_f32_sm80.cu
  )

endif()

if (CUTLASS_NVCC_DEVICE_COMPILE)

if (NOT CUTLASS_NVCC_ARCHS MATCHES 101|101a|101f|110|110a|110f)
cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_blas3

  BATCH_SOURCES ON
  BATCH_SIZE 4

  ## SYRK
  # Syrk SM80 f64 tests
  syrk_f64n_f64t_tensor_op_f64_sm80.cu
  syrk_f64t_f64n_tensor_op_f64_sm80.cu

  # Syrk SM80 f32 tests
  syrk_tf32n_f32t_tensor_op_f32_sm80.cu
  syrk_tf32t_f32t_tensor_op_f32_sm80.cu
  syrk_f32n_f32t_tensor_op_fast_f32_sm80.cu
  syrk_f32t_f32t_tensor_op_fast_f32_sm80.cu

  # Syrk SM80 complex f64 tests
  syrk_cf64n_cf64t_tensor_op_f64_sm80.cu
  syrk_cf64n_cf64n_tensor_op_f64_sm80.cu

  # Syrk SM80 complex f32 tests
  syrk_cf32n_cf32t_tensor_op_f32_sm80.cu
  syrk_cf32n_cf32n_tensor_op_f32_sm80.cu
  syrk_cf32n_cf32t_tensor_op_fast_f32_sm80.cu
  syrk_cf32n_cf32n_tensor_op_fast_f32_sm80.cu

  # Syrk SM90 f64 tests
  syrk_f64_f64_tensor_op_f64_sm90.cu

  # Syrk SM90 complex f64 tests
  syrk_cf64_cf64_tensor_op_f64_sm90.cu

  ## HERK
  # Herk SM80 complex f64 tests
  herk_cf64h_cf64n_tensor_op_f64_sm80.cu

  # Herk SM80 complex f32 tests
  herk_cf32h_cf32n_tensor_op_f32_sm80.cu
  herk_cf32h_cf32n_tensor_op_fast_f32_sm80.cu

  # Herk SM90 complex f64 tests
  herk_cf64_cf64_tensor_op_f64_sm90.cu

  ## TRMM
  # Trmm SM80 f64 tests
  trmm_f64n_f64n_f64t_tensor_op_f64_ls_sm80.cu
  trmm_f64n_f64n_f64t_tensor_op_f64_rs_sm80.cu
  trmm_f64t_f64t_f64n_tensor_op_f64_ls_sm80.cu
  trmm_f64t_f64t_f64n_tensor_op_f64_rs_sm80.cu
  trmm_f64n_f64t_f64t_tensor_op_f64_rs_sm80.cu

  # Trmm SM80 f32 tests
  trmm_tf32t_tf32n_f32t_tensor_op_f32_ls_sm80.cu
  trmm_tf32n_tf32t_f32t_tensor_op_f32_ls_sm80.cu
  trmm_tf32n_tf32t_f32t_tensor_op_f32_rs_sm80.cu
  trmm_tf32t_tf32n_f32n_tensor_op_f32_ls_sm80.cu
  trmm_f32t_f32n_f32t_tensor_op_fast_f32_ls_sm80.cu
  trmm_f32n_f32t_f32t_tensor_op_fast_f32_ls_sm80.cu
  trmm_f32n_f32t_f32t_tensor_op_fast_f32_rs_sm80.cu
  trmm_f32t_f32n_f32n_tensor_op_fast_f32_ls_sm80.cu

  # Trmm SM80 complex f64 tests
  trmm_cf64n_cf64n_cf64t_tensor_op_f64_sm80.cu

  # Trmm SM80 complex f32 tests
  trmm_cf32n_cf32n_cf32t_tensor_op_f32_sm80.cu
  trmm_cf32n_cf32n_cf32t_tensor_op_fast_f32_sm80.cu

  # Trmm SM90 f64 tests
  trmm_f64_f64_f64_tensor_op_f64_sm90.cu

  # Trmm SM90 complex f64 tests
  trmm_cf64_cf64_cf64_tensor_op_f64_sm90.cu

  ## SYR2K
  # Syr2k SM80 f64 tests
  syr2k_f64n_f64t_tensor_op_f64_sm80.cu
  syr2k_f64n_f64n_tensor_op_f64_sm80.cu
  syr2k_f64t_f64n_tensor_op_f64_sm80.cu

  # Syr2k SM80 f32 tests
  syr2k_tf32n_f32n_tensor_op_f32_sm80.cu
  syr2k_tf32t_f32n_tensor_op_f32_sm80.cu
  syr2k_f32n_f32n_tensor_op_fast_f32_sm80.cu
  syr2k_f32t_f32n_tensor_op_fast_f32_sm80.cu

  # Syr2k SM80 complex f64 tests
  syr2k_cf64n_cf64t_tensor_op_f64_sm80.cu
  syr2k_cf64n_cf64n_tensor_op_f64_sm80.cu

  # Syr2k SM80 complex f32 tests
  syr2k_cf32n_cf32n_tensor_op_f32_sm80.cu
  syr2k_cf32n_cf32t_tensor_op_f32_sm80.cu
  syr2k_cf32n_cf32n_tensor_op_fast_f32_sm80.cu
  syr2k_cf32n_cf32t_tensor_op_fast_f32_sm80.cu

  # Syr2k SM90 f64 tests
  syr2k_f64_f64_tensor_op_f64_sm90.cu

  # Syr2k SM90 complex f64 tests
  syr2k_cf64_cf64_tensor_op_f64_sm90.cu

  ## HER2K
  # Her2k SM80 complex f64 tests
  her2k_cf64n_cf64n_tensor_op_f64_sm80.cu

  # Her2k SM80 complex f32 tests
  her2k_cf32h_cf32n_tensor_op_f32_sm80.cu
  her2k_cf32h_cf32n_tensor_op_fast_f32_sm80.cu

  # Her2k SM90 complex f64 tests
  her2k_cf64_cf64_tensor_op_f64_sm90.cu

  ## SYMM
  # Symm SM80 f64 tests
  symm_f64n_f64n_tensor_op_f64_ls_sm80.cu
  symm_f64n_f64n_tensor_op_f64_rs_sm80.cu
  symm_f64n_f64t_tensor_op_f64_ls_sm80.cu
  symm_f64n_f64t_tensor_op_f64_rs_sm80.cu
  symm_f64t_f64n_tensor_op_f64_ls_sm80.cu
  symm_f64t_f64n_tensor_op_f64_rs_sm80.cu
  symm_f64t_f64t_tensor_op_f64_ls_sm80.cu
  symm_f64t_f64t_tensor_op_f64_rs_sm80.cu

  # Symm SM80 f32 tests
  symm_tf32n_f32n_tensor_op_f32_ls_sm80.cu
  symm_tf32n_f32n_tensor_op_f32_rs_sm80.cu
  symm_tf32t_f32t_tensor_op_f32_ls_sm80.cu
  symm_f32n_f32n_tensor_op_fast_f32_ls_sm80.cu
  symm_f32n_f32n_tensor_op_fast_f32_rs_sm80.cu
  symm_f32t_f32t_tensor_op_fast_f32_ls_sm80.cu

  # Symm SM80 complex f64 tests
  symm_cf64n_cf64n_cf64n_tensor_op_ls_f64_sm80.cu
  symm_cf64n_cf64n_cf64n_tensor_op_rs_f64_sm80.cu

  # Symm SM80 complex f32 tests
  symm_cf32n_cf32n_tensor_op_f32_ls_sm80.cu
  symm_cf32n_cf32n_tensor_op_f32_rs_sm80.cu
  symm_cf32n_cf32n_tensor_op_fast_f32_ls_sm80.cu
  symm_cf32n_cf32n_tensor_op_fast_f32_rs_sm80.cu

  # Symm SM90 f64 tests
  symm_f64_f64_tensor_op_f64_sm90.cu

  # Symm SM90 complex f64 tests
  symm_cf64_cf64_cf64_tensor_op_f64_sm90.cu

  # Hemm SM80 complex f64 tests
  hemm_cf64h_cf64n_cf64n_tensor_op_ls_f64_sm80.cu
  hemm_cf64h_cf64n_cf64n_tensor_op_rs_f64_sm80.cu

  # Hemm SM80 complex f32 tests
  hemm_cf32h_cf32n_tensor_op_f32_ls_sm80.cu
  hemm_cf32h_cf32n_tensor_op_f32_rs_sm80.cu
  hemm_cf32h_cf32n_tensor_op_fast_f32_ls_sm80.cu
  hemm_cf32h_cf32n_tensor_op_fast_f32_rs_sm80.cu

  # Hemm SM90 complex f64 tests
  hemm_cf64_cf64_cf64_tensor_op_f64_sm90.cu
)

if (NOT CUTLASS_NVCC_ARCHS MATCHES 100f|101|101a|101f|110|110a|110f|103|103a|103f)
cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_blas3_gaussian

  BATCH_SOURCES ON
  BATCH_SIZE 4

  syrk_cf64n_cf64t_tensor_op_f64_gaussian_sm80.cu
  trmm_cf64n_cf64n_cf64t_tensor_op_f64_gaussian_sm80.cu
  symm_cf64n_cf64n_cf64n_tensor_op_ls_f64_gaussian_sm80.cu
  hemm_cf64h_cf64n_cf64n_tensor_op_ls_f64_gaussian_sm80.cu
)
endif()

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_grouped_blas3

  BATCH_SOURCES ON
  BATCH_SIZE 4

  # Grouped SYR2K SM80 f64 tests
  syr2k_f64n_f64n_tensor_op_f64_grouped_sm80.cu
  syr2k_f64n_f64t_tensor_op_f64_grouped_sm80.cu
  syr2k_f64t_f64n_tensor_op_f64_grouped_sm80.cu
  syr2k_f64t_f64t_tensor_op_f64_grouped_sm80.cu

  # Grouped SYR2K SM80 cf64 tests
  syr2k_cf64n_cf64n_tensor_op_f64_grouped_sm80.cu
  syr2k_cf64n_cf64t_tensor_op_f64_grouped_sm80.cu
  syr2k_cf64t_cf64n_tensor_op_f64_grouped_sm80.cu
  syr2k_cf64t_cf64t_tensor_op_f64_grouped_sm80.cu

  # Grouped HER2K SM80 f64 tests
  her2k_cf64n_cf64n_tensor_op_f64_grouped_sm80.cu
  her2k_cf64h_cf64n_tensor_op_f64_grouped_sm80.cu
)
endif()

endif()

if (CUTLASS_NVCC_DEVICE_COMPILE)

  cutlass_test_unit_gemm_device_add_executable(
    cutlass_test_unit_gemm_device_broadcast
    gemm_f16t_f16n_f16t_tensor_op_f16_broadcast_sm80.cu
  )

endif()



if (CUTLASS_NVCC_ARCHS MATCHES 100a)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_16b_tensorop_sm100_ptr_array

  # 14 (9 + 5) unit tests 
  sm100_gemm_f16_f16_f16_tensor_op_f32_ptr_array.cu
  sm100_gemm_bf16_bf16_bf16_tensor_op_f32_ptr_array.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_16b_tensorop_sm100_group_gemm

  sm100_gemm_f16_f16_f16_tensor_op_f32_group_gemm.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_16b_mixed_tensorop_sm100_ptr_array

  # 14 (9 + 5) unit tests 
  sm100_gemm_f16_f16_f32_tensor_op_f32_ptr_array.cu
  sm100_gemm_f16_f16_f16_tensor_op_f16_ptr_array.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_32b_tensorop_sm100_ptr_array

  # 10 unit tests 
  sm100_gemm_f32_f32_f32_tensor_op_f32_ptr_array.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_32b_tensorop_sm100_group_gemm

  # 10 unit tests 
  sm100_gemm_f32_f32_f32_tensor_op_f32_group_gemm.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_8b_tensorop_sm100_ptr_array

  # 12 unit tests
  sm100_gemm_i8_i8_i8_tensor_op_s32_ptr_array.cu
  sm100_gemm_f8_f8_f8_tensor_op_f32_ptr_array.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_8b_tensorop_sm100_group_gemm

  # 8 unit tests
  sm100_gemm_f8_f8_f8_tensor_op_f32_group_gemm.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_mxf8_training_sm100_group_gemm

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_gemm_mxf8_mxf8_mxf8_tensor_op_f32_group_gemm.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_mxf4xmxf8_sm100_group_gemm

  # 8 unit tests
  sm100_gemm_mxf4_mxf8_mxf8_tensor_op_f32_group_gemm.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_blockscaled_gemm_device_fp4_tensorop_sm100_ptr_array

  # 8 unit tests
  sm100_gemm_f4_f4_f32_tensor_op_f32_ptr_array.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_blockscaled_gemm_device_fp4_tensorop_sm100_group_gemm_1

  # 8 unit tests
  sm100_gemm_f4_f4_f32_tensor_op_f32_group_gemm.cu
)
cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_blockscaled_gemm_device_fp6_tensorop_sm100_ptr_array

  # 8 unit tests
  sm100_gemm_f6_f6_f32_tensor_op_f32_ptr_array.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_blockwise_gemm_sm100

  sm100_gemm_f8_f8_f8_tensor_op_f32_blockwise.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_runtime_datatype_alignx_sm100

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_gemm_f8_f8_f8_tensor_op_f32_runtime_datatype_alignx.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_alignx_sm100

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1
  sm100_gemm_f8_f8_f8_tensor_op_f32_alignx.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_alignx_streamK_sm100

  # setting batch size fo 1 to control memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_gemm_f8_f8_f8_tensor_op_f32_alignx_streamK.cu
)

endif()

if (CUTLASS_NVCC_ARCHS MATCHES 103a|103f)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_1sm

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_1sm.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_f4_tensorop_sm103_nosmem

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_tensor_op_f32_nosmem.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_2sm

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_2sm.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_group_1sm_128x128

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_group_1sm_128x128.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_group_1sm_128x192

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_group_1sm_128x192.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_group_2sm_256x192

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_group_2sm_256x192.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_group_2sm_256x256

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_group_2sm_256x256.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_ptr_array_1sm_128x128

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_ptr_array_1sm_128x128.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_ptr_array_1sm_128x192

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_ptr_array_1sm_128x192.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_ptr_array_2sm_256x192

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_ptr_array_2sm_256x192.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_ptr_array_2sm_256x256

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_ptr_array_2sm_256x256.cu
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_streamk

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_stream_k.cu
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_2sm_256x256

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_2sm_256x256.cu
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_2sm_256x192

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_2sm_256x192.cu
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_2sm_256x128

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_2sm_256x128.cu
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_1sm_128x128

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_1sm_128x128.cu
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_1sm_128x192

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_1sm_128x192.cu
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm103_1sm_128x256

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm103_gemm_f4_f4_f32_tensor_op_f32_1sm_128x256.cu
)

endif()


cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_simt_sm100

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_gemm_f32_f32_f32_simt_align1.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_simt_sm100_bias_relu

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_gemm_f32_f32_f32_simt_align1_bias_relu.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_simt_sm100_ptr_array

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1
  sm100_gemm_f32_f32_f32_simt_align1_ptr_array.cu
)

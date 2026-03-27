![ALT](https://raw.githubusercontent.com/intel/sycl-tla/main/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Complete CUDA GEMM decomposition")

# SYCL\* Templates for Linear Algebra (SYCL\*TLA)

**This repository is forked from the NVIDIA CUTLASS repository and extends CUTLASS and CuTe API support to Intel GPUs through SYCL enablement.**
*This project was previously referred to as CUTLASS-SYCL, you may see references to CUTLASS-SYCL in the code and documentation.*
*For SYCL support instructions, refer to the [SYCL build documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/build/building_with_sycl_support.md)*

_Latest upstream: CUTLASS 4.4.2 - March 2026_

*SYCL is a trademark of the Khronos Group Inc, Other names and brands may be claimed as the property of others.*
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/intel/sycl-tla/badge)](https://scorecard.dev/viewer/?uri=github.com/intel/sycl-tla)

SYCL\*TLA is a modular, header‑only C++ template framework for high‑performance 
GEMM, and fused epilogue kernels. It applies hierarchical tiling, composable policy 
abstractions, and efficient data‑movement primitives to build flexible, reusable 
building blocks for dense linear algebra. The SYCL implementation brings those 
optimizations to Intel GPUs with tuned kernels for modern execution units and memory 
hierarchies. It adds mixed‑precision and epilogue fusion pathways designed to 
simplify integrating advanced quantization and post‑processing into custom pipelines.

To support a wide variety of applications, SYCL\*TLA provides extensive
support for mixed-precision computations on Intel hardware, providing
specialized data-movement and multiply-accumulate abstractions for FP64, FP32,
FP16, BF16, 8b floating point types (E5M2 and E4M3 for FP8), narrow integer
types (4 and 8b signed and unsigned integers with support for zero-point
quantization), and mixed-precision operations with tensor-wise, channel-wise,
and group-wise quantization support. SYCL\*TLA demonstrates optimal matrix
multiply operations targeting Intel's programmable, high-throughput execution
units implemented in Intel Data Center GPU Max/Flex Series (Intel Xe
architecture, codename: Ponte-Vecchio) and Intel Arc B580 GPUs.

See the [Quick Start Guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md) to get started quickly.

See the [functionality docs](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) for a more comprehensive
list of kernel level features, data types, instructions, and minimum supported by CUTLASS on each GPU
architecture.

This project fast follows NVIDIA CUTLASS releases to ensure parity of APIs and features.

Base NVIDIA CUTLASS Versions for SYCL*TLA releases:
| SYCL*TLA | NVIDIA CUTLASS |
|-----------------|----------|
|0.1| 3.9|
|0.2 | 3.9.2 |
|0.3 | 3.9.2 |
|0.5 | 4.2.0 |
|0.6 | 4.2.0 |
|0.7 | 4.2.1 |
|0.8 | 4.2.1 |

# What's New in SYCL*TLA 0.8

## [SYCL*TLA 0.8](https://github.com/intel/sycl-tla/releases/tag/v0.8) (2026-03-25)

### Major Architecture Changes
- **Support BMG G31 Platform ([#755](https://github.com/intel/sycl-tla/pull/755))**
- **SLM Copy API functionalities and examples**
  - Support CuTe copy engines for 1D LDSM/STSM operations with vISA ([#753](https://github.com/intel/sycl-tla/pull/753))
  - Enable fusion example of 2 matmul operations through SLM Copy API ([#747](https://github.com/intel/sycl-tla/pull/747))
  - Enable subgroup specialization example with SLM Copy API ([#735](https://github.com/intel/sycl-tla/pull/735))
- **Support default sub-byte reorder for low-precision data types ([#709](https://github.com/intel/sycl-tla/pull/709))**

### Enhancements
- **Flash Attention Performance Improvements (for BMG and BF16)**:
  - Fix long context OOM issue ([#728](https://github.com/intel/sycl-tla/pull/728))
  - Overall performance improved from ~45% to ~78% of peak([#728](https://github.com/intel/sycl-tla/pull/728), [#743](https://github.com/intel/sycl-tla/pull/743),[#749](https://github.com/intel/sycl-tla/pull/749),[#750](https://github.com/intel/sycl-tla/pull/750))
  - Refine code and fix bugs ([#715](https://github.com/intel/sycl-tla/pull/715), [#716](https://github.com/intel/sycl-tla/pull/716),[#720](https://github.com/intel/sycl-tla/pull/720))
- **Epilogue Visitor Tree (EVT) Enhancements**:
  - Combine with SIGMOID function ([#686](https://github.com/intel/sycl-tla/pull/686))
  - Add Relu variation test cases ([#693](https://github.com/intel/sycl-tla/pull/693))
  - Enhance and refine code and test case([#703](https://github.com/intel/sycl-tla/pull/703), [#717](https://github.com/intel/sycl-tla/pull/717))
- **GEMM Enhancements**:
  - Support all GEMM tile shapes ([#738](https://github.com/intel/sycl-tla/pull/738))
  - Enhance examples ([#726](https://github.com/intel/sycl-tla/pull/726))

**See the [CHANGELOG](https://github.com/intel/sycl-tla/blob/main/CHANGELOG-SYCL.md) for details of all past releases and updates.**

To get started quickly - please refer:
  - [CUTLASS C++ Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html).
  - [CuTe DSL Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html).

# What's New in CUTLASS 4.4

## CuTe DSL
* New features
  - CuTe DSL now supports CUDA toolkit 13.1!
    + Set up with cutlass/python/CuTeDSL/setup.sh --cu13
    + Refer to https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html for more details
  - GB300 is now supported in CuTe DSL with CTK 13.1
    + Refer to [SM103 batched 3xFP4 blockscaled GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py) for example kernel
  - cute.experimental: introduce a higher-level, composable layer on top of existing CuTe DSL APIs (not a separate abstraction), which can be mixed with existing Cute DSL building blocks.
    + Fragment-free programming model: copy/dot APIs take memrefs directly instead of descriptors/fragments.
    + Automatic TMA descriptor generation and update insertion.
    + Automatic vectorization and predication for SIMT copies.
    + New pipeline abstraction with convenience wrappers
    + New Partition ops to simplify partitioning logic.
    + Device-side TMA descriptor allocation, initialization, and management
    + These examples can be found here https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/experimental
  - Ahead of Time (AoT) compilation is now available!
    + Refer to files under https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/export for example usage
  - JAX support - you can now use CuTeDSL along with JAX
    + Refer to files under https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/jax for example usage
  - Introduced versioning support in DSL:
    + cutlass.__version__ for a string representation of DSL version
    + cutlass.CUDA_VERSION for a version class to tell the CUDA version used for DSL
  - Added CopyDsmemStoreOp to store data to distributed shared memory with explicit synchronization.
  - Grouped GEMM example now supports device-only problem shapes.
  - We allow grid carve-out without problem shapes being available on host.
  - Tma+LdMatrix features for loading+unpacking narrow-width types (refer to mixed_input_fmha_decode.py for example usage).
  - It is possible now to have customized epilogue fusion for persistent dense GEMM through a Python Epilogue Fusion Configuration (EFC) function, somewhat similar to CUTLASS C++ EVT. It also provides a PyTorch evaluator to compare the results.
  - CuTe DSL now supports Python 3.14 for both x86_64 and aarch64
  - Runtime Pointer/Tensor/FakeTensor now supports __cache_key__, providing a stable, hashable representation that simplifies and improves compiled function caching.

* More examples of authorizing peak-performance kernels
  - [SM103 batched 3xFP4 blockscaled GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py)
  - Mixed input FMHA decode example with support for int4 KV (int8 KV supported in 4.3)
  - New acc_scale grouped mixed input gemm kernel variant is introduced to deliver better performance for decoding cases.
  - All mixed_input_gemm examples are moved into a separate folder `mixed_input_gemm`. Common utility functions are also extracted into mixed_input_host_utils.py under the same folder.

* Bug fixing and improvements
  - Fixed an issue that both branches of if are executed
  - Fixed `cute.printf` with f-string
  - Fixed an indexing issue of scalar tensor
  - Fixed small K reference check error for cta_tile_n = 256 case with overlapping accumulator optimization in [Blackwell SM100 persistent dense blockscaled GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py).
  - Fixed a segfault issue with tvm-ffi on aarch64
  - Fixed Hopper FMHA causal attention performance regression on CUDA toolkit 13.1 by
 optimizing mbarrier synchronization to avoid unnecessary convergence barriers.
  - Fix kernel loading race condition when multiple GPU are present in the same process in JAX.

* API changes
  - Deprecate get_num_tmem_alloc_cols from blackwell_helpers.py. Use the one from tmem_allocator.py instead.
  - Deprecate SM100_TMEM_CAPACITY_COLUMNS and SM100_TMEM_MIN_ALLOC_COLUMNS.
  - LdMatrix16x16x8bOp and StMatrix16x8x8bOp now require explicit transpose=True when calling __init__, to avoid ambiguity in data transposition.
  - LdMatrix16x16x8bOp copy traits updated to be faithful to PTX without permutations. Permuted variant is renamed to LdMatrix16x8x8bOp.
  - Grouped GEMM example takes the argument --host_problem_shape_available. If the argument is provided, grid is carved out based upon the host problem shapes, otherwise, we launch maximum possible SMs.
  - hardware_info.get_max_active_cluster support pass in specific stream to query. Useful for green context based SM partition.
  - group_bulk_copy_modes in async bulk copy example is now deprecated, use group_modes directly instead.
  - Deprecate nvvm wrapper from using nvvm enum, use str instead.
  - cute.arch.calc_packed_f32x2_op default enable ftz to default disable ftz
  - In CuTe DSL with CTK 13.1, following APIs in cutlass.cute.arch now require string literal instead of enum as argument:
    + fence_proxy
    + fence_view_async_tmem_op
    + calc_packed_f32x2_op
    + warp_redux_sync
    + atomic_add
    + atomic_and
    + atomic_or
    + atomic_xor
    + atomic_max
    + atomic_min
    + atomic_exch
    + atomic_cas
    + store
    + load

* Use 'Advanced control file' for mixed input gemm examples for better performance.
  - Advanced control file is an experimental feature of CUDA compiler. The controls file contains internal compiler settings tuned for specific kernels with a specific version of CUDA toolkit to get better GPU kernel code. More details and documentation on how to create these controls files will be provided in future CUDA toolkit release.  Note: The advanced compiler control file is not expected to work for kernels that it was not tuned for. There is no compatibility guarantee, and the controls file will not work for CUDA toolkit with a different version.

## CUTLASS C++
* Add [example 93](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa/) for Blackwell low latency generation phase GQA kernel.
    - Flash Decoding with cluster reduction.
    - Kernel design details please check [Readme](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa/readme.md).
* Add Blackwell SM100 State Space Decomposition (SSD) kernel in [example 112](https://github.com/NVIDIA/cutlass/tree/main/examples/112_blackwell_ssd).
* Add Hopper SM90 State Space Decomposition (SSD) kernel in [example 111](https://github.com/NVIDIA/cutlass/tree/main/examples/111_hopper_ssd).
* Add Hopper e2m1 to fp32 optimized conversion and e2m1 * TF32 tensor core GEMM.
    - Enable [example 55](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm) with TF32 support
* Add [example 94](https://github.com/NVIDIA/cutlass/tree/main/examples/94_ada_fp8_blockwise/) for Ada FP8xFP8 -> BF16 GEMM with blockwise dequantization of input matrices in the MMA loop with FP32 accumulation.
* Add support for arbitrary application-provided strides for block-scale tensors.
    - Users and applications now must pass valid block-scale strides in all cases, even when the tensor is packed.
* Support 4x blockscaled public ptx for CUDA 13.1.
* Enable Blackwell SM120f compilation of examples and exposes NVFP4/MX Grouped GEMM in the CUTLASS Profiler.
* Allow non-static `TmaGbasis` in `AuxTmaParams`.
    - Some cases in attention kernel may require non-static `tma_gbasis`.
    - Relax the restriction on `TmaGbasis` parameter of `AuxTmaParams` and users are allowed to manually construct a dynamic gbasis.
* Fix some kernel issues:
    - Fix MSVC pre process issue.
    - Fix a self assign issue in GEMV kernel.
    - Fix a TMA descriptor bug where the CUDA driver is not properly setting the OOB address gen mode correctly.
    - Fix memory fence for clc scheduler in Blackwell SM120 pingpong kernel.
    - Fix missing SMEM alignment in Blackwell SM120 scale factors.
    - Fix a PDL issue for grouped gemm.
    - Fix divide-by-zero issue in canimplement for sm100 implicit gemm kernels.
    - Fix cluster swizzle for Grouped GEMMs.
        + Move host-side swizzling heuristics to device.
        + Apply swizzle per group based on problem shape and max swizzle size.
        + Improve examples and unit tests.
* Fix some profiler issues:
    - Fix a core dump issue for nvfp4 grouped GEMM kernel.
    - Fix inconsistent GEMM verification logic.
    - Rework grouped gemm verification logic for different types.
    - Fix api break change in using nvMatmulHeuristics.
* Fix some failed links under `media/docs`.

Note: CUTLASS 4.x builds are known to be down on Windows platforms for all CUDA toolkits.
CUTLASS team is working on a fix.

**See the [CHANGELOG](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html) for details of all past releases and updates.**

# Performance

CUTLASS primitives are very efficient.  When used to construct device-wide GEMM kernels,
they exhibit nearly optimal utilization of peak theoretical throughput. The figure below
shows CUTLASS 3.8's performance as a % of theoretical peak utilization
on various input and output data types when run on NVIDIA Blackwell SM100 architecture GPU.

![ALT](media/images/cutlass-3.8-blackwell-gemm-peak-performance.svg "")

The two figures below show the continual CUTLASS performance improvements
on an [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) (NVIDIA Hopper architecture) since
CUTLASS 3.1.
CUTLASS 3.5.1 was compiled with the [CUDA 12.5u1 Toolkit](https://developer.nvidia.com/cuda-downloads).
Tensor Core operations are implemented using CUDA's
[mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma) and
[wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions) instructions.

![ALT](media/images/cutlass-3.5.1-gemm-peak-performance.png "")
![ALT](media/images/cutlass-3.5.1-gemm-peak-performance-fp8.png "")

# CuTe

SYCL\*TLA supports the newly introduced core library, CuTe, to describe and manipulate tensors of threads and data.
CuTe in SYCL\*TLA is a collection of C++ SYCL template abstractions for
defining and operating on hierarchically multidimensional layouts of threads and data.
CuTe provides `Layout` and `Tensor` objects that compactly package the type,
shape, memory space, and layout of data, while performing the complicated indexing for the user.
This lets programmers focus on the logical descriptions of their algorithms while
CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design,
implement, and modify all dense linear algebra operations.

The core abstractions of CuTe are hierarchically multidimensional layouts
which can be composed with data arrays to represent tensors.
The representation of layouts is powerful enough to represent nearly
everything we need to implement efficient dense linear algebra.
Layouts can also be combined and manipulated via functional composition, on which we build a large set of common operations such as tiling and partitioning.

SYCL\*TLA and beyond adopts CuTe throughout the GEMM hierarchy in its templates.
This greatly simplifies the design and improves code composability and readability.
More documentation specific to CuTe can be found in its
[dedicated documentation directory](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/cute/00_quickstart.md) (Intel/SYCL fork) or
[upstream documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html) (NVIDIA CUTLASS).

# Compatibility

Minimum requirements:

- Architecture: Intel Data Center GPU Max Series (codename: Ponte-Vecchio)
- Compiler: Must support at least C++17
- DPC++ Compiler Version: oneAPI 2025.1 and onwards
- Intel Compute Runtime and Graphics Compiler: 
  - For Intel Data Center GPU Max Series: Runtime from [LTS driver installation guide](https://dgpu-docs.intel.com/driver/installation-lts2.html), IGC [v2.27.10](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.27.10) or later

## Hardware Support

SYCL*TLA runs successfully on the following Intel GPUs.

|**GPU**|**Intel GPU Architecture**
|---|---|
|Intel Data Center GPU Max Series            |Xe-HPC|
|Intel Arc GPU B580 Graphics                       |Xe2|

## Validated Software Configurations

We are regularly testing following setup in CI.

|**Platform**|**Operating System** | **DPC++ Compiler** | **G++** | **Intel Compute Runtime** |**Intel Graphics Compiler** |
|-----------------|----------|-----------------|--------|---------------------|-----------------------|
|Xe-HPC| Ubuntu 24.04 |2025.3+ |G++13  | 25.48 | 2.24 |
|Xe2| Ubuntu 25.04 |2025.3+  |G++13  | 26.01 | 2.27 |

## NVIDIA GPU Support (Upstream CUTLASS)

|**GPU**|**CUDA Compute Capability**|**Minimum CUDA Toolkit Required by CUTLASS-3**|
|---|---|---|
|NVIDIA V100 Tensor Core GPU            |7.0|11.4|
|NVIDIA TitanV                          |7.0|11.4|
|NVIDIA GeForce RTX 20x0 series         |7.5|11.4|
|NVIDIA T4                              |7.5|11.4|
|NVIDIA A100 Tensor Core GPU            |8.0|11.4|
|NVIDIA A10                             |8.6|11.4|
|NVIDIA GeForce RTX 30x0 series         |8.6|11.4|
|NVIDIA GeForce RTX 40x0 series         |8.9|11.8|
|NVIDIA L40                             |8.9|11.8|
|NVIDIA H100 Tensor Core GPU            |9.0|11.8|
|NVIDIA H200 Tensor Core GPU            |9.0|11.8|
|NVIDIA B200 Tensor Core GPU            |10.0|12.8|
|NVIDIA B300 Tensor Core GPU            |10.3|13.0|
|NVIDIA DRIVE Thor                      |11.0|13.0|
|NVIDIA GeForce RTX 50x0 series         |12.0|12.8|
|NVIDIA DGX Spark                       |12.1|13.0|

## Target Architecture

The target architecture information is passed on to SYCL*TLA via the cmake flag
`DPCPP_SYCL_TARGET`. 

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_pvc"
```
Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21"
```

Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31"
```

> Note: `-DDPCPP_SYCL_TARGET="bmg"` will compile for both `intel_gpu_bmg_g21`, `intel_gpu_bmg_g31` targets.

Please refer to the [functionality documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) for Intel/SYCL-specific information, or the [upstream CUTLASS functionality documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html)
for details on which kernels require which target architectures.

# Documentation

CUTLASS is described in the following documents and the accompanying
[Doxygen documentation](https://nvidia.github.io/cutlass).

## SYCL*TLA Documentation (Intel GPU Fork)

- [Quick Start Guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md) - basics of building and running CUTLASS
- [Functionality](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) - summarizes functionality available in CUTLASS
- [Efficient GEMM in CUDA](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/efficient_gemm.md) - describes how GEMM kernels may be implemented efficiently in CUDA
- [CUTLASS 3.x Design](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/cutlass_3x_design.md) - describes the CUTLASS 3.x design, its benefits, and how CuTe enables us to write much more composable components
- [GEMM API 3.x](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/gemm_api_3x.md) - describes the CUTLASS 3.x GEMM model and C++ template concepts
- [Implicit GEMM Convolution](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/implicit_gemm_convolution.md) - describes 2-D and 3-D convolution in CUTLASS
- [Code Organization](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/code_organization.md) - describes the organization and contents of the CUTLASS project
- [Terminology](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/terminology.md) - describes terms used in the code
- [Programming Guidelines](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/programming_guidelines.md) - guidelines for writing efficient modern CUDA C++
- [Fundamental types](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/fundamental_types.md) - describes basic C++ classes used in CUTLASS to represent numeric quantities and arrays
- [Layouts](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/layout.md) - describes layouts of matrices and tensors in memory
- [Tile Iterators](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/tile_iterator_concept.md) - describes C++ concepts for iterating over tiles of matrices in memory
- [CUTLASS Utilities](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/utilities.md) - additional templates used to facilitate rapid development

## Upstream CUTLASS Documentation (NVIDIA)

- [Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html) - basics of building and running CUTLASS
- [Functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html) - summarizes functionality available in CUTLASS
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) - describes how GEMM kernels may be implemented efficiently in CUDA
- [CUTLASS 3.x Design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html) - describes the CUTLASS 3.x design, its benefits, and how CuTe enables us to write much more composable components
- [GEMM API 3.x](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) - describes the CUTLASS 3.x GEMM model and C++ template concepts
- [GEMM API 2.x](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html) - describes the CUTLASS 2.x GEMM model and C++ template concepts
- [Implicit GEMM Convolution](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html) - describes 2-D and 3-D convolution in CUTLASS
- [Code Organization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html) - describes the organization and contents of the CUTLASS project
- [Terminology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/terminology.html) - describes terms used in the code
- [Programming Guidelines](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html) - guidelines for writing efficient modern CUDA C++
- [Fundamental types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html) - describes basic C++ classes used in CUTLASS to represent numeric quantities and arrays
- [Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html) - describes layouts of matrices and tensors in memory
- [Tile Iterators](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/tile_iterator_concept.html) - describes C++ concepts for iterating over tiles of matrices in memory
- [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html) - command-line driven profiling application
- [CUTLASS Utilities](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html) - additional templates used to facilitate rapid development
- [Dependent kernel launch](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html) - describes a new feature in Hopper which allows overlapping dependent
kernels in the same stream, and how it is used in CUTLASS.

# Resources


# Building SYCL*TLA

SYCL*TLA is a header-only template library and does not need to be built to be used by other
projects. Client applications should target SYCL*TLA's `include/` directory in their include
paths.

SYCL*TLA unit tests, examples, and utilities can be built with CMake.
The minimum version of CMake is given in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).
Make sure you have Intel oneAPI DPC++ compiler installed and the environment is properly set up.

**Note for NVIDIA GPU users**: CUTLASS unit tests, examples, and utilities can be built with CMake.
The minimum version of CMake is given in the [Quickstart guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html).
Make sure the `CUDACXX` environment variable points to NVCC in the CUDA Toolkit installed
on your system.

```bash
$ source /opt/intel/oneapi/setvars.sh
```

Create a build directory within the SYCL*TLA project, then run CMake. You need to specify
the target Intel GPU architecture using the `DPCPP_SYCL_TARGET` flag.
For Intel Data Center GPU Max Series (Ponte Vecchio), use `intel_gpu_pvc`.
For Intel Arc GPU B580 Graphics, use `intel_gpu_bmg_g21`.
For Intel Arc GPU Battlemage (G31), use `intel_gpu_bmg_g31`.

```bash
$ mkdir build && cd build

$ CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_pvc"     # compiles for Intel Data Center GPU Max Series
```

Or for Intel Arc GPU B580 Graphics:

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" # compiles for Intel Arc GPU B580 Graphics
```

Or for Intel Arc GPU Battlemage (G31):

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31" # compiles for Intel Arc GPU Battlemage (G31)
```

To compile with G++ as host compiler, add the flag `-DDPCPP_HOST_COMPILER=g++-13` to the cmake command. Please note that the build system must be able to find `g++-13` in your PATH.

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_HOST_COMPILER=g++-13 -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" # compiles for Intel Arc GPU B580 Graphics with G++ as host compiler
```

From the `build/` directory, compile and run the SYCL*TLA unit tests by building the target `test_unit` with make.

The unit tests are organized as several binaries mirroring the top-level namespaces of SYCL*TLA,
and they may be executed in parallel via make's `-j` command line argument.

```bash
$ make test_unit -j
...
...
...
[----------] Global test environment tear-down
[==========] XXX tests from YY test cases ran. (ZZZZ ms total)
[  PASSED  ] XXX tests.
```

All tests should pass on supported Intel GPU platforms, though the exact number of tests may vary over time.


# Project Structure

SYCL*TLA is arranged as a header-only library along with Utilities, Tools, Examples, and unit tests.

A detailed explanation of the source code organization may be found in the
[SYCL*TLA documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/code_organization.md) or
[CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html), but several main components are summarized below.

## SYCL*TLA

```
include/                     # client applications should target this directory in their build's include paths

  cutlass/                   # SYCL Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of Intel GPU architecture features (including instruction-level GEMMs)

    conv/                    # code specialized for convolution on Intel GPUs

    epilogue/                # code specialized for the epilogue of gemm/convolution using SYCL

    gemm/                    # code specialized for general matrix product computations with SYCL

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory

    platform/                # SYCL-capable Standard Library components for Intel GPUs

    reduction/               # bandwidth-limited reduction kernels optimized for Intel GPU architectures

    thread/                  # SYCL workgroup and subgroup code for Intel GPU execution units
    
    transform/               # code specialized for layout, type, and domain transformations using SYCL

    *                        # core vocabulary types, containers, and basic numeric operations

  cute/                      # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy for SYCL

    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples

    arch/                    # Intel GPU architecture wrapper structs for copy and math instructions

    atom/                    # Meta-information for Intel GPU operators and SYCL kernels

      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma for Intel GPU architectures

      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy optimized for SYCL

      *xe*.hpp               # Intel Xe architecture specific meta-information for copy and math operations

    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations

```

### SYCL*TLA Examples

[SYCL*TLA examples](https://github.com/intel/sycl-tla/tree/main/examples) apply SYCL*TLA templates to implement basic computations.

### Tools

```
tools/
  library/                   # SYCL*TLA Instance Library - contains instantiations of all supported SYCL*TLA templates
    include/
      cutlass/
        library/

  profiler/                  # Profiler                 - SYCL support not yet available
                             #                            (command-line utility for executing operations)
  
  util/                      # Utilities               - contains numerous helper classes for
    include/                 #                            managing tensors in Intel GPU device memory, reference
      cutlass/               #                            implementations for SYCL GEMM, random initialization
        util/                #                            of tensors, and I/O for Intel GPU environments.
```

### Test

The `test/unit/` directory consist of unit tests implemented with Google Test that demonstrate
basic usage of Core API components and complete tests of the CUTLASS GEMM computations.

Instructions for building and running the Unit tests are described in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).

# Performance Profiling

**Note**: The CUTLASS Profiler is primarily designed for NVIDIA GPUs. For Intel GPU profiling, refer to the SYCL*TLA examples and benchmarks.

The `tools/profiler/` directory contains a command-line utility for launching each of the GEMM kernels.
It can be built as follows:

```bash
$ make cutlass_profiler -j16
```
## Building all GEMM and Convolution kernels (_long_ build times)

By default, only one tile size is instantiated for each data type, math instruction, and layout.
To instantiate all, set the following environment variable when running CMake from an empty `build/` directory.
Beware, this results in *tens of thousands* of kernels and long build times.
This would also result in a large binary size and on some platforms linker to fail on building the library.
Therefore, it's highly recommended to generate only a subset of kernels as demonstrated in the sub-section below.
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=all
...
$ make cutlass_profiler -j16
```

## Building a subset of GEMM and Convolution kernels (_reduced_ build times)

To compile strictly one kernel or a small set of kernels, a comma-delimited list of kernel names with
wildcard characters may be used to reduce the set of kernels. The following examples show building exactly one
or a subset of kernels for NVIDIA Ampere and Turing architecture:

### Building a subset Tensor Core GEMM kernels

To compile a subset of Tensor Core GEMM kernels with FP32 accumulation and FP16 input targeting NVIDIA Ampere and Turing architecture,
use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8
...
$ make cutlass_profiler -j16
```

Example command line for profiling a subset of Tensor Core GEMM kernels is as follows:
```bash
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_*_nt_align8 --m=3456 --n=4096 --k=4096

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          cuBLAS: Passed

       Arguments: --gemm_kind=universal --m=3456 --n=4096 --k=4096 --A=f16:column --B=f16:row --C=f32:column --alpha=1  \
                  --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128  \
                  --cta_k=32 --stages=2 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=8 --min_cc=75  \
                  --max_cc=1024

           Bytes: 118489088  bytes
           FLOPs: 115992428544  flops

         Runtime: 1.55948  ms
          Memory: 70.7616 GiB/s

            Math: 74378.8 GFLOP/s



=============================
...
```

### Building one CUDA Core GEMM kernel

To compile one SGEMM kernel targeting NVIDIA Ampere and Turing architecture, use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sgemm_128x128_8x2_nn_align1
...
$ make cutlass_profiler -j16
```

Example command line for profiling single SGEMM CUDA kernel is as follows:
```bash
$ ./tools/profiler/cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=4096 --A=f32:column --B=f32:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 115992428544  flops

         Runtime: 6.73655  ms
          Memory: 24.934 GiB/s

            Math: 17218.4 GFLOP/s

=============================
```

### Building a subset of Tensor Core Convolution kernels

To compile a subset of Tensor core convolution kernels implementing forward propagation (fprop) with FP32 accumulation
and FP16 input targeting NVIDIA Ampere and Turing architecture, use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*fprop_optimized_f16
...
$ make cutlass_profiler -j16
```

Example command line for profiling a subset of Tensor Core convolution kernels is as follows:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*fprop_optimized_f16 --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_tensorop_s16816fprop_optimized_f16_128x128_32x5_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f16:nhwc --Filter=f16:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=32 --stages=5  \
                  --warps_m=2 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 1130659840  bytes
           FLOPs: 118482796544  flops

         Runtime: 0.711496  ms
          Memory: 1479.99 GiB/s

            Math: 166526 GFLOP/s

=============================
...
```


### Building one Convolution CUDA kernel

To compile and run one CUDA Core convolution kernel implementing forward propagation (fprop) with F32 accumulation
and FP32 input targeting NVIDIA Ampere and Turing architecture, use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sfprop_optimized_128x128_8x2_nhwc
...
$ make cutlass_profiler -j16
```

Example command line for profiling one CUDA Core convolution kernel:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sfprop_optimized_128x128_8x2_nhwc --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3


=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_simt_sfprop_optimized_128x128_8x2_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f32:nhwc --Filter=f32:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 2055798784  bytes
           FLOPs: 118482796544  flops

         Runtime: 7.34266  ms
          Memory: 260.752 GiB/s

            Math: 16136.2 GFLOP/s


=============================

```

## More Details on Compiling CUTLASS Kernels and CUTLASS Profiler
- Please follow the links for more CMake examples on selectively compiling CUTLASS kernels:
  - [GEMM CMake Examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#gemm-cmake-examples)
  - [Implicit GEMM convolution CMake Examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#convolution-cmake-examples)
- [Further details about the CUTLASS Profiler are described here.](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)

# About

SYCL*TLA is released by INTEL Corporation as Open Source software under the
[3-clause "New" BSD license](https://github.com/intel/sycl-tla/blob/main/LICENSE.txt).

# Contributors

The official list of SYCL*TLA developers and contributors is available here: [CONTRIBUTORS](https://github.com/intel/sycl-tla/blob/main/CONTRIBUTORS.md).

# Contributing

## Pull Request Templates

We provide concise PR templates to streamline documentation:

### Quick Start

**GitHub CLI:**
```bash
gh pr create --template .github/PULL_REQUEST_TEMPLATE/bug_fix.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/performance.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/feature.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/refactoring.md
```

**GitHub Web:** Add `?template=<name>.md` to PR URL (e.g., `?template=bug_fix.md`)

### Which Template?

- 🐛 **Bug fixes** → `bug_fix.md` - Root cause + verification
- ⚡ **Performance** → `performance.md` - Profiling data + benchmarks
- ✨ **Features** → `feature.md` - API design + examples
- 🔨 **Refactoring** → `refactoring.md` - Refactored/Redesigned code
- 📝 **Mixed/Other** → Default template

See [`.github/PULL_REQUEST_TEMPLATE`](https://github.com/intel/sycl-tla/tree/main/.github/PULL_REQUEST_TEMPLATE) for details.

# Copyright

Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2025 Intel Corporation. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

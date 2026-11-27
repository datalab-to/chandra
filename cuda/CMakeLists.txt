cmake_minimum_required(VERSION 3.24...3.31)
project(chandra_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# The GPUs to compile for: CHANDRA_CUDA_ARCHS (e.g. "80;86;89;90;120"), else the
# GPU of the machine that builds.
if (DEFINED ENV{CHANDRA_CUDA_ARCHS})
  set(CMAKE_CUDA_ARCHITECTURES "$ENV{CHANDRA_CUDA_ARCHS}")
elseif (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit 12.8 REQUIRED)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI NB_STATIC
  src/cpp/module.cpp
  src/cpp/decode_gemm.cu
  src/cpp/attn_decode.cu
  src/cpp/gdn_decode.cu
  src/cpp/small_ops.cu
  src/cpp/vit_attn.cu)
target_include_directories(_core PRIVATE src/cpp)
# The kernels run on the caller's (torch's) CUDA context and stream.
target_link_libraries(_core PRIVATE CUDA::cudart_static)
target_compile_options(_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr -Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O2>)

install(TARGETS _core LIBRARY DESTINATION chandra_cuda)

cmake_minimum_required(VERSION 3.24)
# FCB_WITH_CUDA=OFF builds the CPU driver and CPU tests only (CI runners have no GPU / CUDA toolkit).
option(FCB_WITH_CUDA "Build the CUDA driver and CUDA tests" ON)
# Must be set before project(): enabling CUDA otherwise fixes the toolkit default (sm_75 PTX), and that PTX cannot
# be JIT-compiled by an older driver than the toolkit (CUDA 13.3 toolkit, 13.0 driver here). sm_120 SASS loads.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120-real)  # RTX 5090 Laptop (sm_120)
endif()
if(FCB_WITH_CUDA)
  project(fcb_mppi LANGUAGES CXX CUDA VERSION 0.1)
else()
  project(fcb_mppi LANGUAGES CXX VERSION 0.1)
endif()

# C++17/CUDA MPPI for Friction-Cliff-Bench (docs/BASELINES.md §5). Builds libfcb_mppi.so (loaded from Python with
# ctypes, zero-copy on Warp/PyTorch device pointers) and the GoogleTest suite.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# No fast math on either side: the port must reproduce Warp's float32 results (Warp's default is fast_math off).
# -ffp-contract=off keeps the CPU build from fusing multiply-adds differently from the reference.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

if(FCB_WITH_CUDA)
  add_library(fcb_mppi SHARED src/fcb_cpu.cpp src/fcb_cuda.cu)
  set_target_properties(fcb_mppi PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
else()
  add_library(fcb_mppi SHARED src/fcb_cpu.cpp)
endif()
target_include_directories(fcb_mppi PUBLIC include)

option(FCB_BUILD_TESTS "Build the GoogleTest suite" ON)
if(FCB_BUILD_TESTS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  if(FCB_WITH_CUDA)
    add_executable(fcb_tests tests/test_vehicle.cpp tests/test_mppi.cpp tests/test_cuda.cu)
  else()
    add_executable(fcb_tests tests/test_vehicle.cpp tests/test_mppi.cpp)
  endif()
  target_link_libraries(fcb_tests PRIVATE fcb_mppi GTest::gtest_main)
  target_compile_definitions(fcb_tests PRIVATE FCB_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  include(GoogleTest)
  gtest_discover_tests(fcb_tests)
endif()

cmake_minimum_required(VERSION 3.16)

# 1. 启用ASC工程
find_package(ASC REQUIRED)
project(HierarchicalKV-ascend
  LANGUAGES ASC CXX)

# 2. cmake基础参数设置
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# 3. hkv依赖库
set(hkv_intf ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(hkv_depends INTERFACE)
target_include_directories(hkv_depends INTERFACE ${hkv_intf})
target_link_directories(hkv_depends INTERFACE /usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend950PR_957b)
target_link_libraries(hkv_depends INTERFACE
  # c++
  m
  dl # platform依赖

  runtime_camodel npu_drv tiling_api platform opapi nnopbase pthread
)
target_compile_options(hkv_depends INTERFACE
  # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
  -g -O0
  # 优化SIMT算子寄存器数
  -mllvm -disable-machine-licm
  # 打印寄存器使用情况
  --cce-res-usage
)

# 5. 编译示例
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hkv_hashtable_benchmark.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark_find_miss.cpp
  PROPERTIES LANGUAGE ASC
)

add_executable(hkv_demo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(hkv_demo PRIVATE hkv_depends)

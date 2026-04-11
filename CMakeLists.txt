cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(Ascend_c LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu" CACHE STRING "sim/npu")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
  CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

add_library(hkv_includes INTERFACE)
target_include_directories(hkv_includes INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_compile_options(-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)

if(ENABLE_TEST STREQUAL 1)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of GTest" FORCE)
  set(INSTALL_GMOCK OFF CACHE BOOL "Disable installation of GMock" FORCE)
  add_subdirectory(3rdparty/gtest)
  add_subdirectory(tests)
endif()

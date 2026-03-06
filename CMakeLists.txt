cmake_minimum_required(VERSION 3.17)
project(grecov LANGUAGES CXX)

find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_ext src/ext/grecov_ext.cpp)

target_compile_features(_ext PRIVATE cxx_std_17)

option(GRECOV_PROFILE "Build with profiling symbols and frame pointers" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Emscripten")
  target_compile_options(_ext PRIVATE -ffast-math)
  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_compile_options(_ext PRIVATE -fexceptions)
    target_link_options(_ext PRIVATE -fexceptions)
    # nanobind's static library must also be compiled with -fexceptions
    # so it can catch C++ exceptions and translate them to Python errors
    if(TARGET nanobind-static)
      target_compile_options(nanobind-static PRIVATE -fexceptions)
    endif()
  else()
    target_compile_options(_ext PRIVATE -march=native)
  endif()
  if(GRECOV_PROFILE)
    target_compile_options(_ext PRIVATE -g -fno-omit-frame-pointer -fno-optimize-sibling-calls)
  endif()
endif()

install(TARGETS _ext LIBRARY DESTINATION grecov)

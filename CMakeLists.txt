cmake_minimum_required(VERSION 3.17)
project(grecov LANGUAGES CXX)

find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_ext src/ext/grecov_ext.cpp)

target_compile_features(_ext PRIVATE cxx_std_17)

# Performance flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(_ext PRIVATE -march=native -ffast-math)
endif()

install(TARGETS _ext LIBRARY DESTINATION grecov)

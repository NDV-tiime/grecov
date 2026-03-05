cmake_minimum_required(VERSION 3.17)
project(grecov LANGUAGES CXX)

find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_ext src/ext/grecov_ext.cpp)

target_compile_features(_ext PRIVATE cxx_std_17)

install(TARGETS _ext LIBRARY DESTINATION grecov)

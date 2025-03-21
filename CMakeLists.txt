cmake_minimum_required(VERSION 3.10)
project(pybind_jax_repro)

find_package(Python3 COMPONENTS Development REQUIRED)

add_subdirectory(pybind11)

add_executable(${PROJECT_NAME}
  main.cpp
  common.hpp
  jaxInterface.cpp
  jaxInterface.hpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed Python3::Python)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
)

target_compile_definitions(${PROJECT_NAME} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

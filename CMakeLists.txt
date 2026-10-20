cmake_minimum_required(VERSION 3.22)
project(torchcli VERSION 0.3.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Two ways to get libtorch:
#   TORCHCLI_LIBTORCH_PREFIX  a static, CPU-only, Python-free libtorch built by scripts/build-libtorch.sh (the release:
#                             one fully static executable with no runtime dependencies)
#   Torch_DIR / CMAKE_PREFIX_PATH  any libtorch CMake package, e.g. the PyTorch wheel's (a development build that loads
#                             libtorch shared libraries at run time)
set(TORCHCLI_LIBTORCH_PREFIX "" CACHE PATH "Prefix of a static libtorch built by scripts/build-libtorch.sh")
option(TORCHCLI_FULL_STATIC "Link the executable fully statically (static libtorch only)" ON)
option(TORCHCLI_PYTHON_TESTS "Register the Python oracle tests (need python3 with torch and safetensors)" OFF)
set(LMCPP_SOURCE_DIR "" CACHE PATH "lmcpp source tree at the pinned revision (scripts/build-native.sh extracts it)")

find_package(nlohmann_json 3.2.0 REQUIRED)
if(LMCPP_SOURCE_DIR)
  set(LMCPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${LMCPP_SOURCE_DIR}" lmcpp EXCLUDE_FROM_ALL)
  set(LMCPP_TARGET lmcpp_core)
else()
  find_package(lmcpp CONFIG REQUIRED)
  set(LMCPP_TARGET lmcpp::core)
endif()

# The DSL schema is compiled in byte for byte: `torchcli --dsl-schema --json` prints it, and bytecode binds its SHA-256.
set(TORCHCLI_DSL_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/schema/torchcli.dsl.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TORCHCLI_DSL_SCHEMA}")
file(READ "${TORCHCLI_DSL_SCHEMA}" TORCHCLI_SCHEMA_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," TORCHCLI_SCHEMA_INITIALIZER "${TORCHCLI_SCHEMA_HEX}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/dsl_schema_embedded.hpp.tmp"
  "#pragma once\n#include <cstddef>\nnamespace torchcli::dsl { inline constexpr unsigned char embedded_dsl_schema[] = {${TORCHCLI_SCHEMA_INITIALIZER}}; inline constexpr std::size_t embedded_dsl_schema_size = sizeof(embedded_dsl_schema); }\n")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/generated/dsl_schema_embedded.hpp.tmp" "${CMAKE_CURRENT_BINARY_DIR}/generated/dsl_schema_embedded.hpp" COPYONLY)

add_executable(torchcli src/main.cpp src/safetensors.cpp src/ops.cpp src/dsl_host.cpp)
target_include_directories(torchcli PRIVATE include third_party/dslvm "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_options(torchcli PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(torchcli PRIVATE ${LMCPP_TARGET})
set_target_properties(torchcli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(TORCHCLI_LIBTORCH_PREFIX)
  if(NOT EXISTS "${TORCHCLI_LIBTORCH_PREFIX}/lib/libtorch_cpu.a")
    message(FATAL_ERROR "TORCHCLI_LIBTORCH_PREFIX has no lib/libtorch_cpu.a: run scripts/build-libtorch.sh ('${TORCHCLI_LIBTORCH_PREFIX}')")
  endif()
  set(P "${TORCHCLI_LIBTORCH_PREFIX}/lib")
  target_include_directories(torchcli SYSTEM PRIVATE "${TORCHCLI_LIBTORCH_PREFIX}/include" "${TORCHCLI_LIBTORCH_PREFIX}/include/torch/csrc/api/include")
  # Operator and dispatcher registrations are static initializers: the libraries that hold them are linked whole.
  set(TORCHCLI_WHOLE "")
  foreach(lib torch torch_cpu c10)
    if(EXISTS "${P}/lib${lib}.a")
      list(APPEND TORCHCLI_WHOLE "${P}/lib${lib}.a")
    endif()
  endforeach()
  file(GLOB TORCHCLI_ARCHIVES "${P}/*.a")
  list(REMOVE_ITEM TORCHCLI_ARCHIVES ${TORCHCLI_WHOLE})
  # Build tools and test-only archives never belong in the executable.
  list(FILTER TORCHCLI_ARCHIVES EXCLUDE REGEX "lib(protoc|gtest|gmock|benchmark|gtest_main|gmock_main|benchmark_main)\\.a$")
  target_link_libraries(torchcli PRIVATE
    "-Wl,--whole-archive" ${TORCHCLI_WHOLE} "-Wl,--no-whole-archive"
    "-Wl,--start-group" ${TORCHCLI_ARCHIVES} "-Wl,--end-group"
    pthread m dl rt)
  target_link_options(torchcli PRIVATE -static-libstdc++ -static-libgcc)
  if(TORCHCLI_FULL_STATIC)
    target_link_options(torchcli PRIVATE -static)
  endif()
  target_compile_definitions(torchcli PRIVATE TORCHCLI_STATIC_LIBTORCH=1)
else()
  find_package(Torch REQUIRED)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(torchcli PRIVATE ${TORCH_CXX_FLAGS_LIST})
  target_link_libraries(torchcli PRIVATE ${TORCH_LIBRARIES})
  get_filename_component(TORCH_PACKAGE_DIR "${TORCH_INSTALL_PREFIX}" ABSOLUTE)
  get_filename_component(TORCH_SITE_PACKAGES "${TORCH_PACKAGE_DIR}" DIRECTORY)
  if(EXISTS "${TORCH_SITE_PACKAGES}/torch.libs")
    target_link_options(torchcli PRIVATE
      "-Wl,-rpath-link,${TORCH_SITE_PACKAGES}/torch.libs"
      "-Wl,-rpath,${TORCH_PACKAGE_DIR}/lib:${TORCH_SITE_PACKAGES}/torch.libs")
  endif()
endif()

enable_testing()
add_executable(torchcli_unit tests/safetensors_test.cpp src/safetensors.cpp)
target_include_directories(torchcli_unit PRIVATE include)
target_compile_options(torchcli_unit PRIVATE -UNDEBUG)
target_link_libraries(torchcli_unit PRIVATE ${LMCPP_TARGET})
add_test(NAME torchcli_unit COMMAND torchcli_unit)
set_tests_properties(torchcli_unit PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
if(TORCHCLI_PYTHON_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME torchcli_integration COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration.py $<TARGET_FILE:torchcli> $<TARGET_FILE:torchcli_unit> ${CMAKE_CURRENT_SOURCE_DIR}/tests/torch_oracle.py)
  # The docker shim test belongs to the legacy image (docker/Dockerfile enables these tests).
  add_test(NAME torchcli_shim COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/shim_test.py ${CMAKE_CURRENT_SOURCE_DIR}/bin/torchcli)
endif()

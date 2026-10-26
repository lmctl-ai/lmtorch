cmake_minimum_required(VERSION 3.22)
# macOS floor: the release binary and the static libtorch it links are both built for this deployment target.
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
# Windows: the static MSVC runtime (/MT) in every target, matching scripts/build-libtorch.sh, so lmtorch.exe needs no
# Visual C++ redistributable.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "MSVC runtime library")
project(lmtorch VERSION 0.5.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Two ways to get libtorch:
#   LMTORCH_LIBTORCH_PREFIX  a static, CPU-only, Python-free libtorch built by scripts/build-libtorch.sh (the release:
#                             one fully static executable with no runtime dependencies)
#   Torch_DIR / CMAKE_PREFIX_PATH  any libtorch CMake package, e.g. the PyTorch wheel's (a development build that loads
#                             libtorch shared libraries at run time)
set(LMTORCH_LIBTORCH_PREFIX "" CACHE PATH "Prefix of a static libtorch built by scripts/build-libtorch.sh")
# Default release form: everything static except glibc (no LGPL relinking duty; runs on glibc >= the build host's).
option(LMTORCH_FULL_STATIC "Also link glibc statically (static libtorch only)" OFF)
option(LMTORCH_PYTHON_TESTS "Register the Python oracle tests (need python3 with torch and safetensors)" OFF)
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

# The DSL schema is compiled in byte for byte: `lmtorch --dsl-schema --json` prints it, and bytecode binds its SHA-256.
set(LMTORCH_DSL_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/schema/lmtorch.dsl.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LMTORCH_DSL_SCHEMA}")
file(READ "${LMTORCH_DSL_SCHEMA}" LMTORCH_SCHEMA_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," LMTORCH_SCHEMA_INITIALIZER "${LMTORCH_SCHEMA_HEX}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/dsl_schema_embedded.hpp.tmp"
  "#pragma once\n#include <cstddef>\nnamespace lmtorch::dsl { inline constexpr unsigned char embedded_dsl_schema[] = {${LMTORCH_SCHEMA_INITIALIZER}}; inline constexpr std::size_t embedded_dsl_schema_size = sizeof(embedded_dsl_schema); }\n")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/generated/dsl_schema_embedded.hpp.tmp" "${CMAKE_CURRENT_BINARY_DIR}/generated/dsl_schema_embedded.hpp" COPYONLY)

add_executable(lmtorch src/main.cpp src/safetensors.cpp src/ops.cpp src/dsl_host.cpp)
# One version: --version (host flag standard, R45 @latest compares it with the release tag) and Torch.version().
target_compile_definitions(lmtorch PRIVATE LMTORCH_VERSION="${PROJECT_VERSION}")
target_include_directories(lmtorch PRIVATE include third_party/dslvm "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(MSVC)
  # /Zc:preprocessor and /permissive- as libtorch is built; /bigobj for the dispatcher-heavy translation units.
  set(LMTORCH_MSVC_FLAGS /O2 /W3 /EHsc /bigobj /permissive- /Zc:preprocessor /Zc:__cplusplus /utf-8 /DNOMINMAX /DWIN32_LEAN_AND_MEAN)
  target_compile_options(lmtorch PRIVATE ${LMTORCH_MSVC_FLAGS})
else()
  target_compile_options(lmtorch PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
endif()
target_link_libraries(lmtorch PRIVATE ${LMCPP_TARGET})
set_target_properties(lmtorch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(LMTORCH_LIBTORCH_PREFIX)
  if(MSVC)
    set(LMTORCH_LIB_PREFIX "")
    set(LMTORCH_LIB_SUFFIX ".lib")
  else()
    set(LMTORCH_LIB_PREFIX "lib")
    set(LMTORCH_LIB_SUFFIX ".a")
  endif()
  if(NOT EXISTS "${LMTORCH_LIBTORCH_PREFIX}/lib/${LMTORCH_LIB_PREFIX}torch_cpu${LMTORCH_LIB_SUFFIX}")
    message(FATAL_ERROR "LMTORCH_LIBTORCH_PREFIX has no lib/${LMTORCH_LIB_PREFIX}torch_cpu${LMTORCH_LIB_SUFFIX}: run scripts/build-libtorch.sh ('${LMTORCH_LIBTORCH_PREFIX}')")
  endif()
  set(P "${LMTORCH_LIBTORCH_PREFIX}/lib")
  target_include_directories(lmtorch SYSTEM PRIVATE "${LMTORCH_LIBTORCH_PREFIX}/include" "${LMTORCH_LIBTORCH_PREFIX}/include/torch/csrc/api/include")
  # Operator and dispatcher registrations are static initializers: the libraries that hold them are linked whole.
  set(LMTORCH_WHOLE "")
  foreach(lib torch torch_cpu c10)
    if(EXISTS "${P}/${LMTORCH_LIB_PREFIX}${lib}${LMTORCH_LIB_SUFFIX}")
      list(APPEND LMTORCH_WHOLE "${P}/${LMTORCH_LIB_PREFIX}${lib}${LMTORCH_LIB_SUFFIX}")
    endif()
  endforeach()
  file(GLOB LMTORCH_ARCHIVES "${P}/*${LMTORCH_LIB_SUFFIX}")
  list(REMOVE_ITEM LMTORCH_ARCHIVES ${LMTORCH_WHOLE})
  # Build tools and test-only archives never belong in the executable.
  list(FILTER LMTORCH_ARCHIVES EXCLUDE REGEX "/(lib)?(protoc|gtest|gmock|benchmark|gtest_main|gmock_main|benchmark_main)(\\.a|\\.lib)$")
  if(MSVC)
    # link.exe searches every library until nothing new resolves, so no grouping; /WHOLEARCHIVE keeps the registrations.
    foreach(lib ${LMTORCH_WHOLE})
      target_link_options(lmtorch PRIVATE "/WHOLEARCHIVE:${lib}")
    endforeach()
    target_link_libraries(lmtorch PRIVATE ${LMTORCH_WHOLE} ${LMTORCH_ARCHIVES})
    # A PDB next to the executable for crash stacks (not shipped); /OPT:REF,ICF keep the image as without /DEBUG.
    target_link_options(lmtorch PRIVATE /DEBUG /OPT:REF /OPT:ICF)
  elseif(APPLE)
    # ld64 rescans archives as needed (no --start-group); -force_load is its --whole-archive. Only libc++ and libSystem
    # stay dynamic: they are part of macOS.
    set(LMTORCH_FORCE_LOAD "")
    foreach(lib ${LMTORCH_WHOLE})
      list(APPEND LMTORCH_FORCE_LOAD "-Wl,-force_load,${lib}")
    endforeach()
    target_link_libraries(lmtorch PRIVATE ${LMTORCH_FORCE_LOAD} ${LMTORCH_ARCHIVES})
  else()
    target_link_libraries(lmtorch PRIVATE
      "-Wl,--whole-archive" ${LMTORCH_WHOLE} "-Wl,--no-whole-archive"
      "-Wl,--start-group" ${LMTORCH_ARCHIVES} "-Wl,--end-group"
      pthread m dl rt)
    target_link_options(lmtorch PRIVATE -static-libstdc++ -static-libgcc)
    if(LMTORCH_FULL_STATIC)
      target_link_options(lmtorch PRIVATE -static)
    endif()
  endif()
  target_compile_definitions(lmtorch PRIVATE LMTORCH_STATIC_LIBTORCH=1)
else()
  find_package(Torch REQUIRED)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(lmtorch PRIVATE ${TORCH_CXX_FLAGS_LIST})
  target_link_libraries(lmtorch PRIVATE ${TORCH_LIBRARIES})
  get_filename_component(TORCH_PACKAGE_DIR "${TORCH_INSTALL_PREFIX}" ABSOLUTE)
  get_filename_component(TORCH_SITE_PACKAGES "${TORCH_PACKAGE_DIR}" DIRECTORY)
  if(EXISTS "${TORCH_SITE_PACKAGES}/torch.libs")
    target_link_options(lmtorch PRIVATE
      "-Wl,-rpath-link,${TORCH_SITE_PACKAGES}/torch.libs"
      "-Wl,-rpath,${TORCH_PACKAGE_DIR}/lib:${TORCH_SITE_PACKAGES}/torch.libs")
  endif()
endif()

enable_testing()
add_executable(lmtorch_unit tests/safetensors_test.cpp src/safetensors.cpp)
target_include_directories(lmtorch_unit PRIVATE include)
if(MSVC)
  target_compile_options(lmtorch_unit PRIVATE /EHsc /utf-8 /UNDEBUG)
else()
  target_compile_options(lmtorch_unit PRIVATE -UNDEBUG)
endif()
target_link_libraries(lmtorch_unit PRIVATE ${LMCPP_TARGET})
add_test(NAME lmtorch_unit COMMAND lmtorch_unit)
set_tests_properties(lmtorch_unit PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
if(LMTORCH_PYTHON_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME lmtorch_integration COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration.py $<TARGET_FILE:lmtorch> $<TARGET_FILE:lmtorch_unit> ${CMAKE_CURRENT_SOURCE_DIR}/tests/torch_oracle.py)
endif()


cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(dynamicparallelism LANGUAGES CXX CUDA)


find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 11)
set(DEFAULT_BUILD_TYPE "Release")


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)

add_executable(dynamicparallelism
  main.cu
  common/fmt.hpp
  common/catch.hpp
  common/clara.hpp
  common/utils.hpp
)



CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

# Request that dynamicparallelism be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(dynamicparallelism PUBLIC cxx_std_11)
target_compile_options(dynamicparallelism PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( dynamicparallelism PROPERTIES
                       CUDA_SEPARABLE_COMPILATION ON
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      )

target_link_libraries(dynamicparallelism ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})


include_directories(dynamicparallelism
  ${PROJECT_SOURCE_DIR}/src
  ${CUDA_INCLUDE_DIRS}
)

if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET dynamicparallelism PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

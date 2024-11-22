cmake_minimum_required(VERSION 3.18)
project(ReactiveTransport2D CUDA CXX)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)

# Find required packages
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Print debug information
message(STATUS "CUDA_FOUND: ${CUDA_FOUND}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "HDF5_FOUND: ${HDF5_FOUND}")
message(STATUS "HDF5_VERSION: ${HDF5_VERSION}")
message(STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")
message(STATUS "HDF5_CXX_INCLUDE_DIRS: ${HDF5_CXX_INCLUDE_DIRS}")
message(STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")

# Source files
set(SOURCES
    src/main.cpp
    src/transport2d.cu
    src/reactions2d.cu
    src/io_utils.cpp
)

# Create executable
add_executable(reactive_transport ${SOURCES})

# Include directories
target_include_directories(reactive_transport
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${HDF5_CXX_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(reactive_transport
    PRIVATE
    ${CUDA_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
)

# CUDA specific properties
set_target_properties(reactive_transport PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optimization flags for Release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(reactive_transport PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reactive_transport PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# Installation
install(TARGETS reactive_transport
    RUNTIME DESTINATION bin
)

# Optional: Add version information
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

# Configure include directories for IDEs
set_target_properties(reactive_transport PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optional: Add testing support
option(BUILD_TESTING "Build tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
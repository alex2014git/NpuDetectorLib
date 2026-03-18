cmake_minimum_required(VERSION 3.10)
# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the project name
project(NpuLibrary VERSION 2.0.0)

# Include common macros and definitions
include(${PROJECT_SOURCE_DIR}/common.cmake)

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
message("HAILORT_INCLUDE: ${HAILORT_INCLUDE}")
message("HAILORT_LIB value: ${HAILORT_LIB}")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS value: ${OpenCV_LIBS}")
message("LETTER_BOX value: ${LETTER_BOX}")
message("SHOW_LABEL value: ${SHOW_LABEL}")
message("TIME_TRACE_DEBUG value: ${TIME_TRACE_DEBUG}")
message("BUILD_TESTER value: ${BUILD_TESTER}")
message("BUILD_VALIDATION_TESTS value: ${BUILD_VALIDATION_TESTS}")

# Add include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${HAILORT_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src)

if (UNIX)
    # Optionally, install the library and headers
    install(TARGETS NpuLibrary
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
    install(DIRECTORY include/ DESTINATION include)
    install(FILES ${PROJECT_SOURCE_DIR}/common.cmake DESTINATION share/NpuLibrary)
endif()
# Test configuration
option(BUILD_TESTER "Build tests (legacy, enables all test categories)" OFF)
option(BUILD_UNIT_TESTS "Build unit tests (requires NpuLibrary)" OFF)
option(BUILD_INTEGRATION_TESTS "Build integration tests (requires NpuLibrary)" OFF)
option(BUILD_HARDWARE_TESTS "Build hardware tests (requires Hailo NPU)" OFF)
option(BUILD_VALIDATION_TESTS "Build validation tests (requires Hailo NPU)" OFF)

# Determine if any tests should be built
set(BUILD_ANY_TESTS OFF)
if(BUILD_TESTER OR BUILD_UNIT_TESTS OR BUILD_INTEGRATION_TESTS OR BUILD_HARDWARE_TESTS OR BUILD_VALIDATION_TESTS)
    set(BUILD_ANY_TESTS ON)
endif()

if(BUILD_ANY_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


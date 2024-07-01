cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NpuLibrary VERSION 2.0.0)

# Include common macros and definitions
include(${PROJECT_SOURCE_DIR}/common.cmake)

message("HAILORT_INCLUDE: ${HAILORT_INCLUDE}")
message("HAILORT_LIB value: ${HAILORT_LIB}")
message("LETTER_BOX value: ${LETTER_BOX}")
message("SHOW_LABEL value: ${SHOW_LABEL}")
message("TIME_TRACE_DEBUG value: ${TIME_TRACE_DEBUG}")
message("BUILD_TESTER value: ${BUILD_TESTER}")
# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

# Add include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    #${HAILORT_INCLUDE}
    #${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src)
if (BUILD_TESTER)
    enable_testing()
    add_subdirectory(tests)
endif()


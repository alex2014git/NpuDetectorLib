cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NpuLibrary)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include common macros and definitions
include(${PROJECT_SOURCE_DIR}/common.cmake)

# Find required packages
find_package(Threads REQUIRED)
#find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

# Add include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${HAILORT_INCLUDE}
    #${OpenCV_INCLUDE_DIRS}
)

# Add source files
file(GLOB SOURCES
    src/*.cpp
    src/yolov8/*.cpp
    src/MultiNetworkPipeline/*.cpp
)

# Create the library
add_library(NpuLibrary SHARED ${SOURCES})

# Link libraries
target_link_libraries(NpuLibrary
    ${HAILORT_LIB}
    ${OpenCV_LIBS}
    Threads::Threads
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Specify the output directory
set_target_properties(NpuLibrary PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)

# Optionally, install the library and headers
install(TARGETS NpuLibrary
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(FILES ${PROJECT_SOURCE_DIR}/common.cmake DESTINATION share/NpuLibrary)

# Create an executable for testing/demo purposes
file(GLOB TEST_SOURCES tests/*.cpp)

add_executable(test ${TEST_SOURCES})
target_link_libraries(test
    NpuLibrary
    ${HAILORT_LIB}
    ${OpenCV_LIBS}
    Threads::Threads
)

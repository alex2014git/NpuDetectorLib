cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NpuLibrary)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (UNIX)
set(HAILORT_INCLUDE "/usr/include/hailo")
set(HAILORT_LIB "/usr/lib/libhailort.so")
else ()
set(HAILORT_ROOT "C:/Program Files/HailoRT")
set(HAILORT_INCLUDE "${HAILORT_ROOT}/include")
set(HAILORT_LIB "${HAILORT_ROOT}/lib/libhailort.lib")
endif (UNIX)

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

# Add compile definitions
target_compile_definitions(NpuLibrary PRIVATE LETTER_BOX)
#target_compile_definitions(NpuLibrary PRIVATE TIME_TRACE_DEBUG)
#target_compile_definitions(NpuLibrary PRIVATE SHOW_LABEL)

# Specify the output directory
set_target_properties(NpuLibrary PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)

# Optionally, install the library and headers
install(TARGETS NpuLibrary
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Create an executable for testing/demo purposes
add_executable(test main.cpp)
target_link_libraries(test
    NpuLibrary
    ${HAILORT_LIB}
    ${OpenCV_LIBS}
    Threads::Threads
)

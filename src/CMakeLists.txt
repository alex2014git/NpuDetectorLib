#cmake_minimum_required(VERSION 3.10)

# Set the project name
#project(NpuLibrary)

# Include common macros and definitions
#include(${PROJECT_SOURCE_DIR}/common.cmake)

#message("HAILORT_INCLUDE: ${HAILORT_INCLUDE}")
#message("HAILORT_LIB value: ${HAILORT_LIB}")
#message("LETTER_BOX value: ${LETTER_BOX}")
#message("SHOW_LABEL value: ${SHOW_LABEL}")
#message("TIME_TRACE_DEBUG value: ${TIME_TRACE_DEBUG}")
#message("BUILD_TESTER value: ${BUILD_TESTER}")
# Set the C++ standard
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
#find_package(Threads REQUIRED)
#find_package(HailoRT REQUIRED)
#find_package(OpenCV REQUIRED)

# Add include directories
include_directories(
    #${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include
    ${HAILORT_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

# Add source files
file(GLOB SOURCES
    *.cpp
    yolov8/*.cpp
    MultiNetworkPipeline/*.cpp
)

# Create the library
add_library(NpuLibrary SHARED ${SOURCES})

# Link libraries
target_link_libraries(NpuLibrary
    ${HAILORT_LIB}
    ${OpenCV_LIBS}
    Threads::Threads
)

if (WIN32)
    target_compile_definitions( NpuLibrary PRIVATE NOMINMAX)
    target_compile_definitions( NpuLibrary PRIVATE WIN32_LEAN_AND_MEAN)
    set_target_properties(NpuLibrary PROPERTIES COMPILE_FLAGS "/MD")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Add compile definitions based on options
if (LETTER_BOX)
    target_compile_definitions(NpuLibrary PRIVATE LETTER_BOX)
endif()

if (TIME_TRACE_DEBUG)
    target_compile_definitions(NpuLibrary PRIVATE TIME_TRACE_DEBUG)
endif()

if (SHOW_LABEL)
    target_compile_definitions(NpuLibrary PRIVATE SHOW_LABEL)
endif()

# Specify the output directory
set_target_properties(NpuLibrary PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)

# Optionally, install the library and headers
#install(TARGETS NpuLibrary
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib)
#install(DIRECTORY include/ DESTINATION include)
#install(FILES ${PROJECT_SOURCE_DIR}/common.cmake DESTINATION share/NpuLibrary)

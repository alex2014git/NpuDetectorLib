add_executable(TestExecutable main.cpp)

# Add include directories
include_directories(
    ${HAILORT_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(TestExecutable
    ${HAILORT_LIB}
    ${OpenCV_LIBS}
    Threads::Threads
    NpuLibrary
)
if (WIN32)
    target_link_directories(TestExecutable PRIVATE ${PROJECT_BINARY_DIR}/lib)
    set_target_properties(TestExecutable PROPERTIES COMPILE_FLAGS "/MD")
endif()
target_link_libraries(TestExecutable opencv_core opencv_imgcodecs opencv_highgui opencv_videoio)

add_test(NAME MyTest COMMAND TestExecutable)

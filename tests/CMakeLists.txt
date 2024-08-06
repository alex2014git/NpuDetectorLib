add_executable(TestExecutable main.cpp)

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
add_test(NAME MyTest COMMAND TestExecutable)

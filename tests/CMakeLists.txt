add_executable(TestExecutable main.cpp)

# Link libraries
target_link_libraries(TestExecutable
    ${HAILORT_LIB}
    ${OpenCV_LIBS}
    Threads::Threads
    NpuLibrary
)
add_test(NAME MyTest COMMAND TestExecutable)

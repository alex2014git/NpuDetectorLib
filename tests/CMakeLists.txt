add_executable(TestExecutable main.cpp)

# Include directories for tests
include_directories(
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/include/async_backend
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
add_test(NAME MyTest COMMAND TestExecutable)

# Validation Tests
if(BUILD_VALIDATION_TESTS)
    # Test 1: Model Loading
    add_executable(TestModelLoading test_model_loading.cpp)
    target_link_libraries(TestModelLoading
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if (WIN32)
        target_link_directories(TestModelLoading PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestModelLoading PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME TestModelLoading COMMAND TestModelLoading)

    # Test 2: Single Inference
    add_executable(TestSingleInference test_single_inference.cpp)
    target_link_libraries(TestSingleInference
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if (WIN32)
        target_link_directories(TestSingleInference PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestSingleInference PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME TestSingleInference COMMAND TestSingleInference)

    # Test 3: Pipeline LPR
    add_executable(TestPipelineLpr test_pipeline_lpr.cpp)
    target_link_libraries(TestPipelineLpr
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if (WIN32)
        target_link_directories(TestPipelineLpr PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestPipelineLpr PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME TestPipelineLpr COMMAND TestPipelineLpr)

    # Test 4: Thread Safety
    add_executable(TestThreadSafety test_thread_safety.cpp)
    target_link_libraries(TestThreadSafety
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if (WIN32)
        target_link_directories(TestThreadSafety PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestThreadSafety PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME TestThreadSafety COMMAND TestThreadSafety)

    # Test 5: Async Backend
    add_executable(TestAsyncBackend test_async_backend.cpp)
    target_link_libraries(TestAsyncBackend
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if (WIN32)
        target_link_directories(TestAsyncBackend PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestAsyncBackend PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME TestAsyncBackend COMMAND TestAsyncBackend)

    # Test 6: Factory Mapping
    add_executable(TestFactoryMapping test_factory_mapping.cpp)
    target_link_libraries(TestFactoryMapping
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if (WIN32)
        target_link_directories(TestFactoryMapping PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestFactoryMapping PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME TestFactoryMapping COMMAND TestFactoryMapping)

    # Test 7: LPR and Classification
    add_executable(TestLprClassification test_lpr_classification.cpp)
    target_link_libraries(TestLprClassification
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if (WIN32)
        target_link_directories(TestLprClassification PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestLprClassification PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME TestLprClassification COMMAND TestLprClassification)
endif()

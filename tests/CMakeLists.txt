# Test build configuration
# Note: All tests currently require NpuLibrary which links HailoRT.
# True hardware-free unit tests require Phase 2 library refactoring.

# Include directories for all tests
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/pipeline
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/include/async_backend
    ${PROJECT_SOURCE_DIR}/src/async_backend
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/mock
)

# ==================== UNIT TESTS ====================
# Placeholder for Phase 2 - currently no true unit tests possible
# without Hailo dependency. Tests here will be added when pipeline/
# components are split from NpuLibrary core.
if(BUILD_UNIT_TESTS)
    message(STATUS "Unit tests: reserved for Phase 2 (library splitting)")
    # Future: add_executable(TestPipelineGraph unit/test_pipeline_graph.cpp)
    # Future: target_link_libraries(TestPipelineGraph PipelineCore) # No Hailo
endif()

# ==================== INTEGRATION TESTS ====================
# Tests that link NpuLibrary but don't require Hailo hardware at runtime
# Placeholder for Phase 2 - currently all integration tests need hardware
if(BUILD_INTEGRATION_TESTS)
    message(STATUS "Integration tests: reserved for Phase 2")
    # Future: Tests with mock backend injection
endif()

# ==================== HARDWARE TESTS ====================
# All current tests require Hailo NPU hardware
if(BUILD_HARDWARE_TESTS OR BUILD_TESTER OR BUILD_VALIDATION_TESTS)
    message(STATUS "Building hardware tests (Hailo NPU required)")

    # Main test executable (keeps original name for compatibility)
    add_executable(TestExecutable hardware/main.cpp)
    target_link_libraries(TestExecutable
        ${HAILORT_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
        NpuLibrary
    )
    if(WIN32)
        target_link_directories(TestExecutable PRIVATE ${PROJECT_BINARY_DIR}/lib)
        set_target_properties(TestExecutable PROPERTIES COMPILE_FLAGS "/MD")
    endif()
    add_test(NAME hardware_main COMMAND TestExecutable)

    # Individual validation tests
    set(HARDWARE_TESTS
        test_model_loading
        test_single_inference
        test_thread_safety
        test_lpr_classification
        test_pipeline_lpr
        test_async_backend
        test_factory_mapping
        test_lpr_decoder
        test_classification_decoder
    )

    foreach(test_name ${HARDWARE_TESTS})
        add_executable(${test_name} hardware/${test_name}.cpp)
        target_link_libraries(${test_name}
            ${HAILORT_LIB}
            ${OpenCV_LIBS}
            Threads::Threads
            gtest
            NpuLibrary
        )
        if(WIN32)
            target_link_directories(${test_name} PRIVATE ${PROJECT_BINARY_DIR}/lib)
            set_target_properties(${test_name} PROPERTIES COMPILE_FLAGS "/MD")
        endif()
        add_test(NAME hardware_${test_name} COMMAND ${test_name})
    endforeach()
endif()

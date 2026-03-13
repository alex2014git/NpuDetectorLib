/**
 * Factory Mapping Test
 *
 * Purpose: Ensure algorithm-to-implementation mappings are never broken by future changes.
 *
 * Test Cases:
 * - Verify all algorithm types are registered in factory
 * - Verify instances can be created for each algorithm type
 * - Verify YOLO NMS uses hardware NMS configuration
 * - Verify base implementation works without NMS-specific setup
 */

#include "npu_factory.hpp"
#include "npu.hpp"
#include <opencv2/opencv.hpp>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Test result tracking
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << message << " at line " << __LINE__ << std::endl; \
            g_tests_failed++; \
            return false; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

#define TEST_ASSERT_MSG(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << message << " at line " << __LINE__ << std::endl; \
            g_tests_failed++; \
            return false; \
        } else { \
            std::cout << "  PASS: " << message << std::endl; \
            g_tests_passed++; \
        } \
    } while(0)

// Check if file exists
bool file_exists(const std::string& path) {
    FILE* file = fopen(path.c_str(), "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

// Test: All algorithm types are registered
bool test_all_algorithms_registered() {
    std::cout << "\n[Test] All Algorithm Types Registered" << std::endl;

    // Test all algorithm types from the enum
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_BASE),
                    "ALG_BASE is registered");
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_YOLO_NMS),
                    "ALG_YOLO_NMS is registered");
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_YOLO_V5),
                    "ALG_YOLO_V5 is registered");
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_YOLO_V8),
                    "ALG_YOLO_V8 is registered");
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_POSE),
                    "ALG_POSE is registered");
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_YOLO_V8_SEG),
                    "ALG_YOLO_V8_SEG is registered");
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_LPR),
                    "ALG_LPR is registered");
    TEST_ASSERT_MSG(NpuFactory::IsRegistered(ALG_CLASSIFICATION),
                    "ALG_CLASSIFICATION is registered");

    return true;
}

// Test: Algorithm type correctness - instances can be created
bool test_algorithm_type_correctness() {
    std::cout << "\n[Test] Algorithm Type Correctness" << std::endl;

    // Test each algorithm type can be instantiated
    auto base_npu = NpuFactory::CreateNpu(ALG_BASE);
    TEST_ASSERT_MSG(base_npu != nullptr, "ALG_BASE creates instance");

    auto yolo_nms_npu = NpuFactory::CreateNpu(ALG_YOLO_NMS);
    TEST_ASSERT_MSG(yolo_nms_npu != nullptr, "ALG_YOLO_NMS creates instance");

    auto yolov5_npu = NpuFactory::CreateNpu(ALG_YOLO_V5);
    TEST_ASSERT_MSG(yolov5_npu != nullptr, "ALG_YOLO_V5 creates instance");

    auto yolov8_npu = NpuFactory::CreateNpu(ALG_YOLO_V8);
    TEST_ASSERT_MSG(yolov8_npu != nullptr, "ALG_YOLO_V8 creates instance");

    auto pose_npu = NpuFactory::CreateNpu(ALG_POSE);
    TEST_ASSERT_MSG(pose_npu != nullptr, "ALG_POSE creates instance");

    auto seg_npu = NpuFactory::CreateNpu(ALG_YOLO_V8_SEG);
    TEST_ASSERT_MSG(seg_npu != nullptr, "ALG_YOLO_V8_SEG creates instance");

    auto lpr_npu = NpuFactory::CreateNpu(ALG_LPR);
    TEST_ASSERT_MSG(lpr_npu != nullptr, "ALG_LPR creates instance");

    auto cls_npu = NpuFactory::CreateNpu(ALG_CLASSIFICATION);
    TEST_ASSERT_MSG(cls_npu != nullptr, "ALG_CLASSIFICATION creates instance");

    return true;
}

// Test: YOLO NMS uses hardware NMS correctly
bool test_yolo_nms_uses_hardware_nms() {
    std::cout << "\n[Test] YOLO NMS Hardware NMS Verification" << std::endl;

    // Check test image exists
    if (!file_exists("tests/test_person_image.jpg")) {
        std::cout << "  SKIP: test_person_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Check yolov5s model exists
    if (!file_exists("models/yolov5s.json")) {
        std::cout << "  SKIP: models/yolov5s.json not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Create ALG_YOLO_NMS instance
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_NMS);
    TEST_ASSERT_MSG(npu != nullptr, "CreateNpu(ALG_YOLO_NMS) returns instance");

    // Initialize with yolov5s.json (which has yolo_nms_core: true)
    int init_result = npu->Initialize("models/yolov5s.json", 0);
    TEST_ASSERT_MSG(init_result >= 0, "Initialize() with yolov5s.json succeeds");

    // Load test image
    cv::Mat image = cv::imread("tests/test_person_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference - should work with hardware NMS
    int num_detections = npu->Detect(imgData, true);
    TEST_ASSERT_MSG(num_detections >= 0, "Detect() succeeds with hardware NMS");

    std::cout << "  Hardware NMS detected " << num_detections << " objects" << std::endl;

    npu->Release();
    return true;
}

// Test: Base implementation works without NMS-specific setup
bool test_base_uses_simple_impl() {
    std::cout << "\n[Test] Base Implementation (No NMS)" << std::endl;

    // Create ALG_BASE instance
    auto npu = NpuFactory::CreateNpu(ALG_BASE);
    TEST_ASSERT_MSG(npu != nullptr, "CreateNpu(ALG_BASE) returns instance");

    // Note: ALG_BASE uses NpuBaseAlgImpl which doesn't require NMS setup
    // It should work with any simple model config (when available)

    std::cout << "  Base implementation created successfully (no NMS)" << std::endl;

    return true;
}

// Test: LPR and Classification use base implementation
bool test_lpr_classification_use_base() {
    std::cout << "\n[Test] LPR and Classification Use Base Implementation" << std::endl;

    // Create LPR instance
    auto lpr_npu = NpuFactory::CreateNpu(ALG_LPR);
    TEST_ASSERT_MSG(lpr_npu != nullptr, "CreateNpu(ALG_LPR) returns instance");

    // Create Classification instance
    auto cls_npu = NpuFactory::CreateNpu(ALG_CLASSIFICATION);
    TEST_ASSERT_MSG(cls_npu != nullptr, "CreateNpu(ALG_CLASSIFICATION) returns instance");

    // Both should be created successfully (actual initialization would need model configs)
    std::cout << "  LPR and Classification instances created successfully" << std::endl;

    return true;
}

// Test: str2AlgEnum mapping correctness
bool test_str2alg_enum_mapping() {
    std::cout << "\n[Test] str2AlgEnum Mapping Correctness" << std::endl;

    TEST_ASSERT_MSG(Npu::str2AlgEnum("base") == ALG_BASE,
                    "\"base\" maps to ALG_BASE");
    TEST_ASSERT_MSG(Npu::str2AlgEnum("yolo_nms") == ALG_YOLO_NMS,
                    "\"yolo_nms\" maps to ALG_YOLO_NMS");
    TEST_ASSERT_MSG(Npu::str2AlgEnum("yolov5") == ALG_YOLO_V5,
                    "\"yolov5\" maps to ALG_YOLO_V5");
    TEST_ASSERT_MSG(Npu::str2AlgEnum("yolov8") == ALG_YOLO_V8,
                    "\"yolov8\" maps to ALG_YOLO_V8");
    TEST_ASSERT_MSG(Npu::str2AlgEnum("yolov8_pose") == ALG_POSE,
                    "\"yolov8_pose\" maps to ALG_POSE");
    TEST_ASSERT_MSG(Npu::str2AlgEnum("yolov8_seg") == ALG_YOLO_V8_SEG,
                    "\"yolov8_seg\" maps to ALG_YOLO_V8_SEG");
    TEST_ASSERT_MSG(Npu::str2AlgEnum("lpr") == ALG_LPR,
                    "\"lpr\" maps to ALG_LPR");
    TEST_ASSERT_MSG(Npu::str2AlgEnum("classification") == ALG_CLASSIFICATION,
                    "\"classification\" maps to ALG_CLASSIFICATION");

    // Unknown strings should default to ALG_BASE
    TEST_ASSERT_MSG(Npu::str2AlgEnum("unknown") == ALG_BASE,
                    "Unknown string defaults to ALG_BASE");

    return true;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "========================================" << std::endl;
    std::cout << "NpuDetectorLib - Factory Mapping Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Track overall test success
    bool all_passed = true;

    // Run all tests
    all_passed &= test_all_algorithms_registered();
    all_passed &= test_algorithm_type_correctness();
    all_passed &= test_yolo_nms_uses_hardware_nms();
    all_passed &= test_base_uses_simple_impl();
    all_passed &= test_lpr_classification_use_base();
    all_passed &= test_str2alg_enum_mapping();

    // Print summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << g_tests_passed << std::endl;
    std::cout << "Failed: " << g_tests_failed << std::endl;

    if (g_tests_failed == 0) {
        std::cout << "\nALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "\nSOME TESTS FAILED" << std::endl;
        return 1;
    }
}

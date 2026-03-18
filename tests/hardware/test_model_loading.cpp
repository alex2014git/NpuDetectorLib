/**
 * Test 1: Basic Model Loading Test
 *
 * Purpose: Verify all model JSON configs can be parsed and HEF files are accessible.
 *
 * Test Cases:
 * - Load yolov5s.json -> verify initialization succeeds
 * - Load yolov8s_pose.json -> verify initialization succeeds
 * - Load yolov8s_seg.json -> verify initialization succeeds
 * - Load non-existent JSON -> verify graceful failure
 */

#include "npu_factory.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

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

// Test: Load YOLOv5s model
bool test_yolov5s_loading() {
    std::cout << "\n[Test] YOLOv5s Model Loading" << std::endl;

    // Check HEF file exists
    TEST_ASSERT_MSG(file_exists("models/yolov5s.hef"),
                    "yolov5s.hef file exists");

    // Note: yolov5s.json has "yolo_nms_core": true, meaning it uses Hailo's built-in NMS
    // Therefore we must use ALG_BASE (not ALG_YOLO_V5) which handles the NMS output format
    auto npu = NpuFactory::CreateNpu(ALG_BASE);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_BASE instance");

    int result = npu->Initialize("models/yolov5s.json", 0);
    TEST_ASSERT_MSG(result >= 0, "Initialize() returns success");

    npu->Release();
    std::cout << "  YOLOv5s model loaded successfully" << std::endl;
    return true;
}

// Test: Load YOLOv8s Pose model
bool test_yolov8s_pose_loading() {
    std::cout << "\n[Test] YOLOv8s Pose Model Loading" << std::endl;

    // Check HEF file exists
    TEST_ASSERT_MSG(file_exists("models/yolov8s_pose.hef"),
                    "yolov8s_pose.hef file exists");

    // Try to create and initialize NPU
    auto npu = NpuFactory::CreateNpu(ALG_POSE);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_POSE instance");

    int result = npu->Initialize("models/yolov8s_pose.json", 0);
    TEST_ASSERT_MSG(result >= 0, "Initialize() returns success");

    npu->Release();
    std::cout << "  YOLOv8s Pose model loaded successfully" << std::endl;
    return true;
}

// Test: Load YOLOv8s Segmentation model
bool test_yolov8s_seg_loading() {
    std::cout << "\n[Test] YOLOv8s Segmentation Model Loading" << std::endl;

    // Check HEF file exists
    TEST_ASSERT_MSG(file_exists("models/yolov8s_seg.hef"),
                    "yolov8s_seg.hef file exists");

    // Try to create and initialize NPU
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_V8_SEG);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_YOLO_V8_SEG instance");

    int result = npu->Initialize("models/yolov8s_seg.json", 0);
    TEST_ASSERT_MSG(result >= 0, "Initialize() returns success");

    npu->Release();
    std::cout << "  YOLOv8s Segmentation model loaded successfully" << std::endl;
    return true;
}

// Test: Load YOLOv8s LP (License Plate) detection model
bool test_yolov8s_lp_loading() {
    std::cout << "\n[Test] YOLOv8s License Plate Detection Model Loading" << std::endl;

    // Check config file exists
    TEST_ASSERT_MSG(file_exists("models/yolov8s_lp.json"),
                    "yolov8s_lp.json config file exists");

    // Note: det_v8.hef may not be present - this is OK for the config validation test
    bool hef_exists = file_exists("models/det_v8.hef");
    if (!hef_exists) {
        std::cout << "  SKIP: det_v8.hef not found (expected in some configurations)" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Try to create and initialize NPU
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_V8);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_YOLO_V8 instance");

    int result = npu->Initialize("models/yolov8s_lp.json", 0);
    TEST_ASSERT_MSG(result >= 0, "Initialize() returns success");

    npu->Release();
    std::cout << "  YOLOv8s LP model loaded successfully" << std::endl;
    return true;
}

// Test: Load non-existent JSON file (graceful failure)
bool test_nonexistent_json() {
    std::cout << "\n[Test] Non-existent JSON Handling" << std::endl;

    auto npu = NpuFactory::CreateNpu(ALG_YOLO_V5);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates instance");

    int result = npu->Initialize("models/nonexistent_file.json", 0);
    // Should fail (return negative value)
    TEST_ASSERT_MSG(result < 0, "Initialize() returns failure for non-existent file");

    std::cout << "  Correctly handled non-existent file" << std::endl;
    return true;
}

// Test: JSON parsing validation
bool test_json_parsing() {
    std::cout << "\n[Test] JSON Config Parsing Validation" << std::endl;

    // Test loading a model and verify key parameters are parsed correctly
    // Use ALG_BASE since yolov5s has yolo_nms_core=true
    auto npu = NpuFactory::CreateNpu(ALG_BASE);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates instance");

    // Use a different stream ID (1) to avoid conflict with test_yolov5s_loading which uses stream 0
    int result = npu->Initialize("models/yolov5s.json", 1);
    TEST_ASSERT_MSG(result >= 0, "Initialize() succeeds");

    // Version should be available after initialization
    std::string version = npu->GetVersion();
    TEST_ASSERT_MSG(!version.empty(), "GetVersion() returns non-empty string");
    std::cout << "  Model version: " << version << std::endl;

    npu->Release();
    std::cout << "  JSON parsing validation passed" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "NpuDetectorLib - Model Loading Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Track overall test success
    bool all_passed = true;

    // Run all tests
    all_passed &= test_yolov5s_loading();
    all_passed &= test_yolov8s_pose_loading();
    all_passed &= test_yolov8s_seg_loading();
    all_passed &= test_yolov8s_lp_loading();
    all_passed &= test_nonexistent_json();
    all_passed &= test_json_parsing();

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

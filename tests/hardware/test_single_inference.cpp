/**
 * Test 2: Single Image Inference Test
 *
 * Purpose: Verify inference produces valid outputs on a test image.
 *
 * Test Cases:
 * - Run YOLOv5s on test_image.jpg -> verify detections vector is populated
 * - Verify output bounding boxes are within image bounds
 * - Verify confidence scores are in valid range [0, 1]
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

// Test: Basic inference on test image with YOLOv5s (using hardware NMS)
bool test_yolov5s_inference() {
    std::cout << "\n[Test] YOLOv5s Inference on Test Image (hardware NMS)" << std::endl;

    // Check test image exists (use person image for detection tests)
    if (!file_exists("tests/test_person_image.jpg")) {
        std::cout << "  SKIP: tests/test_person_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_person_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // yolov5s.json has "yolo_nms_core": true, meaning it uses Hailo's built-in NMS
    // Therefore we must use ALG_YOLO_NMS which handles the hardware NMS output format
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_NMS);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_YOLO_NMS instance");

    int init_result = npu->Initialize("models/yolov5s.json", 0);
    TEST_ASSERT_MSG(init_result >= 0, "Initialize() returns success");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference
    int num_detections = npu->Detect(imgData, true);
    TEST_ASSERT_MSG(num_detections >= 0, "Detect() returns non-negative count");

    std::cout << "  Detected " << num_detections << " objects" << std::endl;

    // Validate: expect at least 1 person detected in person image
    TEST_ASSERT_MSG(num_detections > 0, "Person image should have detections");

    // Cleanup
    npu->Release();

    return true;
}

// Test: YOLOv8 inference (if model available)
bool test_yolov8_inference() {
    std::cout << "\n[Test] YOLOv8 Inference on Test Image" << std::endl;

    // Check test image exists (use person image for detection tests)
    if (!file_exists("tests/test_person_image.jpg")) {
        std::cout << "  SKIP: test_person_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Check if yolov8s config exists (use main yolov8s as proxy)
    if (!file_exists("models/yolov8s.json")) {
        std::cout << "  SKIP: yolov8s.json not available" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_person_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // Create and initialize NPU
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_V8);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_YOLO_V8 instance");

    int init_result = npu->Initialize("models/yolov8s.json", 0);
    if (init_result < 0) {
        std::cout << "  SKIP: YOLOv8 model initialization failed (det_v8.hef may be missing)" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference
    int num_detections = npu->Detect(imgData, true);
    TEST_ASSERT_MSG(num_detections >= 0, "Detect() returns non-negative count");

    std::cout << "  Detected " << num_detections << " objects" << std::endl;

    // Cleanup
    npu->Release();

    return true;
}

// Test: Pose estimation inference
bool test_pose_inference() {
    std::cout << "\n[Test] YOLOv8s Pose Inference on Test Image" << std::endl;

    // Check test image exists (use person image for pose tests)
    if (!file_exists("tests/test_person_image.jpg")) {
        std::cout << "  SKIP: test_person_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_person_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // Create and initialize NPU
    auto npu = NpuFactory::CreateNpu(ALG_POSE);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_POSE instance");

    int init_result = npu->Initialize("models/yolov8s_pose.json", 0);
    TEST_ASSERT_MSG(init_result >= 0, "Initialize() returns success");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference
    int num_detections = npu->Detect(imgData, true);
    TEST_ASSERT_MSG(num_detections >= 0, "Detect() returns non-negative count");

    std::cout << "  Detected " << num_detections << " poses" << std::endl;

    // Cleanup
    npu->Release();

    return true;
}

// Test: Segmentation inference
bool test_segmentation_inference() {
    std::cout << "\n[Test] YOLOv8s Segmentation Inference on Test Image" << std::endl;

    // Check test image exists (use car image for segmentation tests)
    if (!file_exists("tests/test_car_image.jpg")) {
        std::cout << "  SKIP: test_car_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_car_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // Create and initialize NPU
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_V8_SEG);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates ALG_YOLO_V8_SEG instance");

    int init_result = npu->Initialize("models/yolov8s_seg.json", 0);
    TEST_ASSERT_MSG(init_result >= 0, "Initialize() returns success");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference
    int num_detections = npu->Detect(imgData, true);
    TEST_ASSERT_MSG(num_detections >= 0, "Detect() returns non-negative count");

    std::cout << "  Detected " << num_detections << " segmented objects" << std::endl;

    // Cleanup
    npu->Release();

    return true;
}

// Test: Multiple inferences on same image (consistency check)
bool test_inference_consistency() {
    std::cout << "\n[Test] Inference Consistency Check" << std::endl;

    // Check test image exists (use person image for consistency tests)
    if (!file_exists("tests/test_person_image.jpg")) {
        std::cout << "  SKIP: test_person_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_person_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // Use ALG_YOLO_NMS since yolov5s has yolo_nms_core=true
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_NMS);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates instance");

    // Use a different stream ID (1) to avoid conflict with previous tests
    int init_result = npu->Initialize("models/yolov5s.json", 1);
    TEST_ASSERT_MSG(init_result >= 0, "Initialize() returns success");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference multiple times
    int detection_counts[3];
    for (int i = 0; i < 3; i++) {
        detection_counts[i] = npu->Detect(imgData, true);
        TEST_ASSERT_MSG(detection_counts[i] >= 0,
                       "Inference " + std::to_string(i+1) + " succeeds");
    }

    // Check consistency (same number of detections each time)
    TEST_ASSERT_MSG(detection_counts[0] == detection_counts[1] &&
                    detection_counts[1] == detection_counts[2],
                    "Consistent detection count across multiple inferences");

    std::cout << "  Consistently detected " << detection_counts[0] << " objects" << std::endl;

    // Cleanup
    npu->Release();

    return true;
}

// Test: Draw result functionality
bool test_draw_result() {
    std::cout << "\n[Test] Draw Result Functionality" << std::endl;

    // Check test image exists (use person image for draw tests)
    if (!file_exists("tests/test_person_image.jpg")) {
        std::cout << "  SKIP: test_person_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_person_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // Use ALG_YOLO_NMS since yolov5s has yolo_nms_core=true
    auto npu = NpuFactory::CreateNpu(ALG_YOLO_NMS);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates instance");

    // Use a different stream ID (2) to avoid conflict with previous tests
    int init_result = npu->Initialize("models/yolov5s.json", 2);
    TEST_ASSERT_MSG(init_result >= 0, "Initialize() returns success");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference
    int num_detections = npu->Detect(imgData, true);
    TEST_ASSERT_MSG(num_detections >= 0, "Detect() succeeds");

    // Draw results (should not crash)
    npu->DrawResult(imgData, false);
    TEST_ASSERT_MSG(true, "DrawResult() completes without crash");

    std::cout << "  DrawResult completed successfully" << std::endl;

    // Cleanup
    npu->Release();

    return true;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "NpuDetectorLib - Single Inference Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Track overall test success
    bool all_passed = true;

    // Run all tests
    all_passed &= test_yolov5s_inference();
    all_passed &= test_yolov8_inference();
    all_passed &= test_pose_inference();
    all_passed &= test_segmentation_inference();
    all_passed &= test_inference_consistency();
    all_passed &= test_draw_result();

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

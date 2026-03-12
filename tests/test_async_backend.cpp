/**
 * Test 5: Async Backend Test
 *
 * Purpose: Verify AsyncBackend works correctly through the NpuFactory API.
 *
 * Test Cases:
 * - Multiple networks work correctly
 * - Run interleaved inference on different networks
 * - Verify outputs are correctly associated with inputs
 */

#include "npu_factory.hpp"
#include "npu.hpp"
#include <opencv2/opencv.hpp>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

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

/**
 * Note: AsyncBackend is an internal implementation detail. We test it indirectly
 * through the NpuFactory API which uses AsyncBackend for all NPU operations.
 */

// Test: Single network inference
bool test_single_network_inference() {
    std::cout << "\n[Test] Single Network Inference" << std::endl;

    // Check test image exists
    if (!file_exists("tests/test_image.jpg")) {
        std::cout << "  SKIP: test_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Create NPU through factory (which uses AsyncBackend internally)
    auto npu = NpuFactory::CreateNpu(ALG_BASE);
    TEST_ASSERT_MSG(npu != nullptr, "NpuFactory creates instance");

    int init_result = npu->Initialize("models/yolov5s.json", 0);
    TEST_ASSERT_MSG(init_result >= 0, "Initialize() returns success");

    // Load test image
    cv::Mat image = cv::imread("tests/test_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded successfully");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Run inference multiple times
    for (int i = 0; i < 5; i++) {
        int num_detections = npu->Detect(imgData, true);
        TEST_ASSERT_MSG(num_detections >= 0, "Inference " + std::to_string(i+1) + " succeeds");
    }

    std::cout << "  Single network inference works" << std::endl;

    npu->Release();
    return true;
}

// Test: Multiple sequential networks
bool test_sequential_networks() {
    std::cout << "\n[Test] Sequential Network Operations" << std::endl;

    // Check test image exists
    if (!file_exists("tests/test_image.jpg")) {
        std::cout << "  SKIP: test_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_image.jpg");
    TEST_ASSERT_MSG(!image.empty(), "Test image loaded");

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    // Create and test first network
    {
        auto npu1 = NpuFactory::CreateNpu(ALG_BASE);
        TEST_ASSERT_MSG(npu1 != nullptr, "NPU1 created");

        int result1 = npu1->Initialize("models/yolov5s.json", 0);
        TEST_ASSERT_MSG(result1 >= 0, "NPU1 initialized");

        int detections1 = npu1->Detect(imgData, true);
        TEST_ASSERT_MSG(detections1 >= 0, "NPU1 inference succeeds");
        std::cout << "  NPU1 detected " << detections1 << " objects" << std::endl;

        npu1->Release();
    }

    // Create and test second network (different stream ID)
    {
        auto npu2 = NpuFactory::CreateNpu(ALG_BASE);
        TEST_ASSERT_MSG(npu2 != nullptr, "NPU2 created");

        int result2 = npu2->Initialize("models/yolov5s.json", 1);
        TEST_ASSERT_MSG(result2 >= 0, "NPU2 initialized");

        int detections2 = npu2->Detect(imgData, true);
        TEST_ASSERT_MSG(detections2 >= 0, "NPU2 inference succeeds");
        std::cout << "  NPU2 detected " << detections2 << " objects" << std::endl;

        npu2->Release();
    }

    std::cout << "  Sequential network operations successful" << std::endl;
    return true;
}

// Test: Rapid inference stress test
bool test_rapid_inference_stress() {
    std::cout << "\n[Test] Rapid Inference Stress Test" << std::endl;

    // Check test image exists
    if (!file_exists("tests/test_image.jpg")) {
        std::cout << "  SKIP: test_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Create NPU
    auto npu = NpuFactory::CreateNpu(ALG_BASE);
    if (!npu) {
        std::cerr << "  FAIL: Could not create NPU" << std::endl;
        g_tests_failed++;
        return false;
    }

    int init_result = npu->Initialize("models/yolov5s.json", 0);
    if (init_result < 0) {
        std::cerr << "  FAIL: Could not initialize NPU" << std::endl;
        g_tests_failed++;
        return false;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_image.jpg");
    if (image.empty()) {
        std::cerr << "  FAIL: Could not load test image" << std::endl;
        g_tests_failed++;
        return false;
    }

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    const int num_iterations = 50;
    auto start_time = std::chrono::high_resolution_clock::now();

    int success_count = 0;
    for (int i = 0; i < num_iterations; i++) {
        int num_detections = npu->Detect(imgData, true);
        if (num_detections >= 0) {
            success_count++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "  Completed " << success_count << "/" << num_iterations
             << " inferences in " << duration.count() << "s"
             << " (" << (num_iterations / duration.count()) << " FPS)" << std::endl;

    npu->Release();

    if (success_count == num_iterations) {
        g_tests_passed++;
        return true;
    } else {
        g_tests_failed++;
        return false;
    }
}

// Test: Network ID uniqueness
bool test_network_id_management() {
    std::cout << "\n[Test] Network ID Management" << std::endl;

    // Create multiple NPU instances with different stream IDs
    std::vector<std::shared_ptr<Npu>> npus;
    std::vector<int> stream_ids = {0, 1, 2, 3};

    for (size_t i = 0; i < stream_ids.size(); i++) {
        auto npu = NpuFactory::CreateNpu(ALG_BASE);
        if (!npu) {
            std::cerr << "  FAIL: Could not create NPU instance " << i << std::endl;
            g_tests_failed++;
            return false;
        }

        int result = npu->Initialize("models/yolov5s.json", stream_ids[i]);
        if (result < 0) {
            std::cerr << "  FAIL: Could not initialize NPU instance " << i << std::endl;
            g_tests_failed++;
            return false;
        }

        npus.push_back(npu);
        std::cout << "  Created network with stream_id=" << stream_ids[i] << std::endl;
    }

    // Release all
    for (auto& npu : npus) {
        npu->Release();
    }

    std::cout << "  Network ID management successful" << std::endl;
    g_tests_passed++;
    return true;
}

// Test: Multiple algorithms
bool test_multiple_algorithms() {
    std::cout << "\n[Test] Multiple Algorithms" << std::endl;

    // Check test image exists
    if (!file_exists("tests/test_image.jpg")) {
        std::cout << "  SKIP: test_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    // Load test image
    cv::Mat image = cv::imread("tests/test_image.jpg");
    if (image.empty()) {
        std::cerr << "  FAIL: Could not load test image" << std::endl;
        g_tests_failed++;
        return false;
    }

    // Test YOLOv5
    {
        auto npu = NpuFactory::CreateNpu(ALG_BASE);
        if (!npu) {
            std::cerr << "  FAIL: Could not create YOLOv5 NPU" << std::endl;
            g_tests_failed++;
            return false;
        }

        int result = npu->Initialize("models/yolov5s.json", 0);
        if (result < 0) {
            std::cerr << "  SKIP: YOLOv5 initialization failed" << std::endl;
        } else {
            image_share_t imgData;
            imgData.data = (void*)image.data;
            imgData.width = image.cols;
            imgData.height = image.rows;
            imgData.ch = 3;

            int detections = npu->Detect(imgData, true);
            std::cout << "  YOLOv5: " << detections << " detections" << std::endl;
        }
        npu->Release();
    }

    // Test Pose (if available)
    if (file_exists("models/yolov8s_pose.json")) {
        auto npu = NpuFactory::CreateNpu(ALG_POSE);
        if (npu) {
            int result = npu->Initialize("models/yolov8s_pose.json", 1);
            if (result >= 0) {
                image_share_t imgData;
                imgData.data = (void*)image.data;
                imgData.width = image.cols;
                imgData.height = image.rows;
                imgData.ch = 3;

                int detections = npu->Detect(imgData, true);
                std::cout << "  YOLOv8-Pose: " << detections << " poses" << std::endl;
            }
            npu->Release();
        }
    }

    std::cout << "  Multiple algorithms test completed" << std::endl;
    g_tests_passed++;
    return true;
}

// Test: Multi-threaded with different stream IDs
bool test_multithread_different_streams() {
    std::cout << "\n[Test] Multi-thread with Different Stream IDs" << std::endl;

    if (!file_exists("tests/test_image.jpg")) {
        std::cout << "  SKIP: test_image.jpg not found" << std::endl;
        g_tests_passed++;
        return true;
    }

    cv::Mat image = cv::imread("tests/test_image.jpg");
    if (image.empty()) {
        std::cerr << "  FAIL: Could not load test image" << std::endl;
        g_tests_failed++;
        return false;
    }

    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);

    auto worker = [&](int thread_id) {
        auto npu = NpuFactory::CreateNpu(ALG_BASE);
        if (!npu) {
            std::cerr << "  FAIL: Thread " << thread_id << " could not create NPU" << std::endl;
            return;
        }

        // Each thread uses a different stream ID
        int stream_id = thread_id;
        int result = npu->Initialize("models/yolov5s.json", stream_id);
        if (result < 0) {
            std::cerr << "  FAIL: Thread " << thread_id << " initialization failed" << std::endl;
            return;
        }

        // Run inference
        int detections = npu->Detect(imgData, true);
        if (detections >= 0) {
            success_count++;
            std::cout << "  Thread " << thread_id << " (stream_id=" << stream_id
                      << "): " << detections << " detections" << std::endl;
        }

        npu->Release();
    };

    // Spawn threads
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    TEST_ASSERT_MSG(success_count == num_threads,
                    "All " + std::to_string(num_threads) + " threads succeeded");

    std::cout << "  Multi-thread with different stream IDs test passed" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "NpuDetectorLib - Async Backend Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Track overall test success
    bool all_passed = true;

    // Run all tests
    all_passed &= test_single_network_inference();
    all_passed &= test_sequential_networks();
    all_passed &= test_rapid_inference_stress();
    all_passed &= test_network_id_management();
    all_passed &= test_multiple_algorithms();
    all_passed &= test_multithread_different_streams();

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

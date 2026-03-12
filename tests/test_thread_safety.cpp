/**
 * Test 4: Thread Safety Test
 *
 * Purpose: Verify concurrent access doesn't crash or corrupt.
 *
 * Test Cases:
 * - Spawn 4 threads, each creating its own Npu instance
 * - Each thread runs 10 inferences on same image
 * - Verify no crashes, all threads complete successfully
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
#include <mutex>
#include <atomic>
#include <chrono>

// Test result tracking
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static std::mutex g_test_mutex;

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::lock_guard<std::mutex> lock(g_test_mutex); \
            std::cerr << "FAIL: " << message << " at line " << __LINE__ << std::endl; \
            g_tests_failed++; \
            success = false; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

#define TEST_ASSERT_MSG(condition, message) \
    do { \
        if (!(condition)) { \
            std::lock_guard<std::mutex> lock(g_test_mutex); \
            std::cerr << "FAIL: " << message << " at line " << __LINE__ << std::endl; \
            g_tests_failed++; \
            success = false; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

// Thread-local results
struct ThreadResult {
    int thread_id;
    int inferences_completed;
    int total_detections;
    bool success;
    std::string error_message;
};

// Worker function for each thread
void thread_worker(int thread_id, const cv::Mat& image, const std::string& model_path,
                   algorithm alg_type, int num_inferences, ThreadResult& result) {
    result.thread_id = thread_id;
    result.inferences_completed = 0;
    result.total_detections = 0;
    result.success = true;

    // Create NPU instance for this thread
    auto npu = NpuFactory::CreateNpu(alg_type);
    if (!npu) {
        result.error_message = "Failed to create NPU instance";
        result.success = false;
        return;
    }

    // Initialize with stream ID offset by 100 to avoid conflicts between test runs
    int init_result = npu->Initialize(model_path, thread_id + 100);
    if (init_result < 0) {
        result.error_message = "Failed to initialize NPU";
        result.success = false;
        return;
    }

    // Prepare image data
    image_share_t imgData;
    imgData.data = (void*)image.data;
    imgData.width = image.cols;
    imgData.height = image.rows;
    imgData.ch = 3;

    bool success = true;

    // Run multiple inferences
    for (int i = 0; i < num_inferences && success; i++) {
        int num_detections = npu->Detect(imgData, true);
        if (num_detections < 0) {
            std::lock_guard<std::mutex> lock(g_test_mutex);
            result.error_message = "Detect failed at iteration " + std::to_string(i);
            success = false;
            break;
        }
        result.total_detections += num_detections;
        result.inferences_completed++;
    }

    result.success = success;
    npu->Release();
}

// Check if file exists
bool file_exists(const std::string& path) {
    FILE* file = fopen(path.c_str(), "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

// Test: Multiple threads with individual NPU instances
bool test_multi_thread_inference() {
    std::cout << "\n[Test] Multi-Thread Inference (4 threads, 10 inferences each)" << std::endl;

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

    const int num_threads = 4;
    const int inferences_per_thread = 10;

    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(num_threads);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Spawn threads
    // Note: yolov5s.json has "yolo_nms_core": true, so we use ALG_BASE
    // Use stream IDs 4-7 to avoid conflict with previous tests
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(thread_worker, i, std::ref(image), "models/yolov5s.json",
                            ALG_BASE, inferences_per_thread, std::ref(results[i]));
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Verify all threads completed successfully
    bool all_success = true;
    int total_inferences = 0;
    int total_detections = 0;

    for (int i = 0; i < num_threads; i++) {
        if (!results[i].success) {
            std::cerr << "  Thread " << i << " failed: " << results[i].error_message << std::endl;
            all_success = false;
        } else {
            total_inferences += results[i].inferences_completed;
            total_detections += results[i].total_detections;
            std::cout << "  Thread " << i << ": " << results[i].inferences_completed
                     << " inferences, " << results[i].total_detections << " total detections" << std::endl;
        }
    }

    if (all_success) {
        std::cout << "  Total: " << total_inferences << " inferences in "
                 << duration.count() << "s (" << (total_inferences / duration.count()) << " FPS)" << std::endl;
        std::cout << "  Total detections: " << total_detections << std::endl;
        g_tests_passed++;
        return true;
    } else {
        g_tests_failed++;
        return false;
    }
}

// Test: Concurrent initialization (stress test)
bool test_concurrent_initialization() {
    std::cout << "\n[Test] Concurrent Initialization (4 threads initializing simultaneously)" << std::endl;

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    auto init_worker = [&](int thread_id) {
        // Note: yolov5s.json has "yolo_nms_core": true, so we use ALG_BASE
        // Use stream IDs 10-13 to avoid conflict with previous tests
        auto npu = NpuFactory::CreateNpu(ALG_BASE);
        if (!npu) {
            failure_count++;
            return;
        }

        int result = npu->Initialize("models/yolov5s.json", 10 + thread_id);
        if (result >= 0) {
            success_count++;
            npu->Release();
        } else {
            failure_count++;
        }
    };

    // Spawn all threads simultaneously
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(init_worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "  Successful initializations: " << success_count << "/" << num_threads << std::endl;
    std::cout << "  Failed initializations: " << failure_count << "/" << num_threads << std::endl;

    if (success_count == num_threads) {
        g_tests_passed++;
        return true;
    } else {
        g_tests_failed++;
        return false;
    }
}

// Test: Mixed algorithm types in threads
bool test_mixed_algorithms_threads() {
    std::cout << "\n[Test] Mixed Algorithms in Threads" << std::endl;

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

    struct AlgTest {
        algorithm alg;
        std::string model;
        std::string name;
    };

    std::vector<AlgTest> algs = {
        // Note: yolov5s.json has "yolo_nms_core": true, so we use ALG_BASE
        // Use stream ID 20 to avoid conflict with previous tests
        {ALG_BASE, "models/yolov5s.json", "YOLOv5"},
    };

    // Add pose if available
    if (file_exists("models/yolov8s_pose.json")) {
        algs.push_back({ALG_POSE, "models/yolov8s_pose.json", "YOLOv8-Pose"});
    }

    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(algs.size());

    // Spawn threads with different algorithms
    // Use thread IDs starting from 10 to avoid conflict with previous tests
    for (size_t i = 0; i < algs.size(); i++) {
        threads.emplace_back(thread_worker, (int)(10 + i), std::ref(image), algs[i].model,
                            algs[i].alg, 5, std::ref(results[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    bool all_success = true;
    for (size_t i = 0; i < algs.size(); i++) {
        if (results[i].success) {
            std::cout << "  " << algs[i].name << ": " << results[i].inferences_completed
                     << " inferences OK" << std::endl;
        } else {
            std::cerr << "  " << algs[i].name << " FAILED: " << results[i].error_message << std::endl;
            all_success = false;
        }
    }

    if (all_success) {
        g_tests_passed++;
        return true;
    } else {
        g_tests_failed++;
        return false;
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "NpuDetectorLib - Thread Safety Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Track overall test success
    bool all_passed = true;

    // Run all tests
    all_passed &= test_multi_thread_inference();
    all_passed &= test_concurrent_initialization();
    all_passed &= test_mixed_algorithms_threads();

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

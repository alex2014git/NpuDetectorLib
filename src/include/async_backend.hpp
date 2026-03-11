#ifndef _ASYNC_BACKEND_H_
#define _ASYNC_BACKEND_H_

#include <time.h>
#include <vector>
#include <ctype.h>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <unordered_map>
#include <future>
#include <atomic>
#include "hailo/hailort.hpp"
#include "npu_handler.hpp"

// Import types from MultiNetworkPipeline for compatibility
#include "MultiNetworkPipeline/MultiNetworkPipeline.hpp"

/**
 * AsyncBackend is a Singleton class that wraps NPUHandler to provide
 * async inference capabilities while maintaining backward compatibility
 * with the MultiNetworkPipeline interface.
 */
class AsyncBackend
{

public:
    struct NetworkConfig {
        std::string id_name;
        std::string hef_path;
        size_t batch_size = 1;
        hailo_format_type_t out_format = HAILO_FORMAT_TYPE_FLOAT32;
        bool out_quantized = false;
        std::vector<std::string> output_order_by_name;
    };

    /**
     * This is the static method that controls the access to the singleton
     * instance. On the first run, it creates a singleton object and places it
     * into the static field. On subsequent runs, it returns the client existing
     * object stored in the static field.
     */
    static AsyncBackend& GetInstance();

    /**
     * Initialize the VDevice for async inference
     * @return true if initialization succeeded, false otherwise
     */
    bool Initialize();

    /**
     * Add new network to the async backend
     * @param config Network configuration
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode AddNetwork(const NetworkConfig& config);

    /**
     * Run inference on the specified network (blocking call for backward compatibility)
     * @param id_name The network id name to infer
     * @param data The input data
     * @param input_stream_index Input stream index (reserved for future use)
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode Infer(const std::string& id_name, const std::vector<uint8_t>& data, size_t input_stream_index = 0);

    /**
     * Read output data by network name (float32 version)
     * @param id_name The network id name to read the output
     * @param output_buffer Output buffer to store results
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode ReadOutputById(const std::string& id_name, std::vector<std::vector<float>>& output_buffer);

    /**
     * Read output data by network name (uint8 version)
     * @param id_name The network id name to read the output
     * @param output_buffer Output buffer to store results
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode ReadOutputById(const std::string& id_name, std::vector<std::vector<uint8_t>>& output_buffer);

    /**
     * Initialize output buffer for the specified network (float version)
     * @param id_name The network id name
     * @param output_buffer Output buffer to resize
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<float>>& buffer);

    /**
     * Initialize output buffer for the specified network (uint8 version)
     * @param id_name The network id name
     * @param output_buffer Output buffer to resize
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<uint8_t>>& buffer);

    /**
     * Get network input size
     * @param id_name The network id name
     * @param size Output parameter for input size
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode GetNetworkInputSize(const std::string& id_name, size_t& size);

    /**
     * Get network quantization info
     * @param id_name The network id name
     * @param info Output vector for quantization info
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode GetNetworkQuantizationInfo(const std::string& id_name, std::vector<qp_zp_scale_t>& info);

    /**
     * Get network vstream info
     * @param id_name The network id name
     * @param info Output vector for vstream info
     * @param get_from_output_stream Get from output stream (default true)
     * @return MnpReturnCode::SUCCESS on success, error code otherwise
     */
    MnpReturnCode GetNetworkVstream_Info(const std::string& id_name, std::vector<hailo_vstream_info_t>& info, bool get_from_output_stream = true);

    /**
     * Release all resources
     */
    void Release();

private:
    struct NetworkInstance {
        std::unique_ptr<NPUHandler> handler;
        std::string id_name;
        size_t input_size;
        size_t batch_index;  // Current batch index for round-robin
        std::vector<qp_zp_scale_t> quant_info;
        std::vector<hailo_vstream_info_t> vstream_info;
        bool is_nms;

        // Double buffering for async inference (ping-pong buffers)
        std::vector<std::vector<std::shared_ptr<uint8_t>>> input_buffers[2];   // [buffer_index][batch][stream]
        std::vector<std::vector<std::shared_ptr<uint8_t>>> output_buffers[2];  // [buffer_index][batch][stream]
        std::vector<hailort::ConfiguredInferModel::Bindings> bindings[2];      // Bindings per buffer
        std::atomic<size_t> current_buffer{0};
        std::atomic<bool> inference_in_progress[2] = {false, false};  // Per-buffer completion flags

        // For tracking completion
        std::promise<void>* pending_promises[2] = {nullptr, nullptr};
    };

    std::unordered_map<std::string, std::shared_ptr<NetworkInstance>> networks_;
    std::mutex mutex_;
    bool initialized_ = false;

private:
    AsyncBackend() = default;
    // ~AsyncBackend();  // Intentionally not defined to avoid static destruction order issues

    // Delete copy constructor and assignment operator
    AsyncBackend(const AsyncBackend&) = delete;
    AsyncBackend& operator=(const AsyncBackend&) = delete;
};

#endif // _ASYNC_BACKEND_H_

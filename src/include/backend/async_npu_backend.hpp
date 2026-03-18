#ifndef _ASYNC_NPU_BACKEND_H_
#define _ASYNC_NPU_BACKEND_H_

#include "core/npu_backend.hpp"

// Forward declaration - implementation includes async_backend.hpp
class AsyncBackend;

// Adapter that wraps AsyncBackend singleton to implement NpuBackend interface
// This allows gradual migration without breaking existing AsyncBackend functionality
//
// NOTE: The implementation (.cpp file) includes async_backend.hpp and uses
// AsyncBackend methods. This header only sees AsyncBackend via forward declaration.
class AsyncNpuBackend : public NpuBackend {
public:
    AsyncNpuBackend();
    ~AsyncNpuBackend() override;

    // Initialize the backend
    bool Initialize() override;

    // Add a network to the backend
    MnpReturnCode AddNetwork(const NetworkConfig& config) override;

    // Remove a network from the backend
    MnpReturnCode RemoveNetwork(const std::string& id_name) override;

    // Run inference on the specified network
    MnpReturnCode Infer(const std::string& id_name, const std::vector<uint8_t>& data) override;

    // Read float output from the specified network
    MnpReturnCode ReadOutput(const std::string& id_name, std::vector<std::vector<float>>& output_buffer) override;

    // Read uint8 output from the specified network
    MnpReturnCode ReadOutput(const std::string& id_name, std::vector<std::vector<uint8_t>>& output_buffer) override;

    // Initialize output buffer for the specified network (float version)
    MnpReturnCode InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<float>>& buffer) override;

    // Initialize output buffer for the specified network (uint8 version)
    MnpReturnCode InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<uint8_t>>& buffer) override;

    // Get network input size
    MnpReturnCode GetNetworkInputSize(const std::string& id_name, size_t& size) override;

    // Get network quantization info
    MnpReturnCode GetNetworkQuantizationInfo(const std::string& id_name, std::vector<qp_zp_scale_t>& info) override;

    // Get network vstream info
    MnpReturnCode GetNetworkVstreamInfo(const std::string& id_name, std::vector<hailo_vstream_info_t>& info) override;

    // Release all resources
    void Release() override;

private:
    AsyncBackend* async_backend_;
};

#endif // _ASYNC_NPU_BACKEND_H_

#ifndef _NPU_BACKEND_H_
#define _NPU_BACKEND_H_

#include <string>
#include <vector>
#include <memory>
#include "core/npu_types.hpp"
#include "hailo/hailort.hpp"

// Abstract backend interface
class NpuBackend {
public:
    virtual ~NpuBackend() = default;

    // Initialize the backend
    virtual bool Initialize() = 0;

    // Add a network to the backend
    virtual MnpReturnCode AddNetwork(const NetworkConfig& config) = 0;

    // Remove a network from the backend
    virtual MnpReturnCode RemoveNetwork(const std::string& id_name) = 0;

    // Run inference on the specified network
    virtual MnpReturnCode Infer(const std::string& id_name, const std::vector<uint8_t>& data) = 0;

    // Read float output from the specified network
    virtual MnpReturnCode ReadOutput(const std::string& id_name, std::vector<std::vector<float>>& output_buffer) = 0;

    // Read uint8 output from the specified network
    virtual MnpReturnCode ReadOutput(const std::string& id_name, std::vector<std::vector<uint8_t>>& output_buffer) = 0;

    // Initialize output buffer for the specified network (float version)
    virtual MnpReturnCode InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<float>>& buffer) = 0;

    // Initialize output buffer for the specified network (uint8 version)
    virtual MnpReturnCode InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<uint8_t>>& buffer) = 0;

    // Get network input size
    virtual MnpReturnCode GetNetworkInputSize(const std::string& id_name, size_t& size) = 0;

    // Get network quantization info
    virtual MnpReturnCode GetNetworkQuantizationInfo(const std::string& id_name, std::vector<qp_zp_scale_t>& info) = 0;

    // Get network vstream info
    virtual MnpReturnCode GetNetworkVstreamInfo(const std::string& id_name, std::vector<hailo_vstream_info_t>& info) = 0;

    // Release all resources
    virtual void Release() = 0;
};

#endif // _NPU_BACKEND_H_

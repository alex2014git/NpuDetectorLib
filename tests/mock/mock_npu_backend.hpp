#ifndef MOCK_NPU_BACKEND_HPP
#define MOCK_NPU_BACKEND_HPP

#include "core/npu_backend.hpp"
#include <unordered_map>
#include <vector>

// Mock NpuBackend for testing without Hailo hardware
// Usage: Create mock, configure outputs, inject into NpuBaseImpl via SetBackend()
class MockNpuBackend : public NpuBackend {
public:
    // Configuration
    bool init_result = true;
    MnpReturnCode add_network_result = MnpReturnCode::SUCCESS;
    MnpReturnCode infer_result = MnpReturnCode::SUCCESS;
    MnpReturnCode read_output_result = MnpReturnCode::SUCCESS;

    // Set expected outputs for a network
    void SetNetworkOutputs(const std::string& id,
                           const std::vector<std::vector<float>>& outputs);

    // NpuBackend interface
    bool Initialize() override { return init_result; }
    MnpReturnCode AddNetwork(const NetworkConfig& config) override;
    MnpReturnCode RemoveNetwork(const std::string& id_name) override;
    MnpReturnCode Infer(const std::string& id_name,
                        const std::vector<uint8_t>& data) override;
    MnpReturnCode ReadOutput(const std::string& id_name,
                             std::vector<std::vector<float>>& output_buffer) override;
    MnpReturnCode ReadOutput(const std::string& id_name,
                             std::vector<std::vector<uint8_t>>& output_buffer) override;
    MnpReturnCode InitializeOutputBuffer(const std::string& id_name,
                                          std::vector<std::vector<float>>& buffer) override;
    MnpReturnCode InitializeOutputBuffer(const std::string& id_name,
                                          std::vector<std::vector<uint8_t>>& buffer) override;
    MnpReturnCode GetNetworkInputSize(const std::string& id_name, size_t& size) override;
    MnpReturnCode GetNetworkQuantizationInfo(const std::string& id_name,
                                              std::vector<qp_zp_scale_t>& info) override;
    MnpReturnCode GetNetworkVstreamInfo(const std::string& id_name,
                                         std::vector<hailo_vstream_info_t>& info) override;
    void Release() override;

    // Test introspection
    size_t GetInferCount(const std::string& id) const;
    bool HasNetwork(const std::string& id) const;

private:
    struct NetworkState {
        NetworkConfig config;
        std::vector<std::vector<float>> float_outputs;
        std::vector<std::vector<uint8_t>> uint8_outputs;
        size_t infer_count = 0;
    };
    std::unordered_map<std::string, NetworkState> networks_;
};

#endif // MOCK_NPU_BACKEND_HPP

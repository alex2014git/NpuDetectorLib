#include "backend/async_npu_backend.hpp"
#include "async_backend.hpp"  // Implementation-only include

AsyncNpuBackend::AsyncNpuBackend()
    : async_backend_(&AsyncBackend::GetInstance())
{
}

AsyncNpuBackend::~AsyncNpuBackend() = default;

bool AsyncNpuBackend::Initialize()
{
    return async_backend_->Initialize();
}

MnpReturnCode AsyncNpuBackend::AddNetwork(const NetworkConfig& config)
{
    AsyncBackend::NetworkConfig async_config;
    async_config.id_name = config.id_name;
    async_config.hef_path = config.hef_path;
    async_config.batch_size = config.batch_size;
    async_config.out_format = config.out_format;
    async_config.out_quantized = config.out_quantized;
    async_config.output_order_by_name = config.output_order_by_name;

    return async_backend_->AddNetwork(async_config);
}

MnpReturnCode AsyncNpuBackend::RemoveNetwork(const std::string& id_name)
{
    return async_backend_->RemoveNetwork(id_name);
}

MnpReturnCode AsyncNpuBackend::Infer(const std::string& id_name, const std::vector<uint8_t>& data)
{
    return async_backend_->Infer(id_name, data);
}

MnpReturnCode AsyncNpuBackend::ReadOutput(const std::string& id_name, std::vector<std::vector<float>>& output_buffer)
{
    return async_backend_->ReadOutputById(id_name, output_buffer);
}

MnpReturnCode AsyncNpuBackend::ReadOutput(const std::string& id_name, std::vector<std::vector<uint8_t>>& output_buffer)
{
    return async_backend_->ReadOutputById(id_name, output_buffer);
}

MnpReturnCode AsyncNpuBackend::InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<float>>& buffer)
{
    return async_backend_->InitializeOutputBuffer(id_name, buffer);
}

MnpReturnCode AsyncNpuBackend::InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<uint8_t>>& buffer)
{
    return async_backend_->InitializeOutputBuffer(id_name, buffer);
}

MnpReturnCode AsyncNpuBackend::GetNetworkInputSize(const std::string& id_name, size_t& size)
{
    return async_backend_->GetNetworkInputSize(id_name, size);
}

MnpReturnCode AsyncNpuBackend::GetNetworkQuantizationInfo(const std::string& id_name, std::vector<qp_zp_scale_t>& info)
{
    return async_backend_->GetNetworkQuantizationInfo(id_name, info);
}

MnpReturnCode AsyncNpuBackend::GetNetworkVstreamInfo(const std::string& id_name, std::vector<hailo_vstream_info_t>& info)
{
    return async_backend_->GetNetworkVstream_Info(id_name, info);
}

void AsyncNpuBackend::Release()
{
    async_backend_->Release();
}

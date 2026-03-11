#include "async_backend.hpp"
#include <chrono>
#include <stdexcept>

// Debug macros (imported from MultiNetworkPipeline style)
//#define LOG_DEBUG_ENABLE
#ifdef LOG_DEBUG_ENABLE
#define DBG_DEBUG(MSG)   std::cout<<"[DEBUG] "<<MSG<<std::endl;
#else
#define DBG_DEBUG(MSG)
#endif

#ifdef LOG_WARN_ENABLE
#define DBG_WARN(MSG)   std::cout<<"[WARNING] "<<MSG<<std::endl;
#else
#define DBG_WARN(MSG)
#endif

#ifdef LOG_ERROR_ENABLE
#define DBG_ERROR(MSG)   std::cout<<"[ERROR] "<<MSG<<std::endl;
#else
#define DBG_ERROR(MSG)
#endif

AsyncBackend& AsyncBackend::GetInstance()
{
    static AsyncBackend* instance = new AsyncBackend();
    return *instance;
}

bool AsyncBackend::Initialize()
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        DBG_DEBUG("AsyncBackend already initialized");
        return true;
    }

    // VDevice initialization is handled per-network in NPUHandler
    // This method is kept for API compatibility
    initialized_ = true;
    DBG_DEBUG("AsyncBackend initialized successfully");
    return true;
}

MnpReturnCode AsyncBackend::AddNetwork(const NetworkConfig& config)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        DBG_ERROR("AsyncBackend not initialized");
        return MnpReturnCode::HAILO_NOT_INITIALIZED;
    }

    if (config.id_name.empty()) {
        DBG_ERROR("Invalid network id_name (empty)");
        return MnpReturnCode::INVALID_PARAMETER;
    }

    // Check for duplicate network ID
    if (networks_.find(config.id_name) != networks_.end()) {
        DBG_ERROR("Network with id_name '" << config.id_name << "' already exists");
        return MnpReturnCode::DUPLICATED;
    }

    try {
        // Create new network instance
        auto network = std::make_shared<NetworkInstance>();
        network->id_name = config.id_name;
        network->batch_index = 0;
        network->is_nms = false;

        // Create NPUHandler
        network->handler = std::make_unique<NPUHandler>(config.hef_path);

        // Configure the handler
        network->handler->setBatchSize(config.batch_size);
        network->handler->setOutputFormat(config.out_format);

        // Configure model
        network->handler->configureModel();

        // Get and store vstream info for post-processing
        auto vstream_infos = network->handler->getOutputVstreamInfos();
        network->vstream_info.assign(vstream_infos.begin(), vstream_infos.end());

        // Get input/output names and sizes
        auto input_names = network->handler->getInputNames();
        auto output_names = network->handler->getOutputNames();

        if (input_names.empty()) {
            DBG_ERROR("No input names found for network: " << config.id_name);
            return MnpReturnCode::FAILED;
        }

        // Get input size (assuming single input for most cases)
        network->input_size = network->handler->getInputSize(input_names[0]);

        // Get input image shape for quantization info
        auto input_shape = network->handler->getInputImageShape(input_names[0]);

        // Calculate quantization info from input shape
        // Note: In async API, quantization info is typically handled by the model
        // We'll extract it from the handler's output shape info
        for (const auto& out_name : output_names) {
            auto out_shape = network->handler->getOutputImageShape(out_name);

            // Create quant info - for async API, we need to get this from model
            // For now, use default values (will be overridden when model provides actual values)
            qp_zp_scale_t q_info;
            q_info.qp_zp = 0.0f;
            q_info.qp_scale = 1.0f;
            network->quant_info.push_back(q_info);

            // Check if this is an NMS output
            if (network->handler->isNMS(out_name)) {
                network->is_nms = true;
            }
        }

        // Create double buffers for async inference (ping-pong buffering)
        // We create two sets of buffers: one for current inference, one for next
        for (size_t buf = 0; buf < 2; buf++) {
            network->input_buffers[buf].resize(config.batch_size);
            network->output_buffers[buf].resize(config.batch_size);

            for (size_t b = 0; b < config.batch_size; b++) {
                // Create input buffers
                for (const auto& in_name : input_names) {
                    size_t in_size = network->handler->getInputSize(in_name);
                    auto buffer = network->handler->pageAlignedAlloc(in_size);
                    network->input_buffers[buf][b].push_back(buffer);
                }

                // Create output buffers
                for (const auto& out_name : output_names) {
                    size_t out_size = network->handler->getOutputSize(out_name);
                    auto buffer = network->handler->pageAlignedAlloc(out_size);
                    network->output_buffers[buf][b].push_back(buffer);
                }
            }
        }

        // Create bindings with both buffer sets (will be reconfigured per-inference)
        network->handler->setExternalInputBuffers(network->input_buffers[0]);
        network->handler->setExternalOutputBuffers(network->output_buffers[0]);
        network->bindings[0] = network->handler->createBindings();

        network->handler->setExternalInputBuffers(network->input_buffers[1]);
        network->handler->setExternalOutputBuffers(network->output_buffers[1]);
        network->bindings[1] = network->handler->createBindings();

        // Store network instance
        networks_[config.id_name] = network;

        DBG_DEBUG("Added network: " << config.id_name << " with batch_size=" << config.batch_size);

        return MnpReturnCode::SUCCESS;

    } catch (const std::exception& e) {
        DBG_ERROR("Failed to add network: " << e.what());
        return MnpReturnCode::FAILED;
    }
}

MnpReturnCode AsyncBackend::Infer(const std::string& id_name, const std::vector<uint8_t>& data, size_t input_stream_index /*= 0*/)
{
    // NO GLOBAL LOCK - lock-free read (shared_ptr keeps network alive)
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    auto& network = it->second;

    try {
        // Get current buffer index (ping-pong)
        size_t buf_idx = network->current_buffer.fetch_add(1) % 2;

        // Wait for previous inference on THIS buffer to complete (if still running)
        while (network->inference_in_progress[buf_idx].load()) {
            std::this_thread::yield();  // Spin-wait briefly
        }

        // For batch_size=1, always use index 0
        size_t batch_idx = 0;

        // Copy data to input buffer (selected by ping-pong)
        if (!network->input_buffers[buf_idx][batch_idx].empty() && !data.empty()) {
            size_t copy_size = std::min(data.size(), network->input_size);
            memcpy(network->input_buffers[buf_idx][batch_idx][0].get(), data.data(), copy_size);
        }

        // Mark THIS buffer as in-progress
        network->inference_in_progress[buf_idx].store(true);

        // Reconfigure bindings with current buffer (creates new DMA mappings) and store them
        network->bindings[buf_idx] = network->handler->rebindBuffers(network->input_buffers[buf_idx], network->output_buffers[buf_idx]);

        // Run async inference WITH CALLBACK (pass the first element since bindings[buf_idx] is a vector)
        network->handler->run(network->bindings[buf_idx], [network, buf_idx](const auto& outputs) {
            // Inference complete - mark THIS buffer as done
            network->inference_in_progress[buf_idx].store(false);
            // Data is now in output_buffers[buf_idx], ready for ReadOutputById
        });

        return MnpReturnCode::SUCCESS;

    } catch (const std::exception& e) {
        DBG_ERROR("Inference failed for network '" << id_name << "': " << e.what());
        return MnpReturnCode::FAILED;
    }
}

MnpReturnCode AsyncBackend::ReadOutputById(const std::string& id_name, std::vector<std::vector<float>>& output_buffer)
{
    // NO GLOBAL LOCK - lock-free read
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    auto& network = it->second;

    try {
        // Get the buffer that should have completed (the one before current)
        // Use (current + 1) % 2 to get the previous buffer in ping-pong
        size_t buf_idx = (network->current_buffer.load() + 1) % 2;

        // Wait for inference on THIS buffer to complete (if still running)
        while (network->inference_in_progress[buf_idx].load()) {
            std::this_thread::yield();
        }

        size_t batch_idx = 0;

        if (batch_idx >= network->output_buffers[buf_idx].size() || network->output_buffers[buf_idx][batch_idx].empty()) {
            return MnpReturnCode::NO_DATA_AVAILABLE;
        }

        // Resize output buffer if needed
        if (output_buffer.size() != network->output_buffers[buf_idx][batch_idx].size()) {
            output_buffer.resize(network->output_buffers[buf_idx][batch_idx].size());
        }

        // Copy data from internal buffers to output buffer
        for (size_t i = 0; i < network->output_buffers[buf_idx][batch_idx].size(); i++) {
            // Get output size
            auto output_names = network->handler->getOutputNames();
            if (i >= output_names.size()) {
                continue;
            }

            size_t out_size = network->handler->getOutputSize(output_names[i]);
            size_t num_floats = out_size / sizeof(float);

            if (output_buffer[i].size() != num_floats) {
                output_buffer[i].resize(num_floats);
            }

            // Copy data - assuming float32 format
            memcpy(output_buffer[i].data(),
                   network->output_buffers[buf_idx][batch_idx][i].get(),
                   out_size);
        }

        return MnpReturnCode::SUCCESS;

    } catch (const std::exception& e) {
        DBG_ERROR("ReadOutputById failed for network '" << id_name << "': " << e.what());
        return MnpReturnCode::FAILED;
    }
}

MnpReturnCode AsyncBackend::ReadOutputById(const std::string& id_name, std::vector<std::vector<uint8_t>>& output_buffer)
{
    // NO GLOBAL LOCK - lock-free read
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    auto& network = it->second;

    try {
        // Get the buffer that should have completed (the one before current)
        // Use (current + 1) % 2 to get the previous buffer in ping-pong
        size_t buf_idx = (network->current_buffer.load() + 1) % 2;

        // Wait for inference on THIS buffer to complete (if still running)
        while (network->inference_in_progress[buf_idx].load()) {
            std::this_thread::yield();
        }

        size_t batch_idx = 0;

        if (batch_idx >= network->output_buffers[buf_idx].size() || network->output_buffers[buf_idx][batch_idx].empty()) {
            return MnpReturnCode::NO_DATA_AVAILABLE;
        }

        // Resize output buffer if needed
        if (output_buffer.size() != network->output_buffers[buf_idx][batch_idx].size()) {
            output_buffer.resize(network->output_buffers[buf_idx][batch_idx].size());
        }

        // Copy data from internal buffers to output buffer
        for (size_t i = 0; i < network->output_buffers[buf_idx][batch_idx].size(); i++) {
            auto output_names = network->handler->getOutputNames();
            if (i >= output_names.size()) {
                continue;
            }

            size_t out_size = network->handler->getOutputSize(output_names[i]);

            if (output_buffer[i].size() != out_size) {
                output_buffer[i].resize(out_size);
            }

            memcpy(output_buffer[i].data(),
                   network->output_buffers[buf_idx][batch_idx][i].get(),
                   out_size);
        }

        return MnpReturnCode::SUCCESS;

    } catch (const std::exception& e) {
        DBG_ERROR("ReadOutputById failed for network '" << id_name << "': " << e.what());
        return MnpReturnCode::FAILED;
    }
}

MnpReturnCode AsyncBackend::InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<float>>& buffer)
{
    // NO GLOBAL LOCK - lock-free read
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    auto& network = it->second;

    try {
        auto output_names = network->handler->getOutputNames();
        buffer.resize(output_names.size());

        for (size_t i = 0; i < output_names.size(); i++) {
            size_t out_size = network->handler->getOutputSize(output_names[i]);
            buffer[i].resize(out_size / sizeof(float));
        }

        return MnpReturnCode::SUCCESS;

    } catch (const std::exception& e) {
        DBG_ERROR("InitializeOutputBuffer failed for network '" << id_name << "': " << e.what());
        return MnpReturnCode::FAILED;
    }
}

MnpReturnCode AsyncBackend::InitializeOutputBuffer(const std::string& id_name, std::vector<std::vector<uint8_t>>& buffer)
{
    // NO GLOBAL LOCK - lock-free read
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    auto& network = it->second;

    try {
        auto output_names = network->handler->getOutputNames();
        buffer.resize(output_names.size());

        for (size_t i = 0; i < output_names.size(); i++) {
            size_t out_size = network->handler->getOutputSize(output_names[i]);
            buffer[i].resize(out_size);
        }

        return MnpReturnCode::SUCCESS;

    } catch (const std::exception& e) {
        DBG_ERROR("InitializeOutputBuffer failed for network '" << id_name << "': " << e.what());
        return MnpReturnCode::FAILED;
    }
}

MnpReturnCode AsyncBackend::GetNetworkInputSize(const std::string& id_name, size_t& size)
{
    // NO GLOBAL LOCK - lock-free read
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    size = it->second->input_size;
    return MnpReturnCode::SUCCESS;
}

MnpReturnCode AsyncBackend::GetNetworkQuantizationInfo(const std::string& id_name, std::vector<qp_zp_scale_t>& info)
{
    // NO GLOBAL LOCK - lock-free read
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    info = it->second->quant_info;
    return MnpReturnCode::SUCCESS;
}

MnpReturnCode AsyncBackend::GetNetworkVstream_Info(const std::string& id_name, std::vector<hailo_vstream_info_t>& info, bool get_from_output_stream /*= true*/)
{
    // NO GLOBAL LOCK - lock-free read
    auto it = networks_.find(id_name);
    if (it == networks_.end()) {
        DBG_ERROR("Network '" << id_name << "' not found");
        return MnpReturnCode::NOT_FOUND;
    }

    info = it->second->vstream_info;
    return MnpReturnCode::SUCCESS;
}

void AsyncBackend::Release()
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [id, network] : networks_) {
        // Clear bindings first (they hold references to buffers)
        for (size_t buf = 0; buf < 2; buf++) {
            network->bindings[buf].clear();
        }
        // Clear buffer tracking (now arrays of buffers)
        for (size_t buf = 0; buf < 2; buf++) {
            network->input_buffers[buf].clear();
            network->output_buffers[buf].clear();
            // Clean up any pending promises
            if (network->pending_promises[buf] != nullptr) {
                delete network->pending_promises[buf];
                network->pending_promises[buf] = nullptr;
            }
            // Reset completion flag
            network->inference_in_progress[buf].store(false);
        }
        // Handler destructor will clean up DMA mappings
        network->handler.reset();
    }
    networks_.clear();
    initialized_ = false;

    DBG_DEBUG("AsyncBackend released all resources");
}

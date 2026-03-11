#include "npu_handler.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstring>

#if defined(__unix__)
#include <sys/mman.h>
#elif defined(_MSC_VER)
#include <windows.h>
#else
#error("Aligned alloc not supported")
#endif

NPUHandler::NPUHandler(const std::string &hef_path)
{
    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    if (HAILO_SUCCESS != status)
    {
        std::cerr << "Failed init vdevice_params, status = " << status << std::endl;
        std::abort();
    }
    params.device_count = 1;
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    params.group_id = "HAILO"; //HAILO_UNIQUE_VDEVICE_GROUP_ID;

    vdevice = hailort::VDevice::create_shared(params).expect("Failed to create VDevice");

    model = vdevice->create_infer_model(hef_path).expect("Failed to create InferModel");
    inited = false;
}

NPUHandler::~NPUHandler()
{
    // Clear external buffers (no ownership)
    external_input_buffers.clear();
    external_output_buffers.clear();

    // Clear internal buffers and mappings
    nhwc_input_buffers.clear();
    nhwc_output_buffers.clear();
    buffer_guards.clear();
    buffer_map_guards.clear();
    model.reset();
}

void NPUHandler::setBatchSize(size_t batch_size)
{
    if (model) {
        this->batch_size = batch_size;
        model->set_batch_size(batch_size);
    }
}

void NPUHandler::setOutputFormat(hailo_format_type_t format)
{
    if (model) {
        for (const auto &output_name : model->get_output_names()) {
            model->output(output_name)->set_format_type(format);
        }
    }
}

void NPUHandler::setOutputOrder(hailo_format_order_t format_order)
{
    if (model) {
        for (const auto &output_name : model->get_output_names()) {
            model->output(output_name)->set_format_order(format_order);
        }
    }
}

void NPUHandler::configureModel()
{
    if (!model) {
        throw std::runtime_error("Model not loaded");
    }
    configured_model = model->configure().expect("Failed to configure model");
    inited = true;
}

void NPUHandler::setExternalInputBuffers(const std::vector<std::vector<std::shared_ptr<uint8_t>>>& buffers) {
    external_input_buffers = buffers;
    use_external_inputs = true;
}

void NPUHandler::setExternalOutputBuffers(const std::vector<std::vector<std::shared_ptr<uint8_t>>>& buffers) {
    external_output_buffers = buffers;
    use_external_outputs = true;
}

std::vector<hailort::ConfiguredInferModel::Bindings> NPUHandler::createBindings() {
    if (!model || !inited) {
        throw std::runtime_error("Model not loaded or not configured");
    }

    std::vector<hailort::ConfiguredInferModel::Bindings> multiple_bindings;
    nhwc_input_buffers.clear();
    nhwc_output_buffers.clear();

    for (size_t i = 0; i < batch_size; i++) {
        auto bindings = configured_model.create_bindings().expect("Failed to create bindings");

        std::vector<std::shared_ptr<uint8_t>> input_batch;
        std::vector<std::shared_ptr<uint8_t>> output_batch;

        // Process inputs
        size_t input_idx = 0;
        for (const auto& input_name : model->get_input_names()) {
            size_t input_size = model->input(input_name)->get_frame_size();
            std::shared_ptr<uint8_t> input_buffer;

            if (use_external_inputs) {
                if (i >= external_input_buffers.size() || input_idx >= external_input_buffers[i].size()) {
                    throw std::runtime_error("External input buffer not provided for batch or input");
                }
                input_buffer = external_input_buffers[i][input_idx];
            } else {
                input_buffer = pageAlignedAlloc(input_size);
                buffer_guards.push_back(input_buffer); // Track internal buffers
            }

            input_batch.push_back(input_buffer);

            // Create DMA mapping (required for all buffers)
            auto input_mapping = hailort::DmaMappedBuffer::create(
                *vdevice, input_buffer.get(), input_size, HAILO_DMA_BUFFER_DIRECTION_H2D
            ).expect("Failed to map input buffer");
            buffer_map_guards.push_back(std::move(input_mapping));

            bindings.input(input_name)->set_buffer(hailort::MemoryView(input_buffer.get(), input_size));
            input_idx++;
        }

        // Process outputs
        size_t output_idx = 0;
        for (const auto& output_name : model->get_output_names()) {
            size_t output_size = model->output(output_name)->get_frame_size();
            std::shared_ptr<uint8_t> output_buffer;

            if (use_external_outputs) {
                if (i >= external_output_buffers.size() || output_idx >= external_output_buffers[i].size()) {
                    throw std::runtime_error("External output buffer not provided for batch or output");
                }
                output_buffer = external_output_buffers[i][output_idx];
            } else {
                output_buffer = pageAlignedAlloc(output_size);
                buffer_guards.push_back(output_buffer); // Track internal buffers
            }

            output_batch.push_back(output_buffer);

            auto output_mapping = hailort::DmaMappedBuffer::create(
                *vdevice, output_buffer.get(), output_size, HAILO_DMA_BUFFER_DIRECTION_D2H
            ).expect("Failed to map output buffer");
            buffer_map_guards.push_back(std::move(output_mapping));

            bindings.output(output_name)->set_buffer(hailort::MemoryView(output_buffer.get(), output_size));
            output_idx++;
        }

        // Store buffers only if internally allocated
        if (!use_external_inputs) {
            nhwc_input_buffers.push_back(std::move(input_batch));
        }
        if (!use_external_outputs) {
            nhwc_output_buffers.push_back(std::move(output_batch));
        }

        multiple_bindings.push_back(std::move(bindings));
    }

    return multiple_bindings;
}

std::vector<std::string> NPUHandler::getInputNames()
{
    return model->get_input_names();
}

std::vector<std::string> NPUHandler::getOutputNames()
{
    return model->get_output_names();
}

size_t NPUHandler::getInputSize(const std::string &input_name)
{
    return model->input(input_name)->get_frame_size();
}

size_t NPUHandler::getOutputSize(const std::string &output_name)
{
    return model->output(output_name)->get_frame_size();
}

hailo_3d_image_shape_t NPUHandler::getInputImageShape(const std::string &input_name)
{
    return model->input(input_name)->shape();
}

hailo_3d_image_shape_t NPUHandler::getOutputImageShape(const std::string &output_name)
{
    return model->output(output_name)->shape();
}

bool NPUHandler::isNMS(const std::string &output_name)
{
    return model->output(output_name)->is_nms();
}

std::vector<hailo_vstream_info_t> NPUHandler::getOutputVstreamInfos()
{
    return model->hef().get_output_vstream_infos().expect("Failed to get output vstream infos");
}

void NPUHandler::run(std::vector<hailort::ConfiguredInferModel::Bindings> &bindings_vec,
                     std::function<void(const std::vector<std::unordered_map<std::string, hailort::MemoryView>>&)> callback)
{
    auto status = configured_model.wait_for_async_ready(std::chrono::milliseconds(10000), bindings_vec.size());
    if (HAILO_SUCCESS != status) {
        throw hailort::hailort_error(status, "Failed to wait for async ready");
    }
    auto job = configured_model.run_async(bindings_vec, [this, &bindings_vec, callback](const hailort::AsyncInferCompletionInfo &completion_info) {
        if (completion_info.status != HAILO_SUCCESS) {
            std::cerr << "Async inference (multi-bindings) failed" << std::endl;
            return;
        }
        if (callback) {
            std::vector<std::unordered_map<std::string, hailort::MemoryView>> all_outputs;
            for (auto &bindings : bindings_vec) {
                std::unordered_map<std::string, hailort::MemoryView> outputs;
                for (const auto &output_name : model->get_output_names()) {
                    outputs[output_name] = bindings.output(output_name)->get_buffer().value();
                }
                all_outputs.push_back(std::move(outputs));
            }
            callback(all_outputs);
        }
    }).expect("Async inference failed");

    if (!callback) {
        job.wait(std::chrono::milliseconds(1000));
    } else {
        job.detach();
    }
}

std::shared_ptr<uint8_t> NPUHandler::pageAlignedAlloc(size_t size)
{
#if defined(__unix__)
    void *addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return {reinterpret_cast<uint8_t*>(addr), [size](void *p){ munmap(p, size); }};
#elif defined(_MSC_VER)
    void *addr = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!addr) throw std::bad_alloc();
    return {reinterpret_cast<uint8_t*>(addr), [](void *p){ VirtualFree(p, 0, MEM_RELEASE); }};
#endif
}

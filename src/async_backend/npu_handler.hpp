/*
 * Copyright (c) 2020-2023 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 */

#pragma once

#include "hailo/hailort.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

class NPUHandler {
public:
    explicit NPUHandler(const std::string &hef_path);
    ~NPUHandler();

    void setBatchSize(size_t batch_size);
    void setOutputFormat(hailo_format_type_t format);
    void setOutputOrder(hailo_format_order_t format_order);
    void configureModel();
    void setExternalInputBuffers(const std::vector<std::vector<std::shared_ptr<uint8_t>>>& buffers);
    void setExternalOutputBuffers(const std::vector<std::vector<std::shared_ptr<uint8_t>>>& buffers);
    std::vector<hailort::ConfiguredInferModel::Bindings> createBindings();

    std::vector<std::string> getInputNames();
    std::vector<std::string> getOutputNames();
    size_t getInputSize(const std::string &input_name);
    size_t getOutputSize(const std::string &output_name);
    hailo_3d_image_shape_t getInputImageShape(const std::string &input_name);
    hailo_3d_image_shape_t getOutputImageShape(const std::string &output_name);
    bool isNMS(const std::string &output_name);
    std::vector<hailo_vstream_info_t> getOutputVstreamInfos();
    void run(std::vector<hailort::ConfiguredInferModel::Bindings> &bindings_vec,
             std::function<void(const std::vector<std::unordered_map<std::string, hailort::MemoryView>>&)> callback = nullptr);

    // Public static method for page-aligned memory allocation
    static std::shared_ptr<uint8_t> pageAlignedAlloc(size_t size);


private:
    std::shared_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::InferModel> model;
    hailort::ConfiguredInferModel configured_model;
    size_t batch_size;
    bool inited;

    bool use_external_inputs = false;
    bool use_external_outputs = false;
    std::vector<std::vector<std::shared_ptr<uint8_t>>> external_input_buffers; // User-provided
    std::vector<std::vector<std::shared_ptr<uint8_t>>> external_output_buffers; // User-provided
    std::vector<std::vector<std::shared_ptr<uint8_t>>> nhwc_input_buffers; // Internal buffers
    std::vector<std::vector<std::shared_ptr<uint8_t>>> nhwc_output_buffers; // Internal buffers

    std::vector<std::shared_ptr<uint8_t>> buffer_guards; // Track internally allocated buffers
    std::vector<hailort::DmaMappedBuffer> buffer_map_guards; // Track DMA mappings
};

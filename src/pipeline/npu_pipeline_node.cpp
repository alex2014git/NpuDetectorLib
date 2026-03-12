#include "pipeline/npu_pipeline_node.hpp"
#include "pipeline/npu_pipeline_context.hpp"
#include "npu.hpp"
#include "npu_factory.hpp"
#include <opencv2/opencv.hpp>

namespace npu_pipeline {

// Check if node is ready to execute
bool PipelineNode::isReady(const FrameResults& frame) const {
    auto inputs = getInputNodes();
    for (const auto& node_id : inputs) {
        const auto& outputs = frame.getNodeOutput(node_id);
        // Check if this upstream node has produced output
        // This is a simplified check - in practice might need more sophisticated logic
    }
    return true;
}

// NpuInferenceNode implementation
NpuInferenceNode::NpuInferenceNode(const std::string& name, int algorithm_type)
    : _name(name)
    , _algorithm_type(algorithm_type)
    , _preferred_batch_size(8) {
}

int NpuInferenceNode::initialize(const std::string& configJson, int streamId) {
    _model_config = configJson;
    _stream_id = streamId;

    // Create NPU instance using factory
    _npu = NpuFactory::CreateNpu(static_cast<::algorithm>(_algorithm_type));
    if (!_npu) {
        return -1;
    }

    return _npu->Initialize(configJson, streamId);
}

void NpuInferenceNode::release() {
    if (_npu) {
        _npu->Release();
        _npu.reset();
    }
}

std::vector<PipelineObject> NpuInferenceNode::processBatch(
    const std::vector<PipelineObject>& inputs,
    const std::vector<uint64_t>& frame_ids,
    PipelineContext& ctx) {

    if (inputs.empty() || !_npu) {
        return {};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<PipelineObject> outputs;
    outputs.reserve(inputs.size());

    // For single input, fall back to single inference
    if (inputs.size() == 1) {
        auto& frame = ctx.getFrame(frame_ids[0]);
        outputs.push_back(processObject(inputs[0], frame, ctx));
        return outputs;
    }

    // Batch preprocessing - collect images
    std::vector<cv::Mat> batch_images;
    batch_images.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& frame = ctx.getFrame(frame_ids[i]);
        if (!frame.source_frame) continue;

        // Convert to cv::Mat
        image_share_t* img = frame.source_frame.get();
        cv::Mat original(img->height, img->width, CV_8UC(img->ch), img->data);

        // Crop ROI
        const auto& roi = inputs[i].roi;
        int x = static_cast<int>(roi.x_min);
        int y = static_cast<int>(roi.y_min);
        int w = static_cast<int>(roi.width());
        int h = static_cast<int>(roi.height());

        // Clamp to image bounds
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, original.cols - x);
        h = std::min(h, original.rows - y);

        if (w > 0 && h > 0) {
            cv::Rect crop_rect(x, y, w, h);
            cv::Mat cropped = original(crop_rect);
            batch_images.push_back(cropped.clone());
        }
    }

    // Run inference on each image
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i >= batch_images.size()) break;

        PipelineObject out = inputs[i];

        // Create image_share_t for inference
        image_share_t img_data;
        img_data.data = batch_images[i].data;
        img_data.width = batch_images[i].cols;
        img_data.height = batch_images[i].rows;
        img_data.ch = batch_images[i].channels();

        // Run inference
        int ret = _npu->Detect(img_data, true);
        if (ret == 0) {
            // Get detection results and convert to PipelineObject format
            // This is simplified - actual implementation depends on NPU interface
            DetectionResult det_result;
            det_result.class_id = out.roi.class_id;
            det_result.confidence = out.roi.confidence;
            det_result.bbox = out.roi;
            out.setResult(_name, det_result);
        }

        outputs.push_back(std::move(out));
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    {
        std::lock_guard<std::mutex> lock(_stats_mutex);
        _inference_count++;
        _avg_latency_ms = (_avg_latency_ms * (_inference_count - 1) + latency_ms) / _inference_count;
    }

    return outputs;
}

std::vector<std::shared_ptr<image_share_t>> NpuInferenceNode::preprocess(
    const std::vector<PipelineObject>& inputs,
    const std::vector<uint64_t>& frame_ids,
    PipelineContext& ctx) {

    std::vector<std::shared_ptr<image_share_t>> result;
    result.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& frame = ctx.getFrame(frame_ids[i]);
        if (!frame.source_frame) continue;

        image_share_t* src = frame.source_frame.get();
        const auto& roi = inputs[i].roi;

        // Calculate crop region
        int x = static_cast<int>(roi.x_min);
        int y = static_cast<int>(roi.y_min);
        int w = static_cast<int>(roi.width());
        int h = static_cast<int>(roi.height());

        // Clamp to image bounds
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, src->width - x);
        h = std::min(h, src->height - y);

        if (w <= 0 || h <= 0) {
            result.push_back(nullptr);
            continue;
        }

        // Create cropped image
        auto cropped = std::make_shared<image_share_t>();
        cropped->width = w;
        cropped->height = h;
        cropped->ch = src->ch;
        cropped->data = new uint8_t[w * h * src->ch];

        // Copy cropped region
        for (int row = 0; row < h; ++row) {
            uint8_t* src_row = static_cast<uint8_t*>(src->data) + ((y + row) * src->width + x) * src->ch;
            uint8_t* dst_row = static_cast<uint8_t*>(cropped->data) + row * w * src->ch;
            memcpy(dst_row, src_row, w * src->ch);
        }

        result.push_back(cropped);
    }

    return result;
}

std::vector<PipelineObject> NpuInferenceNode::postprocess(
    const std::vector<PipelineObject>& inputs,
    const std::vector<std::vector<float>>& npu_outputs) {
    // Implementation depends on specific model output format
    std::vector<PipelineObject> outputs;
    outputs.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        PipelineObject out = inputs[i];
        // Parse npu_outputs[i] into DetectionResult or other result type
        outputs.push_back(std::move(out));
    }

    return outputs;
}

std::vector<PipelineObject> NpuInferenceNode::inferSingle(
    const PipelineObject& input,
    const FrameResults& frame) {
    std::vector<PipelineObject> inputs = {input};
    std::vector<uint64_t> frame_ids = {input.frame_id};
    // Need to create a temporary context - this is not ideal
    // In practice, the scheduler should handle this
    return {};
}

// TransformNode implementation
TransformNode::TransformNode(const std::string& name, TransformOp op)
    : _name(name)
    , _op(op) {
}

int TransformNode::initialize(const std::string& configJson, int streamId) {
    (void)configJson;
    (void)streamId;
    return 0;
}

PipelineObject TransformNode::processObject(const PipelineObject& input,
                                            const FrameResults& frame,
                                            PipelineContext& ctx) {
    (void)ctx;
    PipelineObject output = input;

    switch (_op) {
        case RESIZE: {
            // Resize parameters from _params
            if (_params.size() >= 2) {
                int target_w = static_cast<int>(_params[0]);
                int target_h = static_cast<int>(_params[1]);
                output.metadata["resize_w"] = std::to_string(target_w);
                output.metadata["resize_h"] = std::to_string(target_h);
            }
            break;
        }
        case CROP: {
            // Crop already done by edge transform
            break;
        }
        case NORMALIZE: {
            // Normalization parameters
            if (_params.size() >= 1) {
                output.metadata["normalize"] = std::to_string(_params[0]);
            }
            break;
        }
        case LETTERBOX: {
            output.metadata["letterbox"] = "true";
            break;
        }
        case FLIP: {
            output.metadata["flip"] = "true";
            break;
        }
        case ROTATE: {
            if (_params.size() >= 1) {
                output.metadata["rotate"] = std::to_string(_params[0]);
            }
            break;
        }
    }

    return output;
}

// AggregateNode implementation
AggregateNode::AggregateNode(const std::string& name)
    : _name(name) {
}

int AggregateNode::initialize(const std::string& configJson, int streamId) {
    (void)configJson;
    (void)streamId;
    return 0;
}

std::vector<PipelineObject> AggregateNode::processBatch(
    const std::vector<PipelineObject>& inputs,
    const std::vector<uint64_t>& frame_ids,
    PipelineContext& ctx) {
    // Aggregate results from multiple branches
    // For now, just pass through
    (void)frame_ids;
    (void)ctx;
    return inputs;
}

} // namespace npu_pipeline

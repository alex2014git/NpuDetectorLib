#include "pipeline/npu_pipeline_node.hpp"
#include "pipeline/npu_pipeline_context.hpp"
#include "npu.hpp"
#include "npu_factory.hpp"
#include "common/debug_logger.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

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

// Extract NPU results and set on pipeline output object
void NpuInferenceNode::extractAndSetResults(PipelineObject& output, const npu::NpuResult& result) {
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, npu::LprResult>) {
            // Convert npu::LprResult to npu_pipeline::LprResult
            LprResult pipeline_result;
            pipeline_result.text = arg.text;
            pipeline_result.confidence = arg.confidence;
            pipeline_result.char_confidences = arg.char_confidences;
            output.setResult(_name, pipeline_result);
        } else if constexpr (std::is_same_v<T, npu::ClassificationResult>) {
            // Convert npu::ClassificationResult to npu_pipeline::ClassificationResult
            ClassificationResult pipeline_result;
            pipeline_result.class_id = arg.class_id;
            pipeline_result.label = arg.label;
            pipeline_result.confidence = arg.confidence;
            // Convert top_k
            for (const auto& [id, conf] : arg.top_k) {
                pipeline_result.top_k.push_back({id, conf});
            }
            output.setResult(_name, pipeline_result);
        } else if constexpr (std::is_same_v<T, npu::DetectionResult>) {
            // Convert npu::DetectionResult to npu_pipeline::DetectionResult
            DetectionResult pipeline_result;
            pipeline_result.class_id = arg.class_id;
            pipeline_result.confidence = arg.confidence;
            pipeline_result.bbox.y_min = arg.bbox.y_min;
            pipeline_result.bbox.x_min = arg.bbox.x_min;
            pipeline_result.bbox.y_max = arg.bbox.y_max;
            pipeline_result.bbox.x_max = arg.bbox.x_max;
            pipeline_result.bbox.confidence = arg.confidence;
            pipeline_result.bbox.class_id = arg.class_id;
            pipeline_result.bbox.class_name = arg.class_name;
            output.setResult(_name, pipeline_result);
        }
        // Other types (PoseResult, SegmentationResult) can be added as needed
    }, result);
}

// Single object processing - runs inference and extracts results
PipelineObject NpuInferenceNode::processObject(const PipelineObject& input,
                                                const FrameResults& frame,
                                                PipelineContext& ctx) {
    (void)frame;
    (void)ctx;

    if (!_npu) {
        return input;
    }

    PipelineObject output = input;

    // Prepare image data from source frame or cropped image
    image_share_t img_data = {};
    cv::Mat processed_img;

    if (input.cropped_image) {
        // Use the cropped image from previous transform
        image_share_t* img = input.cropped_image.get();
        img_data.data = img->data;
        img_data.width = img->width;
        img_data.height = img->height;
        img_data.ch = img->ch;
    } else if (frame.source_frame) {
        // Use the full source frame with ROI
        image_share_t* img = frame.source_frame.get();
        cv::Mat original(img->height, img->width, CV_8UC(img->ch), img->data);

        // Crop ROI
        const auto& roi = input.roi;
        int x = static_cast<int>(roi.x_min * img->width);
        int y = static_cast<int>(roi.y_min * img->height);
        int w = static_cast<int>((roi.x_max - roi.x_min) * img->width);
        int h = static_cast<int>((roi.y_max - roi.y_min) * img->height);

        // Clamp to image bounds
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, img->width - x);
        h = std::min(h, img->height - y);

        if (w > 0 && h > 0) {
            cv::Rect crop_rect(x, y, w, h);
            processed_img = original(crop_rect).clone();
            img_data.data = processed_img.data;
            img_data.width = processed_img.cols;
            img_data.height = processed_img.rows;
            img_data.ch = processed_img.channels();
        } else {
            // Invalid ROI, return input as-is
            return input;
        }
    } else {
        // No image data available
        return input;
    }

    // Run inference
    // Skip preprocessing if image already matches model input size (avoid letterbox padding)
    bool needPreProcess = true;
    if (_npu) {
        int model_width = _npu->GetModelWidth();
        int model_height = _npu->GetModelHeight();
        if (img_data.width == model_width && img_data.height == model_height) {
            needPreProcess = false;
        }
    }

    int ret = _npu->Detect(img_data, needPreProcess);
    if (ret < 0) {
        return output;
    }

    // Extract results using unified GetResults() API
    auto results = _npu->GetResults();
    for (const auto& result : results) {
        extractAndSetResults(output, result);
    }

    // Clear results for next inference
    _npu->ClearResults();

    return output;
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

    // For single input, still run through the same batch path to ensure
    // inference is executed and results are extracted properly

    // Batch preprocessing - collect images
    std::vector<cv::Mat> batch_images;
    batch_images.reserve(inputs.size());

    // Track which inputs have been processed (have cropped_image)
    std::vector<bool> inputs_with_cropped(inputs.size(), false);

    for (size_t i = 0; i < inputs.size(); ++i) {
        cv::Mat original;

        NPU_DEBUG("Input[" << i << "]: cropped_image=" << (inputs[i].cropped_image ? "yes" : "no")
                  << ", roi=" << inputs[i].roi.x_min << "," << inputs[i].roi.y_min
                  << "-" << inputs[i].roi.x_max << "," << inputs[i].roi.y_max);

        if (inputs[i].cropped_image) {
            inputs_with_cropped[i] = true;
            image_share_t* img = inputs[i].cropped_image.get();
            NPU_DEBUG("Using cropped image: " << img->width << "x" << img->height);
            original = cv::Mat(img->height, img->width, CV_8UC(img->ch), img->data);

            // Create image_share_t for inference directly from cropped image
            image_share_t img_data;
            img_data.data = img->data;
            img_data.width = img->width;
            img_data.height = img->height;
            img_data.ch = img->ch;

            // Run inference (rest of the logic)
            // Skip preprocessing if image already matches model input size (avoid letterbox padding)
            bool needPreProcess = true;
            if (_npu) {
                int model_width = _npu->GetModelWidth();
                int model_height = _npu->GetModelHeight();
                if (img_data.width == model_width && img_data.height == model_height) {
                    needPreProcess = false;
                }
            }

            PipelineObject out = inputs[i];
            int ret = _npu->Detect(img_data, needPreProcess);
            if (ret >= 0) {
                // Extract results using unified GetResults() API
                auto results = _npu->GetResults();
                for (const auto& result : results) {
                    extractAndSetResults(out, result);
                }
            }
            _npu->ClearResults();
            outputs.push_back(std::move(out));
            continue;  // Skip the normal processing below
        }

        // Normal path: use source frame and apply ROI cropping
        auto& frame = ctx.getFrame(frame_ids[i]);
        if (!frame.source_frame) continue;

        image_share_t* img = frame.source_frame.get();
        original = cv::Mat(img->height, img->width, CV_8UC(img->ch), img->data);

        // Get ROI - for detection nodes with full frame, this should be full frame
        const auto& roi = inputs[i].roi;

        // Check if ROI is full frame (normalized coordinates 0,0,1,1)
        // or if it's already in pixel coordinates
        int x, y, w, h;
        if (roi.x_max <= 1.0f && roi.y_max <= 1.0f) {
            // Normalized coordinates (0-1 range)
            x = static_cast<int>(roi.x_min * original.cols);
            y = static_cast<int>(roi.y_min * original.rows);
            w = static_cast<int>((roi.x_max - roi.x_min) * original.cols);
            h = static_cast<int>((roi.y_max - roi.y_min) * original.rows);
        } else {
            // Pixel coordinates
            x = static_cast<int>(roi.x_min);
            y = static_cast<int>(roi.y_min);
            w = static_cast<int>(roi.width());
            h = static_cast<int>(roi.height());
        }

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

    // Run inference on each image (only for inputs without cropped_image)
    size_t batch_idx = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs_with_cropped[i]) continue;  // Skip inputs already processed
        if (batch_idx >= batch_images.size()) break;

        PipelineObject out = inputs[i];

        // Create image_share_t for inference
        image_share_t img_data;
        img_data.data = batch_images[batch_idx].data;
        img_data.width = batch_images[batch_idx].cols;
        img_data.height = batch_images[batch_idx].rows;
        img_data.ch = batch_images[batch_idx].channels();

        // Run inference
        // Skip preprocessing if image already matches model input size (avoid letterbox padding)
        bool needPreProcess = true;
        if (_npu) {
            int model_width = _npu->GetModelWidth();
            int model_height = _npu->GetModelHeight();
            if (img_data.width == model_width && img_data.height == model_height) {
                needPreProcess = false;
            }
        }

        int ret = _npu->Detect(img_data, needPreProcess);
        if (ret >= 0) {
            // Extract results using unified GetResults() API
            auto results = _npu->GetResults();
            for (const auto& result : results) {
                // Handle detection results specially - create new PipelineObject per detection
                bool is_detection = std::visit([](auto&& arg) -> bool {
                    using T = std::decay_t<decltype(arg)>;
                    return std::is_same_v<T, npu::DetectionResult>;
                }, result);

                if (is_detection) {
                    // For detection models, create one PipelineObject per detection
                    const auto& det_result = std::get<npu::DetectionResult>(result);
                    PipelineObject det_out = out;
                    DetectionResult pipeline_result;
                    pipeline_result.class_id = det_result.class_id;
                    pipeline_result.confidence = det_result.confidence;
                    pipeline_result.bbox.y_min = det_result.bbox.y_min;
                    pipeline_result.bbox.x_min = det_result.bbox.x_min;
                    pipeline_result.bbox.y_max = det_result.bbox.y_max;
                    pipeline_result.bbox.x_max = det_result.bbox.x_max;
                    pipeline_result.bbox.class_id = det_result.class_id;
                    pipeline_result.bbox.confidence = det_result.confidence;
                    det_out.roi = pipeline_result.bbox;
                    det_out.setResult(_name, pipeline_result);
                    outputs.push_back(std::move(det_out));
                } else {
                    extractAndSetResults(out, result);
                }
            }
            // Clear the detection results for next inference
            _npu->ClearResults();
            continue;  // Skip the outputs.push_back at the end since we already added objects
        }

        outputs.push_back(std::move(out));
        batch_idx++;
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

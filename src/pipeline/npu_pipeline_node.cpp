#include "pipeline/npu_pipeline_node.hpp"
#include "pipeline/npu_pipeline_context.hpp"
#include "npu.hpp"
#include "npu_factory.hpp"
#include "core/npu_base_impl.hpp"
#include "common/debug_logger.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

namespace npu_pipeline {

// Helper function to transform coordinates from model input space to original image space
// This reverses the preprocessing transformation (letterbox or resize)
static ObjectRoi transformCoordinates(
    const ObjectRoi& model_roi,
    int model_width, int model_height,
    int original_width, int original_height,
    bool used_letterbox,
    float scale, int offset_x, int offset_y) {

    ObjectRoi original_roi;
    original_roi.class_id = model_roi.class_id;
    original_roi.class_name = model_roi.class_name;
    original_roi.confidence = model_roi.confidence;

    if (used_letterbox) {
        // LETTER_BOX mode: aspect ratio preserved, padding added
        // model_roi coordinates are in [0,1] normalized space of model input
        // Step 1: Convert to pixel coordinates in model input space
        float x_min_pixel = model_roi.x_min * model_width;
        float y_min_pixel = model_roi.y_min * model_height;
        float x_max_pixel = model_roi.x_max * model_width;
        float y_max_pixel = model_roi.y_max * model_height;

        // Step 2: Remove letterbox padding offset
        x_min_pixel -= offset_x;
        y_min_pixel -= offset_y;
        x_max_pixel -= offset_x;
        y_max_pixel -= offset_y;

        // Step 3: Scale back to original image size
        // scale = new_size / original_size, so original = pixel / scale
        x_min_pixel /= scale;
        y_min_pixel /= scale;
        x_max_pixel /= scale;
        y_max_pixel /= scale;

        // Step 4: Normalize to [0,1] based on original image dimensions
        original_roi.x_min = x_min_pixel / original_width;
        original_roi.y_min = y_min_pixel / original_height;
        original_roi.x_max = x_max_pixel / original_width;
        original_roi.y_max = y_max_pixel / original_height;
    } else {
        // Simple resize mode: aspect ratio distorted
        // Coordinates are stretched to fill the model input
        // Step 1: Convert to pixel coordinates in model input space
        float x_min_pixel = model_roi.x_min * model_width;
        float y_min_pixel = model_roi.y_min * model_height;
        float x_max_pixel = model_roi.x_max * model_width;
        float y_max_pixel = model_roi.y_max * model_height;

        // Step 2: Scale to original image dimensions independently
        // Note: This assumes uniform scaling in both dimensions during resize
        // The scale factor from PreProcessing is based on width
        float scale_x = static_cast<float>(original_width) / model_width;
        float scale_y = static_cast<float>(original_height) / model_height;

        x_min_pixel *= scale_x;
        y_min_pixel *= scale_y;
        x_max_pixel *= scale_x;
        y_max_pixel *= scale_y;

        // Step 3: Normalize to [0,1] based on original image dimensions
        original_roi.x_min = x_min_pixel / original_width;
        original_roi.y_min = y_min_pixel / original_height;
        original_roi.x_max = x_max_pixel / original_width;
        original_roi.y_max = y_max_pixel / original_height;
    }

    // Clamp to valid range [0, 1]
    original_roi.x_min = std::max(0.0f, std::min(1.0f, original_roi.x_min));
    original_roi.y_min = std::max(0.0f, std::min(1.0f, original_roi.y_min));
    original_roi.x_max = std::max(0.0f, std::min(1.0f, original_roi.x_max));
    original_roi.y_max = std::max(0.0f, std::min(1.0f, original_roi.y_max));

    return original_roi;
}

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
void NpuInferenceNode::extractAndSetResults(PipelineObject& output, const npu::NpuResult& result,
                                             int original_width, int original_height) {
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
            // Transform coordinates from model space to original image space
            ObjectRoi model_roi;
            model_roi.y_min = arg.bbox.y_min;
            model_roi.x_min = arg.bbox.x_min;
            model_roi.y_max = arg.bbox.y_max;
            model_roi.x_max = arg.bbox.x_max;
            model_roi.confidence = arg.confidence;
            model_roi.class_id = arg.class_id;
            model_roi.class_name = arg.class_name;

            // Get preprocessing parameters from NPU
            int model_width = _npu->GetModelWidth();
            int model_height = _npu->GetModelHeight();

            // Get the preprocessing parameters - need to cast to NpuBaseImpl
            // Since we know the actual type, we can use static_pointer_cast
            // But to avoid exposing implementation details, let's use dynamic_cast
            auto* base_impl = dynamic_cast<NpuBaseImpl*>(_npu.get());
            bool used_letterbox = false;
            float scale = 1.0f;
            int offset_x = 0;
            int offset_y = 0;

            if (base_impl) {
                used_letterbox = base_impl->GetLastPreprocessUsedLetterbox();
                scale = base_impl->GetLastPreprocessScale();
                offset_x = base_impl->GetLastPreprocessOffsetX();
                offset_y = base_impl->GetLastPreprocessOffsetY();
            }

            // Transform coordinates to original image space
            ObjectRoi original_roi = transformCoordinates(
                model_roi,
                model_width, model_height,
                original_width, original_height,
                used_letterbox,
                scale, offset_x, offset_y);

            DetectionResult pipeline_result;
            pipeline_result.class_id = arg.class_id;
            pipeline_result.confidence = arg.confidence;
            pipeline_result.bbox = original_roi;
            output.setResult(_name, pipeline_result);
            // Set ROI from transformed detection bbox for downstream cropping
            output.roi = original_roi;
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

    // Get original image dimensions for coordinate transformation
    int original_width = img_data.width;
    int original_height = img_data.height;

    // Extract results using unified GetResults() API
    auto results = _npu->GetResults();
    for (const auto& result : results) {
        extractAndSetResults(output, result, original_width, original_height);
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

                // Check if any results are detection results
                bool has_detections = false;
                for (const auto& result : results) {
                    bool is_detection = std::visit([](auto&& arg) -> bool {
                        using T = std::decay_t<decltype(arg)>;
                        return std::is_same_v<T, npu::DetectionResult>;
                    }, result);
                    if (is_detection) {
                        has_detections = true;
                        break;
                    }
                }

                // Get original image dimensions for coordinate transformation
                int original_width = img_data.width;
                int original_height = img_data.height;

                if (has_detections) {
                    // Get preprocessing parameters from NPU
                    int model_width = _npu->GetModelWidth();
                    int model_height = _npu->GetModelHeight();
                    auto* base_impl = dynamic_cast<NpuBaseImpl*>(_npu.get());
                    bool used_letterbox = false;
                    float scale = 1.0f;
                    int offset_x = 0;
                    int offset_y = 0;
                    if (base_impl) {
                        used_letterbox = base_impl->GetLastPreprocessUsedLetterbox();
                        scale = base_impl->GetLastPreprocessScale();
                        offset_x = base_impl->GetLastPreprocessOffsetX();
                        offset_y = base_impl->GetLastPreprocessOffsetY();
                    }

                    // For detection models, create one PipelineObject per detection
                    for (const auto& result : results) {
                        bool is_detection = std::visit([](auto&& arg) -> bool {
                            using T = std::decay_t<decltype(arg)>;
                            return std::is_same_v<T, npu::DetectionResult>;
                        }, result);

                        if (is_detection) {
                            const auto& det_result = std::get<npu::DetectionResult>(result);
                            PipelineObject det_out = out;

                            // Transform coordinates from model space to original image space
                            ObjectRoi model_roi;
                            model_roi.y_min = det_result.bbox.y_min;
                            model_roi.x_min = det_result.bbox.x_min;
                            model_roi.y_max = det_result.bbox.y_max;
                            model_roi.x_max = det_result.bbox.x_max;
                            model_roi.confidence = det_result.confidence;
                            model_roi.class_id = det_result.class_id;
                            model_roi.class_name = det_result.class_name;

                            ObjectRoi original_roi = transformCoordinates(
                                model_roi, model_width, model_height,
                                original_width, original_height,
                                used_letterbox, scale, offset_x, offset_y);

                            DetectionResult pipeline_result;
                            pipeline_result.class_id = det_result.class_id;
                            pipeline_result.confidence = det_result.confidence;
                            pipeline_result.bbox = original_roi;
                            det_out.roi = original_roi;
                            det_out.setResult(_name, pipeline_result);
                            outputs.push_back(std::move(det_out));
                        }
                    }
                } else {
                    // Non-detection results (LPR, classification, etc.)
                    for (const auto& result : results) {
                        extractAndSetResults(out, result, original_width, original_height);
                    }
                    outputs.push_back(std::move(out));
                }
            }
            _npu->ClearResults();
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

            // Check if any results are detection results
            bool has_detections = false;
            for (const auto& result : results) {
                bool is_detection = std::visit([](auto&& arg) -> bool {
                    using T = std::decay_t<decltype(arg)>;
                    return std::is_same_v<T, npu::DetectionResult>;
                }, result);
                if (is_detection) {
                    has_detections = true;
                    break;
                }
            }

            // Get original image dimensions for coordinate transformation
            int original_width = img_data.width;
            int original_height = img_data.height;

            if (has_detections) {
                // Get preprocessing parameters from NPU
                int model_width = _npu->GetModelWidth();
                int model_height = _npu->GetModelHeight();
                auto* base_impl = dynamic_cast<NpuBaseImpl*>(_npu.get());
                bool used_letterbox = false;
                float scale = 1.0f;
                int offset_x = 0;
                int offset_y = 0;
                if (base_impl) {
                    used_letterbox = base_impl->GetLastPreprocessUsedLetterbox();
                    scale = base_impl->GetLastPreprocessScale();
                    offset_x = base_impl->GetLastPreprocessOffsetX();
                    offset_y = base_impl->GetLastPreprocessOffsetY();
                }

                // For detection models, create one PipelineObject per detection
                for (const auto& result : results) {
                    bool is_detection = std::visit([](auto&& arg) -> bool {
                        using T = std::decay_t<decltype(arg)>;
                        return std::is_same_v<T, npu::DetectionResult>;
                    }, result);

                    if (is_detection) {
                        const auto& det_result = std::get<npu::DetectionResult>(result);
                        PipelineObject det_out = out;

                        // Transform coordinates from model space to original image space
                        ObjectRoi model_roi;
                        model_roi.y_min = det_result.bbox.y_min;
                        model_roi.x_min = det_result.bbox.x_min;
                        model_roi.y_max = det_result.bbox.y_max;
                        model_roi.x_max = det_result.bbox.x_max;
                        model_roi.confidence = det_result.confidence;
                        model_roi.class_id = det_result.class_id;
                        model_roi.class_name = det_result.class_name;

                        ObjectRoi original_roi = transformCoordinates(
                            model_roi, model_width, model_height,
                            original_width, original_height,
                            used_letterbox, scale, offset_x, offset_y);

                        DetectionResult pipeline_result;
                        pipeline_result.class_id = det_result.class_id;
                        pipeline_result.confidence = det_result.confidence;
                        pipeline_result.bbox = original_roi;
                        det_out.roi = original_roi;
                        det_out.setResult(_name, pipeline_result);
                        outputs.push_back(std::move(det_out));
                    }
                }
            } else {
                // Non-detection results (LPR, classification, etc.)
                for (const auto& result : results) {
                    extractAndSetResults(out, result, original_width, original_height);
                }
                outputs.push_back(std::move(out));
            }

            // Clear the detection results for next inference
            _npu->ClearResults();
        } else {
            // Inference failed, still add the output
            outputs.push_back(std::move(out));
        }

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

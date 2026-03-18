#include "pipeline/npu_pipeline_node.hpp"
#include "pipeline/npu_pipeline_context.hpp"
#include "npu.hpp"
#include "npu_factory.hpp"
#include "core/npu_base_alg_impl.hpp"
#include "core/npu_detection_impl.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

namespace npu_pipeline {

// LPR charset for decoding (same as yolo_lpr_async reference)
static const char* g_lpr_charset[] = {
    "#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川",
    "贵","云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民","航","危",
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","险","品","I","O","-"
};
static constexpr size_t g_lpr_charset_size = sizeof(g_lpr_charset) / sizeof(g_lpr_charset[0]);

// Decode LPR output using CTC-style decoding (skip duplicates and blanks)
static std::string decode_lpr_output(const float* output, int output_size) {
    std::string plate;
    std::string prev = "#";

    for (int i = 0; i < output_size; ++i) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            continue;
        }
        int idx = static_cast<int>(std::round(output[i]));
        if (idx < 0 || idx >= static_cast<int>(g_lpr_charset_size)) {
            continue;
        }
        const std::string& c = g_lpr_charset[idx];
        if (c != "#" && c != prev) {
            plate += c;
        }
        prev = c;
    }

    return plate;
}

// Decode classification output (argmax + top-k)
static ClassificationResult decode_classification_output(const float* output, int output_size, const std::vector<std::string>& labels) {
    ClassificationResult result;

    // Find argmax
    auto max_it = std::max_element(output, output + output_size);
    int max_idx = std::distance(output, max_it);

    result.class_id = max_idx;
    result.confidence = *max_it;
    if (max_idx >= 0 && max_idx < static_cast<int>(labels.size())) {
        result.label = labels[max_idx];
    } else {
        result.label = "class_" + std::to_string(max_idx);
    }

    // Build top-5
    std::vector<std::pair<float, int>> scored;
    scored.reserve(output_size);
    for (int i = 0; i < output_size; ++i) {
        scored.push_back({output[i], i});
    }
    std::partial_sort(scored.begin(), scored.begin() + std::min(5, output_size), scored.end(), std::greater<>());

    for (int i = 0; i < std::min(5, output_size); ++i) {
        result.top_k.push_back({scored[i].second, scored[i].first});
    }

    return result;
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
    // Note: For LPR and classification models, always apply preprocessing to ensure
    // proper color conversion (BGR to RGB) and normalization, even if dimensions match.
    bool needPreProcess = true;
    if (_npu && _algorithm_type != ALG_LPR && _algorithm_type != ALG_CLASSIFICATION) {
        int model_width = _npu->GetModelWidth();
        int model_height = _npu->GetModelHeight();
        if (img_data.width == model_width && img_data.height == model_height) {
            needPreProcess = false;
        }
    }

    // Debug output for LPR
    if (_algorithm_type == ALG_LPR) {
        int model_width = _npu->GetModelWidth();
        int model_height = _npu->GetModelHeight();
        std::cout << "[LPR DEBUG] Image: " << img_data.width << "x" << img_data.height
                  << "x" << img_data.ch << ", Model: " << model_width << "x" << model_height
                  << ", needPreProcess=" << needPreProcess << std::endl;
    }

    int ret = _npu->Detect(img_data, needPreProcess);
    if (ret < 0) {
        return output;
    }

    // Extract results based on algorithm type
    switch (_algorithm_type) {
        case ALG_LPR: {
            auto* alg_impl = dynamic_cast<NpuBaseAlgImpl*>(_npu.get());
            if (alg_impl && !alg_impl->GetRawOutputFloat().empty()) {
                const auto& output_buffer = alg_impl->GetRawOutputFloat()[0];
                LprResult lpr_result;
                lpr_result.text = decode_lpr_output(output_buffer.data(), static_cast<int>(output_buffer.size()));
                lpr_result.confidence = 1.0f;
                output.setResult(_name, lpr_result);
            }
            break;
        }
        case ALG_CLASSIFICATION: {
            auto* alg_impl = dynamic_cast<NpuBaseAlgImpl*>(_npu.get());
            if (alg_impl && !alg_impl->GetRawOutputFloat().empty()) {
                const auto& output_buffer = alg_impl->GetRawOutputFloat()[0];
                ClassificationResult cls_result = decode_classification_output(
                    output_buffer.data(), static_cast<int>(output_buffer.size()), {});
                output.setResult(_name, cls_result);
            }
            break;
        }
        case ALG_YOLO_V5:
        case ALG_YOLO_V8:
        case ALG_YOLO_NMS:
        case ALG_POSE:
        case ALG_YOLO_V8_SEG:
        default: {
            // Detection models - extract results from NPU's internal storage
            auto* det_impl = dynamic_cast<NpuDetectionImpl*>(_npu.get());
            if (det_impl) {
                const auto& detections = det_impl->GetDetectionResults();
                if (!detections.empty()) {
                    // Create DetectionResult for each detected object
                    for (const auto& obj : detections) {
                        DetectionResult det_result;
                        det_result.class_id = obj.category;
                        det_result.confidence = obj.confidence;
                        det_result.bbox.x_min = obj.x_min;
                        det_result.bbox.y_min = obj.y_min;
                        det_result.bbox.x_max = obj.x_max;
                        det_result.bbox.y_max = obj.y_max;
                        output.setResult(_name, det_result);
                    }
                } else {
                    // No detections
                    DetectionResult det_result;
                    det_result.class_id = -1;
                    det_result.confidence = 0.0f;
                    output.setResult(_name, det_result);
                }
                // Clear the detection results for next inference
                det_impl->ClearDetectionResults();
            } else {
                // Fallback: create result from ROI info
                DetectionResult det_result;
                det_result.class_id = output.roi.class_id;
                det_result.confidence = output.roi.confidence;
                det_result.bbox = output.roi;
                output.setResult(_name, det_result);
            }
            break;
        }
    }

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

        // First check if we have a cropped image from a previous transform (e.g., crop ROI edge)
        std::cout << "[DEBUG] Input[" << i << "]: cropped_image=" << (inputs[i].cropped_image ? "yes" : "no")
                  << ", roi=" << inputs[i].roi.x_min << "," << inputs[i].roi.y_min
                  << "-" << inputs[i].roi.x_max << "," << inputs[i].roi.y_max << std::endl;

        if (inputs[i].cropped_image) {
            inputs_with_cropped[i] = true;
            image_share_t* img = inputs[i].cropped_image.get();
            std::cout << "[DEBUG] Using cropped image: " << img->width << "x" << img->height << std::endl;
            original = cv::Mat(img->height, img->width, CV_8UC(img->ch), img->data);

            // Create image_share_t for inference directly from cropped image
            image_share_t img_data;
            img_data.data = img->data;
            img_data.width = img->width;
            img_data.height = img->height;
            img_data.ch = img->ch;

            // Run inference (rest of the logic)
            // Skip preprocessing if image already matches model input size (avoid letterbox padding)
            // Note: For LPR and classification models, always apply preprocessing to ensure
            // proper color conversion (BGR to RGB) and normalization, even if dimensions match.
            bool needPreProcess = true;
            if (_npu && _algorithm_type != ALG_LPR && _algorithm_type != ALG_CLASSIFICATION) {
                int model_width = _npu->GetModelWidth();
                int model_height = _npu->GetModelHeight();
                if (img_data.width == model_width && img_data.height == model_height) {
                    needPreProcess = false;
                }
            }

            // Debug output for LPR
            if (_algorithm_type == ALG_LPR) {
                int model_width = _npu->GetModelWidth();
                int model_height = _npu->GetModelHeight();
                std::cout << "[LPR DEBUG] Batch cropped: Image: " << img_data.width << "x" << img_data.height
                          << "x" << img_data.ch << ", Model: " << model_width << "x" << model_height
                          << ", needPreProcess=" << needPreProcess << std::endl;
            }

            PipelineObject out = inputs[i];
            int ret = _npu->Detect(img_data, needPreProcess);
            if (ret >= 0) {
                switch (_algorithm_type) {
                    case ALG_LPR: {
                        auto* alg_impl = dynamic_cast<NpuBaseAlgImpl*>(_npu.get());
                        if (alg_impl && !alg_impl->GetRawOutputFloat().empty()) {
                            const auto& output_buffer = alg_impl->GetRawOutputFloat()[0];
                            LprResult lpr_result;
                            lpr_result.text = decode_lpr_output(output_buffer.data(), static_cast<int>(output_buffer.size()));
                            lpr_result.confidence = 1.0f;
                            out.setResult(_name, lpr_result);
                        }
                        break;
                    }
                    case ALG_CLASSIFICATION: {
                        auto* alg_impl = dynamic_cast<NpuBaseAlgImpl*>(_npu.get());
                        if (alg_impl && !alg_impl->GetRawOutputFloat().empty()) {
                            const auto& output_buffer = alg_impl->GetRawOutputFloat()[0];
                            ClassificationResult cls_result = decode_classification_output(
                                output_buffer.data(), static_cast<int>(output_buffer.size()), {});
                            out.setResult(_name, cls_result);
                        }
                        break;
                    }
                    case ALG_YOLO_V5:
                    case ALG_YOLO_V8:
                    case ALG_YOLO_NMS:
                    case ALG_POSE:
                    case ALG_YOLO_V8_SEG: {
                        // Detection models - extract results from NPU's internal storage
                        auto* det_impl = dynamic_cast<NpuDetectionImpl*>(_npu.get());
                        if (det_impl) {
                            const auto& detections = det_impl->GetDetectionResults();
                            if (!detections.empty()) {
                                // Create one PipelineObject per detection
                                for (const auto& obj : detections) {
                                    PipelineObject det_out = out;
                                    DetectionResult det_result;
                                    det_result.class_id = obj.category;
                                    det_result.confidence = obj.confidence;
                                    det_result.bbox.x_min = obj.x_min;
                                    det_result.bbox.y_min = obj.y_min;
                                    det_result.bbox.x_max = obj.x_max;
                                    det_result.bbox.y_max = obj.y_max;
                                    det_result.bbox.class_id = obj.category;
                                    det_result.bbox.confidence = obj.confidence;
                                    det_out.roi = det_result.bbox;
                                    det_out.setResult(_name, det_result);
                                    outputs.push_back(std::move(det_out));
                                }
                            } else {
                                // No detections - create empty result
                                DetectionResult det_result;
                                det_result.class_id = -1;
                                det_result.confidence = 0.0f;
                                out.setResult(_name, det_result);
                                outputs.push_back(std::move(out));
                            }
                            det_impl->ClearDetectionResults();
                        } else {
                            outputs.push_back(std::move(out));
                        }
                        break;
                    }
                    default:
                        outputs.push_back(std::move(out));
                        break;
                }
            }
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
        // Note: For LPR and classification models, always apply preprocessing to ensure
        // proper color conversion (BGR to RGB) and normalization, even if dimensions match.
        bool needPreProcess = true;
        if (_npu && _algorithm_type != ALG_LPR && _algorithm_type != ALG_CLASSIFICATION) {
            int model_width = _npu->GetModelWidth();
            int model_height = _npu->GetModelHeight();
            if (img_data.width == model_width && img_data.height == model_height) {
                needPreProcess = false;
            }
        }

        // Debug output for LPR
        if (_algorithm_type == ALG_LPR) {
            int model_width = _npu->GetModelWidth();
            int model_height = _npu->GetModelHeight();
            std::cout << "[LPR DEBUG] Batch: Image: " << img_data.width << "x" << img_data.height
                      << "x" << img_data.ch << ", Model: " << model_width << "x" << model_height
                      << ", needPreProcess=" << needPreProcess << std::endl;
        }

        int ret = _npu->Detect(img_data, needPreProcess);
        if (ret >= 0) {
            // Get algorithm-specific results
            switch (_algorithm_type) {
                case ALG_LPR: {
                    // Get raw output from NpuBaseAlgImpl
                    auto* alg_impl = dynamic_cast<NpuBaseAlgImpl*>(_npu.get());
                    if (alg_impl && !alg_impl->GetRawOutputFloat().empty()) {
                        const auto& output_buffer = alg_impl->GetRawOutputFloat()[0];
                        LprResult lpr_result;
                        lpr_result.text = decode_lpr_output(output_buffer.data(), static_cast<int>(output_buffer.size()));
                        lpr_result.confidence = 1.0f; // Could calculate from output if needed
                        out.setResult(_name, lpr_result);
                    }
                    break;
                }
                case ALG_CLASSIFICATION: {
                    // Get raw output from NpuBaseAlgImpl
                    auto* alg_impl = dynamic_cast<NpuBaseAlgImpl*>(_npu.get());
                    if (alg_impl && !alg_impl->GetRawOutputFloat().empty()) {
                        const auto& output_buffer = alg_impl->GetRawOutputFloat()[0];
                        // Get labels from the NPU implementation if available
                        ClassificationResult cls_result = decode_classification_output(
                            output_buffer.data(), static_cast<int>(output_buffer.size()), {});
                        out.setResult(_name, cls_result);
                    }
                    break;
                }
                case ALG_YOLO_V5:
                case ALG_YOLO_V8:
                case ALG_YOLO_NMS:
                case ALG_POSE:
                case ALG_YOLO_V8_SEG:
                default: {
                    // Detection models - extract results from NPU's internal storage
                    // The NPU has already stored detection results in _objects vector
                    auto* det_impl = dynamic_cast<NpuDetectionImpl*>(_npu.get());
                    if (det_impl) {
                        const auto& detections = det_impl->GetDetectionResults();
                        if (!detections.empty()) {
                            // Create one PipelineObject per detection
                            for (const auto& obj : detections) {
                                PipelineObject det_out = out;
                                DetectionResult det_result;
                                det_result.class_id = obj.category;
                                det_result.confidence = obj.confidence;
                                // Copy bbox from object_roi_t to ObjectRoi
                                det_result.bbox.x_min = obj.x_min;
                                det_result.bbox.y_min = obj.y_min;
                                det_result.bbox.x_max = obj.x_max;
                                det_result.bbox.y_max = obj.y_max;
                                det_result.bbox.class_id = obj.category;
                                det_result.bbox.confidence = obj.confidence;
                                det_out.roi = det_result.bbox;  // Set the ROI for downstream processing
                                det_out.setResult(_name, det_result);
                                outputs.push_back(std::move(det_out));
                            }
                        } else {
                            // No detections - still create a result indicating this
                            DetectionResult det_result;
                            det_result.class_id = -1;
                            det_result.confidence = 0.0f;
                            out.setResult(_name, det_result);
                            outputs.push_back(std::move(out));
                        }
                        // Clear the detection results for next inference
                        det_impl->ClearDetectionResults();
                        continue;  // Skip the outputs.push_back at the end since we already added objects
                    } else {
                        // Fallback: create result from ROI info
                        DetectionResult det_result;
                        det_result.class_id = out.roi.class_id;
                        det_result.confidence = out.roi.confidence;
                        det_result.bbox = out.roi;
                        out.setResult(_name, det_result);
                    }
                    break;
                }
            }
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

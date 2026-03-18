#include "pipeline/npu_pipeline_edge.hpp"
#include "pipeline/npu_pipeline_context.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace npu_pipeline {

// Execute transform
std::vector<PipelineObject> PipelineEdge::execute(const std::vector<PipelineObject>& input,
                                                 const FrameResults& frame,
                                                 PipelineContext& ctx) const {
    if (transform_type == CUSTOM && custom_transform) {
        return custom_transform(input, frame, ctx);
    }
    // Handle transforms that need the destination node ID
    if (transform_type == FILTER_TRACK_UNPROCESSED) {
        return Transforms::filterUnprocessed(input, frame, ctx, to_node);
    }
    if (transform_type == BATCH_ACCUMULATE) {
        return Transforms::batchAccumulate(input, frame, ctx, to_node, params);
    }
    return makeTransform(transform_type, params)(input, frame, ctx);
}

// Built-in transforms factory
TransformFunc PipelineEdge::makeTransform(TransformType type, const Params& p) {
    switch (type) {
        case PASS_THROUGH:
            return Transforms::passThrough;

        case FILTER_CLASS:
            return [p](const auto& input, const auto& frame, auto& ctx) {
                return Transforms::filterByClass(input, frame, ctx, p.target_classes);
            };

        case FILTER_CONFIDENCE:
            return [p](const auto& input, const auto& frame, auto& ctx) {
                return Transforms::filterByConfidence(input, frame, ctx, p.confidence_threshold);
            };

        case CROP_ROI:
            return [p](const auto& input, const auto& frame, auto& ctx) {
                return Transforms::cropRoi(input, frame, ctx, p);
            };

        case CROP_ROI_PADDED:
            return [p](const auto& input, const auto& frame, auto& ctx) {
                return Transforms::cropRoiPadded(input, frame, ctx, p.padding_ratio);
            };

        case FILTER_TRACK_NEW:
            return Transforms::filterNewTracks;

        case FILTER_TRACK_UNPROCESSED:
            return [](const auto& input, const auto& frame, auto& ctx) {
                return Transforms::filterUnprocessed(input, frame, ctx, "");
            };

        case FILTER_TRACK_ACTIVE:
            return Transforms::filterActiveTracks;

        case BATCH_ACCUMULATE:
            // Batch accumulation returns empty - handled by scheduler
            return [p](const auto& input, const auto& frame, auto& ctx) {
                return Transforms::batchAccumulate(input, frame, ctx, "", p);
            };

        default:
            return Transforms::passThrough;
    }
}

// Transform implementations
namespace Transforms {

std::vector<PipelineObject> passThrough(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx) {
    (void)frame;
    (void)ctx;
    return input;
}

std::vector<PipelineObject> filterByClass(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx,
    const std::vector<int>& target_classes) {
    (void)frame;
    (void)ctx;

    std::vector<PipelineObject> output;
    output.reserve(input.size());

    for (const auto& obj : input) {
        if (std::find(target_classes.begin(), target_classes.end(),
                      obj.roi.class_id) != target_classes.end()) {
            output.push_back(obj);
        }
    }

    return output;
}

std::vector<PipelineObject> filterByConfidence(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx,
    float threshold) {
    (void)frame;
    (void)ctx;

    std::vector<PipelineObject> output;
    output.reserve(input.size());

    for (const auto& obj : input) {
        if (obj.roi.confidence >= threshold) {
            output.push_back(obj);
        }
    }

    return output;
}

std::vector<PipelineObject> cropRoi(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx,
    const PipelineEdge::Params& params) {
    (void)ctx;

    std::vector<PipelineObject> output;
    output.reserve(input.size());

    // Get source frame for cropping
    if (!frame.source_frame) {
        return output;
    }

    image_share_t* src = frame.source_frame.get();
    if (!src->data || src->width <= 0 || src->height <= 0) {
        return output;
    }

    // Create OpenCV Mat from source frame (no copy, just wrapper)
    cv::Mat src_mat(src->height, src->width, CV_8UC(src->ch), src->data);

    for (auto obj : input) {
        // Filter by class if specified
        if (!params.target_classes.empty()) {
            if (std::find(params.target_classes.begin(), params.target_classes.end(),
                          obj.roi.class_id) == params.target_classes.end()) {
                continue;
            }
        }

        // Calculate pixel coordinates from normalized ROI
        int x = static_cast<int>(obj.roi.x_min * src->width);
        int y = static_cast<int>(obj.roi.y_min * src->height);
        int w = static_cast<int>(obj.roi.width() * src->width);
        int h = static_cast<int>(obj.roi.height() * src->height);

        // Clamp to image bounds
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, src->width - x);
        h = std::min(h, src->height - y);

        // Skip invalid crops
        if (w <= 0 || h <= 0) {
            continue;
        }

        // Apply min/max size constraints if specified
        if (params.min_crop_width > 0 && w < params.min_crop_width) {
            continue;
        }
        if (params.min_crop_height > 0 && h < params.min_crop_height) {
            continue;
        }
        if (params.max_crop_width > 0 && w > params.max_crop_width) {
            continue;
        }
        if (params.max_crop_height > 0 && h > params.max_crop_height) {
            continue;
        }

        // Perform the crop using OpenCV (clone() makes a deep copy)
        cv::Rect crop_rect(x, y, w, h);
        std::cout << "[CROP DEBUG] Cropping ROI: x=" << x << " y=" << y << " w=" << w << " h=" << h
                  << " from source " << src->width << "x" << src->height << std::endl;
        cv::Mat cropped_mat = src_mat(crop_rect).clone();

        // Resize to target dimensions if specified (e.g., for LPR model input)
        if (params.target_width > 0 && params.target_height > 0) {
            cv::Mat resized_mat;
            cv::resize(cropped_mat, resized_mat,
                       cv::Size(params.target_width, params.target_height),
                       0, 0, cv::INTER_LINEAR);
            cropped_mat = std::move(resized_mat);
            std::cout << "[CROP DEBUG] Resized crop from " << w << "x" << h
                      << " to " << params.target_width << "x" << params.target_height << std::endl;
        }

        // Create new image_share_t for the cropped image with custom deleter
        auto cropped = std::shared_ptr<image_share_t>(new image_share_t{},
            [](image_share_t* img) {
                if (img && img->data) {
                    delete[] static_cast<uint8_t*>(img->data);
                    img->data = nullptr;
                }
                delete img;
            });
        cropped->width = cropped_mat.cols;
        cropped->height = cropped_mat.rows;
        cropped->ch = src->ch;
        cropped->data = new uint8_t[cropped->width * cropped->height * src->ch];

        // Copy pixel data from OpenCV Mat to our buffer
        memcpy(cropped->data, cropped_mat.data, cropped->width * cropped->height * src->ch);

        std::cout << "[CROP DEBUG] Created cropped image: " << cropped->width << "x" << cropped->height << std::endl;

        // Store crop metadata for reference
        obj.metadata["crop_x"] = std::to_string(x);
        obj.metadata["crop_y"] = std::to_string(y);
        obj.metadata["crop_w"] = std::to_string(w);
        obj.metadata["crop_h"] = std::to_string(h);

        // Set the cropped image on the object
        obj.cropped_image = std::move(cropped);

        output.push_back(std::move(obj));
    }

    return output;
}

std::vector<PipelineObject> cropRoiPadded(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx,
    float padding_ratio) {
    (void)ctx;

    std::vector<PipelineObject> output;
    output.reserve(input.size());

    for (auto obj : input) {
        // Calculate padded coordinates
        float w = obj.roi.width();
        float h = obj.roi.height();
        float pad_x = w * padding_ratio;
        float pad_y = h * padding_ratio;

        obj.metadata["crop_x"] = std::to_string(static_cast<int>(obj.roi.x_min - pad_x));
        obj.metadata["crop_y"] = std::to_string(static_cast<int>(obj.roi.y_min - pad_y));
        obj.metadata["crop_w"] = std::to_string(static_cast<int>(w + 2 * pad_x));
        obj.metadata["crop_h"] = std::to_string(static_cast<int>(h + 2 * pad_y));
        obj.metadata["padded"] = "true";

        output.push_back(std::move(obj));
    }

    return output;
}

std::vector<PipelineObject> filterNewTracks(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx) {

    std::vector<PipelineObject> output;
    output.reserve(input.size());

    for (const auto& obj : input) {
        if (obj.track_id != 0) {
            auto& track = ctx.getTrackState(obj.frame_id, obj.track_id);
            if (track.isNew()) {
                output.push_back(obj);
            }
        }
    }

    return output;
}

std::vector<PipelineObject> filterUnprocessed(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx,
    const std::string& node_id) {

    std::vector<PipelineObject> output;
    output.reserve(input.size());

    for (const auto& obj : input) {
        if (obj.track_id != 0) {
            auto& track = ctx.getTrackState(obj.frame_id, obj.track_id);
            if (!track.wasProcessed(node_id)) {
                output.push_back(obj);
            }
        } else {
            // Untracked objects - pass through
            output.push_back(obj);
        }
    }

    return output;
}

std::vector<PipelineObject> filterActiveTracks(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx) {

    std::vector<PipelineObject> output;
    output.reserve(input.size());

    for (const auto& obj : input) {
        if (obj.track_id != 0) {
            auto& track = ctx.getTrackState(obj.frame_id, obj.track_id);
            if (track.isActive()) {
                output.push_back(obj);
            }
        } else {
            // Untracked objects - pass through
            output.push_back(obj);
        }
    }

    return output;
}

std::vector<PipelineObject> batchAccumulate(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx,
    const std::string& node_id,
    const PipelineEdge::Params& params) {

    // Add items to batch accumulator
    for (const auto& obj : input) {
        BatchItem item;
        item.frame_id = obj.frame_id;
        item.object_id = obj.object_id;
        item.object = obj;

        ctx.accumulateForBatch(node_id, std::move(item),
                               params.batch_size, params.batch_timeout);
    }

    // Return empty - batch processing happens in scheduler
    return {};
}

} // namespace Transforms

} // namespace npu_pipeline

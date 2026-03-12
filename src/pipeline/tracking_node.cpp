#include "pipeline/npu_pipeline_types.hpp"
#include "pipeline/npu_pipeline_context.hpp"
#include "pipeline/npu_pipeline_node.hpp"
#include <algorithm>
#include <cmath>

namespace npu_pipeline {

// Simple IoU calculation for bounding boxes
float calculateIoU(const ObjectRoi& a, const ObjectRoi& b) {
    float x_left = std::max(a.x_min, b.x_min);
    float y_top = std::max(a.y_min, b.y_min);
    float x_right = std::min(a.x_max, b.x_max);
    float y_bottom = std::min(a.y_max, b.y_max);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float a_area = a.area();
    float b_area = b.area();

    return intersection_area / (a_area + b_area - intersection_area);
}

// Tracking node implementation
class TrackingNode : public PipelineNode {
public:
    TrackingNode(const std::string& name, float iou_threshold = 0.5f,
                 int max_frames_without_update = 30);
    ~TrackingNode() override = default;

    const std::string& getName() const override { return _name; }
    const std::string& getType() const override { return _type; }

    int initialize(const std::string& configJson, int streamId) override;

    // Single object processing (not used - we process frame-level)
    PipelineObject processObject(const PipelineObject& input,
                                  const FrameResults& frame,
                                  PipelineContext& ctx) override;

    // Frame-level batch processing
    std::vector<PipelineObject> processBatch(
        const std::vector<PipelineObject>& inputs,
        const std::vector<uint64_t>& frame_ids,
        PipelineContext& ctx) override;

    std::vector<std::string> getInputNodes() const override { return _input_nodes; }
    void setInputNodes(const std::vector<std::string>& nodes) override { _input_nodes = nodes; }

    void reset() override;

private:
    std::string _name;
    std::string _type = "tracking";
    std::vector<std::string> _input_nodes;

    // Tracking parameters
    float _iou_threshold;
    int _max_frames_without_update;

    // Track state across frames (persists beyond single frame)
    struct InternalTrack {
        uint64_t id;
        ObjectRoi roi;
        int class_id;
        int frames_tracked;
        int frames_without_update;
        bool updated_this_frame;
    };

    std::unordered_map<uint64_t, InternalTrack> _active_tracks;
    uint64_t _next_track_id = 1;

    // Hungarian algorithm for optimal assignment
    std::vector<std::pair<size_t, size_t>> hungarianMatching(
        const std::vector<std::vector<float>>& cost_matrix);

    // Update tracks with new detections
    std::vector<PipelineObject> updateTracks(
        const std::vector<PipelineObject>& detections,
        uint64_t frame_id,
        PipelineContext& ctx);
};

TrackingNode::TrackingNode(const std::string& name, float iou_threshold,
                            int max_frames_without_update)
    : _name(name)
    , _iou_threshold(iou_threshold)
    , _max_frames_without_update(max_frames_without_update) {
}

int TrackingNode::initialize(const std::string& configJson, int streamId) {
    (void)streamId;
    // Parse config if needed
    // Example: {"iou_threshold": 0.5, "max_frames_without_update": 30}
    return 0;
}

void TrackingNode::reset() {
    _active_tracks.clear();
    _next_track_id = 1;
}

PipelineObject TrackingNode::processObject(const PipelineObject& input,
                                           const FrameResults& frame,
                                           PipelineContext& ctx) {
    (void)frame;
    (void)ctx;
    return input;
}

std::vector<PipelineObject> TrackingNode::processBatch(
    const std::vector<PipelineObject>& inputs,
    const std::vector<uint64_t>& frame_ids,
    PipelineContext& ctx) {

    if (inputs.empty()) {
        return {};
    }

    // Process each frame's detections separately
    // Note: inputs may come from multiple frames in batch mode
    std::unordered_map<uint64_t, std::vector<PipelineObject>> frame_detections;

    for (size_t i = 0; i < inputs.size(); ++i) {
        frame_detections[frame_ids[i]].push_back(inputs[i]);
    }

    std::vector<PipelineObject> all_outputs;

    for (auto& [frame_id, detections] : frame_detections) {
        auto outputs = updateTracks(detections, frame_id, ctx);
        all_outputs.insert(all_outputs.end(), outputs.begin(), outputs.end());
    }

    return all_outputs;
}

std::vector<PipelineObject> TrackingNode::updateTracks(
    const std::vector<PipelineObject>& detections,
    uint64_t frame_id,
    PipelineContext& ctx) {

    // Reset update flags
    for (auto& [id, track] : _active_tracks) {
        track.updated_this_frame = false;
    }

    // Mark all tracks as aged
    for (auto& [id, track] : _active_tracks) {
        track.frames_without_update++;
    }

    // Build cost matrix (IoU between tracks and detections)
    std::vector<uint64_t> track_ids;
    std::vector<InternalTrack*> track_ptrs;

    for (auto& [id, track] : _active_tracks) {
        if (track.frames_without_update <= _max_frames_without_update) {
            track_ids.push_back(id);
            track_ptrs.push_back(&track);
        }
    }

    std::vector<PipelineObject> outputs;
    outputs.reserve(detections.size());

    if (track_ptrs.empty()) {
        // No active tracks - create new tracks for all detections
        for (const auto& det : detections) {
            uint64_t new_id = _next_track_id++;

            InternalTrack track;
            track.id = new_id;
            track.roi = det.roi;
            track.class_id = det.roi.class_id;
            track.frames_tracked = 1;
            track.frames_without_update = 0;
            track.updated_this_frame = true;

            _active_tracks[new_id] = track;

            PipelineObject out = det;
            out.track_id = new_id;
            outputs.push_back(out);

            // Update context track state
            TrackState& state = ctx.getTrackState(frame_id, new_id);
            state.id = new_id;
            state.last_roi = det.roi;
            state.frames_tracked = 1;
            state.frames_since_update = 0;
        }
    } else {
        // Build cost matrix
        std::vector<std::vector<float>> cost_matrix;
        cost_matrix.reserve(track_ptrs.size());

        for (const auto* track : track_ptrs) {
            std::vector<float> row;
            row.reserve(detections.size());
            for (const auto& det : detections) {
                float iou = calculateIoU(track->roi, det.roi);
                // Cost is negative IoU for maximization
                row.push_back(1.0f - iou);
            }
            cost_matrix.push_back(row);
        }

        // Find optimal assignment (simplified greedy matching)
        std::vector<bool> detection_matched(detections.size(), false);
        std::vector<bool> track_matched(track_ptrs.size(), false);

        // Greedy matching by best IoU
        for (size_t t = 0; t < track_ptrs.size(); ++t) {
            float best_iou = _iou_threshold;
            size_t best_det = detections.size();

            for (size_t d = 0; d < detections.size(); ++d) {
                if (detection_matched[d]) continue;

                float iou = 1.0f - cost_matrix[t][d];
                if (iou > best_iou) {
                    best_iou = iou;
                    best_det = d;
                }
            }

            if (best_det < detections.size()) {
                // Match found
                track_ptrs[t]->roi = detections[best_det].roi;
                track_ptrs[t]->class_id = detections[best_det].roi.class_id;
                track_ptrs[t]->frames_tracked++;
                track_ptrs[t]->frames_without_update = 0;
                track_ptrs[t]->updated_this_frame = true;

                PipelineObject out = detections[best_det];
                out.track_id = track_ptrs[t]->id;
                outputs.push_back(out);

                // Update context
                TrackState& state = ctx.getTrackState(frame_id, track_ptrs[t]->id);
                state.id = track_ptrs[t]->id;
                state.last_roi = detections[best_det].roi;
                state.frames_tracked = track_ptrs[t]->frames_tracked;
                state.frames_since_update = 0;

                detection_matched[best_det] = true;
                track_matched[t] = true;
            }
        }

        // Create new tracks for unmatched detections
        for (size_t d = 0; d < detections.size(); ++d) {
            if (detection_matched[d]) continue;

            uint64_t new_id = _next_track_id++;

            InternalTrack track;
            track.id = new_id;
            track.roi = detections[d].roi;
            track.class_id = detections[d].roi.class_id;
            track.frames_tracked = 1;
            track.frames_without_update = 0;
            track.updated_this_frame = true;

            _active_tracks[new_id] = track;

            PipelineObject out = detections[d];
            out.track_id = new_id;
            outputs.push_back(out);

            // Update context
            TrackState& state = ctx.getTrackState(frame_id, new_id);
            state.id = new_id;
            state.last_roi = detections[d].roi;
            state.frames_tracked = 1;
            state.frames_since_update = 0;
        }
    }

    // Update context track states for unmatched tracks
    for (const auto& [id, track] : _active_tracks) {
        if (!track.updated_this_frame && track.frames_without_update <= _max_frames_without_update) {
            TrackState& state = ctx.getTrackState(frame_id, id);
            state.id = id;
            state.last_roi = track.roi;
            state.frames_tracked = track.frames_tracked;
            state.frames_since_update = track.frames_without_update;
        }
    }

    // Remove lost tracks
    for (auto it = _active_tracks.begin(); it != _active_tracks.end();) {
        if (it->second.frames_without_update > _max_frames_without_update) {
            it = _active_tracks.erase(it);
        } else {
            ++it;
        }
    }

    return outputs;
}

// Simplified Hungarian algorithm (greedy approximation)
std::vector<std::pair<size_t, size_t>> TrackingNode::hungarianMatching(
    const std::vector<std::vector<float>>& cost_matrix) {

    std::vector<std::pair<size_t, size_t>> matches;

    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return matches;
    }

    size_t rows = cost_matrix.size();
    size_t cols = cost_matrix[0].size();

    std::vector<bool> row_covered(rows, false);
    std::vector<bool> col_covered(cols, false);

    // Greedy matching
    for (size_t i = 0; i < rows; ++i) {
        float min_val = std::numeric_limits<float>::max();
        size_t min_col = 0;

        for (size_t j = 0; j < cols; ++j) {
            if (!col_covered[j] && cost_matrix[i][j] < min_val) {
                min_val = cost_matrix[i][j];
                min_col = j;
            }
        }

        if (!col_covered[min_col]) {
            matches.push_back({i, min_col});
            col_covered[min_col] = true;
            row_covered[i] = true;
        }
    }

    return matches;
}

} // namespace npu_pipeline

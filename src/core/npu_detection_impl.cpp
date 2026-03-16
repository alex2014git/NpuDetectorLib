#include "core/npu_detection_impl.hpp"
#include "common.hpp"
#include <opencv2/imgproc.hpp>

NpuDetectionImpl::NpuDetectionImpl() : NpuBaseImpl() {
    // Detection-specific initialization
}

NpuDetectionImpl::~NpuDetectionImpl() {
    // Cleanup handled by base
}

// Get unified results (converts _objects to NpuResult)
std::vector<npu::NpuResult> NpuDetectionImpl::GetResults() const {
    std::vector<npu::NpuResult> results;
    results.reserve(_objects.size());

    for (const auto& obj : _objects) {
        npu::DetectionResult det;
        det.class_id = obj.category;
        det.class_name = obj.name;
        det.confidence = obj.confidence;
        det.bbox.x_min = obj.x_min;
        det.bbox.y_min = obj.y_min;
        det.bbox.x_max = obj.x_max;
        det.bbox.y_max = obj.y_max;
        results.push_back(det);
    }

    return results;
}

// Clear results for next inference
void NpuDetectionImpl::ClearResults() {
    _objects.clear();
    _objects.shrink_to_fit();
}

// Draw detection results on image
void NpuDetectionImpl::DrawResult(image_share_t imgData, bool needFormat) {
    int width = imgData.width;
    int height = imgData.height;
    int channel = imgData.ch;
    float scale = std::min(float(_model_width) / width, float(_model_height) / height);
    int new_width = std::round(_model_width / scale);
    int new_height = std::round(_model_height / scale);
    int w_compen = (new_width - width) / 2;
    int h_compen = (new_height - height) / 2;

    cv::Mat showFrame;
    cv::Size frameSize(width, height);
    if ((channel == 0) || (channel == 3)) {
        showFrame = cv::Mat(frameSize, CV_8UC3, imgData.data);
    } else {
        showFrame = cv::Mat(frameSize, CV_8UC4, imgData.data);
    }

    if (needFormat) {
        memset(showFrame.data, 0, width * height * channel);
    }

    DrawObject(imgData, showFrame, new_width, new_height, w_compen, h_compen);
}

// Draw bounding boxes for detected objects
void NpuDetectionImpl::DrawObject(image_share_t imgData, cv::Mat &showFrame,
                                   int new_width, int new_height,
                                   int w_compen, int h_compen) {
    (void)imgData;  // Unused but kept for API compatibility

    for (const auto& object : _objects) {
        int x1, y1, x2, y2;
#ifdef LETTER_BOX
        x1 = static_cast<int>(object.x_min * float(new_width) - w_compen);
        y1 = static_cast<int>(object.y_min * float(new_height) - h_compen);
        x2 = static_cast<int>(object.x_max * float(new_width) - w_compen);
        y2 = static_cast<int>(object.y_max * float(new_height) - h_compen);
#else
        x1 = static_cast<int>(object.x_min * showFrame.cols);
        y1 = static_cast<int>(object.y_min * showFrame.rows);
        x2 = static_cast<int>(object.x_max * showFrame.cols);
        y2 = static_cast<int>(object.y_max * showFrame.rows);
#endif
        cv::rectangle(showFrame, cv::Point(x1, y1),
                      cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 2);
        cv::putText(showFrame, object.name, cv::Point(x1, y1 - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255, 255), 1, 0);
    }

    _objects.clear();
    _objects.shrink_to_fit();
}

// Format label with confidence percentage
std::string NpuDetectionImpl::GetFinalLabel(float conf, std::string label) {
#ifdef SHOW_LABEL
    float rounded_probability = floorf(conf * 10000) / 100;
    std::ostringstream os_label;
    os_label << label;
    os_label << "(" << rounded_probability << "%)";
    return os_label.str();
#else
    (void)conf;  // Unused when SHOW_LABEL is not defined
    return label;
#endif
}

// Parse hardware NMS results from NPU output buffer
int NpuDetectionImpl::ParseHardwareNmsResults(image_share_t imgData) {
    (void)imgData;  // Hardware NMS doesn't need original image data

    if (!_nms_core || (_out_format != HAILO_FORMAT_TYPE_FLOAT32)) {
        return 0;
    }

    _objects.clear();
    _objects.shrink_to_fit();

    size_t num_dets = 0;

    /*
     * Hardware NMS output format:
     * Data is sorted by class. For each class:
     * - First element: number of boxes in this class
     * - Following elements: boxes (x_min, y_min, x_max, y_max, score) each
     * Each box is sizeof(common::hailo_bbox_t) = 5 * uint16_t or float32
     */

    for (size_t i = 0, class_id = 0; i < _output_buffer_float[0].size(); i++) {
        int obj_num = static_cast<int>(_output_buffer_float[0][i]);
        if (obj_num != 0) {
            for (int j = 0; j < obj_num; j++) {
                if (i + 5 >= _output_buffer_float[0].size()) break;

                float x_min = _output_buffer_float[0][i + 1];
                float y_min = _output_buffer_float[0][i + 2];
                float x_max = _output_buffer_float[0][i + 3];
                float y_max = _output_buffer_float[0][i + 4];
                float score = _output_buffer_float[0][i + 5];

                if (score >= _conf_threshold) {
                    object_roi_t cobj;
                    cobj.x_min = x_min;
                    cobj.y_min = y_min;
                    cobj.x_max = x_max;
                    cobj.y_max = y_max;
                    cobj.confidence = score;
                    cobj.category = static_cast<int>(class_id);
                    if (cobj.category + 1 < static_cast<int>(_labels.size())) {
                        cobj.name = GetFinalLabel(score, _labels[cobj.category + 1]);
                    } else {
                        cobj.name = GetFinalLabel(score, "unknown");
                    }
                    _objects.push_back(cobj);
                    num_dets++;
                }
                i += 5;
            }
        }
        class_id++;
        if (class_id > static_cast<size_t>(_nclasses)) {
            break;
        }
    }

    return static_cast<int>(num_dets);
}

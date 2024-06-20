#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sys/time.h>
#include "npu_yolov8_pose_impl.hpp"
#include "common/hailo_objects.hpp"
#include "yolov8pose_postprocess.hpp"

//#define TIME_TRACE_DEBUG

NpuYolov8PoseImpl::NpuYolov8PoseImpl(void)
{
    //
}

size_t NpuYolov8PoseImpl::post_processing_all(std::vector<std::vector<uint8_t>> &output_buffer_uint8,
                                    std::vector<hailo_vstream_info_t> &vstream_infos,
                                    std::pair<std::vector<KeyPt>,
                                    std::vector<PairPairs>> &keypoints_and_pairs)
{
    auto status = HAILO_SUCCESS;

    HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));

    for (uint j = 0; j < output_buffer_uint8.size(); j++) {
        roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(output_buffer_uint8[j].data()), vstream_infos[j]));
    }

    std::vector<int> strides = {_model_width/_feature_map_sizes[0], _model_width/_feature_map_sizes[1], _model_width/_feature_map_sizes[2]};
    std::vector<int> network_dims = {_model_width, _model_height};
    keypoints_and_pairs = filter(roi, _conf_threshold, strides, network_dims);
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
    for (auto &detection : detections) {
        if (detection->get_confidence() < _conf_threshold) {
              continue;
          }
          object_roi_t cobj;
          HailoBBox bbox = detection->get_bbox();
          cobj.y_min = bbox.ymin();
          cobj.x_min = bbox.xmin();
          cobj.y_max = bbox.ymax();
          cobj.x_max = bbox.xmax();
          cobj.confidence = detection->get_confidence();
          cobj.category = detection->get_class_id() + 1;
          cobj.name = GetFinalLabel(detection->get_confidence(), detection->get_label());
          _objects.push_back(cobj);
        #ifdef TIME_TRACE_DEBUG
          std::cout << "Detection: " << detection->get_label() << ", Confidence: " << detection->get_confidence() * 100.0 << "%" << std::endl;
        #endif
    }

    return _objects.size();
}

int NpuYolov8PoseImpl::Initialize(std::string configJsonFile, int streamId)
{
    InitConfig(configJsonFile, streamId);
    //get my extra json parameter from the _dom.
    if (_dom.HasMember("feature_map_size") && _dom["feature_map_size"].IsArray()) {
        const rapidjson::Value& arr = _dom["feature_map_size"];
        for (int i = 0; i < arr.Size(); ++i) {
        if (arr[i].IsInt())
            _feature_map_sizes.push_back(arr[i].GetInt());
        }
    }
    return InitNPU();
}

int NpuYolov8PoseImpl::Detect(image_share_t imgData, bool needPreProcess)
{
    MnpReturnCode ReadOutRet = MnpReturnCode::NO_DATA_AVAILABLE;
    size_t num_dets = 0;

    ReadOutRet = NpuPorcessing<uint8_t>(imgData, needPreProcess);
    if (ReadOutRet == MnpReturnCode::SUCCESS)
    {
#ifdef TIME_TRACE_DEBUG
      std::chrono::duration<double> total_time;
      std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
#endif

      num_dets = post_processing_all(_output_buffer_uint8, _vstream_infos, _pose_extra);
#ifdef TIME_TRACE_DEBUG
      std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
      total_time = t_end - t_start;
      std::cout << "-I- postprocessing run time: " << (double)total_time.count() << " sec" << std::endl;
#endif
    }
    return num_dets;
}

void NpuYolov8PoseImpl::DrawResult(image_share_t imgData, bool needFormat)
{
    //the image data is RGB
    int width = imgData.width;
    int height = imgData.height;
    int max_dim = ( width >= height ) ? width : height;
    int w_compen = ( width >= height ) ? 0 : ((height - width) / 2);
    int h_compen = ( width >= height ) ? ((width - height) / 2) : 0;
    cv::Mat showFrame(cv::Size(width, height), CV_8UC3, (void *)imgData.data);
    int x1, y1, x2, y2;

    if(needFormat) {
        memset(showFrame.data, 0, width * height * 3);
    }
    DrawObject(imgData, showFrame, max_dim, w_compen, h_compen);
    for (auto &keypoint : _pose_extra.first){
      #ifdef LETTER_BOX
        x1 = keypoint.xs * float(max_dim) - w_compen;
        y1 = keypoint.ys * float(max_dim) - h_compen;
      #else
        x1 = keypoint.xs * showFrame.cols;
        y1 = keypoint.ys * showFrame.rows;
      #endif
        circle(showFrame, cv::Point(x1, y1), 3, cv::Scalar(0, 255, 0, 255), 2);
    }
    for (PairPairs &p : _pose_extra.second){
      #ifdef LETTER_BOX
        x1 = p.pt1.first * float(max_dim) - w_compen;
        y1 = p.pt1.second * float(max_dim) - h_compen;
        x2 = p.pt2.first * float(max_dim) - w_compen;
        y2 = p.pt2.second * float(max_dim) - h_compen;
      #else
        x1 = p.pt1.first * showFrame.cols;
        y1 = p.pt1.second * showFrame.rows;
        x2 = p.pt2.first * showFrame.cols;
        y2 = p.pt2.second * showFrame.rows;
      #endif
        auto pt1 = cv::Point(x1, y1);
        auto pt2 = cv::Point(x2, y2);
        line(showFrame, pt1, pt2, cv::Scalar(0, 255, 0, 255), 2);
    }
}



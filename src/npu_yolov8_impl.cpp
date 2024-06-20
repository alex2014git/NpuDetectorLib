#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sys/time.h>
#include "npu_yolov8_impl.hpp"
#include "common/hailo_objects.hpp"
#include "yolov8_postprocess.hpp"

//#define TIME_TRACE_DEBUG

NpuYolov8Impl::NpuYolov8Impl(void)
{
    //
}

int NpuYolov8Impl::Initialize(std::string configJsonFile, int streamId)
{
    InitConfig(configJsonFile, streamId);
    if (_dom.HasMember("feature_map_size") && _dom["feature_map_size"].IsArray()) {
        const rapidjson::Value& arr = _dom["feature_map_size"];
        for (int i = 0; i < arr.Size(); ++i) {
        if (arr[i].IsInt())
            _feature_map_sizes.push_back(arr[i].GetInt());
        }
    }
    if (_dom.HasMember("yolo_nms_core") && _dom["yolo_nms_core"].IsBool()) {
        _nms_core = _dom["yolo_nms_core"].GetBool();
    }
    if (_dom.HasMember("out_sigmoid") && _dom["out_sigmoid"].IsBool()) {
        _out_sigmoid = _dom["out_sigmoid"].GetBool();
    }
    return InitNPU();
}

int NpuYolov8Impl::Detect(image_share_t imgData, bool needPreProcess)
{
    MnpReturnCode ReadOutRet = MnpReturnCode::NO_DATA_AVAILABLE;
    std::vector<float32_t> detectionsResult;

    ReadOutRet = NpuPorcessing<uint8_t>(imgData, needPreProcess);
    if (ReadOutRet == MnpReturnCode::SUCCESS)
    {
#ifdef TIME_TRACE_DEBUG
      std::chrono::duration<double> total_time;
      std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
#endif
      HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
      for (uint j = 0; j < _output_buffer_uint8.size(); j++) {
          roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(_output_buffer_uint8[j].data()), _vstream_infos[j]));
      }

      //get my extra json parameter from the _dom.
      std::vector<int> strides = {_model_width/_feature_map_sizes[0], _model_width/_feature_map_sizes[1], _model_width/_feature_map_sizes[2]};
      std::vector<int> network_dims = {_model_width, _model_height};
      filter_yolov8(roi, _nclasses, _labels, _out_sigmoid, _conf_threshold, strides, network_dims);
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
#ifdef TIME_TRACE_DEBUG
      std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
      total_time = t_end - t_start;
      std::cout << "-I- postprocessing run time: " << (double)total_time.count() << " sec" << std::endl;
#endif
    }
    return _objects.size();
}



#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <sys/time.h>
#include "npu_yolov8_seg_impl.hpp"
#include "common/hailo_objects.hpp"
#include "yolov8seg_postprocess.hpp"

//#define TIME_TRACE_DEBUG

NpuYolov8SegImpl::NpuYolov8SegImpl(void)
{
    //
}

int NpuYolov8SegImpl::Initialize(std::string configJsonFile, int streamId)
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
    return InitNPU();
}

int NpuYolov8SegImpl::Detect(image_share_t imgData, bool needPreProcess)
{
    MnpReturnCode ReadOutRet = MnpReturnCode::NO_DATA_AVAILABLE;

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
      _filtered_masks = filter_seg(roi, imgData.height, imgData.width, _nclasses, _labels, _conf_threshold, strides, network_dims);
      std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
      //cv::resize(frames[0], frames[0], cv::Size((int)org_width, (int)org_height), 1);
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

cv::Mat NpuYolov8SegImpl::crop_mask(cv::Mat& mask, int x_min, int y_min, int x_max, int y_max)
{
    int rows = mask.rows;
    int cols = mask.cols;

    // Ensure ROI coordinates are within the valid range
    int top_start = std::max(0, y_min);
    int bottom_end = std::min(rows, y_max);
    int left_start = std::max(0, x_min);
    int right_end = std::min(cols, x_max);

    // Create ROI rectangles
    cv::Rect top_roi(0, 0, cols, top_start);
    cv::Rect bottom_roi(0, bottom_end, cols, rows - bottom_end);
    cv::Rect left_roi(0, 0, left_start, rows);
    cv::Rect right_roi(right_end, 0, cols - right_end, rows);

    // Set values to zero in the specified ROIs
    mask(top_roi) = 0;
    mask(bottom_roi) = 0;
    mask(left_roi) = 0;
    mask(right_roi) = 0;

    return mask;
}

void NpuYolov8SegImpl::DrawResult(image_share_t imgData, bool needFormat)
{
    int x1, y1, x2, y2;
    std::map<int, cv::Vec3b> COLORS = {
        {0, cv::Vec3b(244, 67, 54)},
        {1, cv::Vec3b(233, 30, 99)},
        {2, cv::Vec3b(156, 39, 176)},
        {3, cv::Vec3b(103, 58, 183)},
        {4, cv::Vec3b(63, 81, 181)},
        {5, cv::Vec3b(33, 150, 243)},
        {6, cv::Vec3b(3, 169, 244)},
        {7, cv::Vec3b(0, 188, 212)},
        {8, cv::Vec3b(0, 150, 136)},
        {9, cv::Vec3b(76, 175, 80)},
        {10, cv::Vec3b(139, 195, 74)},
        {11, cv::Vec3b(205, 220, 57)},
        {12, cv::Vec3b(255, 235, 59)},
        {13, cv::Vec3b(255, 193, 7)},
        {14, cv::Vec3b(255, 152, 0)},
        {15, cv::Vec3b(255, 87, 34)},
        {16, cv::Vec3b(121, 85, 72)},
        {17, cv::Vec3b(158, 158, 158)},
        {18, cv::Vec3b(96, 125, 139)}
    };
    //the image data is RGB
    int width = imgData.width;
    int height = imgData.height;
    int max_dim = ( width >= height ) ? width : height;
    int w_compen = ( width >= height ) ? 0 : ((height - width) / 2);
    int h_compen = ( width >= height ) ? ((width - height) / 2) : 0;
    cv::Mat showFrame(cv::Size(width, height), CV_8UC3, (void *)imgData.data);

    if(needFormat) {
        memset(showFrame.data, 0, width * height * 3);
    }

    //DrawObject(imgData, showFrame, max_dim, w_compen, h_compen);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random_index(0, COLORS.size() - 1);
    cv::Mat mask_full(height, width, CV_32FC1);
    for (size_t i = 0; i < _objects.size() && i < _filtered_masks.size(); ++i) {
        const auto& object = _objects[i];
        const auto& mask = _filtered_masks[i];
        // Draw the objects
      #ifdef LETTER_BOX
        x1 = object.x_min * float(max_dim) - w_compen;
        y1 = object.y_min * float(max_dim) - h_compen;
        x2 = object.x_max * float(max_dim) - w_compen;
        y2 = object.y_max * float(max_dim) - h_compen;
      #else
        x1 = object.x_min * width;
        y1 = object.y_min * height;
        x2 = object.x_max * width;
        y2 = object.y_max * height;
      #endif
        rectangle(showFrame, cv::Point(x1, y1),
                    cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 2);
        putText(showFrame, object.name, cv::Point(x1, y1 - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0, 0, 255, 255), 1 , 0);

        //draw the mask.
        cv::Mat overlay = cv::Mat(height, width, CV_8UC3, cv::Scalar(0));
        cv::resize(mask, mask_full, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        crop_mask(mask_full, x1,y1,x2,y2);
      #ifdef TIME_TRACE_DEBUG
        printf("mask %dx%d\n", mask.cols, mask.rows);
        printf("overlay %dx%d\n", overlay.cols, overlay.rows);
      #endif
        auto pixel_color = COLORS[random_index(gen)];
        for (int r = 0; r < mask_full.rows; r++) {
            for (int c = 0; c < mask_full.cols ; c++) {
                if (mask_full.at<float>(r, c) > 0.55) {
                    overlay.at<cv::Vec3b>(r,c) = pixel_color;
                }
            }
        }
        //std::stringstream ss;
        //ss << "mask_" << i << ".bmp";
        //std::string filename = ss.str();
        //cv::imwrite(filename, overlay);
        cv::addWeighted(showFrame, 1, overlay, 0.5, 0.0, showFrame);
        overlay.release();
        mask_full.release();
    }
    _objects.clear();
    _objects.shrink_to_fit();
}



#ifndef _NPU_API_H
#define _NPU_API_H
#include <vector>
#include <string>
#include <functional>
#include <memory>

enum algorithm {
    // Generic base implementation for simple models (no NMS)
    ALG_BASE,              // Generic/simple models: LPR, Classification

    // Detection models using HailoRT built-in NMS (yolo_nms_core: true in JSON)
    ALG_YOLO_NMS,          // YOLO with HailoRT hardware NMS

    // YOLO models with software post-processing (yolo_nms_core: false in JSON)
    ALG_YOLO_V5,           // YOLOv5-v7 detection with self NMS
    ALG_YOLO_V8,           // YOLOv8 detection with self NMS

    // Specialized YOLO variants (all use yolo_nms_core: false)
    ALG_POSE,              // YOLOv8-pose (keypoint detection)
    ALG_YOLO_V8_SEG,       // YOLOv8 instance segmentation

    // Non-YOLO models (use ALG_BASE implementation)
    ALG_LPR,               // License Plate Recognition
    ALG_CLASSIFICATION     // Image classification
};

typedef struct _image_share
{
    void *data;       // image bitmap data point
    int width;        // width, in pixel unit
    int height;       // height, in pixel unit
    int ch;           // image channel. for now is 3 for RGB888 or 4 for RGBA8888
} image_share_t;

// Define the _object_roi structure
typedef struct _object_roi
{
    float y_min;
    float x_min;
    float y_max;
    float x_max;
    float confidence;
    int category;
    std::string name;
} object_roi_t;

class Npu {

public:
    virtual int Initialize(std::string configJsonFile, int streamId) = 0;

    /// @brief 获取算法版本号
    /// @return 算法版本号
    virtual std::string GetVersion() = 0;

    virtual int Detect(image_share_t imgData, bool needPreProcess) = 0;

    virtual void DrawResult(image_share_t imgData, bool needFormat) = 0;

    /// @brief 释放npu
    virtual void Release() = 0;

    /// @brief Get model input width
    virtual int GetModelWidth() const = 0;

    /// @brief Get model input height
    virtual int GetModelHeight() const = 0;

    static algorithm str2AlgEnum(const char* enumStr) {
        static const std::unordered_map<std::string, algorithm> strToEnumMap = {
            {"base", ALG_BASE},              // Generic/simple models (LPR, classification)
            {"yolo_nms", ALG_YOLO_NMS},      // YOLO with hardware NMS
            {"yolov5", ALG_YOLO_V5},         // YOLOv5 with software NMS
            {"yolov8", ALG_YOLO_V8},         // YOLOv8 with software NMS
            {"yolov8_pose", ALG_POSE},       // YOLOv8 pose estimation
            {"yolov8_seg", ALG_YOLO_V8_SEG}, // YOLOv8 segmentation
            {"lpr", ALG_LPR},                // License plate recognition
            {"classification", ALG_CLASSIFICATION}  // Image classification
        };
        auto it = strToEnumMap.find(enumStr);
        if (it != strToEnumMap.end()) {
            return it->second;
        }
        return ALG_BASE;
    }
};


#endif // #ifndef _NPU_API_H


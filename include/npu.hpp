#ifndef _NPU_API_H
#define _NPU_API_H
#include <vector>
#include <string>
#include <functional>
#include <memory>

enum algorithm {
    ALG_BASE,         //hailo PP detection.
    ALG_YOLO_V5,      //yolov5-yolov7 self post processing.
    ALG_YOLO_V8,       //yolov8 self post processing.
    ALG_POSE,         //yolov8-pose.
    ALG_YOLO_V8_SEG,   //yolov8-seg
    ALG_SCRFD         //SCRFD10G.
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

    static algorithm str2AlgEnum(const char* enumStr) {
        static const std::unordered_map<std::string, algorithm> strToEnumMap = {
            {"base", ALG_BASE},
            {"yolov5", ALG_YOLO_V5},
            {"yolov8", ALG_YOLO_V8},
            {"yolov8_pose", ALG_POSE},
            {"yolov8_seg", ALG_YOLO_V8_SEG}
        };
        auto it = strToEnumMap.find(enumStr);
        if (it != strToEnumMap.end()) {
            return it->second;
        }
        return ALG_BASE;
    }
};


#endif // #ifndef _NPU_API_H


#ifndef _YOLO_NMS_DECODER_H_
#define _YOLO_NMS_DECODER_H_

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <type_traits>
#include <algorithm>

struct DetectionObject {
    float ymin, xmin, ymax, xmax, confidence;
    int class_id;

    DetectionObject(float ymin, float xmin, float ymax, float xmax, float confidence, int class_id):
        ymin(ymin), xmin(xmin), ymax(ymax), xmax(xmax), confidence(confidence), class_id(class_id)
        {}
    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

struct DecodedBox {
    float x;
    float y;
    float w;
    float h;
};

struct QunatizationInfo {
    float qp_zp;
    float qp_scale;
};

struct YoloOutputInfo {

    int FeatureSizeRow;
    int FeatureSizeCol; 
    std::vector<int> anchors;    
    QunatizationInfo qInfo;
};


template <class T>
class YoloNmsDecoder {

    const float IOU_THRESHOLD = 0.45f;
    const int MAX_BOXES = 100;
    const int CONF_CHANNEL_OFFSET = 4;
    const int CLASS_CHANNEL_OFFSET = 5;
    int ImageSizeWidth = 640;
    int ImageSizeHeight = 640;
    int YoloAnchorNum = 0;
    int TotalOutputAdded = 0;
    int TotalClass = 80;
    int TotalFeatureMapChannel = TotalClass+5;
    float ConfidenceThreshold = 0.3f;
    std::vector<YoloOutputInfo> OutputInfoList;


    float iou_calc(const DetectionObject &box_1, const DetectionObject &box_2) {
        const float width_of_overlap_area = std::min(box_1.xmax, box_2.xmax) - std::max(box_1.xmin, box_2.xmin);
        const float height_of_overlap_area = std::min(box_1.ymax, box_2.ymax) - std::max(box_1.ymin, box_2.ymin);
        const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
        const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
        const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;
        const float box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
        const float box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
        return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
    }

    void extract_boxes( std::vector<T> &fm, QunatizationInfo &q_info, int feature_map_size_row, int feature_map_size_col, 
                        std::vector<int> &Anchors, std::vector<DetectionObject> &objects, float& thr) {
        float  confidence, xmin, ymin, xmax, ymax, conf_max = 0.0f;
        int add = 0, anchor = 0, chosen_row = 0, chosen_col = 0, chosen_cls = -1;
        float cls_prob, prob_max;

        //PreCalculation for optimization
        int lcTotalFeatureMapChannel = this->TotalFeatureMapChannel;
        int lcYoloAnchorNum = this->YoloAnchorNum;
        int TotalFeatMapChXAnchorNum = lcTotalFeatureMapChannel * lcYoloAnchorNum;
        int AggregateCalc = TotalFeatMapChXAnchorNum * feature_map_size_col;
        //int isRawData = std::is_same<uint8_t, decltype(fm[0])>::value; 
        int isRawData = (typeid(uint8_t).name() == typeid(fm[0]).name());

        for (int row = 0; row < feature_map_size_row; ++row) {
            for (int col = 0; col < feature_map_size_col; ++col) {
                prob_max = 0;
                for (int a = 0; a < lcYoloAnchorNum; ++a) {
                    add =   AggregateCalc * row + 
                            TotalFeatMapChXAnchorNum * col + 
                            lcTotalFeatureMapChannel * a + CONF_CHANNEL_OFFSET;

                    confidence = (isRawData) ? this->output_data_trasnform(this->dequantize(fm[add], q_info)) : this->output_data_trasnform(fm[add]);
                    if (confidence < thr)
                        continue;
                    for (int c = CLASS_CHANNEL_OFFSET; c < lcTotalFeatureMapChannel; ++c) {
                        add =   AggregateCalc * row + 
                                TotalFeatMapChXAnchorNum * col + 
                                lcTotalFeatureMapChannel * a + c;

                        // final confidence: box confidence * class probability
                        cls_prob = fm[add];
                        if (cls_prob > prob_max) {
                            conf_max = (isRawData) ? this->output_data_trasnform(this->dequantize(cls_prob, q_info)) * confidence : this->output_data_trasnform(cls_prob) * confidence;
                            
                            chosen_cls = c - CLASS_CHANNEL_OFFSET + 1;
                            prob_max = cls_prob;
                            anchor = a;
                            chosen_row = row;
                            chosen_col = col;
                        }
                    }
                }
                if (conf_max >= thr) {

                    add =   AggregateCalc * chosen_row + 
                            TotalFeatMapChXAnchorNum * chosen_col + 
                            lcTotalFeatureMapChannel * anchor;

                    DecodedBox DBox = this->box_decode(fm, add, Anchors, anchor, q_info, feature_map_size_row, feature_map_size_col, chosen_col, chosen_row);

                    // x,y,h,w to xmin,ymin,xmax,ymax
                    xmin = (DBox.x - (DBox.w / 2.0f));
                    ymin = (DBox.y - (DBox.h / 2.0f));
                    xmax = (DBox.x + (DBox.w / 2.0f));
                    ymax = (DBox.y + (DBox.h / 2.0f));
                    objects.push_back(DetectionObject(ymin, xmin, ymax, xmax, conf_max, chosen_cls));
                }
            }
        }
    }    

public:

    virtual float      output_data_trasnform(float data) {
        return data;
    }

    virtual DecodedBox box_decode(  std::vector<T> &fm, int fm_offset, 
                                    std::vector<int> &Anchors, int anchor_index, QunatizationInfo &q_info,
                                    int feature_map_size_row, int feature_map_size_col, int chosen_col, int chosen_row) = 0;

    float dequantize(float input, QunatizationInfo &q_info) {
        return (input - q_info.qp_zp) * q_info.qp_scale;
    }

    int  getImageWidth() {
        return this->ImageSizeWidth;
    }

    int  getImageHeight() {
        return this->ImageSizeHeight;
    }

    void YoloConfig (int ImageWidth, int ImageHeight, int TotalClass, float ConfidenceThreshold, int totalAnchorNum = 3) {
        this->ImageSizeWidth = ImageWidth;
        this->ImageSizeHeight = ImageHeight;
        this->TotalClass = TotalClass;     
        this->TotalFeatureMapChannel = this->TotalClass + 5;
        this->ConfidenceThreshold = ConfidenceThreshold;
        this->YoloAnchorNum = totalAnchorNum;
    }

    int YoloAddOutput (int FeatureSizeRow, int FeatureSizeCol, std::vector<int> anchors, QunatizationInfo* pQInfo = nullptr) {

        YoloOutputInfo OutputInfo;
        OutputInfo.FeatureSizeRow = FeatureSizeRow;
        OutputInfo.FeatureSizeCol = FeatureSizeCol;
        OutputInfo.anchors = anchors;

        if (pQInfo == nullptr) {
            OutputInfo.qInfo.qp_scale = 1.0f;
            OutputInfo.qInfo.qp_zp = 0.0f;
            if ((typeid(uint8_t).name() == typeid(T).name()))
                printf("WARNING: Raw data declaration detected while quantization info is not proivded (for dequantization from raw int to float)");
        }
        else {
            OutputInfo.qInfo.qp_scale = pQInfo->qp_scale;
            OutputInfo.qInfo.qp_zp = pQInfo->qp_zp;
        }

        this->OutputInfoList.push_back(OutputInfo);
        this->TotalOutputAdded++;

        return this->TotalOutputAdded;
    }

    std::vector<float> decode(std::vector<T> &Output1, std::vector<T> &Output2, std::vector<T> &Output3) {
        size_t num_boxes = 0;
        std::vector<DetectionObject> objects;
        objects.reserve(MAX_BOXES);
        
        if (Output1.size() > 0) {
            this->extract_boxes(Output1, this->OutputInfoList[0].qInfo, this->OutputInfoList[0].FeatureSizeRow, this->OutputInfoList[0].FeatureSizeCol, 
                                this->OutputInfoList[0].anchors, objects, this->ConfidenceThreshold);
        }

        if (Output2.size() > 0) {
            this->extract_boxes(Output2, this->OutputInfoList[1].qInfo, this->OutputInfoList[1].FeatureSizeRow, this->OutputInfoList[1].FeatureSizeCol, 
                                this->OutputInfoList[1].anchors, objects, this->ConfidenceThreshold);
        }

        if (Output3.size() > 0) {
            this->extract_boxes(Output3, this->OutputInfoList[2].qInfo, this->OutputInfoList[2].FeatureSizeRow, this->OutputInfoList[2].FeatureSizeCol, 
                                this->OutputInfoList[2].anchors, objects, this->ConfidenceThreshold);
        }

        // filter by overlapping boxes
        num_boxes = objects.size();
        if (objects.size() > 0) {
            std::sort(objects.begin(), objects.end());
            for (unsigned int i = 0; i < objects.size(); ++i) {
                if (objects[i].confidence <= this->ConfidenceThreshold)
                    continue;
                for (unsigned int j = i + 1; j < objects.size(); ++j) {
                    if (objects[i].class_id == objects[j].class_id && objects[j].confidence >= this->ConfidenceThreshold) {
                        if (iou_calc(objects[i], objects[j]) >= IOU_THRESHOLD) {
                            objects[j].confidence = 0;
                            num_boxes -= 1;
                        }
                    }
                }
            }
        }

        // Copy the results
        std::vector<float> results;
        if (num_boxes > 0) {
            int box_ptr = 0;

            results.resize(num_boxes * 6);
            for (const auto &obj: objects) {
                if (obj.confidence >= this->ConfidenceThreshold) {
                    results[box_ptr*6 + 0] = obj.ymin;
                    results[box_ptr*6 + 1] = obj.xmin;
                    results[box_ptr*6 + 2] = obj.ymax;
                    results[box_ptr*6 + 3] = obj.xmax;
                    results[box_ptr*6 + 4] = (float)obj.class_id;
                    results[box_ptr*6 + 5] = obj.confidence;                
                    box_ptr += 1;                    
                }
            }
            return results;
        } else {        
            results.resize(0);
            return results;
        }
    }
};


template <class T>
class Yolov5NmsDecoder : public YoloNmsDecoder <T> {

    bool        bActivationIsSigmoid = false;

    float sigmoid(float x) {
        // returns the value of the sigmoid function f(x) = 1/(1 + e^-x)
        return 1.0f / (1.0f + expf(-x));
    }

    //Override parent
    float output_data_trasnform(float data) {

        if (this->bActivationIsSigmoid)
            return data;
        
        return sigmoid(data);  

    }
    
public:

    Yolov5NmsDecoder(bool bSigmoidActivation) {
        this->bActivationIsSigmoid = bSigmoidActivation;
    }
    
    void set_sigmoid(bool bSigmoidActivation) {
        this->bActivationIsSigmoid = bSigmoidActivation;
    }

    //Override parent
    DecodedBox box_decode(  std::vector<T> &fm, int fm_offset, 
                            std::vector<int> &Anchors, int anchor_index, QunatizationInfo &q_info,
                            int feature_map_size_row, int feature_map_size_col, int chosen_col, int chosen_row) {
        
        int isRawData = (typeid(uint8_t).name() == typeid(fm[fm_offset]).name());
        DecodedBox DBox;
        // box centers
        float x = (isRawData) ? this->dequantize(fm[fm_offset], q_info) : fm[fm_offset];
        DBox.x = (this->output_data_trasnform(x) * 2.0f - 0.5f + chosen_col) / feature_map_size_col;

        float y = (isRawData) ? this->dequantize(fm[fm_offset + 1], q_info) : fm[fm_offset + 1];
        DBox.y = (this->output_data_trasnform(y) * 2.0f - 0.5f +  chosen_row) / feature_map_size_row;

        // box scales
        float w = (isRawData) ? this->dequantize(fm[fm_offset + 2], q_info) : fm[fm_offset + 2];
        w = this->output_data_trasnform(w);
        DBox.w = pow(2.0f * w, 2.0f) * Anchors[anchor_index * 2] / this->getImageWidth();

        float h = (isRawData) ? this->dequantize(fm[fm_offset + 3], q_info) : fm[fm_offset + 3];
        h = this->output_data_trasnform(h);
        DBox.h = pow(2.0f * h, 2.0f) * Anchors[anchor_index * 2 + 1] / this->getImageHeight();

        return DBox;
    }

};


#endif

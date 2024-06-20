/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
// General includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

// Hailo includes
#include "yolov8_common.hpp"
#include "yolov8_postprocess.hpp"

using namespace xt::placeholders;

#define IOU_THRESHOLD 0.7

/**
 * @brief Split the raw output tensors into boxes and scores
 * 
 * @param tensors  -  std::vector<HailoTensorPtr>
 *        The network output tensors
 * 
 * @param num_classes  -  int
 *        Number of classes
 * 
 * @param regression_length  -  int
 *        Regression length of anchors
 *  
 * @return std::pair<std::vector<xt::xarray<float>>, xt::xarray<float>> 
 *         The separated score and box vectors
 */
std::pair<std::vector<HailoTensorPtr>, xt::xarray<float>> get_boxes_and_scores(std::vector<HailoTensorPtr> &tensors,
                                                                                  int num_classes,
                                                                                  int regression_length)
{
    std::vector<HailoTensorPtr> outputs_boxes(tensors.size() / 2);
    
    // Prepare the scores xarray at the size we will fill in in-place
    int total_scores = 0;
    for (uint i = 0; i < tensors.size(); i = i + 2) { 
        total_scores += tensors[i+1]->width() * tensors[i+1]->height(); 
    }

    std::vector<size_t> shape = { (long unsigned int)total_scores, (long unsigned int)num_classes};
    
    xt::xarray<float> scores(shape);
    int view_index = 0;

    for (uint i = 0; i < tensors.size(); i = i + 2)
    {
        // Bounding boxes extraction will be done later on only on the boxes that surpass the score threshold
        outputs_boxes[i / 2] = tensors[i];

        // Extract and dequantize the scores outputs
        auto dequantized_output_s = common::dequantize(common::get_xtensor(tensors[i+1]), tensors[i+1]->vstream_info().quant_info.qp_scale, tensors[i+1]->vstream_info().quant_info.qp_zp);
        int num_proposals_scores = dequantized_output_s.shape(0)*dequantized_output_s.shape(1);

        // From the layer extract the scores
        auto output_scores = xt::view(dequantized_output_s, xt::all(), xt::all(), xt::all());
        xt::view(scores, xt::range(view_index, view_index + num_proposals_scores), xt::all()) = xt::reshape_view(output_scores, {num_proposals_scores, num_classes});
        view_index += num_proposals_scores;
    }

    return std::pair<std::vector<HailoTensorPtr>, xt::xarray<float>>( outputs_boxes, scores );
}

std::vector<HailoDetection> decode_boxes(std::vector<HailoTensorPtr> raw_boxes_outputs,
                                            xt::xarray<float> scores,
                                            std::vector<int> network_dims,
                                            std::vector<int> strides,
                                            int regression_length,
                                            std::vector<std::string> labels, float conf_thr)
{
    int strided_width, strided_height, class_index;
    std::vector<HailoDetection> detections;
    int instance_index = 0;
    float confidence = 0.0;
    std::string label;

    auto centers = get_centers(std::ref(strides), std::ref(network_dims), raw_boxes_outputs.size(), strided_width, strided_height);

    // Box distribution to distance
    auto regression_distance =  xt::reshape_view(xt::arange(0, regression_length + 1), {1, 1, regression_length + 1});

    for (uint i = 0; i < raw_boxes_outputs.size(); i++)
    {
        auto output_b = common::get_xtensor(raw_boxes_outputs[i]);
        int num_proposals = output_b.shape(0) * output_b.shape(1);
        auto output_boxes = xt::view(output_b, xt::all(), xt::all(), xt::all());
        xt::xarray<uint8_t> quantized_boxes = xt::reshape_view(output_boxes, {num_proposals, 4, regression_length + 1});

        float32_t qp_scale = raw_boxes_outputs[i]->vstream_info().quant_info.qp_scale;
        float32_t qp_zp = raw_boxes_outputs[i]->vstream_info().quant_info.qp_zp;

        auto shape = {quantized_boxes.shape(1), quantized_boxes.shape(2)};

        for (uint j = 0; j < num_proposals; j++)
        {
            class_index = xt::argmax(xt::row(scores, instance_index))(0);
            confidence = scores(instance_index, class_index);
            instance_index++;
            if (confidence < conf_thr)
                continue;

            xt::xarray<float> box(shape);

            dequantize_box_values(box, j, quantized_boxes, box.shape(0), box.shape(1), qp_scale, qp_zp);

            common::softmax_2D(box.data(), box.shape(0), box.shape(1));
            
            auto box_distance = box * regression_distance;
            xt::xarray<float> reduced_distances = xt::sum(box_distance, {2});
            auto strided_distances = reduced_distances * strides[i];

            // Decode box
            auto distance_view1 = xt::view(strided_distances, xt::all(), xt::range(_, 2)) * -1;
            auto distance_view2 = xt::view(strided_distances, xt::all(), xt::range(2, _));
            auto distance_view = xt::concatenate(xt::xtuple(distance_view1, distance_view2), 1);
            auto decoded_box = centers[i] + distance_view;

            HailoBBox bbox(decoded_box(j, 0) / network_dims[0],
                           decoded_box(j, 1) / network_dims[1],
                           (decoded_box(j, 2) - decoded_box(j, 0)) / network_dims[0],
                           (decoded_box(j, 3) - decoded_box(j, 1)) / network_dims[1]);

            label = labels[class_index + 1];
            HailoDetection detected_instance(bbox, class_index, label, confidence);
            detections.push_back(detected_instance);
        }
    }
    return detections;
}

std::vector<HailoDetection> yolov8_postprocess(std::vector<HailoTensorPtr> &tensors,
                                                   std::vector<int> network_dims,
                                                   std::vector<int> strides,
                                                   int regression_length,
                                                   int num_classes,
                                                   std::vector<std::string> labels, 
                                                   bool out_sigmoid, float conf_thr)
{
    std::vector<HailoDetection> detections;
    if (tensors.size() == 0)
    {
        return detections;
    }

    auto boxes_and_scores = get_boxes_and_scores(tensors, num_classes, regression_length);
    std::vector<HailoTensorPtr> raw_boxes = boxes_and_scores.first;
    xt::xarray<float> scores = boxes_and_scores.second;

    // Calculate the sigmoid of the scores
    if (!out_sigmoid)
        common::sigmoid(scores.data(), scores.size());

    // Decode the boxes
    detections = decode_boxes(raw_boxes, scores, network_dims, strides, regression_length, labels, conf_thr);

    // Filter with NMS
    common::nms(detections, IOU_THRESHOLD, true);

    return detections;
}

/**
 * @brief yolov8 postprocess
 *        Provides network specific paramters
 * 
 * @param roi  -  HailoROIPtr
 *        The roi that contains the ouput tensors
 */
void yolov8(HailoROIPtr roi, int class_nums, std::vector<std::string> labels,
                           bool out_sigmoid, float conf_thr,
                           std::vector<int> strides,
                           std::vector<int> network_dims)
{
    // anchor params
    int regression_length = 15;

    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    std::vector<HailoDetection> detections = yolov8_postprocess(tensors, network_dims, strides, regression_length, class_nums, labels, out_sigmoid, conf_thr);
    hailo_common::add_detections(roi, detections);
}

//******************************************************************
//  DEFAULT FILTER
//******************************************************************
void filter_yolov8(HailoROIPtr roi, int class_nums,
                           std::vector<std::string> labels,
                           bool out_sigmoid, float conf_thr,
                           std::vector<int> strides,
                           std::vector<int> network_dims)
{
    yolov8(roi, class_nums, labels, out_sigmoid, conf_thr, strides, network_dims);
}

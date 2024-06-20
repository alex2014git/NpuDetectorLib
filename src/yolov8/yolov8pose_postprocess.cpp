/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
// General includes
#include <iostream>
#include <vector>

// Hailo includes
#include "yolov8_common.hpp"
#include "yolov8pose_postprocess.hpp"

using namespace xt::placeholders;

#define IOU_THRESHOLD 0.7
#define NUM_CLASSES 1

std::vector<std::pair<int, int>> JOINT_PAIRS = {
    {0, 1}, {1, 3}, {0, 2}, {2, 4},
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 11}, {6, 12}, {11, 12},
    {11, 13}, {12, 14}, {13, 15}, {14, 16}
};


std::pair<std::vector<KeyPt>, std::vector<PairPairs>> filter_keypoints(std::vector<Decodings> filtered_decodings,
                                                            std::vector<int> network_dims, float joint_threshold=0.5) {
    std::vector<KeyPt> filtered_keypoints;
    std::vector<PairPairs> filtered_pairs;

    for (auto& dec : filtered_decodings){
        auto keypoint_coordinates_and_score = dec.keypoints;
        auto coordinates = keypoint_coordinates_and_score.first;
        auto score = keypoint_coordinates_and_score.second;
        
        // Filter keypoints
        for (int i = 0; i < score.shape(0); i++){
            if (score(i,0) > joint_threshold) {
                filtered_keypoints.push_back(KeyPt({coordinates(i, 0) / network_dims[0], coordinates(i, 1) / network_dims[1], score(i,0)}));
            }
        }

        // Filter joints pair
        for (const auto& pair : JOINT_PAIRS) {
            if (score(pair.first,0) >= joint_threshold && score(pair.second, 0) >= joint_threshold){
                PairPairs pr = PairPairs({
                                std::make_pair(coordinates(pair.first,0) / network_dims[0], coordinates(pair.first,1) / network_dims[1]),
                                std::make_pair(coordinates(pair.second,0) / network_dims[0], coordinates(pair.second,1) / network_dims[1]),
                                score(pair.first, 0), 
                                score(pair.second, 0)
                                });
                filtered_pairs.push_back(pr);
            }
        }
    }

    return std::make_pair(filtered_keypoints, filtered_pairs);
}

std::vector<Decodings> nms(std::vector<Decodings> &decodings, const float iou_thr, bool should_nms_cross_classes = false) {

    std::vector<Decodings> decodings_after_nms;

    for (uint index = 0; index < decodings.size(); index++)
    {
        if (decodings[index].detection_box.get_confidence() != 0.0f)
        {
            for (uint jindex = index + 1; jindex < decodings.size(); jindex++)
            {
                if ((should_nms_cross_classes || (decodings[index].detection_box.get_class_id() == decodings[jindex].detection_box.get_class_id())) &&
                    decodings[jindex].detection_box.get_confidence() != 0.0f)
                {
                    // For each detection, calculate the IOU against each following detection.
                    float iou = iou_calc(decodings[index].detection_box.get_bbox(), decodings[jindex].detection_box.get_bbox());
                    // If the IOU is above threshold, then we have two similar detections,
                    // and want to delete the one.
                    if (iou >= iou_thr)
                    {
                        // The detections are arranged in highest score order,
                        // so we want to erase the latter detection.
                        decodings[jindex].detection_box.set_confidence(0.0f);
                    }
                }
            }
        }
    }
    for (uint index = 0; index < decodings.size(); index++)
    {
        if (decodings[index].detection_box.get_confidence() != 0.0f)
        {
            decodings_after_nms.push_back(Decodings{decodings[index].detection_box, decodings[index].keypoints, decodings[index].joint_pairs});
        }
    }
    return decodings_after_nms;
}


std::vector<Decodings> decode_boxes_and_keypoints(std::vector<HailoTensorPtr> raw_boxes_outputs,
                                                                    xt::xarray<float> scores,
                                                                    std::vector<HailoTensorPtr> raw_keypoints,
                                                                    std::vector<int> network_dims,
                                                                    std::vector<int> strides,
                                                                    int regression_length, float conf_thr) {
    int strided_width, strided_height, class_index;
    std::vector<Decodings> decodings;
    std::vector<PairPairs> joint_pairs;
    int instance_index = 0;
    float confidence = 0.0;
    std::string label;

    auto centers = get_centers(std::ref(strides), std::ref(network_dims), raw_boxes_outputs.size(), strided_width, strided_height);

    // Box distribution to distance
    auto regression_distance =  xt::reshape_view(xt::arange(0, regression_length + 1), {1, 1, regression_length + 1});

    for (uint i = 0; i < raw_boxes_outputs.size(); i++)
    {
        // Boxes setup
        float32_t qp_scale = raw_boxes_outputs[i]->vstream_info().quant_info.qp_scale;
        float32_t qp_zp = raw_boxes_outputs[i]->vstream_info().quant_info.qp_zp;

        auto output_b = common::get_xtensor(raw_boxes_outputs[i]);
        int num_proposals = output_b.shape(0) * output_b.shape(1);
        auto output_boxes = xt::view(output_b, xt::all(), xt::all(), xt::all());
        xt::xarray<uint8_t> quantized_boxes = xt::reshape_view(output_boxes, {num_proposals, 4, regression_length + 1});

        auto shape = {quantized_boxes.shape(1), quantized_boxes.shape(2)};

        // Keypoints setup
        float32_t qp_scale_kpts = raw_keypoints[i]->vstream_info().quant_info.qp_scale;
        float32_t qp_zp_kpts = raw_keypoints[i]->vstream_info().quant_info.qp_zp;

        auto output_keypoints = common::get_xtensor(raw_keypoints[i]);        
        int num_proposals_keypoints = output_keypoints.shape(0) * output_keypoints.shape(1);
        auto output_keypoints_quantized = xt::view(output_keypoints, xt::all(), xt::all(), xt::all());
        xt::xarray<uint8_t> quantized_keypoints = xt::reshape_view(output_keypoints_quantized, {num_proposals_keypoints, 17, 3});

        auto keypoints_shape = {quantized_keypoints.shape(1), quantized_keypoints.shape(2)};

        // Bbox decoding
        for (uint j = 0; j < num_proposals; j++) {
            confidence =    xt::row(scores, instance_index)(0);
            instance_index++;
            if (confidence < conf_thr)
                continue;

            xt::xarray<float> box(shape);
            xt::xarray<float> kpts_corrdinates_and_scores(keypoints_shape);
    
            dequantize_box_values(box, j, quantized_boxes, 
                                    box.shape(0), box.shape(1), 
                                    qp_scale, qp_zp);
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

            label = "person";
            HailoDetection detected_instance(bbox, class_index, label, confidence);

            // Decode keypoints
            dequantize_box_values(kpts_corrdinates_and_scores, j, quantized_keypoints, 
                                    kpts_corrdinates_and_scores.shape(0), kpts_corrdinates_and_scores.shape(1), 
                                    qp_scale_kpts, qp_zp_kpts);

            auto kpts_corrdinates = xt::view(kpts_corrdinates_and_scores, xt::all(), xt::range(0, 2));
            auto keypoints_scores = xt::view(kpts_corrdinates_and_scores, xt::all(), xt::range(2, xt::placeholders::_));

            kpts_corrdinates *= 2;

            auto center = xt::view(centers[i], xt::all(), xt::range(0, 2));
            auto center_values = xt::xarray<float>{(float)center(j,0), (float)center(j,1)};

            kpts_corrdinates = strides[i] * (kpts_corrdinates - 0.5) + center_values;

            // Apply sigmoid to keypoints scores
            auto sigmoided_scores = 1 / (1 + xt::exp(-keypoints_scores));

            auto keypoint = std::make_pair(kpts_corrdinates, sigmoided_scores);

            decodings.push_back(Decodings{detected_instance, keypoint, joint_pairs});
        }
    }

    return decodings;
}


Triple get_boxes_scores_keypoints(std::vector<HailoTensorPtr> &tensors, int num_classes, int regression_length){
    std::vector<HailoTensorPtr> outputs_boxes(tensors.size() / 3);
    std::vector<HailoTensorPtr> outputs_keypoints(tensors.size() / 3);
    
    // Prepare the scores xarray at the size we will fill in in-place
    int total_scores = 0;
    for (uint i = 0; i < tensors.size(); i = i + 3) { 
        total_scores += tensors[i+1]->width() * tensors[i+1]->height(); 
    }

    std::vector<size_t> scores_shape = { (long unsigned int)total_scores, (long unsigned int)num_classes};
    
    xt::xarray<float> scores(scores_shape);

    int view_index_scores = 0;

    for (uint i = 0; i < tensors.size(); i = i + 3)
    {
        // Bounding boxes extraction will be done later on only on the boxes that surpass the score threshold
        outputs_boxes[i / 3] = tensors[i];

        // Extract and dequantize the scores outputs
        auto dequantized_output_s = common::dequantize(common::get_xtensor(tensors[i+1]), tensors[i+1]->vstream_info().quant_info.qp_scale, tensors[i+1]->vstream_info().quant_info.qp_zp);
        int num_proposals_scores = dequantized_output_s.shape(0)*dequantized_output_s.shape(1);

        // From the layer extract the scores
        auto output_scores = xt::view(dequantized_output_s, xt::all(), xt::all(), xt::all());
        xt::view(scores, xt::range(view_index_scores, view_index_scores + num_proposals_scores), xt::all()) = xt::reshape_view(output_scores, {num_proposals_scores, num_classes});
        view_index_scores += num_proposals_scores;

        // Keypoints extraction will be done later according to the boxes that surpass the threshold
        outputs_keypoints[i / 3] = tensors[i+2];

    }
    return Triple{outputs_boxes, scores, outputs_keypoints};
}

std::vector<Decodings> yolov8pose_postprocess(std::vector<HailoTensorPtr> &tensors,
                                std::vector<int> network_dims,
                                std::vector<int> strides,
                                int regression_length,
                                int num_classes, float conf_thr)
{
    std::vector<Decodings> decodings;
    if (tensors.size() == 0)
    {
        return decodings;
    }

    Triple boxes_scores_keypoints = get_boxes_scores_keypoints(tensors, num_classes, regression_length);
    std::vector<HailoTensorPtr> raw_boxes = boxes_scores_keypoints.boxes;
    xt::xarray<float> scores = boxes_scores_keypoints.scores;
    std::vector<HailoTensorPtr> raw_keypoints = boxes_scores_keypoints.keypoints;

    // Decode the boxes and keypoints
    decodings = decode_boxes_and_keypoints(raw_boxes, scores, raw_keypoints, network_dims, strides, regression_length, conf_thr);

    // Filter with NMS
    auto decodings_after_nms = nms(decodings, IOU_THRESHOLD, true);

    return decodings_after_nms;
}

/**
 * @brief yolov8 postprocess
 *        Provides network specific paramters
 * 
 * @param roi  -  HailoROIPtr
 *        The roi that contains the ouput tensors
 */

std::pair<std::vector<KeyPt>, std::vector<PairPairs>> yolov8(HailoROIPtr roi, float conf_thr,
                           std::vector<int> strides,
                           std::vector<int> network_dims)
{
    // anchor params
    int regression_length = 15;

    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    auto filtered_decodings = yolov8pose_postprocess(tensors, network_dims, strides, regression_length, NUM_CLASSES, conf_thr);

    std::vector<HailoDetection> detections;

    for (auto& dec : filtered_decodings){
        detections.push_back(dec.detection_box);
    }

    hailo_common::add_detections(roi, detections);

    std::pair<std::vector<KeyPt>, std::vector<PairPairs>> keypoints_and_pairs = filter_keypoints(filtered_decodings, network_dims);

    return keypoints_and_pairs;
}

//******************************************************************
//  DEFAULT FILTER
//******************************************************************

std::pair<std::vector<KeyPt>, std::vector<PairPairs>> filter(HailoROIPtr roi, float conf_thr,
                           std::vector<int> strides,
                           std::vector<int> network_dims)
{
    return yolov8(roi, conf_thr, strides, network_dims);
}

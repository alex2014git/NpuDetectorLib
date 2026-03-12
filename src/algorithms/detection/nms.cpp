#include "common/nms.hpp"

namespace common
{
  // Implementation of iou_calc function for calculating Intersection over Union (IoU)
  float iou_calc(const HailoBBox &box_1, const HailoBBox &box_2)
  {
    // Calculate the area of overlap
    const float width_of_overlap_area = std::min(box_1.xmax(), box_2.xmax()) - std::max(box_1.xmin(), box_2.xmin());
    const float height_of_overlap_area = std::min(box_1.ymax(), box_2.ymax()) - std::max(box_1.ymin(), box_2.ymin());
    const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
    const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
    const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;

    // Calculate the area of each box
    const float box_1_area = (box_1.ymax() - box_1.ymin()) * (box_1.xmax() - box_1.xmin());
    const float box_2_area = (box_2.ymax() - box_2.ymin()) * (box_2.xmax() - box_2.xmin());

    // Calculate the IoU
    return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
  }

  // Implementation of nms function for performing Non-Maximum Suppression (NMS) on detections
  void nms(std::vector<HailoDetection> &objects, const float iou_thr, bool should_nms_cross_classes)
  {
    // Sort detections by confidence in descending order
    std::sort(objects.begin(), objects.end(),
              [](HailoDetection a, HailoDetection b) { return a.get_confidence() > b.get_confidence(); });

    std::vector<HailoDetection> objects_after_nms;
    for (uint index = 0; index < objects.size(); index++)
    {
      // Only consider detections with non-zero confidence
      if (objects[index].get_confidence() != 0.0f)
      {
        for (uint jindex = index + 1; jindex < objects.size(); jindex++)
        {
          // Consider detections of the same class or all classes if specified
          if ((should_nms_cross_classes || (objects[index].get_class_id() == objects[jindex].get_class_id())) &&
              objects[jindex].get_confidence() != 0.0f)
          {
            // Calculate IoU between the current and subsequent detections
            float iou = iou_calc(objects[index].get_bbox(), objects[jindex].get_bbox());

            // Suppress detection with lower confidence if IoU is above threshold
            if (iou >= iou_thr)
            {
              objects[jindex].set_confidence(0.0f);
            }
          }
        }
        // Add detections with non-zero confidence to the final list
        objects_after_nms.push_back(objects[index]);
      }
    }

    // Replace the original objects with the filtered detections
    objects = objects_after_nms;
  }
}


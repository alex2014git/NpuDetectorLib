#include "yolov8_common.hpp"


float dequantize_value(uint8_t val, float32_t qp_scale, float32_t qp_zp){
    return (float(val) - qp_zp) * qp_scale;
}

                   
void dequantize_box_values(xt::xarray<float>& dequantized_outputs, int index, 
                        xt::xarray<uint8_t>& quantized_outputs,
                        size_t dim1, size_t dim2, float32_t qp_scale, float32_t qp_zp){
    for (size_t i = 0; i < dim1; i++){
        for (size_t j = 0; j < dim2; j++){
            dequantized_outputs(i, j) = dequantize_value(quantized_outputs(index, i, j), qp_scale, qp_zp);
        }
    }
}

float iou_calc(const HailoBBox &box_1, const HailoBBox &box_2)
{
    // Calculate IOU between two detection boxes
    const float width_of_overlap_area = std::min(box_1.xmax(), box_2.xmax()) - std::max(box_1.xmin(), box_2.xmin());
    const float height_of_overlap_area = std::min(box_1.ymax(), box_2.ymax()) - std::max(box_1.ymin(), box_2.ymin());
    const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
    const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
    const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;
    const float box_1_area = (box_1.ymax() - box_1.ymin()) * (box_1.xmax() - box_1.xmin());
    const float box_2_area = (box_2.ymax() - box_2.ymin()) * (box_2.xmax() - box_2.xmin());
    // The IOU is a ratio of how much the boxes overlap vs their size outside the overlap.
    // Boxes that are similar will have a higher overlap threshold.
    return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
}

std::vector<xt::xarray<double>> get_centers(std::vector<int>& strides, std::vector<int>& network_dims,
                                        std::size_t boxes_num, int strided_width, int strided_height){

        std::vector<xt::xarray<double>> centers(boxes_num);

        for (uint i=0; i < boxes_num; i++) {
            strided_width = network_dims[0] / strides[i];
            strided_height = network_dims[1] / strides[i];

            // Create a meshgrid of the proper strides
            xt::xarray<int> grid_x = xt::arange(0, strided_width);
            xt::xarray<int> grid_y = xt::arange(0, strided_height);

            auto mesh = xt::meshgrid(grid_x, grid_y);
            grid_x = std::get<1>(mesh);
            grid_y = std::get<0>(mesh);

            // Use the meshgrid to build up box center prototypes
            auto ct_row = (xt::flatten(grid_y) + 0.5) * strides[i];
            auto ct_col = (xt::flatten(grid_x) + 0.5) * strides[i];

            centers[i] = xt::stack(xt::xtuple(ct_col, ct_row, ct_col, ct_row), 1);
        }

        return centers;
}



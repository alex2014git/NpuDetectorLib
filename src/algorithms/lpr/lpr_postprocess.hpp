/**
 * LPR (License Plate Recognition) Post-Processing
 *
 * Decodes LPR model output to text using charset mapping.
 */

#ifndef LPR_POSTPROCESS_HPP
#define LPR_POSTPROCESS_HPP

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace npu_pipeline {

/**
 * @brief Character set for Chinese license plates
 * Index mapping matches the LPR model training
 */
static const char* LPR_CHARSET[] = {
    "#", "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川",
    "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "学", "警", "港", "澳", "挂", "使", "领", "民", "航", "危",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "险", "品", "I", "O", "-"
};

constexpr size_t LPR_CHARSET_SIZE = sizeof(LPR_CHARSET) / sizeof(LPR_CHARSET[0]);

/**
 * @brief Decode LPR model output to license plate text
 *
 * @param output Raw model output (float array)
 * @param output_size Number of output elements (typically 21 for LPR)
 * @return std::string Decoded license plate text
 */
inline std::string decode_lpr_output(const float* output, int output_size) {
    std::string plate;
    std::string prev = "#";

    for (int i = 0; i < output_size; ++i) {
        float raw = output[i];

        // Skip invalid values
        if (std::isnan(raw) || std::isinf(raw)) {
            continue;
        }

        int idx = static_cast<int>(std::round(raw));

        // Validate index bounds
        if (idx < 0 || idx >= static_cast<int>(LPR_CHARSET_SIZE)) {
            continue;
        }

        const std::string& c = LPR_CHARSET[idx];

        // Skip blank/stop character and duplicates
        if (c != "#" && c != prev) {
            plate += c;
        }
        prev = c;
    }

    return plate;
}

/**
 * @brief Calculate average confidence from LPR output
 *
 * @param output Raw model output
 * @param output_size Number of output elements
 * @return float Average confidence (based on character probability)
 */
inline float calculate_lpr_confidence(const float* output, int output_size) {
    float total_conf = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < output_size; ++i) {
        float raw = output[i];
        if (!std::isnan(raw) && !std::isinf(raw)) {
            int idx = static_cast<int>(std::round(raw));
            if (idx >= 0 && idx < static_cast<int>(LPR_CHARSET_SIZE)) {
                // Simple confidence based on whether index is valid
                total_conf += 1.0f;
            }
            valid_count++;
        }
    }

    return valid_count > 0 ? (total_conf / output_size) : 0.0f;
}

} // namespace npu_pipeline

#endif // LPR_POSTPROCESS_HPP
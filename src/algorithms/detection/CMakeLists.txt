target_sources(NpuLibrary PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_postprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8seg_postprocess.cpp
)
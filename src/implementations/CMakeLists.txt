target_sources(NpuLibrary PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_yolo_nms_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_yolo_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_yolov8_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_yolov8_pose_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_yolov8_seg_impl.cpp
)
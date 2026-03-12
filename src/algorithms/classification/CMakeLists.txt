# Classification Post-Processing
add_library(classification_postprocess
    classification_postprocess.cpp
)
target_include_directories(classification_postprocess PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(classification_postprocess PUBLIC
    npu_core
)
# LPR Post-Processing
add_library(lpr_postprocess
    lpr_postprocess.cpp
)
target_include_directories(lpr_postprocess PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(lpr_postprocess PUBLIC
    npu_core
)
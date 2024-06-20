# Set common include and library paths
if (NOT DEFINED HAILORT_INCLUDE)
    if (UNIX)
        set(HAILORT_INCLUDE "/usr/include/hailo" CACHE PATH "Path to HailoRT include directory")
    else ()
        set(HAILORT_ROOT "C:/Program Files/HailoRT" CACHE PATH "Path to HailoRT root directory")
        set(HAILORT_INCLUDE "${HAILORT_ROOT}/include" CACHE PATH "Path to HailoRT include directory")
    endif (UNIX)
endif ()

if (NOT DEFINED HAILORT_LIB)
    if (UNIX)
        set(HAILORT_LIB "/usr/lib/libhailort.so" CACHE FILEPATH "Path to HailoRT library")
    else ()
        set(HAILORT_LIB "${HAILORT_ROOT}/lib/libhailort.lib" CACHE FILEPATH "Path to HailoRT library")
    endif (UNIX)
endif ()

# Set common compile definitions
set(COMMON_COMPILE_DEFINITIONS
    LETTER_BOX
    SHOW_LABEL
    CACHE INTERNAL "Common compile definitions"
)

# Optional compile definitions
# set(COMMON_COMPILE_DEFINITIONS
#     ${COMMON_COMPILE_DEFINITIONS}
#     TIME_TRACE_DEBUG
#     CACHE INTERNAL "Common compile definitions with optional TIME_TRACE_DEBUG"
# )

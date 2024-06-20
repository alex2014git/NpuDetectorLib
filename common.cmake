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

if(NOT DEFINED LETTER_BOX)
    set(LETTER_BOX OFF CACHE INTERNAL "Enable letterboxing functionality")
endif()

# Allow overriding with command-line argument
option(LETTER_BOX "Enable letterboxing functionality" ${LETTER_BOX})

if(NOT DEFINED SHOW_LABEL)
    set(SHOW_LABEL OFF CACHE INTERNAL "Enable show labels functionality")
endif()

# Allow overriding with command-line argument
option(SHOW_LABEL "Enable show labels functionality" ${SHOW_LABEL})

if(NOT DEFINED TIME_TRACE_DEBUG)
    set(TIME_TRACE_DEBUG OFF CACHE INTERNAL "Enable debuging and time tracking functionality")
endif()

# Allow overriding with command-line argument
option(TIME_TRACE_DEBUG "Enable debuging and time tracking functionality" ${TIME_TRACE_DEBUG})

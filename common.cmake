# Allow setting HAILORT paths through options

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
option(HAILORT_INCLUDE "Path to HailoRT include directory" "/usr/include/hailo")
option(HAILORT_LIB "Path to HailoRT library" "/usr/lib/libhailort.so")

if(NOT DEFINED LETTER_BOX)
    set(LETTER_BOX ON CACHE BOOL "Enable letterboxing functionality")
endif()

option(LETTER_BOX "Enable letterboxing functionality" ${LETTER_BOX})

if(NOT DEFINED SHOW_LABEL)
    set(SHOW_LABEL OFF CACHE BOOL "Enable show labels functionality")
endif()

option(SHOW_LABEL "Enable show labels functionality" ${SHOW_LABEL})

if(NOT DEFINED TIME_TRACE_DEBUG)
    set(TIME_TRACE_DEBUG OFF CACHE BOOL "Enable debugging and time tracking functionality")
endif()

option(TIME_TRACE_DEBUG "Enable debugging and time tracking functionality" ${TIME_TRACE_DEBUG})

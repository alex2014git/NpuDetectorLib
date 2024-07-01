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

option(LETTER_BOX "Enable letterboxing functionality" ON)

option(SHOW_LABEL "Enable show labels functionality" OFF)

option(TIME_TRACE_DEBUG "Enable debugging and time tracking functionality" OFF)

option(BUILD_TESTER "Build the test suite" OFF)

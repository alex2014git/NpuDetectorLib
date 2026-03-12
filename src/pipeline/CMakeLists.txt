file(GLOB PIPELINE_SOURCES
    *.cpp
)

target_sources(NpuLibrary PRIVATE ${PIPELINE_SOURCES})
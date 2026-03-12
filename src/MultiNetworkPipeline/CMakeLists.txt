file(GLOB MULTINETWORK_SOURCES
    *.cpp
)

target_sources(NpuLibrary PRIVATE ${MULTINETWORK_SOURCES})
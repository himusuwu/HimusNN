cmake_minimum_required(VERSION 3.16)
project(HimusNN)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SOURCE_FILES 
    src/main.cpp 
    src/Layer.cpp 
    src/Network.cpp 
    src/Neuron.cpp 
    src/Progress.cpp 
    src/Timing.cpp 
    src/Trainer.cpp 
    src/Metric.cpp 
    src/DataLoader.cpp 
    src/Matrix.mm
)

include_directories(include)

add_executable(HimusNN ${SOURCE_FILES})

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    target_compile_options(HimusNN PRIVATE 
        -O3
        -ffast-math
        -mcpu=native
    )

    target_compile_definitions(HimusNN PRIVATE
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
    )

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)

    if(ipo_supported)
        set_target_properties(HimusNN PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    endif()
endif()

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    find_library(METAL_FRAMEWORK Metal)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(HimusNN PRIVATE
            ${ACCELERATE_FRAMEWORK}
            ${METAL_FRAMEWORK}
            ${MPS_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )
    endif()
endif()

add_custom_target(run
    COMMAND ./HimusNN
    DEPENDS HimusNN
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
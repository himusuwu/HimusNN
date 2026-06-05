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
    src/Matrix.cpp
)

include_directories(include)

add_executable(HimusNN ${SOURCE_FILES})

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    target_compile_options(HimusNN PRIVATE 
        -O3
        -ffast-math
        -mcpu=native
    )

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)

    if(ipo_supported)
        set_target_properties(HimusNN PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    endif()
endif()

add_custom_target(run
    COMMAND ./HimusNN
    DEPENDS HimusNN
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
cmake_minimum_required(VERSION 3.16)
project(HimusNN)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCE_FILES src/main.cpp src/Layer.cpp src/Network.cpp src/Neuron.cpp src/Progress.cpp src/Timing.cpp src/Trainer.cpp src/Metric.cpp src/DataLoader.cpp src/Matrix.cpp)

include_directories(include)

add_executable(HimusNN ${SOURCE_FILES})

add_custom_target(run
    COMMAND ./HimusNN
    DEPENDS HimusNN
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
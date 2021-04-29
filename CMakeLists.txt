cmake_minimum_required(VERSION 3.19)
project(FreeSpaceEstimation_q10)

set(CMAKE_CXX_STANDARD 14)

include_directories(./external/opencv2.3.13.6/include)
link_directories(./external/opencv2.3.13.6/x64/vc14/lib)

include_directories(.)

add_executable(FreeSpaceEstimation_q10
        freespace.cpp
        freespace.hpp
        main.cpp)

set(EXT_LIBS opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d)
target_link_libraries(FreeSpaceEstimation_q10 ${EXT_LIBS})

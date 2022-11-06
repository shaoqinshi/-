cmake_minimum_required(VERSION 3.1)
project(cpp_test)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIB_DIRS})

add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test ${OpenCV_LIBS})

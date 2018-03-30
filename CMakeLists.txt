#CMake minimum version
cmake_minimum_required(VERSION 2.8)

#Project name
#--------------------------------------------
#Remember: Need to be changed in Project->settings->General->Executable to run
#          $(ProjectPath)build\sample
#--------------------------------------------
set(project_name test_DOT)


#Project definition
#--------------------------------------------
project( ${project_name} )

set(OpenCV_DIR /usr/local/opencv-2.4.10_normal/share/OpenCV)

#Find libraries
#--------------------------------------------
find_package( OpenCV REQUIRED )
#find_package( Boost REQUIRED )
find_package( Eigen3 REQUIRED )
#find_package( OpenNI REQUIRED )
#find_package( Nite REQUIRED )


#Include directories
#--------------------------------------------
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
#include_directories(${OpenNI_INCLUDE_DIR})
#include_directories(${Nite_INCLUDE_DIR})


#Link directories
#--------------------------------------------
#link_directories(${Boost_LIBRARY_DIR})

message(${OpenCV_INCLUDE_DIRS})
message(${EIGEN3_INCLUDE_DIR})
message(${OpenCV_LIBS})

#Other flags
#--------------------------------------------
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")  # set gprof
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")       # no gprof
#set(CMAKE_BUILD_TYPE "DEBUG")
set(CMAKE_BUILD_TYPE "RELEASE")


#Files to be compiled
#--------------------------------------------
add_executable( ${project_name} main.cpp ) 


#Link libraries
#--------------------------------------------
target_link_libraries( ${project_name} ${OpenCV_LIBS})
#target_link_libraries( ${project_name} ${Boost_LIBRARIES})
#target_link_libraries( ${project_name} ${OpenNI_LIBRARIES})
#target_link_libraries( ${project_name} ${Nite_LIBRARIES})
#target_link_libraries( ${project_name} -fopenmp )

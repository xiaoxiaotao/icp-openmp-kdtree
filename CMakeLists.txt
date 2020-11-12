project(icp_openmp)
cmake_minimum_required(VERSION 3.8)

SET(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -lopenblas -llapack")
set(CMAKE_CXX_FLAGS_RELEASE "-O0 -Wall -g  -g -ggdb")


# add_subdirectory(cuda)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(CUDA REQUIRED)


include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(
  
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  /home/tao/learning/icp_openmp_kdtree/include/
  
  )

add_executable(
  icp_openmp 
    src/main.cpp
    src/icp.cpp

 )
target_link_libraries(
  icp_openmp 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBRARIES}
 

  )

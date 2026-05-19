cmake_minimum_required(VERSION 3.8)
project(zed_launcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(zed_msgs REQUIRED)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
  cmake_policy(SET CMP0146 OLD)
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
  cmake_policy(SET CMP0153 OLD)
endif()
find_package(ZED REQUIRED)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
  cmake_policy(SET CMP0146 NEW)
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
  cmake_policy(SET CMP0153 NEW)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(zed_body_fusion_node
  src/zed_body_fusion_node.cpp
)

target_include_directories(zed_body_fusion_node PRIVATE
  ${ZED_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(zed_body_fusion_node PRIVATE
  ${ZED_LIBRARY_DIR}
  ${CUDAToolkit_LIBRARY_DIR}
)

target_link_libraries(zed_body_fusion_node
  ${ZED_LIBRARIES}
  CUDA::cudart
)

ament_target_dependencies(zed_body_fusion_node
  rclcpp
  zed_msgs
)

install(
  TARGETS zed_body_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch calibration rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

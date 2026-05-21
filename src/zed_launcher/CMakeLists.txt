cmake_minimum_required(VERSION 3.8)
project(zed_launcher)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(zed_msgs REQUIRED)

find_package(ZED REQUIRED)
find_package(CUDAToolkit REQUIRED)

install(
  DIRECTORY launch calibration rviz
  DESTINATION share/${PROJECT_NAME}
)

# ZED Body Fusion

add_executable(zed_body_fusion_node
  src/zed_body_fusion_node.cpp
)

target_include_directories(zed_body_fusion_node PRIVATE
  ${ZED_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(zed_body_fusion_node
  ${ZED_LIBRARIES}
  ${OpenCV_LIBRARIES}
  CUDA::cudart
)

ament_target_dependencies(zed_body_fusion_node
  geometry_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
  zed_msgs
)

# ZED Body Tracking Overlay

add_executable(zed_skeleton_overlay_node
  src/zed_skeleton_overlay_node.cpp
)

target_include_directories(zed_skeleton_overlay_node PRIVATE
  ${ZED_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(zed_skeleton_overlay_node
  ${ZED_LIBRARIES}
  ${OpenCV_LIBRARIES}
  CUDA::cudart
)

target_link_directories(zed_skeleton_overlay_node PRIVATE
  ${ZED_LIBRARY_DIR}
  ${OpenCV_LIBRARY_DIR}
)

ament_target_dependencies(zed_skeleton_overlay_node
  rclcpp
  sensor_msgs
  zed_msgs
)

install(
  TARGETS
    zed_body_fusion_node
    zed_skeleton_overlay_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

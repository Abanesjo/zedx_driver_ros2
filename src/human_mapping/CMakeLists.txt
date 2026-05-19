cmake_minimum_required(VERSION 3.8)
project(human_mapping)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(g1_cbf_msg REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(zed_msgs REQUIRED)

add_executable(human_mapping_node
  src/human_mapping_node.cpp
)

add_executable(human_collider_viz_node
  src/human_collider_viz_node.cpp
)

ament_target_dependencies(human_mapping_node
  builtin_interfaces
  geometry_msgs
  g1_cbf_msg
  rclcpp
  sensor_msgs
  zed_msgs
)

ament_target_dependencies(human_collider_viz_node
  geometry_msgs
  g1_cbf_msg
  rclcpp
  visualization_msgs
)

install(
  TARGETS
    human_mapping_node
    human_collider_viz_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

#include "human_mapping.cpp"

#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<human_mapping::HumanMappingNode>());
  rclcpp::shutdown();
  return 0;
}

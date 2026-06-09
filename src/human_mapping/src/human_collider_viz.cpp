#include "human_mapping/human_collider_viz_node.hpp"

#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<human_mapping::HumanColliderVizNode>());
  rclcpp::shutdown();
  return 0;
}

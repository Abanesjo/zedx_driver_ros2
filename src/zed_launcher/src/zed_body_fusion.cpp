#include "zed_launcher/zed_body_fusion_node.hpp"

#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<zed_launcher::ZedBodyFusionNode>();
    rclcpp::spin(node);
    node.reset();
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("zed_body_fusion_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}

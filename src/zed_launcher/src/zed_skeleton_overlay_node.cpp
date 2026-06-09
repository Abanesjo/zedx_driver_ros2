#include "zed_skeleton_overlay.cpp"

#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<zed_launcher::ZedSkeletonOverlayNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("zed_skeleton_overlay_node"), "%s",
                 e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}

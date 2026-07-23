#include "zed_launcher/apriltag_fusion_node.hpp"
#include "zed_launcher/zed_body_fusion_node.hpp"

#include <exception>
#include <memory>
#include <utility>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  try {
    const rclcpp::NodeOptions options;
    auto apriltag_node = std::make_shared<zed_launcher::ApriltagFusionNode>(
        options, zed_launcher::ApriltagFusionNode::ImageInputMode::Direct);

    zed_launcher::ZedBodyFusionNode::ImageProcessor image_processor;
    if (apriltag_node->enabled()) {
      const std::weak_ptr<zed_launcher::ApriltagFusionNode> weak_apriltag_node =
          apriltag_node;
      image_processor =
          [weak_apriltag_node](const std::string &camera_name,
                               const sensor_msgs::msg::Image &source,
                               const sensor_msgs::msg::CameraInfo &camera_info,
                               sensor_msgs::msg::Image *debug_overlay) {
            if (const auto node = weak_apriltag_node.lock()) {
              node->processCameraFrame(camera_name, source, camera_info,
                                       debug_overlay);
            }
          };
    }

    auto body_fusion_node = std::make_shared<zed_launcher::ZedBodyFusionNode>(
        options, std::move(image_processor), "/zed_fusion");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(apriltag_node);
    executor.add_node(body_fusion_node);
    executor.spin();

    executor.remove_node(body_fusion_node);
    executor.remove_node(apriltag_node);
    body_fusion_node.reset();
    apriltag_node.reset();
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("body_tracking_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}

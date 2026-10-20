#include "zed_launcher/apriltag_fusion_node.hpp"
#include "zed_launcher/async_apriltag_processor.hpp"
#include "zed_launcher/zed_body_fusion_node.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  try {
    const rclcpp::NodeOptions options;
    auto apriltag_node = std::make_shared<zed_launcher::ApriltagFusionNode>(
        options, zed_launcher::ApriltagFusionNode::ImageInputMode::Direct);

    std::shared_ptr<zed_launcher::AsyncApriltagProcessor>
        async_apriltag_processor;
    zed_launcher::ZedBodyFusionNode::ImageProcessor image_processor;
    if (apriltag_node->enabled()) {
      async_apriltag_processor =
          std::make_shared<zed_launcher::AsyncApriltagProcessor>(
              apriltag_node, apriltag_node->cameraNames());
      const std::weak_ptr<zed_launcher::AsyncApriltagProcessor> weak_processor =
          async_apriltag_processor;

      image_processor.needs_depth = async_apriltag_processor->needsDepth();
      image_processor.should_process =
          [weak_processor](const std::string &camera_name) {
            if (const auto processor = weak_processor.lock()) {
              return processor->shouldProcessFrame(camera_name);
            }
            return false;
          };
      image_processor.submit =
          [weak_processor](
              std::string camera_name, sensor_msgs::msg::Image source,
              sensor_msgs::msg::CameraInfo camera_info,
              std::optional<zed_launcher::OwnedDepthFrame> depth,
              std::optional<sensor_msgs::msg::Image> debug_overlay,
              zed_launcher::ZedBodyFusionNode::ImageProcessor::DebugPublisher
                  debug_publisher) {
            if (const auto processor = weak_processor.lock()) {
              processor->submit(std::move(camera_name), std::move(source),
                                std::move(camera_info), std::move(depth),
                                std::move(debug_overlay),
                                std::move(debug_publisher));
            }
          };
      image_processor.stats = [weak_processor]() {
        std::vector<zed_launcher::ZedBodyFusionNode::ImageProcessorCameraStats>
            result;
        const auto processor = weak_processor.lock();
        if (!processor) {
          return result;
        }
        const auto processor_stats = processor->stats();
        result.reserve(processor_stats.size());
        for (const auto &stats : processor_stats) {
          result.push_back(
              zed_launcher::ZedBodyFusionNode::ImageProcessorCameraStats{
                  stats.camera_name, stats.submitted, stats.processed,
                  stats.overwritten, stats.stale_dropped,
                  stats.last_processing_ms, stats.last_job_age_ms});
        }
        return result;
      };
    }

    auto body_fusion_node = std::make_shared<zed_launcher::ZedBodyFusionNode>(
        options, std::move(image_processor), "/zed_fusion");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(apriltag_node);
    executor.add_node(body_fusion_node);
    executor.spin();

    executor.remove_node(body_fusion_node);
    if (async_apriltag_processor) {
      async_apriltag_processor->stop();
    }
    body_fusion_node.reset();
    async_apriltag_processor.reset();
    executor.remove_node(apriltag_node);
    apriltag_node.reset();
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("body_tracking_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}

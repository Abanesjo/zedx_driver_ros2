#pragma once

#include <mutex>
#include <string>

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>

namespace zed_launcher {

class ZedSkeletonOverlayNode final : public rclcpp::Node {
public:
  explicit ZedSkeletonOverlayNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  bool validPoint(const zed_msgs::msg::Keypoint2Df &keypoint,
                  const cv::Size &size) const;

  cv::Point toPoint(const zed_msgs::msg::Keypoint2Df &keypoint) const;

  void drawObject(cv::Mat &image, const zed_msgs::msg::Object &object) const;

  cv::Point labelAnchor(const zed_msgs::msg::Object &object) const;

  void handleImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

  bool skeletonsAreFresh(const zed_msgs::msg::ObjectsStamped &skeletons) const;

  bool toBgrImage(const sensor_msgs::msg::Image &msg, cv::Mat &image) const;

  sensor_msgs::msg::Image
  toImageMessage(const cv::Mat &image,
                 const std_msgs::msg::Header &header) const;

  std::string image_topic_;

  std::string skeleton_topic_;

  std::string overlay_topic_;

  double min_confidence_ = 70.0;

  double max_skeleton_age_sec_ = 0.5;

  int line_thickness_ = 2;

  int point_radius_ = 4;

  bool draw_labels_ = false;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  rclcpp::Subscription<zed_msgs::msg::ObjectsStamped>::SharedPtr skeleton_sub_;

  zed_msgs::msg::ObjectsStamped::SharedPtr latest_skeletons_;

  std::mutex skeleton_mutex_;
};

} // namespace zed_launcher

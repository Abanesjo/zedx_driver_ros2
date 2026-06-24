#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace zed_launcher {

class ApriltagFusionNode final : public rclcpp::Node {
public:
  explicit ApriltagFusionNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  enum class CameraFrameConvention { RosOptical, ZedXForward };

  void handleCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  void handleImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  bool shouldSkipFrame();

  bool toGrayImage(const sensor_msgs::msg::Image &msg, cv::Mat &gray) const;

  std::optional<cv::Mat>
  cameraMatrixForImage(const sensor_msgs::msg::CameraInfo &camera_info,
                       const sensor_msgs::msg::Image &image) const;

  cv::Mat distortionCoeffs(
      const sensor_msgs::msg::CameraInfo &camera_info) const;

  std::optional<size_t>
  selectMarker(const std::vector<int> &ids,
               const std::vector<std::vector<cv::Point2f>> &corners) const;

  std::optional<tf2::Transform> estimateTagInCameraFrame(
      const std::vector<cv::Point2f> &corners,
      const cv::Mat &camera_matrix,
      const cv::Mat &distortion_coeffs) const;

  std::vector<cv::Point3f> markerObjectPoints() const;

  tf2::Transform transformFromMsg(
      const geometry_msgs::msg::Transform &msg) const;

  geometry_msgs::msg::Pose poseMsgFromTransform(
      const tf2::Transform &transform) const;

  geometry_msgs::msg::Transform transformMsgFromTransform(
      const tf2::Transform &transform) const;

  void publishTagToPelvisTransform();

  cv::aruco::PREDEFINED_DICTIONARY_NAME
  parseDictionary(const std::string &dictionary) const;

  CameraFrameConvention
  parseCameraFrameConvention(const std::string &convention) const;

  std::string image_topic_;

  std::string camera_info_topic_;

  std::string fusion_frame_id_;

  std::string tag_frame_id_;

  std::string pelvis_frame_id_;

  std::string pose_topic_;

  int target_tag_id_ = 0;

  double tag_size_m_ = 0.06;

  double max_publish_rate_hz_ = 30.0;

  double tf_lookup_timeout_sec_ = 0.05;

  bool publish_tag_pose_ = true;

  bool publish_tf_ = true;

  bool publish_tag_to_pelvis_tf_ = false;

  tf2::Transform tag_to_pelvis_;

  CameraFrameConvention camera_frame_convention_ =
      CameraFrameConvention::ZedXForward;

  cv::Ptr<cv::aruco::Dictionary> dictionary_;

  cv::Ptr<cv::aruco::DetectorParameters> detector_params_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;

  std::mutex camera_info_mutex_;

  tf2_ros::Buffer tf_buffer_;

  tf2_ros::TransformListener tf_listener_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  rclcpp::Time last_processed_time_;

  bool has_last_processed_time_ = false;
};

} // namespace zed_launcher

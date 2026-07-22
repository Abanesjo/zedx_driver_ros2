#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR > 4 ||                                                \
    (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#define ZED_LAUNCHER_USE_MODERN_ARUCO 1
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#define ZED_LAUNCHER_USE_MODERN_ARUCO 0
#include <opencv2/aruco.hpp>
#endif
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace zed_launcher {

class ApriltagFusionNode final : public rclcpp::Node {
public:
  explicit ApriltagFusionNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  enum class CameraFrameConvention { RosOptical, ZedXForward };

  class ArucoDetectorAdapter {
  public:
    ArucoDetectorAdapter(int dictionary_id, bool corner_refinement);

    void detectMarkers(
        const cv::Mat &image,
        std::vector<std::vector<cv::Point2f>> &corners,
        std::vector<int> &ids) const;

  private:
#if ZED_LAUNCHER_USE_MODERN_ARUCO
    cv::aruco::ArucoDetector detector_;
#else
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> detector_parameters_;
#endif
  };

  struct Observation {
    tf2::Transform fusion_from_tag;
    rclcpp::Time stamp;
    rclcpp::Time receipt_time;
    double quality = 0.0;
    double marker_area_px2 = 0.0;
    double reprojection_rmse_px = 0.0;
    uint64_t sequence = 0;
  };

  struct PoseEstimate {
    tf2::Transform camera_from_tag;
    double reprojection_rmse_px = 0.0;
  };

  struct CameraContext {
    std::string name;
    std::string image_topic;
    std::string camera_info_topic;
    std::unique_ptr<ArucoDetectorAdapter> detector;
    rclcpp::CallbackGroup::SharedPtr callback_group;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
        camera_info_sub;
    sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info;
    std::optional<Observation> latest_observation;
    rclcpp::Time last_detection_attempt_time;
    bool has_last_detection_attempt_time = false;
    std::mutex mutex;
  };

  void handleCameraInfo(
      size_t camera_index,
      const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  void handleImage(size_t camera_index,
                   const sensor_msgs::msg::Image::ConstSharedPtr msg);

  bool shouldSkipFrame(CameraContext &camera, const rclcpp::Time &current_time);

  void fuseAndPublish();

  bool observationsAgree(const Observation &lhs,
                         const Observation &rhs) const;

  tf2::Transform fuseObservations(
      const std::vector<Observation> &observations,
      const std::vector<size_t> &selected_indices) const;

  tf2::Transform smoothTransform(const tf2::Transform &fusion_from_tag,
                                 const rclcpp::Time &source_stamp,
                                 const rclcpp::Time &receipt_time);

  bool toGrayImage(const sensor_msgs::msg::Image &msg, cv::Mat &gray) const;

  std::optional<cv::Mat>
  cameraMatrixForImage(const sensor_msgs::msg::CameraInfo &camera_info,
                       const sensor_msgs::msg::Image &image) const;

  cv::Mat distortionCoeffs(
      const sensor_msgs::msg::CameraInfo &camera_info) const;

  std::optional<size_t>
  selectMarker(const std::vector<int> &ids,
               const std::vector<std::vector<cv::Point2f>> &corners) const;

  std::optional<PoseEstimate> estimateTagInCameraFrame(
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

  int parseDictionary(const std::string &dictionary) const;

  CameraFrameConvention
  parseCameraFrameConvention(const std::string &convention) const;

  std::vector<std::string> camera_names_;
  std::string fusion_frame_id_;
  std::string tag_frame_id_;
  std::string pose_topic_;

  int target_tag_id_ = 0;
  double tag_size_m_ = 0.06;
  double max_detection_rate_hz_ = 30.0;
  double fusion_publish_rate_hz_ = 30.0;
  double tf_lookup_timeout_sec_ = 0.05;
  bool publish_fusion_pose_ = true;
  bool publish_tf_ = true;
  bool corner_refinement_ = true;
  double max_observation_age_sec_ = 0.15;
  double sync_tolerance_sec_ = 0.05;
  double min_marker_area_px2_ = 64.0;
  double max_reprojection_rmse_px_ = 3.0;
  double max_translation_disagreement_m_ = 0.25;
  double max_rotation_disagreement_deg_ = 25.0;
  double smoothing_time_constant_sec_ = 0.10;
  double smoothing_reset_sec_ = 0.50;

  CameraFrameConvention camera_frame_convention_ =
      CameraFrameConvention::ZedXForward;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  std::vector<std::unique_ptr<CameraContext>> cameras_;
  rclcpp::CallbackGroup::SharedPtr fusion_callback_group_;
  rclcpp::TimerBase::SharedPtr fusion_timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::atomic<uint64_t> next_observation_sequence_{1};
  std::vector<uint64_t> last_published_sequences_;
  tf2::Transform smoothed_fusion_from_tag_;
  rclcpp::Time last_smoothed_stamp_;
  rclcpp::Time last_smoothed_receipt_time_;
  bool has_smoothed_transform_ = false;
};

} // namespace zed_launcher

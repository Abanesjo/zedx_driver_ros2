#pragma once

#include "zed_launcher/depth_frame.hpp"

#include <array>
#include <atomic>
#include <cstddef>
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
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
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
  enum class ImageInputMode { RosTopics, Direct };

  explicit ApriltagFusionNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions(),
      ImageInputMode image_input_mode = ImageInputMode::RosTopics);

  bool enabled() const { return enabled_; }

  void processCameraFrame(const std::string &camera_name,
                          const sensor_msgs::msg::Image &source,
                          const sensor_msgs::msg::CameraInfo &camera_info,
                          const DepthFrameProvider &depth_provider,
                          sensor_msgs::msg::Image *debug_overlay);

private:
  friend class ApriltagFusionNodeTestPeer;

  static constexpr size_t kFrontTagSlot = 0;
  static constexpr size_t kBackTagSlot = 1;
  static constexpr size_t kTagCount = 2;

  enum class CameraFrameConvention { RosOptical, ZedXForward };

  class ArucoDetectorAdapter {
  public:
    ArucoDetectorAdapter(int dictionary_id, bool corner_refinement);

    void detectMarkers(const cv::Mat &image,
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

  struct FusedTagEstimate {
    Observation observation;
    std::vector<uint64_t> source_sequences;
  };

  struct PoseEstimate {
    tf2::Transform camera_from_tag;
    cv::Mat rvec;
    cv::Mat tvec;
    double reprojection_rmse_px = 0.0;
    bool used_depth = false;
    std::size_t depth_valid_samples = 0;
    std::size_t depth_inlier_samples = 0;
    double depth_valid_fraction = 0.0;
    double depth_plane_rmse_m = 0.0;
    double depth_size_error_fraction = 0.0;
    double depth_pnp_translation_delta_m = 0.0;
    double depth_pnp_rotation_delta_deg = 0.0;
    double depth_blend_weight = 0.0;
    double estimator_weight = 1.0;
  };

  struct DepthPlane {
    cv::Vec3d normal;
    double offset = 0.0;
    std::size_t valid_samples = 0;
    std::size_t inlier_samples = 0;
    double valid_fraction = 0.0;
    double rmse_m = 0.0;
  };

  struct CameraContext {
    std::string name;
    std::string image_topic;
    std::string camera_info_topic;
    std::string debug_topic;
    std::unique_ptr<ArucoDetectorAdapter> detector;
    rclcpp::CallbackGroup::SharedPtr callback_group;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
        camera_info_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub;
    sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info;
    std::array<std::optional<Observation>, kTagCount> latest_observations;
    rclcpp::Time last_detection_attempt_time;
    bool has_last_detection_attempt_time = false;
    std::mutex mutex;
  };

  void handleCameraInfo(size_t camera_index,
                        const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  void handleImage(size_t camera_index,
                   const sensor_msgs::msg::Image::ConstSharedPtr msg);

  std::optional<sensor_msgs::msg::Image>
  processImage(size_t camera_index, const sensor_msgs::msg::Image &source,
               const DepthFrameProvider &depth_provider,
               const sensor_msgs::msg::Image *debug_base, bool render_debug);

  bool shouldSkipFrame(CameraContext &camera, const rclcpp::Time &current_time);

  void fuseAndPublish();

  std::optional<FusedTagEstimate>
  fuseTag(size_t tag_slot, const rclcpp::Time &current_time) const;

  bool observationsAgree(const Observation &lhs, const Observation &rhs) const;

  Observation
  fuseObservations(const std::vector<Observation> &observations,
                   const std::vector<size_t> &selected_indices) const;

  tf2::Transform
  tagFrameFromSingleTag(size_t tag_slot,
                        const tf2::Transform &fusion_from_tag) const;

  tf2::Transform
  tagFrameFromBothTags(const tf2::Transform &fusion_from_front_tag,
                       const tf2::Transform &fusion_from_back_tag,
                       double front_quality = 1.0,
                       double back_quality = 1.0) const;

  void updateTagTransform(const FusedTagEstimate &front,
                          const FusedTagEstimate &back);

  void updateTagSeparation(const FusedTagEstimate &front,
                           const FusedTagEstimate &back);

  tf2::Transform activeFrontFromBackTag() const;

  std::optional<tf2::Transform>
  robustAverageTagTransforms(const std::vector<tf2::Transform> &samples) const;

  std::optional<size_t> selectFallbackTagSlot() const;

  tf2::Transform smoothTransform(const tf2::Transform &fusion_from_tag_frame,
                                 const rclcpp::Time &receipt_time);

  void publishResult(const tf2::Transform &fusion_from_tag_frame,
                     const rclcpp::Time &stamp);

  bool toGrayImage(const sensor_msgs::msg::Image &msg, cv::Mat &gray) const;

  bool toBgrImage(const sensor_msgs::msg::Image &msg, cv::Mat &bgr) const;

  static bool drawDebugMarkerOutline(cv::Mat &bgr,
                                     const std::vector<cv::Point2f> &corners);

  sensor_msgs::msg::Image makeDebugImage(const sensor_msgs::msg::Image &source,
                                         const cv::Mat &bgr) const;

  std::optional<cv::Mat>
  cameraMatrixForImage(const sensor_msgs::msg::CameraInfo &camera_info,
                       const sensor_msgs::msg::Image &image) const;

  cv::Mat
  distortionCoeffs(const sensor_msgs::msg::CameraInfo &camera_info) const;

  std::array<std::optional<size_t>, kTagCount>
  selectMarkers(const std::vector<int> &ids,
                const std::vector<std::vector<cv::Point2f>> &corners) const;

  std::optional<PoseEstimate> estimateTagInCameraFrame(
      const std::vector<cv::Point2f> &corners, const cv::Mat &camera_matrix,
      const cv::Mat &distortion_coeffs, double tag_size_m,
      const DepthFrameView *depth_frame, std::size_t image_width,
      std::size_t image_height) const;

  std::optional<PoseEstimate>
  estimateTagWithPnp(const std::vector<cv::Point2f> &corners,
                     const cv::Mat &camera_matrix,
                     const cv::Mat &distortion_coeffs, double tag_size_m) const;

  std::optional<PoseEstimate> estimateTagWithDepth(
      const std::vector<cv::Point2f> &corners, const cv::Mat &camera_matrix,
      const cv::Mat &distortion_coeffs, double tag_size_m,
      const DepthFrameView &depth_frame, std::size_t image_width,
      std::size_t image_height, const PoseEstimate &pnp_estimate) const;

  std::optional<DepthPlane>
  fitDepthPlane(const std::vector<cv::Point2f> &corners,
                const cv::Mat &camera_matrix, const DepthFrameView &depth_frame,
                std::size_t image_width, std::size_t image_height,
                const PoseEstimate *pnp_estimate) const;

  std::vector<cv::Point3f> markerObjectPoints(double tag_size_m) const;

  tf2::Transform
  transformFromMsg(const geometry_msgs::msg::Transform &msg) const;

  geometry_msgs::msg::Pose
  poseMsgFromTransform(const tf2::Transform &transform) const;

  geometry_msgs::msg::Transform
  transformMsgFromTransform(const tf2::Transform &transform) const;

  int parseDictionary(const std::string &dictionary) const;

  CameraFrameConvention
  parseCameraFrameConvention(const std::string &convention) const;

  int tagId(size_t tag_slot) const;
  double tagSize(size_t tag_slot) const;

  std::vector<std::string> camera_names_;
  std::string fusion_frame_id_;
  std::string tag_frame_id_;
  std::string pose_topic_;

  ImageInputMode image_input_mode_ = ImageInputMode::RosTopics;
  bool enabled_ = true;
  int front_tag_id_ = 0;
  int back_tag_id_ = 1;
  double front_tag_size_m_ = 0.12;
  double back_tag_size_m_ = 0.12;
  double initial_tag_frame_offset_m_ = 0.03;
  bool learn_tag_separation_ = true;
  double tag_separation_ema_alpha_ = 0.05;
  double tag_separation_max_innovation_m_ = 0.02;
  double learned_tag_separation_m_ = 0.06;
  bool use_depth_ = true;
  double depth_inner_margin_ratio_ = 0.20;
  int depth_min_valid_samples_ = 25;
  double depth_min_valid_fraction_ = 0.25;
  double depth_plane_inlier_threshold_m_ = 0.015;
  double depth_plane_max_rmse_m_ = 0.010;
  double depth_max_pnp_translation_delta_m_ = 0.20;
  double depth_max_pnp_rotation_delta_deg_ = 20.0;
  double depth_max_size_error_fraction_ = 0.25;
  bool learn_tag_transform_ = true;
  double tag_transform_bootstrap_duration_sec_ = 2.5;
  int tag_transform_bootstrap_min_samples_ = 30;
  double tag_transform_pair_max_age_sec_ = 0.10;
  double tag_transform_bootstrap_translation_outlier_m_ = 0.03;
  double tag_transform_bootstrap_rotation_outlier_deg_ = 8.0;
  double tag_transform_online_alpha_ = 0.01;
  double tag_transform_max_translation_step_m_ = 0.002;
  double tag_transform_max_rotation_step_deg_ = 0.25;
  double max_detection_rate_hz_ = 30.0;
  double fusion_publish_rate_hz_ = 30.0;
  double tf_lookup_timeout_sec_ = 0.05;
  bool publish_fusion_pose_ = true;
  bool publish_tf_ = true;
  bool publish_debug_images_ = false;
  double debug_axis_length_m_ = 0.06;
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
  std::array<std::optional<FusedTagEstimate>, kTagCount> last_tag_estimates_;
  std::vector<uint64_t> last_published_sequences_;
  std::vector<uint64_t> last_separation_update_sequences_;
  std::vector<uint64_t> last_tag_transform_update_sequences_;
  tf2::Transform ideal_front_from_back_tag_;
  tf2::Transform learned_front_from_back_tag_;
  std::vector<tf2::Transform> tag_transform_bootstrap_samples_;
  rclcpp::Time tag_transform_bootstrap_start_time_;
  bool has_tag_transform_bootstrap_start_time_ = false;
  bool tag_transform_calibrated_ = false;
  tf2::Transform smoothed_fusion_from_tag_frame_;
  rclcpp::Time last_smoothed_receipt_time_;
  bool has_smoothed_transform_ = false;
};

} // namespace zed_launcher

#include "zed_launcher/apriltag_fusion_node.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.hpp>

namespace zed_launcher {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadiansToDegrees = 180.0 / kPi;

std::string upper(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

double markerArea(const std::vector<cv::Point2f> &corners) {
  if (corners.size() != 4) {
    return 0.0;
  }
  return std::abs(cv::contourArea(corners));
}

cv::Mat cvOpticalToZedXForwardRotation() {
  return (cv::Mat_<double>(3, 3) << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0,
          0.0);
}

tf2::Vector3 tagZAxis(const tf2::Transform &fusion_from_tag) {
  return fusion_from_tag.getBasis() * tf2::Vector3(0.0, 0.0, 1.0);
}

double rotationDistanceDegrees(const tf2::Quaternion &lhs_value,
                               const tf2::Quaternion &rhs_value) {
  auto lhs = lhs_value;
  auto rhs = rhs_value;
  if (lhs.length2() <= 0.0 || rhs.length2() <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  lhs.normalize();
  rhs.normalize();
  const double dot = std::clamp(std::abs(lhs.dot(rhs)), 0.0, 1.0);
  return 2.0 * std::acos(dot) * kRadiansToDegrees;
}

bool finiteTransform(const tf2::Transform &transform) {
  const auto origin = transform.getOrigin();
  const auto rotation = transform.getRotation();
  return std::isfinite(origin.x()) && std::isfinite(origin.y()) &&
         std::isfinite(origin.z()) && std::isfinite(rotation.x()) &&
         std::isfinite(rotation.y()) && std::isfinite(rotation.z()) &&
         std::isfinite(rotation.w()) && rotation.length2() > 0.0;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const auto middle =
      values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  const double upper = *middle;
  if (values.size() % 2 != 0) {
    return upper;
  }
  const auto lower = std::max_element(values.begin(), middle);
  return 0.5 * (*lower + upper);
}

double smoothstep01(double value) {
  const double clamped = std::clamp(value, 0.0, 1.0);
  return clamped * clamped * (3.0 - 2.0 * clamped);
}

double upperLimitBlendWeight(double value, double limit) {
  if (!std::isfinite(value) || !std::isfinite(limit) || limit < 0.0) {
    return 0.0;
  }
  if (limit == 0.0) {
    return value <= 0.0 ? 1.0 : 0.0;
  }
  return smoothstep01((limit - value) / (0.5 * limit));
}

double lowerLimitBlendWeight(double value, double limit) {
  if (!std::isfinite(value) || !std::isfinite(limit) || limit < 0.0) {
    return 0.0;
  }
  if (limit == 0.0) {
    return 1.0;
  }
  return smoothstep01((value - limit) / limit);
}

std::optional<tf2::Quaternion>
shortestArcQuaternion(const tf2::Vector3 &from_value,
                      const tf2::Vector3 &to_value) {
  auto from = from_value;
  auto to = to_value;
  if (!std::isfinite(from.length2()) || !std::isfinite(to.length2()) ||
      from.length2() <= 1e-12 || to.length2() <= 1e-12) {
    return std::nullopt;
  }
  from.normalize();
  to.normalize();
  const double dot = std::clamp(from.dot(to), -1.0, 1.0);
  if (dot < -1.0 + 1e-9) {
    return std::nullopt;
  }
  const auto cross = from.cross(to);
  tf2::Quaternion result(cross.x(), cross.y(), cross.z(), 1.0 + dot);
  if (result.length2() <= 1e-12) {
    return std::nullopt;
  }
  result.normalize();
  return result;
}

} // namespace

ApriltagFusionNode::ArucoDetectorAdapter::ArucoDetectorAdapter(
    int dictionary_id, bool corner_refinement)
#if ZED_LAUNCHER_USE_MODERN_ARUCO
    : detector_(cv::aruco::getPredefinedDictionary(dictionary_id)) {
  auto detector_parameters = detector_.getDetectorParameters();
  detector_parameters.cornerRefinementMethod =
      corner_refinement ? cv::aruco::CORNER_REFINE_SUBPIX
                        : cv::aruco::CORNER_REFINE_NONE;
  detector_.setDetectorParameters(detector_parameters);
#else
    : dictionary_(cv::aruco::getPredefinedDictionary(dictionary_id)),
      detector_parameters_(cv::aruco::DetectorParameters::create()) {
  detector_parameters_->cornerRefinementMethod =
      corner_refinement ? cv::aruco::CORNER_REFINE_SUBPIX
                        : cv::aruco::CORNER_REFINE_NONE;
#endif
}

void ApriltagFusionNode::ArucoDetectorAdapter::detectMarkers(
    const cv::Mat &image, std::vector<std::vector<cv::Point2f>> &corners,
    std::vector<int> &ids) const {
#if ZED_LAUNCHER_USE_MODERN_ARUCO
  detector_.detectMarkers(image, corners, ids);
#else
  cv::aruco::detectMarkers(image, dictionary_, corners, ids,
                           detector_parameters_);
#endif
}

ApriltagFusionNode::ApriltagFusionNode(const rclcpp::NodeOptions &options,
                                       ImageInputMode image_input_mode)
    : Node("apriltag_fusion_node", options),
      image_input_mode_(image_input_mode), tf_buffer_(get_clock()),
      tf_listener_(tf_buffer_) {
  enabled_ = declare_parameter<bool>("enable_apriltag_fusion", true);
  camera_names_ = declare_parameter<std::vector<std::string>>(
      "camera_names", {"zed_left", "zed_center", "zed_right"});
  fusion_frame_id_ =
      declare_parameter<std::string>("fusion_frame_id", "fusion_world");
  tag_frame_id_ = declare_parameter<std::string>("tag_frame_id", "tag_frame");
  pose_topic_ = declare_parameter<std::string>(
      "pose_topic", "/fusion_world_pose_in_tag_frame");
  front_tag_id_ = declare_parameter<int>("front_tag_id", 0);
  back_tag_id_ = declare_parameter<int>("back_tag_id", 1);
  front_tag_size_m_ = declare_parameter<double>("front_tag_size_m", 0.12);
  back_tag_size_m_ = declare_parameter<double>("back_tag_size_m", 0.12);
  initial_tag_frame_offset_m_ =
      declare_parameter<double>("initial_tag_frame_offset_m", 0.03);
  learn_tag_separation_ = declare_parameter<bool>("learn_tag_separation", true);
  tag_separation_ema_alpha_ =
      declare_parameter<double>("tag_separation_ema_alpha", 0.05);
  tag_separation_max_innovation_m_ =
      declare_parameter<double>("tag_separation_max_innovation_m", 0.02);
  use_depth_ = declare_parameter<bool>("use_depth", true);
  depth_inner_margin_ratio_ =
      declare_parameter<double>("depth_inner_margin_ratio", 0.20);
  depth_min_valid_samples_ =
      declare_parameter<int>("depth_min_valid_samples", 25);
  depth_min_valid_fraction_ =
      declare_parameter<double>("depth_min_valid_fraction", 0.25);
  depth_plane_inlier_threshold_m_ =
      declare_parameter<double>("depth_plane_inlier_threshold_m", 0.015);
  depth_plane_max_rmse_m_ =
      declare_parameter<double>("depth_plane_max_rmse_m", 0.010);
  depth_max_pnp_translation_delta_m_ =
      declare_parameter<double>("depth_max_pnp_translation_delta_m", 0.20);
  depth_max_pnp_rotation_delta_deg_ =
      declare_parameter<double>("depth_max_pnp_rotation_delta_deg", 20.0);
  depth_max_size_error_fraction_ =
      declare_parameter<double>("depth_max_size_error_fraction", 0.25);
  pnp_ambiguity_reprojection_margin_px_ =
      declare_parameter<double>("pnp_ambiguity_reprojection_margin_px", 0.25);
  pnp_prior_max_age_sec_ =
      declare_parameter<double>("pnp_prior_max_age_sec", 0.25);
  learn_tag_transform_ = declare_parameter<bool>("learn_tag_transform", true);
  tag_transform_bootstrap_duration_sec_ =
      declare_parameter<double>("tag_transform_bootstrap_duration_sec", 2.5);
  tag_transform_bootstrap_min_samples_ =
      declare_parameter<int>("tag_transform_bootstrap_min_samples", 30);
  tag_transform_pair_max_age_sec_ =
      declare_parameter<double>("tag_transform_pair_max_age_sec", 0.10);
  tag_transform_bootstrap_translation_outlier_m_ = declare_parameter<double>(
      "tag_transform_bootstrap_translation_outlier_m", 0.03);
  tag_transform_bootstrap_rotation_outlier_deg_ = declare_parameter<double>(
      "tag_transform_bootstrap_rotation_outlier_deg", 8.0);
  tag_transform_online_alpha_ =
      declare_parameter<double>("tag_transform_online_alpha", 0.01);
  tag_transform_max_translation_step_m_ =
      declare_parameter<double>("tag_transform_max_translation_step_m", 0.002);
  tag_transform_max_rotation_step_deg_ =
      declare_parameter<double>("tag_transform_max_rotation_step_deg", 0.25);
  tag_pair_baseline_orientation_weight_ =
      declare_parameter<double>("tag_pair_baseline_orientation_weight", 0.0);
  max_detection_rate_hz_ =
      declare_parameter<double>("max_detection_rate_hz", 30.0);
  fusion_publish_rate_hz_ =
      declare_parameter<double>("apriltag_fusion_publish_rate_hz", 30.0);
  tf_lookup_timeout_sec_ =
      declare_parameter<double>("tf_lookup_timeout_sec", 0.05);
  publish_fusion_pose_ = declare_parameter<bool>("publish_fusion_pose", true);
  publish_tf_ = declare_parameter<bool>("publish_tf", true);
  publish_debug_images_ =
      declare_parameter<bool>("publish_debug_images", false);
  debug_axis_length_m_ = declare_parameter<double>("debug_axis_length_m", 0.06);
  const auto dictionary_name =
      declare_parameter<std::string>("dictionary", "APRILTAG_36h11");
  const auto frame_convention = declare_parameter<std::string>(
      "camera_frame_convention", "zed_x_forward");
  corner_refinement_ = declare_parameter<bool>("corner_refinement", true);
  max_observation_age_sec_ =
      declare_parameter<double>("max_observation_age_sec", 0.15);
  sync_tolerance_sec_ = declare_parameter<double>("sync_tolerance_sec", 0.05);
  min_marker_area_px2_ = declare_parameter<double>("min_marker_area_px2", 64.0);
  max_reprojection_rmse_px_ =
      declare_parameter<double>("max_reprojection_rmse_px", 3.0);
  max_translation_disagreement_m_ =
      declare_parameter<double>("max_translation_disagreement_m", 0.25);
  max_rotation_disagreement_deg_ =
      declare_parameter<double>("max_rotation_disagreement_deg", 25.0);
  fixed_tag_frame_z_m_ = declare_parameter<double>("fixed_tag_frame_z_m", 1.0);
  kalman_position_measurement_std_m_ =
      declare_parameter<double>("kalman_position_measurement_std_m", 0.010);
  kalman_yaw_measurement_std_deg_ =
      declare_parameter<double>("kalman_yaw_measurement_std_deg", 1.5);
  kalman_linear_acceleration_std_mps2_ =
      declare_parameter<double>("kalman_linear_acceleration_std_mps2", 1.0);
  kalman_yaw_acceleration_std_degps2_ =
      declare_parameter<double>("kalman_yaw_acceleration_std_degps2", 90.0);
  kalman_initial_linear_velocity_std_mps_ =
      declare_parameter<double>("kalman_initial_linear_velocity_std_mps", 1.0);
  kalman_initial_yaw_rate_std_degps_ =
      declare_parameter<double>("kalman_initial_yaw_rate_std_degps", 90.0);
  kalman_reset_sec_ = declare_parameter<double>("kalman_reset_sec", 0.50);

  learned_tag_separation_m_ = 2.0 * initial_tag_frame_offset_m_;
  tf2::Quaternion ideal_back_from_front;
  ideal_back_from_front.setRPY(0.0, kPi, 0.0);
  ideal_front_from_back_tag_ = tf2::Transform(
      ideal_back_from_front, tf2::Vector3(0.0, 0.0, learned_tag_separation_m_));
  learned_front_from_back_tag_ = ideal_front_from_back_tag_;

  if (!enabled_) {
    RCLCPP_INFO(get_logger(), "AprilTag fusion is disabled");
    return;
  }

  if (camera_names_.empty()) {
    throw std::runtime_error("camera_names must contain at least one camera");
  }
  std::unordered_set<std::string> unique_camera_names;
  for (const auto &name : camera_names_) {
    if (name.empty()) {
      throw std::runtime_error("camera_names must not contain empty names");
    }
    if (!unique_camera_names.insert(name).second) {
      throw std::runtime_error("camera_names contains duplicate name: " + name);
    }
  }
  if (fusion_frame_id_.empty() || tag_frame_id_.empty()) {
    throw std::runtime_error(
        "fusion_frame_id and tag_frame_id must not be empty");
  }
  if (fusion_frame_id_ == tag_frame_id_) {
    throw std::runtime_error(
        "fusion_frame_id and tag_frame_id must be different");
  }
  if (front_tag_id_ < 0 || back_tag_id_ < 0 || front_tag_id_ == back_tag_id_) {
    throw std::runtime_error(
        "front_tag_id and back_tag_id must be distinct non-negative IDs");
  }
  if (!std::isfinite(front_tag_size_m_) || front_tag_size_m_ <= 0.0 ||
      !std::isfinite(back_tag_size_m_) || back_tag_size_m_ <= 0.0) {
    throw std::runtime_error("front_tag_size_m and back_tag_size_m "
                             "must be positive");
  }
  if (!std::isfinite(initial_tag_frame_offset_m_) ||
      initial_tag_frame_offset_m_ < 0.0) {
    throw std::runtime_error("initial_tag_frame_offset_m must be non-negative");
  }
  if (!std::isfinite(tag_separation_ema_alpha_) ||
      tag_separation_ema_alpha_ <= 0.0 || tag_separation_ema_alpha_ > 1.0) {
    throw std::runtime_error("tag_separation_ema_alpha must be in (0, 1]");
  }
  if (!std::isfinite(tag_separation_max_innovation_m_) ||
      tag_separation_max_innovation_m_ < 0.0) {
    throw std::runtime_error(
        "tag_separation_max_innovation_m must be non-negative");
  }
  if (!std::isfinite(depth_inner_margin_ratio_) ||
      depth_inner_margin_ratio_ < 0.0 || depth_inner_margin_ratio_ >= 1.0) {
    throw std::runtime_error("depth_inner_margin_ratio must be in [0, 1)");
  }
  if (depth_min_valid_samples_ < 3) {
    throw std::runtime_error("depth_min_valid_samples must be at least 3");
  }
  if (!std::isfinite(depth_min_valid_fraction_) ||
      depth_min_valid_fraction_ < 0.0 || depth_min_valid_fraction_ > 1.0) {
    throw std::runtime_error("depth_min_valid_fraction must be in [0, 1]");
  }
  if (!std::isfinite(depth_plane_inlier_threshold_m_) ||
      depth_plane_inlier_threshold_m_ <= 0.0) {
    throw std::runtime_error("depth_plane_inlier_threshold_m must be positive");
  }
  if (!std::isfinite(depth_plane_max_rmse_m_) ||
      depth_plane_max_rmse_m_ <= 0.0) {
    throw std::runtime_error("depth_plane_max_rmse_m must be positive");
  }
  if (!std::isfinite(depth_max_pnp_translation_delta_m_) ||
      depth_max_pnp_translation_delta_m_ < 0.0) {
    throw std::runtime_error(
        "depth_max_pnp_translation_delta_m must be non-negative");
  }
  if (!std::isfinite(depth_max_pnp_rotation_delta_deg_) ||
      depth_max_pnp_rotation_delta_deg_ < 0.0 ||
      depth_max_pnp_rotation_delta_deg_ > 180.0) {
    throw std::runtime_error(
        "depth_max_pnp_rotation_delta_deg must be in [0, 180]");
  }
  if (!std::isfinite(depth_max_size_error_fraction_) ||
      depth_max_size_error_fraction_ < 0.0) {
    throw std::runtime_error(
        "depth_max_size_error_fraction must be non-negative");
  }
  if (!std::isfinite(pnp_ambiguity_reprojection_margin_px_) ||
      pnp_ambiguity_reprojection_margin_px_ < 0.0) {
    throw std::runtime_error(
        "pnp_ambiguity_reprojection_margin_px must be non-negative");
  }
  if (!std::isfinite(pnp_prior_max_age_sec_) || pnp_prior_max_age_sec_ < 0.0) {
    throw std::runtime_error("pnp_prior_max_age_sec must be non-negative");
  }
  if (!std::isfinite(tag_transform_bootstrap_duration_sec_) ||
      tag_transform_bootstrap_duration_sec_ < 0.0) {
    throw std::runtime_error(
        "tag_transform_bootstrap_duration_sec must be non-negative");
  }
  if (tag_transform_bootstrap_min_samples_ <= 0) {
    throw std::runtime_error(
        "tag_transform_bootstrap_min_samples must be positive");
  }
  if (!std::isfinite(tag_transform_pair_max_age_sec_) ||
      tag_transform_pair_max_age_sec_ < 0.0) {
    throw std::runtime_error(
        "tag_transform_pair_max_age_sec must be non-negative");
  }
  if (!std::isfinite(tag_transform_bootstrap_translation_outlier_m_) ||
      tag_transform_bootstrap_translation_outlier_m_ < 0.0) {
    throw std::runtime_error(
        "tag_transform_bootstrap_translation_outlier_m must be non-negative");
  }
  if (!std::isfinite(tag_transform_bootstrap_rotation_outlier_deg_) ||
      tag_transform_bootstrap_rotation_outlier_deg_ < 0.0 ||
      tag_transform_bootstrap_rotation_outlier_deg_ > 180.0) {
    throw std::runtime_error(
        "tag_transform_bootstrap_rotation_outlier_deg must be in [0, 180]");
  }
  if (!std::isfinite(tag_transform_online_alpha_) ||
      tag_transform_online_alpha_ <= 0.0 || tag_transform_online_alpha_ > 1.0) {
    throw std::runtime_error("tag_transform_online_alpha must be in (0, 1]");
  }
  if (!std::isfinite(tag_transform_max_translation_step_m_) ||
      tag_transform_max_translation_step_m_ < 0.0) {
    throw std::runtime_error(
        "tag_transform_max_translation_step_m must be non-negative");
  }
  if (!std::isfinite(tag_transform_max_rotation_step_deg_) ||
      tag_transform_max_rotation_step_deg_ < 0.0 ||
      tag_transform_max_rotation_step_deg_ > 180.0) {
    throw std::runtime_error(
        "tag_transform_max_rotation_step_deg must be in [0, 180]");
  }
  if (!std::isfinite(tag_pair_baseline_orientation_weight_) ||
      tag_pair_baseline_orientation_weight_ < 0.0 ||
      tag_pair_baseline_orientation_weight_ > 1.0) {
    throw std::runtime_error(
        "tag_pair_baseline_orientation_weight must be in [0, 1]");
  }
  if (!std::isfinite(max_detection_rate_hz_) || max_detection_rate_hz_ < 0.0) {
    throw std::runtime_error("max_detection_rate_hz must be non-negative");
  }
  if (!std::isfinite(fusion_publish_rate_hz_) ||
      fusion_publish_rate_hz_ <= 0.0) {
    throw std::runtime_error(
        "apriltag_fusion_publish_rate_hz must be positive");
  }
  if (!std::isfinite(tf_lookup_timeout_sec_) || tf_lookup_timeout_sec_ < 0.0) {
    throw std::runtime_error("tf_lookup_timeout_sec must be non-negative");
  }
  if (!std::isfinite(debug_axis_length_m_) || debug_axis_length_m_ <= 0.0) {
    throw std::runtime_error("debug_axis_length_m must be positive");
  }
  if (!std::isfinite(max_observation_age_sec_) ||
      max_observation_age_sec_ <= 0.0) {
    throw std::runtime_error("max_observation_age_sec must be positive");
  }
  if (!std::isfinite(sync_tolerance_sec_) || sync_tolerance_sec_ < 0.0) {
    throw std::runtime_error("sync_tolerance_sec must be non-negative");
  }
  if (!std::isfinite(min_marker_area_px2_) || min_marker_area_px2_ < 0.0) {
    throw std::runtime_error("min_marker_area_px2 must be non-negative");
  }
  if (!std::isfinite(max_reprojection_rmse_px_) ||
      max_reprojection_rmse_px_ <= 0.0) {
    throw std::runtime_error("max_reprojection_rmse_px must be positive");
  }
  if (!std::isfinite(max_translation_disagreement_m_) ||
      max_translation_disagreement_m_ < 0.0) {
    throw std::runtime_error(
        "max_translation_disagreement_m must be non-negative");
  }
  if (!std::isfinite(max_rotation_disagreement_deg_) ||
      max_rotation_disagreement_deg_ < 0.0 ||
      max_rotation_disagreement_deg_ > 180.0) {
    throw std::runtime_error(
        "max_rotation_disagreement_deg must be in [0, 180]");
  }
  if (!std::isfinite(fixed_tag_frame_z_m_)) {
    throw std::runtime_error("fixed_tag_frame_z_m must be finite");
  }
  if (!std::isfinite(kalman_position_measurement_std_m_) ||
      kalman_position_measurement_std_m_ <= 0.0) {
    throw std::runtime_error(
        "kalman_position_measurement_std_m must be positive");
  }
  if (!std::isfinite(kalman_yaw_measurement_std_deg_) ||
      kalman_yaw_measurement_std_deg_ <= 0.0) {
    throw std::runtime_error("kalman_yaw_measurement_std_deg must be positive");
  }
  if (!std::isfinite(kalman_linear_acceleration_std_mps2_) ||
      kalman_linear_acceleration_std_mps2_ < 0.0) {
    throw std::runtime_error(
        "kalman_linear_acceleration_std_mps2 must be non-negative");
  }
  if (!std::isfinite(kalman_yaw_acceleration_std_degps2_) ||
      kalman_yaw_acceleration_std_degps2_ < 0.0) {
    throw std::runtime_error(
        "kalman_yaw_acceleration_std_degps2 must be non-negative");
  }
  if (!std::isfinite(kalman_initial_linear_velocity_std_mps_) ||
      kalman_initial_linear_velocity_std_mps_ < 0.0) {
    throw std::runtime_error(
        "kalman_initial_linear_velocity_std_mps must be non-negative");
  }
  if (!std::isfinite(kalman_initial_yaw_rate_std_degps_) ||
      kalman_initial_yaw_rate_std_degps_ < 0.0) {
    throw std::runtime_error(
        "kalman_initial_yaw_rate_std_degps must be non-negative");
  }
  if (!std::isfinite(kalman_reset_sec_) || kalman_reset_sec_ < 0.0) {
    throw std::runtime_error("kalman_reset_sec must be non-negative");
  }
  if (!publish_fusion_pose_ && !publish_tf_ && !publish_debug_images_) {
    throw std::runtime_error("At least one output must be enabled");
  }

  camera_frame_convention_ = parseCameraFrameConvention(frame_convention);
  const int dictionary_id = parseDictionary(dictionary_name);

  if (publish_fusion_pose_) {
    pose_pub_ =
        create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, 10);
  }
  if (publish_tf_) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

  cameras_.reserve(camera_names_.size());
  for (size_t camera_index = 0; camera_index < camera_names_.size();
       ++camera_index) {
    auto camera = std::make_unique<CameraContext>();
    camera->name = camera_names_[camera_index];
    camera->image_topic = "/" + camera->name + "/zed_node/rgb/color/rect/image";
    camera->camera_info_topic =
        "/" + camera->name + "/zed_node/rgb/color/rect/camera_info";
    camera->debug_topic =
        "/" + camera->name + "/zed_node/rgb/color/rect/apriltag_overlay";
    camera->detector = std::make_unique<ArucoDetectorAdapter>(
        dictionary_id, corner_refinement_);
    if (image_input_mode_ == ImageInputMode::RosTopics) {
      camera->callback_group =
          create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

      if (publish_debug_images_) {
        camera->debug_pub = create_publisher<sensor_msgs::msg::Image>(
            camera->debug_topic, rclcpp::SensorDataQoS());
      }

      rclcpp::SubscriptionOptions subscription_options;
      subscription_options.callback_group = camera->callback_group;
      camera->camera_info_sub =
          create_subscription<sensor_msgs::msg::CameraInfo>(
              camera->camera_info_topic, rclcpp::SensorDataQoS(),
              [this,
               camera_index](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                handleCameraInfo(camera_index, std::move(msg));
              },
              subscription_options);
      camera->image_sub = create_subscription<sensor_msgs::msg::Image>(
          camera->image_topic, rclcpp::SensorDataQoS(),
          [this, camera_index](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            handleImage(camera_index, std::move(msg));
          },
          subscription_options);

      RCLCPP_INFO(get_logger(), "AprilTag input %s: %s + %s",
                  camera->name.c_str(), camera->image_topic.c_str(),
                  camera->camera_info_topic.c_str());
      if (publish_debug_images_) {
        RCLCPP_INFO(get_logger(), "AprilTag debug output %s: %s",
                    camera->name.c_str(), camera->debug_topic.c_str());
      }
    } else {
      RCLCPP_INFO(get_logger(), "AprilTag direct image input: %s",
                  camera->name.c_str());
    }
    cameras_.push_back(std::move(camera));
  }

  fusion_callback_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  const auto timer_period =
      std::chrono::duration<double>(1.0 / fusion_publish_rate_hz_);
  fusion_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
      [this]() { fuseAndPublish(); }, fusion_callback_group_);

  RCLCPP_INFO(get_logger(),
              "apriltag_fusion_node ready with %zu cameras, front id=%d, "
              "back id=%d: dynamic TF %s -> %s",
              cameras_.size(), front_tag_id_, back_tag_id_,
              tag_frame_id_.c_str(), fusion_frame_id_.c_str());
}

void ApriltagFusionNode::handleCameraInfo(
    size_t camera_index, const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  auto &camera = *cameras_.at(camera_index);
  std::lock_guard<std::mutex> lock(camera.mutex);
  camera.latest_camera_info = msg;
}

void ApriltagFusionNode::handleImage(
    size_t camera_index, const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  const DepthFrameProvider no_depth;
  auto debug_image = processImage(camera_index, *msg, no_depth, nullptr,
                                  publish_debug_images_, true);
  auto &camera = *cameras_.at(camera_index);
  if (debug_image && camera.debug_pub) {
    camera.debug_pub->publish(std::move(*debug_image));
  }
}

bool ApriltagFusionNode::shouldProcessCameraFrame(
    const std::string &camera_name) {
  if (!enabled_ || image_input_mode_ != ImageInputMode::Direct) {
    return false;
  }

  const auto camera_name_it =
      std::find(camera_names_.begin(), camera_names_.end(), camera_name);
  if (camera_name_it == camera_names_.end()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Ignoring AprilTag frame for unknown camera '%s'",
                         camera_name.c_str());
    return false;
  }
  const auto camera_index =
      static_cast<size_t>(std::distance(camera_names_.begin(), camera_name_it));
  return !shouldSkipFrame(*cameras_.at(camera_index));
}

void ApriltagFusionNode::processCameraFrame(
    const std::string &camera_name, const sensor_msgs::msg::Image &source,
    const sensor_msgs::msg::CameraInfo &camera_info,
    const DepthFrameProvider &depth_provider,
    sensor_msgs::msg::Image *debug_overlay, bool frame_was_admitted) {
  if (!enabled_) {
    return;
  }
  if (image_input_mode_ != ImageInputMode::Direct) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Ignoring direct AprilTag frame for %s while using ROS topic input",
        camera_name.c_str());
    return;
  }

  const auto camera_name_it =
      std::find(camera_names_.begin(), camera_names_.end(), camera_name);
  if (camera_name_it == camera_names_.end()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Ignoring AprilTag frame for unknown camera '%s'",
                         camera_name.c_str());
    return;
  }
  const auto camera_index =
      static_cast<size_t>(std::distance(camera_names_.begin(), camera_name_it));
  auto &camera = *cameras_.at(camera_index);
  {
    std::lock_guard<std::mutex> lock(camera.mutex);
    camera.latest_camera_info =
        std::make_shared<sensor_msgs::msg::CameraInfo>(camera_info);
  }

  auto rendered =
      processImage(camera_index, source, depth_provider, debug_overlay,
                   debug_overlay != nullptr, !frame_was_admitted);
  if (debug_overlay && rendered) {
    *debug_overlay = std::move(*rendered);
  }
}

std::optional<sensor_msgs::msg::Image>
ApriltagFusionNode::processImage(size_t camera_index,
                                 const sensor_msgs::msg::Image &source,
                                 const DepthFrameProvider &depth_provider,
                                 const sensor_msgs::msg::Image *debug_base,
                                 bool render_debug, bool apply_rate_limit) {
  if (!enabled_) {
    return std::nullopt;
  }

  auto &camera = *cameras_.at(camera_index);
  const auto current_time = now();
  if (apply_rate_limit && shouldSkipFrame(camera)) {
    return std::nullopt;
  }

  cv::Mat gray;
  if (!toGrayImage(source, gray)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Unsupported or invalid image on %s: encoding=%s",
                         camera.image_topic.c_str(), source.encoding.c_str());
    return std::nullopt;
  }

  cv::Mat debug_image;
  if (render_debug) {
    const auto &debug_source = debug_base ? *debug_base : source;
    try {
      if (!toBgrImage(debug_source, debug_image)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Cannot convert debug image on %s",
                             camera.image_topic.c_str());
      }
    } catch (const cv::Exception &ex) {
      debug_image.release();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Cannot convert debug image on %s: %s",
                           camera.image_topic.c_str(), ex.what());
    }
  }

  const auto renderedDebugImage =
      [this, &source,
       &debug_image]() -> std::optional<sensor_msgs::msg::Image> {
    if (debug_image.empty()) {
      return std::nullopt;
    }
    return makeDebugImage(source, debug_image);
  };

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  try {
    camera.detector->detectMarkers(gray, corners, ids);
  } catch (const cv::Exception &ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "AprilTag detection failed on %s: %s",
                         camera.name.c_str(), ex.what());
    return renderedDebugImage();
  }

  if (!debug_image.empty()) {
    for (size_t index = 0; index < ids.size() && index < corners.size();
         ++index) {
      if (ids[index] != front_tag_id_ && ids[index] != back_tag_id_) {
        continue;
      }
      try {
        if (!drawDebugMarkerOutline(debug_image, corners[index])) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "Cannot draw AprilTag debug outline for id=%d on %s: "
              "invalid corners",
              ids[index], camera.name.c_str());
          continue;
        }
        const auto &corner = corners[index].front();
        cv::putText(debug_image, "id " + std::to_string(ids[index]),
                    cv::Point(cvRound(corner.x) + 4, cvRound(corner.y) - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2,
                    cv::LINE_AA);
      } catch (const cv::Exception &ex) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Cannot draw AprilTag debug outline for id=%d on %s: %s",
            ids[index], camera.name.c_str(), ex.what());
      }
    }
  }

  const auto selected = selectMarkers(ids, corners);
  if (!selected[kFrontTagSlot] && !selected[kBackTagSlot]) {
    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                          "AprilTags id=%d/id=%d not detected by %s",
                          front_tag_id_, back_tag_id_, camera.name.c_str());
    return renderedDebugImage();
  }

  sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
  std::array<std::optional<PoseEstimate>, kTagCount> pnp_priors;
  {
    std::lock_guard<std::mutex> lock(camera.mutex);
    camera_info = camera.latest_camera_info;
    for (size_t tag_slot = 0; tag_slot < kTagCount; ++tag_slot) {
      auto &timed_estimate = camera.latest_pose_estimates[tag_slot];
      if (!timed_estimate) {
        continue;
      }
      const double age =
          (current_time - timed_estimate->receipt_time).seconds();
      if (age >= 0.0 && age <= pnp_prior_max_age_sec_) {
        pnp_priors[tag_slot] = timed_estimate->estimate;
      } else {
        timed_estimate.reset();
      }
    }
  }
  if (!camera_info) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Waiting for camera info on %s",
                         camera.camera_info_topic.c_str());
    return renderedDebugImage();
  }

  const auto camera_matrix = cameraMatrixForImage(*camera_info, source);
  if (!camera_matrix) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Camera info on %s has invalid intrinsics",
                         camera.camera_info_topic.c_str());
    return renderedDebugImage();
  }
  const auto distortion_coeffs = distortionCoeffs(*camera_info);

  std::optional<DepthFrameView> depth_frame;
  if (use_depth_ && depth_provider) {
    try {
      depth_frame = depth_provider();
      if (depth_frame && !static_cast<bool>(*depth_frame)) {
        depth_frame.reset();
      }
    } catch (const std::exception &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Depth retrieval failed for %s: %s",
                           camera.name.c_str(), ex.what());
    }
  }

  std::array<std::optional<PoseEstimate>, kTagCount> pose_estimates;
  std::array<double, kTagCount> marker_areas{};
  for (size_t tag_slot = 0; tag_slot < kTagCount; ++tag_slot) {
    if (!selected[tag_slot]) {
      continue;
    }
    const auto marker_index = *selected[tag_slot];
    marker_areas[tag_slot] = markerArea(corners[marker_index]);
    if (!std::isfinite(marker_areas[tag_slot]) ||
        marker_areas[tag_slot] < min_marker_area_px2_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Rejected small AprilTag id=%d from %s: area=%.2f px^2",
          tagId(tag_slot), camera.name.c_str(), marker_areas[tag_slot]);
      continue;
    }

    try {
      pose_estimates[tag_slot] = estimateTagInCameraFrame(
          corners[marker_index], *camera_matrix, distortion_coeffs,
          tagSize(tag_slot), depth_frame ? &*depth_frame : nullptr,
          source.width, source.height,
          pnp_priors[tag_slot] ? &*pnp_priors[tag_slot] : nullptr);
    } catch (const cv::Exception &ex) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "AprilTag pose estimation failed for id=%d on %s: %s",
          tagId(tag_slot), camera.name.c_str(), ex.what());
      continue;
    }
    if (!pose_estimates[tag_slot]) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "solvePnP failed for AprilTag id=%d from %s",
                           tagId(tag_slot), camera.name.c_str());
      continue;
    }
    if (!std::isfinite(pose_estimates[tag_slot]->reprojection_rmse_px) ||
        pose_estimates[tag_slot]->reprojection_rmse_px >
            max_reprojection_rmse_px_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Rejected AprilTag id=%d from %s: reprojection RMSE=%.3f px",
          tagId(tag_slot), camera.name.c_str(),
          pose_estimates[tag_slot]->reprojection_rmse_px);
      pose_estimates[tag_slot].reset();
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(camera.mutex);
      camera.latest_pose_estimates[tag_slot] =
          TimedPoseEstimate{*pose_estimates[tag_slot], current_time};
    }
    if (pose_estimates[tag_slot]->used_depth) {
      RCLCPP_DEBUG(get_logger(),
                   "Depth-assisted AprilTag id=%d on %s: samples=%zu/%zu "
                   "valid=%.2f plane_rmse=%.4fm size_error=%.3f "
                   "pnp_delta=%.3fm/%.2fdeg blend_t=%.2f blend_r=%.2f "
                   "weight=%.2f",
                   tagId(tag_slot), camera.name.c_str(),
                   pose_estimates[tag_slot]->depth_inlier_samples,
                   pose_estimates[tag_slot]->depth_valid_samples,
                   pose_estimates[tag_slot]->depth_valid_fraction,
                   pose_estimates[tag_slot]->depth_plane_rmse_m,
                   pose_estimates[tag_slot]->depth_size_error_fraction,
                   pose_estimates[tag_slot]->depth_pnp_translation_delta_m,
                   pose_estimates[tag_slot]->depth_pnp_rotation_delta_deg,
                   pose_estimates[tag_slot]->depth_blend_weight,
                   pose_estimates[tag_slot]->depth_rotation_blend_weight,
                   pose_estimates[tag_slot]->estimator_weight);
    }

    if (!debug_image.empty()) {
      const auto &label_corner = corners[marker_index].front();
      const std::string estimator_label =
          pose_estimates[tag_slot]->used_depth
              ? cv::format(
                    "DEPTH T%.0f%% R%.0f%% %zu/%zu rms=%.1fmm",
                    100.0 * pose_estimates[tag_slot]->depth_blend_weight,
                    100.0 *
                        pose_estimates[tag_slot]->depth_rotation_blend_weight,
                    pose_estimates[tag_slot]->depth_inlier_samples,
                    pose_estimates[tag_slot]->depth_valid_samples,
                    1000.0 * pose_estimates[tag_slot]->depth_plane_rmse_m)
              : "PNP";
      try {
        cv::putText(debug_image, estimator_label,
                    cv::Point(cvRound(label_corner.x) + 4,
                              cvRound(label_corner.y) + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1,
                    cv::LINE_AA);
        cv::drawFrameAxes(debug_image, *camera_matrix, distortion_coeffs,
                          pose_estimates[tag_slot]->rvec,
                          pose_estimates[tag_slot]->tvec,
                          static_cast<float>(debug_axis_length_m_), 2);
      } catch (const cv::Exception &ex) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Cannot draw AprilTag debug axes for id=%d on %s: %s",
            tagId(tag_slot), camera.name.c_str(), ex.what());
      }
    }
  }

  auto debug_result = renderedDebugImage();

  if (!pose_estimates[kFrontTagSlot] && !pose_estimates[kBackTagSlot]) {
    return debug_result;
  }

  const auto camera_frame_id = !source.header.frame_id.empty()
                                   ? source.header.frame_id
                                   : camera_info->header.frame_id;
  if (camera_frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Image and camera_info from %s have empty frame_id",
                         camera.name.c_str());
    return debug_result;
  }

  geometry_msgs::msg::TransformStamped fusion_from_camera_msg;
  try {
    fusion_from_camera_msg = tf_buffer_.lookupTransform(
        fusion_frame_id_, camera_frame_id, tf2::TimePointZero,
        tf2::durationFromSec(tf_lookup_timeout_sec_));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Cannot transform %s -> %s for AprilTag detection from %s: %s",
        camera_frame_id.c_str(), fusion_frame_id_.c_str(), camera.name.c_str(),
        ex.what());
    return debug_result;
  }

  const auto fusion_from_camera =
      transformFromMsg(fusion_from_camera_msg.transform);
  const auto source_stamp =
      rclcpp::Time(source.header.stamp, get_clock()->get_clock_type());
  std::array<std::optional<Observation>, kTagCount> new_observations;
  for (size_t tag_slot = 0; tag_slot < kTagCount; ++tag_slot) {
    if (!pose_estimates[tag_slot]) {
      continue;
    }
    Observation observation;
    observation.fusion_from_tag =
        fusion_from_camera * pose_estimates[tag_slot]->camera_from_tag;
    observation.stamp = source_stamp;
    observation.receipt_time = current_time;
    observation.marker_area_px2 = marker_areas[tag_slot];
    observation.reprojection_rmse_px =
        pose_estimates[tag_slot]->reprojection_rmse_px;
    observation.quality =
        marker_areas[tag_slot] /
        std::max(pose_estimates[tag_slot]->reprojection_rmse_px *
                     pose_estimates[tag_slot]->reprojection_rmse_px,
                 0.25) *
        pose_estimates[tag_slot]->estimator_weight;
    observation.sequence = next_observation_sequence_.fetch_add(1);
    new_observations[tag_slot] = std::move(observation);
  }

  {
    std::lock_guard<std::mutex> lock(camera.mutex);
    for (size_t tag_slot = 0; tag_slot < kTagCount; ++tag_slot) {
      if (new_observations[tag_slot]) {
        camera.latest_observations[tag_slot] =
            std::move(new_observations[tag_slot]);
      }
    }
  }
  return debug_result;
}

bool ApriltagFusionNode::shouldSkipFrame(CameraContext &camera) {
  if (max_detection_rate_hz_ == 0.0) {
    return false;
  }

  const auto current_time = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(camera.mutex);
  if (camera.has_last_detection_attempt_time) {
    const double elapsed =
        std::chrono::duration<double>(current_time -
                                      camera.last_detection_attempt_time)
            .count();
    if (elapsed < 1.0 / max_detection_rate_hz_) {
      return true;
    }
  }

  camera.last_detection_attempt_time = current_time;
  camera.has_last_detection_attempt_time = true;
  return false;
}

std::optional<ApriltagFusionNode::FusedTagEstimate>
ApriltagFusionNode::fuseTag(size_t tag_slot,
                            const rclcpp::Time &current_time) const {
  std::vector<Observation> observations;
  observations.reserve(cameras_.size());
  for (const auto &camera_ptr : cameras_) {
    std::lock_guard<std::mutex> lock(camera_ptr->mutex);
    const auto &latest = camera_ptr->latest_observations.at(tag_slot);
    if (!latest) {
      continue;
    }
    const double age = (current_time - latest->receipt_time).seconds();
    if (age >= 0.0 && age <= max_observation_age_sec_) {
      observations.push_back(*latest);
    }
  }
  if (observations.empty()) {
    return std::nullopt;
  }

  const auto newest_stamp =
      std::max_element(observations.begin(), observations.end(),
                       [](const Observation &lhs, const Observation &rhs) {
                         return lhs.stamp < rhs.stamp;
                       })
          ->stamp;
  observations.erase(
      std::remove_if(observations.begin(), observations.end(),
                     [this, &newest_stamp](const Observation &observation) {
                       return (newest_stamp - observation.stamp).seconds() >
                              sync_tolerance_sec_;
                     }),
      observations.end());
  if (observations.empty()) {
    return std::nullopt;
  }

  std::vector<size_t> best_cluster;
  std::vector<size_t> current_cluster;
  double best_quality = -1.0;
  std::function<void(size_t, double)> find_best_cluster;
  find_best_cluster = [&](size_t next_index, double cluster_quality) {
    if (current_cluster.size() > best_cluster.size() ||
        (current_cluster.size() == best_cluster.size() &&
         cluster_quality > best_quality)) {
      best_cluster = current_cluster;
      best_quality = cluster_quality;
    }
    if (next_index >= observations.size() ||
        current_cluster.size() + observations.size() - next_index <
            best_cluster.size()) {
      return;
    }

    bool agrees_with_cluster = true;
    for (const auto selected_index : current_cluster) {
      if (!observationsAgree(observations[next_index],
                             observations[selected_index])) {
        agrees_with_cluster = false;
        break;
      }
    }
    if (agrees_with_cluster) {
      current_cluster.push_back(next_index);
      find_best_cluster(next_index + 1,
                        cluster_quality + observations[next_index].quality);
      current_cluster.pop_back();
    }
    find_best_cluster(next_index + 1, cluster_quality);
  };
  find_best_cluster(0, 0.0);
  if (best_cluster.empty()) {
    return std::nullopt;
  }

  FusedTagEstimate estimate;
  estimate.observation = fuseObservations(observations, best_cluster);
  estimate.source_sequences.reserve(best_cluster.size());
  for (const auto index : best_cluster) {
    estimate.source_sequences.push_back(observations[index].sequence);
  }
  std::sort(estimate.source_sequences.begin(), estimate.source_sequences.end());
  return estimate;
}

void ApriltagFusionNode::fuseAndPublish() {
  const auto current_time = now();
  auto front = fuseTag(kFrontTagSlot, current_time);
  auto back = fuseTag(kBackTagSlot, current_time);

  if (front) {
    last_tag_estimates_[kFrontTagSlot] = front;
  }
  if (back) {
    last_tag_estimates_[kBackTagSlot] = back;
  }

  tf2::Transform fusion_from_tag_frame;
  rclcpp::Time source_stamp;
  std::vector<uint64_t> source_sequences;
  bool using_stale_fallback = false;

  const bool synchronized_pair =
      front && back &&
      std::abs(
          (front->observation.stamp - back->observation.stamp).seconds()) <=
          sync_tolerance_sec_;
  if (front && back) {
    updateTagTransform(*front, *back);
  }
  bool relation_consistent_pair = false;
  if (synchronized_pair) {
    relation_consistent_pair = !tag_transform_calibrated_;
    if (tag_transform_calibrated_) {
      const auto center_from_front = tagFrameFromSingleTag(
          kFrontTagSlot, front->observation.fusion_from_tag);
      const auto center_from_back = tagFrameFromSingleTag(
          kBackTagSlot, back->observation.fusion_from_tag);
      relation_consistent_pair =
          center_from_front.getOrigin().distance(
              center_from_back.getOrigin()) <=
              tag_transform_bootstrap_translation_outlier_m_ &&
          rotationDistanceDegrees(center_from_front.getRotation(),
                                  center_from_back.getRotation()) <=
              tag_transform_bootstrap_rotation_outlier_deg_;
    }
  }
  if (synchronized_pair && relation_consistent_pair) {
    updateTagSeparation(*front, *back);
    const double pair_stamp_delta_sec = std::abs(
        (front->observation.stamp - back->observation.stamp).seconds());
    const double baseline_orientation_weight =
        tag_pair_baseline_orientation_weight_ *
        upperLimitBlendWeight(pair_stamp_delta_sec, sync_tolerance_sec_);
    fusion_from_tag_frame = tagFrameFromBothTags(
        front->observation.fusion_from_tag, back->observation.fusion_from_tag,
        front->observation.quality, back->observation.quality,
        baseline_orientation_weight);
    source_stamp = std::max(front->observation.stamp, back->observation.stamp);
    source_sequences = front->source_sequences;
    source_sequences.insert(source_sequences.end(),
                            back->source_sequences.begin(),
                            back->source_sequences.end());
  } else if (front || back) {
    size_t selected_slot = kFrontTagSlot;
    const FusedTagEstimate *selected = front ? &*front : &*back;
    if (front && back) {
      if (synchronized_pair && !relation_consistent_pair) {
        if (back->observation.quality > front->observation.quality) {
          selected_slot = kBackTagSlot;
          selected = &*back;
        }
        RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Rejected inconsistent front/back AprilTag pair; using id=%d",
            tagId(selected_slot));
      } else if (back->observation.stamp > front->observation.stamp) {
        selected_slot = kBackTagSlot;
        selected = &*back;
      }
    } else if (!front) {
      selected_slot = kBackTagSlot;
    }
    fusion_from_tag_frame = tagFrameFromSingleTag(
        selected_slot, selected->observation.fusion_from_tag);
    source_stamp = selected->observation.stamp;
    source_sequences = selected->source_sequences;
  } else {
    const auto selected_slot = selectFallbackTagSlot();
    if (!selected_slot) {
      return;
    }
    const auto &selected = *last_tag_estimates_[*selected_slot];
    fusion_from_tag_frame = tagFrameFromSingleTag(
        *selected_slot, selected.observation.fusion_from_tag);
    source_stamp = current_time;
    using_stale_fallback = true;
  }

  if (!finiteTransform(fusion_from_tag_frame)) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Rejected non-finite fused AprilTag transform before Kalman update");
    return;
  }

  std::sort(source_sequences.begin(), source_sequences.end());
  if (!using_stale_fallback && source_sequences == last_published_sequences_) {
    return;
  }

  const auto filtered =
      using_stale_fallback && has_kalman_state_
          ? kalmanFilteredTransform()
          : filterTransform(fusion_from_tag_frame, current_time);
  publishResult(filtered, source_stamp);

  if (!using_stale_fallback) {
    last_published_sequences_ = std::move(source_sequences);
  }
}

bool ApriltagFusionNode::observationsAgree(const Observation &lhs,
                                           const Observation &rhs) const {
  const double translation_distance =
      lhs.fusion_from_tag.getOrigin().distance(rhs.fusion_from_tag.getOrigin());
  if (translation_distance > max_translation_disagreement_m_) {
    return false;
  }

  auto lhs_rotation = lhs.fusion_from_tag.getRotation();
  auto rhs_rotation = rhs.fusion_from_tag.getRotation();
  lhs_rotation.normalize();
  rhs_rotation.normalize();
  const double quaternion_dot =
      std::clamp(std::abs(lhs_rotation.dot(rhs_rotation)), 0.0, 1.0);
  const double rotation_difference_deg =
      2.0 * std::acos(quaternion_dot) * kRadiansToDegrees;
  return rotation_difference_deg <= max_rotation_disagreement_deg_;
}

ApriltagFusionNode::Observation ApriltagFusionNode::fuseObservations(
    const std::vector<Observation> &observations,
    const std::vector<size_t> &selected_indices) const {
  const auto first_index = selected_indices.front();
  auto reference_rotation =
      observations[first_index].fusion_from_tag.getRotation();
  reference_rotation.normalize();

  tf2::Vector3 weighted_translation(0.0, 0.0, 0.0);
  double weighted_qx = 0.0;
  double weighted_qy = 0.0;
  double weighted_qz = 0.0;
  double weighted_qw = 0.0;
  double total_weight = 0.0;

  Observation fused;
  fused.stamp = observations[first_index].stamp;
  fused.receipt_time = observations[first_index].receipt_time;
  for (const auto index : selected_indices) {
    const auto &observation = observations[index];
    const double weight =
        std::isfinite(observation.quality) && observation.quality > 0.0
            ? observation.quality
            : 1.0;
    weighted_translation += observation.fusion_from_tag.getOrigin() * weight;

    auto rotation = observation.fusion_from_tag.getRotation();
    rotation.normalize();
    if (reference_rotation.dot(rotation) < 0.0) {
      rotation = tf2::Quaternion(-rotation.x(), -rotation.y(), -rotation.z(),
                                 -rotation.w());
    }
    weighted_qx += rotation.x() * weight;
    weighted_qy += rotation.y() * weight;
    weighted_qz += rotation.z() * weight;
    weighted_qw += rotation.w() * weight;
    total_weight += weight;

    fused.stamp = std::max(fused.stamp, observation.stamp);
    fused.receipt_time = std::max(fused.receipt_time, observation.receipt_time);
    fused.quality += weight;
    fused.marker_area_px2 += observation.marker_area_px2;
    fused.reprojection_rmse_px += observation.reprojection_rmse_px * weight;
    fused.sequence = std::max(fused.sequence, observation.sequence);
  }

  weighted_translation /= total_weight;
  tf2::Quaternion weighted_rotation(
      weighted_qx / total_weight, weighted_qy / total_weight,
      weighted_qz / total_weight, weighted_qw / total_weight);
  if (weighted_rotation.length2() <= 0.0) {
    weighted_rotation = reference_rotation;
  } else {
    weighted_rotation.normalize();
  }
  fused.fusion_from_tag =
      tf2::Transform(weighted_rotation, weighted_translation);
  fused.reprojection_rmse_px /= total_weight;
  return fused;
}

tf2::Transform ApriltagFusionNode::tagFrameFromSingleTag(
    size_t tag_slot, const tf2::Transform &fusion_from_tag) const {
  const auto front_from_back = activeFrontFromBackTag();
  tf2::Transform fusion_from_front;
  tf2::Transform fusion_from_back;
  if (tag_slot == kFrontTagSlot) {
    fusion_from_front = fusion_from_tag;
    fusion_from_back = fusion_from_front * front_from_back;
  } else {
    fusion_from_back = fusion_from_tag;
    fusion_from_front = fusion_from_back * front_from_back.inverse();
  }
  const auto origin =
      (fusion_from_front.getOrigin() + fusion_from_back.getOrigin()) * 0.5;
  auto rotation = fusion_from_back.getRotation();
  rotation.normalize();
  return tf2::Transform(rotation, origin);
}

tf2::Transform ApriltagFusionNode::tagFrameFromBothTags(
    const tf2::Transform &fusion_from_front_tag,
    const tf2::Transform &fusion_from_back_tag, double front_quality,
    double back_quality, double baseline_orientation_weight) const {
  const auto center_from_front =
      tagFrameFromSingleTag(kFrontTagSlot, fusion_from_front_tag);
  const auto center_from_back =
      tagFrameFromSingleTag(kBackTagSlot, fusion_from_back_tag);
  const double front_weight =
      std::isfinite(front_quality) && front_quality > 0.0 ? front_quality : 1.0;
  const double back_weight =
      std::isfinite(back_quality) && back_quality > 0.0 ? back_quality : 1.0;
  const double total_weight = front_weight + back_weight;
  const auto origin =
      (fusion_from_front_tag.getOrigin() + fusion_from_back_tag.getOrigin()) *
      0.5;

  auto front_rotation = center_from_front.getRotation();
  auto back_rotation = center_from_back.getRotation();
  front_rotation.normalize();
  back_rotation.normalize();
  if (front_rotation.dot(back_rotation) < 0.0) {
    back_rotation = tf2::Quaternion(-back_rotation.x(), -back_rotation.y(),
                                    -back_rotation.z(), -back_rotation.w());
  }
  auto rotation = tf2::Quaternion(
      (front_rotation.x() * front_weight + back_rotation.x() * back_weight) /
          total_weight,
      (front_rotation.y() * front_weight + back_rotation.y() * back_weight) /
          total_weight,
      (front_rotation.z() * front_weight + back_rotation.z() * back_weight) /
          total_weight,
      (front_rotation.w() * front_weight + back_rotation.w() * back_weight) /
          total_weight);
  if (rotation.length2() <= 0.0) {
    rotation = back_rotation;
  } else {
    rotation.normalize();
  }

  const auto front_from_back = activeFrontFromBackTag();
  const auto expected_back_from_front = front_from_back.inverse().getOrigin();
  const auto measured_back_from_front =
      fusion_from_front_tag.getOrigin() - fusion_from_back_tag.getOrigin();
  const double expected_length = expected_back_from_front.length();
  const double measured_length = measured_back_from_front.length();
  if (std::isfinite(baseline_orientation_weight) &&
      baseline_orientation_weight > 0.0 && std::isfinite(expected_length) &&
      expected_length > 1e-6 && std::isfinite(measured_length) &&
      measured_length > 1e-6) {
    const auto predicted_back_from_front =
        tf2::Matrix3x3(rotation) * expected_back_from_front;
    const auto correction = shortestArcQuaternion(predicted_back_from_front,
                                                  measured_back_from_front);
    if (correction) {
      const double length_error = std::abs(measured_length - expected_length);
      const double length_error_limit =
          std::max(tag_transform_bootstrap_translation_outlier_m_,
                   0.25 * expected_length);
      const double axis_error_deg =
          rotationDistanceDegrees(tf2::Quaternion::getIdentity(), *correction);
      const double correction_weight =
          std::clamp(baseline_orientation_weight, 0.0, 1.0) *
          upperLimitBlendWeight(length_error, length_error_limit) *
          upperLimitBlendWeight(axis_error_deg, max_rotation_disagreement_deg_);
      auto partial_correction =
          tf2::Quaternion::getIdentity().slerp(*correction, correction_weight);
      partial_correction.normalize();
      rotation = partial_correction * rotation;
      rotation.normalize();
    }
  }
  return tf2::Transform(rotation, origin);
}

tf2::Quaternion
ApriltagFusionNode::blendTagNormal(const tf2::Quaternion &pnp_rotation_value,
                                   const tf2::Quaternion &depth_rotation_value,
                                   double depth_weight) {
  auto pnp_rotation = pnp_rotation_value;
  auto depth_rotation = depth_rotation_value;
  if (!std::isfinite(depth_weight) || pnp_rotation.length2() <= 1e-12 ||
      depth_rotation.length2() <= 1e-12) {
    return pnp_rotation;
  }
  pnp_rotation.normalize();
  depth_rotation.normalize();
  const auto pnp_normal =
      tf2::Matrix3x3(pnp_rotation) * tf2::Vector3(0.0, 0.0, 1.0);
  const auto depth_normal =
      tf2::Matrix3x3(depth_rotation) * tf2::Vector3(0.0, 0.0, 1.0);
  const auto aligned_depth_normal =
      pnp_normal.dot(depth_normal) < 0.0 ? -depth_normal : depth_normal;
  const auto full_correction =
      shortestArcQuaternion(pnp_normal, aligned_depth_normal);
  if (!full_correction) {
    return pnp_rotation;
  }
  auto partial_correction = tf2::Quaternion::getIdentity().slerp(
      *full_correction, std::clamp(depth_weight, 0.0, 1.0));
  partial_correction.normalize();
  auto result = partial_correction * pnp_rotation;
  result.normalize();
  return result;
}

tf2::Transform ApriltagFusionNode::activeFrontFromBackTag() const {
  return tag_transform_calibrated_ ? learned_front_from_back_tag_
                                   : ideal_front_from_back_tag_;
}

std::optional<tf2::Transform> ApriltagFusionNode::robustAverageTagTransforms(
    const std::vector<tf2::Transform> &samples) const {
  std::vector<const tf2::Transform *> finite_samples;
  finite_samples.reserve(samples.size());
  for (const auto &sample : samples) {
    if (finiteTransform(sample)) {
      finite_samples.push_back(&sample);
    }
  }
  if (finite_samples.empty()) {
    return std::nullopt;
  }

  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<double> zs;
  xs.reserve(finite_samples.size());
  ys.reserve(finite_samples.size());
  zs.reserve(finite_samples.size());
  for (const auto *sample : finite_samples) {
    xs.push_back(sample->getOrigin().x());
    ys.push_back(sample->getOrigin().y());
    zs.push_back(sample->getOrigin().z());
  }
  const tf2::Vector3 median_translation(median(xs), median(ys), median(zs));

  // Sign-align the quaternion samples, then use a component-wise median as a
  // robust O(N log N) rotation center.  A full quaternion medoid is O(N^2)
  // and can otherwise become expensive while a bootstrap window is waiting
  // for enough mutually consistent observations.
  auto sign_reference = finite_samples.front()->getRotation();
  sign_reference.normalize();
  std::vector<double> quaternion_xs;
  std::vector<double> quaternion_ys;
  std::vector<double> quaternion_zs;
  std::vector<double> quaternion_ws;
  quaternion_xs.reserve(finite_samples.size());
  quaternion_ys.reserve(finite_samples.size());
  quaternion_zs.reserve(finite_samples.size());
  quaternion_ws.reserve(finite_samples.size());
  for (const auto *sample : finite_samples) {
    auto rotation = sample->getRotation();
    rotation.normalize();
    if (sign_reference.dot(rotation) < 0.0) {
      rotation = tf2::Quaternion(-rotation.x(), -rotation.y(), -rotation.z(),
                                 -rotation.w());
    }
    quaternion_xs.push_back(rotation.x());
    quaternion_ys.push_back(rotation.y());
    quaternion_zs.push_back(rotation.z());
    quaternion_ws.push_back(rotation.w());
  }
  auto reference_rotation =
      tf2::Quaternion(median(quaternion_xs), median(quaternion_ys),
                      median(quaternion_zs), median(quaternion_ws));
  if (reference_rotation.length2() <= 0.0) {
    reference_rotation = sign_reference;
  } else {
    reference_rotation.normalize();
  }

  tf2::Vector3 translation_sum(0.0, 0.0, 0.0);
  double quaternion_x = 0.0;
  double quaternion_y = 0.0;
  double quaternion_z = 0.0;
  double quaternion_w = 0.0;
  std::size_t inlier_count = 0;
  for (const auto *sample : finite_samples) {
    if (sample->getOrigin().distance(median_translation) >
            tag_transform_bootstrap_translation_outlier_m_ ||
        rotationDistanceDegrees(sample->getRotation(), reference_rotation) >
            tag_transform_bootstrap_rotation_outlier_deg_) {
      continue;
    }
    translation_sum += sample->getOrigin();
    auto rotation = sample->getRotation();
    rotation.normalize();
    if (reference_rotation.dot(rotation) < 0.0) {
      rotation = tf2::Quaternion(-rotation.x(), -rotation.y(), -rotation.z(),
                                 -rotation.w());
    }
    quaternion_x += rotation.x();
    quaternion_y += rotation.y();
    quaternion_z += rotation.z();
    quaternion_w += rotation.w();
    ++inlier_count;
  }
  if (inlier_count == 0) {
    return std::nullopt;
  }

  auto average_rotation =
      tf2::Quaternion(quaternion_x / inlier_count, quaternion_y / inlier_count,
                      quaternion_z / inlier_count, quaternion_w / inlier_count);
  if (average_rotation.length2() <= 0.0) {
    average_rotation = reference_rotation;
  } else {
    average_rotation.normalize();
  }
  return tf2::Transform(average_rotation,
                        translation_sum / static_cast<double>(inlier_count));
}

void ApriltagFusionNode::updateTagTransform(const FusedTagEstimate &front,
                                            const FusedTagEstimate &back) {
  if (!learn_tag_transform_) {
    return;
  }

  std::vector<uint64_t> pair_sequences = front.source_sequences;
  pair_sequences.insert(pair_sequences.end(), back.source_sequences.begin(),
                        back.source_sequences.end());
  std::sort(pair_sequences.begin(), pair_sequences.end());
  if (pair_sequences == last_tag_transform_update_sequences_) {
    return;
  }
  last_tag_transform_update_sequences_ = std::move(pair_sequences);

  const double pair_age_sec =
      std::abs((front.observation.stamp - back.observation.stamp).seconds());
  if (!std::isfinite(pair_age_sec) ||
      pair_age_sec > tag_transform_pair_max_age_sec_) {
    return;
  }
  const auto sample = front.observation.fusion_from_tag.inverse() *
                      back.observation.fusion_from_tag;
  if (!finiteTransform(sample)) {
    return;
  }
  const auto sample_time =
      std::max(front.observation.receipt_time, back.observation.receipt_time);

  if (!tag_transform_calibrated_) {
    const auto minimum_samples =
        static_cast<std::size_t>(tag_transform_bootstrap_min_samples_);
    const std::size_t bootstrap_sample_limit =
        std::max<std::size_t>(120, minimum_samples * 4);
    if (tag_transform_bootstrap_samples_.size() >= bootstrap_sample_limit) {
      tag_transform_bootstrap_samples_.clear();
      has_tag_transform_bootstrap_start_time_ = false;
      RCLCPP_WARN(get_logger(),
                  "Resetting tag-transform bootstrap after %zu samples "
                  "without enough consistent inliers",
                  bootstrap_sample_limit);
    }
    if (!has_tag_transform_bootstrap_start_time_) {
      tag_transform_bootstrap_start_time_ = sample_time;
      has_tag_transform_bootstrap_start_time_ = true;
    }
    tag_transform_bootstrap_samples_.push_back(sample);

    const double elapsed_sec =
        (sample_time - tag_transform_bootstrap_start_time_).seconds();
    if (elapsed_sec < tag_transform_bootstrap_duration_sec_ ||
        tag_transform_bootstrap_samples_.size() <
            static_cast<std::size_t>(tag_transform_bootstrap_min_samples_)) {
      return;
    }
    const auto average =
        robustAverageTagTransforms(tag_transform_bootstrap_samples_);
    if (!average) {
      return;
    }
    std::size_t inlier_count = 0;
    for (const auto &candidate : tag_transform_bootstrap_samples_) {
      if (candidate.getOrigin().distance(average->getOrigin()) <=
              tag_transform_bootstrap_translation_outlier_m_ &&
          rotationDistanceDegrees(candidate.getRotation(),
                                  average->getRotation()) <=
              tag_transform_bootstrap_rotation_outlier_deg_) {
        ++inlier_count;
      }
    }
    if (inlier_count <
        static_cast<std::size_t>(tag_transform_bootstrap_min_samples_)) {
      return;
    }

    learned_front_from_back_tag_ = *average;
    tag_transform_calibrated_ = true;
    learned_tag_separation_m_ =
        learned_front_from_back_tag_.getOrigin().length();
    tag_transform_bootstrap_samples_.clear();
    RCLCPP_INFO(
        get_logger(),
        "Calibrated full front-from-back tag transform from %zu inliers "
        "after %.2fs",
        inlier_count, elapsed_sec);
    return;
  }

  const double translation_error =
      learned_front_from_back_tag_.getOrigin().distance(sample.getOrigin());
  const double rotation_error = rotationDistanceDegrees(
      learned_front_from_back_tag_.getRotation(), sample.getRotation());
  if (translation_error > tag_transform_bootstrap_translation_outlier_m_ ||
      rotation_error > tag_transform_bootstrap_rotation_outlier_deg_) {
    return;
  }

  auto translation_delta =
      (sample.getOrigin() - learned_front_from_back_tag_.getOrigin()) *
      tag_transform_online_alpha_;
  const double translation_step = translation_delta.length();
  if (tag_transform_max_translation_step_m_ == 0.0) {
    translation_delta.setValue(0.0, 0.0, 0.0);
  } else if (translation_step > tag_transform_max_translation_step_m_) {
    translation_delta *=
        tag_transform_max_translation_step_m_ / translation_step;
  }

  auto current_rotation = learned_front_from_back_tag_.getRotation();
  auto sample_rotation = sample.getRotation();
  current_rotation.normalize();
  sample_rotation.normalize();
  const double rotation_distance =
      rotationDistanceDegrees(current_rotation, sample_rotation);
  double interpolation_fraction = tag_transform_online_alpha_;
  if (rotation_distance > 0.0) {
    const double bounded_rotation_step =
        std::min(rotation_distance * tag_transform_online_alpha_,
                 tag_transform_max_rotation_step_deg_);
    interpolation_fraction = bounded_rotation_step / rotation_distance;
  }
  auto updated_rotation =
      current_rotation.slerp(sample_rotation, interpolation_fraction);
  updated_rotation.normalize();
  learned_front_from_back_tag_.setOrigin(
      learned_front_from_back_tag_.getOrigin() + translation_delta);
  learned_front_from_back_tag_.setRotation(updated_rotation);
  learned_tag_separation_m_ = learned_front_from_back_tag_.getOrigin().length();
}

void ApriltagFusionNode::updateTagSeparation(const FusedTagEstimate &front,
                                             const FusedTagEstimate &back) {
  if (!learn_tag_separation_) {
    return;
  }

  std::vector<uint64_t> pair_sequences = front.source_sequences;
  pair_sequences.insert(pair_sequences.end(), back.source_sequences.begin(),
                        back.source_sequences.end());
  std::sort(pair_sequences.begin(), pair_sequences.end());
  if (pair_sequences == last_separation_update_sequences_) {
    return;
  }
  last_separation_update_sequences_ = std::move(pair_sequences);

  auto expected_back_rotation = front.observation.fusion_from_tag.getRotation();
  tf2::Quaternion back_from_front;
  back_from_front.setRPY(0.0, kPi, 0.0);
  expected_back_rotation *= back_from_front;
  expected_back_rotation.normalize();
  auto measured_back_rotation = back.observation.fusion_from_tag.getRotation();
  measured_back_rotation.normalize();
  const double rotation_dot = std::clamp(
      std::abs(expected_back_rotation.dot(measured_back_rotation)), 0.0, 1.0);
  const double rotation_error_deg =
      2.0 * std::acos(rotation_dot) * kRadiansToDegrees;
  if (rotation_error_deg > max_rotation_disagreement_deg_) {
    return;
  }

  const auto front_center =
      tagFrameFromSingleTag(kFrontTagSlot, front.observation.fusion_from_tag);
  const auto back_center =
      tagFrameFromSingleTag(kBackTagSlot, back.observation.fusion_from_tag);
  if (front_center.getOrigin().distance(back_center.getOrigin()) >
      max_translation_disagreement_m_) {
    return;
  }

  const auto front_z = tagZAxis(front.observation.fusion_from_tag);
  const auto back_z = tagZAxis(back.observation.fusion_from_tag);
  auto inward_axis = front_z - back_z;
  if (!std::isfinite(inward_axis.length2()) || inward_axis.length2() < 1e-12) {
    return;
  }
  inward_axis.normalize();

  const double separation_sample =
      (back.observation.fusion_from_tag.getOrigin() -
       front.observation.fusion_from_tag.getOrigin())
          .dot(inward_axis);
  if (!std::isfinite(separation_sample) || separation_sample <= 0.0) {
    return;
  }

  const double innovation = separation_sample - learned_tag_separation_m_;
  const double clipped_innovation =
      std::clamp(innovation, -tag_separation_max_innovation_m_,
                 tag_separation_max_innovation_m_);
  learned_tag_separation_m_ += tag_separation_ema_alpha_ * clipped_innovation;
  if (!learn_tag_transform_) {
    ideal_front_from_back_tag_.setOrigin(
        tf2::Vector3(0.0, 0.0, learned_tag_separation_m_));
  }
  RCLCPP_DEBUG(get_logger(), "Learned tag separation %.4f m (sample %.4f m)",
               learned_tag_separation_m_, separation_sample);
}

std::optional<size_t> ApriltagFusionNode::selectFallbackTagSlot() const {
  const auto &front = last_tag_estimates_[kFrontTagSlot];
  const auto &back = last_tag_estimates_[kBackTagSlot];
  if (!front) {
    return back ? std::optional<size_t>(kBackTagSlot) : std::nullopt;
  }
  if (!back) {
    return kFrontTagSlot;
  }
  if (back->observation.quality > front->observation.quality ||
      (back->observation.quality == front->observation.quality &&
       back->observation.receipt_time > front->observation.receipt_time)) {
    return kBackTagSlot;
  }
  return kFrontTagSlot;
}

std::optional<double>
ApriltagFusionNode::constrainedYaw(const tf2::Quaternion &rotation_value,
                                   bool allow_axis_fallback) {
  if (!std::isfinite(rotation_value.length2()) ||
      rotation_value.length2() <= 1e-12) {
    return std::nullopt;
  }
  auto rotation = rotation_value;
  rotation.normalize();
  const tf2::Matrix3x3 basis(rotation);

  // This is the Frobenius-nearest member of
  // R(yaw) = Rz(yaw) * Rx(-pi / 2).  It uses both horizontal axes instead of
  // extracting Euler angles from a noisy unconstrained rotation.
  const double cosine_term = basis[0][0] + basis[1][2];
  const double sine_term = basis[1][0] - basis[0][2];
  if (std::isfinite(cosine_term) && std::isfinite(sine_term) &&
      std::hypot(cosine_term, sine_term) > 1e-9) {
    return std::atan2(sine_term, cosine_term);
  }
  if (!allow_axis_fallback) {
    return std::nullopt;
  }

  const double x_axis_norm = std::hypot(basis[0][0], basis[1][0]);
  const double z_axis_norm = std::hypot(basis[0][2], basis[1][2]);
  if (std::isfinite(x_axis_norm) && x_axis_norm >= z_axis_norm &&
      x_axis_norm > 1e-9) {
    return std::atan2(basis[1][0], basis[0][0]);
  }
  if (std::isfinite(z_axis_norm) && z_axis_norm > 1e-9) {
    return std::atan2(-basis[0][2], basis[1][2]);
  }
  return std::nullopt;
}

tf2::Quaternion ApriltagFusionNode::constrainedRotation(double yaw) {
  tf2::Quaternion result;
  result.setRPY(-0.5 * kPi, 0.0, yaw);
  result.normalize();
  return result;
}

void ApriltagFusionNode::predictKalmanAxis(KalmanAxisState &state,
                                           double delta_sec,
                                           double acceleration_std) {
  state.position += delta_sec * state.velocity;

  const double delta2 = delta_sec * delta_sec;
  const double delta3 = delta2 * delta_sec;
  const double delta4 = delta2 * delta2;
  const double acceleration_variance = acceleration_std * acceleration_std;
  const double position_variance =
      state.position_variance +
      2.0 * delta_sec * state.position_velocity_covariance +
      delta2 * state.velocity_variance + 0.25 * delta4 * acceleration_variance;
  const double position_velocity_covariance =
      state.position_velocity_covariance + delta_sec * state.velocity_variance +
      0.5 * delta3 * acceleration_variance;
  const double velocity_variance =
      state.velocity_variance + delta2 * acceleration_variance;

  state.position_variance = std::max(0.0, position_variance);
  state.position_velocity_covariance = position_velocity_covariance;
  state.velocity_variance = std::max(0.0, velocity_variance);
}

void ApriltagFusionNode::correctKalmanAxis(KalmanAxisState &state,
                                           double measurement,
                                           double measurement_std,
                                           bool wrap_innovation) {
  const double measurement_variance = measurement_std * measurement_std;
  const double innovation_variance =
      state.position_variance + measurement_variance;
  if (!std::isfinite(innovation_variance) || innovation_variance <= 0.0) {
    return;
  }

  double innovation = measurement - state.position;
  if (wrap_innovation) {
    innovation = std::remainder(innovation, 2.0 * kPi);
  }
  const double position_gain = state.position_variance / innovation_variance;
  const double velocity_gain =
      state.position_velocity_covariance / innovation_variance;

  const double old_position_variance = state.position_variance;
  const double old_position_velocity_covariance =
      state.position_velocity_covariance;
  const double old_velocity_variance = state.velocity_variance;
  state.position += position_gain * innovation;
  state.velocity += velocity_gain * innovation;

  // Joseph-form covariance update for H = [1, 0].
  const double residual_gain = 1.0 - position_gain;
  state.position_variance =
      residual_gain * residual_gain * old_position_variance +
      position_gain * position_gain * measurement_variance;
  state.position_velocity_covariance =
      residual_gain * (old_position_velocity_covariance -
                       velocity_gain * old_position_variance) +
      position_gain * velocity_gain * measurement_variance;
  state.velocity_variance =
      old_velocity_variance -
      2.0 * velocity_gain * old_position_velocity_covariance +
      velocity_gain * velocity_gain *
          (old_position_variance + measurement_variance);
  state.position_variance = std::max(0.0, state.position_variance);
  state.velocity_variance = std::max(0.0, state.velocity_variance);
}

tf2::Transform ApriltagFusionNode::kalmanFilteredTransform() const {
  return tf2::Transform(constrainedRotation(kalman_axes_[3].position),
                        tf2::Vector3(kalman_axes_[0].position,
                                     kalman_axes_[1].position,
                                     fixed_tag_frame_z_m_));
}

tf2::Transform ApriltagFusionNode::filterTransform(
    const tf2::Transform &fusion_from_tag_frame_measurement,
    const rclcpp::Time &filter_time) {
  const auto measurement_origin = fusion_from_tag_frame_measurement.getOrigin();
  const auto measured_yaw = constrainedYaw(
      fusion_from_tag_frame_measurement.getRotation(), !has_kalman_state_);
  const double filter_delta_sec =
      has_kalman_state_ ? (filter_time - last_kalman_filter_time_).seconds()
                        : 0.0;
  const bool reset_filter =
      !has_kalman_state_ || !std::isfinite(filter_delta_sec) ||
      filter_delta_sec < 0.0 || filter_delta_sec > kalman_reset_sec_;

  if (reset_filter) {
    const std::array<double, 4> measurements{
        measurement_origin.x(), measurement_origin.y(), fixed_tag_frame_z_m_,
        measured_yaw.value_or(has_kalman_state_ ? kalman_axes_[3].position
                                                : 0.0)};
    const double position_variance =
        kalman_position_measurement_std_m_ * kalman_position_measurement_std_m_;
    const double yaw_measurement_std =
        kalman_yaw_measurement_std_deg_ / kRadiansToDegrees;
    const double yaw_variance = yaw_measurement_std * yaw_measurement_std;
    const double velocity_variance = kalman_initial_linear_velocity_std_mps_ *
                                     kalman_initial_linear_velocity_std_mps_;
    const double yaw_rate_std =
        kalman_initial_yaw_rate_std_degps_ / kRadiansToDegrees;
    const double yaw_rate_variance = yaw_rate_std * yaw_rate_std;
    for (std::size_t axis = 0; axis < kalman_axes_.size(); ++axis) {
      auto &state = kalman_axes_[axis];
      state.position = measurements[axis];
      state.velocity = 0.0;
      state.position_variance =
          axis == 2 ? 0.0 : (axis < 3 ? position_variance : yaw_variance);
      state.position_velocity_covariance = 0.0;
      state.velocity_variance =
          axis == 2 ? 0.0 : (axis < 3 ? velocity_variance : yaw_rate_variance);
    }
  } else {
    const double yaw_acceleration_std =
        kalman_yaw_acceleration_std_degps2_ / kRadiansToDegrees;
    for (std::size_t axis = 0; axis < kalman_axes_.size(); ++axis) {
      if (axis == 2) {
        continue;
      }
      predictKalmanAxis(kalman_axes_[axis], filter_delta_sec,
                        axis < 3 ? kalman_linear_acceleration_std_mps2_
                                 : yaw_acceleration_std);
    }

    correctKalmanAxis(kalman_axes_[0], measurement_origin.x(),
                      kalman_position_measurement_std_m_, false);
    correctKalmanAxis(kalman_axes_[1], measurement_origin.y(),
                      kalman_position_measurement_std_m_, false);
    if (measured_yaw) {
      correctKalmanAxis(kalman_axes_[3], *measured_yaw,
                        kalman_yaw_measurement_std_deg_ / kRadiansToDegrees,
                        true);
    }
  }

  last_kalman_filter_time_ = filter_time;
  has_kalman_state_ = true;
  return kalmanFilteredTransform();
}

void ApriltagFusionNode::publishResult(
    const tf2::Transform &fusion_from_tag_frame, const rclcpp::Time &stamp) {
  const auto tag_frame_from_fusion = fusion_from_tag_frame.inverse();
  if (publish_fusion_pose_ && pose_pub_) {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = tag_frame_id_;
    pose_msg.pose = poseMsgFromTransform(tag_frame_from_fusion);
    pose_pub_->publish(pose_msg);
  }

  if (publish_tf_ && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = tag_frame_id_;
    tf_msg.child_frame_id = fusion_frame_id_;
    tf_msg.transform = transformMsgFromTransform(tag_frame_from_fusion);
    tf_broadcaster_->sendTransform(tf_msg);
  }
}

bool ApriltagFusionNode::toGrayImage(const sensor_msgs::msg::Image &msg,
                                     cv::Mat &gray) const {
  if (msg.height == 0 || msg.width == 0 || msg.data.empty()) {
    return false;
  }
  const size_t min_size = static_cast<size_t>(msg.height) * msg.step;
  if (msg.data.size() < min_size) {
    return false;
  }

  if (msg.encoding == sensor_msgs::image_encodings::MONO8 ||
      msg.encoding == "8UC1") {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC1,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    gray = view;
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::BGR8 ||
      msg.encoding == "8UC3") {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC3,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, gray, cv::COLOR_BGR2GRAY);
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::RGB8) {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC3,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, gray, cv::COLOR_RGB2GRAY);
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::BGRA8) {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC4,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, gray, cv::COLOR_BGRA2GRAY);
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::RGBA8) {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC4,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, gray, cv::COLOR_RGBA2GRAY);
    return true;
  }
  return false;
}

bool ApriltagFusionNode::toBgrImage(const sensor_msgs::msg::Image &msg,
                                    cv::Mat &bgr) const {
  if (msg.height == 0 || msg.width == 0 || msg.data.empty()) {
    return false;
  }
  const size_t min_size = static_cast<size_t>(msg.height) * msg.step;
  if (msg.data.size() < min_size) {
    return false;
  }

  if (msg.encoding == sensor_msgs::image_encodings::MONO8 ||
      msg.encoding == "8UC1") {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC1,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, bgr, cv::COLOR_GRAY2BGR);
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::BGR8 ||
      msg.encoding == "8UC3") {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC3,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    bgr = view.clone();
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::RGB8) {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC3,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::BGRA8) {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC4,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, bgr, cv::COLOR_BGRA2BGR);
    return true;
  }
  if (msg.encoding == sensor_msgs::image_encodings::RGBA8) {
    const cv::Mat view(static_cast<int>(msg.height),
                       static_cast<int>(msg.width), CV_8UC4,
                       const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(view, bgr, cv::COLOR_RGBA2BGR);
    return true;
  }
  return false;
}

bool ApriltagFusionNode::drawDebugMarkerOutline(
    cv::Mat &bgr, const std::vector<cv::Point2f> &corners) {
  if (corners.size() < 2) {
    return false;
  }

  std::vector<cv::Point> integer_corners;
  integer_corners.reserve(corners.size());
  for (const auto &corner : corners) {
    if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
      return false;
    }
    integer_corners.emplace_back(cvRound(corner.x), cvRound(corner.y));
  }

  const std::vector<std::vector<cv::Point>> outline{integer_corners};
  cv::polylines(bgr, outline, true, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
  return true;
}

sensor_msgs::msg::Image
ApriltagFusionNode::makeDebugImage(const sensor_msgs::msg::Image &source,
                                   const cv::Mat &bgr) const {
  const cv::Mat contiguous = bgr.isContinuous() ? bgr : bgr.clone();
  sensor_msgs::msg::Image output;
  output.header = source.header;
  output.height = static_cast<uint32_t>(contiguous.rows);
  output.width = static_cast<uint32_t>(contiguous.cols);
  output.encoding = sensor_msgs::image_encodings::BGR8;
  output.is_bigendian = false;
  output.step = static_cast<uint32_t>(contiguous.cols * contiguous.elemSize());
  const auto byte_count = static_cast<size_t>(output.step) * output.height;
  output.data.assign(contiguous.data, contiguous.data + byte_count);
  return output;
}

std::optional<cv::Mat> ApriltagFusionNode::cameraMatrixForImage(
    const sensor_msgs::msg::CameraInfo &camera_info,
    const sensor_msgs::msg::Image &image) const {
  if (camera_info.k[0] <= 0.0 || camera_info.k[4] <= 0.0) {
    return std::nullopt;
  }

  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << camera_info.k[0], camera_info.k[1],
       camera_info.k[2], camera_info.k[3], camera_info.k[4], camera_info.k[5],
       camera_info.k[6], camera_info.k[7], camera_info.k[8]);

  if (camera_info.width > 0 && camera_info.height > 0 && image.width > 0 &&
      image.height > 0 &&
      (camera_info.width != image.width ||
       camera_info.height != image.height)) {
    const double scale_x = static_cast<double>(image.width) /
                           static_cast<double>(camera_info.width);
    const double scale_y = static_cast<double>(image.height) /
                           static_cast<double>(camera_info.height);
    camera_matrix.at<double>(0, 0) *= scale_x;
    camera_matrix.at<double>(0, 2) *= scale_x;
    camera_matrix.at<double>(1, 1) *= scale_y;
    camera_matrix.at<double>(1, 2) *= scale_y;
  }
  return camera_matrix;
}

cv::Mat ApriltagFusionNode::distortionCoeffs(
    const sensor_msgs::msg::CameraInfo &camera_info) const {
  if (camera_info.d.empty()) {
    return cv::Mat();
  }

  cv::Mat coeffs(1, static_cast<int>(camera_info.d.size()), CV_64F);
  for (size_t i = 0; i < camera_info.d.size(); ++i) {
    coeffs.at<double>(0, static_cast<int>(i)) = camera_info.d[i];
  }
  return coeffs;
}

std::array<std::optional<size_t>, ApriltagFusionNode::kTagCount>
ApriltagFusionNode::selectMarkers(
    const std::vector<int> &ids,
    const std::vector<std::vector<cv::Point2f>> &corners) const {
  std::array<std::optional<size_t>, kTagCount> selected;
  std::array<double, kTagCount> best_areas{};
  for (size_t index = 0; index < ids.size() && index < corners.size();
       ++index) {
    std::optional<size_t> tag_slot;
    if (ids[index] == front_tag_id_) {
      tag_slot = kFrontTagSlot;
    } else if (ids[index] == back_tag_id_) {
      tag_slot = kBackTagSlot;
    }
    if (!tag_slot) {
      continue;
    }

    const double area = markerArea(corners[index]);
    if (!selected[*tag_slot] || area > best_areas[*tag_slot]) {
      selected[*tag_slot] = index;
      best_areas[*tag_slot] = area;
    }
  }
  return selected;
}

std::optional<ApriltagFusionNode::PoseEstimate>
ApriltagFusionNode::estimateTagInCameraFrame(
    const std::vector<cv::Point2f> &corners, const cv::Mat &camera_matrix,
    const cv::Mat &distortion_coeffs, double tag_size_m,
    const DepthFrameView *depth_frame, std::size_t image_width,
    std::size_t image_height, const PoseEstimate *pnp_prior) const {
  auto pnp_estimate = estimateTagWithPnp(
      corners, camera_matrix, distortion_coeffs, tag_size_m, pnp_prior);
  if (!pnp_estimate) {
    return std::nullopt;
  }
  if (!use_depth_ || depth_frame == nullptr ||
      !static_cast<bool>(*depth_frame)) {
    return pnp_estimate;
  }

  auto depth_estimate = estimateTagWithDepth(
      corners, camera_matrix, distortion_coeffs, tag_size_m, *depth_frame,
      image_width, image_height, *pnp_estimate);
  return depth_estimate ? depth_estimate : pnp_estimate;
}

std::optional<std::size_t> ApriltagFusionNode::selectPnpCandidate(
    const std::vector<cv::Mat> &candidate_rvecs,
    const std::vector<double> &candidate_squared_errors,
    const cv::Mat *prior_rvec, double ambiguity_margin_px,
    std::size_t corner_count) {
  if (candidate_rvecs.empty() ||
      candidate_rvecs.size() != candidate_squared_errors.size() ||
      !std::isfinite(ambiguity_margin_px) || ambiguity_margin_px < 0.0 ||
      corner_count == 0) {
    return std::nullopt;
  }

  std::optional<std::size_t> best_index;
  for (std::size_t index = 0; index < candidate_squared_errors.size();
       ++index) {
    if (!std::isfinite(candidate_squared_errors[index])) {
      continue;
    }
    if (!best_index || candidate_squared_errors[index] <
                           candidate_squared_errors[*best_index]) {
      best_index = index;
    }
  }
  if (!best_index || prior_rvec == nullptr || prior_rvec->empty() ||
      prior_rvec->total() < 3) {
    return best_index;
  }

  cv::Mat prior_rotation;
  cv::Rodrigues(*prior_rvec, prior_rotation);
  const double best_rmse =
      std::sqrt(candidate_squared_errors[*best_index] / corner_count);
  std::optional<std::size_t> closest_index;
  double closest_rotation_delta_deg = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < candidate_rvecs.size(); ++index) {
    const double squared_error = candidate_squared_errors[index];
    if (!std::isfinite(squared_error) ||
        std::sqrt(squared_error / corner_count) >
            best_rmse + ambiguity_margin_px) {
      continue;
    }

    cv::Mat candidate_rotation;
    cv::Rodrigues(candidate_rvecs[index], candidate_rotation);
    const cv::Mat rotation_delta = candidate_rotation * prior_rotation.t();
    const double rotation_trace = rotation_delta.at<double>(0, 0) +
                                  rotation_delta.at<double>(1, 1) +
                                  rotation_delta.at<double>(2, 2);
    const double rotation_delta_deg =
        std::acos(std::clamp((rotation_trace - 1.0) * 0.5, -1.0, 1.0)) *
        kRadiansToDegrees;
    if (!closest_index ||
        rotation_delta_deg < closest_rotation_delta_deg - 1e-12 ||
        (std::abs(rotation_delta_deg - closest_rotation_delta_deg) <= 1e-12 &&
         squared_error < candidate_squared_errors[*closest_index])) {
      closest_index = index;
      closest_rotation_delta_deg = rotation_delta_deg;
    }
  }
  return closest_index ? closest_index : best_index;
}

std::optional<ApriltagFusionNode::PoseEstimate>
ApriltagFusionNode::estimateTagWithPnp(const std::vector<cv::Point2f> &corners,
                                       const cv::Mat &camera_matrix,
                                       const cv::Mat &distortion_coeffs,
                                       double tag_size_m,
                                       const PoseEstimate *pnp_prior) const {
  if (corners.size() != 4) {
    return std::nullopt;
  }

  const auto object_points = markerObjectPoints(tag_size_m);
  cv::Mat rvec;
  cv::Mat tvec;
  double squared_error_sum = std::numeric_limits<double>::infinity();
  const auto candidateSquaredError =
      [&object_points, &corners, &camera_matrix, &distortion_coeffs](
          const cv::Mat &candidate_rvec,
          const cv::Mat &candidate_tvec) -> std::optional<double> {
    if (candidate_tvec.rows * candidate_tvec.cols < 3 ||
        candidate_tvec.type() != CV_64F ||
        !std::isfinite(candidate_tvec.at<double>(2, 0)) ||
        candidate_tvec.at<double>(2, 0) <= 0.0) {
      return std::nullopt;
    }
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, candidate_rvec, candidate_tvec,
                      camera_matrix, distortion_coeffs, projected_points);
    if (projected_points.size() != corners.size()) {
      return std::nullopt;
    }
    double error = 0.0;
    for (std::size_t index = 0; index < corners.size(); ++index) {
      const double dx = projected_points[index].x - corners[index].x;
      const double dy = projected_points[index].y - corners[index].y;
      error += dx * dx + dy * dy;
    }
    return error;
  };

  std::vector<cv::Mat> candidate_rvecs;
  std::vector<cv::Mat> candidate_tvecs;
  cv::solvePnPGeneric(object_points, corners, camera_matrix, distortion_coeffs,
                      candidate_rvecs, candidate_tvecs, false,
                      cv::SOLVEPNP_IPPE);
  std::vector<double> candidate_squared_errors(
      candidate_rvecs.size(), std::numeric_limits<double>::infinity());
  for (std::size_t index = 0;
       index < candidate_rvecs.size() && index < candidate_tvecs.size();
       ++index) {
    const auto error =
        candidateSquaredError(candidate_rvecs[index], candidate_tvecs[index]);
    if (error) {
      candidate_squared_errors[index] = *error;
    }
  }
  const cv::Mat *prior_rvec =
      pnp_prior && !pnp_prior->rvec.empty() ? &pnp_prior->rvec : nullptr;
  const auto selected_candidate =
      selectPnpCandidate(candidate_rvecs, candidate_squared_errors, prior_rvec,
                         pnp_ambiguity_reprojection_margin_px_, corners.size());
  if (selected_candidate && *selected_candidate < candidate_tvecs.size()) {
    squared_error_sum = candidate_squared_errors[*selected_candidate];
    rvec = candidate_rvecs[*selected_candidate].clone();
    tvec = candidate_tvecs[*selected_candidate].clone();
  }

  if (rvec.empty()) {
    const bool ok =
        cv::solvePnP(object_points, corners, camera_matrix, distortion_coeffs,
                     rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    const auto error = ok ? candidateSquaredError(rvec, tvec) : std::nullopt;
    if (!error) {
      return std::nullopt;
    }
    squared_error_sum = *error;
  }

  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);
  cv::Mat translation_camera = (cv::Mat_<double>(3, 1) << tvec.at<double>(0, 0),
                                tvec.at<double>(1, 0), tvec.at<double>(2, 0));
  cv::Mat rotation_camera = rotation_cv;
  if (camera_frame_convention_ == CameraFrameConvention::ZedXForward) {
    const auto optical_to_zed = cvOpticalToZedXForwardRotation();
    rotation_camera = optical_to_zed * rotation_camera;
    translation_camera = optical_to_zed * translation_camera;
  }

  tf2::Matrix3x3 basis(
      rotation_camera.at<double>(0, 0), rotation_camera.at<double>(0, 1),
      rotation_camera.at<double>(0, 2), rotation_camera.at<double>(1, 0),
      rotation_camera.at<double>(1, 1), rotation_camera.at<double>(1, 2),
      rotation_camera.at<double>(2, 0), rotation_camera.at<double>(2, 1),
      rotation_camera.at<double>(2, 2));
  tf2::Vector3 origin(translation_camera.at<double>(0, 0),
                      translation_camera.at<double>(1, 0),
                      translation_camera.at<double>(2, 0));

  PoseEstimate estimate;
  estimate.camera_from_tag = tf2::Transform(basis, origin);
  estimate.rvec = rvec.clone();
  estimate.tvec = tvec.clone();
  estimate.reprojection_rmse_px =
      std::sqrt(squared_error_sum / static_cast<double>(corners.size()));
  return estimate;
}

std::optional<ApriltagFusionNode::DepthPlane> ApriltagFusionNode::fitDepthPlane(
    const std::vector<cv::Point2f> &corners, const cv::Mat &camera_matrix,
    const DepthFrameView &depth_frame, std::size_t image_width,
    std::size_t image_height, const PoseEstimate *pnp_estimate) const {
  if (corners.size() != 4 || !static_cast<bool>(depth_frame) ||
      image_width == 0 || image_height == 0 ||
      camera_matrix.at<double>(0, 0) <= 0.0 ||
      camera_matrix.at<double>(1, 1) <= 0.0) {
    return std::nullopt;
  }

  cv::Point2f center(0.0F, 0.0F);
  for (const auto &corner : corners) {
    center += corner;
  }
  center *= 0.25F;

  const double inner_scale = 1.0 - depth_inner_margin_ratio_;
  const double depth_scale_x =
      static_cast<double>(depth_frame.width) / image_width;
  const double depth_scale_y =
      static_cast<double>(depth_frame.height) / image_height;
  std::vector<cv::Point2f> inner_depth_polygon;
  inner_depth_polygon.reserve(corners.size());
  for (const auto &corner : corners) {
    const auto inner =
        center + (corner - center) * static_cast<float>(inner_scale);
    inner_depth_polygon.emplace_back(
        static_cast<float>((inner.x + 0.5) * depth_scale_x - 0.5),
        static_cast<float>((inner.y + 0.5) * depth_scale_y - 0.5));
  }

  const auto bounds = cv::boundingRect(inner_depth_polygon);
  const int min_x = std::max(bounds.x, 0);
  const int min_y = std::max(bounds.y, 0);
  const int max_x =
      std::min(bounds.x + bounds.width, static_cast<int>(depth_frame.width));
  const int max_y =
      std::min(bounds.y + bounds.height, static_cast<int>(depth_frame.height));
  if (min_x >= max_x || min_y >= max_y) {
    return std::nullopt;
  }

  const double fx = camera_matrix.at<double>(0, 0);
  const double fy = camera_matrix.at<double>(1, 1);
  const double cx = camera_matrix.at<double>(0, 2);
  const double cy = camera_matrix.at<double>(1, 2);
  std::optional<cv::Vec3d> expected_normal;
  double expected_offset = 0.0;
  if (pnp_estimate && !pnp_estimate->rvec.empty() &&
      !pnp_estimate->tvec.empty()) {
    cv::Mat expected_rotation;
    cv::Rodrigues(pnp_estimate->rvec, expected_rotation);
    cv::Vec3d normal(expected_rotation.at<double>(0, 2),
                     expected_rotation.at<double>(1, 2),
                     expected_rotation.at<double>(2, 2));
    const double normal_length = cv::norm(normal);
    if (std::isfinite(normal_length) && normal_length > 1e-9) {
      normal /= normal_length;
      expected_normal = normal;
      expected_offset = -(normal[0] * pnp_estimate->tvec.at<double>(0, 0) +
                          normal[1] * pnp_estimate->tvec.at<double>(1, 0) +
                          normal[2] * pnp_estimate->tvec.at<double>(2, 0));
    }
  }

  constexpr std::size_t kMaximumPlaneSamples = 1000;
  const double bounds_area = static_cast<double>(std::max(bounds.area(), 1));
  const int sampling_stride =
      std::max(1, static_cast<int>(std::ceil(
                      std::sqrt(bounds_area / kMaximumPlaneSamples))));
  std::vector<cv::Vec3d> samples;
  samples.reserve(kMaximumPlaneSamples);
  std::size_t candidate_pixels = 0;
  const auto *depth_bytes =
      reinterpret_cast<const std::uint8_t *>(depth_frame.data);
  for (int depth_y = min_y; depth_y < max_y; depth_y += sampling_stride) {
    const auto *row = reinterpret_cast<const float *>(
        depth_bytes +
        static_cast<std::size_t>(depth_y) * depth_frame.row_stride_bytes);
    for (int depth_x = min_x; depth_x < max_x; depth_x += sampling_stride) {
      if (cv::pointPolygonTest(inner_depth_polygon,
                               cv::Point2f(static_cast<float>(depth_x),
                                           static_cast<float>(depth_y)),
                               false) < 0.0) {
        continue;
      }
      ++candidate_pixels;
      const double depth_m = static_cast<double>(row[depth_x]);
      if (!std::isfinite(depth_m) || depth_m <= 0.0) {
        continue;
      }
      const double image_x =
          (static_cast<double>(depth_x) + 0.5) / depth_scale_x - 0.5;
      const double image_y =
          (static_cast<double>(depth_y) + 0.5) / depth_scale_y - 0.5;
      const cv::Vec3d sample((image_x - cx) * depth_m / fx,
                             (image_y - cy) * depth_m / fy, depth_m);
      const double pnp_range_m = pnp_estimate && !pnp_estimate->tvec.empty()
                                     ? pnp_estimate->tvec.at<double>(2, 0)
                                     : 0.0;
      const double range_aware_gate_m =
          0.015 * std::clamp(pnp_range_m, 0.0, 3.0) +
          0.090 * std::max(0.0, pnp_range_m - 3.0);
      const double pnp_plane_gate_m = std::min(
          depth_max_pnp_translation_delta_m_,
          std::max({2.0 * depth_plane_inlier_threshold_m_,
                    4.0 * depth_plane_max_rmse_m_, range_aware_gate_m}));
      if (expected_normal && std::abs(expected_normal->dot(sample) +
                                      expected_offset) > pnp_plane_gate_m) {
        continue;
      }
      samples.push_back(sample);
    }
  }

  if (candidate_pixels == 0 ||
      samples.size() < static_cast<std::size_t>(depth_min_valid_samples_)) {
    return std::nullopt;
  }
  const double valid_fraction =
      static_cast<double>(samples.size()) / candidate_pixels;
  if (valid_fraction < depth_min_valid_fraction_) {
    return std::nullopt;
  }

  std::vector<double> sample_depths;
  sample_depths.reserve(samples.size());
  for (const auto &sample : samples) {
    sample_depths.push_back(sample[2]);
  }
  const double median_depth = median(sample_depths);
  std::vector<double> depth_deviations;
  depth_deviations.reserve(samples.size());
  for (const auto depth : sample_depths) {
    depth_deviations.push_back(std::abs(depth - median_depth));
  }
  const double depth_mad = median(depth_deviations);
  const double depth_gate =
      std::max(4.0 * 1.4826 * depth_mad, 4.0 * depth_plane_inlier_threshold_m_);
  samples.erase(
      std::remove_if(samples.begin(), samples.end(),
                     [median_depth, depth_gate](const cv::Vec3d &sample) {
                       return std::abs(sample[2] - median_depth) > depth_gate;
                     }),
      samples.end());
  if (samples.size() < static_cast<std::size_t>(depth_min_valid_samples_)) {
    return std::nullopt;
  }

  cv::Vec3d best_normal;
  double best_offset = 0.0;
  std::size_t best_inlier_count = 0;
  double best_squared_error = std::numeric_limits<double>::infinity();
  std::uint64_t random_state =
      0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(samples.size());
  const auto nextIndex = [&random_state, &samples]() {
    random_state = random_state * 6364136223846793005ULL + 1ULL;
    return static_cast<std::size_t>(random_state % samples.size());
  };
  constexpr std::size_t kRansacIterations = 128;
  for (std::size_t iteration = 0; iteration < kRansacIterations; ++iteration) {
    const auto first = nextIndex();
    auto second = nextIndex();
    auto third = nextIndex();
    if (second == first) {
      second = (second + 1) % samples.size();
    }
    if (third == first || third == second) {
      third = (third + 1) % samples.size();
      if (third == first || third == second) {
        third = (third + 1) % samples.size();
      }
    }
    auto normal = (samples[second] - samples[first])
                      .cross(samples[third] - samples[first]);
    const double normal_length = cv::norm(normal);
    if (!std::isfinite(normal_length) || normal_length < 1e-9) {
      continue;
    }
    normal /= normal_length;
    const double offset = -normal.dot(samples[first]);

    std::size_t inlier_count = 0;
    double squared_error = 0.0;
    for (const auto &sample : samples) {
      const double residual = std::abs(normal.dot(sample) + offset);
      if (residual <= depth_plane_inlier_threshold_m_) {
        ++inlier_count;
        squared_error += residual * residual;
      }
    }
    if (inlier_count > best_inlier_count ||
        (inlier_count == best_inlier_count &&
         squared_error < best_squared_error)) {
      best_normal = normal;
      best_offset = offset;
      best_inlier_count = inlier_count;
      best_squared_error = squared_error;
    }
  }
  if (best_inlier_count < static_cast<std::size_t>(depth_min_valid_samples_)) {
    return std::nullopt;
  }

  std::vector<cv::Vec3d> inliers;
  inliers.reserve(best_inlier_count);
  for (const auto &sample : samples) {
    if (std::abs(best_normal.dot(sample) + best_offset) <=
        depth_plane_inlier_threshold_m_) {
      inliers.push_back(sample);
    }
  }

  cv::Vec3d centroid(0.0, 0.0, 0.0);
  for (const auto &point : inliers) {
    centroid += point;
  }
  centroid /= static_cast<double>(inliers.size());
  cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
  for (const auto &point : inliers) {
    const auto delta = point - centroid;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        covariance.at<double>(row, col) += delta[row] * delta[col];
      }
    }
  }
  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
  if (!cv::eigen(covariance, eigenvalues, eigenvectors)) {
    return std::nullopt;
  }
  cv::Vec3d normal(eigenvectors.at<double>(2, 0), eigenvectors.at<double>(2, 1),
                   eigenvectors.at<double>(2, 2));
  const double normal_length = cv::norm(normal);
  if (!std::isfinite(normal_length) || normal_length < 1e-9) {
    return std::nullopt;
  }
  normal /= normal_length;
  double offset = -normal.dot(centroid);
  if ((expected_normal && normal.dot(*expected_normal) < 0.0) ||
      (!expected_normal && normal[2] > 0.0)) {
    normal = -normal;
    offset = -offset;
  }

  double squared_error_sum = 0.0;
  std::size_t refined_inlier_count = 0;
  for (const auto &sample : samples) {
    const double residual = std::abs(normal.dot(sample) + offset);
    if (residual <= depth_plane_inlier_threshold_m_) {
      squared_error_sum += residual * residual;
      ++refined_inlier_count;
    }
  }
  if (refined_inlier_count <
      static_cast<std::size_t>(depth_min_valid_samples_)) {
    return std::nullopt;
  }
  const double rmse = std::sqrt(squared_error_sum / refined_inlier_count);
  if (!std::isfinite(rmse) || rmse > depth_plane_max_rmse_m_) {
    return std::nullopt;
  }

  DepthPlane plane;
  plane.normal = normal;
  plane.offset = offset;
  plane.valid_samples = samples.size();
  plane.inlier_samples = refined_inlier_count;
  plane.valid_fraction = valid_fraction;
  plane.rmse_m = rmse;
  return plane;
}

std::optional<ApriltagFusionNode::PoseEstimate>
ApriltagFusionNode::estimateTagWithDepth(
    const std::vector<cv::Point2f> &corners, const cv::Mat &camera_matrix,
    const cv::Mat &distortion_coeffs, double tag_size_m,
    const DepthFrameView &depth_frame, std::size_t image_width,
    std::size_t image_height, const PoseEstimate &pnp_estimate) const {
  const auto plane = fitDepthPlane(corners, camera_matrix, depth_frame,
                                   image_width, image_height, &pnp_estimate);
  if (!plane) {
    return std::nullopt;
  }

  std::vector<cv::Point2f> normalized_corners;
  cv::undistortPoints(corners, normalized_corners, camera_matrix,
                      distortion_coeffs);
  if (normalized_corners.size() != 4) {
    return std::nullopt;
  }
  std::vector<cv::Vec3d> measured_corners;
  measured_corners.reserve(4);
  for (const auto &corner : normalized_corners) {
    const cv::Vec3d ray(corner.x, corner.y, 1.0);
    const double denominator = plane->normal.dot(ray);
    if (!std::isfinite(denominator) || std::abs(denominator) < 1e-9) {
      return std::nullopt;
    }
    const double distance = -plane->offset / denominator;
    if (!std::isfinite(distance) || distance <= 0.0) {
      return std::nullopt;
    }
    measured_corners.push_back(ray * distance);
  }

  double max_size_error_fraction = 0.0;
  for (std::size_t index = 0; index < measured_corners.size(); ++index) {
    const double side_length =
        cv::norm(measured_corners[(index + 1) % measured_corners.size()] -
                 measured_corners[index]);
    max_size_error_fraction =
        std::max(max_size_error_fraction,
                 std::abs(side_length - tag_size_m) / tag_size_m);
  }
  if (!std::isfinite(max_size_error_fraction) ||
      max_size_error_fraction > depth_max_size_error_fraction_) {
    return std::nullopt;
  }

  const auto object_points = markerObjectPoints(tag_size_m);
  cv::Vec3d object_centroid(0.0, 0.0, 0.0);
  cv::Vec3d measured_centroid(0.0, 0.0, 0.0);
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    object_centroid += cv::Vec3d(object_points[index].x, object_points[index].y,
                                 object_points[index].z);
    measured_centroid += measured_corners[index];
  }
  object_centroid /= static_cast<double>(object_points.size());
  measured_centroid /= static_cast<double>(measured_corners.size());

  cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    const cv::Vec3d object_delta(object_points[index].x - object_centroid[0],
                                 object_points[index].y - object_centroid[1],
                                 object_points[index].z - object_centroid[2]);
    const auto measured_delta = measured_corners[index] - measured_centroid;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        covariance.at<double>(row, col) +=
            object_delta[row] * measured_delta[col];
      }
    }
  }
  cv::SVD svd(covariance, cv::SVD::FULL_UV);
  cv::Mat rotation_optical = svd.vt.t() * svd.u.t();
  if (cv::determinant(rotation_optical) < 0.0) {
    cv::Mat corrected_v = svd.vt.t();
    corrected_v.col(2) *= -1.0;
    rotation_optical = corrected_v * svd.u.t();
  }
  const cv::Mat object_center = (cv::Mat_<double>(3, 1) << object_centroid[0],
                                 object_centroid[1], object_centroid[2]);
  const cv::Mat measured_center =
      (cv::Mat_<double>(3, 1) << measured_centroid[0], measured_centroid[1],
       measured_centroid[2]);
  const cv::Mat translation_optical =
      measured_center - rotation_optical * object_center;

  cv::Mat pnp_rotation;
  cv::Rodrigues(pnp_estimate.rvec, pnp_rotation);
  const cv::Mat translation_delta = translation_optical - pnp_estimate.tvec;
  const double pnp_translation_delta = cv::norm(translation_delta);
  const cv::Vec3d pnp_normal(pnp_rotation.at<double>(0, 2),
                             pnp_rotation.at<double>(1, 2),
                             pnp_rotation.at<double>(2, 2));
  const cv::Vec3d depth_normal(rotation_optical.at<double>(0, 2),
                               rotation_optical.at<double>(1, 2),
                               rotation_optical.at<double>(2, 2));
  const double pnp_rotation_delta =
      std::acos(std::clamp(std::abs(pnp_normal.dot(depth_normal)), 0.0, 1.0)) *
      kRadiansToDegrees;
  if (!std::isfinite(pnp_translation_delta) ||
      !std::isfinite(pnp_rotation_delta)) {
    return std::nullopt;
  }

  const auto quaternionFromRotationMatrix = [](const cv::Mat &rotation) {
    tf2::Matrix3x3 basis(rotation.at<double>(0, 0), rotation.at<double>(0, 1),
                         rotation.at<double>(0, 2), rotation.at<double>(1, 0),
                         rotation.at<double>(1, 1), rotation.at<double>(1, 2),
                         rotation.at<double>(2, 0), rotation.at<double>(2, 1),
                         rotation.at<double>(2, 2));
    tf2::Quaternion quaternion;
    basis.getRotation(quaternion);
    quaternion.normalize();
    return quaternion;
  };
  const auto rotationMatrixFromQuaternion = [](const tf2::Quaternion &value) {
    const tf2::Matrix3x3 basis(value);
    cv::Mat result = (cv::Mat_<double>(3, 3) << basis[0][0], basis[0][1],
                      basis[0][2], basis[1][0], basis[1][1], basis[1][2],
                      basis[2][0], basis[2][1], basis[2][2]);
    return result;
  };
  const auto pnp_quaternion = quaternionFromRotationMatrix(pnp_rotation);
  const auto depth_quaternion = quaternionFromRotationMatrix(rotation_optical);
  const auto full_depth_normal_quaternion =
      blendTagNormal(pnp_quaternion, depth_quaternion, 1.0);
  const auto full_depth_normal_rotation =
      rotationMatrixFromQuaternion(full_depth_normal_quaternion);
  cv::Mat depth_rvec;
  cv::Rodrigues(full_depth_normal_rotation, depth_rvec);
  std::vector<cv::Point2f> depth_projected_points;
  cv::projectPoints(object_points, depth_rvec, translation_optical,
                    camera_matrix, distortion_coeffs, depth_projected_points);
  if (depth_projected_points.size() != corners.size()) {
    return std::nullopt;
  }
  double depth_squared_reprojection_error = 0.0;
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const double dx = depth_projected_points[index].x - corners[index].x;
    const double dy = depth_projected_points[index].y - corners[index].y;
    depth_squared_reprojection_error += dx * dx + dy * dy;
  }
  const double depth_reprojection_rmse =
      std::sqrt(depth_squared_reprojection_error / corners.size());

  const double inlier_fraction =
      static_cast<double>(plane->inlier_samples) / plane->valid_samples;
  const double plane_score =
      std::clamp(1.0 - plane->rmse_m / depth_plane_max_rmse_m_, 0.0, 1.0);
  const double size_score =
      depth_max_size_error_fraction_ > 0.0
          ? std::clamp(1.0 - max_size_error_fraction /
                                 depth_max_size_error_fraction_,
                       0.0, 1.0)
          : 1.0;
  const double depth_confidence = std::clamp(plane->valid_fraction, 0.0, 1.0) *
                                  std::clamp(inlier_fraction, 0.0, 1.0) *
                                  plane_score * size_score;
  const double common_safety_weight = std::min(
      {upperLimitBlendWeight(depth_reprojection_rmse,
                             max_reprojection_rmse_px_),
       lowerLimitBlendWeight(static_cast<double>(plane->inlier_samples),
                             static_cast<double>(depth_min_valid_samples_)),
       lowerLimitBlendWeight(plane->valid_fraction,
                             depth_min_valid_fraction_)});
  const double translation_blend_weight =
      std::clamp(depth_confidence * common_safety_weight *
                     upperLimitBlendWeight(pnp_translation_delta,
                                           depth_max_pnp_translation_delta_m_),
                 0.0, 1.0);
  const double rotation_blend_weight =
      std::clamp(depth_confidence * common_safety_weight *
                     upperLimitBlendWeight(pnp_rotation_delta,
                                           depth_max_pnp_rotation_delta_deg_),
                 0.0, 1.0);
  if (translation_blend_weight <= 1e-6 && rotation_blend_weight <= 1e-6) {
    return std::nullopt;
  }

  const auto blended_quaternion =
      blendTagNormal(pnp_quaternion, depth_quaternion, rotation_blend_weight);
  const auto blended_rotation =
      rotationMatrixFromQuaternion(blended_quaternion);
  const cv::Mat blended_translation =
      pnp_estimate.tvec * (1.0 - translation_blend_weight) +
      translation_optical * translation_blend_weight;

  cv::Mat rvec;
  cv::Rodrigues(blended_rotation, rvec);
  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(object_points, rvec, blended_translation, camera_matrix,
                    distortion_coeffs, projected_points);
  if (projected_points.size() != corners.size()) {
    return std::nullopt;
  }
  double squared_reprojection_error = 0.0;
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const double dx = projected_points[index].x - corners[index].x;
    const double dy = projected_points[index].y - corners[index].y;
    squared_reprojection_error += dx * dx + dy * dy;
  }

  cv::Mat rotation_camera = blended_rotation;
  cv::Mat translation_camera = blended_translation;
  if (camera_frame_convention_ == CameraFrameConvention::ZedXForward) {
    const auto optical_to_zed = cvOpticalToZedXForwardRotation();
    rotation_camera = optical_to_zed * rotation_camera;
    translation_camera = optical_to_zed * translation_camera;
  }
  tf2::Matrix3x3 basis(
      rotation_camera.at<double>(0, 0), rotation_camera.at<double>(0, 1),
      rotation_camera.at<double>(0, 2), rotation_camera.at<double>(1, 0),
      rotation_camera.at<double>(1, 1), rotation_camera.at<double>(1, 2),
      rotation_camera.at<double>(2, 0), rotation_camera.at<double>(2, 1),
      rotation_camera.at<double>(2, 2));
  tf2::Vector3 origin(translation_camera.at<double>(0, 0),
                      translation_camera.at<double>(1, 0),
                      translation_camera.at<double>(2, 0));

  PoseEstimate estimate;
  estimate.camera_from_tag = tf2::Transform(basis, origin);
  estimate.rvec = rvec;
  estimate.tvec = blended_translation.clone();
  estimate.reprojection_rmse_px =
      std::sqrt(squared_reprojection_error / corners.size());
  estimate.used_depth = true;
  estimate.depth_valid_samples = plane->valid_samples;
  estimate.depth_inlier_samples = plane->inlier_samples;
  estimate.depth_valid_fraction = plane->valid_fraction;
  estimate.depth_plane_rmse_m = plane->rmse_m;
  estimate.depth_size_error_fraction = max_size_error_fraction;
  estimate.depth_pnp_translation_delta_m = pnp_translation_delta;
  estimate.depth_pnp_rotation_delta_deg = pnp_rotation_delta;
  estimate.depth_blend_weight = translation_blend_weight;
  estimate.depth_rotation_blend_weight = rotation_blend_weight;
  estimate.estimator_weight =
      1.0 + translation_blend_weight + rotation_blend_weight;
  return estimate;
}

std::vector<cv::Point3f>
ApriltagFusionNode::markerObjectPoints(double tag_size_m) const {
  const float half_size = static_cast<float>(tag_size_m * 0.5);
  return {cv::Point3f(-half_size, -half_size, 0.0F),
          cv::Point3f(half_size, -half_size, 0.0F),
          cv::Point3f(half_size, half_size, 0.0F),
          cv::Point3f(-half_size, half_size, 0.0F)};
}

tf2::Transform ApriltagFusionNode::transformFromMsg(
    const geometry_msgs::msg::Transform &msg) const {
  tf2::Quaternion rotation(msg.rotation.x, msg.rotation.y, msg.rotation.z,
                           msg.rotation.w);
  if (rotation.length2() <= 0.0) {
    rotation.setValue(0.0, 0.0, 0.0, 1.0);
  } else {
    rotation.normalize();
  }
  return tf2::Transform(
      rotation,
      tf2::Vector3(msg.translation.x, msg.translation.y, msg.translation.z));
}

geometry_msgs::msg::Pose ApriltagFusionNode::poseMsgFromTransform(
    const tf2::Transform &transform) const {
  geometry_msgs::msg::Pose msg;
  const auto origin = transform.getOrigin();
  auto rotation = transform.getRotation();
  rotation.normalize();
  msg.position.x = origin.x();
  msg.position.y = origin.y();
  msg.position.z = origin.z();
  msg.orientation.x = rotation.x();
  msg.orientation.y = rotation.y();
  msg.orientation.z = rotation.z();
  msg.orientation.w = rotation.w();
  return msg;
}

geometry_msgs::msg::Transform ApriltagFusionNode::transformMsgFromTransform(
    const tf2::Transform &transform) const {
  geometry_msgs::msg::Transform msg;
  const auto origin = transform.getOrigin();
  auto rotation = transform.getRotation();
  rotation.normalize();
  msg.translation.x = origin.x();
  msg.translation.y = origin.y();
  msg.translation.z = origin.z();
  msg.rotation.x = rotation.x();
  msg.rotation.y = rotation.y();
  msg.rotation.z = rotation.z();
  msg.rotation.w = rotation.w();
  return msg;
}

int ApriltagFusionNode::parseDictionary(const std::string &dictionary) const {
  const auto normalized = upper(dictionary);
  if (normalized == "APRILTAG_16H5" || normalized == "DICT_APRILTAG_16H5") {
    return cv::aruco::DICT_APRILTAG_16h5;
  }
  if (normalized == "APRILTAG_25H9" || normalized == "DICT_APRILTAG_25H9") {
    return cv::aruco::DICT_APRILTAG_25h9;
  }
  if (normalized == "APRILTAG_36H10" || normalized == "DICT_APRILTAG_36H10") {
    return cv::aruco::DICT_APRILTAG_36h10;
  }
  if (normalized == "APRILTAG_36H11" || normalized == "DICT_APRILTAG_36H11") {
    return cv::aruco::DICT_APRILTAG_36h11;
  }
  throw std::invalid_argument("Unsupported AprilTag dictionary: " + dictionary);
}

ApriltagFusionNode::CameraFrameConvention
ApriltagFusionNode::parseCameraFrameConvention(
    const std::string &convention) const {
  const auto normalized = upper(convention);
  if (normalized == "ZED_X_FORWARD" || normalized == "ZED") {
    return CameraFrameConvention::ZedXForward;
  }
  if (normalized == "ROS_OPTICAL" || normalized == "OPTICAL") {
    return CameraFrameConvention::RosOptical;
  }
  throw std::invalid_argument("Unsupported camera_frame_convention: " +
                              convention);
}

int ApriltagFusionNode::tagId(size_t tag_slot) const {
  return tag_slot == kFrontTagSlot ? front_tag_id_ : back_tag_id_;
}

double ApriltagFusionNode::tagSize(size_t tag_slot) const {
  return tag_slot == kFrontTagSlot ? front_tag_size_m_ : back_tag_size_m_;
}

} // namespace zed_launcher

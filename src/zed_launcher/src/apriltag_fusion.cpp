#include "zed_launcher/apriltag_fusion_node.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
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

ApriltagFusionNode::ApriltagFusionNode(const rclcpp::NodeOptions &options)
    : Node("apriltag_fusion_node", options), tf_buffer_(get_clock()),
      tf_listener_(tf_buffer_) {
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
  max_detection_rate_hz_ =
      declare_parameter<double>("max_detection_rate_hz", 30.0);
  fusion_publish_rate_hz_ =
      declare_parameter<double>("fusion_publish_rate_hz", 30.0);
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
  smoothing_time_constant_sec_ =
      declare_parameter<double>("smoothing_time_constant_sec", 0.10);
  smoothing_reset_sec_ = declare_parameter<double>("smoothing_reset_sec", 0.50);

  learned_tag_separation_m_ = 2.0 * initial_tag_frame_offset_m_;

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
  if (!std::isfinite(max_detection_rate_hz_) || max_detection_rate_hz_ < 0.0) {
    throw std::runtime_error("max_detection_rate_hz must be non-negative");
  }
  if (!std::isfinite(fusion_publish_rate_hz_) ||
      fusion_publish_rate_hz_ <= 0.0) {
    throw std::runtime_error("fusion_publish_rate_hz must be positive");
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
  if (!std::isfinite(smoothing_time_constant_sec_) ||
      smoothing_time_constant_sec_ < 0.0) {
    throw std::runtime_error(
        "smoothing_time_constant_sec must be non-negative");
  }
  if (!std::isfinite(smoothing_reset_sec_) || smoothing_reset_sec_ < 0.0) {
    throw std::runtime_error("smoothing_reset_sec must be non-negative");
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
    camera->callback_group =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    if (publish_debug_images_) {
      camera->debug_pub = create_publisher<sensor_msgs::msg::Image>(
          camera->debug_topic, rclcpp::SensorDataQoS());
    }

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group = camera->callback_group;
    camera->camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera->camera_info_topic, rclcpp::SensorDataQoS(),
        [this, camera_index](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
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
  auto &camera = *cameras_.at(camera_index);
  const auto current_time = now();
  if (shouldSkipFrame(camera, current_time)) {
    return;
  }

  cv::Mat gray;
  if (!toGrayImage(*msg, gray)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Unsupported or invalid image on %s: encoding=%s",
                         camera.image_topic.c_str(), msg->encoding.c_str());
    return;
  }

  cv::Mat debug_image;
  if (publish_debug_images_ && !toBgrImage(*msg, debug_image)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Cannot convert debug image on %s",
                         camera.image_topic.c_str());
  }

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  try {
    camera.detector->detectMarkers(gray, corners, ids);
  } catch (const cv::Exception &ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "AprilTag detection failed on %s: %s",
                         camera.name.c_str(), ex.what());
    if (!debug_image.empty()) {
      publishDebugImage(camera, *msg, debug_image);
    }
    return;
  }

  if (!debug_image.empty()) {
    for (size_t index = 0; index < ids.size() && index < corners.size();
         ++index) {
      if (ids[index] != front_tag_id_ && ids[index] != back_tag_id_) {
        continue;
      }
      const std::vector<std::vector<cv::Point2f>> outline{corners[index]};
      cv::polylines(debug_image, outline, true, cv::Scalar(0, 255, 255), 2,
                    cv::LINE_AA);
      if (!corners[index].empty()) {
        cv::putText(debug_image, "id " + std::to_string(ids[index]),
                    corners[index].front() + cv::Point2f(4.0F, -6.0F),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2,
                    cv::LINE_AA);
      }
    }
  }

  const auto selected = selectMarkers(ids, corners);
  if (!selected[kFrontTagSlot] && !selected[kBackTagSlot]) {
    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                          "AprilTags id=%d/id=%d not detected by %s",
                          front_tag_id_, back_tag_id_, camera.name.c_str());
    if (!debug_image.empty()) {
      publishDebugImage(camera, *msg, debug_image);
    }
    return;
  }

  sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
  {
    std::lock_guard<std::mutex> lock(camera.mutex);
    camera_info = camera.latest_camera_info;
  }
  if (!camera_info) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Waiting for camera info on %s",
                         camera.camera_info_topic.c_str());
    if (!debug_image.empty()) {
      publishDebugImage(camera, *msg, debug_image);
    }
    return;
  }

  const auto camera_matrix = cameraMatrixForImage(*camera_info, *msg);
  if (!camera_matrix) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Camera info on %s has invalid intrinsics",
                         camera.camera_info_topic.c_str());
    if (!debug_image.empty()) {
      publishDebugImage(camera, *msg, debug_image);
    }
    return;
  }
  const auto distortion_coeffs = distortionCoeffs(*camera_info);

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
      pose_estimates[tag_slot] =
          estimateTagInCameraFrame(corners[marker_index], *camera_matrix,
                                   distortion_coeffs, tagSize(tag_slot));
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

    if (!debug_image.empty()) {
      cv::drawFrameAxes(debug_image, *camera_matrix, distortion_coeffs,
                        pose_estimates[tag_slot]->rvec,
                        pose_estimates[tag_slot]->tvec,
                        static_cast<float>(debug_axis_length_m_), 2);
    }
  }

  if (!debug_image.empty()) {
    publishDebugImage(camera, *msg, debug_image);
  }

  if (!pose_estimates[kFrontTagSlot] && !pose_estimates[kBackTagSlot]) {
    return;
  }

  const auto camera_frame_id = !msg->header.frame_id.empty()
                                   ? msg->header.frame_id
                                   : camera_info->header.frame_id;
  if (camera_frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Image and camera_info from %s have empty frame_id",
                         camera.name.c_str());
    return;
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
    return;
  }

  const auto fusion_from_camera =
      transformFromMsg(fusion_from_camera_msg.transform);
  const auto source_stamp =
      rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type());
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
                 0.25);
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
}

bool ApriltagFusionNode::shouldSkipFrame(CameraContext &camera,
                                         const rclcpp::Time &current_time) {
  if (max_detection_rate_hz_ == 0.0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(camera.mutex);
  if (camera.has_last_detection_attempt_time) {
    const double elapsed =
        (current_time - camera.last_detection_attempt_time).seconds();
    if (elapsed >= 0.0 && elapsed < 1.0 / max_detection_rate_hz_) {
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
  rclcpp::Time source_receipt;
  std::vector<uint64_t> source_sequences;
  bool using_stale_fallback = false;

  const bool synchronized_pair =
      front && back &&
      std::abs(
          (front->observation.stamp - back->observation.stamp).seconds()) <=
          sync_tolerance_sec_;
  if (synchronized_pair) {
    updateTagSeparation(*front, *back);
    fusion_from_tag_frame = tagFrameFromBothTags(
        front->observation.fusion_from_tag, back->observation.fusion_from_tag);
    source_stamp = std::max(front->observation.stamp, back->observation.stamp);
    source_receipt = std::max(front->observation.receipt_time,
                              back->observation.receipt_time);
    source_sequences = front->source_sequences;
    source_sequences.insert(source_sequences.end(),
                            back->source_sequences.begin(),
                            back->source_sequences.end());
  } else if (front || back) {
    size_t selected_slot = kFrontTagSlot;
    const FusedTagEstimate *selected = front ? &*front : &*back;
    if (front && back && back->observation.stamp > front->observation.stamp) {
      selected_slot = kBackTagSlot;
      selected = &*back;
    } else if (!front) {
      selected_slot = kBackTagSlot;
    }
    fusion_from_tag_frame = tagFrameFromSingleTag(
        selected_slot, selected->observation.fusion_from_tag);
    source_stamp = selected->observation.stamp;
    source_receipt = selected->observation.receipt_time;
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
    source_receipt = selected.observation.receipt_time;
    using_stale_fallback = true;
  }

  std::sort(source_sequences.begin(), source_sequences.end());
  if (!using_stale_fallback && source_sequences == last_published_sequences_) {
    return;
  }

  const auto smoothed =
      smoothTransform(fusion_from_tag_frame, source_stamp, source_receipt);
  publishResult(smoothed, source_stamp);

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
  const double offset_m = 0.5 * learned_tag_separation_m_;
  const auto origin =
      fusion_from_tag.getOrigin() + tagZAxis(fusion_from_tag) * offset_m;

  auto rotation = fusion_from_tag.getRotation();
  if (tag_slot == kFrontTagSlot) {
    tf2::Quaternion tag1_from_tag0;
    tag1_from_tag0.setRPY(0.0, kPi, 0.0);
    rotation *= tag1_from_tag0;
  }
  rotation.normalize();
  return tf2::Transform(rotation, origin);
}

tf2::Transform ApriltagFusionNode::tagFrameFromBothTags(
    const tf2::Transform &fusion_from_front_tag,
    const tf2::Transform &fusion_from_back_tag) const {
  auto rotation = fusion_from_back_tag.getRotation();
  rotation.normalize();
  const auto midpoint =
      (fusion_from_front_tag.getOrigin() + fusion_from_back_tag.getOrigin()) *
      0.5;
  return tf2::Transform(rotation, midpoint);
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

tf2::Transform
ApriltagFusionNode::smoothTransform(const tf2::Transform &fusion_from_tag_frame,
                                    const rclcpp::Time &source_stamp,
                                    const rclcpp::Time &receipt_time) {
  const bool stamp_went_backwards =
      has_smoothed_transform_ && source_stamp < last_smoothed_stamp_;
  const bool detection_gap_exceeded =
      has_smoothed_transform_ &&
      (receipt_time - last_smoothed_receipt_time_).seconds() >
          smoothing_reset_sec_;
  if (!has_smoothed_transform_ || stamp_went_backwards ||
      detection_gap_exceeded || smoothing_time_constant_sec_ == 0.0) {
    smoothed_fusion_from_tag_frame_ = fusion_from_tag_frame;
  } else {
    const double delta_sec =
        std::max(0.0, (source_stamp - last_smoothed_stamp_).seconds());
    const double alpha =
        delta_sec == 0.0
            ? 1.0
            : std::clamp(
                  1.0 - std::exp(-delta_sec / smoothing_time_constant_sec_),
                  0.0, 1.0);
    const auto smoothed_translation =
        smoothed_fusion_from_tag_frame_.getOrigin() * (1.0 - alpha) +
        fusion_from_tag_frame.getOrigin() * alpha;

    auto previous_rotation = smoothed_fusion_from_tag_frame_.getRotation();
    auto target_rotation = fusion_from_tag_frame.getRotation();
    previous_rotation.normalize();
    target_rotation.normalize();
    if (previous_rotation.dot(target_rotation) < 0.0) {
      target_rotation =
          tf2::Quaternion(-target_rotation.x(), -target_rotation.y(),
                          -target_rotation.z(), -target_rotation.w());
    }
    auto smoothed_rotation = previous_rotation.slerp(target_rotation, alpha);
    smoothed_rotation.normalize();
    smoothed_fusion_from_tag_frame_ =
        tf2::Transform(smoothed_rotation, smoothed_translation);
  }

  last_smoothed_stamp_ = source_stamp;
  last_smoothed_receipt_time_ = receipt_time;
  has_smoothed_transform_ = true;
  return smoothed_fusion_from_tag_frame_;
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
    gray = view.clone();
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

void ApriltagFusionNode::publishDebugImage(
    CameraContext &camera, const sensor_msgs::msg::Image &source,
    const cv::Mat &bgr) const {
  if (!camera.debug_pub || bgr.empty()) {
    return;
  }
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
  camera.debug_pub->publish(std::move(output));
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
    const cv::Mat &distortion_coeffs, double tag_size_m) const {
  if (corners.size() != 4) {
    return std::nullopt;
  }

  const auto object_points = markerObjectPoints(tag_size_m);
  cv::Mat rvec;
  cv::Mat tvec;
  const bool ok =
      cv::solvePnP(object_points, corners, camera_matrix, distortion_coeffs,
                   rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
  if (!ok || tvec.rows * tvec.cols < 3 ||
      !std::isfinite(tvec.at<double>(2, 0)) || tvec.at<double>(2, 0) <= 0.0) {
    return std::nullopt;
  }

  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coeffs,
                    projected_points);
  if (projected_points.size() != corners.size()) {
    return std::nullopt;
  }
  double squared_error_sum = 0.0;
  for (size_t i = 0; i < corners.size(); ++i) {
    const double dx = projected_points[i].x - corners[i].x;
    const double dy = projected_points[i].y - corners[i].y;
    squared_error_sum += dx * dx + dy * dy;
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

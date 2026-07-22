#include "zed_launcher/apriltag_fusion_node.hpp"

#include <algorithm>
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

constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

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
  return (cv::Mat_<double>(3, 3) << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0,
          0.0, -1.0, 0.0);
}

} // namespace

ApriltagFusionNode::ApriltagFusionNode(const rclcpp::NodeOptions &options)
    : Node("apriltag_fusion_node", options), tf_buffer_(get_clock()),
      tf_listener_(tf_buffer_) {
  camera_names_ = declare_parameter<std::vector<std::string>>(
      "camera_names", {"zed_left", "zed_center", "zed_right"});
  fusion_frame_id_ =
      declare_parameter<std::string>("fusion_frame_id", "fusion_world");
  tag_frame_id_ =
      declare_parameter<std::string>("tag_frame_id", "apriltag_0");
  pose_topic_ = declare_parameter<std::string>(
      "pose_topic", "/fusion_world_pose_in_apriltag");
  target_tag_id_ = declare_parameter<int>("target_tag_id", 0);
  tag_size_m_ = declare_parameter<double>("tag_size_m", 0.06);
  max_detection_rate_hz_ =
      declare_parameter<double>("max_detection_rate_hz", 30.0);
  fusion_publish_rate_hz_ =
      declare_parameter<double>("fusion_publish_rate_hz", 30.0);
  tf_lookup_timeout_sec_ =
      declare_parameter<double>("tf_lookup_timeout_sec", 0.05);
  publish_fusion_pose_ =
      declare_parameter<bool>("publish_fusion_pose", true);
  publish_tf_ = declare_parameter<bool>("publish_tf", true);
  const auto dictionary_name =
      declare_parameter<std::string>("dictionary", "APRILTAG_36h11");
  const auto frame_convention = declare_parameter<std::string>(
      "camera_frame_convention", "zed_x_forward");
  corner_refinement_ = declare_parameter<bool>("corner_refinement", true);
  max_observation_age_sec_ =
      declare_parameter<double>("max_observation_age_sec", 0.15);
  sync_tolerance_sec_ =
      declare_parameter<double>("sync_tolerance_sec", 0.05);
  min_marker_area_px2_ =
      declare_parameter<double>("min_marker_area_px2", 64.0);
  max_reprojection_rmse_px_ =
      declare_parameter<double>("max_reprojection_rmse_px", 3.0);
  max_translation_disagreement_m_ = declare_parameter<double>(
      "max_translation_disagreement_m", 0.25);
  max_rotation_disagreement_deg_ = declare_parameter<double>(
      "max_rotation_disagreement_deg", 25.0);
  smoothing_time_constant_sec_ =
      declare_parameter<double>("smoothing_time_constant_sec", 0.10);
  smoothing_reset_sec_ =
      declare_parameter<double>("smoothing_reset_sec", 0.50);

  if (camera_names_.empty()) {
    throw std::runtime_error("camera_names must contain at least one camera");
  }
  std::unordered_set<std::string> unique_camera_names;
  for (const auto &name : camera_names_) {
    if (name.empty()) {
      throw std::runtime_error("camera_names must not contain empty names");
    }
    if (!unique_camera_names.insert(name).second) {
      throw std::runtime_error("camera_names contains duplicate name: " +
                               name);
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
  if (target_tag_id_ < 0) {
    throw std::runtime_error("target_tag_id must be non-negative");
  }
  if (!std::isfinite(tag_size_m_) || tag_size_m_ <= 0.0) {
    throw std::runtime_error("tag_size_m must be positive");
  }
  if (!std::isfinite(max_detection_rate_hz_) ||
      max_detection_rate_hz_ < 0.0) {
    throw std::runtime_error(
        "max_detection_rate_hz must be non-negative");
  }
  if (!std::isfinite(fusion_publish_rate_hz_) ||
      fusion_publish_rate_hz_ <= 0.0) {
    throw std::runtime_error("fusion_publish_rate_hz must be positive");
  }
  if (!std::isfinite(tf_lookup_timeout_sec_) ||
      tf_lookup_timeout_sec_ < 0.0) {
    throw std::runtime_error("tf_lookup_timeout_sec must be non-negative");
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
  if (!publish_fusion_pose_ && !publish_tf_) {
    throw std::runtime_error(
        "At least one of publish_fusion_pose or publish_tf is required");
  }

  camera_frame_convention_ = parseCameraFrameConvention(frame_convention);
  dictionary_ =
      cv::aruco::getPredefinedDictionary(parseDictionary(dictionary_name));

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
    camera->image_topic =
        "/" + camera->name + "/zed_node/rgb/color/rect/image";
    camera->camera_info_topic =
        "/" + camera->name + "/zed_node/rgb/color/rect/camera_info";
    camera->detector_params = cv::aruco::DetectorParameters::create();
    camera->detector_params->cornerRefinementMethod =
        corner_refinement_ ? cv::aruco::CORNER_REFINE_SUBPIX
                           : cv::aruco::CORNER_REFINE_NONE;
    camera->callback_group =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group = camera->callback_group;
    camera->camera_info_sub =
        create_subscription<sensor_msgs::msg::CameraInfo>(
            camera->camera_info_topic, rclcpp::SensorDataQoS(),
            [this, camera_index](
                sensor_msgs::msg::CameraInfo::SharedPtr msg) {
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
    cameras_.push_back(std::move(camera));
  }

  fusion_callback_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  const auto timer_period =
      std::chrono::duration<double>(1.0 / fusion_publish_rate_hz_);
  fusion_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
      [this]() { fuseAndPublish(); }, fusion_callback_group_);

  RCLCPP_INFO(
      get_logger(),
      "apriltag_fusion_node ready with %zu cameras: dynamic TF %s -> %s",
      cameras_.size(), tag_frame_id_.c_str(), fusion_frame_id_.c_str());
}

void ApriltagFusionNode::handleCameraInfo(
    size_t camera_index,
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  auto &camera = *cameras_.at(camera_index);
  std::lock_guard<std::mutex> lock(camera.mutex);
  camera.latest_camera_info = msg;
}

void ApriltagFusionNode::handleImage(
    size_t camera_index,
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  auto &camera = *cameras_.at(camera_index);
  const auto current_time = now();
  if (shouldSkipFrame(camera, current_time)) {
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
    return;
  }

  cv::Mat gray;
  if (!toGrayImage(*msg, gray)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Unsupported or invalid image on %s: encoding=%s",
                         camera.image_topic.c_str(), msg->encoding.c_str());
    return;
  }

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  try {
    cv::aruco::detectMarkers(gray, dictionary_, corners, ids,
                             camera.detector_params);
  } catch (const cv::Exception &ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "AprilTag detection failed on %s: %s",
                         camera.name.c_str(), ex.what());
    return;
  }
  const auto selected = selectMarker(ids, corners);
  if (!selected) {
    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                          "Target AprilTag id=%d not detected by %s",
                          target_tag_id_, camera.name.c_str());
    return;
  }

  const double area_px2 = markerArea(corners[*selected]);
  if (!std::isfinite(area_px2) || area_px2 < min_marker_area_px2_) {
    RCLCPP_DEBUG_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Rejected small AprilTag id=%d from %s: area=%.2f px^2",
        target_tag_id_, camera.name.c_str(), area_px2);
    return;
  }

  const auto camera_matrix = cameraMatrixForImage(*camera_info, *msg);
  if (!camera_matrix) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Camera info on %s has invalid intrinsics",
                         camera.camera_info_topic.c_str());
    return;
  }

  std::optional<PoseEstimate> pose_estimate;
  try {
    pose_estimate = estimateTagInCameraFrame(
        corners[*selected], *camera_matrix, distortionCoeffs(*camera_info));
  } catch (const cv::Exception &ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "AprilTag pose estimation failed on %s: %s",
                         camera.name.c_str(), ex.what());
    return;
  }
  if (!pose_estimate) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "solvePnP failed for AprilTag id=%d from %s",
                         target_tag_id_, camera.name.c_str());
    return;
  }
  if (!std::isfinite(pose_estimate->reprojection_rmse_px) ||
      pose_estimate->reprojection_rmse_px > max_reprojection_rmse_px_) {
    RCLCPP_DEBUG_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Rejected AprilTag id=%d from %s: reprojection RMSE=%.3f px",
        target_tag_id_, camera.name.c_str(),
        pose_estimate->reprojection_rmse_px);
    return;
  }

  const auto camera_frame_id =
      !msg->header.frame_id.empty() ? msg->header.frame_id
                                    : camera_info->header.frame_id;
  if (camera_frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
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
        camera_frame_id.c_str(), fusion_frame_id_.c_str(),
        camera.name.c_str(), ex.what());
    return;
  }

  const auto fusion_from_camera =
      transformFromMsg(fusion_from_camera_msg.transform);
  Observation observation;
  observation.fusion_from_tag =
      fusion_from_camera * pose_estimate->camera_from_tag;
  observation.stamp =
      rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type());
  observation.receipt_time = current_time;
  observation.marker_area_px2 = area_px2;
  observation.reprojection_rmse_px =
      pose_estimate->reprojection_rmse_px;
  observation.quality =
      area_px2 /
      std::max(pose_estimate->reprojection_rmse_px *
                   pose_estimate->reprojection_rmse_px,
               0.25);
  observation.sequence = next_observation_sequence_.fetch_add(1);

  {
    std::lock_guard<std::mutex> lock(camera.mutex);
    camera.latest_observation = std::move(observation);
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

void ApriltagFusionNode::fuseAndPublish() {
  const auto current_time = now();
  std::vector<Observation> observations;
  observations.reserve(cameras_.size());

  for (const auto &camera_ptr : cameras_) {
    std::lock_guard<std::mutex> lock(camera_ptr->mutex);
    if (!camera_ptr->latest_observation) {
      continue;
    }
    const double age =
        (current_time - camera_ptr->latest_observation->receipt_time).seconds();
    if (age >= 0.0 && age <= max_observation_age_sec_) {
      observations.push_back(*camera_ptr->latest_observation);
    }
  }
  if (observations.empty()) {
    return;
  }

  const auto newest_stamp = std::max_element(
      observations.begin(), observations.end(),
      [](const Observation &lhs, const Observation &rhs) {
        return lhs.stamp < rhs.stamp;
      })->stamp;
  observations.erase(
      std::remove_if(
          observations.begin(), observations.end(),
          [this, &newest_stamp](const Observation &observation) {
            return (newest_stamp - observation.stamp).seconds() >
                   sync_tolerance_sec_;
          }),
      observations.end());
  if (observations.empty()) {
    return;
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
    return;
  }

  std::vector<uint64_t> selected_sequences;
  selected_sequences.reserve(best_cluster.size());
  rclcpp::Time source_stamp = observations[best_cluster.front()].stamp;
  rclcpp::Time source_receipt =
      observations[best_cluster.front()].receipt_time;
  for (const auto index : best_cluster) {
    selected_sequences.push_back(observations[index].sequence);
    source_stamp = std::max(source_stamp, observations[index].stamp);
    source_receipt =
        std::max(source_receipt, observations[index].receipt_time);
  }
  std::sort(selected_sequences.begin(), selected_sequences.end());
  if (selected_sequences == last_published_sequences_) {
    return;
  }

  const auto fused_fusion_from_tag =
      fuseObservations(observations, best_cluster);
  const auto smoothed_fusion_from_tag =
      smoothTransform(fused_fusion_from_tag, source_stamp, source_receipt);
  const auto tag_from_fusion = smoothed_fusion_from_tag.inverse();

  if (publish_fusion_pose_ && pose_pub_) {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = source_stamp;
    pose_msg.header.frame_id = tag_frame_id_;
    pose_msg.pose = poseMsgFromTransform(tag_from_fusion);
    pose_pub_->publish(pose_msg);
  }

  if (publish_tf_ && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = source_stamp;
    tf_msg.header.frame_id = tag_frame_id_;
    tf_msg.child_frame_id = fusion_frame_id_;
    tf_msg.transform = transformMsgFromTransform(tag_from_fusion);
    tf_broadcaster_->sendTransform(tf_msg);
  }

  last_published_sequences_ = std::move(selected_sequences);
}

bool ApriltagFusionNode::observationsAgree(const Observation &lhs,
                                           const Observation &rhs) const {
  const double translation_distance =
      lhs.fusion_from_tag.getOrigin().distance(
          rhs.fusion_from_tag.getOrigin());
  if (translation_distance > max_translation_disagreement_m_) {
    return false;
  }

  auto lhs_rotation = lhs.fusion_from_tag.getRotation();
  auto rhs_rotation = rhs.fusion_from_tag.getRotation();
  lhs_rotation.normalize();
  rhs_rotation.normalize();
  const double quaternion_dot = std::clamp(
      std::abs(lhs_rotation.dot(rhs_rotation)), 0.0, 1.0);
  const double rotation_difference_deg =
      2.0 * std::acos(quaternion_dot) * kRadiansToDegrees;
  return rotation_difference_deg <= max_rotation_disagreement_deg_;
}

tf2::Transform ApriltagFusionNode::fuseObservations(
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

  for (const auto index : selected_indices) {
    const auto &observation = observations[index];
    const double weight =
        std::isfinite(observation.quality) && observation.quality > 0.0
            ? observation.quality
            : 1.0;
    weighted_translation +=
        observation.fusion_from_tag.getOrigin() * weight;

    auto rotation = observation.fusion_from_tag.getRotation();
    rotation.normalize();
    if (reference_rotation.dot(rotation) < 0.0) {
      rotation = tf2::Quaternion(-rotation.x(), -rotation.y(),
                                 -rotation.z(), -rotation.w());
    }
    weighted_qx += rotation.x() * weight;
    weighted_qy += rotation.y() * weight;
    weighted_qz += rotation.z() * weight;
    weighted_qw += rotation.w() * weight;
    total_weight += weight;
  }

  weighted_translation /= total_weight;
  tf2::Quaternion weighted_rotation(weighted_qx / total_weight,
                                    weighted_qy / total_weight,
                                    weighted_qz / total_weight,
                                    weighted_qw / total_weight);
  if (weighted_rotation.length2() <= 0.0) {
    weighted_rotation = reference_rotation;
  } else {
    weighted_rotation.normalize();
  }
  return tf2::Transform(weighted_rotation, weighted_translation);
}

tf2::Transform ApriltagFusionNode::smoothTransform(
    const tf2::Transform &fusion_from_tag,
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
    smoothed_fusion_from_tag_ = fusion_from_tag;
  } else {
    const double delta_sec =
        std::max(0.0, (source_stamp - last_smoothed_stamp_).seconds());
    const double alpha =
        delta_sec == 0.0
            ? 1.0
            : std::clamp(
                  1.0 -
                      std::exp(-delta_sec / smoothing_time_constant_sec_),
                  0.0, 1.0);
    const auto smoothed_translation =
        smoothed_fusion_from_tag_.getOrigin() * (1.0 - alpha) +
        fusion_from_tag.getOrigin() * alpha;

    auto previous_rotation = smoothed_fusion_from_tag_.getRotation();
    auto target_rotation = fusion_from_tag.getRotation();
    previous_rotation.normalize();
    target_rotation.normalize();
    if (previous_rotation.dot(target_rotation) < 0.0) {
      target_rotation =
          tf2::Quaternion(-target_rotation.x(), -target_rotation.y(),
                          -target_rotation.z(), -target_rotation.w());
    }
    auto smoothed_rotation = previous_rotation.slerp(target_rotation, alpha);
    smoothed_rotation.normalize();
    smoothed_fusion_from_tag_ =
        tf2::Transform(smoothed_rotation, smoothed_translation);
  }

  last_smoothed_stamp_ = source_stamp;
  last_smoothed_receipt_time_ = receipt_time;
  has_smoothed_transform_ = true;
  return smoothed_fusion_from_tag_;
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

std::optional<cv::Mat> ApriltagFusionNode::cameraMatrixForImage(
    const sensor_msgs::msg::CameraInfo &camera_info,
    const sensor_msgs::msg::Image &image) const {
  if (camera_info.k[0] <= 0.0 || camera_info.k[4] <= 0.0) {
    return std::nullopt;
  }

  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << camera_info.k[0], camera_info.k[1],
       camera_info.k[2], camera_info.k[3], camera_info.k[4],
       camera_info.k[5], camera_info.k[6], camera_info.k[7],
       camera_info.k[8]);

  if (camera_info.width > 0 && camera_info.height > 0 && image.width > 0 &&
      image.height > 0 &&
      (camera_info.width != image.width ||
       camera_info.height != image.height)) {
    const double scale_x =
        static_cast<double>(image.width) /
        static_cast<double>(camera_info.width);
    const double scale_y =
        static_cast<double>(image.height) /
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

std::optional<size_t> ApriltagFusionNode::selectMarker(
    const std::vector<int> &ids,
    const std::vector<std::vector<cv::Point2f>> &corners) const {
  std::optional<size_t> best_index;
  double best_area = 0.0;

  for (size_t i = 0; i < ids.size() && i < corners.size(); ++i) {
    if (ids[i] != target_tag_id_) {
      continue;
    }
    const double area = markerArea(corners[i]);
    if (!best_index || area > best_area) {
      best_index = i;
      best_area = area;
    }
  }

  return best_index;
}

std::optional<ApriltagFusionNode::PoseEstimate>
ApriltagFusionNode::estimateTagInCameraFrame(
    const std::vector<cv::Point2f> &corners,
    const cv::Mat &camera_matrix,
    const cv::Mat &distortion_coeffs) const {
  if (corners.size() != 4) {
    return std::nullopt;
  }

  const auto object_points = markerObjectPoints();
  cv::Mat rvec;
  cv::Mat tvec;
  const bool ok = cv::solvePnP(object_points, corners, camera_matrix,
                               distortion_coeffs, rvec, tvec, false,
                               cv::SOLVEPNP_ITERATIVE);
  if (!ok || tvec.rows * tvec.cols < 3 ||
      !std::isfinite(tvec.at<double>(2, 0)) ||
      tvec.at<double>(2, 0) <= 0.0) {
    return std::nullopt;
  }

  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix,
                    distortion_coeffs, projected_points);
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
  cv::Mat translation_camera =
      (cv::Mat_<double>(3, 1) << tvec.at<double>(0, 0),
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
  estimate.reprojection_rmse_px =
      std::sqrt(squared_error_sum / static_cast<double>(corners.size()));
  return estimate;
}

std::vector<cv::Point3f> ApriltagFusionNode::markerObjectPoints() const {
  const float half_size = static_cast<float>(tag_size_m_ * 0.5);
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
      rotation, tf2::Vector3(msg.translation.x, msg.translation.y,
                             msg.translation.z));
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

cv::aruco::PREDEFINED_DICTIONARY_NAME
ApriltagFusionNode::parseDictionary(const std::string &dictionary) const {
  const auto normalized = upper(dictionary);
  if (normalized == "APRILTAG_16H5" ||
      normalized == "DICT_APRILTAG_16H5") {
    return cv::aruco::DICT_APRILTAG_16h5;
  }
  if (normalized == "APRILTAG_25H9" ||
      normalized == "DICT_APRILTAG_25H9") {
    return cv::aruco::DICT_APRILTAG_25h9;
  }
  if (normalized == "APRILTAG_36H10" ||
      normalized == "DICT_APRILTAG_36H10") {
    return cv::aruco::DICT_APRILTAG_36h10;
  }
  if (normalized == "APRILTAG_36H11" ||
      normalized == "DICT_APRILTAG_36H11") {
    return cv::aruco::DICT_APRILTAG_36h11;
  }
  throw std::invalid_argument("Unsupported AprilTag dictionary: " +
                              dictionary);
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

} // namespace zed_launcher

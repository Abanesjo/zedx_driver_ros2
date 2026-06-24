#include "zed_launcher/apriltag_fusion_node.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.hpp>

namespace zed_launcher {
namespace {

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
  const auto camera_name =
      declare_parameter<std::string>("camera_name", "zed_left");
  const auto default_image_topic =
      "/" + camera_name + "/zed_node/rgb/color/rect/image";
  const auto default_camera_info_topic =
      "/" + camera_name + "/zed_node/rgb/color/rect/camera_info";

  image_topic_ =
      declare_parameter<std::string>("image_topic", default_image_topic);
  camera_info_topic_ = declare_parameter<std::string>(
      "camera_info_topic", default_camera_info_topic);
  fusion_frame_id_ =
      declare_parameter<std::string>("fusion_frame_id", "fusion_world");
  tag_frame_id_ =
      declare_parameter<std::string>("tag_frame_id", "apriltag_0");
  pelvis_frame_id_ =
      declare_parameter<std::string>("pelvis_frame_id", "pelvis");
  pose_topic_ =
      declare_parameter<std::string>("pose_topic", "/tag_pose_fusion_world");
  target_tag_id_ = declare_parameter<int>("target_tag_id", 0);
  tag_size_m_ = declare_parameter<double>("tag_size_m", 0.06);
  max_publish_rate_hz_ =
      declare_parameter<double>("max_publish_rate_hz", 30.0);
  tf_lookup_timeout_sec_ =
      declare_parameter<double>("tf_lookup_timeout_sec", 0.05);
  publish_tag_pose_ = declare_parameter<bool>("publish_tag_pose", true);
  publish_tf_ = declare_parameter<bool>("publish_tf", true);
  publish_tag_to_pelvis_tf_ =
      declare_parameter<bool>("publish_tag_to_pelvis_tf", false);
  const auto tag_to_pelvis_x_m =
      declare_parameter<double>("tag_to_pelvis_x_m", 0.0);
  const auto tag_to_pelvis_y_m =
      declare_parameter<double>("tag_to_pelvis_y_m", 0.0);
  const auto tag_to_pelvis_z_m =
      declare_parameter<double>("tag_to_pelvis_z_m", 0.0);
  const auto tag_to_pelvis_qx =
      declare_parameter<double>("tag_to_pelvis_qx", 0.0);
  const auto tag_to_pelvis_qy =
      declare_parameter<double>("tag_to_pelvis_qy", 0.0);
  const auto tag_to_pelvis_qz =
      declare_parameter<double>("tag_to_pelvis_qz", 0.0);
  const auto tag_to_pelvis_qw =
      declare_parameter<double>("tag_to_pelvis_qw", 1.0);

  const auto dictionary_name =
      declare_parameter<std::string>("dictionary", "APRILTAG_36h11");
  const auto frame_convention = declare_parameter<std::string>(
      "camera_frame_convention", "zed_x_forward");
  const auto corner_refinement =
      declare_parameter<bool>("corner_refinement", true);

  if (target_tag_id_ < 0) {
    throw std::runtime_error("target_tag_id must be non-negative");
  }
  if (tag_size_m_ <= 0.0) {
    throw std::runtime_error("tag_size_m must be positive");
  }
  if (max_publish_rate_hz_ < 0.0) {
    throw std::runtime_error("max_publish_rate_hz must be non-negative");
  }
  if (tf_lookup_timeout_sec_ < 0.0) {
    throw std::runtime_error("tf_lookup_timeout_sec must be non-negative");
  }
  if (tag_frame_id_.empty()) {
    throw std::runtime_error("tag_frame_id must not be empty");
  }
  if (pelvis_frame_id_.empty()) {
    throw std::runtime_error("pelvis_frame_id must not be empty");
  }
  if (tag_frame_id_ == pelvis_frame_id_) {
    throw std::runtime_error("tag_frame_id and pelvis_frame_id must differ");
  }
  if (!publish_tag_pose_ && !publish_tf_) {
    throw std::runtime_error(
        "At least one of publish_tag_pose or publish_tf is required");
  }

  tf2::Quaternion tag_to_pelvis_rotation(
      tag_to_pelvis_qx, tag_to_pelvis_qy, tag_to_pelvis_qz,
      tag_to_pelvis_qw);
  if (tag_to_pelvis_rotation.length2() <= 0.0) {
    throw std::runtime_error(
        "tag_to_pelvis quaternion must have non-zero length");
  }
  tag_to_pelvis_rotation.normalize();
  tag_to_pelvis_ = tf2::Transform(
      tag_to_pelvis_rotation,
      tf2::Vector3(tag_to_pelvis_x_m, tag_to_pelvis_y_m, tag_to_pelvis_z_m));

  camera_frame_convention_ = parseCameraFrameConvention(frame_convention);
  dictionary_ =
      cv::aruco::getPredefinedDictionary(parseDictionary(dictionary_name));
  detector_params_ = cv::aruco::DetectorParameters::create();
  detector_params_->cornerRefinementMethod =
      corner_refinement ? cv::aruco::CORNER_REFINE_SUBPIX
                        : cv::aruco::CORNER_REFINE_NONE;

  if (publish_tag_pose_) {
    pose_pub_ =
        create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, 10);
  }
  if (publish_tf_) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }
  if (publish_tag_to_pelvis_tf_) {
    static_tf_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
    publishTagToPelvisTransform();
  }

  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        handleCameraInfo(std::move(msg));
      });
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        handleImage(std::move(msg));
      });

  RCLCPP_INFO(get_logger(),
              "apriltag_fusion_node ready: %s + %s -> %s, TF %s -> %s",
              image_topic_.c_str(), camera_info_topic_.c_str(),
              pose_topic_.c_str(), fusion_frame_id_.c_str(),
              tag_frame_id_.c_str());
  if (publish_tag_to_pelvis_tf_) {
    RCLCPP_INFO(get_logger(), "Publishing static TF %s -> %s",
                tag_frame_id_.c_str(), pelvis_frame_id_.c_str());
  }
}

void ApriltagFusionNode::handleCameraInfo(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(camera_info_mutex_);
  latest_camera_info_ = msg;
}

void ApriltagFusionNode::handleImage(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (shouldSkipFrame()) {
    return;
  }

  sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_info = latest_camera_info_;
  }
  if (!camera_info) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Waiting for camera info on %s",
                         camera_info_topic_.c_str());
    return;
  }

  cv::Mat gray;
  if (!toGrayImage(*msg, gray)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Unsupported or invalid image on %s: encoding=%s",
                         image_topic_.c_str(), msg->encoding.c_str());
    return;
  }

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  cv::aruco::detectMarkers(gray, dictionary_, corners, ids, detector_params_);
  const auto selected = selectMarker(ids, corners);
  if (!selected) {
    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                          "Target AprilTag id=%d not detected",
                          target_tag_id_);
    return;
  }

  const auto camera_matrix = cameraMatrixForImage(*camera_info, *msg);
  if (!camera_matrix) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Camera info on %s has invalid intrinsics",
                         camera_info_topic_.c_str());
    return;
  }

  const auto tag_in_camera = estimateTagInCameraFrame(
      corners[*selected], *camera_matrix, distortionCoeffs(*camera_info));
  if (!tag_in_camera) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "solvePnP failed for AprilTag id=%d", target_tag_id_);
    return;
  }

  const auto camera_frame_id =
      !msg->header.frame_id.empty()
          ? msg->header.frame_id
          : camera_info->header.frame_id;
  if (camera_frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Image and camera_info have empty frame_id");
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
        "Cannot transform %s -> %s for AprilTag pose: %s",
        camera_frame_id.c_str(), fusion_frame_id_.c_str(), ex.what());
    return;
  }

  const auto fusion_from_camera =
      transformFromMsg(fusion_from_camera_msg.transform);
  const auto fusion_from_tag = fusion_from_camera * (*tag_in_camera);

  if (publish_tag_pose_ && pose_pub_) {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = msg->header.stamp;
    pose_msg.header.frame_id = fusion_frame_id_;
    pose_msg.pose = poseMsgFromTransform(fusion_from_tag);
    pose_pub_->publish(pose_msg);
  }

  if (publish_tf_ && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = msg->header.stamp;
    tf_msg.header.frame_id = fusion_frame_id_;
    tf_msg.child_frame_id = tag_frame_id_;
    tf_msg.transform = transformMsgFromTransform(fusion_from_tag);
    tf_broadcaster_->sendTransform(tf_msg);
  }
}

bool ApriltagFusionNode::shouldSkipFrame() {
  if (max_publish_rate_hz_ == 0.0) {
    return false;
  }

  const auto current_time = now();
  if (has_last_processed_time_ &&
      (current_time - last_processed_time_).seconds() <
          1.0 / max_publish_rate_hz_) {
    return true;
  }

  last_processed_time_ = current_time;
  has_last_processed_time_ = true;
  return false;
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

  if (camera_info.width > 0 && camera_info.height > 0 &&
      image.width > 0 && image.height > 0 &&
      (camera_info.width != image.width || camera_info.height != image.height)) {
    const double scale_x =
        static_cast<double>(image.width) / static_cast<double>(camera_info.width);
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

std::optional<tf2::Transform> ApriltagFusionNode::estimateTagInCameraFrame(
    const std::vector<cv::Point2f> &corners,
    const cv::Mat &camera_matrix,
    const cv::Mat &distortion_coeffs) const {
  if (corners.size() != 4) {
    return std::nullopt;
  }

  cv::Mat rvec;
  cv::Mat tvec;
  const bool ok = cv::solvePnP(markerObjectPoints(), corners, camera_matrix,
                               distortion_coeffs, rvec, tvec, false,
                               cv::SOLVEPNP_ITERATIVE);
  if (!ok) {
    return std::nullopt;
  }

  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);
  cv::Mat translation_camera =
      (cv::Mat_<double>(3, 1) << tvec.at<double>(0, 0), tvec.at<double>(1, 0),
       tvec.at<double>(2, 0));
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

  return tf2::Transform(basis, origin);
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

void ApriltagFusionNode::publishTagToPelvisTransform() {
  if (!static_tf_broadcaster_) {
    return;
  }

  geometry_msgs::msg::TransformStamped msg;
  msg.header.stamp = now();
  msg.header.frame_id = tag_frame_id_;
  msg.child_frame_id = pelvis_frame_id_;
  msg.transform = transformMsgFromTransform(tag_to_pelvis_);
  static_tf_broadcaster_->sendTransform(msg);
}

cv::aruco::PREDEFINED_DICTIONARY_NAME
ApriltagFusionNode::parseDictionary(const std::string &dictionary) const {
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

} // namespace zed_launcher

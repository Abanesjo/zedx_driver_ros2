#include "zed_launcher/zed_body_fusion_node.hpp"

#include "zed_launcher/skeleton_draw_utils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <sensor_msgs/distortion_models.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace zed_launcher {
namespace {
constexpr auto kRosCoordinateSystem =
    sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
constexpr auto kRosUnits = sl::UNIT::METER;

std::string upper(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

sl::BODY_TRACKING_MODEL parseBodyModel(const std::string &value) {
  const auto normalized = upper(value);
  if (normalized == "HUMAN_BODY_FAST") {
    return sl::BODY_TRACKING_MODEL::HUMAN_BODY_FAST;
  }
  if (normalized == "HUMAN_BODY_MEDIUM") {
    return sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
  }
  if (normalized == "HUMAN_BODY_ACCURATE") {
    return sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
  }
  throw std::invalid_argument("Unsupported body_model: " + value);
}

sl::BODY_FORMAT parseBodyFormat(const std::string &value) {
  const auto normalized = upper(value);
  if (normalized == "BODY_18") {
    return sl::BODY_FORMAT::BODY_18;
  }
  if (normalized == "BODY_34") {
    return sl::BODY_FORMAT::BODY_34;
  }
  if (normalized == "BODY_38") {
    return sl::BODY_FORMAT::BODY_38;
  }
  throw std::invalid_argument("Unsupported body_format: " + value);
}

sl::DEPTH_MODE parseDepthMode(const std::string &value) {
  const auto normalized = upper(value);
  if (normalized == "PERFORMANCE") {
    return sl::DEPTH_MODE::PERFORMANCE;
  }
  if (normalized == "QUALITY") {
    return sl::DEPTH_MODE::QUALITY;
  }
  if (normalized == "ULTRA") {
    return sl::DEPTH_MODE::ULTRA;
  }
  if (normalized == "NEURAL_LIGHT") {
    return sl::DEPTH_MODE::NEURAL_LIGHT;
  }
  if (normalized == "NEURAL") {
    return sl::DEPTH_MODE::NEURAL;
  }
  if (normalized == "NEURAL_PLUS") {
    return sl::DEPTH_MODE::NEURAL_PLUS;
  }
  throw std::invalid_argument("Unsupported depth_mode: " + value);
}

sl::RESOLUTION parseResolution(const std::string &value) {
  const auto normalized = upper(value);
  if (normalized == "AUTO") {
    return sl::RESOLUTION::AUTO;
  }
  if (normalized == "HD1200") {
    return sl::RESOLUTION::HD1200;
  }
  if (normalized == "HD1080") {
    return sl::RESOLUTION::HD1080;
  }
  if (normalized == "HD720") {
    return sl::RESOLUTION::HD720;
  }
  if (normalized == "SVGA") {
    return sl::RESOLUTION::SVGA;
  }
  if (normalized == "VGA") {
    return sl::RESOLUTION::VGA;
  }
  throw std::invalid_argument("Unsupported camera_resolution: " + value);
}

sl::FUSION_REFERENCE_FRAME parseFusionReferenceFrame(const std::string &value) {
  const auto normalized = upper(value);
  if (normalized == "BASELINK") {
    return sl::FUSION_REFERENCE_FRAME::BASELINK;
  }
  if (normalized == "WORLD") {
    return sl::FUSION_REFERENCE_FRAME::WORLD;
  }
  throw std::invalid_argument("Unsupported fusion_reference_frame: " + value);
}

std::string zedErrorToString(sl::ERROR_CODE code) {
  std::ostringstream out;
  out << sl::toString(code);
  return out.str();
}

std::string fusionErrorToString(sl::FUSION_ERROR_CODE code) {
  std::ostringstream out;
  out << sl::toString(code);
  return out.str();
}

template <typename RosArray, typename SlVector>
void copyVector3(RosArray &dest, const SlVector &src) {
  const auto count = std::min<size_t>(dest.size(), 3);
  for (size_t i = 0; i < count; ++i) {
    dest[i] = static_cast<float>(src[i]);
  }
}

template <typename RosArray, typename SlVector>
void copyFlat(RosArray &dest, const SlVector &src) {
  const auto count = std::min<size_t>(std::size(dest), std::size(src));
  for (size_t i = 0; i < count; ++i) {
    dest[i] = static_cast<float>(src[i]);
  }
}

template <typename RosBox, typename SlBox>
void copyBox2d(RosBox &dest, const SlBox &src) {
  const auto count = std::min<size_t>(dest.corners.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.corners[i].kp[0] = static_cast<unsigned int>(src[i][0]);
    dest.corners[i].kp[1] = static_cast<unsigned int>(src[i][1]);
  }
}

template <typename RosBox, typename SlBox>
void copyBox3d(RosBox &dest, const SlBox &src) {
  const auto count = std::min<size_t>(dest.corners.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.corners[i].kp[0] = static_cast<float>(src[i][0]);
    dest.corners[i].kp[1] = static_cast<float>(src[i][1]);
    dest.corners[i].kp[2] = static_cast<float>(src[i][2]);
  }
}

template <typename RosSkeleton, typename SlKeypoints>
void copySkeleton2d(RosSkeleton &dest, const SlKeypoints &src) {
  const auto count = std::min<size_t>(dest.keypoints.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.keypoints[i].kp[0] = static_cast<float>(src[i][0]);
    dest.keypoints[i].kp[1] = static_cast<float>(src[i][1]);
  }
}

template <typename RosSkeleton, typename SlKeypoints>
void copySkeleton3d(RosSkeleton &dest, const SlKeypoints &src) {
  const auto count = std::min<size_t>(dest.keypoints.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.keypoints[i].kp[0] = static_cast<float>(src[i][0]);
    dest.keypoints[i].kp[1] = static_cast<float>(src[i][1]);
    dest.keypoints[i].kp[2] = static_cast<float>(src[i][2]);
  }
}

} // namespace

ZedBodyFusionNode::ZedBodyFusionNode(const rclcpp::NodeOptions &options)
    : Node("zed_body_fusion_node", options) {
  loadParameters();

  normalizeMode();

  if (role_ != "local") {
    throw std::runtime_error("zed_body_fusion_node currently runs local "
                             "in-process Fusion only; requested role: " +
                             role_);
  }

  pub_bodies_ = create_publisher<zed_msgs::msg::ObjectsStamped>(
      output_topic_, rclcpp::SensorDataQoS());
  static_tf_broadcaster_ =
      std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

  RCLCPP_INFO(get_logger(), "Loading ZED Fusion configuration: %s",
              fusion_config_path_.c_str());
  fusion_configs_ = sl::readFusionConfigurationFile(
      fusion_config_path_, kRosCoordinateSystem, kRosUnits);
  validateFusionConfigurations();

  configureRuntimeParameters();
  publishStaticCameraTransforms();

  try {
    startCameraPublishers();
    startFusion();
  } catch (...) {
    shutdown();
    throw;
  }

  const auto timer_period =
      std::chrono::duration<double>(1.0 / fusion_publish_rate_hz_);
  fusion_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
      [this]() { processFusion(); });
}

ZedBodyFusionNode::~ZedBodyFusionNode() { shutdown(); }

void ZedBodyFusionNode::loadParameters() {
  role_ = declare_parameter<std::string>("role", "local");
  input_mode_ = declare_parameter<std::string>("input_mode", "stream");
  fusion_config_path_ =
      declare_parameter<std::string>("fusion_config_path", "");
  output_topic_ =
      declare_parameter<std::string>("output_topic", "body_trk/skeletons");
  publish_frame_id_ =
      declare_parameter<std::string>("publish_frame_id", "fusion_world");
  stream_address_ = declare_parameter<std::string>("stream_address", "");

  const auto camera_names = declare_parameter<std::vector<std::string>>(
      "camera_names",
      std::vector<std::string>{"zed_left", "zed_center", "zed_right"});
  const auto camera_serials = declare_parameter<std::vector<int64_t>>(
      "camera_serials",
      std::vector<int64_t>{41235597, 46229474, 49967328});
  const auto stream_ports = declare_parameter<std::vector<int64_t>>(
      "stream_ports", std::vector<int64_t>{30000, 30004, 30002});

  if (camera_names.empty()) {
    throw std::runtime_error("camera_names must contain at least one camera");
  }
  if (camera_names.size() != camera_serials.size() ||
      camera_names.size() != stream_ports.size()) {
    throw std::runtime_error(
        "camera_names, camera_serials, and stream_ports must have equal "
        "lengths");
  }

  std::unordered_set<std::string> unique_names;
  std::unordered_set<int64_t> unique_serials;
  std::unordered_set<int64_t> unique_ports;
  camera_specs_.reserve(camera_names.size());
  for (size_t idx = 0; idx < camera_names.size(); ++idx) {
    const auto &name = camera_names[idx];
    const auto serial = camera_serials[idx];
    const auto port = stream_ports[idx];

    if (name.empty()) {
      throw std::runtime_error("camera_names entries must be non-empty");
    }
    if (serial <= 0 ||
        serial >
            static_cast<int64_t>(std::numeric_limits<unsigned int>::max())) {
      throw std::runtime_error(
          "camera_serials entries must be positive valid ZED serials");
    }
    if (port <= 0 || port > 65534 || port % 2 != 0) {
      throw std::runtime_error(
          "stream_ports entries must be even and in the range [2, 65534]");
    }
    if (!unique_names.insert(name).second) {
      throw std::runtime_error("camera_names entries must be unique: " + name);
    }
    if (!unique_serials.insert(serial).second) {
      throw std::runtime_error("camera_serials entries must be unique: " +
                               std::to_string(serial));
    }
    if (!unique_ports.insert(port).second) {
      throw std::runtime_error("stream_ports entries must be unique: " +
                               std::to_string(port));
    }

    camera_specs_.push_back(CameraSpec{name,
                                       static_cast<unsigned int>(serial),
                                       static_cast<int>(port)});
  }

  body_model_ = parseBodyModel(
      declare_parameter<std::string>("body_model", "HUMAN_BODY_ACCURATE"));
  body_format_ =
      parseBodyFormat(declare_parameter<std::string>("body_format", "BODY_38"));
  depth_mode_ = parseDepthMode(
      declare_parameter<std::string>("depth_mode", "NEURAL_LIGHT"));
  camera_resolution_ = parseResolution(
      declare_parameter<std::string>("camera_resolution", "HD1080"));
  fusion_reference_frame_ = parseFusionReferenceFrame(
      declare_parameter<std::string>("fusion_reference_frame", "BASELINK"));

  confidence_threshold_ =
      declare_parameter<double>("confidence_threshold", 70.0);
  single_body_switch_margin_ =
      declare_parameter<double>("single_body_switch_margin", 10.0);
  fusion_skeleton_smoothing_ =
      declare_parameter<double>("fusion_skeleton_smoothing", 0.0);
  fusion_minimum_allowed_cameras_ =
      declare_parameter<int>("fusion_minimum_allowed_cameras", 2);
  fusion_minimum_allowed_keypoints_ =
      declare_parameter<int>("fusion_minimum_allowed_keypoints", 7);
  camera_fps_ = declare_parameter<int>("camera_fps", 60);
  sdk_gpu_id_ = declare_parameter<int>("sdk_gpu_id", -1);
  fusion_publish_rate_hz_ =
      declare_parameter<double>("fusion_publish_rate_hz", 60.0);
  single_body_switch_frames_ =
      declare_parameter<int>("single_body_switch_frames", 5);

  single_body_enabled_ = declare_parameter<bool>("single_body_enabled", true);
  publish_images_ = declare_parameter<bool>("publish_images", false);
  publish_overlay_images_ =
      declare_parameter<bool>("publish_overlay_images", true);
  publish_per_camera_skeletons_ =
      declare_parameter<bool>("publish_per_camera_skeletons", false);
  overlay_min_confidence_ = declare_parameter<double>("overlay_min_confidence",
                                                      confidence_threshold_);
  overlay_max_skeleton_age_sec_ =
      declare_parameter<double>("overlay_max_skeleton_age_sec", 0.5);
  sender_tracking_enabled_ =
      declare_parameter<bool>("sender_tracking_enabled", false);
  fusion_tracking_enabled_ =
      declare_parameter<bool>("fusion_tracking_enabled", true);
  body_fitting_enabled_ =
      declare_parameter<bool>("body_fitting_enabled", false);
  set_as_static_ = declare_parameter<bool>("set_as_static", true);
  allow_reduced_precision_inference_ =
      declare_parameter<bool>("allow_reduced_precision_inference", false);
  sdk_verbose_ = declare_parameter<int>("sdk_verbose", 1);

  if (fusion_config_path_.empty()) {
    throw std::runtime_error("fusion_config_path is required");
  }
  if (fusion_publish_rate_hz_ <= 0.0) {
    throw std::runtime_error("fusion_publish_rate_hz must be positive");
  }
  if (confidence_threshold_ < 0.0 || confidence_threshold_ > 100.0) {
    throw std::runtime_error(
        "confidence_threshold must be in the range [0, 100]");
  }
  if (overlay_min_confidence_ < 0.0 || overlay_min_confidence_ > 100.0) {
    throw std::runtime_error(
        "overlay_min_confidence must be in the range [0, 100]");
  }
  if (overlay_max_skeleton_age_sec_ < 0.0) {
    throw std::runtime_error(
        "overlay_max_skeleton_age_sec must be non-negative");
  }
  if (camera_fps_ <= 0) {
    throw std::runtime_error("camera_fps must be positive");
  }
  if (single_body_switch_margin_ < 0.0) {
    throw std::runtime_error("single_body_switch_margin must be non-negative");
  }
  if (single_body_switch_frames_ <= 0) {
    throw std::runtime_error("single_body_switch_frames must be positive");
  }
  if (fusion_minimum_allowed_cameras_ <= 0 ||
      fusion_minimum_allowed_cameras_ >
          static_cast<int>(camera_specs_.size())) {
    throw std::runtime_error(
        "fusion_minimum_allowed_cameras must be in the range [1, " +
        std::to_string(camera_specs_.size()) + "]");
  }
}

void ZedBodyFusionNode::normalizeMode() {
  role_ = lower(role_);
  input_mode_ = lower(input_mode_);

  if (role_ == "remote" || role_ == "stream_client") {
    role_ = "local";
    input_mode_ = "stream";
  }

  if (input_mode_ != "live" && input_mode_ != "stream") {
    throw std::runtime_error("input_mode must be 'live' or 'stream'");
  }

  if (input_mode_ == "stream" && stream_address_.empty()) {
    throw std::runtime_error(
        "stream_address is required when input_mode:=stream");
  }
}

void ZedBodyFusionNode::validateFusionConfigurations() const {
  if (fusion_configs_.size() != camera_specs_.size()) {
    throw std::runtime_error(
        "Fusion configuration camera count " +
        std::to_string(fusion_configs_.size()) +
        " does not match configured camera count " +
        std::to_string(camera_specs_.size()));
  }

  std::unordered_set<unsigned int> calibration_serials;
  for (const auto &config : fusion_configs_) {
    const auto serial = static_cast<int64_t>(config.serial_number);
    if (serial <= 0 ||
        serial >
            static_cast<int64_t>(std::numeric_limits<unsigned int>::max())) {
      throw std::runtime_error(
          "Fusion configuration contains an invalid camera serial: " +
          std::to_string(serial));
    }

    const auto unsigned_serial = static_cast<unsigned int>(serial);
    if (!calibration_serials.insert(unsigned_serial).second) {
      throw std::runtime_error(
          "Fusion configuration contains duplicate camera serial: " +
          std::to_string(serial));
    }

    (void)cameraSpecForConfig(config);
  }
}

const ZedBodyFusionNode::CameraSpec &
ZedBodyFusionNode::cameraSpecForConfig(
    const sl::FusionConfiguration &config) const {
  const auto serial = static_cast<int64_t>(config.serial_number);
  const auto spec =
      std::find_if(camera_specs_.begin(), camera_specs_.end(),
                   [serial](const CameraSpec &candidate) {
                     return static_cast<int64_t>(candidate.serial_number) ==
                            serial;
                   });
  if (spec == camera_specs_.end()) {
    throw std::runtime_error("Calibration camera serial " +
                             std::to_string(serial) +
                             " is not present in camera_serials");
  }
  return *spec;
}

int ZedBodyFusionNode::streamPortForConfig(
    const sl::FusionConfiguration &config) const {
  return cameraSpecForConfig(config).stream_port;
}

std::string ZedBodyFusionNode::cameraNameForConfig(
    const sl::FusionConfiguration &config) const {
  return cameraSpecForConfig(config).name;
}

std::string
ZedBodyFusionNode::imageTopicForCamera(const std::string &camera_name) const {
  return "/" + camera_name + "/zed_node/rgb/color/rect/image";
}

std::string ZedBodyFusionNode::cameraInfoTopicForCamera(
    const std::string &camera_name) const {
  return "/" + camera_name + "/zed_node/rgb/color/rect/camera_info";
}

std::string ZedBodyFusionNode::overlayImageTopicForCamera(
    const std::string &camera_name) const {
  return "/" + camera_name + "/zed_node/rgb/color/rect/skeleton_overlay";
}

std::string
ZedBodyFusionNode::bodiesTopicForCamera(const std::string &camera_name) const {
  return "/" + camera_name + "/zed_node/body_trk/skeletons";
}

std::string
ZedBodyFusionNode::imageFrameForCamera(const std::string &camera_name) const {
  return camera_name + "_left_camera_optical_frame";
}

geometry_msgs::msg::TransformStamped
ZedBodyFusionNode::staticCameraTransformForConfig(
    const sl::FusionConfiguration &config) {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = now();
  transform.header.frame_id = publish_frame_id_;
  transform.child_frame_id = imageFrameForCamera(cameraNameForConfig(config));

  const auto translation = config.pose.getTranslation();
  const auto orientation = config.pose.getOrientation();
  transform.transform.translation.x = static_cast<double>(translation.x);
  transform.transform.translation.y = static_cast<double>(translation.y);
  transform.transform.translation.z = static_cast<double>(translation.z);
  transform.transform.rotation.x = static_cast<double>(orientation.x);
  transform.transform.rotation.y = static_cast<double>(orientation.y);
  transform.transform.rotation.z = static_cast<double>(orientation.z);
  transform.transform.rotation.w = static_cast<double>(orientation.w);

  return transform;
}

void ZedBodyFusionNode::publishStaticCameraTransforms() {
  std::vector<geometry_msgs::msg::TransformStamped> transforms;
  transforms.reserve(fusion_configs_.size());

  for (const auto &config : fusion_configs_) {
    auto transform = staticCameraTransformForConfig(config);
    RCLCPP_INFO(
        get_logger(), "Publishing static camera TF %s -> %s for serial %u",
        transform.header.frame_id.c_str(), transform.child_frame_id.c_str(),
        static_cast<unsigned int>(config.serial_number));
    transforms.push_back(std::move(transform));
  }

  static_tf_broadcaster_->sendTransform(transforms);
}

sensor_msgs::msg::CameraInfo
ZedBodyFusionNode::makeCameraInfo(sl::Camera &camera,
                                  const std::string &frame_id) const {
  const auto zed_info = camera.getCameraInformation();
  const auto &calibration =
      zed_info.camera_configuration.calibration_parameters;
  const auto &left = calibration.left_cam;

  sensor_msgs::msg::CameraInfo msg;
  msg.header.frame_id = frame_id;
  msg.width = left.image_size.width != 0
                  ? left.image_size.width
                  : zed_info.camera_configuration.resolution.width;
  msg.height = left.image_size.height != 0
                   ? left.image_size.height
                   : zed_info.camera_configuration.resolution.height;
  msg.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
  msg.d.assign(5, 0.0);
  msg.k = {static_cast<double>(left.fx),
           0.0,
           static_cast<double>(left.cx),
           0.0,
           static_cast<double>(left.fy),
           static_cast<double>(left.cy),
           0.0,
           0.0,
           1.0};
  msg.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  msg.p = {static_cast<double>(left.fx),
           0.0,
           static_cast<double>(left.cx),
           0.0,
           0.0,
           static_cast<double>(left.fy),
           static_cast<double>(left.cy),
           0.0,
           0.0,
           0.0,
           1.0,
           0.0};

  return msg;
}

void ZedBodyFusionNode::configureImagePublishing(CameraWorker &worker) {
  if (!publish_images_) {
    return;
  }

  worker.camera_name = cameraNameForConfig(worker.config);
  worker.image_frame_id = imageFrameForCamera(worker.camera_name);
  worker.camera_info = makeCameraInfo(worker.camera, worker.image_frame_id);
  worker.image_pub = create_publisher<sensor_msgs::msg::Image>(
      imageTopicForCamera(worker.camera_name), rclcpp::SensorDataQoS());
  worker.camera_info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(
      cameraInfoTopicForCamera(worker.camera_name), rclcpp::SensorDataQoS());

  RCLCPP_INFO(get_logger(), "Publishing stream images for serial %u on %s",
              worker.serial_number, worker.image_pub->get_topic_name());
  RCLCPP_INFO(get_logger(), "Publishing stream camera info for serial %u on %s",
              worker.serial_number, worker.camera_info_pub->get_topic_name());
}

void ZedBodyFusionNode::configureOverlayPublishing(CameraWorker &worker) {
  if (!publish_overlay_images_) {
    return;
  }

  worker.camera_name = cameraNameForConfig(worker.config);
  worker.image_frame_id = imageFrameForCamera(worker.camera_name);
  worker.overlay_image_pub = create_publisher<sensor_msgs::msg::Image>(
      overlayImageTopicForCamera(worker.camera_name), rclcpp::SensorDataQoS());

  RCLCPP_INFO(get_logger(),
              "Publishing skeleton overlay images for serial %u on %s",
              worker.serial_number, worker.overlay_image_pub->get_topic_name());
}

void ZedBodyFusionNode::configurePerCameraBodyPublishing(
    CameraWorker &worker) {
  if (!publish_per_camera_skeletons_) {
    return;
  }

  worker.camera_name = cameraNameForConfig(worker.config);
  worker.image_frame_id = imageFrameForCamera(worker.camera_name);
  worker.bodies_pub = create_publisher<zed_msgs::msg::ObjectsStamped>(
      bodiesTopicForCamera(worker.camera_name), rclcpp::SensorDataQoS());

  RCLCPP_INFO(get_logger(),
              "Publishing per-camera skeletons for serial %u on %s",
              worker.serial_number, worker.bodies_pub->get_topic_name());
}

bool ZedBodyFusionNode::hasImageSubscribers(const CameraWorker &worker) const {
  if (!publish_images_ || !worker.image_pub || !worker.camera_info_pub) {
    return false;
  }

  return worker.image_pub->get_subscription_count() > 0 ||
         worker.camera_info_pub->get_subscription_count() > 0;
}

bool ZedBodyFusionNode::hasOverlayImageSubscribers(
    const CameraWorker &worker) const {
  return publish_overlay_images_ && worker.overlay_image_pub &&
         worker.overlay_image_pub->get_subscription_count() > 0;
}

bool ZedBodyFusionNode::shouldRetrieveImage(const CameraWorker &worker) const {
  return hasImageSubscribers(worker) || hasOverlayImageSubscribers(worker);
}

builtin_interfaces::msg::Time
ZedBodyFusionNode::timeFromNanoseconds(uint64_t timestamp_ns) const {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(timestamp_ns / 1000000000ULL);
  stamp.nanosec = static_cast<uint32_t>(timestamp_ns % 1000000000ULL);
  return stamp;
}

builtin_interfaces::msg::Time
ZedBodyFusionNode::imageTimestamp(sl::Camera &camera) {
  const uint64_t timestamp_ns =
      camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
  if (timestamp_ns == 0 ||
      timestamp_ns >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return timeFromNanoseconds(static_cast<uint64_t>(now().nanoseconds()));
  }

  return timeFromNanoseconds(timestamp_ns);
}

void ZedBodyFusionNode::publishImage(CameraWorker &worker,
                                     rclcpp::Clock &steady_clock) {
  if (!shouldRetrieveImage(worker)) {
    return;
  }

  const auto err = worker.camera.retrieveImage(worker.image, sl::VIEW::LEFT_BGR,
                                               sl::MEM::CPU);
  if (err != sl::ERROR_CODE::SUCCESS) {
    RCLCPP_WARN_THROTTLE(get_logger(), steady_clock, 2000,
                         "Camera serial %u image retrieval failed: %s",
                         worker.serial_number, zedErrorToString(err).c_str());
    return;
  }

  const auto *src = reinterpret_cast<const uint8_t *>(
      worker.image.getPtr<sl::uchar1>(sl::MEM::CPU));
  if (src == nullptr) {
    RCLCPP_WARN_THROTTLE(get_logger(), steady_clock, 2000,
                         "Camera serial %u returned an empty image buffer",
                         worker.serial_number);
    return;
  }

  sensor_msgs::msg::Image image_msg;
  image_msg.header.stamp = imageTimestamp(worker.camera);
  image_msg.header.frame_id = worker.image_frame_id;
  image_msg.height = worker.image.getHeight();
  image_msg.width = worker.image.getWidth();
  image_msg.encoding = sensor_msgs::image_encodings::BGR8;
  image_msg.is_bigendian = false;
  image_msg.step = image_msg.width * 3;

  const size_t src_step = worker.image.getStepBytes(sl::MEM::CPU);
  if (src_step < image_msg.step) {
    RCLCPP_WARN_THROTTLE(get_logger(), steady_clock, 2000,
                         "Camera serial %u image step is smaller than expected",
                         worker.serial_number);
    return;
  }

  image_msg.data.resize(static_cast<size_t>(image_msg.height) * image_msg.step);
  for (uint32_t row = 0; row < image_msg.height; ++row) {
    std::memcpy(image_msg.data.data() +
                    static_cast<size_t>(row) * image_msg.step,
                src + static_cast<size_t>(row) * src_step, image_msg.step);
  }

  auto camera_info_msg = worker.camera_info;
  camera_info_msg.header.stamp = image_msg.header.stamp;
  camera_info_msg.width = image_msg.width;
  camera_info_msg.height = image_msg.height;

  if (hasImageSubscribers(worker)) {
    worker.image_pub->publish(image_msg);
    worker.camera_info_pub->publish(camera_info_msg);
  }

  if (hasOverlayImageSubscribers(worker)) {
    publishOverlayImage(worker, image_msg);
  }
}

double ZedBodyFusionNode::overlayTimestampDeltaSec(
    const sl::Bodies &bodies, const sensor_msgs::msg::Image &image_msg) const {
  const auto bodies_timestamp_ns = bodies.timestamp.getNanoseconds();
  const rclcpp::Time image_stamp(image_msg.header.stamp);
  if (bodies_timestamp_ns == 0 ||
      bodies_timestamp_ns >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      image_stamp.nanoseconds() == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const auto delta_ns =
      image_stamp.nanoseconds() - static_cast<int64_t>(bodies_timestamp_ns);
  return std::abs(static_cast<double>(delta_ns) / 1e9);
}

bool ZedBodyFusionNode::validOverlayPoint(const sl::float2 &keypoint,
                                          const cv::Size &size) const {
  const float x = keypoint[0];
  const float y = keypoint[1];
  return std::isfinite(x) && std::isfinite(y) && x > 0.0f && y > 0.0f &&
         x < static_cast<float>(size.width) &&
         y < static_cast<float>(size.height);
}

cv::Point ZedBodyFusionNode::overlayPoint(const sl::float2 &keypoint) const {
  return cv::Point(static_cast<int>(std::lround(keypoint[0])),
                   static_cast<int>(std::lround(keypoint[1])));
}

void ZedBodyFusionNode::drawOverlayBody(cv::Mat &image,
                                        const sl::BodyData &body,
                                        int8_t body_format) const {
  if (body.confidence < overlay_min_confidence_) {
    return;
  }

  const auto color = colorForId(body.id);
  const auto &keypoints = body.keypoint_2d;
  const auto &bones = bonesForFormat(body_format);
  const auto image_size = image.size();

  for (const auto &bone : bones) {
    if (bone.first < 0 || bone.second < 0 ||
        static_cast<size_t>(bone.first) >= keypoints.size() ||
        static_cast<size_t>(bone.second) >= keypoints.size()) {
      continue;
    }
    const auto &first = keypoints[static_cast<size_t>(bone.first)];
    const auto &second = keypoints[static_cast<size_t>(bone.second)];
    if (!validOverlayPoint(first, image_size) ||
        !validOverlayPoint(second, image_size)) {
      continue;
    }
    cv::line(image, overlayPoint(first), overlayPoint(second), color, 2,
             cv::LINE_AA);
  }

  for (const auto &keypoint : keypoints) {
    if (!validOverlayPoint(keypoint, image_size)) {
      continue;
    }
    cv::circle(image, overlayPoint(keypoint), 4, color, -1, cv::LINE_AA);
  }
}

void ZedBodyFusionNode::publishOverlayImage(CameraWorker &worker,
                                            sensor_msgs::msg::Image image_msg) {
  const auto bodies_err = worker.camera.retrieveBodies(worker.overlay_bodies);
  if (bodies_err != sl::ERROR_CODE::SUCCESS) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Camera serial %u body retrieval for overlay failed: %s",
        worker.serial_number, zedErrorToString(bodies_err).c_str());
  } else if (worker.overlay_bodies.is_new) {
    const double timestamp_delta =
        overlayTimestampDeltaSec(worker.overlay_bodies, image_msg);
    const bool timestamp_ok =
        overlay_max_skeleton_age_sec_ == 0.0 ||
        (std::isfinite(timestamp_delta) &&
         timestamp_delta <= overlay_max_skeleton_age_sec_);
    if (!timestamp_ok) {
      if (std::isfinite(timestamp_delta)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Skipping skeleton overlay for camera serial %u: "
                             "image/body timestamp delta %.3fs "
                             "exceeds %.3fs",
                             worker.serial_number, timestamp_delta,
                             overlay_max_skeleton_age_sec_);
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Skipping skeleton overlay for camera serial %u: "
                             "invalid image/body timestamp",
                             worker.serial_number);
      }
    } else {
      const auto body_format =
          static_cast<int8_t>(worker.overlay_bodies.body_format);
      cv::Mat image(static_cast<int>(image_msg.height),
                    static_cast<int>(image_msg.width), CV_8UC3,
                    image_msg.data.data(), image_msg.step);
      for (const auto &body : worker.overlay_bodies.body_list) {
        drawOverlayBody(image, body, body_format);
      }
    }
  }

  worker.overlay_image_pub->publish(std::move(image_msg));
}

void ZedBodyFusionNode::configureRuntimeParameters() {
  sender_runtime_params_.detection_confidence_threshold =
      static_cast<float>(confidence_threshold_);
  sender_runtime_params_.skeleton_smoothing = 0.0f;

  fusion_runtime_params_.skeleton_smoothing =
      static_cast<float>(fusion_skeleton_smoothing_);
  fusion_runtime_params_.skeleton_minimum_allowed_camera =
      fusion_minimum_allowed_cameras_;
  fusion_runtime_params_.skeleton_minimum_allowed_keypoints =
      fusion_minimum_allowed_keypoints_;
}

void ZedBodyFusionNode::startCameraPublishers() {
  workers_.reserve(fusion_configs_.size());

  for (size_t idx = 0; idx < fusion_configs_.size(); ++idx) {
    auto &config = fusion_configs_[idx];
    config.communication_parameters.setForSharedMemory();

    auto worker = std::make_unique<CameraWorker>();
    worker->config = config;
    worker->serial_number = static_cast<unsigned int>(config.serial_number);

    sl::InitParameters init_params;
    if (input_mode_ == "stream") {
      const auto stream_port = streamPortForConfig(config);
      init_params.input.setFromStream(stream_address_.c_str(),
                                      static_cast<unsigned short>(stream_port));
      RCLCPP_INFO(
          get_logger(),
          "Opening ZED SDK stream %s:%d for calibrated camera serial %u",
          stream_address_.c_str(), stream_port, worker->serial_number);
    } else {
      init_params.input = config.input_type;
      RCLCPP_INFO(get_logger(),
                  "Opening local ZED camera serial %u for Fusion publishing",
                  worker->serial_number);
    }
    init_params.camera_resolution = camera_resolution_;
    init_params.camera_fps = camera_fps_;
    init_params.depth_mode = depth_mode_;
    init_params.coordinate_system = kRosCoordinateSystem;
    init_params.coordinate_units = kRosUnits;
    init_params.sdk_gpu_id = sdk_gpu_id_;
    init_params.sdk_verbose = sdk_verbose_;

    auto err = worker->camera.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
      throw std::runtime_error("Failed to open ZED camera serial " +
                               std::to_string(worker->serial_number) + ": " +
                               zedErrorToString(err));
    }

    const auto opened_serial =
        worker->camera.getCameraInformation().serial_number;
    if (opened_serial != worker->serial_number) {
      throw std::runtime_error(
          "ZED source on port " +
          std::to_string(streamPortForConfig(config)) + " reports serial " +
          std::to_string(opened_serial) + ", expected " +
          std::to_string(worker->serial_number));
    }

    configurePerCameraBodyPublishing(*worker);
    configureImagePublishing(*worker);
    configureOverlayPublishing(*worker);

    sl::PositionalTrackingParameters tracking_params;
    tracking_params.set_as_static = set_as_static_;
    tracking_params.enable_area_memory = false;
    tracking_params.enable_pose_smoothing = false;
    tracking_params.set_gravity_as_origin = true;

    err = worker->camera.enablePositionalTracking(tracking_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
      throw std::runtime_error(
          "Failed to enable positional tracking for camera serial " +
          std::to_string(worker->serial_number) + ": " + zedErrorToString(err));
    }

    sl::BodyTrackingParameters body_params;
    body_params.detection_model = body_model_;
    body_params.body_format = body_format_;
    body_params.enable_tracking = sender_tracking_enabled_;
    body_params.enable_body_fitting = false;
    body_params.enable_segmentation = false;
    body_params.allow_reduced_precision_inference =
        allow_reduced_precision_inference_;

    err = worker->camera.enableBodyTracking(body_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
      throw std::runtime_error(
          "Failed to enable body tracking for camera serial " +
          std::to_string(worker->serial_number) + ": " + zedErrorToString(err));
    }

    err =
        worker->camera.setBodyTrackingRuntimeParameters(sender_runtime_params_);
    if (err != sl::ERROR_CODE::SUCCESS) {
      throw std::runtime_error(
          "Failed to set body tracking runtime parameters for camera serial " +
          std::to_string(worker->serial_number) + ": " + zedErrorToString(err));
    }

    err = worker->camera.startPublishing(config.communication_parameters);
    if (err != sl::ERROR_CODE::SUCCESS) {
      throw std::runtime_error(
          "Failed to start Fusion publishing for camera serial " +
          std::to_string(worker->serial_number) + ": " + zedErrorToString(err));
    }

    worker->running = true;
    worker->thread = std::thread(
        [this, raw_worker = worker.get()]() { runCameraWorker(*raw_worker); });

    workers_.push_back(std::move(worker));
  }
}

void ZedBodyFusionNode::startFusion() {
  sl::InitFusionParameters init_params;
  init_params.coordinate_system = kRosCoordinateSystem;
  init_params.coordinate_units = kRosUnits;
  init_params.sdk_gpu_id = sdk_gpu_id_;
  init_params.verbose = sdk_verbose_ != 0;
  init_params.output_performance_metrics = true;
  init_params.timeout_period_number = 20;

  auto fusion_err = fusion_.init(init_params);
  if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
    throw std::runtime_error("Failed to initialize Fusion: " +
                             fusionErrorToString(fusion_err));
  }

  for (const auto &config : fusion_configs_) {
    sl::CameraIdentifier uuid;
    uuid.sn = static_cast<unsigned int>(config.serial_number);

    fusion_err = fusion_.subscribe(uuid, config.communication_parameters,
                                   config.pose, config.override_gravity);
    if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
      throw std::runtime_error("Failed to subscribe Fusion to camera serial " +
                               std::to_string(config.serial_number) + ": " +
                               fusionErrorToString(fusion_err));
    }

    RCLCPP_INFO(get_logger(), "Fusion subscribed to camera serial %d",
                config.serial_number);
  }

  sl::BodyTrackingFusionParameters body_fusion_params;
  body_fusion_params.enable_tracking = fusion_tracking_enabled_;
  body_fusion_params.enable_body_fitting = body_fitting_enabled_;

  fusion_err = fusion_.enableBodyTracking(body_fusion_params);
  if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
    throw std::runtime_error("Failed to enable Fusion body tracking: " +
                             fusionErrorToString(fusion_err));
  }

  RCLCPP_INFO(get_logger(),
              "ZED body Fusion ready; publishing %s in frame '%s'",
              pub_bodies_->get_topic_name(), publish_frame_id_.c_str());
}

void ZedBodyFusionNode::runCameraWorker(CameraWorker &worker) {
  rclcpp::Clock steady_clock(RCL_STEADY_TIME);

  while (rclcpp::ok() && worker.running.load()) {
    auto err = worker.camera.grab();
    if (err != sl::ERROR_CODE::SUCCESS) {
      RCLCPP_WARN_THROTTLE(get_logger(), steady_clock, 2000,
                           "Camera serial %u grab failed: %s",
                           worker.serial_number, zedErrorToString(err).c_str());
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } else {
      publishImage(worker, steady_clock);
    }
  }
}

void ZedBodyFusionNode::processFusion() {
  auto fusion_err = fusion_.process();
  if (fusion_err == sl::FUSION_ERROR_CODE::NO_NEW_DATA_AVAILABLE) {
    return;
  }
  if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Fusion process failed: %s",
                         fusionErrorToString(fusion_err).c_str());
    return;
  }

  sl::Bodies bodies;
  fusion_err =
      fusion_.retrieveBodies(bodies, fusion_runtime_params_,
                             sl::CameraIdentifier(), fusion_reference_frame_);
  if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Fusion retrieveBodies failed: %s",
                         fusionErrorToString(fusion_err).c_str());
    return;
  }
  if (!bodies.is_new) {
    return;
  }

  pub_bodies_->publish(toRosMessage(bodies));
  if (publish_per_camera_skeletons_) {
    publishPerCameraBodies();
  }
}

bool ZedBodyFusionNode::bodyPassesConfidence(const sl::BodyData &body) const {
  return body.confidence >= static_cast<float>(confidence_threshold_);
}

int ZedBodyFusionNode::validKeypointCount(const sl::BodyData &body) const {
  int count_2d = 0;
  for (const auto &keypoint : body.keypoint_2d) {
    if (std::isfinite(keypoint[0]) && std::isfinite(keypoint[1]) &&
        keypoint[0] >= 0.0f && keypoint[1] >= 0.0f) {
      ++count_2d;
    }
  }

  int count_3d = 0;
  for (const auto &keypoint : body.keypoint) {
    if (std::isfinite(keypoint[0]) && std::isfinite(keypoint[1]) &&
        std::isfinite(keypoint[2])) {
      ++count_3d;
    }
  }

  return std::max(count_2d, count_3d);
}

bool ZedBodyFusionNode::bodyPassesRosFilter(const sl::BodyData &body) const {
  if (!bodyPassesConfidence(body)) {
    return false;
  }
  if (fusion_minimum_allowed_keypoints_ <= 0) {
    return true;
  }
  return validKeypointCount(body) >= fusion_minimum_allowed_keypoints_;
}

int ZedBodyFusionNode::bestConfidenceIndex(
    const std::vector<sl::BodyData> &bodies) const {
  if (bodies.empty()) {
    return -1;
  }

  int best_index = -1;
  float best_confidence = -1.0f;
  for (size_t idx = 0; idx < bodies.size(); ++idx) {
    const auto &body = bodies[idx];
    if (!bodyPassesRosFilter(body)) {
      continue;
    }
    if (body.confidence > best_confidence) {
      best_confidence = body.confidence;
      best_index = static_cast<int>(idx);
    }
  }

  return best_index;
}

int ZedBodyFusionNode::bodyIndexById(const std::vector<sl::BodyData> &bodies,
                                     int body_id) const {
  if (body_id < 0) {
    return -1;
  }

  const auto it = std::find_if(
      bodies.begin(), bodies.end(), [this, body_id](const sl::BodyData &body) {
        return body.id == body_id && bodyPassesRosFilter(body);
      });

  if (it == bodies.end()) {
    return -1;
  }

  return static_cast<int>(std::distance(bodies.begin(), it));
}

int ZedBodyFusionNode::selectedBodyIndex(
    const std::vector<sl::BodyData> &bodies) {
  const int best_index = bestConfidenceIndex(bodies);
  if (best_index < 0) {
    selected_body_id_ = -1;
    candidate_body_id_ = -1;
    candidate_switch_count_ = 0;
    return -1;
  }

  const auto &best_body = bodies[static_cast<size_t>(best_index)];
  if (best_body.id < 0) {
    selected_body_id_ = -1;
    candidate_body_id_ = -1;
    candidate_switch_count_ = 0;
    return best_index;
  }

  const int current_index = bodyIndexById(bodies, selected_body_id_);
  if (current_index < 0) {
    selected_body_id_ = best_body.id;
    candidate_body_id_ = -1;
    candidate_switch_count_ = 0;
    return best_index;
  }

  if (best_body.id == selected_body_id_) {
    candidate_body_id_ = -1;
    candidate_switch_count_ = 0;
    return current_index;
  }

  const auto &current_body = bodies[static_cast<size_t>(current_index)];
  const bool best_is_clearly_better =
      best_body.confidence >=
      current_body.confidence + static_cast<float>(single_body_switch_margin_);
  if (!best_is_clearly_better) {
    candidate_body_id_ = -1;
    candidate_switch_count_ = 0;
    return current_index;
  }

  if (candidate_body_id_ == best_body.id) {
    ++candidate_switch_count_;
  } else {
    candidate_body_id_ = best_body.id;
    candidate_switch_count_ = 1;
  }

  if (candidate_switch_count_ >= single_body_switch_frames_) {
    selected_body_id_ = best_body.id;
    candidate_body_id_ = -1;
    candidate_switch_count_ = 0;
    return best_index;
  }

  return current_index;
}

void ZedBodyFusionNode::copyBodyToRosObject(const sl::BodyData &body,
                                            zed_msgs::msg::Object &object,
                                            bool tracking_available) {
  object.label = "Body_" + std::to_string(body.id);
  object.sublabel = "";
  object.label_id = body.id;
  object.confidence = body.confidence;
  copyVector3(object.position, body.position);
  copyFlat(object.position_covariance, body.position_covariance);
  copyVector3(object.velocity, body.velocity);
  object.tracking_available = tracking_available;
  object.tracking_state = static_cast<int8_t>(body.tracking_state);
  object.action_state = static_cast<int8_t>(body.action_state);

  if (body.bounding_box_2d.size() == 4) {
    copyBox2d(object.bounding_box_2d, body.bounding_box_2d);
  }
  if (body.bounding_box.size() == 8) {
    copyBox3d(object.bounding_box_3d, body.bounding_box);
  }
  copyVector3(object.dimensions_3d, body.dimensions);

  object.body_format = static_cast<uint8_t>(body_format_);

  if (body.head_bounding_box_2d.size() == 4) {
    copyBox2d(object.head_bounding_box_2d, body.head_bounding_box_2d);
  }
  if (body.head_bounding_box.size() == 8) {
    copyBox3d(object.head_bounding_box_3d, body.head_bounding_box);
  }
  copyVector3(object.head_position, body.head_position);

  object.skeleton_available = true;
  copySkeleton2d(object.skeleton_2d, body.keypoint_2d);
  copySkeleton3d(object.skeleton_3d, body.keypoint);
}

zed_msgs::msg::ObjectsStamped
ZedBodyFusionNode::toRosMessage(const sl::Bodies &bodies) {
  return toRosMessage(bodies, publish_frame_id_, single_body_enabled_,
                      fusion_tracking_enabled_);
}

zed_msgs::msg::ObjectsStamped ZedBodyFusionNode::toRosMessage(
    const sl::Bodies &bodies, const std::string &frame_id,
    bool apply_single_body_filter, bool tracking_available) {
  zed_msgs::msg::ObjectsStamped msg;
  msg.header.stamp = now();
  msg.header.frame_id = frame_id;

  if (apply_single_body_filter) {
    const int selected_index = selectedBodyIndex(bodies.body_list);
    if (selected_index >= 0) {
      msg.objects.resize(1);
      copyBodyToRosObject(bodies.body_list[static_cast<size_t>(selected_index)],
                          msg.objects[0], tracking_available);
    }
    return msg;
  }

  for (const auto &body : bodies.body_list) {
    if (!bodyPassesRosFilter(body)) {
      continue;
    }
    auto &object = msg.objects.emplace_back();
    copyBodyToRosObject(body, object, tracking_available);
  }

  return msg;
}

void ZedBodyFusionNode::publishPerCameraBodies() {
  for (const auto &worker : workers_) {
    const bool publish_bodies =
        worker->bodies_pub && worker->bodies_pub->get_subscription_count() > 0;
    if (!publish_bodies) {
      continue;
    }

    sl::CameraIdentifier uuid;
    uuid.sn = worker->serial_number;

    sl::Bodies camera_bodies;
    const auto fusion_err = fusion_.retrieveBodies(
        camera_bodies, fusion_runtime_params_, uuid, fusion_reference_frame_);
    if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Fusion retrieveBodies failed for camera serial %u: %s",
          worker->serial_number, fusionErrorToString(fusion_err).c_str());
      continue;
    }
    if (!camera_bodies.is_new) {
      continue;
    }

    // Even when retrieving one sender's raw bodies, Fusion expresses
    // coordinates in the requested Fusion reference frame.
    auto msg = toRosMessage(camera_bodies, publish_frame_id_, false,
                            sender_tracking_enabled_);
    worker->bodies_pub->publish(std::move(msg));
  }
}

void ZedBodyFusionNode::shutdown() {
  if (shutting_down_.exchange(true)) {
    return;
  }

  if (fusion_timer_) {
    fusion_timer_->cancel();
  }

  for (auto &worker : workers_) {
    worker->running = false;
  }
  for (auto &worker : workers_) {
    if (worker->thread.joinable()) {
      worker->thread.join();
    }
  }

  fusion_.disableBodyTracking();
  fusion_.close();

  for (auto &worker : workers_) {
    worker->camera.stopPublishing();
    worker->camera.disableBodyTracking();
    worker->camera.disablePositionalTracking();
    worker->camera.close();
  }
}

} // namespace zed_launcher

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/distortion_models.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>

namespace
{
constexpr auto kRosCoordinateSystem = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
constexpr auto kRosUnits = sl::UNIT::METER;

std::string upper(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

sl::BODY_TRACKING_MODEL parseBodyModel(const std::string & value)
{
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

sl::BODY_FORMAT parseBodyFormat(const std::string & value)
{
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

sl::DEPTH_MODE parseDepthMode(const std::string & value)
{
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

sl::RESOLUTION parseResolution(const std::string & value)
{
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

sl::FUSION_REFERENCE_FRAME parseFusionReferenceFrame(const std::string & value)
{
  const auto normalized = upper(value);
  if (normalized == "BASELINK") {
    return sl::FUSION_REFERENCE_FRAME::BASELINK;
  }
  if (normalized == "WORLD") {
    return sl::FUSION_REFERENCE_FRAME::WORLD;
  }
  throw std::invalid_argument("Unsupported fusion_reference_frame: " + value);
}

std::string zedErrorToString(sl::ERROR_CODE code)
{
  std::ostringstream out;
  out << sl::toString(code);
  return out.str();
}

std::string fusionErrorToString(sl::FUSION_ERROR_CODE code)
{
  std::ostringstream out;
  out << sl::toString(code);
  return out.str();
}

template<typename RosArray, typename SlVector>
void copyVector3(RosArray & dest, const SlVector & src)
{
  const auto count = std::min<size_t>(dest.size(), 3);
  for (size_t i = 0; i < count; ++i) {
    dest[i] = static_cast<float>(src[i]);
  }
}

template<typename RosArray, typename SlVector>
void copyFlat(RosArray & dest, const SlVector & src)
{
  const auto count = std::min<size_t>(std::size(dest), std::size(src));
  for (size_t i = 0; i < count; ++i) {
    dest[i] = static_cast<float>(src[i]);
  }
}

template<typename RosBox, typename SlBox>
void copyBox2d(RosBox & dest, const SlBox & src)
{
  const auto count = std::min<size_t>(dest.corners.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.corners[i].kp[0] = static_cast<unsigned int>(src[i][0]);
    dest.corners[i].kp[1] = static_cast<unsigned int>(src[i][1]);
  }
}

template<typename RosBox, typename SlBox>
void copyBox3d(RosBox & dest, const SlBox & src)
{
  const auto count = std::min<size_t>(dest.corners.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.corners[i].kp[0] = static_cast<float>(src[i][0]);
    dest.corners[i].kp[1] = static_cast<float>(src[i][1]);
    dest.corners[i].kp[2] = static_cast<float>(src[i][2]);
  }
}

template<typename RosSkeleton, typename SlKeypoints>
void copySkeleton2d(RosSkeleton & dest, const SlKeypoints & src)
{
  const auto count = std::min<size_t>(dest.keypoints.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.keypoints[i].kp[0] = static_cast<float>(src[i][0]);
    dest.keypoints[i].kp[1] = static_cast<float>(src[i][1]);
  }
}

template<typename RosSkeleton, typename SlKeypoints>
void copySkeleton3d(RosSkeleton & dest, const SlKeypoints & src)
{
  const auto count = std::min<size_t>(dest.keypoints.size(), src.size());
  for (size_t i = 0; i < count; ++i) {
    dest.keypoints[i].kp[0] = static_cast<float>(src[i][0]);
    dest.keypoints[i].kp[1] = static_cast<float>(src[i][1]);
    dest.keypoints[i].kp[2] = static_cast<float>(src[i][2]);
  }
}

}  // namespace

class ZedBodyFusionNode final : public rclcpp::Node
{
public:
  explicit ZedBodyFusionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("zed_body_fusion_node", options)
  {
    loadParameters();

    normalizeMode();

    if (role_ != "local") {
      throw std::runtime_error(
        "zed_body_fusion_node currently runs local in-process Fusion only; requested role: " + role_);
    }

    pub_bodies_ = create_publisher<zed_msgs::msg::ObjectsStamped>(
      output_topic_, rclcpp::SensorDataQoS());

    RCLCPP_INFO(get_logger(), "Loading ZED Fusion configuration: %s", fusion_config_path_.c_str());
    fusion_configs_ = sl::readFusionConfigurationFile(
      fusion_config_path_, kRosCoordinateSystem, kRosUnits);
    if (fusion_configs_.size() != 2) {
      throw std::runtime_error(
        "Fusion configuration must contain exactly two cameras, found " +
        std::to_string(fusion_configs_.size()));
    }

    configureRuntimeParameters();

    try {
      startCameraPublishers();
      startFusion();
    } catch (...) {
      shutdown();
      throw;
    }

    const auto timer_period = std::chrono::duration<double>(1.0 / fusion_publish_rate_hz_);
    fusion_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
      [this]() { processFusion(); });
  }

  ~ZedBodyFusionNode() override
  {
    shutdown();
  }

private:
  struct CameraWorker
  {
    sl::FusionConfiguration config;
    sl::Camera camera;
    sl::Mat image;
    sensor_msgs::msg::CameraInfo camera_info;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
    rclcpp::Publisher<zed_msgs::msg::ObjectsStamped>::SharedPtr bodies_pub;
    std::thread thread;
    std::atomic_bool running{false};
    std::string camera_name;
    std::string image_frame_id;
    std::string bodies_frame_id;
    unsigned int serial_number = 0;
  };

  void loadParameters()
  {
    role_ = declare_parameter<std::string>("role", "local");
    input_mode_ = declare_parameter<std::string>("input_mode", "stream");
    fusion_config_path_ = declare_parameter<std::string>("fusion_config_path", "");
    output_topic_ = declare_parameter<std::string>("output_topic", "body_trk/skeletons");
    publish_frame_id_ = declare_parameter<std::string>("publish_frame_id", "zed_fusion_map");
    stream_address_ = declare_parameter<std::string>("stream_address", "");
    left_camera_name_ = declare_parameter<std::string>("left_camera_name", "zed_left");
    right_camera_name_ = declare_parameter<std::string>("right_camera_name", "zed_right");

    body_model_ = parseBodyModel(declare_parameter<std::string>("body_model", "HUMAN_BODY_FAST"));
    body_format_ = parseBodyFormat(declare_parameter<std::string>("body_format", "BODY_38"));
    depth_mode_ = parseDepthMode(declare_parameter<std::string>("depth_mode", "NEURAL_LIGHT"));
    camera_resolution_ = parseResolution(declare_parameter<std::string>("camera_resolution", "HD1080"));
    fusion_reference_frame_ = parseFusionReferenceFrame(
      declare_parameter<std::string>("fusion_reference_frame", "BASELINK"));

    confidence_threshold_ = declare_parameter<double>("confidence_threshold", 70.0);
    single_body_switch_margin_ =
      declare_parameter<double>("single_body_switch_margin", 10.0);
    fusion_skeleton_smoothing_ = declare_parameter<double>("fusion_skeleton_smoothing", 0.0);
    fusion_minimum_allowed_cameras_ =
      declare_parameter<int>("fusion_minimum_allowed_cameras", -1);
    fusion_minimum_allowed_keypoints_ =
      declare_parameter<int>("fusion_minimum_allowed_keypoints", 7);
    camera_fps_ = declare_parameter<int>("camera_fps", 60);
    left_stream_port_ = declare_parameter<int>("left_stream_port", 30000);
    right_stream_port_ = declare_parameter<int>("right_stream_port", 30002);
    left_serial_ = declare_parameter<int>("left_serial", 41235597);
    right_serial_ = declare_parameter<int>("right_serial", 49967328);
    sdk_gpu_id_ = declare_parameter<int>("sdk_gpu_id", -1);
    fusion_publish_rate_hz_ = declare_parameter<double>("fusion_publish_rate_hz", 60.0);
    single_body_switch_frames_ =
      declare_parameter<int>("single_body_switch_frames", 5);

    single_body_enabled_ = declare_parameter<bool>("single_body_enabled", true);
    publish_images_ = declare_parameter<bool>("publish_images", true);
    sender_tracking_enabled_ = declare_parameter<bool>("sender_tracking_enabled", false);
    fusion_tracking_enabled_ = declare_parameter<bool>("fusion_tracking_enabled", true);
    body_fitting_enabled_ = declare_parameter<bool>("body_fitting_enabled", false);
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
      throw std::runtime_error("confidence_threshold must be in the range [0, 100]");
    }
    if (camera_fps_ <= 0) {
      throw std::runtime_error("camera_fps must be positive");
    }
    if (publish_images_ && (left_camera_name_.empty() || right_camera_name_.empty())) {
      throw std::runtime_error(
        "left_camera_name and right_camera_name must be non-empty when publish_images is true");
    }
    if (single_body_switch_margin_ < 0.0) {
      throw std::runtime_error("single_body_switch_margin must be non-negative");
    }
    if (single_body_switch_frames_ <= 0) {
      throw std::runtime_error("single_body_switch_frames must be positive");
    }
    if (left_stream_port_ <= 0 || left_stream_port_ > 65535) {
      throw std::runtime_error("left_stream_port must be in the range [1, 65535]");
    }
    if (right_stream_port_ <= 0 || right_stream_port_ > 65535) {
      throw std::runtime_error("right_stream_port must be in the range [1, 65535]");
    }
    if (left_stream_port_ == right_stream_port_) {
      throw std::runtime_error("left_stream_port and right_stream_port must be different");
    }
    if (left_serial_ > 0 && right_serial_ > 0 && left_serial_ == right_serial_) {
      throw std::runtime_error("left_serial and right_serial must be different");
    }
  }

  void normalizeMode()
  {
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
      throw std::runtime_error("stream_address is required when input_mode:=stream");
    }
  }

  int streamPortForConfig(const sl::FusionConfiguration & config, size_t index) const
  {
    const auto serial = static_cast<int>(config.serial_number);
    if (left_serial_ > 0 && serial == left_serial_) {
      return left_stream_port_;
    }
    if (right_serial_ > 0 && serial == right_serial_) {
      return right_stream_port_;
    }

    const bool strict_serial_mapping = left_serial_ > 0 && right_serial_ > 0;
    if (strict_serial_mapping) {
      throw std::runtime_error(
        "Calibration camera serial " + std::to_string(serial) +
        " does not match configured left_serial " + std::to_string(left_serial_) +
        " or right_serial " + std::to_string(right_serial_));
    }

    if (index == 0) {
      return left_stream_port_;
    }
    if (index == 1) {
      return right_stream_port_;
    }

    throw std::runtime_error(
      "No stream port configured for camera serial " + std::to_string(serial));
  }

  std::string cameraNameForConfig(const sl::FusionConfiguration & config, size_t index) const
  {
    const auto serial = static_cast<int>(config.serial_number);
    if (left_serial_ > 0 && serial == left_serial_) {
      return left_camera_name_;
    }
    if (right_serial_ > 0 && serial == right_serial_) {
      return right_camera_name_;
    }

    if (index == 0) {
      return left_camera_name_;
    }
    if (index == 1) {
      return right_camera_name_;
    }

    return "zed_" + std::to_string(index);
  }

  std::string imageTopicForCamera(const std::string & camera_name) const
  {
    return "/" + camera_name + "/zed_node/rgb/color/rect/image";
  }

  std::string cameraInfoTopicForCamera(const std::string & camera_name) const
  {
    return "/" + camera_name + "/zed_node/rgb/color/rect/camera_info";
  }

  std::string bodiesTopicForCamera(const std::string & camera_name) const
  {
    return "/" + camera_name + "/zed_node/body_trk/skeletons";
  }

  std::string imageFrameForCamera(const std::string & camera_name) const
  {
    return camera_name + "_left_camera_optical_frame";
  }

  sensor_msgs::msg::CameraInfo makeCameraInfo(
    sl::Camera & camera, const std::string & frame_id) const
  {
    const auto zed_info = camera.getCameraInformation();
    const auto & calibration = zed_info.camera_configuration.calibration_parameters;
    const auto & left = calibration.left_cam;

    sensor_msgs::msg::CameraInfo msg;
    msg.header.frame_id = frame_id;
    msg.width = left.image_size.width != 0 ?
      left.image_size.width : zed_info.camera_configuration.resolution.width;
    msg.height = left.image_size.height != 0 ?
      left.image_size.height : zed_info.camera_configuration.resolution.height;
    msg.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
    msg.d.assign(5, 0.0);
    msg.k = {
      static_cast<double>(left.fx), 0.0, static_cast<double>(left.cx),
      0.0, static_cast<double>(left.fy), static_cast<double>(left.cy),
      0.0, 0.0, 1.0};
    msg.r = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0};
    msg.p = {
      static_cast<double>(left.fx), 0.0, static_cast<double>(left.cx), 0.0,
      0.0, static_cast<double>(left.fy), static_cast<double>(left.cy), 0.0,
      0.0, 0.0, 1.0, 0.0};

    return msg;
  }

  void configureImagePublishing(CameraWorker & worker, size_t index)
  {
    if (!publish_images_) {
      return;
    }

    worker.camera_name = cameraNameForConfig(worker.config, index);
    worker.image_frame_id = imageFrameForCamera(worker.camera_name);
    worker.bodies_frame_id = worker.image_frame_id;
    worker.camera_info = makeCameraInfo(worker.camera, worker.image_frame_id);
    worker.image_pub = create_publisher<sensor_msgs::msg::Image>(
      imageTopicForCamera(worker.camera_name), rclcpp::SensorDataQoS());
    worker.camera_info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(
      cameraInfoTopicForCamera(worker.camera_name), rclcpp::SensorDataQoS());

    RCLCPP_INFO(
      get_logger(), "Publishing stream images for serial %u on %s",
      worker.serial_number, worker.image_pub->get_topic_name());
    RCLCPP_INFO(
      get_logger(), "Publishing stream camera info for serial %u on %s",
      worker.serial_number, worker.camera_info_pub->get_topic_name());
  }

  void configurePerCameraBodyPublishing(CameraWorker & worker, size_t index)
  {
    worker.camera_name = cameraNameForConfig(worker.config, index);
    worker.image_frame_id = imageFrameForCamera(worker.camera_name);
    worker.bodies_frame_id = worker.image_frame_id;
    worker.bodies_pub = create_publisher<zed_msgs::msg::ObjectsStamped>(
      bodiesTopicForCamera(worker.camera_name), rclcpp::SensorDataQoS());

    RCLCPP_INFO(
      get_logger(), "Publishing per-camera skeletons for serial %u on %s",
      worker.serial_number, worker.bodies_pub->get_topic_name());
  }

  bool hasImageSubscribers(const CameraWorker & worker) const
  {
    if (!publish_images_ || !worker.image_pub || !worker.camera_info_pub) {
      return false;
    }

    return worker.image_pub->get_subscription_count() > 0 ||
           worker.camera_info_pub->get_subscription_count() > 0;
  }

  builtin_interfaces::msg::Time timeFromNanoseconds(uint64_t timestamp_ns) const
  {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(timestamp_ns / 1000000000ULL);
    stamp.nanosec = static_cast<uint32_t>(timestamp_ns % 1000000000ULL);
    return stamp;
  }

  builtin_interfaces::msg::Time imageTimestamp(sl::Camera & camera)
  {
    const uint64_t timestamp_ns =
      camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
    if (
      timestamp_ns == 0 ||
      timestamp_ns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    {
      return timeFromNanoseconds(static_cast<uint64_t>(now().nanoseconds()));
    }

    return timeFromNanoseconds(timestamp_ns);
  }

  void publishImage(CameraWorker & worker, rclcpp::Clock & steady_clock)
  {
    if (!hasImageSubscribers(worker)) {
      return;
    }

    const auto err = worker.camera.retrieveImage(worker.image, sl::VIEW::LEFT_BGR, sl::MEM::CPU);
    if (err != sl::ERROR_CODE::SUCCESS) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock, 2000,
        "Camera serial %u image retrieval failed: %s",
        worker.serial_number, zedErrorToString(err).c_str());
      return;
    }

    const auto * src = reinterpret_cast<const uint8_t *>(
      worker.image.getPtr<sl::uchar1>(sl::MEM::CPU));
    if (src == nullptr) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock, 2000,
        "Camera serial %u returned an empty image buffer", worker.serial_number);
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
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock, 2000,
        "Camera serial %u image step is smaller than expected", worker.serial_number);
      return;
    }

    image_msg.data.resize(static_cast<size_t>(image_msg.height) * image_msg.step);
    for (uint32_t row = 0; row < image_msg.height; ++row) {
      std::memcpy(
        image_msg.data.data() + static_cast<size_t>(row) * image_msg.step,
        src + static_cast<size_t>(row) * src_step,
        image_msg.step);
    }

    auto camera_info_msg = worker.camera_info;
    camera_info_msg.header.stamp = image_msg.header.stamp;
    camera_info_msg.width = image_msg.width;
    camera_info_msg.height = image_msg.height;

    worker.image_pub->publish(std::move(image_msg));
    worker.camera_info_pub->publish(camera_info_msg);
  }

  void configureRuntimeParameters()
  {
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

  void startCameraPublishers()
  {
    workers_.reserve(fusion_configs_.size());

    for (size_t idx = 0; idx < fusion_configs_.size(); ++idx) {
      auto & config = fusion_configs_[idx];
      config.communication_parameters.setForSharedMemory();

      auto worker = std::make_unique<CameraWorker>();
      worker->config = config;
      worker->serial_number = static_cast<unsigned int>(config.serial_number);

      sl::InitParameters init_params;
      if (input_mode_ == "stream") {
        const auto stream_port = streamPortForConfig(config, idx);
        init_params.input.setFromStream(
          stream_address_.c_str(), static_cast<unsigned short>(stream_port));
        RCLCPP_INFO(
          get_logger(),
          "Opening ZED SDK stream %s:%d for calibrated camera serial %u",
          stream_address_.c_str(), stream_port, worker->serial_number);
      } else {
        init_params.input = config.input_type;
        RCLCPP_INFO(
          get_logger(), "Opening local ZED camera serial %u for Fusion publishing",
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
        throw std::runtime_error(
          "Failed to open ZED camera serial " + std::to_string(worker->serial_number) +
          ": " + zedErrorToString(err));
      }

      configurePerCameraBodyPublishing(*worker, idx);
      configureImagePublishing(*worker, idx);

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
      body_params.allow_reduced_precision_inference = allow_reduced_precision_inference_;

      err = worker->camera.enableBodyTracking(body_params);
      if (err != sl::ERROR_CODE::SUCCESS) {
        throw std::runtime_error(
          "Failed to enable body tracking for camera serial " +
          std::to_string(worker->serial_number) + ": " + zedErrorToString(err));
      }

      err = worker->camera.setBodyTrackingRuntimeParameters(sender_runtime_params_);
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
      worker->thread = std::thread([this, raw_worker = worker.get()]() {
        runCameraWorker(*raw_worker);
      });

      workers_.push_back(std::move(worker));
    }
  }

  void startFusion()
  {
    sl::InitFusionParameters init_params;
    init_params.coordinate_system = kRosCoordinateSystem;
    init_params.coordinate_units = kRosUnits;
    init_params.sdk_gpu_id = sdk_gpu_id_;
    init_params.verbose = sdk_verbose_ != 0;
    init_params.output_performance_metrics = true;
    init_params.timeout_period_number = 20;

    auto fusion_err = fusion_.init(init_params);
    if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
      throw std::runtime_error("Failed to initialize Fusion: " + fusionErrorToString(fusion_err));
    }

    for (const auto & config : fusion_configs_) {
      sl::CameraIdentifier uuid;
      uuid.sn = static_cast<unsigned int>(config.serial_number);

      fusion_err = fusion_.subscribe(
        uuid, config.communication_parameters, config.pose, config.override_gravity);
      if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
        throw std::runtime_error(
          "Failed to subscribe Fusion to camera serial " +
          std::to_string(config.serial_number) + ": " + fusionErrorToString(fusion_err));
      }

      RCLCPP_INFO(get_logger(), "Fusion subscribed to camera serial %d", config.serial_number);
    }

    sl::BodyTrackingFusionParameters body_fusion_params;
    body_fusion_params.enable_tracking = fusion_tracking_enabled_;
    body_fusion_params.enable_body_fitting = body_fitting_enabled_;

    fusion_err = fusion_.enableBodyTracking(body_fusion_params);
    if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
      throw std::runtime_error(
        "Failed to enable Fusion body tracking: " + fusionErrorToString(fusion_err));
    }

    RCLCPP_INFO(
      get_logger(), "ZED body Fusion ready; publishing %s in frame '%s'",
      pub_bodies_->get_topic_name(), publish_frame_id_.c_str());
  }

  void runCameraWorker(CameraWorker & worker)
  {
    rclcpp::Clock steady_clock(RCL_STEADY_TIME);

    while (rclcpp::ok() && worker.running.load()) {
      auto err = worker.camera.grab();
      if (err != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), steady_clock, 2000,
          "Camera serial %u grab failed: %s",
          worker.serial_number, zedErrorToString(err).c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      } else {
        publishImage(worker, steady_clock);
      }
    }
  }

  void processFusion()
  {
    auto fusion_err = fusion_.process();
    if (fusion_err == sl::FUSION_ERROR_CODE::NO_NEW_DATA_AVAILABLE) {
      return;
    }
    if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "Fusion process failed: %s",
        fusionErrorToString(fusion_err).c_str());
      return;
    }

    sl::Bodies bodies;
    fusion_err = fusion_.retrieveBodies(
      bodies, fusion_runtime_params_, sl::CameraIdentifier(), fusion_reference_frame_);
    if (fusion_err != sl::FUSION_ERROR_CODE::SUCCESS) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "Fusion retrieveBodies failed: %s",
        fusionErrorToString(fusion_err).c_str());
      return;
    }
    if (!bodies.is_new) {
      return;
    }

    pub_bodies_->publish(toRosMessage(bodies));
    publishPerCameraBodies();
  }

  bool bodyPassesConfidence(const sl::BodyData & body) const
  {
    return body.confidence >= static_cast<float>(confidence_threshold_);
  }

  int validKeypointCount(const sl::BodyData & body) const
  {
    int count_2d = 0;
    for (const auto & keypoint : body.keypoint_2d) {
      if (std::isfinite(keypoint[0]) && std::isfinite(keypoint[1]) && keypoint[0] >= 0.0f &&
        keypoint[1] >= 0.0f)
      {
        ++count_2d;
      }
    }

    int count_3d = 0;
    for (const auto & keypoint : body.keypoint) {
      if (std::isfinite(keypoint[0]) && std::isfinite(keypoint[1]) &&
        std::isfinite(keypoint[2]))
      {
        ++count_3d;
      }
    }

    return std::max(count_2d, count_3d);
  }

  bool bodyPassesRosFilter(const sl::BodyData & body) const
  {
    if (!bodyPassesConfidence(body)) {
      return false;
    }
    if (fusion_minimum_allowed_keypoints_ <= 0) {
      return true;
    }
    return validKeypointCount(body) >= fusion_minimum_allowed_keypoints_;
  }

  int bestConfidenceIndex(const std::vector<sl::BodyData> & bodies) const
  {
    if (bodies.empty()) {
      return -1;
    }

    int best_index = -1;
    float best_confidence = -1.0f;
    for (size_t idx = 0; idx < bodies.size(); ++idx) {
      const auto & body = bodies[idx];
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

  int bodyIndexById(const std::vector<sl::BodyData> & bodies, int body_id) const
  {
    if (body_id < 0) {
      return -1;
    }

    const auto it = std::find_if(
      bodies.begin(), bodies.end(),
      [this, body_id](const sl::BodyData & body) {
        return body.id == body_id && bodyPassesRosFilter(body);
      });

    if (it == bodies.end()) {
      return -1;
    }

    return static_cast<int>(std::distance(bodies.begin(), it));
  }

  int selectedBodyIndex(const std::vector<sl::BodyData> & bodies)
  {
    const int best_index = bestConfidenceIndex(bodies);
    if (best_index < 0) {
      selected_body_id_ = -1;
      candidate_body_id_ = -1;
      candidate_switch_count_ = 0;
      return -1;
    }

    const auto & best_body = bodies[static_cast<size_t>(best_index)];
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

    const auto & current_body = bodies[static_cast<size_t>(current_index)];
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

  void copyBodyToRosObject(const sl::BodyData & body, zed_msgs::msg::Object & object)
  {
    object.label = "Body_" + std::to_string(body.id);
    object.sublabel = "";
    object.label_id = body.id;
    object.confidence = body.confidence;
    copyVector3(object.position, body.position);
    copyFlat(object.position_covariance, body.position_covariance);
    copyVector3(object.velocity, body.velocity);
    object.tracking_available = fusion_tracking_enabled_;
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

  zed_msgs::msg::ObjectsStamped toRosMessage(const sl::Bodies & bodies)
  {
    return toRosMessage(bodies, publish_frame_id_, single_body_enabled_);
  }

  zed_msgs::msg::ObjectsStamped toRosMessage(
    const sl::Bodies & bodies, const std::string & frame_id, bool apply_single_body_filter)
  {
    zed_msgs::msg::ObjectsStamped msg;
    msg.header.stamp = now();
    msg.header.frame_id = frame_id;

    if (apply_single_body_filter) {
      const int selected_index = selectedBodyIndex(bodies.body_list);
      if (selected_index >= 0) {
        msg.objects.resize(1);
        copyBodyToRosObject(
          bodies.body_list[static_cast<size_t>(selected_index)], msg.objects[0]);
      }
      return msg;
    }

    for (const auto & body : bodies.body_list) {
      if (!bodyPassesRosFilter(body)) {
        continue;
      }
      auto & object = msg.objects.emplace_back();
      copyBodyToRosObject(body, object);
    }

    return msg;
  }

  void publishPerCameraBodies()
  {
    for (const auto & worker : workers_) {
      if (!worker->bodies_pub || worker->bodies_pub->get_subscription_count() == 0) {
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

      worker->bodies_pub->publish(
        toRosMessage(camera_bodies, worker->bodies_frame_id, false));
    }
  }

  void shutdown()
  {
    if (shutting_down_.exchange(true)) {
      return;
    }

    if (fusion_timer_) {
      fusion_timer_->cancel();
    }

    for (auto & worker : workers_) {
      worker->running = false;
    }
    for (auto & worker : workers_) {
      if (worker->thread.joinable()) {
        worker->thread.join();
      }
    }

    fusion_.disableBodyTracking();
    fusion_.close();

    for (auto & worker : workers_) {
      worker->camera.stopPublishing();
      worker->camera.disableBodyTracking();
      worker->camera.disablePositionalTracking();
      worker->camera.close();
    }
  }

  std::string role_;
  std::string input_mode_;
  std::string fusion_config_path_;
  std::string output_topic_;
  std::string publish_frame_id_;
  std::string stream_address_;
  std::string left_camera_name_;
  std::string right_camera_name_;

  sl::BODY_TRACKING_MODEL body_model_ = sl::BODY_TRACKING_MODEL::HUMAN_BODY_FAST;
  sl::BODY_FORMAT body_format_ = sl::BODY_FORMAT::BODY_38;
  sl::DEPTH_MODE depth_mode_ = sl::DEPTH_MODE::NEURAL_LIGHT;
  sl::RESOLUTION camera_resolution_ = sl::RESOLUTION::HD1080;
  sl::FUSION_REFERENCE_FRAME fusion_reference_frame_ = sl::FUSION_REFERENCE_FRAME::BASELINK;
  sl::BodyTrackingRuntimeParameters sender_runtime_params_;
  sl::BodyTrackingFusionRuntimeParameters fusion_runtime_params_;

  double confidence_threshold_ = 70.0;
  double single_body_switch_margin_ = 10.0;
  double fusion_skeleton_smoothing_ = 0.0;
  int fusion_minimum_allowed_cameras_ = -1;
  int fusion_minimum_allowed_keypoints_ = 7;
  int camera_fps_ = 60;
  int left_stream_port_ = 30000;
  int right_stream_port_ = 30002;
  int left_serial_ = 41235597;
  int right_serial_ = 49967328;
  int sdk_gpu_id_ = -1;
  int single_body_switch_frames_ = 5;
  int selected_body_id_ = -1;
  int candidate_body_id_ = -1;
  int candidate_switch_count_ = 0;
  double fusion_publish_rate_hz_ = 60.0;
  bool single_body_enabled_ = true;
  bool publish_images_ = true;
  bool sender_tracking_enabled_ = false;
  bool fusion_tracking_enabled_ = true;
  bool body_fitting_enabled_ = false;
  bool set_as_static_ = true;
  bool allow_reduced_precision_inference_ = false;
  int sdk_verbose_ = 1;

  std::vector<sl::FusionConfiguration> fusion_configs_;
  std::vector<std::unique_ptr<CameraWorker>> workers_;
  sl::Fusion fusion_;
  rclcpp::Publisher<zed_msgs::msg::ObjectsStamped>::SharedPtr pub_bodies_;
  rclcpp::TimerBase::SharedPtr fusion_timer_;
  std::atomic_bool shutting_down_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<ZedBodyFusionNode>();
    rclcpp::spin(node);
    node.reset();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("zed_body_fusion_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}

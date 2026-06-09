#pragma once

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <zed_msgs/msg/objects_stamped.hpp>

namespace zed_launcher {

class ZedBodyFusionNode final : public rclcpp::Node {
public:
  explicit ZedBodyFusionNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  ~ZedBodyFusionNode() override;

private:
  struct CameraWorker {
    sl::FusionConfiguration config;
    sl::Camera camera;
    sl::Mat image;
    sensor_msgs::msg::CameraInfo camera_info;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
    rclcpp::Publisher<zed_msgs::msg::ObjectsStamped>::SharedPtr bodies_pub;
    sl::Bodies overlay_bodies;
    std::thread thread;
    std::atomic_bool running{false};
    std::string camera_name;
    std::string image_frame_id;
    unsigned int serial_number = 0;
  };

  void loadParameters();

  void normalizeMode();

  int streamPortForConfig(const sl::FusionConfiguration &config,
                          size_t index) const;

  std::string cameraNameForConfig(const sl::FusionConfiguration &config,
                                  size_t index) const;

  std::string imageTopicForCamera(const std::string &camera_name) const;

  std::string cameraInfoTopicForCamera(const std::string &camera_name) const;

  std::string overlayImageTopicForCamera(const std::string &camera_name) const;

  std::string bodiesTopicForCamera(const std::string &camera_name) const;

  std::string imageFrameForCamera(const std::string &camera_name) const;

  geometry_msgs::msg::TransformStamped
  staticCameraTransformForConfig(const sl::FusionConfiguration &config,
                                 size_t index);

  void publishStaticCameraTransforms();

  sensor_msgs::msg::CameraInfo
  makeCameraInfo(sl::Camera &camera, const std::string &frame_id) const;

  void configureImagePublishing(CameraWorker &worker, size_t index);

  void configureOverlayPublishing(CameraWorker &worker, size_t index);

  void configurePerCameraBodyPublishing(CameraWorker &worker, size_t index);

  bool hasImageSubscribers(const CameraWorker &worker) const;

  bool hasOverlayImageSubscribers(const CameraWorker &worker) const;

  bool shouldRetrieveImage(const CameraWorker &worker) const;

  builtin_interfaces::msg::Time
  timeFromNanoseconds(uint64_t timestamp_ns) const;

  builtin_interfaces::msg::Time imageTimestamp(sl::Camera &camera);

  void publishImage(CameraWorker &worker, rclcpp::Clock &steady_clock);

  double
  overlayTimestampDeltaSec(const sl::Bodies &bodies,
                           const sensor_msgs::msg::Image &image_msg) const;

  bool validOverlayPoint(const sl::float2 &keypoint,
                         const cv::Size &size) const;

  cv::Point overlayPoint(const sl::float2 &keypoint) const;

  void drawOverlayBody(cv::Mat &image, const sl::BodyData &body,
                       int8_t body_format) const;

  void publishOverlayImage(CameraWorker &worker,
                           sensor_msgs::msg::Image image_msg);

  void configureRuntimeParameters();

  void startCameraPublishers();

  void startFusion();

  void runCameraWorker(CameraWorker &worker);

  void processFusion();

  bool bodyPassesConfidence(const sl::BodyData &body) const;

  int validKeypointCount(const sl::BodyData &body) const;

  bool bodyPassesRosFilter(const sl::BodyData &body) const;

  int bestConfidenceIndex(const std::vector<sl::BodyData> &bodies) const;

  int bodyIndexById(const std::vector<sl::BodyData> &bodies, int body_id) const;

  int selectedBodyIndex(const std::vector<sl::BodyData> &bodies);

  void copyBodyToRosObject(const sl::BodyData &body,
                           zed_msgs::msg::Object &object,
                           bool tracking_available);

  zed_msgs::msg::ObjectsStamped toRosMessage(const sl::Bodies &bodies);

  zed_msgs::msg::ObjectsStamped toRosMessage(const sl::Bodies &bodies,
                                             const std::string &frame_id,
                                             bool apply_single_body_filter,
                                             bool tracking_available);

  void publishPerCameraBodies();

  void shutdown();

  std::string role_;

  std::string input_mode_;

  std::string fusion_config_path_;

  std::string output_topic_;

  std::string publish_frame_id_;

  std::string stream_address_;

  std::string left_camera_name_;

  std::string right_camera_name_;

  sl::BODY_TRACKING_MODEL body_model_ =
      sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;

  sl::BODY_FORMAT body_format_ = sl::BODY_FORMAT::BODY_38;

  sl::DEPTH_MODE depth_mode_ = sl::DEPTH_MODE::NEURAL_LIGHT;

  sl::RESOLUTION camera_resolution_ = sl::RESOLUTION::HD1080;

  sl::FUSION_REFERENCE_FRAME fusion_reference_frame_ =
      sl::FUSION_REFERENCE_FRAME::BASELINK;

  sl::BodyTrackingRuntimeParameters sender_runtime_params_;

  sl::BodyTrackingFusionRuntimeParameters fusion_runtime_params_;

  double confidence_threshold_ = 70.0;

  double overlay_min_confidence_ = 70.0;

  double overlay_max_skeleton_age_sec_ = 0.5;

  double single_body_switch_margin_ = 10.0;

  double fusion_skeleton_smoothing_ = 0.0;

  int fusion_minimum_allowed_cameras_ = 1;

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

  bool publish_images_ = false;

  bool publish_overlay_images_ = true;

  bool publish_per_camera_skeletons_ = false;

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

  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::atomic_bool shutting_down_{false};
};

} // namespace zed_launcher

#pragma once

#include "zed_launcher/depth_frame.hpp"
#include "zed_launcher/single_body_continuity.hpp"

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
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
  struct ImageProcessorCameraStats {
    std::string camera_name;
    uint64_t submitted = 0;
    uint64_t processed = 0;
    uint64_t overwritten = 0;
    uint64_t stale_dropped = 0;
    double last_processing_ms = 0.0;
    double last_job_age_ms = 0.0;
  };

  struct ImageProcessor {
    using DebugPublisher =
        std::function<void(sensor_msgs::msg::Image &&debug_image)>;
    using Submit = std::function<void(
        std::string camera_name, sensor_msgs::msg::Image source,
        sensor_msgs::msg::CameraInfo camera_info,
        std::optional<OwnedDepthFrame> depth,
        std::optional<sensor_msgs::msg::Image> debug_overlay,
        DebugPublisher debug_publisher)>;

    ImageProcessor() : needs_depth(false) {}

    std::function<bool(const std::string &camera_name)> should_process;
    Submit submit;
    std::function<std::vector<ImageProcessorCameraStats>()> stats;
    bool needs_depth;

    [[nodiscard]] explicit operator bool() const noexcept {
      return static_cast<bool>(should_process) && static_cast<bool>(submit);
    }
  };

  explicit ZedBodyFusionNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions(),
      ImageProcessor image_processor = ImageProcessor{},
      const std::string &node_namespace = "");

  ~ZedBodyFusionNode() override;

private:
  struct CameraSpec {
    std::string name;
    unsigned int serial_number = 0;
    int stream_port = 0;
  };

  struct CameraWorker {
    struct GrabCounterSample {
      std::chrono::steady_clock::time_point timestamp;
      uint64_t successful = 0;
      uint64_t total = 0;
      uint64_t corrupted = 0;
    };

    sl::FusionConfiguration config;
    sl::Camera camera;
    sl::Mat image;
    sl::Mat depth;
    sensor_msgs::msg::CameraInfo camera_info;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_image_pub;
    rclcpp::Publisher<zed_msgs::msg::ObjectsStamped>::SharedPtr bodies_pub;
    sl::Bodies overlay_bodies;
    std::thread thread;
    std::atomic_bool running{false};
    std::string camera_name;
    std::string image_frame_id;
    unsigned int serial_number = 0;
    int opened_camera_fps = 0;
    std::size_t opened_image_width = 0;
    std::size_t opened_image_height = 0;
    std::mutex gravity_pose_mutex;
    std::array<double, 4> gravity_quaternion{};
    bool has_gravity_pose = false;
    std::atomic<uint64_t> grab_success_count{0};
    std::atomic<uint64_t> grab_failure_count{0};
    std::atomic<uint64_t> corrupted_frame_count{0};
    std::atomic<int64_t> last_image_retrieval_ns{0};
    std::atomic<int64_t> last_depth_retrieval_ns{0};
    std::deque<GrabCounterSample> diagnostic_grab_history;
  };

  void loadParameters();

  void normalizeMode();

  void selectConfiguredFusionConfigurations();

  void validateFusionConfigurations() const;

  const CameraSpec &
  cameraSpecForConfig(const sl::FusionConfiguration &config) const;

  int streamPortForConfig(const sl::FusionConfiguration &config) const;

  std::string cameraNameForConfig(const sl::FusionConfiguration &config) const;

  std::string overlayImageTopicForCamera(const std::string &camera_name) const;

  std::string bodiesTopicForCamera(const std::string &camera_name) const;

  std::string imageFrameForCamera(const std::string &camera_name) const;

  geometry_msgs::msg::TransformStamped
  staticCameraTransformForConfig(const sl::FusionConfiguration &config,
                                 const sl::Transform &absolute_pose);

  bool publishStaticCameraTransforms();

  void updateCameraGravityRotation(CameraWorker &worker);

  sensor_msgs::msg::CameraInfo
  makeCameraInfo(sl::Camera &camera, const std::string &frame_id) const;

  void configureImageProcessing(CameraWorker &worker);

  void configureOverlayPublishing(CameraWorker &worker);

  void configurePerCameraBodyPublishing(CameraWorker &worker);

  bool hasOverlayImageSubscribers(const CameraWorker &worker) const;

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

  void drawSkeletonOverlay(CameraWorker &worker,
                           sensor_msgs::msg::Image &image_msg);

  void configureRuntimeParameters();

  void startCameraPublishers();

  void startFusion();

  void runCameraWorker(CameraWorker &worker);

  void processFusion();

  void publishDiagnostics();

  void recordFusedMessage(bool nonempty);

  bool bodyPassesConfidence(const sl::BodyData &body) const;

  int validKeypointCount(const sl::BodyData &body) const;

  bool bodyPassesRosFilter(const sl::BodyData &body) const;

  bool bodyPassesMeasuredFilter(const sl::BodyData &body,
                                bool tracking_available) const;

  int bestConfidenceIndex(const std::vector<sl::BodyData> &bodies) const;

  int bodyIndexById(const std::vector<sl::BodyData> &bodies, int body_id) const;

  int selectedBodyIndex(const std::vector<sl::BodyData> &bodies);

  void copyBodyToRosObject(const sl::Bodies &bodies, const sl::BodyData &body,
                           zed_msgs::msg::Object &object,
                           bool tracking_available);

  zed_msgs::msg::ObjectsStamped toRosMessage(const sl::Bodies &bodies);

  zed_msgs::msg::ObjectsStamped toRosMessage(const sl::Bodies &bodies,
                                             const std::string &frame_id,
                                             bool apply_single_body_filter,
                                             bool tracking_available);

  std::optional<zed_msgs::msg::Object> retrieveCameraBodyFallback();

  void publishPerCameraBodies();

  void shutdown();

  std::string role_;

  std::string input_mode_;

  std::string fusion_config_path_;

  std::string output_topic_;

  std::string publish_frame_id_;

  std::string stream_address_;

  sl::BODY_TRACKING_MODEL body_model_ =
      sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;

  sl::BODY_FORMAT body_format_ = sl::BODY_FORMAT::BODY_18;

  sl::DEPTH_MODE depth_mode_ = sl::DEPTH_MODE::NEURAL;

  sl::RESOLUTION camera_resolution_ = sl::RESOLUTION::HD1080;

  sl::FUSION_REFERENCE_FRAME fusion_reference_frame_ =
      sl::FUSION_REFERENCE_FRAME::BASELINK;

  sl::BodyTrackingRuntimeParameters sender_runtime_params_;

  sl::BodyTrackingFusionRuntimeParameters fusion_runtime_params_;

  double confidence_threshold_ = 40.0;

  double overlay_min_confidence_ = 40.0;

  double overlay_max_skeleton_age_sec_ = 0.5;

  double single_body_switch_margin_ = 10.0;

  double fusion_skeleton_smoothing_ = 0.1;

  int fusion_minimum_allowed_cameras_ = 1;

  int fusion_minimum_allowed_keypoints_ = 7;

  int camera_fps_ = 30;

  int sdk_gpu_id_ = -1;

  int single_body_switch_frames_ = 30;

  int single_body_logical_id_ = 0;

  int selected_body_id_ = -1;

  int candidate_body_id_ = -1;

  int candidate_switch_count_ = 0;

  double fusion_publish_rate_hz_ = 20.0;

  double fusion_diagnostics_rate_hz_ = 1.0;

  double body_prediction_timeout_sec_ = 1.0;

  double single_body_bridge_timeout_sec_ = 0.5;

  double single_body_bridge_max_speed_mps_ = 2.0;

  double camera_body_fallback_consensus_distance_m_ = 0.5;

  double fused_body_max_jump_m_ = 0.5;

  int camera_body_fallback_minimum_cameras_ = 2;

  bool single_body_enabled_ = true;

  bool camera_body_fallback_enabled_ = true;

  bool per_camera_bodies_retrieved_this_cycle_ = false;

  std::unique_ptr<SingleBodyContinuity> single_body_continuity_;

  std::optional<std::array<double, 3>> last_single_body_position_;

  std::optional<SingleBodyContinuity::TimePoint> last_single_body_position_at_;

  bool publish_overlay_images_ = true;

  bool publish_per_camera_skeletons_ = false;

  bool sender_tracking_enabled_ = true;

  bool fusion_tracking_enabled_ = true;

  bool body_fitting_enabled_ = true;

  bool set_as_static_ = true;

  bool allow_reduced_precision_inference_ = false;

  int sdk_verbose_ = 1;

  ImageProcessor image_processor_;

  std::vector<CameraSpec> camera_specs_;

  std::vector<sl::FusionConfiguration> fusion_configs_;

  std::vector<std::unique_ptr<CameraWorker>> workers_;

  sl::Fusion fusion_;

  rclcpp::Publisher<zed_msgs::msg::ObjectsStamped>::SharedPtr pub_bodies_;

  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_pub_;

  rclcpp::TimerBase::SharedPtr fusion_timer_;

  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  bool camera_transforms_published_ = false;

  std::chrono::steady_clock::time_point diagnostics_started_at_;

  std::chrono::steady_clock::time_point diagnostics_last_sample_at_;

  uint64_t diagnostics_last_fused_message_count_ = 0;

  uint64_t diagnostics_last_nonempty_message_count_ = 0;

  std::atomic<uint64_t> fusion_process_success_count_{0};

  std::atomic<uint64_t> fusion_no_data_count_{0};

  std::atomic<uint64_t> fusion_process_failure_count_{0};

  std::atomic<uint64_t> fused_message_count_{0};

  std::atomic<uint64_t> fused_nonempty_message_count_{0};

  std::atomic<int64_t> last_fused_message_steady_ns_{0};

  std::atomic<uint64_t> max_fused_message_gap_ns_{0};

  std::atomic_bool shutting_down_{false};
};

} // namespace zed_launcher

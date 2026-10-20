#pragma once

#include "zed_launcher/apriltag_fusion_node.hpp"
#include "zed_launcher/depth_frame.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace zed_launcher {

/// Runs direct AprilTag detection outside the ZED camera grab threads.
///
/// Each camera owns one in-flight job and one replaceable pending job. New
/// frames replace an older pending frame so processing latency cannot grow
/// without bound.
class AsyncApriltagProcessor final {
public:
  using DebugPublisher =
      std::function<void(sensor_msgs::msg::Image &&debug_image)>;
  using Processor = std::function<void(
      const std::string &camera_name, const sensor_msgs::msg::Image &source,
      const sensor_msgs::msg::CameraInfo &camera_info,
      const DepthFrameProvider &depth_provider,
      sensor_msgs::msg::Image *debug_overlay)>;

  struct Hooks {
    Hooks() : logger(rclcpp::get_logger("async_apriltag_processor")) {}

    std::function<bool(const std::string &camera_name)> should_process;
    Processor process;
    rclcpp::Logger logger;
    double max_job_age_sec = 0.15;
    bool needs_depth = false;
  };

  struct CameraStats {
    std::string camera_name;
    uint64_t submitted = 0;
    uint64_t processed = 0;
    uint64_t overwritten = 0;
    uint64_t stale_dropped = 0;
    double last_processing_ms = 0.0;
    double last_job_age_ms = 0.0;
  };

  AsyncApriltagProcessor(std::shared_ptr<ApriltagFusionNode> apriltag_node,
                         const std::vector<std::string> &camera_names);

  AsyncApriltagProcessor(Hooks hooks,
                         const std::vector<std::string> &camera_names);

  ~AsyncApriltagProcessor();

  AsyncApriltagProcessor(const AsyncApriltagProcessor &) = delete;
  AsyncApriltagProcessor &operator=(const AsyncApriltagProcessor &) = delete;

  [[nodiscard]] bool shouldProcessFrame(const std::string &camera_name);

  [[nodiscard]] bool needsDepth() const;

  void submit(std::string camera_name, sensor_msgs::msg::Image source,
              sensor_msgs::msg::CameraInfo camera_info,
              std::optional<OwnedDepthFrame> depth,
              std::optional<sensor_msgs::msg::Image> debug_overlay,
              DebugPublisher debug_publisher);

  [[nodiscard]] std::vector<CameraStats> stats() const;

  void stop();

private:
  struct FrameJob {
    sensor_msgs::msg::Image source;
    sensor_msgs::msg::CameraInfo camera_info;
    std::optional<OwnedDepthFrame> depth;
    std::optional<sensor_msgs::msg::Image> debug_overlay;
    DebugPublisher debug_publisher;
    std::chrono::steady_clock::time_point enqueue_time;
  };

  struct CameraWorker {
    explicit CameraWorker(std::string camera_name)
        : name(std::move(camera_name)) {}

    std::string name;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::optional<FrameJob> pending;
    std::thread thread;
    bool stopping = false;
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> overwritten{0};
    std::atomic<uint64_t> stale_dropped{0};
    std::atomic<int64_t> last_processing_ns{0};
    std::atomic<int64_t> last_job_age_ns{0};
  };

  void run(CameraWorker &worker);

  std::function<bool(const std::string &camera_name)> should_process_;
  Processor processor_;
  rclcpp::Logger logger_;
  std::vector<std::unique_ptr<CameraWorker>> workers_;
  std::unordered_map<std::string, std::size_t> worker_indices_;
  double max_job_age_sec_ = 0.15;
  bool needs_depth_ = false;
  std::atomic_bool stopping_{false};
};

} // namespace zed_launcher

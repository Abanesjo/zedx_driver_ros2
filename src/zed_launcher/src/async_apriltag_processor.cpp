#include "zed_launcher/async_apriltag_processor.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <rclcpp/rclcpp.hpp>

namespace zed_launcher {
namespace {

AsyncApriltagProcessor::Hooks
makeApriltagHooks(std::shared_ptr<ApriltagFusionNode> apriltag_node) {
  if (!apriltag_node) {
    throw std::invalid_argument("apriltag_node must not be null");
  }

  AsyncApriltagProcessor::Hooks hooks;
  hooks.logger = apriltag_node->get_logger();
  hooks.max_job_age_sec = apriltag_node->maxObservationAgeSec();
  hooks.needs_depth = apriltag_node->usesDepth();
  hooks.should_process = [apriltag_node](const std::string &camera_name) {
    return apriltag_node->shouldProcessCameraFrame(camera_name);
  };
  hooks.process =
      [apriltag_node](const std::string &camera_name,
                      const sensor_msgs::msg::Image &source,
                      const sensor_msgs::msg::CameraInfo &camera_info,
                      const DepthFrameProvider &depth_provider,
                      sensor_msgs::msg::Image *debug_overlay) {
        apriltag_node->processCameraFrame(camera_name, source, camera_info,
                                          depth_provider, debug_overlay, true);
      };
  return hooks;
}

} // namespace

AsyncApriltagProcessor::AsyncApriltagProcessor(
    std::shared_ptr<ApriltagFusionNode> apriltag_node,
    const std::vector<std::string> &camera_names)
    : AsyncApriltagProcessor(makeApriltagHooks(std::move(apriltag_node)),
                             camera_names) {}

AsyncApriltagProcessor::AsyncApriltagProcessor(
    Hooks hooks, const std::vector<std::string> &camera_names)
    : should_process_(std::move(hooks.should_process)),
      processor_(std::move(hooks.process)), logger_(std::move(hooks.logger)),
      max_job_age_sec_(hooks.max_job_age_sec), needs_depth_(hooks.needs_depth) {
  if (!should_process_) {
    throw std::invalid_argument("should_process hook must not be empty");
  }
  if (!processor_) {
    throw std::invalid_argument("process hook must not be empty");
  }
  if (!std::isfinite(max_job_age_sec_) || max_job_age_sec_ <= 0.0) {
    throw std::invalid_argument("max_job_age_sec must be finite and positive");
  }
  if (camera_names.empty()) {
    throw std::invalid_argument("camera_names must not be empty");
  }

  workers_.reserve(camera_names.size());
  for (const auto &camera_name : camera_names) {
    if (camera_name.empty()) {
      throw std::invalid_argument("camera_names entries must be non-empty");
    }
    if (worker_indices_.count(camera_name) != 0U) {
      throw std::invalid_argument("duplicate camera name: " + camera_name);
    }

    const auto index = workers_.size();
    worker_indices_.emplace(camera_name, index);
    workers_.push_back(std::make_unique<CameraWorker>(camera_name));
  }

  try {
    for (auto &worker : workers_) {
      worker->thread = std::thread(
          [this, raw_worker = worker.get()]() { run(*raw_worker); });
    }
  } catch (...) {
    stop();
    throw;
  }
}

AsyncApriltagProcessor::~AsyncApriltagProcessor() { stop(); }

bool AsyncApriltagProcessor::shouldProcessFrame(
    const std::string &camera_name) {
  if (stopping_.load()) {
    return false;
  }
  return should_process_(camera_name);
}

bool AsyncApriltagProcessor::needsDepth() const { return needs_depth_; }

void AsyncApriltagProcessor::submit(
    std::string camera_name, sensor_msgs::msg::Image source,
    sensor_msgs::msg::CameraInfo camera_info,
    std::optional<OwnedDepthFrame> depth,
    std::optional<sensor_msgs::msg::Image> debug_overlay,
    DebugPublisher debug_publisher) {
  if (stopping_.load()) {
    return;
  }

  const auto worker_it = worker_indices_.find(camera_name);
  if (worker_it == worker_indices_.end()) {
    RCLCPP_WARN(logger_,
                "Ignoring queued AprilTag frame for unknown camera '%s'",
                camera_name.c_str());
    return;
  }

  auto &worker = *workers_.at(worker_it->second);
  FrameJob job{std::move(source),          std::move(camera_info),
               std::move(depth),           std::move(debug_overlay),
               std::move(debug_publisher), std::chrono::steady_clock::now()};
  {
    std::lock_guard<std::mutex> lock(worker.mutex);
    if (worker.stopping) {
      return;
    }
    if (worker.pending) {
      worker.overwritten.fetch_add(1);
    }
    worker.pending = std::move(job);
    worker.submitted.fetch_add(1);
  }
  worker.condition.notify_one();
}

std::vector<AsyncApriltagProcessor::CameraStats>
AsyncApriltagProcessor::stats() const {
  std::vector<CameraStats> result;
  result.reserve(workers_.size());
  for (const auto &worker : workers_) {
    result.push_back(CameraStats{
        worker->name, worker->submitted.load(), worker->processed.load(),
        worker->overwritten.load(), worker->stale_dropped.load(),
        static_cast<double>(worker->last_processing_ns.load()) / 1e6,
        static_cast<double>(worker->last_job_age_ns.load()) / 1e6});
  }
  return result;
}

void AsyncApriltagProcessor::stop() {
  if (stopping_.exchange(true)) {
    return;
  }

  for (auto &worker : workers_) {
    {
      std::lock_guard<std::mutex> lock(worker->mutex);
      worker->stopping = true;
      worker->pending.reset();
    }
    worker->condition.notify_one();
  }
  for (auto &worker : workers_) {
    if (worker->thread.joinable()) {
      worker->thread.join();
    }
  }
}

void AsyncApriltagProcessor::run(CameraWorker &worker) {
  while (true) {
    std::optional<FrameJob> job;
    {
      std::unique_lock<std::mutex> lock(worker.mutex);
      worker.condition.wait(
          lock, [&worker]() { return worker.stopping || worker.pending; });
      if (worker.stopping) {
        return;
      }
      job = std::move(worker.pending);
      worker.pending.reset();
    }

    const auto processing_start = std::chrono::steady_clock::now();
    const auto job_age = processing_start - job->enqueue_time;
    worker.last_job_age_ns.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(job_age).count());
    if (std::chrono::duration<double>(job_age).count() > max_job_age_sec_) {
      worker.stale_dropped.fetch_add(1);
      continue;
    }

    DepthFrameProvider depth_provider;
    if (job->depth && *job->depth) {
      depth_provider = [&owned_depth =
                            *job->depth]() -> std::optional<DepthFrameView> {
        return owned_depth.view();
      };
    }

    auto *debug_overlay = job->debug_overlay ? &*job->debug_overlay : nullptr;
    try {
      processor_(worker.name, job->source, job->camera_info, depth_provider,
                 debug_overlay);
      if (debug_overlay && job->debug_publisher) {
        job->debug_publisher(std::move(*debug_overlay));
      }
    } catch (const std::exception &error) {
      RCLCPP_WARN(logger_, "Asynchronous AprilTag processing failed for %s: %s",
                  worker.name.c_str(), error.what());
    } catch (...) {
      RCLCPP_WARN(logger_, "Asynchronous AprilTag processing failed for %s",
                  worker.name.c_str());
    }

    worker.last_processing_ns.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - processing_start)
            .count());
    worker.processed.fetch_add(1);
  }
}

} // namespace zed_launcher

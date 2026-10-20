#include "zed_launcher/async_apriltag_processor.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <sensor_msgs/image_encodings.hpp>

namespace zed_launcher {
namespace {

using namespace std::chrono_literals;

sensor_msgs::msg::Image makeImage(uint8_t value) {
  sensor_msgs::msg::Image image;
  image.header.stamp.sec = value;
  image.header.frame_id = "camera";
  image.height = 1;
  image.width = 1;
  image.encoding = sensor_msgs::image_encodings::MONO8;
  image.step = 1;
  image.data = {value};
  return image;
}

sensor_msgs::msg::CameraInfo makeCameraInfo(uint8_t value) {
  sensor_msgs::msg::CameraInfo camera_info;
  camera_info.header.stamp.sec = value;
  camera_info.header.frame_id = "camera";
  camera_info.height = 1;
  camera_info.width = 1;
  return camera_info;
}

OwnedDepthFrame makeDepth(uint8_t value) {
  OwnedDepthFrame depth;
  depth.width = 1;
  depth.height = 1;
  depth.data = {static_cast<float>(value)};
  return depth;
}

template <typename Predicate>
bool waitUntil(Predicate predicate,
               std::chrono::milliseconds timeout = 1000ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

TEST(AsyncApriltagProcessorTest,
     ReplacesPendingFrameAndPreservesOwnedPayloadAndPublishOrder) {
  std::mutex mutex;
  std::condition_variable condition;
  bool first_started = false;
  bool release_first = false;
  bool payloads_valid = true;
  std::vector<uint8_t> processed;
  std::vector<uint8_t> published;

  AsyncApriltagProcessor::Hooks hooks;
  hooks.needs_depth = true;
  hooks.should_process = [](const std::string &) { return true; };
  hooks.process = [&](const std::string &camera_name,
                      const sensor_msgs::msg::Image &source,
                      const sensor_msgs::msg::CameraInfo &camera_info,
                      const DepthFrameProvider &depth_provider,
                      sensor_msgs::msg::Image *debug_overlay) {
    const uint8_t value = source.data.at(0);
    if (value == 1) {
      std::unique_lock<std::mutex> lock(mutex);
      first_started = true;
      condition.notify_all();
      condition.wait(lock, [&]() { return release_first; });
    }

    const auto depth = depth_provider ? depth_provider() : std::nullopt;
    const bool valid =
        camera_name == "camera_a" &&
        camera_info.header.stamp.sec == source.header.stamp.sec && depth &&
        *depth && depth->data[0] == static_cast<float>(value) &&
        debug_overlay != nullptr;
    if (debug_overlay) {
      debug_overlay->data[0] = static_cast<uint8_t>(value + 10);
    }

    std::lock_guard<std::mutex> lock(mutex);
    payloads_valid = payloads_valid && valid;
    processed.push_back(value);
    condition.notify_all();
  };

  AsyncApriltagProcessor processor(std::move(hooks), {"camera_a"});
  const auto submit = [&](uint8_t value) {
    processor.submit("camera_a", makeImage(value), makeCameraInfo(value),
                     makeDepth(value), makeImage(value),
                     [&](sensor_msgs::msg::Image &&image) {
                       std::lock_guard<std::mutex> lock(mutex);
                       published.push_back(image.data.at(0));
                       condition.notify_all();
                     });
  };

  submit(1);
  {
    std::unique_lock<std::mutex> lock(mutex);
    const bool started =
        condition.wait_for(lock, 1s, [&]() { return first_started; });
    EXPECT_TRUE(started);
    if (!started) {
      processor.stop();
      return;
    }
  }

  submit(2);
  submit(3);
  {
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_TRUE(published.empty());
    release_first = true;
  }
  condition.notify_all();

  const bool completed = waitUntil([&]() {
    std::lock_guard<std::mutex> lock(mutex);
    return processed.size() == 2 && published.size() == 2;
  });
  processor.stop();

  ASSERT_TRUE(completed);
  EXPECT_TRUE(payloads_valid);
  EXPECT_EQ(processed, (std::vector<uint8_t>{1, 3}));
  EXPECT_EQ(published, (std::vector<uint8_t>{11, 13}));

  const auto stats = processor.stats();
  ASSERT_EQ(stats.size(), 1U);
  EXPECT_EQ(stats[0].submitted, 3U);
  EXPECT_EQ(stats[0].processed, 2U);
  EXPECT_EQ(stats[0].overwritten, 1U);
  EXPECT_EQ(stats[0].stale_dropped, 0U);
}

TEST(AsyncApriltagProcessorTest,
     CameraWorkersRunIndependentlyAndShutdownIsIdempotent) {
  std::mutex mutex;
  std::condition_variable condition;
  bool camera_a_started = false;
  bool camera_a_release = false;
  bool camera_b_processed = false;

  AsyncApriltagProcessor::Hooks hooks;
  hooks.should_process = [](const std::string &camera_name) {
    return camera_name != "blocked";
  };
  hooks.process = [&](const std::string &camera_name,
                      const sensor_msgs::msg::Image &,
                      const sensor_msgs::msg::CameraInfo &,
                      const DepthFrameProvider &, sensor_msgs::msg::Image *) {
    std::unique_lock<std::mutex> lock(mutex);
    if (camera_name == "camera_a") {
      camera_a_started = true;
      condition.notify_all();
      condition.wait(lock, [&]() { return camera_a_release; });
    } else {
      camera_b_processed = true;
      condition.notify_all();
    }
  };

  AsyncApriltagProcessor processor(std::move(hooks), {"camera_a", "camera_b"});
  EXPECT_TRUE(processor.shouldProcessFrame("camera_a"));
  EXPECT_FALSE(processor.shouldProcessFrame("blocked"));

  processor.submit("camera_a", makeImage(1), makeCameraInfo(1), std::nullopt,
                   std::nullopt, {});
  {
    std::unique_lock<std::mutex> lock(mutex);
    const bool started =
        condition.wait_for(lock, 1s, [&]() { return camera_a_started; });
    EXPECT_TRUE(started);
    if (!started) {
      processor.stop();
      return;
    }
  }

  processor.submit("camera_b", makeImage(2), makeCameraInfo(2), std::nullopt,
                   std::nullopt, {});
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(
        condition.wait_for(lock, 1s, [&]() { return camera_b_processed; }));
    camera_a_release = true;
  }
  condition.notify_all();

  processor.stop();
  processor.stop();
  EXPECT_FALSE(processor.shouldProcessFrame("camera_a"));
}

TEST(AsyncApriltagProcessorTest, DropsPendingFrameThatBecameStale) {
  std::mutex mutex;
  std::condition_variable condition;
  bool first_started = false;
  bool release_first = false;
  int processed = 0;
  int published = 0;

  AsyncApriltagProcessor::Hooks hooks;
  hooks.max_job_age_sec = 0.05;
  hooks.should_process = [](const std::string &) { return true; };
  hooks.process = [&](const std::string &,
                      const sensor_msgs::msg::Image &source,
                      const sensor_msgs::msg::CameraInfo &,
                      const DepthFrameProvider &, sensor_msgs::msg::Image *) {
    if (source.data.at(0) == 1) {
      std::unique_lock<std::mutex> lock(mutex);
      first_started = true;
      condition.notify_all();
      condition.wait(lock, [&]() { return release_first; });
    }
    std::lock_guard<std::mutex> lock(mutex);
    ++processed;
  };

  AsyncApriltagProcessor processor(std::move(hooks), {"camera_a"});
  processor.submit("camera_a", makeImage(1), makeCameraInfo(1), std::nullopt,
                   makeImage(1),
                   [&](sensor_msgs::msg::Image &&) { ++published; });
  {
    std::unique_lock<std::mutex> lock(mutex);
    const bool started =
        condition.wait_for(lock, 1s, [&]() { return first_started; });
    EXPECT_TRUE(started);
    if (!started) {
      processor.stop();
      return;
    }
  }

  processor.submit("camera_a", makeImage(2), makeCameraInfo(2), std::nullopt,
                   makeImage(2),
                   [&](sensor_msgs::msg::Image &&) { ++published; });
  std::this_thread::sleep_for(80ms);
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_first = true;
  }
  condition.notify_all();

  const bool stale_dropped =
      waitUntil([&]() { return processor.stats().at(0).stale_dropped == 1; });
  processor.stop();

  ASSERT_TRUE(stale_dropped);
  EXPECT_EQ(processed, 1);
  EXPECT_EQ(published, 1);
  const auto stats = processor.stats();
  EXPECT_EQ(stats[0].submitted, 2U);
  EXPECT_EQ(stats[0].processed, 1U);
  EXPECT_EQ(stats[0].stale_dropped, 1U);
}

TEST(AsyncApriltagProcessorTest, RejectsInvalidConstruction) {
  AsyncApriltagProcessor::Hooks missing_hooks;
  EXPECT_THROW(AsyncApriltagProcessor(std::move(missing_hooks), {"camera"}),
               std::invalid_argument);

  AsyncApriltagProcessor::Hooks invalid_age;
  invalid_age.should_process = [](const std::string &) { return true; };
  invalid_age.process = [](const std::string &, const sensor_msgs::msg::Image &,
                           const sensor_msgs::msg::CameraInfo &,
                           const DepthFrameProvider &,
                           sensor_msgs::msg::Image *) {};
  invalid_age.max_job_age_sec = 0.0;
  EXPECT_THROW(AsyncApriltagProcessor(std::move(invalid_age), {"camera"}),
               std::invalid_argument);
}

} // namespace
} // namespace zed_launcher

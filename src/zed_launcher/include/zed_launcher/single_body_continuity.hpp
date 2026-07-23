#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <zed_msgs/msg/object.hpp>

namespace zed_launcher {

// Maintains the ROS-facing identity and a short constant-velocity bridge for a
// single-human stream. This class deliberately has no ROS clock or SDK
// dependency so continuity decisions always use an injectable monotonic time.
class SingleBodyContinuity {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  static constexpr int8_t kSearchingTrackingState = 2;

  SingleBodyContinuity(int logical_id, double bridge_timeout_sec)
      : logical_id_(checkedLogicalId(logical_id)),
        bridge_timeout_(checkedTimeout(bridge_timeout_sec)) {}

  zed_msgs::msg::Object observe(const zed_msgs::msg::Object &observation,
                                TimePoint observed_at) {
    auto logical_observation = observation;
    applyLogicalIdentity(logical_observation);
    last_observation_ = logical_observation;
    last_observed_at_ = observed_at;
    return logical_observation;
  }

  [[nodiscard]] std::optional<zed_msgs::msg::Object>
  bridge(TimePoint now) const {
    if (!last_observation_) {
      return std::nullopt;
    }

    const double elapsed_sec =
        std::chrono::duration<double>(now - last_observed_at_).count();
    if (!std::isfinite(elapsed_sec) || elapsed_sec < 0.0 ||
        elapsed_sec > bridge_timeout_.count()) {
      return std::nullopt;
    }

    auto prediction = *last_observation_;
    prediction.tracking_state = kSearchingTrackingState;
    translateWithVelocity(prediction, elapsed_sec);
    return prediction;
  }

  void reset() { last_observation_.reset(); }

private:
  static int16_t checkedLogicalId(int logical_id) {
    if (logical_id < 0 ||
        logical_id > static_cast<int>(std::numeric_limits<int16_t>::max())) {
      throw std::invalid_argument(
          "single_body_logical_id must be in the range [0, 32767]");
    }
    return static_cast<int16_t>(logical_id);
  }

  static std::chrono::duration<double> checkedTimeout(double timeout_sec) {
    if (!std::isfinite(timeout_sec) || timeout_sec < 0.0) {
      throw std::invalid_argument(
          "single_body_bridge_timeout_sec must be finite and non-negative");
    }
    return std::chrono::duration<double>(timeout_sec);
  }

  void applyLogicalIdentity(zed_msgs::msg::Object &object) const {
    object.label_id = logical_id_;
    object.label = "Body_" + std::to_string(logical_id_);
  }

  static std::size_t skeletonKeypointCount(int8_t body_format) {
    switch (body_format) {
    case 0:
      return 18;
    case 1:
      return 34;
    case 2:
      return 38;
    default:
      return 0;
    }
  }

  template <typename Point>
  static void translatePoint(Point &point, const double delta[3]) {
    const std::size_t count = std::min<std::size_t>(point.size(), 3);
    for (std::size_t axis = 0; axis < count; ++axis) {
      if (std::isfinite(point[axis])) {
        point[axis] =
            static_cast<float>(static_cast<double>(point[axis]) + delta[axis]);
      }
    }
  }

  template <typename Box>
  static void translateBox(Box &box, const double delta[3]) {
    for (auto &corner : box.corners) {
      translatePoint(corner.kp, delta);
    }
  }

  static bool hasDimensions(const zed_msgs::msg::Object &object) {
    return std::any_of(
        object.dimensions_3d.begin(), object.dimensions_3d.end(),
        [](float dimension) {
          return std::isfinite(dimension) && dimension > 0.0F;
        });
  }

  static void translateWithVelocity(zed_msgs::msg::Object &object,
                                    double elapsed_sec) {
    double delta[3]{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (std::isfinite(object.velocity[axis])) {
        delta[axis] =
            static_cast<double>(object.velocity[axis]) * elapsed_sec;
      }
    }

    translatePoint(object.position, delta);
    translatePoint(object.head_position, delta);

    if (hasDimensions(object)) {
      translateBox(object.bounding_box_3d, delta);
      translateBox(object.head_bounding_box_3d, delta);
    }

    if (!object.skeleton_available) {
      return;
    }
    const auto keypoint_count =
        std::min(skeletonKeypointCount(object.body_format),
                 object.skeleton_3d.keypoints.size());
    for (std::size_t index = 0; index < keypoint_count; ++index) {
      translatePoint(object.skeleton_3d.keypoints[index].kp, delta);
    }
  }

  int16_t logical_id_;
  std::chrono::duration<double> bridge_timeout_;
  std::optional<zed_msgs::msg::Object> last_observation_;
  TimePoint last_observed_at_{};
};

} // namespace zed_launcher

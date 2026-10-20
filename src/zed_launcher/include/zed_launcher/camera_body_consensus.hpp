#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <zed_msgs/msg/object.hpp>

namespace zed_launcher {

inline bool hasFiniteBodyPosition(const zed_msgs::msg::Object &body) {
  return std::isfinite(body.position[0]) && std::isfinite(body.position[1]) &&
         std::isfinite(body.position[2]);
}

inline double bodyPositionDistance(const zed_msgs::msg::Object &lhs,
                                   const zed_msgs::msg::Object &rhs) {
  const double dx = static_cast<double>(lhs.position[0]) - rhs.position[0];
  const double dy = static_cast<double>(lhs.position[1]) - rhs.position[1];
  const double dz = static_cast<double>(lhs.position[2]) - rhs.position[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Returns the camera observation with the strongest mutually consistent
// support. Callers provide at most one candidate per camera.
inline std::optional<std::size_t>
selectCameraBodyConsensus(const std::vector<zed_msgs::msg::Object> &candidates,
                          std::size_t minimum_cameras,
                          double maximum_distance_m) {
  if (minimum_cameras == 0 || !std::isfinite(maximum_distance_m) ||
      maximum_distance_m < 0.0) {
    return std::nullopt;
  }

  std::optional<std::size_t> best_index;
  std::size_t best_support = 0;
  double best_distance_sum = std::numeric_limits<double>::infinity();
  float best_confidence = -std::numeric_limits<float>::infinity();
  for (std::size_t candidate_index = 0; candidate_index < candidates.size();
       ++candidate_index) {
    const auto &candidate = candidates[candidate_index];
    if (!hasFiniteBodyPosition(candidate)) {
      continue;
    }

    std::size_t support = 0;
    double distance_sum = 0.0;
    for (const auto &other : candidates) {
      if (!hasFiniteBodyPosition(other)) {
        continue;
      }
      const double distance = bodyPositionDistance(candidate, other);
      if (distance <= maximum_distance_m) {
        ++support;
        distance_sum += distance;
      }
    }
    if (support < minimum_cameras) {
      continue;
    }

    const bool better =
        !best_index || support > best_support ||
        (support == best_support && distance_sum < best_distance_sum) ||
        (support == best_support && distance_sum == best_distance_sum &&
         candidate.confidence > best_confidence);
    if (better) {
      best_index = candidate_index;
      best_support = support;
      best_distance_sum = distance_sum;
      best_confidence = candidate.confidence;
    }
  }
  return best_index;
}

} // namespace zed_launcher

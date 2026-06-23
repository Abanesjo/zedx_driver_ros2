#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <g1_cbf_msg/msg/capsule_array.hpp>
#include <rclcpp/rclcpp.hpp>

namespace human_mapping {

using MergeCapsuleArray = g1_cbf_msg::msg::CapsuleArray;

class HumanColliderMergeNode : public rclcpp::Node {
public:
  HumanColliderMergeNode();

private:
  void bodyCallback(const MergeCapsuleArray::SharedPtr msg);

  void handCallback(const MergeCapsuleArray::SharedPtr msg);

  void timerCallback();

  void publishMerged();

  bool fresh(const std::optional<int64_t> &recv_ns,
             double timeout_sec) const;

  bool framesCompatible(const MergeCapsuleArray &body,
                        const MergeCapsuleArray &hand);

  std::string body_collider_topic_;

  std::string hand_collider_topic_;

  std::string output_collider_topic_;

  std::string fallback_frame_id_;

  bool include_hand_colliders_ = true;

  double publish_rate_hz_ = 60.0;

  double body_stale_timeout_sec_ = 1.0;

  double hand_stale_timeout_sec_ = 1.0;

  int64_t last_frame_warn_ns_ = 0;

  MergeCapsuleArray::SharedPtr latest_body_;

  MergeCapsuleArray::SharedPtr latest_hand_;

  std::optional<int64_t> latest_body_recv_ns_;

  std::optional<int64_t> latest_hand_recv_ns_;

  rclcpp::Subscription<MergeCapsuleArray>::SharedPtr body_sub_;

  rclcpp::Subscription<MergeCapsuleArray>::SharedPtr hand_sub_;

  rclcpp::Publisher<MergeCapsuleArray>::SharedPtr merged_pub_;

  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace human_mapping

#include "human_mapping/human_collider_merge_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>

namespace human_mapping {

HumanColliderMergeNode::HumanColliderMergeNode()
    : Node("human_collider_merge_node") {
  body_collider_topic_ = declare_parameter<std::string>(
      "body_collider_topic", "/human/body_colliders");
  hand_collider_topic_ = declare_parameter<std::string>(
      "hand_collider_topic", "/human/hand_colliders");
  output_collider_topic_ = declare_parameter<std::string>(
      "output_collider_topic", "/human/colliders");
  fallback_frame_id_ =
      declare_parameter<std::string>("fallback_frame_id", "fusion_world");
  include_hand_colliders_ =
      declare_parameter<bool>("include_hand_colliders", true);
  publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 60.0);
  body_stale_timeout_sec_ =
      declare_parameter<double>("body_stale_timeout_sec", 1.0);
  hand_stale_timeout_sec_ =
      declare_parameter<double>("hand_stale_timeout_sec", 1.0);

  if (publish_rate_hz_ <= 0.0) {
    throw std::runtime_error("publish_rate_hz must be positive");
  }
  if (body_stale_timeout_sec_ < 0.0 || hand_stale_timeout_sec_ < 0.0) {
    throw std::runtime_error("stale timeouts must be non-negative");
  }

  rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
  sensor_qos.best_effort();
  sensor_qos.durability_volatile();

  body_sub_ = create_subscription<MergeCapsuleArray>(
      body_collider_topic_, sensor_qos,
      std::bind(&HumanColliderMergeNode::bodyCallback, this,
                std::placeholders::_1));
  hand_sub_ = create_subscription<MergeCapsuleArray>(
      hand_collider_topic_, sensor_qos,
      std::bind(&HumanColliderMergeNode::handCallback, this,
                std::placeholders::_1));
  merged_pub_ =
      create_publisher<MergeCapsuleArray>(output_collider_topic_, sensor_qos);

  const auto period = std::chrono::nanoseconds(
      static_cast<int64_t>(1e9 / std::max(publish_rate_hz_, 1.0)));
  timer_ = create_wall_timer(
      period, std::bind(&HumanColliderMergeNode::timerCallback, this));

  RCLCPP_INFO(get_logger(), "human_collider_merge_node ready: %s + %s -> %s",
              body_collider_topic_.c_str(), hand_collider_topic_.c_str(),
              output_collider_topic_.c_str());
}

void HumanColliderMergeNode::bodyCallback(
    const MergeCapsuleArray::SharedPtr msg) {
  latest_body_ = msg;
  latest_body_recv_ns_ = now().nanoseconds();
}

void HumanColliderMergeNode::handCallback(
    const MergeCapsuleArray::SharedPtr msg) {
  latest_hand_ = msg;
  latest_hand_recv_ns_ = now().nanoseconds();
}

void HumanColliderMergeNode::timerCallback() { publishMerged(); }

void HumanColliderMergeNode::publishMerged() {
  if (!latest_body_ ||
      !fresh(latest_body_recv_ns_, body_stale_timeout_sec_)) {
    return;
  }

  MergeCapsuleArray merged = *latest_body_;
  if (merged.header.frame_id.empty()) {
    merged.header.frame_id = fallback_frame_id_;
  }

  if (include_hand_colliders_ && latest_hand_ &&
      fresh(latest_hand_recv_ns_, hand_stale_timeout_sec_) &&
      framesCompatible(merged, *latest_hand_)) {
    merged.capsules.insert(merged.capsules.end(), latest_hand_->capsules.begin(),
                           latest_hand_->capsules.end());
  }

  merged_pub_->publish(merged);
}

bool HumanColliderMergeNode::fresh(const std::optional<int64_t> &recv_ns,
                                   double timeout_sec) const {
  if (!recv_ns) {
    return false;
  }
  if (timeout_sec == 0.0) {
    return true;
  }
  return static_cast<double>(now().nanoseconds() - *recv_ns) * 1e-9 <=
         timeout_sec;
}

bool HumanColliderMergeNode::framesCompatible(const MergeCapsuleArray &body,
                                              const MergeCapsuleArray &hand) {
  if (hand.header.frame_id.empty() ||
      hand.header.frame_id == body.header.frame_id) {
    return true;
  }

  const int64_t now_ns = now().nanoseconds();
  if (now_ns - last_frame_warn_ns_ > 2'000'000'000LL) {
    last_frame_warn_ns_ = now_ns;
    RCLCPP_WARN(get_logger(),
                "Skipping hand colliders with frame_id='%s'; body frame_id='%s'",
                hand.header.frame_id.c_str(), body.header.frame_id.c_str());
  }
  return false;
}

} // namespace human_mapping

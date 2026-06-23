#pragma once

#include "human_mapping/geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <zed_msgs/msg/object.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>

namespace human_mapping {

using HandFloat32MultiArray = std_msgs::msg::Float32MultiArray;
using ZedObject = zed_msgs::msg::Object;
using ZedObjectsStamped = zed_msgs::msg::ObjectsStamped;

namespace body38 {
inline constexpr std::size_t kNumPoints = 38;
inline constexpr int kLeftElbow = 14;
inline constexpr int kRightElbow = 15;
inline constexpr int kLeftWrist = 16;
inline constexpr int kRightWrist = 17;
inline constexpr int kLeftThumbTip = 30;
inline constexpr int kRightThumbTip = 31;
inline constexpr int kLeftIndexKnuckle = 32;
inline constexpr int kRightIndexKnuckle = 33;
inline constexpr int kLeftMiddleTip = 34;
inline constexpr int kRightMiddleTip = 35;
inline constexpr int kLeftPinkyKnuckle = 36;
inline constexpr int kRightPinkyKnuckle = 37;
} // namespace body38

enum class HandOutputLayout {
  Finger10,
  Finger12,
  Hand2,
};

struct HandSideConfig {
  std::string name;
  int wrist_idx = 0;
  int elbow_idx = 0;
  std::array<int, 4> finger_indices{};
  double open_score_ref = 0.45;
  double fist_score_ref = 0.20;
};

struct HandEstimate {
  double closure = 1.0;
  double open_score = 0.0;
  double spread = 0.0;
  double extension = 0.0;
  int valid_points = 0;
};

class SideGestureState {
public:
  SideGestureState(double initial_state, double ema_alpha);

  double updateClosure(double closure);

  double state = 1.0;
  std::optional<double> closure_ema;
  std::optional<double> last_valid_time_sec;

private:
  double ema_alpha_ = 0.30;
};

class ZedHandMappingNode : public rclcpp::Node {
public:
  ZedHandMappingNode();

private:
  void skeletonCallback(const ZedObjectsStamped::SharedPtr msg);

  const ZedObject *selectBody(const ZedObjectsStamped &msg);

  bool isBody38(int body_format) const;

  void warnBodyFormat(int body_format);

  HandSideConfig makeSideConfig(const std::string &side) const;

  void validateParams() const;

  HandOutputLayout parseOutputLayout(const std::string &layout) const;

  std::optional<Vec3> pointAt(const ZedObject &obj, int idx) const;

  std::optional<HandEstimate> estimateClosure(const ZedObject &obj,
                                              const HandSideConfig &cfg) const;

  double updateSide(SideGestureState &state,
                    const std::optional<HandEstimate> &estimate,
                    double now_sec) const;

  std::array<float, 5> finger5FromClosure(double closure) const;

  std::vector<float> outputData(double left_closure,
                                double right_closure) const;

  double meanPairwiseDistance(const std::vector<Vec3> &points) const;

  bool validPoint(const Vec3 &point) const;

  void maybeLogDebug(const std::optional<HandEstimate> &left_est,
                     const std::optional<HandEstimate> &right_est,
                     double left_out, double right_out,
                     const std::vector<float> &out_data);

  std::string formatEstimate(const std::string &label,
                             const std::optional<HandEstimate> &estimate) const;

  std::string input_skeleton_topic_;

  std::string output_topic_;

  double min_scale_m_ = 0.05;

  double fallback_scale_m_ = 0.25;

  int min_valid_hand_points_ = 1;

  double spread_weight_ = 0.60;

  double extension_weight_ = 0.40;

  double closure_ema_alpha_ = 0.30;

  double open_threshold_ = 0.35;

  double fist_threshold_ = 0.65;

  bool publish_continuous_open_amount_ = false;

  double missing_timeout_sec_ = 0.50;

  double timeout_state_ = 1.0;

  bool hold_last_on_timeout_ = true;

  HandOutputLayout output_layout_ = HandOutputLayout::Finger10;

  bool ring_follows_middle_ = true;

  bool require_body_38_ = true;

  bool debug_log_ = false;

  double debug_log_period_sec_ = 1.0;

  HandSideConfig left_cfg_;

  HandSideConfig right_cfg_;

  SideGestureState left_state_;

  SideGestureState right_state_;

  int64_t last_body_format_warn_ns_ = 0;

  int64_t last_debug_log_ns_ = 0;

  rclcpp::Publisher<HandFloat32MultiArray>::SharedPtr hand_pub_;

  rclcpp::Subscription<ZedObjectsStamped>::SharedPtr skeleton_sub_;
};

} // namespace human_mapping

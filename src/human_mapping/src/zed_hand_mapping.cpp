#include "human_mapping/zed_hand_mapping_node.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace human_mapping {
namespace {

std::string layoutName(HandOutputLayout layout) {
  switch (layout) {
  case HandOutputLayout::Finger10:
    return "finger10";
  case HandOutputLayout::Finger12:
    return "finger12";
  case HandOutputLayout::Hand2:
    return "hand2";
  }
  return "unknown";
}

} // namespace

SideGestureState::SideGestureState(double initial_state, double ema_alpha)
    : state(initial_state), ema_alpha_(ema_alpha) {}

double SideGestureState::updateClosure(double closure) {
  closure = std::clamp(closure, 0.0, 1.0);
  if (!closure_ema) {
    closure_ema = closure;
  } else {
    closure_ema = ema_alpha_ * closure + (1.0 - ema_alpha_) * *closure_ema;
  }
  return *closure_ema;
}

ZedHandMappingNode::ZedHandMappingNode()
    : Node("zed_hand_mapping_node"), left_state_(1.0, 0.30),
      right_state_(1.0, 0.30) {
  input_skeleton_topic_ = declare_parameter<std::string>(
      "input_skeleton_topic", "/zed_fusion/body_trk/skeletons");
  output_topic_ =
      declare_parameter<std::string>("output_topic", "/hand_finger_angles");

  min_valid_hand_points_ = declare_parameter<int>("min_valid_hand_points", 1);
  min_scale_m_ = declare_parameter<double>("min_scale_m", 0.05);
  fallback_scale_m_ = declare_parameter<double>("fallback_scale_m", 0.25);

  declare_parameter<double>("open_score_ref", 0.45);
  declare_parameter<double>("fist_score_ref", 0.20);
  declare_parameter<double>("left_open_score_ref", -1.0);
  declare_parameter<double>("left_fist_score_ref", -1.0);
  declare_parameter<double>("right_open_score_ref", -1.0);
  declare_parameter<double>("right_fist_score_ref", -1.0);

  spread_weight_ = declare_parameter<double>("spread_weight", 0.60);
  extension_weight_ = declare_parameter<double>("extension_weight", 0.40);
  closure_ema_alpha_ = declare_parameter<double>("closure_ema_alpha", 0.30);

  open_threshold_ = declare_parameter<double>("open_threshold", 0.35);
  fist_threshold_ = declare_parameter<double>("fist_threshold", 0.65);

  publish_continuous_open_amount_ =
      declare_parameter<bool>("publish_continuous_open_amount", false);
  missing_timeout_sec_ = declare_parameter<double>("missing_timeout_sec", 0.50);
  timeout_state_ = declare_parameter<double>("timeout_state", 1.0);
  hold_last_on_timeout_ = declare_parameter<bool>("hold_last_on_timeout", true);

  const auto output_layout =
      declare_parameter<std::string>("output_layout", "finger10");
  output_layout_ = parseOutputLayout(output_layout);
  ring_follows_middle_ = declare_parameter<bool>("ring_follows_middle", true);
  require_body_38_ = declare_parameter<bool>("require_body_38", true);

  debug_log_ = declare_parameter<bool>("debug_log", false);
  debug_log_period_sec_ =
      declare_parameter<double>("debug_log_period_sec", 1.0);

  validateParams();

  left_cfg_ = makeSideConfig("left");
  right_cfg_ = makeSideConfig("right");
  left_state_ = SideGestureState(timeout_state_, closure_ema_alpha_);
  right_state_ = SideGestureState(timeout_state_, closure_ema_alpha_);

  rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
  sensor_qos.best_effort();
  sensor_qos.durability_volatile();

  hand_pub_ =
      create_publisher<HandFloat32MultiArray>(output_topic_, sensor_qos);
  skeleton_sub_ = create_subscription<ZedObjectsStamped>(
      input_skeleton_topic_, sensor_qos,
      std::bind(&ZedHandMappingNode::skeletonCallback, this,
                std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "zed_hand_mapping_node ready: %s -> %s",
              input_skeleton_topic_.c_str(), output_topic_.c_str());
  RCLCPP_INFO(get_logger(), "output_layout=%s, require_body_38=%s",
              layoutName(output_layout_).c_str(),
              require_body_38_ ? "true" : "false");
  RCLCPP_INFO(
      get_logger(),
      "BODY_38 hand indices: L thumb=%d, L index=%d, L middle=%d, "
      "L pinky=%d; R thumb=%d, R index=%d, R middle=%d, R pinky=%d",
      body38::kLeftThumbTip, body38::kLeftIndexKnuckle,
      body38::kLeftMiddleTip, body38::kLeftPinkyKnuckle,
      body38::kRightThumbTip, body38::kRightIndexKnuckle,
      body38::kRightMiddleTip, body38::kRightPinkyKnuckle);
}

void ZedHandMappingNode::skeletonCallback(
    const ZedObjectsStamped::SharedPtr msg) {
  const ZedObject *obj = selectBody(*msg);
  if (obj == nullptr) {
    return;
  }

  const double now_sec = static_cast<double>(now().nanoseconds()) * 1e-9;
  const auto left_est = estimateClosure(*obj, left_cfg_);
  const auto right_est = estimateClosure(*obj, right_cfg_);

  const double left_out = updateSide(left_state_, left_est, now_sec);
  const double right_out = updateSide(right_state_, right_est, now_sec);

  HandFloat32MultiArray out;
  out.data = outputData(left_out, right_out);
  hand_pub_->publish(out);

  maybeLogDebug(left_est, right_est, left_out, right_out, out.data);
}

const ZedObject *ZedHandMappingNode::selectBody(const ZedObjectsStamped &msg) {
  const ZedObject *best = nullptr;
  double best_confidence = -std::numeric_limits<double>::infinity();
  for (const auto &obj : msg.objects) {
    if (!obj.skeleton_available) {
      continue;
    }
    if (require_body_38_ && !isBody38(obj.body_format)) {
      warnBodyFormat(obj.body_format);
      continue;
    }
    if (obj.skeleton_3d.keypoints.size() <=
        static_cast<std::size_t>(body38::kRightPinkyKnuckle)) {
      continue;
    }
    const double confidence = static_cast<double>(obj.confidence);
    if (confidence > best_confidence) {
      best = &obj;
      best_confidence = confidence;
    }
  }
  return best;
}

bool ZedHandMappingNode::isBody38(int body_format) const {
  return body_format == 2 || body_format == 38;
}

void ZedHandMappingNode::warnBodyFormat(int body_format) {
  const int64_t now_ns = now().nanoseconds();
  if (now_ns - last_body_format_warn_ns_ < 2'000'000'000LL) {
    return;
  }
  last_body_format_warn_ns_ = now_ns;
  RCLCPP_WARN(get_logger(),
              "Ignoring skeleton with body_format=%d; BODY_38 is required.",
              body_format);
}

HandSideConfig
ZedHandMappingNode::makeSideConfig(const std::string &side) const {
  const double global_open = get_parameter("open_score_ref").as_double();
  const double global_fist = get_parameter("fist_score_ref").as_double();
  const double side_open =
      get_parameter(side + "_open_score_ref").as_double();
  const double side_fist =
      get_parameter(side + "_fist_score_ref").as_double();

  const double open_ref = side_open > 0.0 ? side_open : global_open;
  const double fist_ref = side_fist > 0.0 ? side_fist : global_fist;
  if (open_ref <= fist_ref) {
    throw std::runtime_error(side +
                             " open_score_ref must be greater than "
                             "fist_score_ref");
  }

  if (side == "left") {
    return HandSideConfig{
        "left",
        body38::kLeftWrist,
        body38::kLeftElbow,
        {body38::kLeftThumbTip, body38::kLeftIndexKnuckle,
         body38::kLeftMiddleTip, body38::kLeftPinkyKnuckle},
        open_ref,
        fist_ref};
  }

  return HandSideConfig{
      "right",
      body38::kRightWrist,
      body38::kRightElbow,
      {body38::kRightThumbTip, body38::kRightIndexKnuckle,
       body38::kRightMiddleTip, body38::kRightPinkyKnuckle},
      open_ref,
      fist_ref};
}

void ZedHandMappingNode::validateParams() const {
  if (min_valid_hand_points_ < 1 || min_valid_hand_points_ > 4) {
    throw std::runtime_error("min_valid_hand_points must be in [1, 4]");
  }
  if (min_scale_m_ < 0.0) {
    throw std::runtime_error("min_scale_m must be non-negative");
  }
  if (fallback_scale_m_ <= 0.0) {
    throw std::runtime_error("fallback_scale_m must be positive");
  }
  if (closure_ema_alpha_ < 0.0 || closure_ema_alpha_ > 1.0) {
    throw std::runtime_error("closure_ema_alpha must be in [0, 1]");
  }
  if (timeout_state_ < 0.0 || timeout_state_ > 1.0) {
    throw std::runtime_error("timeout_state must be in [0, 1]");
  }
  if (open_threshold_ >= fist_threshold_) {
    throw std::runtime_error(
        "open_threshold must be smaller than fist_threshold");
  }
  if (missing_timeout_sec_ < 0.0) {
    throw std::runtime_error("missing_timeout_sec must be non-negative");
  }
  if (debug_log_period_sec_ <= 0.0) {
    throw std::runtime_error("debug_log_period_sec must be positive");
  }
}

HandOutputLayout
ZedHandMappingNode::parseOutputLayout(const std::string &layout) const {
  if (layout == "finger10") {
    return HandOutputLayout::Finger10;
  }
  if (layout == "finger12") {
    return HandOutputLayout::Finger12;
  }
  if (layout == "hand2") {
    return HandOutputLayout::Hand2;
  }
  throw std::runtime_error(
      "output_layout must be one of: finger10, finger12, hand2");
}

std::optional<Vec3> ZedHandMappingNode::pointAt(const ZedObject &obj,
                                                int idx) const {
  if (idx < 0 ||
      static_cast<std::size_t>(idx) >= obj.skeleton_3d.keypoints.size()) {
    return std::nullopt;
  }

  const auto &kp = obj.skeleton_3d.keypoints[static_cast<std::size_t>(idx)].kp;
  const Vec3 point{static_cast<double>(kp[0]), static_cast<double>(kp[1]),
                   static_cast<double>(kp[2])};
  if (!validPoint(point)) {
    return std::nullopt;
  }
  return point;
}

std::optional<HandEstimate>
ZedHandMappingNode::estimateClosure(const ZedObject &obj,
                                    const HandSideConfig &cfg) const {
  const auto wrist = pointAt(obj, cfg.wrist_idx);
  const auto elbow = pointAt(obj, cfg.elbow_idx);
  if (!wrist) {
    return std::nullopt;
  }

  std::vector<Vec3> finger_points;
  finger_points.reserve(cfg.finger_indices.size());
  for (const int idx : cfg.finger_indices) {
    const auto point = pointAt(obj, idx);
    if (point) {
      finger_points.push_back(*point);
    }
  }

  if (finger_points.size() < static_cast<std::size_t>(min_valid_hand_points_)) {
    return std::nullopt;
  }

  double scale = fallback_scale_m_;
  if (elbow) {
    const double forearm_len = norm(*wrist - *elbow);
    if (forearm_len >= min_scale_m_) {
      scale = forearm_len;
    }
  }

  double extension_sum = 0.0;
  for (const auto &point : finger_points) {
    extension_sum += norm(point - *wrist);
  }

  const double spread = meanPairwiseDistance(finger_points) / scale;
  const double extension =
      extension_sum / static_cast<double>(finger_points.size()) / scale;
  const double open_score =
      spread_weight_ * spread + extension_weight_ * extension;
  const double denom = cfg.open_score_ref - cfg.fist_score_ref;
  const double open_amount =
      std::clamp((open_score - cfg.fist_score_ref) / denom, 0.0, 1.0);

  return HandEstimate{1.0 - open_amount, open_score, spread, extension,
                      static_cast<int>(finger_points.size())};
}

double ZedHandMappingNode::updateSide(
    SideGestureState &state, const std::optional<HandEstimate> &estimate,
    double now_sec) const {
  if (!estimate) {
    const bool timed_out =
        !state.last_valid_time_sec ||
        now_sec - *state.last_valid_time_sec > missing_timeout_sec_;
    if (timed_out && !hold_last_on_timeout_) {
      state.state = timeout_state_;
      state.closure_ema = timeout_state_;
    }
    return state.state;
  }

  state.last_valid_time_sec = now_sec;
  const double closure_smoothed = state.updateClosure(estimate->closure);
  if (closure_smoothed >= fist_threshold_) {
    state.state = 1.0;
  } else if (closure_smoothed <= open_threshold_) {
    state.state = 0.0;
  }

  if (publish_continuous_open_amount_) {
    return closure_smoothed;
  }
  return state.state;
}

std::array<float, 5>
ZedHandMappingNode::finger5FromClosure(double closure) const {
  const float open_amount =
      static_cast<float>(1.0 - std::clamp(closure, 0.0, 1.0));
  std::array<float, 5> fingers{open_amount, open_amount, open_amount,
                               open_amount, open_amount};
  if (ring_follows_middle_) {
    fingers[3] = fingers[2];
  }
  return fingers;
}

std::vector<float> ZedHandMappingNode::outputData(double left_closure,
                                                  double right_closure) const {
  const auto left5 = finger5FromClosure(left_closure);
  const auto right5 = finger5FromClosure(right_closure);

  if (output_layout_ == HandOutputLayout::Hand2) {
    return {static_cast<float>(1.0 - std::clamp(left_closure, 0.0, 1.0)),
            static_cast<float>(1.0 - std::clamp(right_closure, 0.0, 1.0))};
  }

  if (output_layout_ == HandOutputLayout::Finger12) {
    return {left5[0],  left5[0],  left5[1],  left5[2],
            left5[3],  left5[4],  right5[0], right5[0],
            right5[1], right5[2], right5[3], right5[4]};
  }

  return {left5[0], left5[1], left5[2], left5[3], left5[4],
          right5[0], right5[1], right5[2], right5[3], right5[4]};
}

double
ZedHandMappingNode::meanPairwiseDistance(const std::vector<Vec3> &points) const {
  double sum = 0.0;
  int count = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1; j < points.size(); ++j) {
      sum += norm(points[i] - points[j]);
      ++count;
    }
  }

  if (count == 0) {
    return 0.0;
  }
  return sum / static_cast<double>(count);
}

bool ZedHandMappingNode::validPoint(const Vec3 &point) const {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z) && norm(point) > kEps;
}

void ZedHandMappingNode::maybeLogDebug(
    const std::optional<HandEstimate> &left_est,
    const std::optional<HandEstimate> &right_est, double left_out,
    double right_out, const std::vector<float> &out_data) {
  if (!debug_log_) {
    return;
  }

  const int64_t now_ns = now().nanoseconds();
  if (now_ns - last_debug_log_ns_ <
      static_cast<int64_t>(debug_log_period_sec_ * 1e9)) {
    return;
  }
  last_debug_log_ns_ = now_ns;

  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << "[zed_hand_mapping] left=" << left_out << ", right=" << right_out
      << ", out=[";
  for (std::size_t i = 0; i < out_data.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << std::setprecision(3) << out_data[i];
  }
  out << "] | " << formatEstimate("left", left_est) << " | "
      << formatEstimate("right", right_est);

  RCLCPP_INFO(get_logger(), "%s", out.str().c_str());
}

std::string ZedHandMappingNode::formatEstimate(
    const std::string &label,
    const std::optional<HandEstimate> &estimate) const {
  if (!estimate) {
    return label + "=missing";
  }

  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << label
      << ": closure=" << estimate->closure << ", score="
      << estimate->open_score << ", spread=" << estimate->spread
      << ", extension=" << estimate->extension
      << ", valid=" << estimate->valid_points;
  return out.str();
}

} // namespace human_mapping

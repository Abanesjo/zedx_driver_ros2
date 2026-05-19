#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "g1_cbf_msg/msg/capsule.hpp"
#include "g1_cbf_msg/msg/capsule_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "zed_msgs/msg/object.hpp"
#include "zed_msgs/msg/objects_stamped.hpp"

namespace
{

using builtin_interfaces::msg::Time;
using g1_cbf_msg::msg::Capsule;
using g1_cbf_msg::msg::CapsuleArray;
using sensor_msgs::msg::JointState;
using zed_msgs::msg::Object;
using zed_msgs::msg::ObjectsStamped;

constexpr std::size_t kNumJoints = 8;
constexpr std::size_t kNumBody38Points = 38;
constexpr double kEps = 1e-8;
constexpr double kPi = 3.14159265358979323846;

constexpr int kPelvis = 0;
constexpr int kNeck = 4;
constexpr int kLeftShoulder = 12;
constexpr int kRightShoulder = 13;
constexpr int kLeftElbow = 14;
constexpr int kRightElbow = 15;
constexpr int kLeftWrist = 16;
constexpr int kRightWrist = 17;
constexpr int kLeftHip = 18;
constexpr int kRightHip = 19;
constexpr int kLeftKnee = 20;
constexpr int kRightKnee = 21;
constexpr int kLeftAnkle = 22;
constexpr int kRightAnkle = 23;

const std::vector<std::string> kControlledJoints = {
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_elbow_joint",
};

struct Vec3
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Vec3 operator+(const Vec3 & a, const Vec3 & b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 & a, const Vec3 & b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 & v, double s)
{
  return {v.x * s, v.y * s, v.z * s};
}

Vec3 operator*(double s, const Vec3 & v)
{
  return v * s;
}

double dot(const Vec3 & a, const Vec3 & b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 & a, const Vec3 & b)
{
  return {
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x};
}

double norm(const Vec3 & v)
{
  return std::sqrt(dot(v, v));
}

std::optional<Vec3> normalize(const Vec3 & v)
{
  const double n = norm(v);
  if (n < kEps || !std::isfinite(n)) {
    return std::nullopt;
  }
  return v * (1.0 / n);
}

bool validPoint(const Vec3 & p)
{
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && norm(p) > kEps;
}

bool validPoint(const std::optional<Vec3> & p)
{
  return p.has_value() && validPoint(*p);
}

std::optional<double> angleBetween(const Vec3 & a, const Vec3 & b)
{
  const auto an = normalize(a);
  const auto bn = normalize(b);
  if (!an || !bn) {
    return std::nullopt;
  }
  return std::acos(std::clamp(dot(*an, *bn), -1.0, 1.0));
}

struct TorsoFrame
{
  Vec3 right;
  Vec3 forward;
  Vec3 up;
};

std::optional<TorsoFrame> buildTorsoFrame(
  const Vec3 & pelvis, const Vec3 & left_shoulder, const Vec3 & right_shoulder)
{
  const Vec3 shoulder_center = 0.5 * (left_shoulder + right_shoulder);
  const auto torso_up = normalize(shoulder_center - pelvis);
  if (!torso_up) {
    return std::nullopt;
  }

  Vec3 shoulder_right = right_shoulder - left_shoulder;
  shoulder_right = shoulder_right - (*torso_up * dot(shoulder_right, *torso_up));
  const auto torso_right = normalize(shoulder_right);
  if (!torso_right) {
    return std::nullopt;
  }

  const auto torso_forward = normalize(cross(*torso_up, *torso_right));
  if (!torso_forward) {
    return std::nullopt;
  }

  return TorsoFrame{*torso_right, *torso_forward, *torso_up};
}

Vec3 transposeMultiply(const TorsoFrame & frame, const Vec3 & v)
{
  return {dot(frame.right, v), dot(frame.forward, v), dot(frame.up, v)};
}

std::array<double, kNumJoints> vectorToArray(
  const std::vector<double> & values, const std::string & name)
{
  if (values.size() != kNumJoints) {
    throw std::runtime_error("Parameter '" + name + "' must contain 8 values");
  }

  std::array<double, kNumJoints> out{};
  std::copy(values.begin(), values.end(), out.begin());
  return out;
}

std::array<double, kNumJoints> zeros()
{
  std::array<double, kNumJoints> out{};
  out.fill(0.0);
  return out;
}

class EMAJumpFilter
{
public:
  EMAJumpFilter(double alpha, double max_jump, int max_reject_count)
  : alpha_(alpha), max_jump_(max_jump), max_reject_count_(max_reject_count)
  {}

  Vec3 update(const Vec3 & x)
  {
    if (!value_) {
      value_ = x;
      reject_count_ = 0;
      last_rejected_.reset();
      return *value_;
    }

    const double jump = norm(x - *value_);
    if (jump > max_jump_) {
      if (!last_rejected_) {
        reject_count_ = 1;
        last_rejected_ = x;
        return *value_;
      }

      const double spread = norm(x - *last_rejected_);
      reject_count_ = spread <= max_jump_ ? reject_count_ + 1 : 1;
      last_rejected_ = x;
      if (reject_count_ >= max_reject_count_) {
        value_ = x;
        reject_count_ = 0;
        last_rejected_.reset();
      }
      return *value_;
    }

    reject_count_ = 0;
    last_rejected_.reset();
    value_ = x * alpha_ + *value_ * (1.0 - alpha_);
    return *value_;
  }

private:
  double alpha_;
  double max_jump_;
  int max_reject_count_;
  std::optional<Vec3> value_;
  int reject_count_ = 0;
  std::optional<Vec3> last_rejected_;
};

class AngleFilter
{
public:
  AngleFilter(double alpha, double max_rate_deg)
  : alpha_(alpha), max_rate_(max_rate_deg * kPi / 180.0)
  {
    ema_.fill(0.0);
    prev_.fill(0.0);
    has_ema_.fill(false);
    has_prev_.fill(false);
  }

  std::array<double, kNumJoints> update(
    const std::array<double, kNumJoints> & values, double dt)
  {
    dt = std::clamp(dt, 1e-3, 0.2);

    std::array<double, kNumJoints> out{};
    for (std::size_t i = 0; i < kNumJoints; ++i) {
      double y = values[i];
      if (has_ema_[i]) {
        y = alpha_ * values[i] + (1.0 - alpha_) * ema_[i];
      }
      ema_[i] = y;
      has_ema_[i] = true;

      if (has_prev_[i]) {
        const double max_step = max_rate_ * dt;
        y = prev_[i] + std::clamp(y - prev_[i], -max_step, max_step);
      }

      y = std::clamp(y, limits_[i].first, limits_[i].second);
      prev_[i] = y;
      has_prev_[i] = true;
      out[i] = y;
    }
    return out;
  }

private:
  double alpha_;
  double max_rate_;
  std::array<double, kNumJoints> ema_;
  std::array<double, kNumJoints> prev_;
  std::array<bool, kNumJoints> has_ema_;
  std::array<bool, kNumJoints> has_prev_;
  const std::array<std::pair<double, double>, kNumJoints> limits_ = {
    std::pair<double, double>{-60.0 * kPi / 180.0, 60.0 * kPi / 180.0},
    std::pair<double, double>{-60.0 * kPi / 180.0, 60.0 * kPi / 180.0},
    std::pair<double, double>{-180.0 * kPi / 180.0, 180.0 * kPi / 180.0},
    std::pair<double, double>{-90.0 * kPi / 180.0, 150.0 * kPi / 180.0},
    std::pair<double, double>{0.0, 180.0 * kPi / 180.0},
    std::pair<double, double>{-180.0 * kPi / 180.0, 180.0 * kPi / 180.0},
    std::pair<double, double>{-90.0 * kPi / 180.0, 150.0 * kPi / 180.0},
    std::pair<double, double>{0.0, 180.0 * kPi / 180.0}};
};

struct CapsuleData
{
  std::string name;
  Vec3 a;
  Vec3 b;
  double radius = 0.0;
};

struct MappingResult
{
  Time stamp;
  std::string frame_id;
  std::array<double, kNumJoints> q_des{};
  std::vector<CapsuleData> capsules;
};

}  // namespace

class HumanMappingNode : public rclcpp::Node
{
public:
  HumanMappingNode()
  : Node("human_mapping_node")
  {
    input_skeleton_topic_ =
      declare_parameter<std::string>("input_skeleton_topic", "/zed_fusion/body_trk/skeletons");
    joint_command_topic_ =
      declare_parameter<std::string>("joint_command_topic", "/joint_commands_unsafe");
    collider_topic_ = declare_parameter<std::string>("collider_topic", "/human/colliders");
    fallback_frame_id_ = declare_parameter<std::string>("fallback_frame_id", "fusion_world");
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 50.0);
    stale_timeout_sec_ = declare_parameter<double>("stale_timeout_sec", 0.5);
    min_confidence_ = declare_parameter<double>("min_confidence", 70.0);
    require_body_38_ = declare_parameter<bool>("require_body_38", true);

    const double point_alpha = declare_parameter<double>("point_ema_alpha", 0.30);
    const double point_max_jump = declare_parameter<double>("point_max_jump", 0.6);
    const int point_max_reject_count = declare_parameter<int>("point_max_reject_count", 3);
    const double angle_alpha = declare_parameter<double>("angle_ema_alpha", 0.25);
    const double angle_max_rate_deg = declare_parameter<double>("angle_max_rate_deg", 100.0);

    enable_neutral_calibration_ = declare_parameter<bool>("enable_neutral_calibration", true);
    startup_delay_sec_ = declare_parameter<double>("startup_delay_sec", 5.0);
    neutral_calibration_duration_ =
      declare_parameter<double>("neutral_calibration_duration", 10.0);

    joint_names_ = declare_parameter<std::vector<std::string>>("joint_names", kControlledJoints);
    q_home_ = vectorToArray(
      declare_parameter<std::vector<double>>(
        "q_home", {0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708}),
      "q_home");
    signs_ = vectorToArray(
      declare_parameter<std::vector<double>>(
        "signs", {-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0}),
      "signs");
    gains_ = vectorToArray(
      declare_parameter<std::vector<double>>("gains", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
      "gains");
    bias_ = vectorToArray(
      declare_parameter<std::vector<double>>("bias", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      "bias");
    q_min_ = vectorToArray(
      declare_parameter<std::vector<double>>(
        "q_min", {-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472}),
      "q_min");
    q_max_ = vectorToArray(
      declare_parameter<std::vector<double>>(
        "q_max", {0.52, 0.52, 2.6704, 2.2515, 2.0944, 2.6704, 1.5882, 2.0944}),
      "q_max");

    human_radius_scale_ = declare_parameter<double>("human_radius_scale", 1.5);
    torso_radius_ = declare_parameter<double>("torso_radius", 0.10);
    shoulder_radius_ = declare_parameter<double>("shoulder_radius", 0.05);
    arm_radius_ = declare_parameter<double>("arm_radius", 0.05);
    thigh_radius_ = declare_parameter<double>("thigh_radius", 0.065);
    shin_radius_ = declare_parameter<double>("shin_radius", 0.065);
    hand_extension_length_ = declare_parameter<double>("hand_extension_length", 0.16);

    if (joint_names_.size() != kNumJoints) {
      throw std::runtime_error("Parameter 'joint_names' must contain 8 values");
    }
    if (publish_rate_hz_ <= 0.0) {
      throw std::runtime_error("Parameter 'publish_rate_hz' must be positive");
    }

    for (std::size_t i = 0; i < kNumBody38Points; ++i) {
      point_filters_.emplace_back(point_alpha, point_max_jump, point_max_reject_count);
    }
    angle_filter_ = std::make_unique<AngleFilter>(angle_alpha, angle_max_rate_deg);

    node_start_ns_ = now().nanoseconds();
    calibration_done_ = !enable_neutral_calibration_;
    neutral_offset_.fill(0.0);

    rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
    sensor_qos.best_effort();
    sensor_qos.durability_volatile();

    joint_pub_ = create_publisher<JointState>(joint_command_topic_, sensor_qos);
    collider_pub_ = create_publisher<CapsuleArray>(collider_topic_, sensor_qos);
    skeleton_sub_ = create_subscription<ObjectsStamped>(
      input_skeleton_topic_, sensor_qos,
      std::bind(&HumanMappingNode::skeletonCallback, this, std::placeholders::_1));

    const auto period = std::chrono::nanoseconds(
      static_cast<int64_t>(1e9 / std::max(publish_rate_hz_, 1.0)));
    timer_ = create_wall_timer(period, std::bind(&HumanMappingNode::timerCallback, this));

    RCLCPP_INFO(
      get_logger(), "human_mapping_node ready: %s -> %s, %s",
      input_skeleton_topic_.c_str(), joint_command_topic_.c_str(), collider_topic_.c_str());
  }

private:
  void skeletonCallback(const ObjectsStamped::SharedPtr msg)
  {
    const Object * obj = selectBody(*msg);
    if (obj == nullptr) {
      return;
    }

    const auto points = filteredPoints(*obj);
    const auto raw_angles = estimateAngles(points);
    if (!raw_angles) {
      return;
    }

    const int64_t now_ns = now().nanoseconds();
    const double angle_dt = angleDt(now_ns);
    const auto filtered_angles = angle_filter_->update(*raw_angles, angle_dt);
    const auto delta = neutralDelta(filtered_angles, now_ns);

    std::array<double, kNumJoints> q_des{};
    for (std::size_t i = 0; i < kNumJoints; ++i) {
      q_des[i] = std::clamp(
        q_home_[i] + signs_[i] * gains_[i] * delta[i] + bias_[i],
        q_min_[i],
        q_max_[i]);
    }

    latest_result_ = MappingResult{
      msg->header.stamp,
      msg->header.frame_id.empty() ? fallback_frame_id_ : msg->header.frame_id,
      q_des,
      buildCapsules(points)};
    latest_result_time_ns_ = now_ns;
  }

  const Object * selectBody(const ObjectsStamped & msg)
  {
    const Object * best = nullptr;
    double best_confidence = -1.0;
    for (const auto & obj : msg.objects) {
      if (!obj.skeleton_available) {
        continue;
      }
      if (require_body_38_ && !isBody38(obj.body_format)) {
        warnBodyFormat(obj.body_format);
        continue;
      }
      if (static_cast<double>(obj.confidence) < min_confidence_) {
        continue;
      }
      if (obj.skeleton_3d.keypoints.size() <= static_cast<std::size_t>(kRightAnkle)) {
        continue;
      }
      if (static_cast<double>(obj.confidence) > best_confidence) {
        best = &obj;
        best_confidence = static_cast<double>(obj.confidence);
      }
    }
    return best;
  }

  bool isBody38(int body_format) const
  {
    return body_format == 2 || body_format == 38;
  }

  void warnBodyFormat(int body_format)
  {
    const int64_t now_ns = now().nanoseconds();
    if (now_ns - last_body_format_warn_ns_ < 2'000'000'000LL) {
      return;
    }
    last_body_format_warn_ns_ = now_ns;
    RCLCPP_WARN(
      get_logger(), "Ignoring skeleton with body_format=%d; BODY_38 is required.",
      body_format);
  }

  std::array<std::optional<Vec3>, kNumBody38Points> filteredPoints(const Object & obj)
  {
    std::array<std::optional<Vec3>, kNumBody38Points> points{};
    for (std::size_t idx = 0; idx < kNumBody38Points; ++idx) {
      const auto & kp = obj.skeleton_3d.keypoints[idx].kp;
      const Vec3 p{
        static_cast<double>(kp[0]),
        static_cast<double>(kp[1]),
        static_cast<double>(kp[2])};
      if (validPoint(p)) {
        points[idx] = point_filters_[idx].update(p);
      }
    }
    return points;
  }

  std::optional<std::array<double, kNumJoints>> estimateAngles(
    const std::array<std::optional<Vec3>, kNumBody38Points> & points)
  {
    const std::array<int, 7> required = {
      kPelvis, kLeftShoulder, kRightShoulder, kLeftElbow,
      kRightElbow, kLeftWrist, kRightWrist};
    for (const int idx : required) {
      if (!validPoint(points[idx])) {
        return std::nullopt;
      }
    }

    const Vec3 pelvis = *points[kPelvis];
    const Vec3 left_shoulder = *points[kLeftShoulder];
    const Vec3 right_shoulder = *points[kRightShoulder];
    const Vec3 left_elbow = *points[kLeftElbow];
    const Vec3 right_elbow = *points[kRightElbow];
    const Vec3 left_wrist = *points[kLeftWrist];
    const Vec3 right_wrist = *points[kRightWrist];

    const auto torso_frame = buildTorsoFrame(pelvis, left_shoulder, right_shoulder);
    if (!torso_frame) {
      return std::nullopt;
    }

    const Vec3 world_up{0.0, 0.0, 1.0};
    double torso_roll = std::atan2(
      -dot(torso_frame->forward, cross(world_up, torso_frame->up)),
      dot(world_up, torso_frame->up));
    if (torso_roll < -kPi / 2.0) {
      torso_roll += kPi;
    } else if (torso_roll > kPi / 2.0) {
      torso_roll -= kPi;
    }

    double torso_pitch = std::atan2(
      -dot(torso_frame->right, cross(world_up, torso_frame->up)),
      dot(world_up, torso_frame->up));
    torso_pitch = minAbs(torso_pitch, torso_pitch + kPi, torso_pitch - kPi);

    const Vec3 left_upper = left_elbow - left_shoulder;
    const Vec3 right_upper = right_elbow - right_shoulder;
    const Vec3 left_forearm = left_wrist - left_elbow;
    const Vec3 right_forearm = right_wrist - right_elbow;

    const Vec3 left_upper_t = transposeMultiply(*torso_frame, left_upper);
    const Vec3 right_upper_t = transposeMultiply(*torso_frame, right_upper);

    const auto left_elbow_pitch = angleBetween(left_upper, left_forearm);
    const auto right_elbow_pitch = angleBetween(right_upper, right_forearm);
    if (!left_elbow_pitch || !right_elbow_pitch) {
      return std::nullopt;
    }

    return std::array<double, kNumJoints>{
      torso_roll,
      torso_pitch,
      std::atan2(left_upper_t.y, -left_upper_t.z),
      std::atan2(-left_upper_t.x, -left_upper_t.z),
      *left_elbow_pitch,
      std::atan2(right_upper_t.y, -right_upper_t.z),
      std::atan2(right_upper_t.x, -right_upper_t.z),
      *right_elbow_pitch};
  }

  double minAbs(double a, double b, double c) const
  {
    double best = a;
    if (std::abs(b) < std::abs(best)) {
      best = b;
    }
    if (std::abs(c) < std::abs(best)) {
      best = c;
    }
    return best;
  }

  double angleDt(int64_t now_ns)
  {
    double dt = 1.0 / std::max(publish_rate_hz_, 1.0);
    if (last_angle_time_ns_) {
      dt = static_cast<double>(now_ns - *last_angle_time_ns_) * 1e-9;
    }
    last_angle_time_ns_ = now_ns;
    return std::clamp(dt, 1e-3, 0.2);
  }

  std::array<double, kNumJoints> neutralDelta(
    const std::array<double, kNumJoints> & angles, int64_t now_ns)
  {
    if (!enable_neutral_calibration_) {
      return angles;
    }

    const double startup_elapsed = static_cast<double>(now_ns - node_start_ns_) * 1e-9;
    if (startup_elapsed < startup_delay_sec_) {
      return zeros();
    }

    if (calibration_done_) {
      return subtract(angles, neutral_offset_);
    }

    if (!calibration_start_ns_) {
      calibration_start_ns_ = now_ns;
      calibration_samples_.clear();
      RCLCPP_INFO(
        get_logger(), "Neutral calibration started for %.2fs.",
        neutral_calibration_duration_);
    }

    calibration_samples_.push_back(angles);
    const double elapsed = static_cast<double>(now_ns - *calibration_start_ns_) * 1e-9;
    if (elapsed >= neutral_calibration_duration_) {
      if (!calibration_samples_.empty()) {
        neutral_offset_.fill(0.0);
        for (const auto & sample : calibration_samples_) {
          for (std::size_t i = 0; i < kNumJoints; ++i) {
            neutral_offset_[i] += sample[i];
          }
        }
        for (double & value : neutral_offset_) {
          value /= static_cast<double>(calibration_samples_.size());
        }
      }
      calibration_done_ = true;
      RCLCPP_INFO(get_logger(), "Neutral calibration completed.");
      return subtract(angles, neutral_offset_);
    }

    return zeros();
  }

  std::array<double, kNumJoints> subtract(
    const std::array<double, kNumJoints> & a,
    const std::array<double, kNumJoints> & b) const
  {
    std::array<double, kNumJoints> out{};
    for (std::size_t i = 0; i < kNumJoints; ++i) {
      out[i] = a[i] - b[i];
    }
    return out;
  }

  std::vector<CapsuleData> buildCapsules(
    const std::array<std::optional<Vec3>, kNumBody38Points> & points)
  {
    std::vector<CapsuleData> capsules;

    auto add = [&](const std::string & name, int a_idx, int b_idx, double radius) {
      const auto & a = points[a_idx];
      const auto & b = points[b_idx];
      if (validPoint(a) && validPoint(b)) {
        capsules.push_back({name, *a, *b, radius * human_radius_scale_});
      }
    };

    add("torso", kPelvis, kNeck, torso_radius_);

    const auto left_hand = armDistalPoint(points[kLeftElbow], points[kLeftWrist]);
    const auto right_hand = armDistalPoint(points[kRightElbow], points[kRightWrist]);
    if (validPoint(points[kLeftElbow]) && validPoint(left_hand)) {
      capsules.push_back(
        {"left_arm", *points[kLeftElbow], *left_hand, arm_radius_ * human_radius_scale_});
    }
    if (validPoint(points[kRightElbow]) && validPoint(right_hand)) {
      capsules.push_back(
        {"right_arm", *points[kRightElbow], *right_hand, arm_radius_ * human_radius_scale_});
    }

    add("left_shoulder", kLeftShoulder, kLeftElbow, shoulder_radius_);
    add("right_shoulder", kRightShoulder, kRightElbow, shoulder_radius_);
    add("left_thigh", kLeftHip, kLeftKnee, thigh_radius_);
    add("right_thigh", kRightHip, kRightKnee, thigh_radius_);
    add("left_shin", kLeftKnee, kLeftAnkle, shin_radius_);
    add("right_shin", kRightKnee, kRightAnkle, shin_radius_);

    return capsules;
  }

  std::optional<Vec3> armDistalPoint(
    const std::optional<Vec3> & elbow, const std::optional<Vec3> & wrist) const
  {
    if (!validPoint(wrist)) {
      return std::nullopt;
    }
    if (!validPoint(elbow)) {
      return wrist;
    }
    const auto unit = normalize(*wrist - *elbow);
    if (!unit) {
      return wrist;
    }
    return *wrist + *unit * hand_extension_length_;
  }

  void timerCallback()
  {
    const auto now_time = now();
    if (!latest_result_ || isStale(now_time.nanoseconds())) {
      publishEmptyAndNeutral(now_time);
      return;
    }

    publishJointCommand(latest_result_->stamp, latest_result_->q_des);
    publishColliders(
      latest_result_->stamp, latest_result_->frame_id, latest_result_->capsules);
  }

  bool isStale(int64_t now_ns) const
  {
    if (!latest_result_time_ns_) {
      return true;
    }
    return static_cast<double>(now_ns - *latest_result_time_ns_) * 1e-9 > stale_timeout_sec_;
  }

  void publishEmptyAndNeutral(const rclcpp::Time & stamp)
  {
    publishJointCommand(stamp, q_home_);
    publishColliders(stamp, fallback_frame_id_, {});
  }

  void publishJointCommand(const Time & stamp, const std::array<double, kNumJoints> & q_des)
  {
    JointState msg;
    msg.header.stamp = stamp;
    msg.name = joint_names_;
    msg.position.assign(q_des.begin(), q_des.end());
    msg.velocity.assign(joint_names_.size(), 0.0);
    joint_pub_->publish(msg);
  }

  void publishJointCommand(
    const rclcpp::Time & stamp, const std::array<double, kNumJoints> & q_des)
  {
    publishJointCommand(toMsg(stamp), q_des);
  }

  void publishColliders(
    const Time & stamp, const std::string & frame_id, const std::vector<CapsuleData> & capsules)
  {
    CapsuleArray msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id.empty() ? fallback_frame_id_ : frame_id;
    msg.capsules.reserve(capsules.size());
    for (const auto & src : capsules) {
      Capsule cap;
      cap.name = src.name;
      cap.a.x = src.a.x;
      cap.a.y = src.a.y;
      cap.a.z = src.a.z;
      cap.b.x = src.b.x;
      cap.b.y = src.b.y;
      cap.b.z = src.b.z;
      cap.radius = src.radius;
      msg.capsules.push_back(cap);
    }
    collider_pub_->publish(msg);
  }

  void publishColliders(
    const rclcpp::Time & stamp,
    const std::string & frame_id,
    const std::vector<CapsuleData> & capsules)
  {
    publishColliders(toMsg(stamp), frame_id, capsules);
  }

  Time toMsg(const rclcpp::Time & stamp) const
  {
    Time msg;
    const int64_t ns = stamp.nanoseconds();
    msg.sec = static_cast<int32_t>(ns / 1'000'000'000LL);
    msg.nanosec = static_cast<uint32_t>(ns % 1'000'000'000LL);
    return msg;
  }

  std::string input_skeleton_topic_;
  std::string joint_command_topic_;
  std::string collider_topic_;
  std::string fallback_frame_id_;
  double publish_rate_hz_ = 50.0;
  double stale_timeout_sec_ = 0.5;
  double min_confidence_ = 70.0;
  bool require_body_38_ = true;

  bool enable_neutral_calibration_ = true;
  double startup_delay_sec_ = 5.0;
  double neutral_calibration_duration_ = 10.0;
  int64_t node_start_ns_ = 0;
  std::optional<int64_t> calibration_start_ns_;
  std::vector<std::array<double, kNumJoints>> calibration_samples_;
  std::array<double, kNumJoints> neutral_offset_{};
  bool calibration_done_ = false;
  std::optional<int64_t> last_angle_time_ns_;

  std::vector<std::string> joint_names_;
  std::array<double, kNumJoints> q_home_{};
  std::array<double, kNumJoints> signs_{};
  std::array<double, kNumJoints> gains_{};
  std::array<double, kNumJoints> bias_{};
  std::array<double, kNumJoints> q_min_{};
  std::array<double, kNumJoints> q_max_{};

  double human_radius_scale_ = 1.5;
  double torso_radius_ = 0.10;
  double shoulder_radius_ = 0.05;
  double arm_radius_ = 0.05;
  double thigh_radius_ = 0.065;
  double shin_radius_ = 0.065;
  double hand_extension_length_ = 0.16;

  std::vector<EMAJumpFilter> point_filters_;
  std::unique_ptr<AngleFilter> angle_filter_;
  std::optional<MappingResult> latest_result_;
  std::optional<int64_t> latest_result_time_ns_;
  int64_t last_body_format_warn_ns_ = 0;

  rclcpp::Publisher<JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<CapsuleArray>::SharedPtr collider_pub_;
  rclcpp::Subscription<ObjectsStamped>::SharedPtr skeleton_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HumanMappingNode>());
  rclcpp::shutdown();
  return 0;
}

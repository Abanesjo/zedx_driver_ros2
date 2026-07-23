#include "human_mapping/human_mapping_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace human_mapping {
namespace {

const std::vector<std::string> kControlledJoints = {
    "waist_yaw_joint",           "waist_roll_joint",
    "waist_pitch_joint",         "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",          "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
};

constexpr double kMinShoulderYawFlex = 20.0 * kPi / 180.0;
constexpr double kMaxShoulderYawFlex = 160.0 * kPi / 180.0;
constexpr double kOverheadSingularity = 15.0 * kPi / 180.0;
constexpr double kShoulderEulerMinCondition = 0.1;

bool validPoint(const Vec3 &p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
         norm(p) > kEps;
}

bool validPoint(const std::optional<Vec3> &p) {
  return p.has_value() && validPoint(*p);
}

std::optional<double> angleBetween(const Vec3 &a, const Vec3 &b) {
  const auto an = normalize(a);
  const auto bn = normalize(b);
  if (!an || !bn) {
    return std::nullopt;
  }
  return std::acos(std::clamp(dot(*an, *bn), -1.0, 1.0));
}

struct TorsoFrame {
  Vec3 right;
  Vec3 forward;
  Vec3 up;
};

std::optional<TorsoFrame> buildTorsoFrame(const Vec3 &pelvis,
                                          const Vec3 &left_shoulder,
                                          const Vec3 &right_shoulder) {
  const Vec3 shoulder_center = 0.5 * (left_shoulder + right_shoulder);
  const auto torso_up = normalize(shoulder_center - pelvis);
  if (!torso_up) {
    return std::nullopt;
  }

  Vec3 shoulder_right = right_shoulder - left_shoulder;
  shoulder_right =
      shoulder_right - (*torso_up * dot(shoulder_right, *torso_up));
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

Vec3 transposeMultiply(const TorsoFrame &frame, const Vec3 &v) {
  return {dot(frame.right, v), dot(frame.forward, v), dot(frame.up, v)};
}

std::optional<double> shoulderYaw(const TorsoFrame &frame,
                                  const Vec3 &upper_arm, const Vec3 &forearm) {
  const auto upper = normalize(upper_arm);
  const auto lower = normalize(forearm);
  const auto flex = angleBetween(upper_arm, forearm);
  if (!upper || !lower || !flex || *flex < kMinShoulderYawFlex ||
      *flex > kMaxShoulderYawFlex) {
    return std::nullopt;
  }

  const Vec3 arm_down = -1.0 * frame.up;
  if (dot(arm_down, *upper) < -std::cos(kOverheadSingularity)) {
    return std::nullopt;
  }

  // G1 applies shoulder pitch before roll, then yaw about the upper arm.
  // Transport torso-forward with that same pitch/roll gauge so combined
  // pitch and roll do not appear as a false shoulder-yaw rotation.
  const Vec3 upper_t = transposeMultiply(frame, *upper);
  const double pitch_plane_norm = std::hypot(upper_t.y, upper_t.z);
  if (pitch_plane_norm < kShoulderEulerMinCondition) {
    return std::nullopt;
  }
  const Vec3 zero_bend = (frame.forward * -upper_t.z + frame.up * upper_t.y) *
                         (1.0 / pitch_plane_norm);
  const auto zero_bend_n =
      normalize(zero_bend - *upper * dot(zero_bend, *upper));
  const auto actual_bend_n = normalize(*lower - *upper * dot(*lower, *upper));
  if (!zero_bend_n || !actual_bend_n) {
    return std::nullopt;
  }

  const Vec3 yaw_axis = -1.0 * *upper;
  return std::atan2(dot(yaw_axis, cross(*zero_bend_n, *actual_bend_n)),
                    dot(*zero_bend_n, *actual_bend_n));
}

std::array<double, kNumJoints> vectorToArray(const std::vector<double> &values,
                                             const std::string &name) {
  if (values.size() != kNumJoints) {
    throw std::runtime_error("Parameter '" + name + "' must contain " +
                             std::to_string(kNumJoints) + " values");
  }

  std::array<double, kNumJoints> out{};
  std::copy(values.begin(), values.end(), out.begin());
  return out;
}

} // namespace

bool isCircularJoint(std::size_t index) {
  return index == jointIndex(JointIndex::WaistYaw) ||
         index == jointIndex(JointIndex::LeftShoulderYaw) ||
         index == jointIndex(JointIndex::RightShoulderYaw);
}

double wrapAngle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

SkeletonBodyFormat decodeBodyFormat(int body_format) {
  switch (body_format) {
  case 0:
  case 18:
    return SkeletonBodyFormat::Body18;
  case 1:
  case 34:
    return SkeletonBodyFormat::Body34;
  case 2:
  case 38:
    return SkeletonBodyFormat::Body38;
  default:
    return SkeletonBodyFormat::Unsupported;
  }
}

std::size_t bodyFormatKeypointCount(SkeletonBodyFormat body_format) {
  switch (body_format) {
  case SkeletonBodyFormat::Body18:
    return 18;
  case SkeletonBodyFormat::Body34:
    return 34;
  case SkeletonBodyFormat::Body38:
    return 38;
  case SkeletonBodyFormat::Unsupported:
    return 0;
  }
  return 0;
}

bool bodyFormatSupportsCapsules(int body_format) {
  return decodeBodyFormat(body_format) != SkeletonBodyFormat::Unsupported;
}

bool bodyFormatSupportsJointAngles(int body_format) {
  return decodeBodyFormat(body_format) == SkeletonBodyFormat::Body38;
}

BodyPoints canonicalBodyPoints(const Object &obj) {
  BodyPoints points{};

  auto copy = [&](int destination, int source) {
    if (source < 0 ||
        static_cast<std::size_t>(source) >=
            obj.skeleton_3d.keypoints.size()) {
      return;
    }
    const auto &kp = obj.skeleton_3d.keypoints[source].kp;
    const Vec3 point{static_cast<double>(kp[0]), static_cast<double>(kp[1]),
                     static_cast<double>(kp[2])};
    if (validPoint(point)) {
      points[destination] = point;
    }
  };

  switch (decodeBodyFormat(obj.body_format)) {
  case SkeletonBodyFormat::Body18:
    // BODY_18 has no pelvis joint. All canonical segments remain the same;
    // the midpoint of the two tracked hips is the anatomical pelvis proxy.
    copy(kNeck, 1);
    copy(kRightShoulder, 2);
    copy(kRightElbow, 3);
    copy(kRightWrist, 4);
    copy(kLeftShoulder, 5);
    copy(kLeftElbow, 6);
    copy(kLeftWrist, 7);
    copy(kRightHip, 8);
    copy(kRightKnee, 9);
    copy(kRightAnkle, 10);
    copy(kLeftHip, 11);
    copy(kLeftKnee, 12);
    copy(kLeftAnkle, 13);
    if (validPoint(points[kLeftHip]) && validPoint(points[kRightHip])) {
      points[kPelvis] =
          0.5 * (*points[kLeftHip] + *points[kRightHip]);
    }
    break;

  case SkeletonBodyFormat::Body34:
    copy(kPelvis, 0);
    copy(kNeck, 3);
    copy(kLeftShoulder, 5);
    copy(kLeftElbow, 6);
    copy(kLeftWrist, 7);
    copy(kRightShoulder, 12);
    copy(kRightElbow, 13);
    copy(kRightWrist, 14);
    copy(kLeftHip, 18);
    copy(kLeftKnee, 19);
    copy(kLeftAnkle, 20);
    copy(kRightHip, 22);
    copy(kRightKnee, 23);
    copy(kRightAnkle, 24);
    break;

  case SkeletonBodyFormat::Body38:
    for (std::size_t index = 0; index < kNumBody38Points; ++index) {
      copy(static_cast<int>(index), static_cast<int>(index));
    }
    break;

  case SkeletonBodyFormat::Unsupported:
    break;
  }

  return points;
}

JointAngles estimateJointAngles(const BodyPoints &points) {
  JointAngles result;
  auto set = [&](std::size_t index, double value) {
    if (std::isfinite(value)) {
      result.values[index] = value;
      result.valid[index] = true;
    }
  };

  const Vec3 world_up{0.0, 0.0, 1.0};
  if (validPoint(points[kLeftHip]) && validPoint(points[kRightHip]) &&
      validPoint(points[kLeftShoulder]) && validPoint(points[kRightShoulder])) {
    Vec3 hip_right = *points[kRightHip] - *points[kLeftHip];
    Vec3 shoulder_right = *points[kRightShoulder] - *points[kLeftShoulder];
    hip_right.z = 0.0;
    shoulder_right.z = 0.0;
    const auto hip_right_n = normalize(hip_right);
    const auto shoulder_right_n = normalize(shoulder_right);
    if (hip_right_n && shoulder_right_n) {
      set(jointIndex(JointIndex::WaistYaw),
          std::atan2(dot(world_up, cross(*hip_right_n, *shoulder_right_n)),
                     dot(*hip_right_n, *shoulder_right_n)));
    }
  }

  if (!validPoint(points[kPelvis]) || !validPoint(points[kLeftShoulder]) ||
      !validPoint(points[kRightShoulder])) {
    return result;
  }

  const Vec3 pelvis = *points[kPelvis];
  const Vec3 left_shoulder = *points[kLeftShoulder];
  const Vec3 right_shoulder = *points[kRightShoulder];
  const auto torso_frame =
      buildTorsoFrame(pelvis, left_shoulder, right_shoulder);
  if (!torso_frame) {
    return result;
  }

  double torso_roll =
      std::atan2(-dot(torso_frame->forward, cross(world_up, torso_frame->up)),
                 dot(world_up, torso_frame->up));
  if (torso_roll < -kPi / 2.0) {
    torso_roll += kPi;
  } else if (torso_roll > kPi / 2.0) {
    torso_roll -= kPi;
  }
  set(jointIndex(JointIndex::WaistRoll), torso_roll);

  double torso_pitch =
      std::atan2(-dot(torso_frame->right, cross(world_up, torso_frame->up)),
                 dot(world_up, torso_frame->up));
  double best_pitch = torso_pitch;
  for (const double candidate : {torso_pitch + kPi, torso_pitch - kPi}) {
    if (std::abs(candidate) < std::abs(best_pitch)) {
      best_pitch = candidate;
    }
  }
  set(jointIndex(JointIndex::WaistPitch), best_pitch);

  auto estimate_arm = [&](int shoulder_idx, int elbow_idx, int wrist_idx,
                          bool is_left) {
    if (!validPoint(points[elbow_idx])) {
      return;
    }

    const Vec3 shoulder = *points[shoulder_idx];
    const Vec3 upper = *points[elbow_idx] - shoulder;
    if (!normalize(upper)) {
      return;
    }

    const Vec3 upper_t = transposeMultiply(*torso_frame, upper);
    const std::size_t pitch_idx =
        jointIndex(is_left ? JointIndex::LeftShoulderPitch
                           : JointIndex::RightShoulderPitch);
    const std::size_t roll_idx = jointIndex(
        is_left ? JointIndex::LeftShoulderRoll : JointIndex::RightShoulderRoll);
    set(pitch_idx, std::atan2(upper_t.y, -upper_t.z));
    const double roll_denominator = std::hypot(upper_t.y, upper_t.z);
    set(roll_idx, is_left ? std::atan2(-upper_t.x, roll_denominator)
                          : std::atan2(upper_t.x, roll_denominator));

    if (!validPoint(points[wrist_idx])) {
      return;
    }

    const Vec3 forearm = *points[wrist_idx] - *points[elbow_idx];
    const auto flex = angleBetween(upper, forearm);
    if (flex) {
      set(jointIndex(is_left ? JointIndex::LeftElbow : JointIndex::RightElbow),
          *flex);
    }
    const auto yaw = shoulderYaw(*torso_frame, upper, forearm);
    if (yaw) {
      set(jointIndex(is_left ? JointIndex::LeftShoulderYaw
                             : JointIndex::RightShoulderYaw),
          *yaw);
    }
  };

  estimate_arm(kLeftShoulder, kLeftElbow, kLeftWrist, true);
  estimate_arm(kRightShoulder, kRightElbow, kRightWrist, false);
  return result;
}

std::vector<CapsuleData> buildBodyCapsules(const BodyPoints &points,
                                           const BodyCapsuleConfig &config) {
  std::vector<CapsuleData> capsules;

  auto add = [&](const std::string &name, int a_idx, int b_idx, double radius) {
    const auto &a = points[a_idx];
    const auto &b = points[b_idx];
    if (validPoint(a) && validPoint(b)) {
      capsules.push_back({name, *a, *b, radius * config.radius_scale});
    }
  };

  add("torso", kPelvis, kNeck, config.torso_radius);
  add("left_arm", kLeftElbow, kLeftWrist, config.arm_radius);
  add("right_arm", kRightElbow, kRightWrist, config.arm_radius);
  add("left_shoulder", kLeftShoulder, kLeftElbow, config.shoulder_radius);
  add("right_shoulder", kRightShoulder, kRightElbow, config.shoulder_radius);
  add("left_thigh", kLeftHip, kLeftKnee, config.thigh_radius);
  add("right_thigh", kRightHip, kRightKnee, config.thigh_radius);
  add("left_shin", kLeftKnee, kLeftAnkle, config.shin_radius);
  add("right_shin", kRightKnee, kRightAnkle, config.shin_radius);

  return capsules;
}

CapsuleAnatomyFilter::CapsuleAnatomyFilter(double max_length_change_fraction)
    : max_length_change_fraction_(max_length_change_fraction) {
  if (!std::isfinite(max_length_change_fraction_) ||
      max_length_change_fraction_ < 0.0 || max_length_change_fraction_ > 1.0) {
    throw std::invalid_argument(
        "Capsule max length change fraction must be in [0, 1]");
  }
}

std::vector<CapsuleData>
CapsuleAnatomyFilter::update(const std::vector<CapsuleData> &candidates) {
  std::vector<CapsuleData> filtered;
  filtered.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    const auto previous = previous_.find(candidate.name);
    if (plausible(candidate)) {
      previous_[candidate.name] = candidate;
      filtered.push_back(candidate);
    } else if (previous != previous_.end()) {
      filtered.push_back(previous->second);
    }
  }
  return filtered;
}

void CapsuleAnatomyFilter::reset() { previous_.clear(); }

bool CapsuleAnatomyFilter::plausible(const CapsuleData &candidate) const {
  const double length = norm(candidate.a - candidate.b);
  if (!std::isfinite(length)) {
    return false;
  }

  double minimum_length = 0.05;
  double maximum_length = 1.0;
  if (candidate.name == "torso") {
    minimum_length = 0.20;
  } else if (candidate.name == "left_arm" || candidate.name == "right_arm" ||
             candidate.name == "left_shoulder" ||
             candidate.name == "right_shoulder") {
    minimum_length = 0.08;
    maximum_length = 0.65;
  } else if (candidate.name == "left_thigh" ||
             candidate.name == "right_thigh" || candidate.name == "left_shin" ||
             candidate.name == "right_shin") {
    minimum_length = 0.12;
    maximum_length = 0.90;
  }
  if (length < minimum_length || length > maximum_length) {
    return false;
  }

  const auto previous = previous_.find(candidate.name);
  if (previous == previous_.end()) {
    return true;
  }
  const double previous_length = norm(previous->second.a - previous->second.b);
  const double minimum_change =
      previous_length * (1.0 - max_length_change_fraction_);
  const double maximum_change =
      previous_length * (1.0 + max_length_change_fraction_);
  return length >= minimum_change && length <= maximum_change;
}

bool hasUsableObservation(const JointAngles &angles,
                          const std::vector<CapsuleData> &capsules) {
  if (!capsules.empty()) {
    return true;
  }
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    if (angles.valid[i] && std::isfinite(angles.values[i])) {
      return true;
    }
  }
  return false;
}

EMAJumpFilter::EMAJumpFilter(double alpha, double max_jump,
                             int max_reject_count)
    : alpha_(alpha), max_jump_(max_jump), max_reject_count_(max_reject_count) {}

Vec3 EMAJumpFilter::update(const Vec3 &x) {
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

void EMAJumpFilter::reset() {
  value_.reset();
  reject_count_ = 0;
  last_rejected_.reset();
}

bool shouldResetPointFilters(
    const std::optional<int> &previous_body_id,
    const std::optional<int64_t> &previous_observation_ns, int current_body_id,
    int64_t current_observation_ns, double reset_gap_sec) {
  if (previous_body_id && *previous_body_id != current_body_id) {
    return true;
  }
  if (!previous_observation_ns) {
    return false;
  }
  if (current_observation_ns < *previous_observation_ns) {
    return true;
  }
  return static_cast<double>(current_observation_ns -
                             *previous_observation_ns) *
             1e-9 >
         reset_gap_sec;
}

AngleFilter::AngleFilter(double alpha, double max_rate_deg)
    : alpha_(alpha), max_rate_(max_rate_deg * kPi / 180.0) {
  ema_.fill(0.0);
  prev_.fill(0.0);
  has_ema_.fill(false);
  has_prev_.fill(false);
}

JointAngles AngleFilter::update(const JointAngles &angles, double dt) {
  dt = std::clamp(dt, 1e-3, 0.2);

  JointAngles out;
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    if (!angles.valid[i] || !std::isfinite(angles.values[i])) {
      out.values[i] = has_prev_[i] ? prev_[i] : 0.0;
      continue;
    }

    double y = angles.values[i];
    if (has_ema_[i]) {
      if (isCircularJoint(i)) {
        y = wrapAngle(ema_[i] + alpha_ * wrapAngle(angles.values[i] - ema_[i]));
      } else {
        y = alpha_ * angles.values[i] + (1.0 - alpha_) * ema_[i];
      }
    }
    ema_[i] = y;
    has_ema_[i] = true;

    if (has_prev_[i]) {
      const double max_step = max_rate_ * dt;
      const double difference =
          isCircularJoint(i) ? wrapAngle(y - prev_[i]) : y - prev_[i];
      y = prev_[i] + std::clamp(difference, -max_step, max_step);
    }

    y = isCircularJoint(i) ? wrapAngle(y)
                           : std::clamp(y, limits_[i].first, limits_[i].second);
    prev_[i] = y;
    has_prev_[i] = true;
    out.values[i] = y;
    out.valid[i] = true;
  }
  return out;
}

HumanMappingNode::HumanMappingNode() : Node("human_mapping_node") {
  input_skeleton_topic_ = declare_parameter<std::string>(
      "input_skeleton_topic", "/zed_fusion/body_trk/skeletons");
  joint_command_topic_ = declare_parameter<std::string>(
      "joint_command_topic", "/human/joint_commands");
  collider_topic_ =
      declare_parameter<std::string>("collider_topic", "/human/body_colliders");
  fallback_frame_id_ =
      declare_parameter<std::string>("fallback_frame_id", "fusion_world");
  publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 30.0);
  stale_timeout_sec_ = declare_parameter<double>("stale_timeout_sec", 0.5);
  min_confidence_ = declare_parameter<double>("min_confidence", 70.0);
  require_body_38_ = declare_parameter<bool>("require_body_38", false);

  const double point_alpha = declare_parameter<double>("point_ema_alpha", 0.30);
  const double point_max_jump =
      declare_parameter<double>("point_max_jump", 0.6);
  const int point_max_reject_count =
      declare_parameter<int>("point_max_reject_count", 3);
  point_filter_reset_gap_sec_ =
      declare_parameter<double>("point_filter_reset_gap_sec", 0.5);
  const double capsule_max_length_change_fraction =
      declare_parameter<double>("capsule_max_length_change_fraction", 0.5);
  const double angle_alpha = declare_parameter<double>("angle_ema_alpha", 0.25);
  const double angle_max_rate_deg =
      declare_parameter<double>("angle_max_rate_deg", 100.0);

  enable_neutral_calibration_ =
      declare_parameter<bool>("enable_neutral_calibration", true);
  startup_delay_sec_ = declare_parameter<double>("startup_delay_sec", 5.0);
  neutral_calibration_duration_ =
      declare_parameter<double>("neutral_calibration_duration", 10.0);

  joint_names_ = declare_parameter<std::vector<std::string>>("joint_names",
                                                             kControlledJoints);
  q_home_ = vectorToArray(declare_parameter<std::vector<double>>(
                              "q_home", {0.0, 0.0, 0.0, 0.35, 0.18, 0.0, 0.87,
                                         0.35, -0.18, 0.0, 0.87}),
                          "q_home");
  signs_ = vectorToArray(declare_parameter<std::vector<double>>(
                             "signs", {1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
                                       -1.0, -1.0, 1.0, -1.0}),
                         "signs");
  gains_ = vectorToArray(
      declare_parameter<std::vector<double>>(
          "gains", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
      "gains");
  bias_ = vectorToArray(
      declare_parameter<std::vector<double>>(
          "bias", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      "bias");
  q_min_ = vectorToArray(
      declare_parameter<std::vector<double>>(
          "q_min", {-2.094, -0.416, -0.416, -2.471, -1.271, -2.094, -0.838,
                    -2.471, -1.801, -2.094, -0.838}),
      "q_min");
  q_max_ = vectorToArray(declare_parameter<std::vector<double>>(
                             "q_max", {2.094, 0.416, 0.416, 2.136, 1.801, 2.094,
                                       1.676, 2.136, 1.271, 2.094, 1.676}),
                         "q_max");

  human_radius_scale_ = declare_parameter<double>("human_radius_scale", 1.5);
  torso_radius_ = declare_parameter<double>("torso_radius", 0.10);
  shoulder_radius_ = declare_parameter<double>("shoulder_radius", 0.05);
  arm_radius_ = declare_parameter<double>("arm_radius", 0.05);
  thigh_radius_ = declare_parameter<double>("thigh_radius", 0.065);
  shin_radius_ = declare_parameter<double>("shin_radius", 0.065);

  if (joint_names_.size() != kNumJoints) {
    throw std::runtime_error("Parameter 'joint_names' must contain " +
                             std::to_string(kNumJoints) + " values");
  }
  if (publish_rate_hz_ <= 0.0) {
    throw std::runtime_error("Parameter 'publish_rate_hz' must be positive");
  }
  if (!std::isfinite(point_filter_reset_gap_sec_) ||
      point_filter_reset_gap_sec_ <= 0.0) {
    throw std::runtime_error(
        "Parameter 'point_filter_reset_gap_sec' must be finite and positive");
  }

  for (std::size_t i = 0; i < kNumBody38Points; ++i) {
    point_filters_.emplace_back(point_alpha, point_max_jump,
                                point_max_reject_count);
  }
  capsule_filter_ = std::make_unique<CapsuleAnatomyFilter>(
      capsule_max_length_change_fraction);
  angle_filter_ =
      std::make_unique<AngleFilter>(angle_alpha, angle_max_rate_deg);

  node_start_ns_ = now().nanoseconds();
  calibration_done_ = !enable_neutral_calibration_;
  neutral_offset_.fill(0.0);
  neutral_initialized_.fill(!enable_neutral_calibration_);
  last_delta_.fill(0.0);
  has_last_delta_.fill(false);
  calibration_linear_sum_.fill(0.0);
  calibration_sin_sum_.fill(0.0);
  calibration_cos_sum_.fill(0.0);
  calibration_sample_count_.fill(0);

  rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
  sensor_qos.best_effort();
  sensor_qos.durability_volatile();

  joint_pub_ = create_publisher<JointState>(joint_command_topic_, sensor_qos);
  collider_pub_ = create_publisher<CapsuleArray>(collider_topic_, sensor_qos);
  skeleton_sub_ = create_subscription<ObjectsStamped>(
      input_skeleton_topic_, sensor_qos,
      std::bind(&HumanMappingNode::skeletonCallback, this,
                std::placeholders::_1));

  const auto period = std::chrono::nanoseconds(
      static_cast<int64_t>(1e9 / std::max(publish_rate_hz_, 1.0)));
  timer_ = create_wall_timer(period,
                             std::bind(&HumanMappingNode::timerCallback, this));

  RCLCPP_INFO(get_logger(), "human_mapping_node ready: %s -> %s, %s",
              input_skeleton_topic_.c_str(), joint_command_topic_.c_str(),
              collider_topic_.c_str());
}

void HumanMappingNode::skeletonCallback(const ObjectsStamped::SharedPtr msg) {
  const Object *obj = selectBody(*msg);
  if (obj == nullptr) {
    return;
  }

  const int64_t now_ns = now().nanoseconds();
  preparePointFilters(obj->label_id, now_ns);
  const auto points = filteredPoints(*obj);
  JointAngles raw_angles;
  const bool estimate_angles = bodyFormatSupportsJointAngles(obj->body_format);
  if (estimate_angles) {
    raw_angles = estimateJointAngles(points);
  }
  auto capsules = capsule_filter_->update(buildCapsules(points));
  if (!hasUsableObservation(raw_angles, capsules)) {
    return;
  }

  std::array<double, kNumJoints> delta{};
  if (estimate_angles) {
    const double angle_dt = angleDt(now_ns);
    const auto filtered_angles = angle_filter_->update(raw_angles, angle_dt);
    delta = neutralDelta(filtered_angles, now_ns);
  }

  std::array<double, kNumJoints> q_des{};
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    q_des[i] =
        std::clamp(q_home_[i] + signs_[i] * gains_[i] * delta[i] + bias_[i],
                   q_min_[i], q_max_[i]);
  }

  latest_result_ = MappingResult{
      msg->header.stamp,
      msg->header.frame_id.empty() ? fallback_frame_id_ : msg->header.frame_id,
      q_des, std::move(capsules)};
  latest_result_time_ns_ = now_ns;
}

const Object *HumanMappingNode::selectBody(const ObjectsStamped &msg) {
  const Object *best = nullptr;
  double best_confidence = -1.0;
  for (const auto &obj : msg.objects) {
    if (!obj.skeleton_available) {
      continue;
    }
    const auto body_format = decodeBodyFormat(obj.body_format);
    if (body_format == SkeletonBodyFormat::Unsupported ||
        (require_body_38_ && body_format != SkeletonBodyFormat::Body38)) {
      warnRejectedBodyFormat(obj.body_format);
      continue;
    }
    if (static_cast<double>(obj.confidence) < min_confidence_) {
      continue;
    }
    if (obj.skeleton_3d.keypoints.size() <
        bodyFormatKeypointCount(body_format)) {
      continue;
    }
    if (static_cast<double>(obj.confidence) > best_confidence) {
      best = &obj;
      best_confidence = static_cast<double>(obj.confidence);
    }
  }
  return best;
}

void HumanMappingNode::warnRejectedBodyFormat(int body_format) {
  const int64_t now_ns = now().nanoseconds();
  if (now_ns - last_body_format_warn_ns_ < 2'000'000'000LL) {
    return;
  }
  last_body_format_warn_ns_ = now_ns;
  if (bodyFormatSupportsCapsules(body_format)) {
    RCLCPP_WARN(
        get_logger(),
        "Ignoring skeleton with body_format=%d because require_body_38=true.",
        body_format);
  } else {
    RCLCPP_WARN(get_logger(),
                "Ignoring skeleton with unsupported body_format=%d.",
                body_format);
  }
}

BodyPoints HumanMappingNode::filteredPoints(const Object &obj) {
  const BodyPoints raw_points = canonicalBodyPoints(obj);
  BodyPoints points{};
  for (std::size_t idx = 0; idx < kNumBody38Points; ++idx) {
    if (validPoint(raw_points[idx])) {
      points[idx] = point_filters_[idx].update(*raw_points[idx]);
    }
  }
  return points;
}

void HumanMappingNode::preparePointFilters(int body_id,
                                           int64_t observation_ns) {
  if (shouldResetPointFilters(filtered_body_id_, last_point_observation_ns_,
                              body_id, observation_ns,
                              point_filter_reset_gap_sec_)) {
    for (auto &filter : point_filters_) {
      filter.reset();
    }
    capsule_filter_->reset();
  }
  filtered_body_id_ = body_id;
  last_point_observation_ns_ = observation_ns;
}

double HumanMappingNode::angleDt(int64_t now_ns) {
  double dt = 1.0 / std::max(publish_rate_hz_, 1.0);
  if (last_angle_time_ns_) {
    dt = static_cast<double>(now_ns - *last_angle_time_ns_) * 1e-9;
  }
  last_angle_time_ns_ = now_ns;
  return std::clamp(dt, 1e-3, 0.2);
}

std::array<double, kNumJoints>
HumanMappingNode::neutralDelta(const JointAngles &angles, int64_t now_ns) {
  std::array<double, kNumJoints> out{};

  auto holdOrUpdate = [&](std::size_t index, double value, bool valid) {
    if (valid) {
      last_delta_[index] = value;
      has_last_delta_[index] = true;
    }
    out[index] = has_last_delta_[index] ? last_delta_[index] : 0.0;
  };

  if (!enable_neutral_calibration_) {
    for (std::size_t i = 0; i < kNumJoints; ++i) {
      holdOrUpdate(i, angles.values[i], angles.valid[i]);
    }
    return out;
  }

  const double startup_elapsed =
      static_cast<double>(now_ns - node_start_ns_) * 1e-9;
  if (startup_elapsed < startup_delay_sec_) {
    return out;
  }

  if (!calibration_done_) {
    if (!calibration_start_ns_) {
      calibration_start_ns_ = now_ns;
      calibration_linear_sum_.fill(0.0);
      calibration_sin_sum_.fill(0.0);
      calibration_cos_sum_.fill(0.0);
      calibration_sample_count_.fill(0);
      RCLCPP_INFO(get_logger(), "Neutral calibration started for %.2fs.",
                  neutral_calibration_duration_);
    }

    for (std::size_t i = 0; i < kNumJoints; ++i) {
      if (!angles.valid[i]) {
        continue;
      }
      if (isCircularJoint(i)) {
        calibration_sin_sum_[i] += std::sin(angles.values[i]);
        calibration_cos_sum_[i] += std::cos(angles.values[i]);
      } else {
        calibration_linear_sum_[i] += angles.values[i];
      }
      ++calibration_sample_count_[i];
    }

    const double elapsed =
        static_cast<double>(now_ns - *calibration_start_ns_) * 1e-9;
    if (elapsed < neutral_calibration_duration_) {
      return out;
    }

    for (std::size_t i = 0; i < kNumJoints; ++i) {
      if (calibration_sample_count_[i] == 0) {
        continue;
      }
      if (isCircularJoint(i)) {
        neutral_offset_[i] =
            std::atan2(calibration_sin_sum_[i], calibration_cos_sum_[i]);
      } else {
        neutral_offset_[i] = calibration_linear_sum_[i] /
                             static_cast<double>(calibration_sample_count_[i]);
      }
      neutral_initialized_[i] = true;
    }
    calibration_done_ = true;
    RCLCPP_INFO(get_logger(), "Neutral calibration completed.");
  }

  for (std::size_t i = 0; i < kNumJoints; ++i) {
    if (!angles.valid[i]) {
      holdOrUpdate(i, 0.0, false);
      continue;
    }

    if (!neutral_initialized_[i]) {
      neutral_offset_[i] = angles.values[i];
      neutral_initialized_[i] = true;
      holdOrUpdate(i, 0.0, true);
      RCLCPP_INFO(get_logger(),
                  "Initialized neutral for previously unobserved joint '%s'.",
                  joint_names_[i].c_str());
      continue;
    }

    double delta = angles.values[i] - neutral_offset_[i];
    if (isCircularJoint(i)) {
      delta = wrapAngle(delta);
    }
    holdOrUpdate(i, delta, true);
  }
  return out;
}

std::vector<CapsuleData>
HumanMappingNode::buildCapsules(const BodyPoints &points) {
  return buildBodyCapsules(points,
                           BodyCapsuleConfig{human_radius_scale_, torso_radius_,
                                             shoulder_radius_, arm_radius_,
                                             thigh_radius_, shin_radius_});
}

void HumanMappingNode::timerCallback() {
  const auto now_time = now();
  if (!latest_result_ || isStale(now_time.nanoseconds())) {
    publishEmptyAndNeutral(now_time);
    return;
  }

  publishJointCommand(latest_result_->stamp, latest_result_->q_des);
  publishColliders(latest_result_->stamp, latest_result_->frame_id,
                   latest_result_->capsules);
}

bool HumanMappingNode::isStale(int64_t now_ns) const {
  if (!latest_result_time_ns_) {
    return true;
  }
  return static_cast<double>(now_ns - *latest_result_time_ns_) * 1e-9 >
         stale_timeout_sec_;
}

void HumanMappingNode::publishEmptyAndNeutral(const rclcpp::Time &stamp) {
  publishJointCommand(stamp, q_home_);
  publishColliders(stamp, fallback_frame_id_, {});
}

void HumanMappingNode::publishJointCommand(
    const Time &stamp, const std::array<double, kNumJoints> &q_des) {
  JointState msg;
  msg.header.stamp = stamp;
  msg.name = joint_names_;
  msg.position.assign(q_des.begin(), q_des.end());
  msg.velocity.assign(joint_names_.size(), 0.0);
  joint_pub_->publish(msg);
}

void HumanMappingNode::publishJointCommand(
    const rclcpp::Time &stamp, const std::array<double, kNumJoints> &q_des) {
  publishJointCommand(toMsg(stamp), q_des);
}

void HumanMappingNode::publishColliders(
    const Time &stamp, const std::string &frame_id,
    const std::vector<CapsuleData> &capsules) {
  CapsuleArray msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id.empty() ? fallback_frame_id_ : frame_id;
  msg.capsules.reserve(capsules.size());
  for (const auto &src : capsules) {
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

void HumanMappingNode::publishColliders(
    const rclcpp::Time &stamp, const std::string &frame_id,
    const std::vector<CapsuleData> &capsules) {
  publishColliders(toMsg(stamp), frame_id, capsules);
}

Time HumanMappingNode::toMsg(const rclcpp::Time &stamp) const {
  Time msg;
  const int64_t ns = stamp.nanoseconds();
  msg.sec = static_cast<int32_t>(ns / 1'000'000'000LL);
  msg.nanosec = static_cast<uint32_t>(ns % 1'000'000'000LL);
  return msg;
}

} // namespace human_mapping

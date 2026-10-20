#pragma once

#include "human_mapping/geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <g1_cbf_msg/msg/capsule.hpp>
#include <g1_cbf_msg/msg/capsule_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <zed_msgs/msg/object.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>

namespace human_mapping {

using Time = builtin_interfaces::msg::Time;
using Capsule = g1_cbf_msg::msg::Capsule;
using CapsuleArray = g1_cbf_msg::msg::CapsuleArray;
using JointState = sensor_msgs::msg::JointState;
using Object = zed_msgs::msg::Object;
using ObjectsStamped = zed_msgs::msg::ObjectsStamped;

inline constexpr std::size_t kNumJoints = 11;
inline constexpr std::size_t kNumBody38Points = 38;
inline constexpr double kPi = 3.14159265358979323846;

inline constexpr int kPelvis = 0;
inline constexpr int kNeck = 4;
inline constexpr int kLeftShoulder = 12;
inline constexpr int kRightShoulder = 13;
inline constexpr int kLeftElbow = 14;
inline constexpr int kRightElbow = 15;
inline constexpr int kLeftWrist = 16;
inline constexpr int kRightWrist = 17;
inline constexpr int kLeftHip = 18;
inline constexpr int kRightHip = 19;
inline constexpr int kLeftKnee = 20;
inline constexpr int kRightKnee = 21;
inline constexpr int kLeftAnkle = 22;
inline constexpr int kRightAnkle = 23;

enum class JointIndex : std::size_t {
  WaistYaw = 0,
  WaistRoll,
  WaistPitch,
  LeftShoulderPitch,
  LeftShoulderRoll,
  LeftShoulderYaw,
  LeftElbow,
  RightShoulderPitch,
  RightShoulderRoll,
  RightShoulderYaw,
  RightElbow,
};

inline constexpr std::size_t jointIndex(JointIndex index) {
  return static_cast<std::size_t>(index);
}

using BodyPoints = std::array<std::optional<Vec3>, kNumBody38Points>;

enum class SkeletonBodyFormat {
  Body18,
  Body34,
  Body38,
  Unsupported,
};

SkeletonBodyFormat decodeBodyFormat(int body_format);

std::size_t bodyFormatKeypointCount(SkeletonBodyFormat body_format);

bool bodyFormatSupportsCapsules(int body_format);

bool bodyFormatSupportsJointAngles(int body_format);

BodyPoints canonicalBodyPoints(const Object &obj);

enum class BodyTrackingQuality {
  Measured,
  Predicted,
  Invalid,
};

BodyTrackingQuality bodyTrackingQuality(const Object &obj);

struct JointAngles {
  std::array<double, kNumJoints> values{};
  std::array<bool, kNumJoints> valid{};
};

bool isCircularJoint(std::size_t index);

double wrapAngle(double angle);

JointAngles estimateJointAngles(const BodyPoints &points);

class TimeAwarePointFilter {
public:
  TimeAwarePointFilter(double time_constant_sec, double max_speed_mps,
                       double max_step_m);

  Vec3 update(const Vec3 &measurement, double dt_sec);

  void reset();

  const std::optional<Vec3> &value() const;

private:
  double time_constant_sec_;
  double max_speed_mps_;
  double max_step_m_;
  std::optional<Vec3> value_;
};

class RootRelativeBodyFilter {
public:
  RootRelativeBodyFilter(double root_position_time_constant_sec,
                         double root_velocity_time_constant_sec,
                         double relative_position_time_constant_sec,
                         double root_max_speed_mps,
                         double relative_max_speed_mps, double root_max_step_m,
                         double relative_max_step_m,
                         double velocity_decay_time_constant_sec,
                         double missing_point_hold_sec,
                         double prediction_timeout_sec);

  BodyPoints update(const BodyPoints &raw_points,
                    BodyTrackingQuality tracking_quality,
                    int64_t observation_ns);

  void reset();

private:
  Vec3 updateMeasuredRoot(const Vec3 &measurement, double dt_sec,
                          int64_t observation_ns);

  Vec3 predictRoot(double dt_sec, int64_t observation_ns);

  bool predictionActive(int64_t observation_ns) const;

  double root_position_time_constant_sec_;
  double root_velocity_time_constant_sec_;
  double root_max_speed_mps_;
  double root_max_step_m_;
  double velocity_decay_time_constant_sec_;
  double missing_point_hold_sec_;
  double prediction_timeout_sec_;
  std::optional<Vec3> root_;
  Vec3 root_velocity_{};
  std::optional<int64_t> last_update_ns_;
  std::optional<int64_t> last_measured_root_ns_;
  std::vector<TimeAwarePointFilter> relative_filters_;
  std::array<std::optional<int64_t>, kNumBody38Points>
      last_relative_observation_ns_{};
};

bool shouldResetPointFilters(
    const std::optional<int> &previous_body_id,
    const std::optional<int64_t> &previous_observation_ns, int current_body_id,
    int64_t current_observation_ns, double reset_gap_sec);

class AngleFilter {
public:
  AngleFilter(double alpha, double max_rate_deg);

  JointAngles update(const JointAngles &angles, double dt);

private:
  double alpha_;
  double max_rate_;
  std::array<double, kNumJoints> ema_;
  std::array<double, kNumJoints> prev_;
  std::array<bool, kNumJoints> has_ema_;
  std::array<bool, kNumJoints> has_prev_;
  const std::array<std::pair<double, double>, kNumJoints> limits_ = {
      std::pair<double, double>{-kPi, kPi},
      std::pair<double, double>{-60.0 * kPi / 180.0, 60.0 * kPi / 180.0},
      std::pair<double, double>{-60.0 * kPi / 180.0, 60.0 * kPi / 180.0},
      std::pair<double, double>{-180.0 * kPi / 180.0, 180.0 * kPi / 180.0},
      std::pair<double, double>{-90.0 * kPi / 180.0, 150.0 * kPi / 180.0},
      std::pair<double, double>{-kPi, kPi},
      std::pair<double, double>{0.0, 180.0 * kPi / 180.0},
      std::pair<double, double>{-180.0 * kPi / 180.0, 180.0 * kPi / 180.0},
      std::pair<double, double>{-90.0 * kPi / 180.0, 150.0 * kPi / 180.0},
      std::pair<double, double>{-kPi, kPi},
      std::pair<double, double>{0.0, 180.0 * kPi / 180.0}};
};

struct CapsuleData {
  std::string name;
  Vec3 a;
  Vec3 b;
  double radius = 0.0;
};

struct BodyCapsuleConfig {
  double radius_scale = 1.5;
  double torso_radius = 0.10;
  double shoulder_radius = 0.05;
  double arm_radius = 0.05;
  double thigh_radius = 0.065;
  double shin_radius = 0.065;
};

std::vector<CapsuleData> buildBodyCapsules(const BodyPoints &points,
                                           const BodyCapsuleConfig &config);

bool hasUsableObservation(const JointAngles &angles,
                          const std::vector<CapsuleData> &capsules);

class CapsuleAnatomyFilter {
public:
  explicit CapsuleAnatomyFilter(double max_length_change_fraction = 0.5,
                                double missing_hold_sec = 0.25);

  std::vector<CapsuleData>
  update(const std::vector<CapsuleData> &candidates,
         const std::optional<Vec3> &body_root = std::nullopt,
         std::optional<int64_t> observation_ns = std::nullopt);

  void reset();

private:
  bool plausible(const CapsuleData &candidate) const;

  double max_length_change_fraction_;
  double missing_hold_sec_;
  std::unordered_map<std::string, CapsuleData> previous_;
  std::unordered_map<std::string, int64_t> last_accepted_ns_;
  std::optional<Vec3> previous_body_root_;
};

struct MappingResult {
  Time stamp;
  std::string frame_id;
  std::array<double, kNumJoints> q_des{};
  std::vector<CapsuleData> capsules;
};

class HumanMappingNode : public rclcpp::Node {
public:
  HumanMappingNode();

private:
  void skeletonCallback(const ObjectsStamped::SharedPtr msg);

  const Object *selectBody(const ObjectsStamped &msg);

  void warnRejectedBodyFormat(int body_format);

  BodyPoints filteredPoints(const Object &obj, int64_t observation_ns);

  void preparePointFilters(int body_id, int64_t observation_ns);

  double angleDt(int64_t now_ns);

  std::array<double, kNumJoints> neutralDelta(const JointAngles &angles,
                                              int64_t now_ns);

  std::vector<CapsuleData> buildCapsules(const BodyPoints &points);

  void timerCallback();

  bool isStale(int64_t now_ns) const;

  void publishEmptyAndNeutral(const rclcpp::Time &stamp);

  void publishJointCommand(const Time &stamp,
                           const std::array<double, kNumJoints> &q_des);

  void publishJointCommand(const rclcpp::Time &stamp,
                           const std::array<double, kNumJoints> &q_des);

  void publishColliders(const Time &stamp, const std::string &frame_id,
                        const std::vector<CapsuleData> &capsules);

  void publishColliders(const rclcpp::Time &stamp, const std::string &frame_id,
                        const std::vector<CapsuleData> &capsules);

  Time toMsg(const rclcpp::Time &stamp) const;

  std::string input_skeleton_topic_;

  std::string joint_command_topic_;

  std::string collider_topic_;

  std::string fallback_frame_id_;

  double publish_rate_hz_ = 30.0;

  double stale_timeout_sec_ = 0.5;

  double min_confidence_ = 70.0;

  bool require_body_38_ = false;

  double point_filter_reset_gap_sec_ = 0.5;

  bool enable_neutral_calibration_ = true;

  double startup_delay_sec_ = 5.0;

  double neutral_calibration_duration_ = 10.0;

  int64_t node_start_ns_ = 0;

  std::optional<int64_t> calibration_start_ns_;

  std::array<double, kNumJoints> calibration_linear_sum_{};

  std::array<double, kNumJoints> calibration_sin_sum_{};

  std::array<double, kNumJoints> calibration_cos_sum_{};

  std::array<std::size_t, kNumJoints> calibration_sample_count_{};

  std::array<double, kNumJoints> neutral_offset_{};

  std::array<bool, kNumJoints> neutral_initialized_{};

  std::array<double, kNumJoints> last_delta_{};

  std::array<bool, kNumJoints> has_last_delta_{};

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

  std::unique_ptr<RootRelativeBodyFilter> point_filter_;

  std::unique_ptr<CapsuleAnatomyFilter> capsule_filter_;

  std::optional<int> filtered_body_id_;

  std::optional<int64_t> last_point_observation_ns_;

  std::unique_ptr<AngleFilter> angle_filter_;

  std::optional<MappingResult> latest_result_;

  std::optional<int64_t> latest_result_time_ns_;

  int64_t last_body_format_warn_ns_ = 0;

  rclcpp::Publisher<JointState>::SharedPtr joint_pub_;

  rclcpp::Publisher<CapsuleArray>::SharedPtr collider_pub_;

  rclcpp::Subscription<ObjectsStamped>::SharedPtr skeleton_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace human_mapping

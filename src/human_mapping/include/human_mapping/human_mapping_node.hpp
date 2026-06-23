#pragma once

#include "human_mapping/geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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

inline constexpr std::size_t kNumJoints = 8;
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

class EMAJumpFilter {
public:
  EMAJumpFilter(double alpha, double max_jump, int max_reject_count);

  Vec3 update(const Vec3 &x);

private:
  double alpha_;
  double max_jump_;
  int max_reject_count_;
  std::optional<Vec3> value_;
  int reject_count_ = 0;
  std::optional<Vec3> last_rejected_;
};

class AngleFilter {
public:
  AngleFilter(double alpha, double max_rate_deg);

  std::array<double, kNumJoints>
  update(const std::array<double, kNumJoints> &values, double dt);

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

struct CapsuleData {
  std::string name;
  Vec3 a;
  Vec3 b;
  double radius = 0.0;
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

  bool isBody38(int body_format) const;

  void warnBodyFormat(int body_format);

  std::array<std::optional<Vec3>, kNumBody38Points>
  filteredPoints(const Object &obj);

  std::optional<std::array<double, kNumJoints>> estimateAngles(
      const std::array<std::optional<Vec3>, kNumBody38Points> &points);

  double minAbs(double a, double b, double c) const;

  double angleDt(int64_t now_ns);

  std::array<double, kNumJoints>
  neutralDelta(const std::array<double, kNumJoints> &angles, int64_t now_ns);

  std::array<double, kNumJoints>
  subtract(const std::array<double, kNumJoints> &a,
           const std::array<double, kNumJoints> &b) const;

  std::vector<CapsuleData> buildCapsules(
      const std::array<std::optional<Vec3>, kNumBody38Points> &points);

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

} // namespace human_mapping

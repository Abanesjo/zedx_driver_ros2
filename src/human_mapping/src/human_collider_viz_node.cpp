#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "g1_cbf_msg/msg/capsule_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace
{

using g1_cbf_msg::msg::Capsule;
using g1_cbf_msg::msg::CapsuleArray;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

constexpr double kEps = 1e-8;

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

Vec3 normalizeOr(const Vec3 & v, const Vec3 & fallback)
{
  const double n = norm(v);
  if (!std::isfinite(n) || n < kEps) {
    return fallback;
  }
  return v * (1.0 / n);
}

Vec3 toVec3(const geometry_msgs::msg::Point & p)
{
  return {p.x, p.y, p.z};
}

geometry_msgs::msg::Point toPoint(const Vec3 & p)
{
  geometry_msgs::msg::Point out;
  out.x = p.x;
  out.y = p.y;
  out.z = p.z;
  return out;
}

geometry_msgs::msg::Quaternion quaternionFromRotationColumns(
  const Vec3 & x_axis, const Vec3 & y_axis, const Vec3 & z_axis)
{
  const double m00 = x_axis.x;
  const double m01 = y_axis.x;
  const double m02 = z_axis.x;
  const double m10 = x_axis.y;
  const double m11 = y_axis.y;
  const double m12 = z_axis.y;
  const double m20 = x_axis.z;
  const double m21 = y_axis.z;
  const double m22 = z_axis.z;

  geometry_msgs::msg::Quaternion q;
  const double trace = m00 + m11 + m22;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * s;
    q.x = (m21 - m12) / s;
    q.y = (m02 - m20) / s;
    q.z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
    q.w = (m21 - m12) / s;
    q.x = 0.25 * s;
    q.y = (m01 + m10) / s;
    q.z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
    q.w = (m02 - m20) / s;
    q.x = (m01 + m10) / s;
    q.y = 0.25 * s;
    q.z = (m12 + m21) / s;
  } else {
    const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
    q.w = (m10 - m01) / s;
    q.x = (m02 + m20) / s;
    q.y = (m12 + m21) / s;
    q.z = 0.25 * s;
  }
  return q;
}

geometry_msgs::msg::Quaternion axisToQuaternion(const Vec3 & a, const Vec3 & b)
{
  const Vec3 z_axis = normalizeOr(a - b, {0.0, 0.0, 1.0});
  Vec3 up{0.0, 0.0, 1.0};
  if (std::abs(dot(z_axis, up)) > 0.999) {
    up = {1.0, 0.0, 0.0};
  }
  const Vec3 x_axis = normalizeOr(cross(up, z_axis), {1.0, 0.0, 0.0});
  const Vec3 y_axis = cross(z_axis, x_axis);
  return quaternionFromRotationColumns(x_axis, y_axis, z_axis);
}

geometry_msgs::msg::Quaternion identityQuaternion()
{
  geometry_msgs::msg::Quaternion q;
  q.w = 1.0;
  return q;
}

}  // namespace

class HumanColliderVizNode : public rclcpp::Node
{
public:
  HumanColliderVizNode()
  : Node("human_collider_viz_node")
  {
    input_collider_topic_ =
      declare_parameter<std::string>("input_collider_topic", "/human/colliders");
    collider_markers_topic_ =
      declare_parameter<std::string>("collider_markers_topic", "/human/collider_markers");
    collider_marker_rate_ = declare_parameter<double>("collider_marker_rate", 5.0);
    collision_geometry_ = declare_parameter<std::string>("collision_geometry", "capsules");
    sphere_interpolation_level_ = declare_parameter<int>("sphere_interpolation_level", 0);
    sphere_radius_gain_ = declare_parameter<double>("sphere_radius_gain", 1.0);

    if (collider_marker_rate_ <= 0.0) {
      RCLCPP_WARN(get_logger(), "collider_marker_rate must be positive; using 5 Hz");
      collider_marker_rate_ = 5.0;
    }
    if (
      collision_geometry_ != "capsules" &&
      collision_geometry_ != "boxes" &&
      collision_geometry_ != "spheres")
    {
      RCLCPP_WARN(
        get_logger(), "Unsupported collision_geometry '%s'; using capsules",
        collision_geometry_.c_str());
      collision_geometry_ = "capsules";
    }

    rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
    sensor_qos.best_effort();
    sensor_qos.durability_volatile();

    collider_sub_ = create_subscription<CapsuleArray>(
      input_collider_topic_, sensor_qos,
      std::bind(&HumanColliderVizNode::colliderCallback, this, std::placeholders::_1));
    marker_pub_ = create_publisher<MarkerArray>(collider_markers_topic_, 10);

    const auto period = std::chrono::nanoseconds(
      static_cast<int64_t>(1e9 / std::max(collider_marker_rate_, 1.0)));
    timer_ = create_wall_timer(period, std::bind(&HumanColliderVizNode::timerCallback, this));

    RCLCPP_INFO(
      get_logger(), "human_collider_viz_node ready: %s -> %s (%s at %.1f Hz)",
      input_collider_topic_.c_str(), collider_markers_topic_.c_str(),
      collision_geometry_.c_str(), collider_marker_rate_);
  }

private:
  void colliderCallback(const CapsuleArray::SharedPtr msg)
  {
    latest_colliders_ = msg;
    latest_dirty_ = true;
  }

  void timerCallback()
  {
    if (!latest_colliders_) {
      return;
    }

    MarkerArray markers;
    if (collision_geometry_ == "spheres") {
      markers = makeSphereMarkers(*latest_colliders_);
    } else if (collision_geometry_ == "boxes") {
      markers = makeBoxMarkers(*latest_colliders_);
    } else {
      markers = makeCapsuleMarkers(*latest_colliders_);
    }

    marker_pub_->publish(markers);
    latest_dirty_ = false;
  }

  MarkerArray makeCapsuleMarkers(const CapsuleArray & colliders)
  {
    MarkerArray msg;
    int id = 0;
    for (const auto & cap : colliders.capsules) {
      const Vec3 a = toVec3(cap.a);
      const Vec3 b = toVec3(cap.b);
      const Vec3 center = (a + b) * 0.5;
      const double diameter = 2.0 * std::max(0.0, cap.radius);
      const double shaft_length = norm(a - b);

      msg.markers.push_back(makeMarker(
        colliders, id++, Marker::CYLINDER, center, axisToQuaternion(a, b),
        diameter, diameter, shaft_length));
      msg.markers.push_back(makeMarker(
        colliders, id++, Marker::SPHERE, a, identityQuaternion(),
        diameter, diameter, diameter));
      msg.markers.push_back(makeMarker(
        colliders, id++, Marker::SPHERE, b, identityQuaternion(),
        diameter, diameter, diameter));
    }
    cleanupStale(colliders, msg, id);
    return msg;
  }

  MarkerArray makeBoxMarkers(const CapsuleArray & colliders)
  {
    MarkerArray msg;
    int id = 0;
    for (const auto & cap : colliders.capsules) {
      const Vec3 a = toVec3(cap.a);
      const Vec3 b = toVec3(cap.b);
      const Vec3 center = (a + b) * 0.5;
      const double diameter = 2.0 * std::max(0.0, cap.radius);
      const double full_length = norm(a - b) + diameter;

      msg.markers.push_back(makeMarker(
        colliders, id++, Marker::CUBE, center, axisToQuaternion(a, b),
        diameter, diameter, full_length));
    }
    cleanupStale(colliders, msg, id);
    return msg;
  }

  MarkerArray makeSphereMarkers(const CapsuleArray & colliders)
  {
    MarkerArray msg;
    int id = 0;
    const auto identity = identityQuaternion();
    const int interp = std::max(0, sphere_interpolation_level_);
    for (const auto & cap : colliders.capsules) {
      const Vec3 a = toVec3(cap.a);
      const Vec3 b = toVec3(cap.b);
      const double radius = std::max(0.0, cap.radius);
      const double diameter = 2.0 * radius * sphere_radius_gain_;
      const double full_length = norm(a - b) + 2.0 * radius;
      const int base_count = radius > kEps ?
        std::max(1, static_cast<int>(std::llround(full_length / (2.0 * radius)))) : 1;
      const int total_count = base_count + std::max(0, base_count - 1) * interp;

      for (int i = 0; i < total_count; ++i) {
        const double t = total_count == 1 ?
          0.5 : static_cast<double>(i) / static_cast<double>(total_count - 1);
        const Vec3 center = b * (1.0 - t) + a * t;
        msg.markers.push_back(makeMarker(
          colliders, id++, Marker::SPHERE, center, identity,
          diameter, diameter, diameter));
      }
    }
    cleanupStale(colliders, msg, id);
    return msg;
  }

  Marker makeMarker(
    const CapsuleArray & colliders,
    int id,
    int type,
    const Vec3 & center,
    const geometry_msgs::msg::Quaternion & quat,
    double sx,
    double sy,
    double sz) const
  {
    Marker marker;
    marker.header = colliders.header;
    marker.ns = "human_colliders";
    marker.id = id;
    marker.type = type;
    marker.action = Marker::ADD;
    marker.pose.position = toPoint(center);
    marker.pose.orientation = quat;
    marker.scale.x = sx;
    marker.scale.y = sy;
    marker.scale.z = sz;
    marker.color.r = 0.9;
    marker.color.g = 0.5;
    marker.color.b = 0.1;
    marker.color.a = 0.3;
    return marker;
  }

  void cleanupStale(const CapsuleArray & colliders, MarkerArray & msg, int next_id)
  {
    for (int id = next_id; id < previous_marker_count_; ++id) {
      Marker marker;
      marker.header = colliders.header;
      marker.ns = "human_colliders";
      marker.id = id;
      marker.action = Marker::DELETE;
      msg.markers.push_back(marker);
    }
    previous_marker_count_ = next_id;
  }

  std::string input_collider_topic_;
  std::string collider_markers_topic_;
  double collider_marker_rate_ = 5.0;
  std::string collision_geometry_ = "capsules";
  int sphere_interpolation_level_ = 0;
  double sphere_radius_gain_ = 1.0;

  CapsuleArray::SharedPtr latest_colliders_;
  bool latest_dirty_ = false;
  int previous_marker_count_ = 0;

  rclcpp::Subscription<CapsuleArray>::SharedPtr collider_sub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HumanColliderVizNode>());
  rclcpp::shutdown();
  return 0;
}

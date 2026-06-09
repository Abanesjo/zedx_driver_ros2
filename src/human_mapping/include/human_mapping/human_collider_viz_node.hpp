#pragma once

#include "human_mapping/geometry.hpp"

#include <string>

#include <g1_cbf_msg/msg/capsule_array.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace human_mapping {

using CapsuleArray = g1_cbf_msg::msg::CapsuleArray;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;

class HumanColliderVizNode : public rclcpp::Node {
public:
  HumanColliderVizNode();

private:
  void colliderCallback(const CapsuleArray::SharedPtr msg);

  void timerCallback();

  MarkerArray makeCapsuleMarkers(const CapsuleArray &colliders);

  MarkerArray makeBoxMarkers(const CapsuleArray &colliders);

  MarkerArray makeSphereMarkers(const CapsuleArray &colliders);

  Marker makeMarker(const CapsuleArray &colliders, int id, int type,
                    const Vec3 &center,
                    const geometry_msgs::msg::Quaternion &quat, double sx,
                    double sy, double sz) const;

  void cleanupStale(const CapsuleArray &colliders, MarkerArray &msg,
                    int next_id);

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

} // namespace human_mapping

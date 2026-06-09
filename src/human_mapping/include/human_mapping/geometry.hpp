#pragma once

#include <cmath>
#include <optional>

#include <geometry_msgs/msg/point.hpp>

namespace human_mapping {

inline constexpr double kEps = 1e-8;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3 &v, double s) {
  return {v.x * s, v.y * s, v.z * s};
}

inline Vec3 operator*(double s, const Vec3 &v) { return v * s; }

inline double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double norm(const Vec3 &v) { return std::sqrt(dot(v, v)); }

inline std::optional<Vec3> normalize(const Vec3 &v) {
  const double n = norm(v);
  if (n < kEps || !std::isfinite(n)) {
    return std::nullopt;
  }
  return v * (1.0 / n);
}

inline Vec3 normalizeOr(const Vec3 &v, const Vec3 &fallback) {
  const double n = norm(v);
  if (!std::isfinite(n) || n < kEps) {
    return fallback;
  }
  return v * (1.0 / n);
}

inline Vec3 toVec3(const geometry_msgs::msg::Point &p) {
  return {p.x, p.y, p.z};
}

inline geometry_msgs::msg::Point toPoint(const Vec3 &p) {
  geometry_msgs::msg::Point out;
  out.x = p.x;
  out.y = p.y;
  out.z = p.z;
  return out;
}

} // namespace human_mapping

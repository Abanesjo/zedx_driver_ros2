#pragma once

#include <sl/Camera.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace zed_launcher {

using Bone = std::pair<int, int>;

template <typename PartT>
std::vector<Bone>
toIndexBones(const std::vector<std::pair<PartT, PartT>> &bones) {
  std::vector<Bone> indexed;
  indexed.reserve(bones.size());
  for (const auto &bone : bones) {
    indexed.emplace_back(sl::getIdx(bone.first), sl::getIdx(bone.second));
  }
  return indexed;
}

inline const std::vector<Bone> &bonesForFormat(int8_t body_format) {
  static const auto body_18_bones = toIndexBones(sl::BODY_18_BONES);
  static const auto body_34_bones = toIndexBones(sl::BODY_34_BONES);
  static const auto body_38_bones = toIndexBones(sl::BODY_38_BONES);

  if (body_format == 0) {
    return body_18_bones;
  }
  if (body_format == 1) {
    return body_34_bones;
  }
  return body_38_bones;
}

inline cv::Scalar colorForId(int id) {
  static const std::vector<cv::Scalar> colors = {
      cv::Scalar(232.0, 176.0, 59.0),  cv::Scalar(175.0, 208.0, 25.0),
      cv::Scalar(102.0, 205.0, 105.0), cv::Scalar(185.0, 0.0, 255.0),
      cv::Scalar(99.0, 107.0, 252.0),  cv::Scalar(252.0, 225.0, 8.0),
      cv::Scalar(167.0, 130.0, 141.0), cv::Scalar(194.0, 72.0, 113.0)};

  if (id < 0) {
    return cv::Scalar(236.0, 184.0, 36.0);
  }
  return colors[static_cast<std::size_t>(id) % colors.size()];
}

} // namespace zed_launcher

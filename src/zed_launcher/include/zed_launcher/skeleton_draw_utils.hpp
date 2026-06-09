#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace zed_launcher {

using Bone = std::pair<int, int>;

const std::vector<Bone> &bonesForFormat(int8_t body_format);
cv::Scalar colorForId(int id);

} // namespace zed_launcher

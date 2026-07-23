#include "zed_launcher/camera_body_consensus.hpp"

#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace {

zed_msgs::msg::Object bodyAt(float x, float y, float z, float confidence) {
  zed_msgs::msg::Object body;
  body.position = {x, y, z};
  body.confidence = confidence;
  return body;
}

TEST(CameraBodyConsensus, SelectsMedoidAndRejectsDistantOutlier) {
  const std::vector<zed_msgs::msg::Object> candidates = {
      bodyAt(1.00F, 2.00F, 0.75F, 80.0F),
      bodyAt(1.04F, 2.01F, 0.76F, 90.0F),
      bodyAt(4.00F, -2.00F, 0.75F, 100.0F),
  };

  const auto selected =
      zed_launcher::selectCameraBodyConsensus(candidates, 2, 0.50);
  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(*selected, 1U);
}

TEST(CameraBodyConsensus, RequiresConfiguredNumberOfCameras) {
  const std::vector<zed_msgs::msg::Object> candidates = {
      bodyAt(1.0F, 2.0F, 0.75F, 95.0F),
      bodyAt(2.0F, 2.0F, 0.75F, 95.0F),
  };

  EXPECT_FALSE(
      zed_launcher::selectCameraBodyConsensus(candidates, 2, 0.25).has_value());
  EXPECT_TRUE(
      zed_launcher::selectCameraBodyConsensus(candidates, 1, 0.25).has_value());
}

TEST(CameraBodyConsensus, IgnoresNonFinitePositions) {
  auto invalid = bodyAt(1.0F, 2.0F, 0.75F, 100.0F);
  invalid.position[0] = std::numeric_limits<float>::quiet_NaN();
  const std::vector<zed_msgs::msg::Object> candidates = {
      invalid, bodyAt(1.0F, 2.0F, 0.75F, 80.0F)};

  const auto selected =
      zed_launcher::selectCameraBodyConsensus(candidates, 1, 0.50);
  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(*selected, 1U);
}

} // namespace

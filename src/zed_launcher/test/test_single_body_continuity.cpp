#include "zed_launcher/single_body_continuity.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

using zed_launcher::SingleBodyContinuity;
using namespace std::chrono_literals;

zed_msgs::msg::Object makeObservation() {
  zed_msgs::msg::Object object;
  object.label = "Body_73";
  object.label_id = 73;
  object.position = {1.0F, 2.0F, 3.0F};
  object.velocity = {2.0F, -4.0F, 0.5F};
  object.tracking_state = 1;
  object.skeleton_available = true;
  object.body_format = 0;
  object.skeleton_3d.keypoints[0].kp = {1.5F, 2.5F, 3.5F};
  object.skeleton_3d.keypoints[17].kp = {4.0F, 5.0F, 6.0F};
  object.skeleton_3d.keypoints[18].kp = {9.0F, 9.0F, 9.0F};
  object.head_position = {1.0F, 2.0F, 4.0F};
  object.dimensions_3d = {0.5F, 0.4F, 1.8F};
  object.bounding_box_3d.corners[0].kp = {0.0F, 1.0F, 2.0F};
  object.head_bounding_box_3d.corners[0].kp = {0.5F, 1.5F, 3.5F};
  return object;
}

TEST(SingleBodyContinuity, RewritesOnlyTheRosFacingIdentityOnObservation) {
  SingleBodyContinuity continuity(0, 0.5, 10.0);
  const auto now = SingleBodyContinuity::TimePoint{10s};
  const auto result = continuity.observe(makeObservation(), now);

  EXPECT_EQ(result.label_id, 0);
  EXPECT_EQ(result.label, "Body_0");
  EXPECT_EQ(result.tracking_state, 1);
  EXPECT_FLOAT_EQ(result.position[0], 1.0F);
  EXPECT_FLOAT_EQ(result.velocity[1], -4.0F);
}

TEST(SingleBodyContinuity,
     BridgesDropoutWithMonotonicConstantVelocityPrediction) {
  SingleBodyContinuity continuity(4, 0.5, 10.0);
  const auto observed_at = SingleBodyContinuity::TimePoint{10s};
  continuity.observe(makeObservation(), observed_at);

  const auto prediction = continuity.bridge(observed_at + 250ms);
  ASSERT_TRUE(prediction.has_value());
  EXPECT_EQ(prediction->label_id, 4);
  EXPECT_EQ(prediction->label, "Body_4");
  EXPECT_EQ(prediction->tracking_state,
            SingleBodyContinuity::kSearchingTrackingState);
  EXPECT_FLOAT_EQ(prediction->position[0], 1.5F);
  EXPECT_FLOAT_EQ(prediction->position[1], 1.0F);
  EXPECT_FLOAT_EQ(prediction->position[2], 3.125F);
  EXPECT_FLOAT_EQ(prediction->head_position[0], 1.5F);
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[0].kp[0], 2.0F);
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[17].kp[1], 4.0F);
  EXPECT_FLOAT_EQ(prediction->bounding_box_3d.corners[0].kp[2], 2.125F);
  EXPECT_FLOAT_EQ(prediction->head_bounding_box_3d.corners[0].kp[1], 0.5F);

  // BODY_18 has only 18 meaningful entries. Unused message capacity must not
  // turn into phantom translated keypoints.
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[18].kp[0], 9.0F);
}

TEST(SingleBodyContinuity, TranslatesAllAndOnlyBody34Keypoints) {
  SingleBodyContinuity continuity(0, 0.5, 10.0);
  auto observation = makeObservation();
  observation.body_format = 1;
  observation.skeleton_3d.keypoints[33].kp = {3.0F, 4.0F, 5.0F};
  observation.skeleton_3d.keypoints[34].kp = {9.0F, 9.0F, 9.0F};
  const auto observed_at = SingleBodyContinuity::TimePoint{10s};
  continuity.observe(observation, observed_at);

  const auto prediction = continuity.bridge(observed_at + 250ms);
  ASSERT_TRUE(prediction.has_value());
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[33].kp[0], 3.5F);
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[33].kp[1], 3.0F);
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[33].kp[2], 5.125F);
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[34].kp[0], 9.0F);
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[34].kp[1], 9.0F);
  EXPECT_FLOAT_EQ(prediction->skeleton_3d.keypoints[34].kp[2], 9.0F);
}

TEST(SingleBodyContinuity, StopsBridgingAfterConfiguredTimeout) {
  SingleBodyContinuity continuity(0, 0.5, 10.0);
  const auto observed_at = SingleBodyContinuity::TimePoint{10s};

  EXPECT_FALSE(continuity.bridge(observed_at).has_value());
  continuity.observe(makeObservation(), observed_at);
  EXPECT_TRUE(continuity.bridge(observed_at + 500ms).has_value());
  EXPECT_FALSE(continuity.bridge(observed_at + 501ms).has_value());
  EXPECT_FALSE(continuity.bridge(observed_at - 1ms).has_value());
}

TEST(SingleBodyContinuity, ClampsUntrustedSdkVelocityVector) {
  SingleBodyContinuity continuity(0, 0.5, 2.0);
  const auto observed_at = SingleBodyContinuity::TimePoint{10s};
  auto observation = makeObservation();
  observation.velocity = {30.0F, 40.0F, 0.0F};
  continuity.observe(observation, observed_at);

  const auto prediction = continuity.bridge(observed_at + 500ms);
  ASSERT_TRUE(prediction.has_value());
  EXPECT_NEAR(prediction->position[0], 1.6, 1e-6);
  EXPECT_NEAR(prediction->position[1], 2.8, 1e-6);
  EXPECT_NEAR(prediction->velocity[0], 1.2, 1e-6);
  EXPECT_NEAR(prediction->velocity[1], 1.6, 1e-6);
}

TEST(SingleBodyContinuity,
     CameraFallbackMovesLastFittedSkeletonWithBoundedRootStep) {
  SingleBodyContinuity continuity(0, 0.5, 2.0);
  const auto observed_at = SingleBodyContinuity::TimePoint{10s};
  auto fitted = makeObservation();
  fitted.body_format = 1;
  fitted.skeleton_3d.keypoints[0].kp = {1.5F, 2.5F, 3.5F};
  continuity.observe(fitted, observed_at);

  auto camera_fallback = makeObservation();
  camera_fallback.body_format = 0;
  camera_fallback.position = {5.0F, 2.0F, 3.0F};
  camera_fallback.skeleton_3d.keypoints[0].kp = {99.0F, 99.0F, 99.0F};
  camera_fallback.velocity = {0.5F, 0.0F, 0.0F};
  camera_fallback.confidence = 87.0F;

  const auto result =
      continuity.observeFallbackPosition(camera_fallback, observed_at + 100ms);

  EXPECT_EQ(result.body_format, 1);
  EXPECT_NEAR(result.position[0], 1.2, 1e-6);
  EXPECT_NEAR(result.skeleton_3d.keypoints[0].kp[0], 1.7, 1e-6);
  EXPECT_FLOAT_EQ(result.skeleton_3d.keypoints[0].kp[1], 2.5F);
  EXPECT_FLOAT_EQ(result.velocity[0], 0.5F);
  EXPECT_FLOAT_EQ(result.confidence, 87.0F);
}

TEST(SingleBodyContinuity, CameraFallbackUsesRawSkeletonAfterReset) {
  SingleBodyContinuity continuity(0, 0.5, 2.0);
  const auto observed_at = SingleBodyContinuity::TimePoint{10s};
  continuity.observe(makeObservation(), observed_at);
  continuity.reset();

  auto camera_fallback = makeObservation();
  camera_fallback.position = {5.0F, 2.0F, 3.0F};
  camera_fallback.skeleton_3d.keypoints[0].kp = {5.5F, 2.5F, 3.5F};
  const auto result =
      continuity.observeFallbackPosition(camera_fallback, observed_at + 1s);

  EXPECT_FLOAT_EQ(result.position[0], 5.0F);
  EXPECT_FLOAT_EQ(result.skeleton_3d.keypoints[0].kp[0], 5.5F);
}

TEST(SingleBodyContinuity, RejectsInvalidConfiguration) {
  EXPECT_THROW(SingleBodyContinuity(-1, 0.5, 2.0), std::invalid_argument);
  EXPECT_THROW(SingleBodyContinuity(32768, 0.5, 2.0), std::invalid_argument);
  EXPECT_THROW(SingleBodyContinuity(0, -0.1, 2.0), std::invalid_argument);
  EXPECT_THROW(
      SingleBodyContinuity(0, std::numeric_limits<double>::quiet_NaN(), 2.0),
      std::invalid_argument);
  EXPECT_THROW(
      SingleBodyContinuity(0, 0.5, std::numeric_limits<double>::quiet_NaN()),
      std::invalid_argument);
  EXPECT_THROW(SingleBodyContinuity(0, 0.5, -0.1), std::invalid_argument);
}

} // namespace

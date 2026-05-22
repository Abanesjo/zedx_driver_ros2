#define HUMAN_MAPPING_NODE_DISABLE_MAIN
#include "../src/human_mapping_node.cpp"

#include <gtest/gtest.h>

namespace
{

constexpr int64_t kMs = 1'000'000LL;

void expectPointNear(const Vec3 & point, double x, double y, double z)
{
  EXPECT_NEAR(point.x, x, 1e-9);
  EXPECT_NEAR(point.y, y, 1e-9);
  EXPECT_NEAR(point.z, z, 1e-9);
}

TEST(KeypointFilter, RejectsSingleFrameJumpAndKeepsLastPoint)
{
  KeypointFilter filter(0.5, 0.6, 3, 0.25, 0.5);

  const auto initial = filter.update(Vec3{0.0, 0.0, 1.0}, 0);
  ASSERT_TRUE(initial.point);
  expectPointNear(*initial.point, 0.0, 0.0, 1.0);

  const auto rejected = filter.update(Vec3{3.0, 0.0, 1.0}, 10 * kMs);
  ASSERT_TRUE(rejected.point);
  EXPECT_EQ(rejected.event, PointFilterEvent::kJumpRejected);
  expectPointNear(*rejected.point, 0.0, 0.0, 1.0);

  const auto recovered = filter.update(Vec3{0.2, 0.0, 1.0}, 20 * kMs);
  ASSERT_TRUE(recovered.point);
  EXPECT_EQ(recovered.event, PointFilterEvent::kNone);
  expectPointNear(*recovered.point, 0.1, 0.0, 1.0);
}

TEST(KeypointFilter, AcceptsPersistentStableJump)
{
  KeypointFilter filter(0.5, 0.6, 3, 0.25, 0.5);

  ASSERT_TRUE(filter.update(Vec3{0.0, 0.0, 1.0}, 0).point);
  EXPECT_EQ(
    filter.update(Vec3{2.0, 0.0, 1.0}, 10 * kMs).event,
    PointFilterEvent::kJumpRejected);
  EXPECT_EQ(
    filter.update(Vec3{2.1, 0.0, 1.0}, 20 * kMs).event,
    PointFilterEvent::kJumpRejected);

  const auto accepted = filter.update(Vec3{2.2, 0.0, 1.0}, 30 * kMs);
  ASSERT_TRUE(accepted.point);
  EXPECT_EQ(accepted.event, PointFilterEvent::kJumpAccepted);
  expectPointNear(*accepted.point, 2.2, 0.0, 1.0);
}

TEST(KeypointFilter, HoldsShortDropoutThenResetsStaleState)
{
  KeypointFilter filter(0.5, 0.6, 3, 0.25, 0.5);

  ASSERT_TRUE(filter.update(Vec3{1.0, 0.0, 1.0}, 0).point);

  const auto held = filter.update(std::nullopt, 200 * kMs);
  ASSERT_TRUE(held.point);
  EXPECT_EQ(held.event, PointFilterEvent::kHeldMissing);
  expectPointNear(*held.point, 1.0, 0.0, 1.0);

  const auto expired = filter.update(std::nullopt, 300 * kMs);
  EXPECT_FALSE(expired.point);
  EXPECT_EQ(expired.event, PointFilterEvent::kMissingExpired);

  const auto reset = filter.update(std::nullopt, 600 * kMs);
  EXPECT_FALSE(reset.point);
  EXPECT_EQ(reset.event, PointFilterEvent::kResetStale);

  const auto reinitialized = filter.update(Vec3{5.0, 0.0, 1.0}, 610 * kMs);
  ASSERT_TRUE(reinitialized.point);
  EXPECT_EQ(reinitialized.event, PointFilterEvent::kNone);
  expectPointNear(*reinitialized.point, 5.0, 0.0, 1.0);
}

}  // namespace

#include "human_mapping/human_mapping_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

namespace human_mapping {
namespace {

constexpr std::size_t kWaistYaw = jointIndex(JointIndex::WaistYaw);
constexpr std::size_t kLeftShoulderPitch =
    jointIndex(JointIndex::LeftShoulderPitch);
constexpr std::size_t kLeftShoulderRoll =
    jointIndex(JointIndex::LeftShoulderRoll);
constexpr std::size_t kLeftShoulderYaw =
    jointIndex(JointIndex::LeftShoulderYaw);
constexpr std::size_t kLeftElbowPitch = jointIndex(JointIndex::LeftElbow);
constexpr std::size_t kRightShoulderPitch =
    jointIndex(JointIndex::RightShoulderPitch);
constexpr std::size_t kRightShoulderRoll =
    jointIndex(JointIndex::RightShoulderRoll);
constexpr std::size_t kRightShoulderYaw =
    jointIndex(JointIndex::RightShoulderYaw);

double radians(double degrees) { return degrees * kPi / 180.0; }

Vec3 rotateZ(const Vec3 &point, double angle) {
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return {cosine * point.x - sine * point.y, sine * point.x + cosine * point.y,
          point.z};
}

Vec3 rotateAroundAxis(const Vec3 &point, const Vec3 &axis, double angle) {
  const Vec3 unit = axis * (1.0 / norm(axis));
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return point * cosine + cross(unit, point) * sine +
         unit * (dot(unit, point) * (1.0 - cosine));
}

BodyPoints makePose(double waist_yaw = 0.0, double left_shoulder_yaw = 0.0,
                    double right_shoulder_yaw = 0.0) {
  BodyPoints points{};
  points[kPelvis] = Vec3{0.0, 0.0, 1.0};
  points[kNeck] = Vec3{0.0, 0.0, 1.7};
  points[kLeftHip] = Vec3{-0.2, 0.0, 0.9};
  points[kRightHip] = Vec3{0.2, 0.0, 0.9};
  points[kLeftKnee] = Vec3{-0.2, 0.0, 0.5};
  points[kRightKnee] = Vec3{0.2, 0.0, 0.5};
  points[kLeftAnkle] = Vec3{-0.2, 0.0, 0.1};
  points[kRightAnkle] = Vec3{0.2, 0.0, 0.1};

  const Vec3 shoulder_center{0.0, 0.0, 1.5};
  const Vec3 left_offset = rotateZ({-0.3, 0.0, 0.0}, waist_yaw);
  const Vec3 right_offset = rotateZ({0.3, 0.0, 0.0}, waist_yaw);
  points[kLeftShoulder] = shoulder_center + left_offset;
  points[kRightShoulder] = shoulder_center + right_offset;

  const Vec3 upper_arm{0.0, 0.0, -0.4};
  points[kLeftElbow] = *points[kLeftShoulder] + upper_arm;
  points[kRightElbow] = *points[kRightShoulder] + upper_arm;

  const Vec3 torso_forward = rotateZ({0.0, 0.3, 0.0}, waist_yaw);
  points[kLeftWrist] =
      *points[kLeftElbow] + rotateZ(torso_forward, left_shoulder_yaw);
  points[kRightWrist] =
      *points[kRightElbow] + rotateZ(torso_forward, right_shoulder_yaw);
  return points;
}

BodyPoints rotatePoseAroundWorldZ(const BodyPoints &input, double angle) {
  BodyPoints output = input;
  for (auto &point : output) {
    if (point) {
      point = rotateZ(*point, angle);
    }
  }
  return output;
}

void setArmFromG1Shoulder(BodyPoints &points, bool is_left, double pitch,
                          double roll, double yaw) {
  const int shoulder = is_left ? kLeftShoulder : kRightShoulder;
  const int elbow = is_left ? kLeftElbow : kRightElbow;
  const int wrist = is_left ? kLeftWrist : kRightWrist;

  const Vec3 upper{-std::sin(roll), -std::sin(pitch) * std::cos(roll),
                   -std::cos(pitch) * std::cos(roll)};
  const Vec3 zero_bend{0.0, std::cos(pitch), -std::sin(pitch)};
  const Vec3 forearm = rotateAroundAxis(zero_bend, -1.0 * upper, yaw);
  points[elbow] = *points[shoulder] + upper * 0.4;
  points[wrist] = *points[elbow] + forearm * 0.3;
}

void expectNear(double actual, double expected, double tolerance = 1e-9) {
  EXPECT_NEAR(wrapAngle(actual - expected), 0.0, tolerance);
}

const CapsuleData *findCapsule(const std::vector<CapsuleData> &capsules,
                               const std::string &name) {
  const auto match = std::find_if(
      capsules.begin(), capsules.end(),
      [&](const CapsuleData &capsule) { return capsule.name == name; });
  return match == capsules.end() ? nullptr : &*match;
}

void expectVecNear(const Vec3 &actual, const Vec3 &expected,
                   double tolerance = 1e-9) {
  EXPECT_NEAR(actual.x, expected.x, tolerance);
  EXPECT_NEAR(actual.y, expected.y, tolerance);
  EXPECT_NEAR(actual.z, expected.z, tolerance);
}

void setObjectPoint(Object &object, std::size_t index, const Vec3 &point) {
  object.skeleton_3d.keypoints[index].kp = {
      static_cast<float>(point.x), static_cast<float>(point.y),
      static_cast<float>(point.z)};
}

Object encodePose(const BodyPoints &points, int body_format) {
  Object object;
  object.body_format = static_cast<int8_t>(body_format);

  auto copy = [&](std::size_t destination, int source) {
    if (points[source]) {
      setObjectPoint(object, destination, *points[source]);
    }
  };

  switch (decodeBodyFormat(body_format)) {
  case SkeletonBodyFormat::Body18:
    copy(1, kNeck);
    copy(2, kRightShoulder);
    copy(3, kRightElbow);
    copy(4, kRightWrist);
    copy(5, kLeftShoulder);
    copy(6, kLeftElbow);
    copy(7, kLeftWrist);
    copy(8, kRightHip);
    copy(9, kRightKnee);
    copy(10, kRightAnkle);
    copy(11, kLeftHip);
    copy(12, kLeftKnee);
    copy(13, kLeftAnkle);
    break;
  case SkeletonBodyFormat::Body34:
    copy(0, kPelvis);
    copy(3, kNeck);
    copy(5, kLeftShoulder);
    copy(6, kLeftElbow);
    copy(7, kLeftWrist);
    copy(12, kRightShoulder);
    copy(13, kRightElbow);
    copy(14, kRightWrist);
    copy(18, kLeftHip);
    copy(19, kLeftKnee);
    copy(20, kLeftAnkle);
    copy(22, kRightHip);
    copy(23, kRightKnee);
    copy(24, kRightAnkle);
    break;
  case SkeletonBodyFormat::Body38:
    for (std::size_t index = 0; index < kNumBody38Points; ++index) {
      copy(index, static_cast<int>(index));
    }
    break;
  case SkeletonBodyFormat::Unsupported:
    break;
  }
  return object;
}

TEST(BodyFormat, SupportsSdkAndLegacyValuesForCapsules) {
  EXPECT_EQ(decodeBodyFormat(0), SkeletonBodyFormat::Body18);
  EXPECT_EQ(decodeBodyFormat(1), SkeletonBodyFormat::Body34);
  EXPECT_EQ(decodeBodyFormat(2), SkeletonBodyFormat::Body38);
  EXPECT_EQ(decodeBodyFormat(18), SkeletonBodyFormat::Body18);
  EXPECT_EQ(decodeBodyFormat(34), SkeletonBodyFormat::Body34);
  EXPECT_EQ(decodeBodyFormat(38), SkeletonBodyFormat::Body38);
  EXPECT_EQ(decodeBodyFormat(17), SkeletonBodyFormat::Unsupported);

  EXPECT_EQ(bodyFormatKeypointCount(SkeletonBodyFormat::Body18), 18U);
  EXPECT_EQ(bodyFormatKeypointCount(SkeletonBodyFormat::Body34), 34U);
  EXPECT_EQ(bodyFormatKeypointCount(SkeletonBodyFormat::Body38), 38U);
  EXPECT_TRUE(bodyFormatSupportsCapsules(0));
  EXPECT_TRUE(bodyFormatSupportsCapsules(1));
  EXPECT_TRUE(bodyFormatSupportsCapsules(2));
  EXPECT_FALSE(bodyFormatSupportsCapsules(17));

  EXPECT_FALSE(bodyFormatSupportsJointAngles(0));
  EXPECT_FALSE(bodyFormatSupportsJointAngles(1));
  EXPECT_TRUE(bodyFormatSupportsJointAngles(2));
}

TEST(BodyFormat, Body18SynthesizesPelvisAtHipMidpoint) {
  BodyPoints source = makePose();
  source[kLeftHip] = Vec3{-0.24, 0.08, 0.92};
  source[kRightHip] = Vec3{0.18, 0.04, 0.96};

  const BodyPoints canonical = canonicalBodyPoints(encodePose(source, 0));
  ASSERT_TRUE(canonical[kPelvis]);
  ASSERT_TRUE(canonical[kLeftHip]);
  ASSERT_TRUE(canonical[kRightHip]);
  expectVecNear(*canonical[kPelvis], {-0.03, 0.06, 0.94}, 1e-7);
  expectVecNear(*canonical[kLeftHip], *source[kLeftHip], 1e-7);
  expectVecNear(*canonical[kRightHip], *source[kRightHip], 1e-7);
}

TEST(BodyFormat, AllSupportedFormatsBuildEquivalentCapsules) {
  BodyPoints source = makePose();
  source[kPelvis] = 0.5 * (*source[kLeftHip] + *source[kRightHip]);

  const auto reference = buildBodyCapsules(
      canonicalBodyPoints(encodePose(source, 2)), BodyCapsuleConfig{});
  ASSERT_EQ(reference.size(), 9U);

  for (const int body_format : {0, 1, 2}) {
    const BodyPoints canonical =
        canonicalBodyPoints(encodePose(source, body_format));
    const auto capsules =
        buildBodyCapsules(canonical, BodyCapsuleConfig{});
    ASSERT_EQ(capsules.size(), reference.size())
        << "body_format=" << body_format;
    for (std::size_t index = 0; index < reference.size(); ++index) {
      EXPECT_EQ(capsules[index].name, reference[index].name);
      expectVecNear(capsules[index].a, reference[index].a, 1e-7);
      expectVecNear(capsules[index].b, reference[index].b, 1e-7);
      EXPECT_DOUBLE_EQ(capsules[index].radius, reference[index].radius);
    }
  }
}

TEST(EstimateJointAngles, WaistYawTracksRelativeShoulderTwist) {
  for (const double expected : {radians(45.0), radians(-35.0)}) {
    const JointAngles angles = estimateJointAngles(makePose(expected));
    ASSERT_TRUE(angles.valid[kWaistYaw]);
    expectNear(angles.values[kWaistYaw], expected);
  }
}

TEST(EstimateJointAngles, ShoulderYawTracksBentArmPlane) {
  const double left_expected = radians(45.0);
  const double right_expected = radians(-35.0);
  const JointAngles angles =
      estimateJointAngles(makePose(0.0, left_expected, right_expected));

  ASSERT_TRUE(angles.valid[kLeftShoulderYaw]);
  ASSERT_TRUE(angles.valid[kRightShoulderYaw]);
  expectNear(angles.values[kLeftShoulderYaw], left_expected);
  expectNear(angles.values[kRightShoulderYaw], right_expected);
}

TEST(EstimateJointAngles, ShoulderYawDoesNotLeakCombinedPitchAndRoll) {
  BodyPoints zero_yaw = makePose();
  setArmFromG1Shoulder(zero_yaw, true, 0.8, 0.55, 0.0);
  setArmFromG1Shoulder(zero_yaw, false, -0.65, -0.45, 0.0);
  const JointAngles zero_angles = estimateJointAngles(zero_yaw);
  ASSERT_TRUE(zero_angles.valid[kLeftShoulderYaw]);
  ASSERT_TRUE(zero_angles.valid[kRightShoulderYaw]);
  expectNear(zero_angles.values[kLeftShoulderPitch], -0.8);
  expectNear(zero_angles.values[kLeftShoulderRoll], 0.55);
  expectNear(zero_angles.values[kRightShoulderPitch], 0.65);
  expectNear(zero_angles.values[kRightShoulderRoll], 0.45);
  expectNear(zero_angles.values[kLeftShoulderYaw], 0.0);
  expectNear(zero_angles.values[kRightShoulderYaw], 0.0);

  BodyPoints yawed = makePose();
  setArmFromG1Shoulder(yawed, true, 0.8, 0.55, radians(40.0));
  setArmFromG1Shoulder(yawed, false, -0.65, -0.45, radians(-35.0));
  const JointAngles yawed_angles = estimateJointAngles(yawed);
  ASSERT_TRUE(yawed_angles.valid[kLeftShoulderYaw]);
  ASSERT_TRUE(yawed_angles.valid[kRightShoulderYaw]);
  expectNear(yawed_angles.values[kLeftShoulderYaw], radians(40.0));
  expectNear(yawed_angles.values[kRightShoulderYaw], radians(-35.0));
}

TEST(EstimateJointAngles, RelativeYawsIgnoreRigidWorldHeading) {
  const double waist_expected = radians(30.0);
  const double left_expected = radians(40.0);
  const double right_expected = radians(-25.0);
  const BodyPoints original =
      makePose(waist_expected, left_expected, right_expected);
  const BodyPoints rotated = rotatePoseAroundWorldZ(original, radians(70.0));

  const JointAngles before = estimateJointAngles(original);
  const JointAngles after = estimateJointAngles(rotated);
  for (const std::size_t index :
       {kWaistYaw, kLeftShoulderYaw, kRightShoulderYaw}) {
    ASSERT_TRUE(before.valid[index]);
    ASSERT_TRUE(after.valid[index]);
    expectNear(after.values[index], before.values[index]);
  }
}

TEST(EstimateJointAngles, ShoulderYawIsInvalidAtArmSingularities) {
  auto expect_left_yaw_only_invalid = [](const BodyPoints &points) {
    const JointAngles angles = estimateJointAngles(points);
    EXPECT_FALSE(angles.valid[kLeftShoulderYaw]);
    EXPECT_TRUE(angles.valid[kLeftShoulderPitch]);
    EXPECT_TRUE(angles.valid[kLeftShoulderRoll]);
    EXPECT_TRUE(angles.valid[kLeftElbowPitch]);
    EXPECT_TRUE(angles.valid[kRightShoulderYaw]);
  };

  BodyPoints straight = makePose();
  straight[kLeftWrist] = *straight[kLeftElbow] + Vec3{0.0, 0.0, -0.3};
  expect_left_yaw_only_invalid(straight);

  BodyPoints folded = makePose();
  folded[kLeftWrist] = *folded[kLeftElbow] + Vec3{0.0, 0.0, 0.3};
  expect_left_yaw_only_invalid(folded);

  BodyPoints overhead = makePose();
  overhead[kLeftElbow] = *overhead[kLeftShoulder] + Vec3{0.0, 0.0, 0.4};
  overhead[kLeftWrist] = *overhead[kLeftElbow] + Vec3{0.0, 0.3, 0.0};
  expect_left_yaw_only_invalid(overhead);

  BodyPoints sideways = makePose();
  sideways[kLeftElbow] = *sideways[kLeftShoulder] + Vec3{-0.4, 0.0, 0.0};
  sideways[kLeftWrist] = *sideways[kLeftElbow] + Vec3{0.0, 0.3, 0.0};
  expect_left_yaw_only_invalid(sideways);
}

TEST(BodyCapsules, WaistTwistDoesNotRotateIndependentTrackedLimbs) {
  const BodyPoints neutral = makePose();
  BodyPoints twisted = makePose(radians(55.0), radians(30.0), radians(-25.0));
  twisted[kLeftWrist] = *twisted[kLeftElbow] + Vec3{0.0, 0.0, -0.3};

  const auto neutral_capsules = buildBodyCapsules(neutral, BodyCapsuleConfig{});
  const auto twisted_capsules = buildBodyCapsules(twisted, BodyCapsuleConfig{});

  struct ExpectedCapsule {
    const char *name;
    int a;
    int b;
    double radius;
  };
  const std::array<ExpectedCapsule, 9> expected = {{
      {"torso", kPelvis, kNeck, 0.15},
      {"left_arm", kLeftElbow, kLeftWrist, 0.075},
      {"right_arm", kRightElbow, kRightWrist, 0.075},
      {"left_shoulder", kLeftShoulder, kLeftElbow, 0.075},
      {"right_shoulder", kRightShoulder, kRightElbow, 0.075},
      {"left_thigh", kLeftHip, kLeftKnee, 0.0975},
      {"right_thigh", kRightHip, kRightKnee, 0.0975},
      {"left_shin", kLeftKnee, kLeftAnkle, 0.0975},
      {"right_shin", kRightKnee, kRightAnkle, 0.0975},
  }};
  ASSERT_EQ(twisted_capsules.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(twisted_capsules[i].name, expected[i].name);
    expectVecNear(twisted_capsules[i].a, *twisted[expected[i].a]);
    expectVecNear(twisted_capsules[i].b, *twisted[expected[i].b]);
    EXPECT_NEAR(twisted_capsules[i].radius, expected[i].radius, 1e-12);
  }

  for (const std::string &name :
       {"left_thigh", "right_thigh", "left_shin", "right_shin"}) {
    const CapsuleData *before = findCapsule(neutral_capsules, name);
    const CapsuleData *after = findCapsule(twisted_capsules, name);
    ASSERT_NE(before, nullptr);
    ASSERT_NE(after, nullptr);
    expectVecNear(after->a, before->a);
    expectVecNear(after->b, before->b);
  }

  const JointAngles angles = estimateJointAngles(twisted);
  EXPECT_FALSE(angles.valid[kLeftShoulderYaw]);
  const CapsuleData *left_upper =
      findCapsule(twisted_capsules, "left_shoulder");
  const CapsuleData *left_lower = findCapsule(twisted_capsules, "left_arm");
  ASSERT_NE(left_upper, nullptr);
  ASSERT_NE(left_lower, nullptr);
  expectVecNear(left_upper->a, *twisted[kLeftShoulder]);
  expectVecNear(left_upper->b, *twisted[kLeftElbow]);
  expectVecNear(left_lower->a, *twisted[kLeftElbow]);
  expectVecNear(left_lower->b, *twisted[kLeftWrist]);
}

TEST(CapsuleAnatomyFilter, HoldsPreviousSegmentForImpossibleLength) {
  CapsuleAnatomyFilter filter;
  const CapsuleData valid{"left_arm", {0.0, 0.0, 1.0}, {0.26, 0.0, 1.0}, 0.075};
  const CapsuleData impossible{
      "left_arm", {0.0, 0.0, 1.0}, {3.04, 0.0, 1.0}, 0.075};

  const auto first = filter.update({valid});
  ASSERT_EQ(first.size(), 1U);
  expectVecNear(first.front().a, valid.a);
  expectVecNear(first.front().b, valid.b);

  const auto held = filter.update({impossible});
  ASSERT_EQ(held.size(), 1U);
  expectVecNear(held.front().a, valid.a);
  expectVecNear(held.front().b, valid.b);
}

TEST(CapsuleAnatomyFilter, ResetDoesNotHoldThePreviousTrack) {
  CapsuleAnatomyFilter filter;
  const CapsuleData valid{"left_arm", {0.0, 0.0, 1.0}, {0.26, 0.0, 1.0}, 0.075};
  const CapsuleData impossible{
      "left_arm", {0.0, 0.0, 1.0}, {3.04, 0.0, 1.0}, 0.075};

  ASSERT_EQ(filter.update({valid}).size(), 1U);
  filter.reset();
  EXPECT_TRUE(filter.update({impossible}).empty());
}

TEST(CapsuleAnatomyFilter, HoldsSuddenBoneLengthChangeWithinAbsoluteBounds) {
  CapsuleAnatomyFilter filter(0.25);
  const CapsuleData valid{
      "right_thigh", {0.0, 0.0, 1.0}, {0.4, 0.0, 1.0}, 0.0975};
  const CapsuleData stretched{
      "right_thigh", {0.0, 0.0, 1.0}, {0.7, 0.0, 1.0}, 0.0975};

  ASSERT_EQ(filter.update({valid}).size(), 1U);
  const auto held = filter.update({stretched});
  ASSERT_EQ(held.size(), 1U);
  expectVecNear(held.front().a, valid.a);
  expectVecNear(held.front().b, valid.b);
}

TEST(MappingObservation, CompletelyInvalidSkeletonDoesNotRefreshState) {
  JointAngles angles;
  std::vector<CapsuleData> capsules;
  EXPECT_FALSE(hasUsableObservation(angles, capsules));

  angles.valid[kWaistYaw] = true;
  angles.values[kWaistYaw] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(hasUsableObservation(angles, capsules));

  angles.values[kWaistYaw] = 0.2;
  EXPECT_TRUE(hasUsableObservation(angles, capsules));

  angles.valid[kWaistYaw] = false;
  capsules.push_back({"tracked", Vec3{}, Vec3{}, 0.1});
  EXPECT_TRUE(hasUsableObservation(angles, capsules));
}

TEST(EMAJumpFilter, ResetDropsStateFromThePreviousTrack) {
  EMAJumpFilter filter(0.30, 0.60, 3);
  const Vec3 previous_track{0.0, 0.0, 1.0};
  const Vec3 next_track{4.0, 0.0, 1.0};

  expectVecNear(filter.update(previous_track), previous_track);
  expectVecNear(filter.update(next_track), previous_track);

  filter.reset();
  expectVecNear(filter.update(next_track), next_track);
}

TEST(PointFilterResetPolicy, ResetsOnIdentityChangeOrObservationGap) {
  constexpr int64_t kSecond = 1'000'000'000LL;
  const std::optional<int> body_id = 7;
  const std::optional<int64_t> previous_time = 10 * kSecond;

  EXPECT_FALSE(shouldResetPointFilters(std::nullopt, std::nullopt, 7,
                                       10 * kSecond, 0.5));
  EXPECT_FALSE(shouldResetPointFilters(body_id, previous_time, 7,
                                       10 * kSecond + 500'000'000LL, 0.5));
  EXPECT_TRUE(shouldResetPointFilters(body_id, previous_time, 8,
                                      10 * kSecond + 1, 0.5));
  EXPECT_TRUE(shouldResetPointFilters(body_id, previous_time, 7,
                                      10 * kSecond + 500'000'001LL, 0.5));
  EXPECT_TRUE(
      shouldResetPointFilters(body_id, previous_time, 7, 9 * kSecond, 0.5));
}

TEST(AngleFilter, CircularJointsCrossPiAndInvalidValuesHold) {
  AngleFilter filter(0.5, 1.0e9);
  JointAngles first;
  for (const std::size_t index :
       {kWaistYaw, kLeftShoulderYaw, kRightShoulderYaw}) {
    first.values[index] = radians(179.0);
    first.valid[index] = true;
  }
  const JointAngles first_output = filter.update(first, 0.1);

  JointAngles second;
  for (const std::size_t index :
       {kWaistYaw, kLeftShoulderYaw, kRightShoulderYaw}) {
    second.values[index] = radians(-179.0);
    second.valid[index] = true;
  }
  const JointAngles second_output = filter.update(second, 0.1);
  for (const std::size_t index :
       {kWaistYaw, kLeftShoulderYaw, kRightShoulderYaw}) {
    ASSERT_TRUE(first_output.valid[index]);
    ASSERT_TRUE(second_output.valid[index]);
    EXPECT_NEAR(std::abs(second_output.values[index]), kPi, 1e-9);
  }

  JointAngles missing;
  const JointAngles held = filter.update(missing, 0.1);
  for (const std::size_t index :
       {kWaistYaw, kLeftShoulderYaw, kRightShoulderYaw}) {
    EXPECT_FALSE(held.valid[index]);
    expectNear(held.values[index], second_output.values[index]);
  }
}

TEST(AngleUtilities, WrapAngleAndCircularJointIndicesMatchContract) {
  EXPECT_NEAR(wrapAngle(2.0 * kPi + 0.25), 0.25, 1e-12);
  EXPECT_NEAR(wrapAngle(-2.0 * kPi - 0.25), -0.25, 1e-12);
  EXPECT_NEAR(std::abs(wrapAngle(kPi)), kPi, 1e-12);

  for (std::size_t index = 0; index < kNumJoints; ++index) {
    EXPECT_EQ(isCircularJoint(index), index == kWaistYaw ||
                                          index == kLeftShoulderYaw ||
                                          index == kRightShoulderYaw);
  }
}

class HumanMappingNodeDefaultsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      const char *argv[] = {"test_human_mapping", nullptr};
      rclcpp::init(1, argv);
    }
  }

  static void TearDownTestSuite() {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

TEST_F(HumanMappingNodeDefaultsTest, MatchesG1ManualCommandContract) {
  const auto node = std::make_shared<HumanMappingNode>();

  EXPECT_EQ(node->get_parameter("joint_command_topic").as_string(),
            "/human/joint_commands");

  const std::vector<std::string> expected_names = {
      "waist_yaw_joint",           "waist_roll_joint",
      "waist_pitch_joint",         "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
      "left_elbow_joint",          "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
      "right_elbow_joint",
  };
  EXPECT_EQ(node->get_parameter("joint_names").as_string_array(),
            expected_names);

  const std::vector<double> expected_home = {0.0,  0.0,  0.0,   0.35, 0.18, 0.0,
                                             0.87, 0.35, -0.18, 0.0,  0.87};
  const std::vector<double> expected_min = {-2.094, -0.416, -0.416, -2.471,
                                            -1.271, -2.094, -0.838, -2.471,
                                            -1.801, -2.094, -0.838};
  const std::vector<double> expected_max = {2.094, 0.416, 0.416, 2.136,
                                            1.801, 2.094, 1.676, 2.136,
                                            1.271, 2.094, 1.676};
  const std::vector<double> expected_signs = {1.0,  -1.0, 1.0,  -1.0, 1.0, 1.0,
                                              -1.0, -1.0, -1.0, 1.0,  -1.0};
  const std::vector<double> expected_gains(kNumJoints, 1.0);
  const std::vector<double> expected_bias(kNumJoints, 0.0);

  EXPECT_EQ(node->get_parameter("q_home").as_double_array(), expected_home);
  EXPECT_EQ(node->get_parameter("q_min").as_double_array(), expected_min);
  EXPECT_EQ(node->get_parameter("q_max").as_double_array(), expected_max);
  EXPECT_EQ(node->get_parameter("signs").as_double_array(), expected_signs);
  EXPECT_EQ(node->get_parameter("gains").as_double_array(), expected_gains);
  EXPECT_EQ(node->get_parameter("bias").as_double_array(), expected_bias);
  EXPECT_DOUBLE_EQ(node->get_parameter("publish_rate_hz").as_double(), 30.0);
  EXPECT_DOUBLE_EQ(
      node->get_parameter("point_filter_reset_gap_sec").as_double(), 0.5);
  EXPECT_DOUBLE_EQ(
      node->get_parameter("capsule_max_length_change_fraction").as_double(),
      0.5);
  EXPECT_FALSE(node->get_parameter("require_body_38").as_bool());
  EXPECT_EQ(node->get_parameter("collider_topic").as_string(),
            "/human/body_colliders");
}

} // namespace
} // namespace human_mapping

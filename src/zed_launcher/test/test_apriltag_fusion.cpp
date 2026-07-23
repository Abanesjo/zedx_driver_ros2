#include "zed_launcher/apriltag_fusion_node.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

namespace zed_launcher {

class ApriltagFusionNodeTestPeer {
public:
  using Observation = ApriltagFusionNode::Observation;
  using FusedTagEstimate = ApriltagFusionNode::FusedTagEstimate;

  static size_t frontSlot() { return ApriltagFusionNode::kFrontTagSlot; }

  static size_t backSlot() { return ApriltagFusionNode::kBackTagSlot; }

  static tf2::Transform single(const ApriltagFusionNode &node, size_t slot,
                               const tf2::Transform &fusion_from_tag) {
    return node.tagFrameFromSingleTag(slot, fusion_from_tag);
  }

  static tf2::Transform both(const ApriltagFusionNode &node,
                             const tf2::Transform &fusion_from_front,
                             const tf2::Transform &fusion_from_back) {
    return node.tagFrameFromBothTags(fusion_from_front, fusion_from_back);
  }

  static void updateSeparation(ApriltagFusionNode &node,
                               const FusedTagEstimate &front,
                               const FusedTagEstimate &back) {
    node.updateTagSeparation(front, back);
  }

  static double learnedSeparation(const ApriltagFusionNode &node) {
    return node.learned_tag_separation_m_;
  }

  static void setObservation(ApriltagFusionNode &node, size_t camera_index,
                             size_t tag_slot, Observation observation) {
    auto &camera = *node.cameras_.at(camera_index);
    std::lock_guard<std::mutex> lock(camera.mutex);
    camera.latest_observations.at(tag_slot) = std::move(observation);
  }

  static std::optional<FusedTagEstimate>
  fuseTag(const ApriltagFusionNode &node, size_t tag_slot,
          const rclcpp::Time &current_time) {
    return node.fuseTag(tag_slot, current_time);
  }

  static void setLastEstimate(ApriltagFusionNode &node, size_t tag_slot,
                              FusedTagEstimate estimate) {
    node.last_tag_estimates_.at(tag_slot) = std::move(estimate);
  }

  static std::optional<size_t> fallbackSlot(const ApriltagFusionNode &node) {
    return node.selectFallbackTagSlot();
  }

  static void fuseAndPublish(ApriltagFusionNode &node) {
    node.fuseAndPublish();
  }

  static rclcpp::Time lastSmoothedReceipt(const ApriltagFusionNode &node) {
    return node.last_smoothed_receipt_time_;
  }

  static bool drawDebugMarkerOutline(cv::Mat &bgr,
                                     const std::vector<cv::Point2f> &corners) {
    return ApriltagFusionNode::drawDebugMarkerOutline(bgr, corners);
  }

  static std::optional<sensor_msgs::msg::Image>
  processImage(ApriltagFusionNode &node, size_t camera_index,
               const sensor_msgs::msg::Image &source,
               const sensor_msgs::msg::Image *debug_base, bool render_debug) {
    return node.processImage(camera_index, source, debug_base, render_debug);
  }
};

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTolerance = 1e-9;

tf2::Quaternion rotation(double roll, double pitch, double yaw) {
  tf2::Quaternion result;
  result.setRPY(roll, pitch, yaw);
  result.normalize();
  return result;
}

void expectVectorNear(const tf2::Vector3 &actual, const tf2::Vector3 &expected,
                      double tolerance = kTolerance) {
  EXPECT_NEAR(actual.x(), expected.x(), tolerance);
  EXPECT_NEAR(actual.y(), expected.y(), tolerance);
  EXPECT_NEAR(actual.z(), expected.z(), tolerance);
}

void expectRotationNear(const tf2::Quaternion &actual_value,
                        const tf2::Quaternion &expected_value,
                        double tolerance = kTolerance) {
  auto actual = actual_value;
  auto expected = expected_value;
  actual.normalize();
  expected.normalize();
  EXPECT_NEAR(std::abs(actual.dot(expected)), 1.0, tolerance);
}

ApriltagFusionNodeTestPeer::Observation
makeObservation(const tf2::Transform &transform, const rclcpp::Time &stamp,
                double quality, uint64_t sequence) {
  ApriltagFusionNodeTestPeer::Observation result;
  result.fusion_from_tag = transform;
  result.stamp = stamp;
  result.receipt_time = stamp;
  result.quality = quality;
  result.marker_area_px2 = quality;
  result.reprojection_rmse_px = 1.0;
  result.sequence = sequence;
  return result;
}

ApriltagFusionNodeTestPeer::FusedTagEstimate
makeEstimate(const tf2::Transform &transform, const rclcpp::Time &stamp,
             double quality, uint64_t sequence) {
  ApriltagFusionNodeTestPeer::FusedTagEstimate result;
  result.observation = makeObservation(transform, stamp, quality, sequence);
  result.source_sequences = {sequence};
  return result;
}

class ApriltagFusionTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      int argc = 0;
      char **argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  static void TearDownTestSuite() {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  static std::shared_ptr<ApriltagFusionNode> makeNode() {
    const std::vector<rclcpp::Parameter> parameters{
        rclcpp::Parameter(
            "camera_names",
            std::vector<std::string>{"camera_0", "camera_1", "camera_2"}),
        rclcpp::Parameter("publish_tf", false),
        rclcpp::Parameter("publish_fusion_pose", true),
        rclcpp::Parameter("smoothing_time_constant_sec", 0.0)};
    rclcpp::NodeOptions options;
    options.parameter_overrides(parameters);
    return std::make_shared<ApriltagFusionNode>(options);
  }
};

TEST_F(ApriltagFusionTest, DefaultInterfaceMatchesDualTagContract) {
  auto node = std::make_shared<ApriltagFusionNode>();
  EXPECT_EQ(node->get_parameter("front_tag_id").as_int(), 0);
  EXPECT_EQ(node->get_parameter("back_tag_id").as_int(), 1);
  EXPECT_DOUBLE_EQ(node->get_parameter("front_tag_size_m").as_double(), 0.12);
  EXPECT_DOUBLE_EQ(node->get_parameter("back_tag_size_m").as_double(), 0.12);
  EXPECT_EQ(node->get_parameter("tag_frame_id").as_string(), "tag_frame");
  EXPECT_EQ(node->get_parameter("pose_topic").as_string(),
            "/fusion_world_pose_in_tag_frame");
  EXPECT_DOUBLE_EQ(
      node->get_parameter("initial_tag_frame_offset_m").as_double(), 0.03);
  EXPECT_TRUE(node->get_parameter("learn_tag_separation").as_bool());
  EXPECT_FALSE(node->get_parameter("publish_debug_images").as_bool());
  EXPECT_EQ(node->get_parameter("camera_names").as_string_array().size(), 3U);
}

TEST(ApriltagFusionOverlayTest, DrawsFloatingPointDetectorCorners) {
  cv::Mat image = cv::Mat::zeros(64, 64, CV_8UC3);
  const std::vector<cv::Point2f> corners{
      cv::Point2f(8.25F, 8.75F), cv::Point2f(48.25F, 8.75F),
      cv::Point2f(48.25F, 48.75F), cv::Point2f(8.25F, 48.75F)};

  bool rendered = false;
  EXPECT_NO_THROW(rendered = ApriltagFusionNodeTestPeer::drawDebugMarkerOutline(
                      image, corners));
  EXPECT_TRUE(rendered);
  EXPECT_GT(cv::countNonZero(image.reshape(1)), 0);
}

TEST_F(ApriltagFusionTest, CombinedOverlayPreservesSkeletonDebugBase) {
  const std::vector<rclcpp::Parameter> parameters{
      rclcpp::Parameter("camera_names", std::vector<std::string>{"camera_0"}),
      rclcpp::Parameter("publish_tf", false),
      rclcpp::Parameter("publish_fusion_pose", true),
      rclcpp::Parameter("max_detection_rate_hz", 0.0)};
  rclcpp::NodeOptions options;
  options.parameter_overrides(parameters);
  auto node = std::make_shared<ApriltagFusionNode>(
      options, ApriltagFusionNode::ImageInputMode::Direct);

  sensor_msgs::msg::Image source;
  source.header.stamp = node->now();
  source.header.frame_id = "camera_0_left_camera_optical_frame";
  source.height = 32;
  source.width = 32;
  source.encoding = "bgr8";
  source.step = source.width * 3;
  source.data.assign(static_cast<size_t>(source.height) * source.step, 0U);

  auto skeleton_base = source;
  const size_t marked_pixel =
      static_cast<size_t>(12) * skeleton_base.step + static_cast<size_t>(9) * 3;
  skeleton_base.data.at(marked_pixel) = 41U;
  skeleton_base.data.at(marked_pixel + 1) = 199U;
  skeleton_base.data.at(marked_pixel + 2) = 83U;

  const auto combined = ApriltagFusionNodeTestPeer::processImage(
      *node, 0, source, &skeleton_base, true);

  ASSERT_TRUE(combined);
  EXPECT_EQ(combined->header.frame_id, source.header.frame_id);
  EXPECT_EQ(combined->header.stamp.sec, source.header.stamp.sec);
  EXPECT_EQ(combined->header.stamp.nanosec, source.header.stamp.nanosec);
  EXPECT_EQ(combined->encoding, "bgr8");
  EXPECT_EQ(combined->height, source.height);
  EXPECT_EQ(combined->width, source.width);
  EXPECT_EQ(combined->step, source.step);
  EXPECT_EQ(combined->data, skeleton_base.data);
}

TEST_F(ApriltagFusionTest, SingleTagsProduceSameRobotCenterAndOrientation) {
  const auto node = makeNode();
  const tf2::Transform front(tf2::Quaternion::getIdentity(),
                             tf2::Vector3(0.0, 0.0, 0.0));
  const tf2::Transform back(rotation(0.0, kPi, 0.0),
                            tf2::Vector3(0.0, 0.0, 0.06));

  const auto from_front = ApriltagFusionNodeTestPeer::single(
      *node, ApriltagFusionNodeTestPeer::frontSlot(), front);
  const auto from_back = ApriltagFusionNodeTestPeer::single(
      *node, ApriltagFusionNodeTestPeer::backSlot(), back);

  expectVectorNear(from_front.getOrigin(), tf2::Vector3(0.0, 0.0, 0.03));
  expectVectorNear(from_back.getOrigin(), tf2::Vector3(0.0, 0.0, 0.03));
  expectRotationNear(from_front.getRotation(), rotation(0.0, kPi, 0.0));
  expectRotationNear(from_back.getRotation(), rotation(0.0, kPi, 0.0));
}

TEST_F(ApriltagFusionTest, BothTagsAlwaysUseMidpointAndBackOrientation) {
  const auto node = makeNode();
  const tf2::Transform front(rotation(0.1, -0.2, 0.3),
                             tf2::Vector3(1.0, 2.0, 3.0));
  const tf2::Transform back(rotation(-0.4, 0.5, -0.6),
                            tf2::Vector3(9.0, 8.0, 7.0));

  const auto result = ApriltagFusionNodeTestPeer::both(*node, front, back);
  expectVectorNear(result.getOrigin(), tf2::Vector3(5.0, 5.0, 5.0));
  expectRotationNear(result.getRotation(), back.getRotation());
}

TEST_F(ApriltagFusionTest, SeparationLearningClipsAndUsesEachPairOnce) {
  auto node = makeNode();
  const auto stamp = node->now();
  const auto front = makeEstimate(tf2::Transform(tf2::Quaternion::getIdentity(),
                                                 tf2::Vector3(0.0, 0.0, 0.0)),
                                  stamp, 1.0, 10);
  auto back = makeEstimate(
      tf2::Transform(rotation(0.0, kPi, 0.0), tf2::Vector3(0.0, 0.0, 0.16)),
      stamp, 1.0, 11);

  EXPECT_DOUBLE_EQ(ApriltagFusionNodeTestPeer::learnedSeparation(*node), 0.06);
  ApriltagFusionNodeTestPeer::updateSeparation(*node, front, back);
  EXPECT_NEAR(ApriltagFusionNodeTestPeer::learnedSeparation(*node), 0.061,
              kTolerance);

  ApriltagFusionNodeTestPeer::updateSeparation(*node, front, back);
  EXPECT_NEAR(ApriltagFusionNodeTestPeer::learnedSeparation(*node), 0.061,
              kTolerance);

  back.source_sequences = {12};
  ApriltagFusionNodeTestPeer::updateSeparation(*node, front, back);
  EXPECT_NEAR(ApriltagFusionNodeTestPeer::learnedSeparation(*node), 0.062,
              kTolerance);

  back.observation.fusion_from_tag.setRotation(tf2::Quaternion::getIdentity());
  back.source_sequences = {13};
  ApriltagFusionNodeTestPeer::updateSeparation(*node, front, back);
  EXPECT_NEAR(ApriltagFusionNodeTestPeer::learnedSeparation(*node), 0.062,
              kTolerance);

  const auto reset_node = makeNode();
  EXPECT_DOUBLE_EQ(ApriltagFusionNodeTestPeer::learnedSeparation(*reset_node),
                   0.06);
}

TEST_F(ApriltagFusionTest, MultiCameraFusionPrefersLargestAgreeingSet) {
  auto node = makeNode();
  const auto stamp = node->now();
  ApriltagFusionNodeTestPeer::setObservation(
      *node, 0, ApriltagFusionNodeTestPeer::frontSlot(),
      makeObservation(tf2::Transform(tf2::Quaternion::getIdentity(),
                                     tf2::Vector3(0.0, 0.0, 0.0)),
                      stamp, 1.0, 20));
  ApriltagFusionNodeTestPeer::setObservation(
      *node, 1, ApriltagFusionNodeTestPeer::frontSlot(),
      makeObservation(tf2::Transform(tf2::Quaternion::getIdentity(),
                                     tf2::Vector3(0.1, 0.0, 0.0)),
                      stamp, 3.0, 21));
  ApriltagFusionNodeTestPeer::setObservation(
      *node, 2, ApriltagFusionNodeTestPeer::frontSlot(),
      makeObservation(tf2::Transform(tf2::Quaternion::getIdentity(),
                                     tf2::Vector3(1.0, 0.0, 0.0)),
                      stamp, 100.0, 22));

  const auto result = ApriltagFusionNodeTestPeer::fuseTag(
      *node, ApriltagFusionNodeTestPeer::frontSlot(), node->now());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->source_sequences, (std::vector<uint64_t>{20, 21}));
  EXPECT_NEAR(result->observation.fusion_from_tag.getOrigin().x(), 0.075,
              kTolerance);
}

TEST_F(ApriltagFusionTest, StaleFallbackUsesQualityThenRecency) {
  auto node = makeNode();
  const rclcpp::Time earlier(10, 0, RCL_ROS_TIME);
  const rclcpp::Time later(11, 0, RCL_ROS_TIME);
  ApriltagFusionNodeTestPeer::setLastEstimate(
      *node, ApriltagFusionNodeTestPeer::frontSlot(),
      makeEstimate(tf2::Transform::getIdentity(), later, 5.0, 30));
  ApriltagFusionNodeTestPeer::setLastEstimate(
      *node, ApriltagFusionNodeTestPeer::backSlot(),
      makeEstimate(tf2::Transform::getIdentity(), earlier, 6.0, 31));
  ASSERT_TRUE(ApriltagFusionNodeTestPeer::fallbackSlot(*node));
  EXPECT_EQ(*ApriltagFusionNodeTestPeer::fallbackSlot(*node),
            ApriltagFusionNodeTestPeer::backSlot());

  ApriltagFusionNodeTestPeer::setLastEstimate(
      *node, ApriltagFusionNodeTestPeer::backSlot(),
      makeEstimate(tf2::Transform::getIdentity(), later, 5.0, 32));
  ApriltagFusionNodeTestPeer::setLastEstimate(
      *node, ApriltagFusionNodeTestPeer::frontSlot(),
      makeEstimate(tf2::Transform::getIdentity(), earlier, 5.0, 33));
  EXPECT_EQ(*ApriltagFusionNodeTestPeer::fallbackSlot(*node),
            ApriltagFusionNodeTestPeer::backSlot());
}

TEST_F(ApriltagFusionTest, StaleFallbackPreservesRealObservationReceipt) {
  auto node = makeNode();
  const auto stale_receipt = node->now() - rclcpp::Duration::from_seconds(2.0);
  ApriltagFusionNodeTestPeer::setLastEstimate(
      *node, ApriltagFusionNodeTestPeer::frontSlot(),
      makeEstimate(tf2::Transform::getIdentity(), stale_receipt, 5.0, 40));

  ApriltagFusionNodeTestPeer::fuseAndPublish(*node);

  EXPECT_EQ(
      ApriltagFusionNodeTestPeer::lastSmoothedReceipt(*node).nanoseconds(),
      stale_receipt.nanoseconds());
}

} // namespace
} // namespace zed_launcher

#include "zed_launcher/apriltag_fusion_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

namespace zed_launcher {

class ApriltagFusionNodeTestPeer {
public:
  using Observation = ApriltagFusionNode::Observation;
  using FusedTagEstimate = ApriltagFusionNode::FusedTagEstimate;
  using PoseEstimate = ApriltagFusionNode::PoseEstimate;

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

  static void updateTagTransform(ApriltagFusionNode &node,
                                 const FusedTagEstimate &front,
                                 const FusedTagEstimate &back) {
    node.updateTagTransform(front, back);
  }

  static double learnedSeparation(const ApriltagFusionNode &node) {
    return node.learned_tag_separation_m_;
  }

  static tf2::Transform activeFrontFromBackTag(const ApriltagFusionNode &node) {
    return node.activeFrontFromBackTag();
  }

  static bool tagTransformCalibrated(const ApriltagFusionNode &node) {
    return node.tag_transform_calibrated_;
  }

  static size_t
  tagTransformBootstrapSampleCount(const ApriltagFusionNode &node) {
    return node.tag_transform_bootstrap_samples_.size();
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

  static tf2::Transform smoothedTransform(const ApriltagFusionNode &node) {
    return node.smoothed_fusion_from_tag_frame_;
  }

  static tf2::Transform smooth(ApriltagFusionNode &node,
                               const tf2::Transform &transform,
                               const rclcpp::Time &receipt_time) {
    return node.smoothTransform(transform, receipt_time);
  }

  static bool drawDebugMarkerOutline(cv::Mat &bgr,
                                     const std::vector<cv::Point2f> &corners) {
    return ApriltagFusionNode::drawDebugMarkerOutline(bgr, corners);
  }

  static std::optional<sensor_msgs::msg::Image>
  processImage(ApriltagFusionNode &node, size_t camera_index,
               const sensor_msgs::msg::Image &source,
               const sensor_msgs::msg::Image *debug_base, bool render_debug) {
    return node.processImage(camera_index, source, DepthFrameProvider{},
                             debug_base, render_debug);
  }

  static std::optional<PoseEstimate> estimateTagInCameraFrame(
      const ApriltagFusionNode &node, const std::vector<cv::Point2f> &corners,
      const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs,
      double tag_size_m, const DepthFrameView *depth_frame, size_t image_width,
      size_t image_height) {
    return node.estimateTagInCameraFrame(
        corners, camera_matrix, distortion_coeffs, tag_size_m, depth_frame,
        image_width, image_height);
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

double rotationErrorDegrees(const tf2::Quaternion &actual_value,
                            const tf2::Quaternion &expected_value) {
  auto actual = actual_value;
  auto expected = expected_value;
  actual.normalize();
  expected.normalize();
  const double dot = std::clamp(std::abs(actual.dot(expected)), 0.0, 1.0);
  return 2.0 * std::acos(dot) * 180.0 / kPi;
}

void expectTransformNear(const tf2::Transform &actual,
                         const tf2::Transform &expected,
                         double translation_tolerance_m,
                         double rotation_tolerance_deg) {
  EXPECT_LE(actual.getOrigin().distance(expected.getOrigin()),
            translation_tolerance_m);
  EXPECT_LE(rotationErrorDegrees(actual.getRotation(), expected.getRotation()),
            rotation_tolerance_deg);
}

tf2::Transform transformFromCvPose(const cv::Mat &rvec, const cv::Mat &tvec) {
  cv::Mat rotation_matrix;
  cv::Rodrigues(rvec, rotation_matrix);
  tf2::Matrix3x3 basis(
      rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
      rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
      rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
      rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
      rotation_matrix.at<double>(2, 2));
  return tf2::Transform(basis, tf2::Vector3(tvec.at<double>(0, 0),
                                            tvec.at<double>(1, 0),
                                            tvec.at<double>(2, 0)));
}

struct SyntheticDepthScene {
  static constexpr size_t kWidth = 320;
  static constexpr size_t kHeight = 240;
  static constexpr double kTagSizeM = 0.12;

  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 600.0, 0.0, 160.0, 0.0,
                           600.0, 120.0, 0.0, 0.0, 1.0);
  cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
  cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.10, -0.08, 0.03);
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.02, -0.015, 1.20);
  std::vector<cv::Point2f> corners;
  std::vector<float> depth;

  SyntheticDepthScene() : depth(kWidth * kHeight) {
    const float half_size = static_cast<float>(0.5 * kTagSizeM);
    const std::vector<cv::Point3f> object_points{
        cv::Point3f(-half_size, -half_size, 0.0F),
        cv::Point3f(half_size, -half_size, 0.0F),
        cv::Point3f(half_size, half_size, 0.0F),
        cv::Point3f(-half_size, half_size, 0.0F)};
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion,
                      corners);

    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    const cv::Vec3d normal(rotation_matrix.at<double>(0, 2),
                           rotation_matrix.at<double>(1, 2),
                           rotation_matrix.at<double>(2, 2));
    const cv::Vec3d translation(tvec.at<double>(0, 0), tvec.at<double>(1, 0),
                                tvec.at<double>(2, 0));
    const double numerator = normal.dot(translation);

    for (size_t y = 0; y < kHeight; ++y) {
      for (size_t x = 0; x < kWidth; ++x) {
        const cv::Vec3d ray(
            (static_cast<double>(x) - camera_matrix.at<double>(0, 2)) /
                camera_matrix.at<double>(0, 0),
            (static_cast<double>(y) - camera_matrix.at<double>(1, 2)) /
                camera_matrix.at<double>(1, 1),
            1.0);
        const double ideal_depth = numerator / normal.dot(ray);
        const size_t index = y * kWidth + x;
        const double noise_m =
            0.0006 * std::sin(static_cast<double>(x * 17 + y * 29));
        depth[index] = static_cast<float>(ideal_depth + noise_m);
        if (index % 97 == 0) {
          depth[index] = std::numeric_limits<float>::quiet_NaN();
        } else if (index % 89 == 0) {
          depth[index] += 0.06F;
        }
      }
    }
  }

  DepthFrameView view() const {
    return DepthFrameView{depth.data(), kWidth, kHeight,
                          kWidth * sizeof(float)};
  }

  tf2::Transform expectedCameraFromTag() const {
    return transformFromCvPose(rvec, tvec);
  }
};

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

void expectPnpFallback(ApriltagFusionNode &node,
                       const SyntheticDepthScene &scene,
                       const std::vector<float> &depth) {
  const DepthFrameView depth_view{depth.data(), SyntheticDepthScene::kWidth,
                                  SyntheticDepthScene::kHeight,
                                  SyntheticDepthScene::kWidth * sizeof(float)};
  const auto pnp = ApriltagFusionNodeTestPeer::estimateTagInCameraFrame(
      node, scene.corners, scene.camera_matrix, scene.distortion,
      SyntheticDepthScene::kTagSizeM, nullptr, SyntheticDepthScene::kWidth,
      SyntheticDepthScene::kHeight);
  const auto result = ApriltagFusionNodeTestPeer::estimateTagInCameraFrame(
      node, scene.corners, scene.camera_matrix, scene.distortion,
      SyntheticDepthScene::kTagSizeM, &depth_view, SyntheticDepthScene::kWidth,
      SyntheticDepthScene::kHeight);

  ASSERT_TRUE(pnp);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->used_depth);
  expectTransformNear(result->camera_from_tag, pnp->camera_from_tag, 1e-9,
                      1e-7);
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

  static std::shared_ptr<ApriltagFusionNode>
  makeNode(std::vector<rclcpp::Parameter> additional_parameters = {}) {
    std::vector<rclcpp::Parameter> parameters{
        rclcpp::Parameter(
            "camera_names",
            std::vector<std::string>{"camera_0", "camera_1", "camera_2"}),
        rclcpp::Parameter("publish_tf", false),
        rclcpp::Parameter("publish_fusion_pose", true),
        rclcpp::Parameter("smoothing_time_constant_sec", 0.0)};
    parameters.insert(parameters.end(), additional_parameters.begin(),
                      additional_parameters.end());
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
  EXPECT_TRUE(node->get_parameter("use_depth").as_bool());
  EXPECT_TRUE(node->get_parameter("learn_tag_transform").as_bool());
  EXPECT_DOUBLE_EQ(
      node->get_parameter("tag_transform_bootstrap_duration_sec").as_double(),
      2.5);
  EXPECT_EQ(node->get_parameter("tag_transform_bootstrap_min_samples").as_int(),
            30);
  EXPECT_FALSE(node->get_parameter("publish_debug_images").as_bool());
  EXPECT_EQ(node->get_parameter("camera_names").as_string_array().size(), 3U);
}

TEST_F(ApriltagFusionTest,
       SmoothingUsesReceiptTimeWhenImageTimestampsRepeatOrReorder) {
  auto node =
      makeNode({rclcpp::Parameter("smoothing_time_constant_sec", 0.10)});
  const auto clock_type = node->get_clock()->get_clock_type();
  const auto identity = tf2::Transform::getIdentity();
  const tf2::Transform target(rotation(0.0, 0.0, 0.5),
                              tf2::Vector3(1.0, 0.0, 0.0));

  ApriltagFusionNodeTestPeer::smooth(*node, identity,
                                     rclcpp::Time(1, 0, clock_type));
  const auto first = ApriltagFusionNodeTestPeer::smooth(
      *node, target, rclcpp::Time(1, 10000000, clock_type));
  const auto second = ApriltagFusionNodeTestPeer::smooth(
      *node, target, rclcpp::Time(1, 20000000, clock_type));

  const double first_alpha = 1.0 - std::exp(-0.01 / 0.10);
  const double second_alpha = 1.0 - std::exp(-0.02 / 0.10);
  EXPECT_NEAR(first.getOrigin().x(), first_alpha, 1e-9);
  EXPECT_NEAR(second.getOrigin().x(), second_alpha, 1e-9);
  EXPECT_GT(second.getOrigin().x(), first.getOrigin().x());
  EXPECT_LT(second.getOrigin().x(), target.getOrigin().x());
  EXPECT_LT(rotationErrorDegrees(second.getRotation(), identity.getRotation()),
            0.5 * 180.0 / kPi);
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

TEST_F(ApriltagFusionTest,
       DepthPlaneRecoversPoseWithNoiseOutliersAndInvalidSamples) {
  auto node =
      makeNode({rclcpp::Parameter("camera_frame_convention", "ros_optical")});
  const SyntheticDepthScene scene;
  const auto depth_view = scene.view();

  const auto result = ApriltagFusionNodeTestPeer::estimateTagInCameraFrame(
      *node, scene.corners, scene.camera_matrix, scene.distortion,
      SyntheticDepthScene::kTagSizeM, &depth_view, SyntheticDepthScene::kWidth,
      SyntheticDepthScene::kHeight);

  ASSERT_TRUE(result);
  EXPECT_TRUE(result->used_depth);
  EXPECT_GE(result->depth_valid_samples, 25U);
  EXPECT_GE(result->depth_inlier_samples, 25U);
  EXPECT_LE(result->depth_plane_rmse_m, 0.010);
  expectTransformNear(result->camera_from_tag, scene.expectedCameraFromTag(),
                      0.005, 1.0);
}

TEST_F(ApriltagFusionTest, InvalidDepthFallsBackToPnpWithoutDroppingTag) {
  auto node =
      makeNode({rclcpp::Parameter("camera_frame_convention", "ros_optical")});
  const SyntheticDepthScene scene;

  auto sparse_depth = std::vector<float>(
      scene.depth.size(), std::numeric_limits<float>::quiet_NaN());
  const int center_x = static_cast<int>(
      std::lround(0.25 * (scene.corners[0].x + scene.corners[1].x +
                          scene.corners[2].x + scene.corners[3].x)));
  const int center_y = static_cast<int>(
      std::lround(0.25 * (scene.corners[0].y + scene.corners[1].y +
                          scene.corners[2].y + scene.corners[3].y)));
  for (int y = center_y - 1; y <= center_y + 1; ++y) {
    for (int x = center_x - 1; x <= center_x + 1; ++x) {
      const size_t index =
          static_cast<size_t>(y) * SyntheticDepthScene::kWidth +
          static_cast<size_t>(x);
      sparse_depth[index] = scene.depth[index];
    }
  }
  expectPnpFallback(*node, scene, sparse_depth);

  auto poor_coverage_depth = std::vector<float>(
      scene.depth.size(), std::numeric_limits<float>::quiet_NaN());
  for (int y = center_y - 3; y < center_y + 3; ++y) {
    for (int x = center_x - 3; x < center_x + 3; ++x) {
      const size_t index =
          static_cast<size_t>(y) * SyntheticDepthScene::kWidth +
          static_cast<size_t>(x);
      poor_coverage_depth[index] = scene.depth[index];
    }
  }
  expectPnpFallback(*node, scene, poor_coverage_depth);

  auto background_depth = scene.depth;
  for (auto &value : background_depth) {
    if (std::isfinite(value)) {
      value += 0.05F;
    }
  }
  expectPnpFallback(*node, scene, background_depth);

  auto wrong_depth = scene.depth;
  for (auto &value : wrong_depth) {
    if (std::isfinite(value)) {
      value += 0.25F;
    }
  }
  expectPnpFallback(*node, scene, wrong_depth);
}

TEST_F(ApriltagFusionTest,
       DepthBlendConvergesContinuouslyToPnpAtSafetyBoundary) {
  auto node =
      makeNode({rclcpp::Parameter("camera_frame_convention", "ros_optical"),
                rclcpp::Parameter("depth_max_pnp_translation_delta_m", 0.04)});
  const SyntheticDepthScene scene;
  const auto pnp = ApriltagFusionNodeTestPeer::estimateTagInCameraFrame(
      *node, scene.corners, scene.camera_matrix, scene.distortion,
      SyntheticDepthScene::kTagSizeM, nullptr, SyntheticDepthScene::kWidth,
      SyntheticDepthScene::kHeight);
  ASSERT_TRUE(pnp);

  auto inside_depth = scene.depth;
  for (auto &value : inside_depth) {
    if (std::isfinite(value)) {
      value += 0.038F;
    }
  }
  const DepthFrameView inside_view{inside_depth.data(),
                                   SyntheticDepthScene::kWidth,
                                   SyntheticDepthScene::kHeight,
                                   SyntheticDepthScene::kWidth * sizeof(float)};
  const auto inside = ApriltagFusionNodeTestPeer::estimateTagInCameraFrame(
      *node, scene.corners, scene.camera_matrix, scene.distortion,
      SyntheticDepthScene::kTagSizeM, &inside_view, SyntheticDepthScene::kWidth,
      SyntheticDepthScene::kHeight);
  ASSERT_TRUE(inside);
  EXPECT_TRUE(inside->used_depth);
  EXPECT_GT(inside->depth_blend_weight, 0.0);
  EXPECT_LT(inside->depth_blend_weight, 0.1);

  auto outside_depth = scene.depth;
  for (auto &value : outside_depth) {
    if (std::isfinite(value)) {
      value += 0.042F;
    }
  }
  const DepthFrameView outside_view{
      outside_depth.data(), SyntheticDepthScene::kWidth,
      SyntheticDepthScene::kHeight,
      SyntheticDepthScene::kWidth * sizeof(float)};
  const auto outside = ApriltagFusionNodeTestPeer::estimateTagInCameraFrame(
      *node, scene.corners, scene.camera_matrix, scene.distortion,
      SyntheticDepthScene::kTagSizeM, &outside_view,
      SyntheticDepthScene::kWidth, SyntheticDepthScene::kHeight);
  ASSERT_TRUE(outside);
  EXPECT_FALSE(outside->used_depth);

  EXPECT_LT(inside->camera_from_tag.getOrigin().distance(
                pnp->camera_from_tag.getOrigin()),
            0.002);
  expectTransformNear(outside->camera_from_tag, pnp->camera_from_tag, 1e-9,
                      1e-7);
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

TEST_F(ApriltagFusionTest, BothTagsUseRelationConsistentCanonicalCenter) {
  const auto node = makeNode();
  const tf2::Transform front(rotation(0.1, -0.2, 0.3),
                             tf2::Vector3(1.0, 2.0, 3.0));
  const auto back =
      front * ApriltagFusionNodeTestPeer::activeFrontFromBackTag(*node);

  const auto result = ApriltagFusionNodeTestPeer::both(*node, front, back);
  const auto expected = ApriltagFusionNodeTestPeer::single(
      *node, ApriltagFusionNodeTestPeer::frontSlot(), front);
  expectTransformNear(result, expected, kTolerance, 1e-6);
}

TEST_F(ApriltagFusionTest,
       FullTagTransformLearnsNonIdealRelationAndCanonicalCenter) {
  auto node = makeNode();
  const tf2::Transform expected_front_from_back(
      rotation(0.06, kPi - 0.08, -0.04), tf2::Vector3(0.012, -0.006, 0.070));
  const auto start = node->now();

  for (int sample_index = 0; sample_index < 36; ++sample_index) {
    const bool outlier = sample_index == 4 || sample_index == 11 ||
                         sample_index == 18 || sample_index == 25;
    tf2::Transform sampled_front_from_back;
    if (outlier) {
      auto outlier_rotation =
          expected_front_from_back.getRotation() * rotation(0.45, 0.0, 0.0);
      outlier_rotation.normalize();
      sampled_front_from_back = tf2::Transform(
          outlier_rotation, expected_front_from_back.getOrigin() +
                                tf2::Vector3(0.20, -0.10, 0.08));
    } else {
      const double signed_step = static_cast<double>((sample_index % 5) - 2);
      auto noisy_rotation =
          expected_front_from_back.getRotation() *
          rotation(0.0008 * signed_step, -0.0005 * signed_step,
                   0.0006 * signed_step);
      noisy_rotation.normalize();
      sampled_front_from_back = tf2::Transform(
          noisy_rotation,
          expected_front_from_back.getOrigin() +
              tf2::Vector3(0.0003 * signed_step, -0.0002 * signed_step,
                           0.0001 * signed_step));
    }

    const tf2::Transform fusion_from_front(
        rotation(0.01 * sample_index, -0.004 * sample_index,
                 0.006 * sample_index),
        tf2::Vector3(0.002 * sample_index, -0.001 * sample_index,
                     1.0 + 0.0005 * sample_index));
    const auto fusion_from_back = fusion_from_front * sampled_front_from_back;
    const auto stamp =
        start + rclcpp::Duration::from_seconds(0.1 * sample_index);
    const auto front = makeEstimate(fusion_from_front, stamp, 1.0,
                                    static_cast<uint64_t>(100 + sample_index));
    const auto back = makeEstimate(fusion_from_back, stamp, 1.0,
                                   static_cast<uint64_t>(1000 + sample_index));
    ApriltagFusionNodeTestPeer::updateTagTransform(*node, front, back);
  }

  ASSERT_TRUE(ApriltagFusionNodeTestPeer::tagTransformCalibrated(*node));
  const auto learned =
      ApriltagFusionNodeTestPeer::activeFrontFromBackTag(*node);
  expectTransformNear(learned, expected_front_from_back, 0.002, 0.5);

  const tf2::Transform fusion_from_front(rotation(0.2, -0.15, 0.35),
                                         tf2::Vector3(1.0, -0.5, 0.8));
  const auto fusion_from_back = fusion_from_front * expected_front_from_back;
  const auto center_from_front = ApriltagFusionNodeTestPeer::single(
      *node, ApriltagFusionNodeTestPeer::frontSlot(), fusion_from_front);
  const auto center_from_back = ApriltagFusionNodeTestPeer::single(
      *node, ApriltagFusionNodeTestPeer::backSlot(), fusion_from_back);
  expectTransformNear(center_from_front, center_from_back, 0.003, 0.6);
}

TEST_F(ApriltagFusionTest, TagTransformRejectsDuplicateStaleAndOutlierPairs) {
  auto node =
      makeNode({rclcpp::Parameter("tag_transform_bootstrap_duration_sec", 0.2),
                rclcpp::Parameter("tag_transform_bootstrap_min_samples", 3)});
  const tf2::Transform expected_front_from_back(
      rotation(0.03, kPi - 0.04, -0.02), tf2::Vector3(0.006, -0.003, 0.065));
  const tf2::Transform fusion_from_front(rotation(0.1, -0.2, 0.3),
                                         tf2::Vector3(0.5, -0.4, 1.2));
  const auto fusion_from_back = fusion_from_front * expected_front_from_back;
  const auto start = node->now();

  auto front = makeEstimate(fusion_from_front, start, 1.0, 200);
  auto back = makeEstimate(fusion_from_back, start, 1.0, 1200);
  ApriltagFusionNodeTestPeer::updateTagTransform(*node, front, back);
  EXPECT_EQ(ApriltagFusionNodeTestPeer::tagTransformBootstrapSampleCount(*node),
            1U);

  front.observation.receipt_time += rclcpp::Duration::from_seconds(0.05);
  back.observation.receipt_time += rclcpp::Duration::from_seconds(0.05);
  ApriltagFusionNodeTestPeer::updateTagTransform(*node, front, back);
  EXPECT_EQ(ApriltagFusionNodeTestPeer::tagTransformBootstrapSampleCount(*node),
            1U);

  auto stale_front = makeEstimate(
      fusion_from_front, start + rclcpp::Duration::from_seconds(0.1), 1.0, 201);
  auto stale_back =
      makeEstimate(fusion_from_back,
                   start + rclcpp::Duration::from_seconds(0.25), 1.0, 1201);
  ApriltagFusionNodeTestPeer::updateTagTransform(*node, stale_front,
                                                 stale_back);
  EXPECT_EQ(ApriltagFusionNodeTestPeer::tagTransformBootstrapSampleCount(*node),
            1U);

  for (int sample_index = 1; sample_index <= 2; ++sample_index) {
    const auto stamp =
        start + rclcpp::Duration::from_seconds(0.1 * sample_index);
    ApriltagFusionNodeTestPeer::updateTagTransform(
        *node,
        makeEstimate(fusion_from_front, stamp, 1.0,
                     static_cast<uint64_t>(210 + sample_index)),
        makeEstimate(fusion_from_back, stamp, 1.0,
                     static_cast<uint64_t>(1210 + sample_index)));
  }
  ASSERT_TRUE(ApriltagFusionNodeTestPeer::tagTransformCalibrated(*node));
  const auto before_outlier =
      ApriltagFusionNodeTestPeer::activeFrontFromBackTag(*node);

  auto outlier_rotation =
      expected_front_from_back.getRotation() * rotation(0.6, 0.0, 0.0);
  outlier_rotation.normalize();
  const tf2::Transform outlier_relation(outlier_rotation,
                                        expected_front_from_back.getOrigin() +
                                            tf2::Vector3(0.5, 0.2, -0.1));
  const auto outlier_stamp = start + rclcpp::Duration::from_seconds(0.3);
  ApriltagFusionNodeTestPeer::updateTagTransform(
      *node, makeEstimate(fusion_from_front, outlier_stamp, 1.0, 220),
      makeEstimate(fusion_from_front * outlier_relation, outlier_stamp, 1.0,
                   1220));

  expectTransformNear(ApriltagFusionNodeTestPeer::activeFrontFromBackTag(*node),
                      before_outlier, kTolerance, 1e-6);
}

TEST_F(ApriltagFusionTest,
       InconsistentCalibratedPairUsesHigherQualitySingleTag) {
  auto node =
      makeNode({rclcpp::Parameter("tag_transform_bootstrap_duration_sec", 0.0),
                rclcpp::Parameter("tag_transform_bootstrap_min_samples", 1)});
  const auto stamp = node->now();
  const tf2::Transform fusion_from_front(rotation(0.1, -0.2, 0.3),
                                         tf2::Vector3(0.5, -0.4, 1.2));
  const auto ideal_relation =
      ApriltagFusionNodeTestPeer::activeFrontFromBackTag(*node);
  const auto fusion_from_back = fusion_from_front * ideal_relation;
  ApriltagFusionNodeTestPeer::updateTagTransform(
      *node, makeEstimate(fusion_from_front, stamp, 10.0, 300),
      makeEstimate(fusion_from_back, stamp, 10.0, 1300));
  ASSERT_TRUE(ApriltagFusionNodeTestPeer::tagTransformCalibrated(*node));

  auto inconsistent_back = fusion_from_back;
  inconsistent_back.setOrigin(inconsistent_back.getOrigin() +
                              tf2::Vector3(0.12, -0.05, 0.02));
  ApriltagFusionNodeTestPeer::setObservation(
      *node, 0, ApriltagFusionNodeTestPeer::frontSlot(),
      makeObservation(fusion_from_front, stamp, 10.0, 301));
  ApriltagFusionNodeTestPeer::setObservation(
      *node, 1, ApriltagFusionNodeTestPeer::backSlot(),
      makeObservation(inconsistent_back, stamp, 1.0, 1301));

  ApriltagFusionNodeTestPeer::fuseAndPublish(*node);

  const auto expected = ApriltagFusionNodeTestPeer::single(
      *node, ApriltagFusionNodeTestPeer::frontSlot(), fusion_from_front);
  expectTransformNear(ApriltagFusionNodeTestPeer::smoothedTransform(*node),
                      expected, kTolerance, 1e-6);
}

TEST_F(ApriltagFusionTest,
       LegacySeparationSeedRetainsClippedCompatibilityBehavior) {
  auto node = makeNode({rclcpp::Parameter("learn_tag_transform", false)});
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

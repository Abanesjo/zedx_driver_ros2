#include <sl/Camera.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>

namespace
{
using Bone = std::pair<int, int>;

template<typename PartT>
std::vector<Bone> toIndexBones(const std::vector<std::pair<PartT, PartT>> & bones)
{
  std::vector<Bone> indexed;
  indexed.reserve(bones.size());
  for (const auto & bone : bones) {
    indexed.emplace_back(sl::getIdx(bone.first), sl::getIdx(bone.second));
  }
  return indexed;
}

const std::vector<Bone> & bonesForFormat(int8_t body_format)
{
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

cv::Scalar colorForId(int id)
{
  static const std::vector<cv::Scalar> colors = {
    cv::Scalar(232.0, 176.0, 59.0),
    cv::Scalar(175.0, 208.0, 25.0),
    cv::Scalar(102.0, 205.0, 105.0),
    cv::Scalar(185.0, 0.0, 255.0),
    cv::Scalar(99.0, 107.0, 252.0),
    cv::Scalar(252.0, 225.0, 8.0),
    cv::Scalar(167.0, 130.0, 141.0),
    cv::Scalar(194.0, 72.0, 113.0)};

  if (id < 0) {
    return cv::Scalar(236.0, 184.0, 36.0);
  }
  return colors[static_cast<size_t>(id) % colors.size()];
}

}  // namespace

class ZedSkeletonOverlayNode final : public rclcpp::Node
{
public:
  explicit ZedSkeletonOverlayNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("zed_skeleton_overlay_node", options)
  {
    const auto camera_name = declare_parameter<std::string>("camera_name", "zed_left");
    const auto default_image_topic = "/" + camera_name + "/zed_node/rgb/color/rect/image";
    const auto default_skeleton_topic = "/" + camera_name + "/zed_node/body_trk/skeletons";
    const auto default_overlay_topic =
      "/" + camera_name + "/zed_node/rgb/color/rect/skeleton_overlay";

    image_topic_ = declare_parameter<std::string>("image_topic", default_image_topic);
    skeleton_topic_ = declare_parameter<std::string>("skeleton_topic", default_skeleton_topic);
    overlay_topic_ = declare_parameter<std::string>("overlay_topic", default_overlay_topic);
    min_confidence_ = declare_parameter<double>("min_confidence", 70.0);
    max_skeleton_age_sec_ = declare_parameter<double>("max_skeleton_age_sec", 0.5);
    line_thickness_ = declare_parameter<int>("line_thickness", 2);
    point_radius_ = declare_parameter<int>("point_radius", 4);
    draw_labels_ = declare_parameter<bool>("draw_labels", false);

    if (min_confidence_ < 0.0 || min_confidence_ > 100.0) {
      throw std::runtime_error("min_confidence must be in the range [0, 100]");
    }
    if (max_skeleton_age_sec_ < 0.0) {
      throw std::runtime_error("max_skeleton_age_sec must be non-negative");
    }
    if (line_thickness_ <= 0) {
      throw std::runtime_error("line_thickness must be positive");
    }
    if (point_radius_ <= 0) {
      throw std::runtime_error("point_radius must be positive");
    }

    overlay_pub_ = create_publisher<sensor_msgs::msg::Image>(
      overlay_topic_, rclcpp::SensorDataQoS());
    skeleton_sub_ = create_subscription<zed_msgs::msg::ObjectsStamped>(
      skeleton_topic_, rclcpp::SensorDataQoS(),
      [this](zed_msgs::msg::ObjectsStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(skeleton_mutex_);
        latest_skeletons_ = msg;
      });
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        handleImage(msg);
      });

    RCLCPP_INFO(
      get_logger(), "Publishing skeleton overlays from %s and %s to %s",
      image_topic_.c_str(), skeleton_topic_.c_str(), overlay_topic_.c_str());
  }

private:
  bool validPoint(const zed_msgs::msg::Keypoint2Df & keypoint, const cv::Size & size) const
  {
    const float x = keypoint.kp[0];
    const float y = keypoint.kp[1];
    return std::isfinite(x) && std::isfinite(y) && x > 0.0f && y > 0.0f &&
           x < static_cast<float>(size.width) && y < static_cast<float>(size.height);
  }

  cv::Point toPoint(const zed_msgs::msg::Keypoint2Df & keypoint) const
  {
    return cv::Point(
      static_cast<int>(std::lround(keypoint.kp[0])),
      static_cast<int>(std::lround(keypoint.kp[1])));
  }

  void drawObject(cv::Mat & image, const zed_msgs::msg::Object & object) const
  {
    if (!object.skeleton_available || object.confidence < min_confidence_) {
      return;
    }

    const auto color = colorForId(object.label_id);
    const auto & keypoints = object.skeleton_2d.keypoints;
    const auto & bones = bonesForFormat(object.body_format);
    const auto image_size = image.size();

    for (const auto & bone : bones) {
      if (bone.first < 0 || bone.second < 0 ||
        static_cast<size_t>(bone.first) >= keypoints.size() ||
        static_cast<size_t>(bone.second) >= keypoints.size())
      {
        continue;
      }
      const auto & first = keypoints[static_cast<size_t>(bone.first)];
      const auto & second = keypoints[static_cast<size_t>(bone.second)];
      if (!validPoint(first, image_size) || !validPoint(second, image_size)) {
        continue;
      }
      cv::line(image, toPoint(first), toPoint(second), color, line_thickness_, cv::LINE_AA);
    }

    for (const auto & keypoint : keypoints) {
      if (!validPoint(keypoint, image_size)) {
        continue;
      }
      cv::circle(image, toPoint(keypoint), point_radius_, color, -1, cv::LINE_AA);
    }

    if (draw_labels_) {
      const std::string label =
        std::to_string(object.label_id) + " " + std::to_string(static_cast<int>(object.confidence));
      cv::putText(
        image, label, labelAnchor(object), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }
  }

  cv::Point labelAnchor(const zed_msgs::msg::Object & object) const
  {
    for (const auto & keypoint : object.skeleton_2d.keypoints) {
      if (std::isfinite(keypoint.kp[0]) && std::isfinite(keypoint.kp[1]) &&
        keypoint.kp[0] > 0.0f && keypoint.kp[1] > 0.0f)
      {
        return toPoint(keypoint) + cv::Point(6, -6);
      }
    }
    return cv::Point(10, 24);
  }

  void handleImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    if (overlay_pub_->get_subscription_count() == 0) {
      return;
    }

    zed_msgs::msg::ObjectsStamped::SharedPtr skeletons;
    {
      std::lock_guard<std::mutex> lock(skeleton_mutex_);
      skeletons = latest_skeletons_;
    }

    cv::Mat image;
    if (!toBgrImage(*msg, image)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Unsupported or invalid image on %s: encoding=%s",
        image_topic_.c_str(), msg->encoding.c_str());
      return;
    }

    if (skeletons && skeletonsAreFresh(*skeletons)) {
      for (const auto & object : skeletons->objects) {
        drawObject(image, object);
      }
    }

    overlay_pub_->publish(toImageMessage(image, msg->header));
  }

  bool skeletonsAreFresh(const zed_msgs::msg::ObjectsStamped & skeletons) const
  {
    if (max_skeleton_age_sec_ == 0.0) {
      return true;
    }

    const rclcpp::Time stamp(skeletons.header.stamp);
    if (stamp.nanoseconds() == 0) {
      return true;
    }

    const double age_sec = std::abs((now() - stamp).seconds());
    return age_sec <= max_skeleton_age_sec_;
  }

  bool toBgrImage(const sensor_msgs::msg::Image & msg, cv::Mat & image) const
  {
    if (msg.height == 0 || msg.width == 0 || msg.data.empty()) {
      return false;
    }

    const size_t min_size = static_cast<size_t>(msg.height) * msg.step;
    if (msg.data.size() < min_size) {
      return false;
    }

    if (msg.encoding == sensor_msgs::image_encodings::BGR8) {
      const cv::Mat view(
        static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3,
        const_cast<uint8_t *>(msg.data.data()), msg.step);
      image = view.clone();
      return true;
    }

    if (msg.encoding == sensor_msgs::image_encodings::RGB8) {
      const cv::Mat view(
        static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3,
        const_cast<uint8_t *>(msg.data.data()), msg.step);
      cv::cvtColor(view, image, cv::COLOR_RGB2BGR);
      return true;
    }

    if (msg.encoding == sensor_msgs::image_encodings::BGRA8) {
      const cv::Mat view(
        static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC4,
        const_cast<uint8_t *>(msg.data.data()), msg.step);
      cv::cvtColor(view, image, cv::COLOR_BGRA2BGR);
      return true;
    }

    if (msg.encoding == sensor_msgs::image_encodings::RGBA8) {
      const cv::Mat view(
        static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC4,
        const_cast<uint8_t *>(msg.data.data()), msg.step);
      cv::cvtColor(view, image, cv::COLOR_RGBA2BGR);
      return true;
    }

    if (msg.encoding == sensor_msgs::image_encodings::MONO8) {
      const cv::Mat view(
        static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC1,
        const_cast<uint8_t *>(msg.data.data()), msg.step);
      cv::cvtColor(view, image, cv::COLOR_GRAY2BGR);
      return true;
    }

    return false;
  }

  sensor_msgs::msg::Image toImageMessage(
    const cv::Mat & image, const std_msgs::msg::Header & header) const
  {
    sensor_msgs::msg::Image msg;
    msg.header = header;
    msg.height = static_cast<uint32_t>(image.rows);
    msg.width = static_cast<uint32_t>(image.cols);
    msg.encoding = sensor_msgs::image_encodings::BGR8;
    msg.is_bigendian = false;
    msg.step = static_cast<uint32_t>(image.cols * image.elemSize());
    msg.data.resize(static_cast<size_t>(msg.height) * msg.step);

    for (uint32_t row = 0; row < msg.height; ++row) {
      const auto * src = image.ptr<uint8_t>(static_cast<int>(row));
      auto * dest = msg.data.data() + static_cast<size_t>(row) * msg.step;
      std::copy(src, src + msg.step, dest);
    }

    return msg;
  }

  std::string image_topic_;
  std::string skeleton_topic_;
  std::string overlay_topic_;
  double min_confidence_ = 70.0;
  double max_skeleton_age_sec_ = 0.5;
  int line_thickness_ = 2;
  int point_radius_ = 4;
  bool draw_labels_ = false;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<zed_msgs::msg::ObjectsStamped>::SharedPtr skeleton_sub_;
  zed_msgs::msg::ObjectsStamped::SharedPtr latest_skeletons_;
  std::mutex skeleton_mutex_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<ZedSkeletonOverlayNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("zed_skeleton_overlay_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}

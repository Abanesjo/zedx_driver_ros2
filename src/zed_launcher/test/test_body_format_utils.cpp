#include "zed_launcher/body_format_utils.hpp"

#include <gtest/gtest.h>

namespace {

TEST(BodyFormatUtils, UsesActualSdkResultFormat) {
  sl::Bodies bodies;

  bodies.body_format = sl::BODY_FORMAT::BODY_18;
  EXPECT_EQ(zed_launcher::rosBodyFormat(bodies),
            static_cast<uint8_t>(sl::BODY_FORMAT::BODY_18));

  // Fusion body fitting turns BODY_18 sender output into BODY_34 output. The
  // ROS message must report that result, irrespective of sender configuration.
  bodies.body_format = sl::BODY_FORMAT::BODY_34;
  EXPECT_EQ(zed_launcher::rosBodyFormat(bodies),
            static_cast<uint8_t>(sl::BODY_FORMAT::BODY_34));

  bodies.body_format = sl::BODY_FORMAT::BODY_38;
  EXPECT_EQ(zed_launcher::rosBodyFormat(bodies),
            static_cast<uint8_t>(sl::BODY_FORMAT::BODY_38));
}

} // namespace

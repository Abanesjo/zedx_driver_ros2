#pragma once

#include <sl/Camera.hpp>

#include <cstdint>

namespace zed_launcher {

// Fusion can change the skeleton format it returns. In particular, fitting a
// BODY_18 sender skeleton produces BODY_34 output, so the result container is
// the authoritative format rather than the sender's configured input format.
[[nodiscard]] inline uint8_t rosBodyFormat(const sl::Bodies &bodies) noexcept {
  return static_cast<uint8_t>(bodies.body_format);
}

} // namespace zed_launcher

#pragma once

#include <sl/Camera.hpp>

namespace zed_launcher {

// With tracking enabled, OK is the only state backed by a current measured
// body. SEARCHING and TERMINATE are retained SDK tracks, and OFF means
// tracking has not initialized. When tracking is intentionally disabled, OFF
// is the normal measured-detection state.
[[nodiscard]] inline bool
isMeasuredBodyTrackingState(sl::OBJECT_TRACKING_STATE state,
                            bool tracking_available) noexcept {
  if (tracking_available) {
    return state == sl::OBJECT_TRACKING_STATE::OK;
  }
  return state == sl::OBJECT_TRACKING_STATE::OFF ||
         state == sl::OBJECT_TRACKING_STATE::OK;
}

} // namespace zed_launcher

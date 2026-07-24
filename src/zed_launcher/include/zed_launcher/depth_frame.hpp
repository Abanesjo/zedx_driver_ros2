#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace zed_launcher {

/// Non-owning CPU view of a row-major F32_C1 depth image in meters.
///
/// The data and provider are valid only for the duration of the synchronous
/// image-processing callback that receives the provider.
struct DepthFrameView {
  const float *data = nullptr;
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t row_stride_bytes = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return data != nullptr && width > 0 && height > 0 &&
           row_stride_bytes / sizeof(float) >= width;
  }
};

/// Lazily retrieves depth for the current camera grab.
///
/// A provider returns std::nullopt when depth is unavailable. Consumers must
/// not retain or invoke it after their synchronous image callback returns.
using DepthFrameProvider = std::function<std::optional<DepthFrameView>()>;

/// Owning, tightly packed CPU depth frame used by asynchronous consumers.
struct OwnedDepthFrame {
  std::vector<float> data;
  std::size_t width = 0;
  std::size_t height = 0;

  [[nodiscard]] DepthFrameView view() const noexcept {
    return DepthFrameView{data.data(), width, height, width * sizeof(float)};
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return width > 0 && height > 0 && data.size() == width * height;
  }
};

} // namespace zed_launcher

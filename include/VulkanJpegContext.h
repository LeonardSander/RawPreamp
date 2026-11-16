#pragma once

#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include <libplacebo/vulkan.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <memory>
#include <vector>

namespace rawpreamp {
// Shared by raw/JPEG decode, proxy reduction and resident processing. Callers
// hold this mutex while issuing commands on this libplacebo GPU.
struct VulkanJpegContext {
    // A consumer may release its last image lease during a submission. Allow
    // that nested retirement while serializing GPU work across threads.
    std::recursive_mutex mutex;
    pl_log log = nullptr;
    pl_vulkan vulkan = nullptr;
    bool unavailable = false;
    // Pool access and GPU destruction use the same submission lock. Leases
    // retain this context and may retire on any consumer thread.
    std::vector<pl_buf> retiredImages;
    size_t retiredImageBytes = 0;
    std::shared_ptr<void> acquireImage(size_t bytes, pl_buf& buffer);
    void retireImage(pl_buf buffer);

    ~VulkanJpegContext();
    bool init();
};
VulkanJpegContext& jpegContext();
// Resident consumers retain the device independently of static cache order.
std::shared_ptr<void> jpegContextLease();
// Call only while holding jpegContext().mutex. The buffer remains valid until
// the next predictor scan; a source pointer mismatch signals that it changed.
pl_buf jpegResidentImageVulkan(const uint16_t* pixels, uint32_t width,
                               uint32_t height, uint32_t channels,
                               uint32_t bits);
}
#endif

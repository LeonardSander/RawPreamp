#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>

namespace rawpreamp {
// Codec-neutral packed image storage. Texture imports keep their native layout
// and enter the shared processing pipeline without conversion to this buffer.
struct VulkanResidentImage {
    const void* gpu = nullptr;
    const void* buffer = nullptr;
    // GPU word containing 1 only if compressed-frame decoding succeeded.
    // Terminal processing validates this without an image readback.
    const void* validationBuffer = nullptr;
    uint32_t width = 0, height = 0, channels = 0, bits = 0, wordsPerRow = 0;
    // Optional ownership transferred from the decoder for deferred consumers.
    std::shared_ptr<void> lease;
    std::shared_ptr<void> deviceLease;
};
// Nearest reduction preserves CFA parity; packed RGB uses pixel centers.
bool downscaleVulkanResidentImage(const VulkanResidentImage& source,
                                  uint32_t scale, bool cfa, VulkanResidentImage& result);
// Average each same-color CFA block with integer half-up rounding. A 4x4
// CFA becomes ordinary 2x2 Bayer at half width/height. The caller supplies
// factor 2..4; input must contain complete color blocks and even output rows.
// No pixel readback.
bool binVulkanResidentCfa(const VulkanResidentImage& source, uint32_t factor,
                           VulkanResidentImage& result);
// Explicit output/test operation; never used to bridge GPU processing stages.
bool downloadVulkanResidentImage(const VulkanResidentImage&, uint16_t*, size_t);
}

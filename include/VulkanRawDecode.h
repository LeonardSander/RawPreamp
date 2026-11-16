#pragma once
#include "VulkanResidentImage.h"

namespace motioncam {
enum class EncodedFrameFormat;
}

namespace rawpreamp {
// Capability-based dispatch shared by the raw-container adapters. New codec
// backends join here; sources are never selected by filename extension.
bool supportsRawFrameDecodeVulkan(motioncam::EncodedFrameFormat format);
bool rawFrameDecodeVulkan(motioncam::EncodedFrameFormat format, uint32_t compression,
                          const uint8_t* bytes, size_t size, uint32_t width, uint32_t height,
                          VulkanResidentImage& image, uint32_t maximumCode = 65535);

// MotionCam compression type 7. Uploads compressed bytes and block descriptors;
// the returned allocation has the same packed uint16 layout as resident DNG
// JPEG decode. No pixel/status readback is needed for this bounded block codec.
// Legacy compression, invalid geometry and unavailable GPUs return false.
bool mcrawDecodeVulkan(const uint8_t* bytes, size_t size, uint32_t width, uint32_t height,
                       VulkanResidentImage& image);
// Native MC24 record and UFR1 payload decoders. MC24 unpacking and predictor
// reconstruction are separate GPU passes; no image samples cross the CPU.
bool mc24DecodeVulkan(const uint8_t*, size_t, uint32_t, uint32_t, VulkanResidentImage&,
                      uint32_t maximumCode = 65535);
// Checks only the decoder's control word, never image samples.
bool validateVulkanRawFrame(const VulkanResidentImage& image);
bool unspektraDecodeVulkan(const uint8_t*, size_t, uint32_t, uint32_t, VulkanResidentImage&);
} // namespace rawpreamp

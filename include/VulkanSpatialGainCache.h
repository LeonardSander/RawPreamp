#pragma once
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include "DebugInstrumentation.h"
#include <libplacebo/gpu.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

namespace rawpreamp {
#if RAWPREAMP_DEBUG_INSTRUMENTATION
inline std::atomic<uint64_t> spatialGainCacheBuilds{0}, spatialGainCacheHits{0};
#endif
// Source/codec-independent storage and stability policy. The caller supplies
// its existing interpolation shader and serializes access on the owning GPU.
class VulkanSpatialGainCache {
    pl_gpu gpu = nullptr;
    pl_buf field = nullptr;
    std::vector<uint8_t> active, pending, rejected;
    unsigned unsharedCaptures = 0;
public:
    VulkanSpatialGainCache() = default;
    VulkanSpatialGainCache(const VulkanSpatialGainCache&) = delete;
    VulkanSpatialGainCache& operator=(const VulkanSpatialGainCache&) = delete;
    ~VulkanSpatialGainCache() { reset(); }
    void reset() {
        if (field) pl_buf_destroy(gpu, &field);
        active.clear(); pending.clear(); rejected.clear(); unsharedCaptures = 0;
    }
    template<class Generate>
    pl_buf acquire(pl_gpu device, const std::vector<uint8_t>& key,
                   size_t bytes, uint32_t channels, Generate&& generate,
                   bool captureDuringProcessing = false) {
        const size_t budget = size_t(64u * 1024u * 1024u) * std::min(channels, 3u);
        if (gpu != device) { reset(); gpu = device; }
        if (key.empty() || !bytes || !channels || bytes > budget ||
            bytes > device->limits.max_ssbo_size ||
            debugEnvironmentIsSet("RAWPREAMP_GAIN_FIELD_DISABLE")) {
            reset(); return nullptr;
        }
        if (field && active == key) {
            pending.clear(); unsharedCaptures = 0;
#if RAWPREAMP_DEBUG_INSTRUMENTATION
            spatialGainCacheHits.fetch_add(1, std::memory_order_relaxed);
#endif
            return field;
        }
        if (rejected == key) { pending.clear(); return nullptr; }
        // Separate interpolation passes need evidence of reuse. A caller
        // capturing gains in its existing pixel pass avoids that duplicate work.
        // After two captures without reuse, wait for a repeat so continuously
        // changing maps retain the ordinary pass and avoid field writes.
        const bool repeated = pending == key;
        if ((!captureDuringProcessing || unsharedCaptures >= 2) && !repeated) {
            pending = key;
            return nullptr;
        }
        active.clear();
        pl_buf_params params{};
        params.size = bytes; params.storable = true;
        params.memory_type = PL_BUF_MEM_DEVICE;
        if (!pl_buf_recreate(gpu, &field, &params) || !generate(field)) {
            if (field) pl_buf_destroy(gpu, &field);
            rejected = key; pending.clear();
            return nullptr;
        }
        active = key; pending.clear(); rejected.clear();
        if (captureDuringProcessing) unsharedCaptures = repeated ? 1u : unsharedCaptures + 1u;
#if RAWPREAMP_DEBUG_INSTRUMENTATION
        spatialGainCacheBuilds.fetch_add(1, std::memory_order_relaxed);
#endif
        return field;
    }
};
}
#endif

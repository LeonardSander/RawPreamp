#pragma once
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include "DebugInstrumentation.h"
#include <libplacebo/gpu.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>

namespace rawpreamp {
// Shared by gallery gain/profile resources and DNG bake lookup tables. Call
// under the owning device's submission lock; invalidate on allocation changes.
class VulkanUploadCache {
    std::vector<uint8_t> previous;
    pl_gpu previousGpu = nullptr;
    pl_buf previousTarget = nullptr;
public:
    void invalidate() {
        previous.clear();
        previousGpu = nullptr;
        previousTarget = nullptr;
    }
    size_t upload(pl_gpu gpu, pl_buf target, pl_buf staging,
                  const void* source, size_t bytes) {
        if (!bytes) return 0;
        if (gpu != previousGpu || target != previousTarget) invalidate();
        previousGpu = gpu;
        previousTarget = target;
        const auto* data = static_cast<const uint8_t*>(source);
        size_t first = 0, last = bytes;
        const bool disabled = debugEnvironmentIsSet("RAWPREAMP_GPU_UPLOAD_CACHE_DISABLE");
        if (!disabled && previous.size() == bytes) {
            if (!std::memcmp(previous.data(), data, bytes)) return 0;
            constexpr size_t block = 4096;
            while (first + block <= bytes &&
                   !std::memcmp(previous.data() + first, data + first, block)) first += block;
            while (first < bytes && previous[first] == data[first]) ++first;
            while (last >= first + block &&
                   !std::memcmp(previous.data() + last - block, data + last - block, block)) last -= block;
            while (last > first && previous[last - 1] == data[last - 1]) --last;
            // Storage transfers are word aligned; surrounding bytes are valid.
            first &= ~size_t(3);
            last = std::min(bytes, (last + 3u) & ~size_t(3));
        }
        pl_buf_write(gpu, staging, first, data + first, last - first);
        pl_buf_copy(gpu, target, first, staging, first, last - first);
        if (disabled) previous.clear();
        else if (previous.size() != bytes) previous.assign(data, data + bytes);
        else std::memcpy(previous.data() + first, data + first, last - first);
        return last - first;
    }
};
}
#endif

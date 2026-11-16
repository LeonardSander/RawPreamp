#include "VulkanResidentImage.h"
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include "VulkanJpegContext.h"
#include <libplacebo/gpu.h>
#include <mutex>
#include <string>
namespace rawpreamp {
namespace {
constexpr char reductionShader[] = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Source { uint words[]; } src;
layout(std430, binding = 1) writeonly buffer Destination { uint words[]; } dst;
layout(push_constant) uniform Constants {
    uint sourceWidth; uint width; uint height; uint channels; uint scale; uint cfa;
} p;
uint sampleAt(uint index) {
#ifdef CFA_BIN
    uint pixel = index;
#else
    uint channel = index % p.channels;
    uint pixel = index / p.channels;
#endif
    uint x = pixel % p.width, y = pixel / p.width;
#ifdef CFA_BIN
    uint sum = 0u;
    for (uint by = 0u; by < p.scale; ++by)
        for (uint bx = 0u; bx < p.scale; ++bx) {
            uint source = (y * p.scale + by) * p.sourceWidth + x * p.scale + bx;
            sum += (src.words[source / 2u] >> ((source & 1u) * 16u)) & 65535u;
        }
    uint area = p.scale * p.scale;
    return (sum + area / 2u) / area;
#else
    uint sx = x * p.scale + (p.cfa != 0u ? (x & 1u) : (p.scale - 1u) / 2u);
    uint sy = y * p.scale + (p.cfa != 0u ? (y & 1u) : (p.scale - 1u) / 2u);
    uint source = (sy * p.sourceWidth + sx) * p.channels + channel;
    return (src.words[source / 2u] >> ((source & 1u) * 16u)) & 65535u;
#endif
}
void main() {
    uint word = gl_GlobalInvocationID.x;
    if (word >= p.width * p.height * p.channels / 2u) return;
    dst.words[word] = sampleAt(word * 2u) | (sampleAt(word * 2u + 1u) << 16u);
}
)glsl";
struct ReductionState {
    pl_gpu gpu = nullptr;
    pl_pass proxy = nullptr, bin = nullptr;
    std::shared_ptr<void> deviceLease;
    ~ReductionState() {
        if (gpu) { pl_pass_destroy(gpu, &proxy); pl_pass_destroy(gpu, &bin); }
    }
};
}
static bool reduceResidentImage(const VulkanResidentImage& source,
                                uint32_t scale, bool cfa,
                                VulkanResidentImage& result, bool bin) {
    if (!source.gpu || !source.buffer || !source.lease || scale <= 1 ||
        (cfa ? source.channels != 1 || (!bin && (scale & 1u)) : source.channels != 3) ||
        (bin && (scale > 4 || !cfa || source.width % (2u * scale) || source.height % (2u * scale))) ||
        uint64_t(source.wordsPerRow) * 2u != uint64_t(source.width) * source.channels)
        return false;
    if (!source.width || !source.height ||
        uint64_t(source.width) * source.height > UINT32_MAX / source.channels ||
        static_cast<pl_buf>(source.buffer)->params.size <
            uint64_t(source.width) * source.height * source.channels * sizeof(uint16_t))
        return false;
    uint32_t width = source.width / scale, height = source.height / scale;
    if (cfa) { width &= ~1u; height &= ~1u; }
    // Gallery buffer inputs use tightly packed rows. Odd RGB row widths retain
    // the CPU fallback rather than exposing padding as image samples.
    if (!width || !height || (width & 1u)) return false;
    const uint64_t samples = uint64_t(width) * height * source.channels;
    if (samples > UINT32_MAX) return false;
    auto& context = jpegContext();
    std::lock_guard lock(context.mutex);
    if (!context.init() || source.gpu != context.vulkan->gpu) return false;
    static ReductionState s;
    if (!s.gpu) {
        s.gpu = context.vulkan->gpu;
        s.deviceLease = jpegContextLease();
    }
    if ((samples / 2u + 255u) / 256u > uint32_t(s.gpu->limits.max_dispatch[0])) return false;
    auto& pass = bin ? s.bin : s.proxy;
    if (!pass) {
        pl_desc descriptors[] = {
            {"src", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
            {"dst", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_WRITEONLY},
        };
        pl_pass_params params{};
        params.type = PL_PASS_COMPUTE;
        std::string shader = reductionShader;
        if (bin) shader.insert(shader.find('\n') + 1u, "#define CFA_BIN\n");
        params.glsl_shader = shader.c_str();
        params.descriptors = descriptors;
        params.num_descriptors = 2;
        params.push_constants_size = 6 * sizeof(uint32_t);
        pass = pl_pass_create(s.gpu, &params);
    }
    pl_buf buffer = nullptr;
    if (!pass) return false;
    auto owner = context.acquireImage(samples * sizeof(uint16_t), buffer);
    if (!owner) return false;
    uint32_t constants[] = {source.width, width, height, source.channels, scale, cfa ? 1u : 0u};
    pl_desc_binding bindings[2]{};
    bindings[0].object = static_cast<pl_buf>(source.buffer);
    bindings[1].object = buffer;
    pl_pass_run_params run{};
    run.pass = pass;
    run.desc_bindings = bindings;
    run.push_constants = constants;
    run.compute_groups[0] = int((samples / 2u + 255u) / 256u);
    run.compute_groups[1] = run.compute_groups[2] = 1;
    pl_pass_run(s.gpu, &run);
    result = source;
    result.buffer = buffer;
    result.width = width; result.height = height;
    result.wordsPerRow = width * source.channels / 2u;
    // Preserve source decode validity; reduction does not validate the frame.
    // Keep its control word and pixels alive until this output is released.
    struct Leases { std::shared_ptr<void> source, output; };
    result.lease = std::make_shared<Leases>(Leases{source.lease, std::move(owner)});
    return true;
}

bool downscaleVulkanResidentImage(const VulkanResidentImage& source,
                                  uint32_t scale, bool cfa, VulkanResidentImage& result) {
    return reduceResidentImage(source, scale, cfa, result, false);
}
bool binVulkanResidentCfa(const VulkanResidentImage& source, uint32_t factor,
                           VulkanResidentImage& result) {
    return reduceResidentImage(source, factor, true, result, true);
}

bool downloadVulkanResidentImage(const VulkanResidentImage& image,
                                  uint16_t* pixels, size_t samples) {
    if (!pixels || !image.gpu || !image.buffer || !image.lease ||
        !image.width || !image.height || (image.channels != 1 && image.channels != 3) ||
        uint64_t(image.width) * image.height > SIZE_MAX / sizeof(uint16_t) / image.channels ||
        samples != size_t(image.width) * image.height * image.channels ||
        uint64_t(image.wordsPerRow) * 2u != uint64_t(image.width) * image.channels ||
        static_cast<pl_buf>(image.buffer)->params.size < samples * sizeof(uint16_t)) return false;
    auto& context = jpegContext();
    std::lock_guard lock(context.mutex);
    const auto gpu = static_cast<pl_gpu>(image.gpu);
    pl_buf_params params{};
    params.size = samples * sizeof(uint16_t);
    params.host_readable = true;
    params.memory_type = PL_BUF_MEM_HOST;
    pl_buf readback = pl_buf_create(gpu, &params);
    if (!readback) return false;
    pl_buf_copy(gpu, readback, 0, static_cast<pl_buf>(image.buffer), 0, params.size);
    const bool okay = pl_buf_read(gpu, readback, 0, pixels, params.size);
    pl_buf_destroy(gpu, &readback);
    return okay;
}
}
#else
namespace rawpreamp {
bool downscaleVulkanResidentImage(const VulkanResidentImage&, uint32_t, bool,
                                  VulkanResidentImage&) { return false; }
bool binVulkanResidentCfa(const VulkanResidentImage&, uint32_t, VulkanResidentImage&) { return false; }
bool downloadVulkanResidentImage(const VulkanResidentImage&, uint16_t*, size_t) { return false; }
}
#endif

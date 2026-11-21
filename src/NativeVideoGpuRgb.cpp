#include "DebugInstrumentation.h"
#ifdef __ANDROID__
#include "android/VulkanHardwareBuffer.h"
#include "android/VulkanInteropFence.h"
#include <media/NdkImage.h>
#include <dlfcn.h>
#endif
#include "NativeVideoGpuRgb.h"
#include "SharedVulkanDevice.h"
#include "VulkanGainBake.h"
#include "VulkanJpegFullEncode.h"
#include "VulkanGalleryPreview.h"

#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/pixdesc.h>
#ifdef RAWPREAMP_HAS_CUDA_INTEROP
#include <libavutil/hwcontext_cuda.h>
#endif
}
#include <libplacebo/filters.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/custom.h>
#define PL_LIBAV_IMPLEMENTATION 0
#include <libplacebo/utils/libav.h>
#include <libplacebo/vulkan.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef RAWPREAMP_HAS_CUDA_INTEROP
#include <dlfcn.h>
#include <unistd.h>
#endif
#ifdef __ANDROID__
#include <QDebug>
#endif

namespace {

void lockQueue(void* context, uint32_t family, uint32_t index) {
    auto* device = static_cast<AVHWDeviceContext*>(context);
    auto* vulkan = static_cast<AVVulkanDeviceContext*>(device->hwctx);
    vulkan->lock_queue(device, family, index);
}

void unlockQueue(void* context, uint32_t family, uint32_t index) {
    auto* device = static_cast<AVHWDeviceContext*>(context);
    auto* vulkan = static_cast<AVVulkanDeviceContext*>(device->hwctx);
    vulkan->unlock_queue(device, family, index);
}

#ifdef RAWPREAMP_HAS_CUDA_INTEROP
struct CudaDriver {
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    template<class T>
    T symbol(const char* name) const {
        return reinterpret_cast<T>(library ? dlsym(library, name) : nullptr);
    }
    decltype(&cuCtxPushCurrent) push = symbol<decltype(push)>("cuCtxPushCurrent_v2");
    decltype(&cuCtxPopCurrent) pop = symbol<decltype(pop)>("cuCtxPopCurrent_v2");
    decltype(&cuCtxGetDevice) device = symbol<decltype(device)>("cuCtxGetDevice");
    decltype(&cuDeviceGetUuid) uuid = symbol<decltype(uuid)>("cuDeviceGetUuid");
    decltype(&cuImportExternalMemory) import = symbol<decltype(import)>("cuImportExternalMemory");
    decltype(&cuExternalMemoryGetMappedBuffer) map = symbol<decltype(map)>("cuExternalMemoryGetMappedBuffer");
    decltype(&cuDestroyExternalMemory) destroy = symbol<decltype(destroy)>("cuDestroyExternalMemory");
    decltype(&cuMemFree) free = symbol<decltype(free)>("cuMemFree_v2");
    decltype(&cuMemcpyDtoDAsync) copy = symbol<decltype(copy)>("cuMemcpyDtoDAsync_v2");
    decltype(&cuStreamSynchronize) sync = symbol<decltype(sync)>("cuStreamSynchronize");

    bool available() const {
        return push && pop && device && uuid && import && map && destroy && free && copy && sync;
    }
};

const CudaDriver& cudaDriver() {
    static const CudaDriver driver;
    return driver;
}

struct CudaContext {
    const CudaDriver& driver = cudaDriver();
    bool active;
    explicit CudaContext(CUcontext context)
        : active(driver.available() && driver.push(context) == CUDA_SUCCESS) {}
    ~CudaContext() {
        CUcontext ignored;
        if (active) driver.pop(&ignored);
    }
};

struct CudaVulkanInput {
    CUexternalMemory memory = nullptr;
    CUdeviceptr address = 0;
    pl_buf buffer = nullptr;

    void reset(pl_gpu gpu, const CudaDriver& cuda) {
        if (address) cuda.free(address);
        if (memory) cuda.destroy(memory);
        address = 0;
        memory = nullptr;
        pl_buf_destroy(gpu, &buffer);
    }

    bool prepare(pl_vulkan vk, const CudaDriver& cuda, size_t bytes) {
        if (buffer && buffer->params.size >= bytes && address) return true;
        reset(vk->gpu, cuda);
        if (!(vk->gpu->export_caps.buf & PL_HANDLE_FD) || !vk->gpu->limits.buf_transfer)
            return false;

        CUdevice device;
        CUuuid cudaUuid;
        if (cuda.device(&device) != CUDA_SUCCESS || cuda.uuid(&cudaUuid, device) != CUDA_SUCCESS ||
            std::memcmp(cudaUuid.bytes, vk->gpu->uuid, VK_UUID_SIZE) != 0)
            return false;

        pl_buf_params params{};
        params.size = bytes;
        params.memory_type = PL_BUF_MEM_DEVICE;
        params.export_handle = PL_HANDLE_FD;
        buffer = pl_buf_create(vk->gpu, &params);
        if (!buffer) return false;

        const int fd = dup(buffer->shared_mem.handle.fd);
        if (fd < 0) return false;
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
        handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        handle.handle.fd = fd;
        handle.size = buffer->shared_mem.size;
        if (cuda.import(&memory, &handle) != CUDA_SUCCESS) {
            close(fd);
            return false;
        }
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC region{};
        region.offset = buffer->shared_mem.offset;
        region.size = bytes;
        return cuda.map(&address, memory, &region) == CUDA_SUCCESS;
    }

    bool upload(pl_vulkan vk, AVCUDADeviceContext& device, AVFrame* source,
                pl_tex textures[4], pl_frame& target) {
        CudaContext ctx(device.cuda_ctx);
        if (!ctx.active) return false;
        const auto* frames = reinterpret_cast<const AVHWFramesContext*>(source->hw_frames_ctx->data);
        const auto format = frames->sw_format;
        if (format != AV_PIX_FMT_P010LE && format != AV_PIX_FMT_NV12) return false;
        source->format = format;
        pl_frame_from_avframe(&target, source);
        source->format = AV_PIX_FMT_CUDA;
        pl_plane_data planes[4]{};
        if (pl_plane_data_from_pixfmt(planes, &target.repr.bits, format) != 2)
            return false;
        size_t size = 0;
        size_t planeBytes[2]{};
        for (int i = 0; i < 2; ++i) {
            auto& plane = planes[i];
            plane.width = i ? (source->width + 1) / 2 : source->width;
            plane.height = i ? (source->height + 1) / 2 : source->height;
            if (!source->data[i] || source->linesize[i] <= 0) return false;
            plane.row_stride = static_cast<size_t>(source->linesize[i]);
            const size_t rowBytes = plane.width * plane.pixel_stride;
            if (rowBytes > plane.row_stride) return false;
            planeBytes[i] = (plane.height - 1) * plane.row_stride + rowBytes;
            plane.buf_offset = size;
            size += (planeBytes[i] + 255) & ~size_t{255};
        }
        if (!prepare(vk, ctx.driver, size) || !pl_buf_export(vk->gpu, buffer))
            return false;
        while (pl_buf_poll(vk->gpu, buffer, UINT64_MAX)) {}
        bool copied = true;
        for (int i = 0; i < 2; ++i) {
            copied &= ctx.driver.copy(address + planes[i].buf_offset,
                reinterpret_cast<CUdeviceptr>(source->data[i]), planeBytes[i],
                device.stream) == CUDA_SUCCESS;
        }
        if (ctx.driver.sync(device.stream) != CUDA_SUCCESS || !copied) return false;
        for (int i = 0; i < 2; ++i) {
            planes[i].buf = buffer;
            if (!pl_upload_plane(vk->gpu, &target.planes[i], &textures[i], &planes[i]))
                return false;
        }
        return true;
    }
};
#endif

} // namespace
#endif

namespace rawpreamp {

const char* NativeVideoGpuRgb::transferShader() {
    // Quantization matches the RGBA16 intermediate used by the ordinary path.
    return R"glsl(
vec3 decodeNativeVideo(vec3 encoded, uint curve) {
    if (curve == 0u) return encoded;
    vec3 e = floor(clamp(encoded, 0.0, 1.0) * 65535.0 + 0.5) / 65535.0;
    if (curve == 1u)
        return mix(e * e / 3.0,
            (exp((e - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
            greaterThan(e, vec3(0.5)));
    if (curve == 3u) {
        // LLog60 inverse; keep constants in sync with the CPU curve.
        const float a = 0.980913053208808, m = 0.244314924085355, join = 0.971335276138720, k = 12.594751265244078;
        vec3 d = max(e - join, vec3(0.0));
        vec3 transition = vec3(0.96) + 2.0 * d / (m + sqrt(m*m + 4.0*k*d));
        vec3 lower = (pow(vec3(61.0), e / a) - 1.0) / 60.0;
        return mix(mix(lower, transition, greaterThan(e, vec3(join))), e,
            greaterThanEqual(e, vec3(0.99)));
    }
    return (pow(vec3(61.0), e) - 1.0) / 60.0;
}
)glsl";
}

struct NativeVideoGpuRgb::State {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#ifdef __ANDROID__
    struct AndroidImport {
        AHardwareBuffer* buffer = nullptr;
        uint64_t id = 0;
        bool hasId = false;
        bool fullRange = false, smoothChroma = false;
        std::unique_ptr<VulkanHardwareBuffer> hardware;
    };
    struct AndroidInput {
        pl_vulkan vk = nullptr;
        pl_tex texture = nullptr;
        VkImage output = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSemaphore available = VK_NULL_HANDLE;
        std::shared_ptr<AndroidImport> imported;
        ~AndroidInput() {
            // The external compute queue can still be writing this image
            // when a cancelled preview never submits a libplacebo read.
            if (imported) imported->hardware->waitForSampling();
            imported.reset();
            if (view) vkDestroyImageView(vk->device, view, nullptr);
            if (available) vkDestroySemaphore(vk->device, available, nullptr);
            if (texture) pl_tex_destroy(vk->gpu, &texture);
        }
    };
    std::vector<std::unique_ptr<AndroidInput>> androidInputs;
    std::vector<std::shared_ptr<AndroidImport>> androidImports;
#endif
    AVBufferRef* deviceRef = nullptr;
#ifdef RAWPREAMP_HAS_CUDA_INTEROP
    AVCUDADeviceContext* cudaDevice = nullptr;
    CudaVulkanInput cudaInput;
#endif
    AVFrame* output = nullptr;
    pl_log log = nullptr;
    pl_vulkan vulkan = nullptr;
    pl_renderer renderer = nullptr;
    pl_tex sourceTextures[4] = {};
    pl_tex sampleableOutput = nullptr;
    pl_frame outputTarget = {};
    bool outputTargetReady = false;
    pl_frame pendingTarget = {};
    bool pending = false;
    const pl_hook* hlgHook = nullptr;
    const pl_hook* log60Hook = nullptr;
    const pl_hook* llog60Hook = nullptr;
    const pl_hook* limited10Hook = nullptr;
    const pl_hook* mosaicHook = nullptr;
    std::array<uint8_t, 4> mosaicPhase{};
    bool bayer = false;
    bool rgbaOutput = false;
    std::vector<uint16_t> rgbaDownload;

    bool downloadRgb(const pl_frame& target, uint16_t* pixels) {
        if (!rgbaOutput)
            return pl_download_avframe(vulkan->gpu, &target, output);
        rgbaDownload.resize(static_cast<size_t>(width) * height * 4u);
        output->data[0] = reinterpret_cast<uint8_t*>(rgbaDownload.data());
        output->linesize[0] = width * 4 * sizeof(uint16_t);
        const bool ok = pl_download_avframe(vulkan->gpu, &target, output);
        if (ok) {
            const size_t count = static_cast<size_t>(width) * height;
            for (size_t i = 0; i < count; ++i)
                std::memcpy(pixels + i * 3u, rgbaDownload.data() + i * 4u,
                            3u * sizeof(uint16_t));
        }
        return ok;
    }
    int width = 0;
    int height = 0;

    bool prepareForSeek() {
        if (!vulkan) return true;
        // Finish work that may still reference the decoder's input surface
        // before FFmpeg reuses it. The renderer and its output textures are
        // independent of the input AVFrame, so keep them warm across seeks.
        pl_gpu_finish(vulkan->gpu);
        pending = false;
        pendingTarget = {};
        return renderer != nullptr;
    }

    ~State() {
        if (vulkan) {
            releaseVulkanGainBakeTextureDevice(vulkan->gpu);
            releaseVulkanJpegFullEncodeDevice(vulkan->gpu);
            releaseVulkanGalleryTextureDevice(vulkan->gpu);
        }
        if (vulkan) pl_gpu_finish(vulkan->gpu);
#ifdef RAWPREAMP_HAS_CUDA_INTEROP
        if (cudaDevice && cudaInput.buffer) {
            CudaContext ctx(cudaDevice->cuda_ctx);
            if (ctx.active) cudaInput.reset(vulkan->gpu, ctx.driver);
        }
#endif
#ifdef __ANDROID__
        androidInputs.clear();
        androidImports.clear();
#endif
        av_frame_free(&output);
        if (vulkan) {
            for (auto& texture : sourceTextures)
                pl_tex_destroy(vulkan->gpu, &texture);
            pl_tex_destroy(vulkan->gpu, &sampleableOutput);
        }
        pl_mpv_user_shader_destroy(&hlgHook);
        pl_mpv_user_shader_destroy(&log60Hook);
        pl_mpv_user_shader_destroy(&llog60Hook);
        pl_mpv_user_shader_destroy(&limited10Hook);
        pl_mpv_user_shader_destroy(&mosaicHook);
        pl_renderer_destroy(&renderer);
        pl_vulkan_destroy(&vulkan);
        pl_log_destroy(&log);
        av_buffer_unref(&deviceRef);
    }

    bool init(const AVFrame* source, int outputWidth, int outputHeight,
              bool outputBayer) {
        const bool softwareSource = !source->hw_frames_ctx;
        bool useOwnVulkan = softwareSource;
        AVHWFramesContext* frames = softwareSource ? nullptr :
            reinterpret_cast<AVHWFramesContext*>(source->hw_frames_ctx->data);
        if (!softwareSource && (!frames || !frames->device_ref)) return false;
        auto* device = softwareSource ? nullptr :
            reinterpret_cast<AVHWDeviceContext*>(frames->device_ref->data);
#ifdef RAWPREAMP_HAS_CUDA_INTEROP
        if (device && device->type == AV_HWDEVICE_TYPE_CUDA) {
            cudaDevice = static_cast<AVCUDADeviceContext*>(device->hwctx);
            useOwnVulkan = cudaDevice && cudaDriver().available();
        }
#endif
        if (!useOwnVulkan && (!device || device->type != AV_HWDEVICE_TYPE_VULKAN))
            return false;
        auto* hw = useOwnVulkan ? nullptr :
            static_cast<AVVulkanDeviceContext*>(device->hwctx);
        if (!useOwnVulkan && (!hw || !hw->lock_queue || !hw->unlock_queue))
            return false;

        pl_log_params logParams{};
        logParams.log_level = PL_LOG_WARN;
        logParams.log_cb = [](void*, pl_log_level, const char* message) {
            spdlog::warn("Native Video GPU conversion: {}", message);
        };
        log = pl_log_create(PL_API_VER, &logParams);
        if (!log) return false;
        if (useOwnVulkan) {
            vulkan = createGalleryVulkanContext(log);
        } else {
        pl_vulkan_import_params params = {};
        params.instance = hw->inst;
        params.get_proc_addr = hw->get_proc_addr;
        params.phys_device = hw->phys_dev;
        params.device = hw->act_dev;
        params.extensions = hw->enabled_dev_extensions;
        params.num_extensions = hw->nb_enabled_dev_extensions;
        params.features = &hw->device_features;
        params.lock_queue = lockQueue;
        params.unlock_queue = unlockQueue;
        params.queue_ctx = device;
        params.max_api_version = VK_API_VERSION_1_3;
        params.queue_graphics.index = VK_QUEUE_FAMILY_IGNORED;
        params.queue_compute.index = VK_QUEUE_FAMILY_IGNORED;
        params.queue_transfer.index = VK_QUEUE_FAMILY_IGNORED;
#if LIBAVUTIL_VERSION_MAJOR < 59
        // FFmpeg 6 exposes each Vulkan queue role as a separate field.
        // Later releases expose the queue families through qf[].
        if (hw->queue_family_index >= 0)
            params.queue_graphics = {uint32_t(hw->queue_family_index),
                                     uint32_t(hw->nb_graphics_queues)};
        if (hw->queue_family_comp_index >= 0)
            params.queue_compute = {uint32_t(hw->queue_family_comp_index),
                                    uint32_t(hw->nb_comp_queues)};
        if (hw->queue_family_tx_index >= 0)
            params.queue_transfer = {uint32_t(hw->queue_family_tx_index),
                                     uint32_t(hw->nb_tx_queues)};
#else
        for (int i = 0; i < hw->nb_qf; ++i) {
            const auto& queue = hw->qf[i];
            pl_vulkan_queue mapped = {};
            mapped.index = queue.idx;
            mapped.count = queue.num;
#if PL_API_VER >= 365
            mapped.flags = hw->queue_flags;
#endif
            if (queue.flags & VK_QUEUE_GRAPHICS_BIT) params.queue_graphics = mapped;
            if (queue.flags & VK_QUEUE_COMPUTE_BIT) params.queue_compute = mapped;
            if (queue.flags & VK_QUEUE_TRANSFER_BIT) params.queue_transfer = mapped;
        }
#endif
        vulkan = pl_vulkan_import(log, &params);
        }
        if (!vulkan) return false;
        renderer = pl_renderer_create(log, vulkan->gpu);
        if (!renderer) return false;
        static const std::string hlgShader =
            std::string("//!HOOK SCALED\n//!BIND HOOKED\n") + NativeVideoGpuRgb::transferShader() +
            "vec4 hook() { vec4 c = HOOKED_texOff(0); c.rgb = decodeNativeVideo(c.rgb, 1u); return c; }\n";
        static const std::string log60Shader =
            std::string("//!HOOK SCALED\n//!BIND HOOKED\n") + NativeVideoGpuRgb::transferShader() +
            "vec4 hook() { vec4 c = HOOKED_texOff(0); c.rgb = decodeNativeVideo(c.rgb, 2u); return c; }\n";
        static const std::string llog60Shader =
            std::string("//!HOOK SCALED\n//!BIND HOOKED\n") + NativeVideoGpuRgb::transferShader() +
            "vec4 hook() { vec4 c = HOOKED_texOff(0); c.rgb = decodeNativeVideo(c.rgb, 3u); return c; }\n";
        static constexpr char limited10Shader[] = R"(
//!HOOK NATIVE
//!BIND HOOKED
vec4 hook() {
    vec4 c = HOOKED_texOff(0);
    c.rgb *= 1020.0 / 1023.0;
    return c;
}
)";
        hlgHook = pl_mpv_user_shader_parse(vulkan->gpu, hlgShader.data(), hlgShader.size());
        log60Hook = pl_mpv_user_shader_parse(vulkan->gpu, log60Shader.data(), log60Shader.size());
        limited10Hook = pl_mpv_user_shader_parse(vulkan->gpu, limited10Shader,
                                                sizeof(limited10Shader) - 1);
        llog60Hook = pl_mpv_user_shader_parse(vulkan->gpu, llog60Shader.data(), llog60Shader.size());
        if (!hlgHook || !log60Hook || !llog60Hook || !limited10Hook) return false;
        // Android Vulkan drivers can mishandle three-component 16-bit render
        // targets when the texture is sampled by the gallery. Use a native
        // four-component target, then discard alpha on CPU download.
#ifdef __ANDROID__
        rgbaOutput = !outputBayer;
#endif
        const auto outputFormat = outputBayer ? AV_PIX_FMT_GRAY16LE :
            rgbaOutput ? AV_PIX_FMT_RGBA64LE : AV_PIX_FMT_RGB48LE;
        if (!pl_test_pixfmt(vulkan->gpu, outputFormat)) {
#ifdef __ANDROID__
            qWarning() << "Native Video Vulkan output format unsupported"
                       << int(outputFormat);
#endif
            return false;
        }
        if (frames) deviceRef = av_buffer_ref(frames->device_ref);
        output = av_frame_alloc();
        if ((!softwareSource && !deviceRef) || !output) return false;
        output->format = outputFormat;
        output->width = outputWidth;
        output->height = outputHeight;
        output->color_range = AVCOL_RANGE_JPEG;
        output->colorspace = AVCOL_SPC_RGB;
        output->color_primaries = AVCOL_PRI_BT2020;
        output->color_trc = AVCOL_TRC_LINEAR;
        width = outputWidth;
        height = outputHeight;
        bayer = outputBayer;
        return true;
    }

    bool render(const AVFrame* source, bool smoothChroma,
                NativeVideoGpuRgb::TransferCurve curve, bool fullRange,
                uint16_t* packedRgb,
                const std::array<uint8_t, 4>* cfaPhase,
                bool deferDownload, const pl_frame* externalInput = nullptr) {
        pending = false;
        if (!outputTargetReady &&
            !vulkanUsesUnifiedMemory(vulkan->instance, vulkan->phys_device, vulkan->get_proc_addr)) {
            // The renderer needs a destination plus intermediate textures.
            const size_t targetBytes = size_t(width) * height * 4 * sizeof(uint16_t);
            if (!vulkanDeviceMemoryFits(vulkan, 3 * targetBytes))
                return false;
        }
        output->data[0] = reinterpret_cast<uint8_t*>(packedRgb);
        output->linesize[0] = width * (cfaPhase ? 1 : rgbaOutput ? 4 : 3) * sizeof(uint16_t);
        AVFrame* tagged = av_frame_clone(source);
        if (!tagged) {
            output->data[0] = nullptr;
            output->linesize[0] = 0;
            return false;
        }
        tagged->color_trc = AVCOL_TRC_LINEAR;
        tagged->colorspace = AVCOL_SPC_BT2020_NCL;
        tagged->color_primaries = AVCOL_PRI_BT2020;
        tagged->color_range = fullRange ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;

        pl_avframe_params inputParams = {};
        inputParams.frame = tagged;
        if (!tagged->hw_frames_ctx) inputParams.tex = sourceTextures;
        inputParams.map_dovi = false;
        pl_frame input = externalInput ? *externalInput : pl_frame{};
        const auto mapStarted = rawpreamp::diagnosticNow();
        bool mapped;
        bool mappedInternally = !externalInput;
#ifdef RAWPREAMP_HAS_CUDA_INTEROP
        if (cudaDevice) {
            mappedInternally = false;
            mapped = cudaInput.upload(vulkan, *cudaDevice, tagged, sourceTextures, input);
        } else
#endif
            mapped = externalInput || pl_map_avframe_ex(vulkan->gpu, &input, &inputParams);
        av_frame_free(&tagged);
        if (!mapped) {
            output->data[0] = nullptr;
            output->linesize[0] = 0;
            return false;
        }
        // The decoder returns storage-oriented pixels. The gallery applies the
        // clip display transform later; rotating here stretches a 4:3 frame.
        input.rotation = PL_ROTATION_0;

        if (!outputTargetReady) {
            pl_frame target = {};
            pl_frame_from_avframe(&target, output);
            pl_plane_data planes[4]{};
            if (pl_plane_data_from_pixfmt(planes, &target.repr.bits,
                    static_cast<AVPixelFormat>(output->format)) == 1) {
                planes[0].width = output->width;
                planes[0].height = output->height;
                auto& plane = target.planes[0];
                const auto format = pl_plane_find_fmt(vulkan->gpu, plane.component_mapping, planes);
                pl_tex_params sampleable = {};
                sampleable.w = output->width;
                sampleable.h = output->height;
                sampleable.format = format;
                sampleable.renderable = true;
                sampleable.sampleable = true;
                sampleable.host_readable = true;
                if (format && pl_tex_recreate(vulkan->gpu, &sampleableOutput, &sampleable)) {
                    plane.components = 0;
                    for (int i = 0; i < 4; ++i)
                        if (plane.component_mapping[i] >= 0) plane.components = i + 1;
                    plane.texture = sampleableOutput;
                    target.num_planes = 1;
                    outputTarget = target;
                    outputTargetReady = true;
                }
            }
        }
        pl_frame target = outputTarget;
        bool rendered = false;
        if (outputTargetReady) {
            pl_render_params params = pl_render_default_params;
            params.upscaler = smoothChroma ? &pl_filter_mitchell : &pl_filter_nearest;
            // Proxy rendering averages the source area represented by each
            // output pixel. This also avoids ringing at high-contrast edges.
            params.downscaler = source->width > width || source->height > height
                ? &pl_filter_box : &pl_filter_mitchell;
            params.plane_upscaler = params.upscaler;
            params.peak_detect_params = nullptr;
            params.deband_params = nullptr;
            const pl_hook* transferHook = curve == NativeVideoGpuRgb::TransferCurve::HLG
                ? hlgHook : curve == NativeVideoGpuRgb::TransferCurve::LOG60
                    ? log60Hook : curve == NativeVideoGpuRgb::TransferCurve::LLOG60
                        ? llog60Hook : nullptr;
            const auto* frames = source->hw_frames_ctx
                ? reinterpret_cast<const AVHWFramesContext*>(
                    source->hw_frames_ctx->data) : nullptr;
            const pl_hook* hooks[3] = {};
            if (!fullRange && source->format == AV_PIX_FMT_VULKAN &&
                frames && frames->sw_format == AV_PIX_FMT_P010LE)
                hooks[params.num_hooks++] = limited10Hook;
            if (transferHook) hooks[params.num_hooks++] = transferHook;
            if (cfaPhase) {
                if (!mosaicHook || mosaicPhase != *cfaPhase) {
                    pl_mpv_user_shader_destroy(&mosaicHook);
                    mosaicPhase = *cfaPhase;
                    const std::string shader =
                        "//!HOOK SCALED\n//!BIND HOOKED\n"
                        "vec4 hook() {\n"
                        "    vec3 c = HOOKED_texOff(0).rgb;\n"
                        "    ivec2 p = ivec2(floor(HOOKED_pos * HOOKED_size));\n"
                        "    ivec4 phase = ivec4(" +
                        std::to_string((*cfaPhase)[0]) + "," +
                        std::to_string((*cfaPhase)[1]) + "," +
                        std::to_string((*cfaPhase)[2]) + "," +
                        std::to_string((*cfaPhase)[3]) + ");\n"
                        "    float v = c[phase[(p.y & 1) * 2 + (p.x & 1)]];\n"
                        "    return vec4(v, v, v, 1.0);\n"
                        "}\n";
                    mosaicHook = pl_mpv_user_shader_parse(vulkan->gpu,
                        shader.c_str(), shader.size());
                }
                if (!mosaicHook) {
                    if (mappedInternally) pl_unmap_avframe(vulkan->gpu, &input);
                    pl_gpu_flush(vulkan->gpu);
                    output->data[0] = nullptr;
                    output->linesize[0] = 0;
                    return false;
                }
                hooks[params.num_hooks++] = mosaicHook;
            }
            params.hooks = hooks;
            const auto renderStart = rawpreamp::diagnosticNow();
            const bool renderOk = pl_render_image(renderer, &input, &target, &params);
            // A failed intermediate allocation/hook can still return an image
            // with the transfer curve omitted. Do not publish it or reuse the
            // damaged renderer cache on the next frame (libplacebo 7.360 can
            // leave a null FBO in that cache). The caller releases this state
            // and converts the hardware-decoded frame on CPU instead.
            const auto errors = pl_renderer_get_errors(renderer).errors;
            const bool requiredPassFailed = errors & (PL_RENDER_ERR_FBO | PL_RENDER_ERR_HOOKS);
            if (requiredPassFailed)
                spdlog::warn("Native Video GPU conversion required render pass failed (flags={})",
                             static_cast<unsigned int>(errors));
            const auto downloadStart = rawpreamp::diagnosticNow();
            rendered = renderOk && !requiredPassFailed &&
                (deferDownload || downloadRgb(target, packedRgb));
            if (rendered && deferDownload) {
                pendingTarget = target;
                pending = true;
            }
            const auto downloadEnd = rawpreamp::diagnosticNow();
            if (rawpreamp::debugEnvironment("RAWPREAMP_NATIVE_VIDEO_GPU_RGB_PROFILE")) {
                const auto ms = [](auto a, auto b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                std::fprintf(stderr, "direct GPU RGB render_ms=%.3f download_ms=%.3f\n",
                             ms(renderStart, downloadStart), ms(downloadStart, downloadEnd));
            }
        }
        if (mappedInternally) pl_unmap_avframe(vulkan->gpu, &input);
        pl_gpu_flush(vulkan->gpu);
        output->data[0] = nullptr;
        output->linesize[0] = 0;
        return rendered;
    }
#endif
};

NativeVideoGpuRgb::NativeVideoGpuRgb() : mState(std::make_shared<State>()) {}
NativeVideoGpuRgb::~NativeVideoGpuRgb() = default;

bool NativeVideoGpuRgb::render(const AVFrame* source, int width, int height,
                             bool smoothChroma, TransferCurve curve,
                             bool fullRange, uint16_t* packedRgb,
                             const std::array<uint8_t, 4>* cfaPhase,
                             bool deferDownload) {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
    if (!source || !packedRgb) return false;
    auto* frames = source->hw_frames_ctx ?
        reinterpret_cast<AVHWFramesContext*>(source->hw_frames_ctx->data) : nullptr;
    if (source->hw_frames_ctx && (!frames || !frames->device_ref)) return false;
    if (!mState->vulkan ||
        bool(mState->deviceRef) != bool(frames) ||
        (frames && mState->deviceRef->data != frames->device_ref->data) ||
        mState->width != width || mState->height != height ||
        mState->bayer != static_cast<bool>(cfaPhase)) {
        mState = std::make_shared<State>();
        if (!mState->init(source, width, height, cfaPhase != nullptr)) return false;
    }
    return mState->render(source, smoothChroma, curve, fullRange, packedRgb,
                          cfaPhase, deferDownload);
#else
    (void)source; (void)width; (void)height; (void)smoothChroma;
    (void)curve; (void)fullRange; (void)packedRgb; (void)cfaPhase;
    (void)deferDownload;
    return false;
#endif
}

#ifdef __ANDROID__
bool NativeVideoGpuRgb::renderAndroidImage(const AVFrame* source, AImage* image,
        int acquireFenceFd, int width, int height, bool smoothChroma,
        TransferCurve curve, bool fullRange, uint16_t* packedRgb,
        const std::array<uint8_t, 4>* cfaPhase, bool deferDownload) {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
    auto fail = [&] {
        if (acquireFenceFd >= 0) ::close(acquireFenceFd);
        if (image) AImage_delete(image);
        return false;
    };
    if (!source || !image || !packedRgb || source->hw_frames_ctx) return fail();
    if (!mState->vulkan || mState->deviceRef || mState->width != width ||
        mState->height != height || mState->bayer != bool(cfaPhase)) {
        mState = std::make_shared<State>();
        if (!mState->init(source, width, height, bool(cfaPhase))) return fail();
    }
    AHardwareBuffer* hardware = nullptr;
    if (AImage_getHardwareBuffer(image, &hardware) != AMEDIA_OK) return fail();
    using GetBufferId = int (*)(const AHardwareBuffer*, uint64_t*);
    static const auto getBufferId = reinterpret_cast<GetBufferId>(
        dlsym(RTLD_DEFAULT, "AHardwareBuffer_getId"));
    uint64_t bufferId = 0;
    const bool hasBufferId = getBufferId && getBufferId(hardware, &bufferId) == 0;
    std::unique_ptr<State::AndroidInput> lease;
    for (auto it = mState->androidInputs.begin(); it != mState->androidInputs.end(); ++it) {
        if (!pl_tex_poll(mState->vulkan->gpu, (*it)->texture, 0)) {
            lease = std::move(*it);
            mState->androidInputs.erase(it);
            break;
        }
    }
    // Import caching must not retain a full-resolution RGBA image per codec
    // allocation. Reuse at most three output images independently of imports.
    if (!lease && mState->androidInputs.size() >= 3) {
        while (pl_tex_poll(mState->vulkan->gpu, mState->androidInputs.front()->texture,
                           UINT64_MAX)) {}
        lease = std::move(mState->androidInputs.front());
        mState->androidInputs.erase(mState->androidInputs.begin());
    }
    if (!lease) {
        lease = std::make_unique<State::AndroidInput>();
        lease->vk = mState->vulkan;
        pl_fmt format = pl_find_fmt(lease->vk->gpu, PL_FMT_UNORM, 4, 16, 16,
            static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_STORABLE));
        if (!format) return fail();
        pl_tex_params textureInfo{};
        textureInfo.w = source->width; textureInfo.h = source->height;
        textureInfo.format = format;
        textureInfo.sampleable = textureInfo.storable = true;
        lease->texture = pl_tex_create(lease->vk->gpu, &textureInfo);
        if (!lease->texture) return fail();
        VkFormat vkFormat = VK_FORMAT_UNDEFINED;
        lease->output = pl_vulkan_unwrap(lease->vk->gpu, lease->texture, &vkFormat, nullptr);
        if (vkFormat != VK_FORMAT_R16G16B16A16_UNORM) return fail();
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = lease->output; view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = vkFormat;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(lease->vk->device, &view, nullptr, &lease->view) != VK_SUCCESS)
            return fail();
        lease->available = createAndroidReleaseSemaphore(lease->vk);
        if (!lease->available) return fail();
    }
    pl_vulkan_hold_params hold{};
    hold.tex = lease->texture;
    hold.layout = VK_IMAGE_LAYOUT_GENERAL;
    hold.qf = VK_QUEUE_FAMILY_IGNORED;
    hold.semaphore = {lease->available, 0};
    const bool recycled = bool(lease->imported);
    if (recycled) {
        if (!pl_vulkan_hold_ex(lease->vk->gpu, &hold)) return fail();
        // A cancelled fused frame may never have submitted a read. Holding
        // the texture submits a wait for its previous sampling semaphore.
        // Finish that wait before making the import available to the cache.
        pl_gpu_finish(lease->vk->gpu);
        lease->imported.reset();
    }
    for (auto it = mState->androidImports.begin(); it != mState->androidImports.end();) {
        // Only reuse submission resources after their output read has completed.
        const bool sameBuffer = hasBufferId && (*it)->hasId
            ? (*it)->id == bufferId : (*it)->buffer == hardware;
        if (it->use_count() == 1 && sameBuffer &&
            (*it)->fullRange == fullRange && (*it)->smoothChroma == smoothChroma) {
            lease->imported = *it;
            break;
        }
        if (it->use_count() == 1 && mState->androidImports.size() >= 8)
            it = mState->androidImports.erase(it);
        else ++it;
    }
    if (!lease->imported) {
        auto imported = std::make_shared<State::AndroidImport>();
        imported->buffer = hardware;
        imported->id = bufferId;
        imported->hasId = hasBufferId;
        imported->fullRange = fullRange;
        imported->smoothChroma = smoothChroma;
        imported->hardware = std::make_unique<VulkanHardwareBuffer>();
        if (!imported->hardware->initialize(lease->vk, hardware, fullRange, smoothChroma) ||
            imported->hardware->width != static_cast<uint32_t>(source->width) ||
            imported->hardware->height != static_cast<uint32_t>(source->height)) return fail();
        lease->imported = imported;
        mState->androidImports.push_back(std::move(imported));
    }
    if (!recycled && !pl_vulkan_hold_ex(lease->vk->gpu, &hold)) return fail();
    const char* fusionSetting = rawpreamp::debugEnvironment("RAWPREAMP_ANDROID_FUSED_DECODE");
    const bool fused = deferDownload && !cfaPhase &&
        width == source->width && height == source->height &&
        (!fusionSetting || fusionSetting[0] != '0');
    VkSemaphore ready = VK_NULL_HANDLE;
    int releaseFd = -1;
    const int consumedFd = acquireFenceFd;
    acquireFenceFd = -1;
    if (!lease->imported->hardware->submitSampling(lease->output, lease->view, VK_IMAGE_LAYOUT_GENERAL,
            consumedFd, lease->available, &ready, &releaseFd,
            fused ? curve : TransferCurve::None)) {
        pl_gpu_finish(lease->vk->gpu);
        return fail();
    }
    AImage_deleteAsync(image, releaseFd);
    image = nullptr;
    pl_vulkan_release_params release{};
    release.tex = lease->texture;
    release.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    release.qf = VK_QUEUE_FAMILY_IGNORED;
    release.semaphore = {ready, 0};
    pl_vulkan_release_ex(lease->vk->gpu, &release);
    pl_frame input{};
    input.num_planes = 1;
    input.planes[0].texture = lease->texture;
    input.planes[0].components = 4;
    for (int i = 0; i < 4; ++i) input.planes[0].component_mapping[i] = i;
    input.repr = pl_color_repr_rgb;
    input.repr.alpha = PL_ALPHA_NONE;
    input.color = pl_color_space_bt2020_hlg;
    input.color.transfer = PL_COLOR_TRC_LINEAR;
    bool rendered = true;
    if (fused) {
        // YCbCr sampling already produced the linear RGB texture. Pass that
        // allocation to the gallery directly instead of rendering it again.
        mState->pendingTarget = input;
        mState->pending = true;
    } else {
        rendered = mState->render(source, smoothChroma, curve, fullRange,
            packedRgb, cfaPhase, deferDownload, &input);
    }
    if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_PROFILE"))
        spdlog::info("GALLERY_PERF event=android_decode_pipeline fused={} width={} height={}",
            fused, width, height);
    // Keep both the imported decoder allocation and output sampler alive until
    // libplacebo's queued reads have completed; recycle without a host wait.
    mState->androidInputs.push_back(std::move(lease));
    return rendered;
#else
    (void)source; (void)width; (void)height; (void)smoothChroma; (void)curve;
    (void)fullRange; (void)packedRgb; (void)cfaPhase; (void)deferDownload;
    if (acquireFenceFd >= 0) ::close(acquireFenceFd);
    if (image) AImage_delete(image);
    return false;
#endif
}
#endif

std::shared_ptr<void> NativeVideoGpuRgb::deviceLease() const { return mState; }

bool NativeVideoGpuRgb::pendingTexture(const void*& gpu, const void*& texture) const {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
    if (!mState->pending || !mState->vulkan || !mState->pendingTarget.planes[0].texture)
        return false;
    gpu = mState->vulkan->gpu;
    texture = mState->pendingTarget.planes[0].texture;
    return true;
#else
    (void)gpu; (void)texture;
    return false;
#endif
}

bool NativeVideoGpuRgb::downloadPending(uint16_t* pixels, size_t samples) {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
    if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_REQUIRE_ZERO_COPY")) {
        spdlog::error("Full-frame Native Video readback rejected by zero-copy validation");
        return false;
    }
    if (!mState->pending || !pixels || samples != static_cast<size_t>(mState->width) *
        mState->height * (mState->bayer ? 1u : 3u)) return false;
    mState->output->data[0] = reinterpret_cast<uint8_t*>(pixels);
    mState->output->linesize[0] = mState->width *
        (mState->bayer ? 1 : mState->rgbaOutput ? 4 : 3) * sizeof(uint16_t);
    const bool ok = mState->downloadRgb(mState->pendingTarget, pixels);
    mState->output->data[0] = nullptr;
    mState->output->linesize[0] = 0;
    mState->pending = false;
    return ok;
#else
    (void)pixels; (void)samples;
    return false;
#endif
}

bool NativeVideoGpuRgb::prepareForSeek() {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
    return mState->prepareForSeek();
#else
    return true;
#endif
}

} // namespace rawpreamp

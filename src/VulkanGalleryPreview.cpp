#include "VulkanSpatialGainCache.h"
#include "VulkanUploadCache.h"
#include "DebugInstrumentation.h"
#include "VulkanGalleryPreview.h"
#include "SharedVulkanDevice.h"
#include "DNGImage.h"
#include "RawLogTransform.h"
#include "VulkanGainBakeShader.h"

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif

namespace rawpreamp {
VulkanGalleryGpuBuffer::~VulkanGalleryGpuBuffer() {
    if (release) release();
    release = {};
    rejectSharedImage = {};
    resourceLease.reset();
    sourceLease.reset();
#ifndef _WIN32
    if (fd >= 0) ::close(fd);
#endif
}
}

#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include <libplacebo/vulkan.h>
#include <libplacebo/gpu.h>
#include <libplacebo/cache.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <future>
#include <filesystem>
#include <iterator>
#include <string>
#include <unordered_map>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <mutex>
#include <spdlog/spdlog.h>
#ifdef __ANDROID__
#include <QDebug>
#endif

namespace rawpreamp {
namespace {

#ifdef __ANDROID__
void galleryPlaceboLog(void*, pl_log_level level, const char* message) {
    if (level <= PL_LOG_ERR)
        qWarning().noquote() << "Gallery libplacebo:" << message;
}
#endif

// Five header vectors followed by four arrays of 32 vectors in GalleryBayerGain.
constexpr size_t bayerGainHeaderBytes = (5u + 4u * 32u) * 16u;

constexpr char shader[] = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Source { uint words[]; } src;
layout(std430, binding = 1) writeonly buffer Destination { uint words[]; } dst;
layout(push_constant) uniform Constants {
    uvec4 dimensions; // width, height, packed CFA phase, output word count
    vec4 black;
    vec4 inverseRange;
} p;

uint colorAt(int x, int y) {
    uint shift = uint(((y & 1) * 2 + (x & 1)) * 2);
    return (p.dimensions.z >> shift) & 3u;
}
uint sampleAt(int x, int y) {
    uint i = uint(y) * p.dimensions.x + uint(x);
    uint word = src.words[i >> 1u];
    return (i & 1u) == 0u ? word & 65535u : word >> 16u;
}
uint averageColor(int x, int y, uint channel) {
    uint native = colorAt(x, y);
    if (native == channel) return sampleAt(x, y);
    int dx[4] = int[4](-1, 1, 0, 0);
    int dy[4] = int[4](0, 0, -1, 1);
    int count = 4;
    if (native == 1u) {
        if (colorAt(x + 1, y) == channel) count = 2;
        else {
            dx[0] = 0; dx[1] = 0;
            dy[0] = -1; dy[1] = 1;
            count = 2;
        }
    } else if (channel != 1u) {
        dx[0] = -1; dy[0] = -1;
        dx[1] =  1; dy[1] = -1;
        dx[2] = -1; dy[2] =  1;
        dx[3] =  1; dy[3] =  1;
    }
    uint sum = 0u, valid = 0u;
    for (int i = 0; i < count; ++i) {
        int sx = x + dx[i], sy = y + dy[i];
        if (sx >= 0 && sy >= 0 && sx < int(p.dimensions.x) &&
            sy < int(p.dimensions.y)) {
            sum += sampleAt(sx, sy);
            valid++;
        }
    }
    return (sum + valid / 2u) / valid;
}
uint codeAt(uint sampleIndex) {
    uint channel = sampleIndex % 3u;
    uint pixel = sampleIndex / 3u;
    int x = int(pixel % p.dimensions.x), y = int(pixel / p.dimensions.x);
    uint value = averageColor(x, y, channel);
    float normalized = (float(value) - p.black[channel]) * p.inverseRange[channel];
    return uint(floor(clamp(normalized, 0.0, 1.0) * 65535.0 + 0.5));
}
void main() {
    uint word = gl_GlobalInvocationID.x;
    if (word >= p.dimensions.w) return;
    uint first = word * 2u;
    uint total = p.dimensions.x * p.dimensions.y * 3u;
    uint low = codeAt(first);
    uint high = first + 1u < total ? codeAt(first + 1u) : 0u;
    dst.words[word] = low | (high << 16u);
}
)glsl";

// Diagnostic-only full-buffer comparison; image data stays on the GPU.
constexpr char comparePreparedShader[] = R"glsl(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer A { uint words[]; } a;
layout(std430,binding=1) readonly buffer B { uint words[]; } b;
layout(std430,binding=2) buffer Result { uint maximum; uint different; } result;
layout(push_constant) uniform Size { uint samples; } p;
void main() {
    uint word=gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*gl_NumWorkGroups.x*256u;
    if(word*2u>=p.samples) return;
    uint x=a.words[word], y=b.words[word];
    for(uint lane=0u;lane<2u && word*2u+lane<p.samples;++lane) {
        uint shift=lane*16u;
        uint delta=uint(abs(int((x>>shift)&65535u)-int((y>>shift)&65535u)));
        if(delta!=0u) { atomicMax(result.maximum,delta); atomicAdd(result.different,1u); }
    }
}
)glsl";

constexpr char srgbLookupShader[] = R"glsl(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) writeonly buffer Lookup { float values[]; } dst;
void main() {
    uint index=gl_GlobalInvocationID.x;
    if(index>=65536u) return;
    float linear=float(index)/65535.0;
    float encoded=linear<=0.0031308 ? linear*12.92 :
        1.055*pow(linear,1.0/2.4)-0.055;
    dst.values[index]=floor(clamp(encoded,0.0,1.0)*65535.0+0.5);
}
)glsl";

constexpr char colorShader[] = R"glsl(#version 450
#ifdef GALLERY_IMAGE_OUTPUT
layout(local_size_x = 16, local_size_y = 16) in;
#else
layout(local_size_x = 256) in;
#endif
#if defined(SOURCE_TEXTURE) || defined(SOURCE_BAYER_TEXTURE)
layout(binding = 0) uniform sampler2D sourceImage;
#else
layout(std430, binding = 0) readonly buffer Source { uint words[]; } src;
#endif
#ifdef SOURCE_TEXTURE
layout(std430, binding = 4) readonly buffer GalleryGain {
    uvec4 shape; // width, height, encoded white, dither seed
    uvec4 bounds;
    vec4 axes; // origin H/V, spacing H/V
    uvec4 flags; // high quality dither, total map value count, secondary map present
    uvec4 secondaryShape; // width, height, channels, offset in values
    uvec4 secondaryBounds;
    vec4 secondaryAxes;
    float values[]; // gain grids, followed by linearization codes
} galleryGain;
#else
layout(std430, binding = 4) readonly buffer GalleryBayerGain {
    uvec4 header; // map count, LOG white, dither seed, flags
    vec4 black;
    vec4 scale;
    uvec4 lookup; // first float in values used by LOG linearization
    uvec4 sourceGeometry; // original width, height, total scale, Bayer parity offset
    uvec4 bounds[32];
    vec4 axes[32];
    uvec4 shape[32]; // width, height, channels, float offset
    uvec4 pitch[32];
    float values[];
} bayerGain;
#endif
#ifdef GALLERY_IMAGE_OUTPUT
#ifdef GALLERY_IMAGE_RGB10A2
layout(rgb10_a2, binding = 1) writeonly uniform image2D dstImage;
#else
layout(rgba8, binding = 1) writeonly uniform image2D dstImage;
#endif
#elif defined(GALLERY_GAIN_FIELD)
layout(std430, binding = 1) writeonly buffer Destination { float values[]; } dst;
#else
layout(std430, binding = 1) writeonly buffer Destination { uint words[]; } dst;
#endif
#if defined(GALLERY_CACHED_GAIN) || defined(GALLERY_CAPTURE_GAIN)
layout(std430, binding = 7)
#ifdef GALLERY_CAPTURE_GAIN
writeonly
#else
readonly
#endif
buffer CachedGain { float values[]; } cachedGain;
#endif
layout(std430, binding = 2) readonly buffer Profiles {
    uvec4 shapes[2]; // hue, saturation, value, encoding
    uvec4 offsets[2]; // first float, count
    float values[];
} profiles;
#ifdef GALLERY_PROFILE_TEXTURES
layout(binding = 5) uniform sampler3D hueTexture;
layout(binding = 6) uniform sampler3D lookTexture;
vec3 profileTexel(uint table, uint v, uint h, uint s) {
    return table == 0u ? texelFetch(hueTexture, ivec3(s,h,v), 0).rgb
                       : texelFetch(lookTexture, ivec3(s,h,v), 0).rgb;
}
#endif
layout(std430, binding = 3) readonly buffer Clipping { uint words[]; } clipping;
layout(push_constant) uniform Constants {
    uvec4 dimensions; // pixels, output words, mode, gain-map debug
    vec4 gains; // RGB gains, exposure
    vec4 matrixRows[3];
    vec4 xyzRows[3];
    uvec4 bayer; // width, height, packed phase, enabled
    vec4 bayerBlack;
    vec4 bayerInverseRange;
    uvec4 geometry; // input width, input height, rotation, flags
    vec4 textureBlack;
    vec4 textureInverseRange;
    uvec4 thumbnail; // output width, output height, enabled, unused
} p;
// @DISPLAY_MATH@
uint packedCode(uint sampleIndex) {
#ifdef SOURCE_TEXTURE
    uint pixel = sampleIndex / 3u, channel = sampleIndex % 3u;
    ivec2 xy = ivec2(int(pixel % p.geometry.x), int(pixel / p.geometry.x));
    float code = floor(clamp(texelFetch(sourceImage, xy, 0)[channel],
                             0.0, 1.0) * 65535.0 + 0.5);
    float normalized = (code - p.textureBlack[channel]) *
        p.textureInverseRange[channel];
    if (galleryGain.shape.x != 0u) {
        uint x = pixel % p.geometry.x, y = pixel / p.geometry.x;
        float gain = 1.0;
        if (x >= galleryGain.bounds.x && y >= galleryGain.bounds.y &&
            x < galleryGain.bounds.z && y < galleryGain.bounds.w) {
            uvec2 a, b;
            vec2 t;
            rawpreampGainGrid(uvec2(x, y), p.geometry.xy, galleryGain.shape.xy,
                             galleryGain.axes, a, b, t);
            uint stride = galleryGain.shape.x * 3u;
            uint ia = a.y * stride + a.x * 3u + channel;
            uint ib = a.y * stride + b.x * 3u + channel;
            uint ic = b.y * stride + a.x * 3u + channel;
            uint id = b.y * stride + b.x * 3u + channel;
            gain = RAWPREAMP_GAIN_BILERP(
                galleryGain.values[ia], galleryGain.values[ib],
                galleryGain.values[ic], galleryGain.values[id], t);
            gain = rawpreampGainOrIdentity(gain);
        }
        if (galleryGain.flags.z != 0u &&
            x >= galleryGain.secondaryBounds.x && y >= galleryGain.secondaryBounds.y &&
            x < galleryGain.secondaryBounds.z && y < galleryGain.secondaryBounds.w) {
            uvec2 a, b;
            vec2 t;
            rawpreampGainGrid(uvec2(x, y), p.geometry.xy, galleryGain.secondaryShape.xy,
                             galleryGain.secondaryAxes, a, b, t);
            uint stride = galleryGain.secondaryShape.x;
            uint start = galleryGain.secondaryShape.w;
            uint ia = start + a.y * stride + a.x;
            uint ib = start + a.y * stride + b.x;
            uint ic = start + b.y * stride + a.x;
            uint id = start + b.y * stride + b.x;
            float luminance = RAWPREAMP_GAIN_BILERP(
                galleryGain.values[ia], galleryGain.values[ib],
                galleryGain.values[ic], galleryGain.values[id], t);
            if (!isnan(luminance) && !isinf(luminance) && luminance > 0.0)
                gain *= luminance;
        }
        float linear = galleryGain.flags.w != 0u
            ? floor(clamp(1.0 / gain, 0.0, 1.0) * 65535.0 + 0.5)
            : floor(clamp(normalized * gain, 0.0, 1.0) * 65535.0 + 0.5);
        if (galleryGain.shape.z == 0u) return uint(linear);
        float dither = rawpreampDither(x, y, galleryGain.shape.w,
                                      galleryGain.flags.x != 0u);
        float encoded = rawpreampLog60(linear / 65535.0, galleryGain.shape.z);
        uint logCode = uint(floor(clamp(encoded + dither, 0.0,
            float(galleryGain.shape.z)) + 0.5));
        return uint(galleryGain.values[galleryGain.flags.y + logCode]);
    }
    return uint(floor(clamp(normalized, 0.0, 1.0) * 65535.0 + 0.5));
#else
#ifdef SOURCE_BAYER_TEXTURE
    ivec2 xy = ivec2(int(sampleIndex % p.bayer.x), int(sampleIndex / p.bayer.x));
    uint code = uint(floor(clamp(texelFetch(sourceImage, xy, 0).r, 0.0, 1.0) * 65535.0 + 0.5));
#else
    uint word = src.words[sampleIndex >> 1u];
    uint code = (sampleIndex & 1u) == 0u ? word & 65535u : word >> 16u;
#endif
#ifdef GALLERY_BAYER_PREPARE
    if (p.bayer.w != 0u && (bayerGain.header.w & 4u) != 0u)
        code = uint(bayerGain.values[bayerGain.lookup.y +
            min(code, bayerGain.lookup.z - 1u)]);
#endif
    return code;
#endif
}
uint bayerColor(int x, int y) {
    uint shift = uint(((y & 1) * 2 + (x & 1)) * 2);
    return (p.bayer.z >> shift) & 3u;
}
#ifndef SOURCE_TEXTURE
float bayerGainProduct(int x, int y) {
    uint phase = uint(((y & 1) << 1) | (x & 1));
    uint color = (bayerGain.header.w >> (8u + phase * 2u)) & 3u;
    uint sampleScale = max(1u, bayerGain.sourceGeometry.z);
    uvec2 sourcePixel = uvec2(x, y) * sampleScale;
    if (sampleScale > 1u)
        sourcePixel += (uvec2(x, y) & uvec2(1u)) * bayerGain.sourceGeometry.w;
    uvec2 sourceSize = bayerGain.sourceGeometry.xy;
    float gain = 1.0;
    for (uint mapIndex = 0u; mapIndex < bayerGain.header.x; ++mapIndex) {
        uvec4 bounds = bayerGain.bounds[mapIndex];
        if (sourcePixel.x < bounds.x || sourcePixel.y < bounds.y ||
            sourcePixel.x >= bounds.z || sourcePixel.y >= bounds.w) continue;
        uvec4 shape = bayerGain.shape[mapIndex];
        uvec4 pitch = bayerGain.pitch[mapIndex];
        if (shape.z == 1u &&
            ((sourcePixel.y - bounds.y) % pitch.x != 0u ||
             (sourcePixel.x - bounds.x) % pitch.y != 0u)) continue;
        vec4 axes = bayerGain.axes[mapIndex];
        uvec2 a, b;
        vec2 fraction;
        rawpreampGainGrid(sourcePixel, sourceSize, shape.xy, axes, a, b, fraction);
        uint channel = shape.z == 1u ? 0u : shape.z == 3u ? color : phase;
        uint ia = shape.w + (a.y * shape.x + a.x) * shape.z + channel;
        uint ib = shape.w + (a.y * shape.x + b.x) * shape.z + channel;
        uint ic = shape.w + (b.y * shape.x + a.x) * shape.z + channel;
        uint id = shape.w + (b.y * shape.x + b.x) * shape.z + channel;
        gain *= rawpreampGainOrIdentity(RAWPREAMP_GAIN_BILERP(
            bayerGain.values[ia], bayerGain.values[ib],
            bayerGain.values[ic], bayerGain.values[id], fraction));
    }
    return gain;
}
#endif
uint bayerCode(int x, int y) {
    uint raw = packedCode(uint(y) * p.bayer.x + uint(x));
#if defined(SOURCE_TEXTURE) || !defined(GALLERY_BAYER_PREPARE)
    // CFA gain/LUT/LOG work is completed by the preparation pass. Display
    // variants consume prepared samples and must not retain those branches.
    return raw;
#else
    if (p.bayer.w != 2u) return raw;
    uint phase = uint(((y & 1) << 1) | (x & 1));
#ifdef GALLERY_CACHED_GAIN
    float gain = cachedGain.values[uint(y) * p.bayer.x + uint(x)];
#else
    float gain = bayerGainProduct(x, y);
#ifdef GALLERY_CAPTURE_GAIN
    cachedGain.values[uint(y) * p.bayer.x + uint(x)] = gain;
#endif
#endif
    float linear = (bayerGain.header.w & 2u) != 0u
        ? clamp(1.0 / gain, 0.0, 1.0) * 65535.0
        : (float(raw) - bayerGain.black[phase]) *
          bayerGain.scale[phase] * gain;
    uint code = uint(floor(clamp(linear, 0.0, 65535.0) + 0.5));
    if (bayerGain.header.y == 0u) return code;
    float encoded = bayerGain.values[bayerGain.lookup.w + code];
    encoded += rawpreampDither(uint(x), uint(y), bayerGain.header.z,
                              (bayerGain.header.w & 1u) != 0u);
    uint logCode = uint(floor(clamp(encoded, 0.0,
                         float(bayerGain.header.y)) + 0.5));
    return uint(bayerGain.values[bayerGain.lookup.x + logCode]);
#endif
}
uint bayerAverage(int x, int y, uint channel) {
    uint native = bayerColor(x, y);
    if (native == channel) return bayerCode(x, y);
    int dx[4] = int[4](-1, 1, 0, 0);
    int dy[4] = int[4](0, 0, -1, 1);
    int count = 4;
    if (native == 1u) {
        if (bayerColor(x + 1, y) == channel) count = 2;
        else {
            dx[0] = 0; dx[1] = 0;
            dy[0] = -1; dy[1] = 1;
            count = 2;
        }
    } else if (channel != 1u) {
        dx[0] = -1; dy[0] = -1;
        dx[1] =  1; dy[1] = -1;
        dx[2] = -1; dy[2] =  1;
        dx[3] =  1; dy[3] =  1;
    }
    uint sum = 0u, valid = 0u;
    for (int i = 0; i < count; ++i) {
        int sx = x + dx[i], sy = y + dy[i];
        if (sx >= 0 && sy >= 0 && sx < int(p.bayer.x) && sy < int(p.bayer.y)) {
            sum += bayerCode(sx, sy);
            valid++;
        }
    }
    return (sum + valid / 2u) / valid;
}
#ifdef SOURCE_TEXTURE
// Fetch and normalize RGB once per pixel; gain coordinates and dither are
// identical across channels. Keep the scalar implementation for measured A/B.
vec3 textureGainProduct(uvec2 xy) {
    uint x = xy.x, y = xy.y;
    vec3 gain = vec3(1.0);
    if (x >= galleryGain.bounds.x && y >= galleryGain.bounds.y &&
        x < galleryGain.bounds.z && y < galleryGain.bounds.w) {
        uvec2 a, b;
        vec2 t;
        rawpreampGainGrid(uvec2(x, y), p.geometry.xy, galleryGain.shape.xy,
                         galleryGain.axes, a, b, t);
        uint stride = galleryGain.shape.x * 3u;
        uint ia = a.y * stride + a.x * 3u;
        uint ib = a.y * stride + b.x * 3u;
        uint ic = b.y * stride + a.x * 3u;
        uint id = b.y * stride + b.x * 3u;
        gain = RAWPREAMP_GAIN_BILERP(
            vec3(galleryGain.values[ia], galleryGain.values[ia + 1u], galleryGain.values[ia + 2u]), vec3(galleryGain.values[ib], galleryGain.values[ib + 1u], galleryGain.values[ib + 2u]),
            vec3(galleryGain.values[ic], galleryGain.values[ic + 1u], galleryGain.values[ic + 2u]), vec3(galleryGain.values[id], galleryGain.values[id + 1u], galleryGain.values[id + 2u]), t);
        gain = vec3(rawpreampGainOrIdentity(gain.r),
                    rawpreampGainOrIdentity(gain.g), rawpreampGainOrIdentity(gain.b));
    }
    if (galleryGain.flags.z != 0u &&
        x >= galleryGain.secondaryBounds.x && y >= galleryGain.secondaryBounds.y &&
        x < galleryGain.secondaryBounds.z && y < galleryGain.secondaryBounds.w) {
        uvec2 a, b;
        vec2 t;
        rawpreampGainGrid(uvec2(x, y), p.geometry.xy, galleryGain.secondaryShape.xy,
                         galleryGain.secondaryAxes, a, b, t);
        uint stride = galleryGain.secondaryShape.x;
        uint start = galleryGain.secondaryShape.w;
        uint ia = start + a.y * stride + a.x;
        uint ib = start + a.y * stride + b.x;
        uint ic = start + b.y * stride + a.x;
        uint id = start + b.y * stride + b.x;
        float luminance = RAWPREAMP_GAIN_BILERP(
            galleryGain.values[ia], galleryGain.values[ib],
            galleryGain.values[ic], galleryGain.values[id], t);
        if (!isnan(luminance) && !isinf(luminance) && luminance > 0.0)
            gain *= luminance;
    }
    return gain;
}
uvec3 textureCodes(uint pixel) {
    ivec2 xy = ivec2(int(pixel % p.geometry.x), int(pixel / p.geometry.x));
    vec3 code = floor(clamp(texelFetch(sourceImage, xy, 0).rgb,
                             0.0, 1.0) * 65535.0 + 0.5);
    vec3 normalized = (code - p.textureBlack.xyz) *
        p.textureInverseRange.xyz;
    if (galleryGain.shape.x != 0u) {
        uint x = pixel % p.geometry.x, y = pixel / p.geometry.x;
#ifdef GALLERY_CACHED_GAIN
        uint at = pixel * 3u;
        vec3 gain = vec3(cachedGain.values[at], cachedGain.values[at+1u], cachedGain.values[at+2u]);
#else
        vec3 gain = textureGainProduct(uvec2(x, y));
#endif
        vec3 linear = galleryGain.flags.w != 0u
            ? floor(clamp(1.0 / gain, 0.0, 1.0) * 65535.0 + 0.5)
            : floor(clamp(normalized * gain, 0.0, 1.0) * 65535.0 + 0.5);
        if (galleryGain.shape.z == 0u) return uvec3(linear);
        float dither = rawpreampDither(x, y, galleryGain.shape.w,
                                      galleryGain.flags.x != 0u);
        vec3 encoded = rawpreampLog60(linear / 65535.0, galleryGain.shape.z);
        // One scalar sample is broadcast to RGB: the hash has no channel term.
        uvec3 logCode = uvec3(floor(clamp(encoded + dither, 0.0,
            float(galleryGain.shape.z)) + 0.5));
        return uvec3(galleryGain.values[galleryGain.flags.y + logCode.r],
                     galleryGain.values[galleryGain.flags.y + logCode.g],
                     galleryGain.values[galleryGain.flags.y + logCode.b]);
    }
    return uvec3(floor(clamp(normalized, 0.0, 1.0) * 65535.0 + 0.5));
}
#endif
uvec3 sourceCodes(uint pixel) {
    uint first = pixel * 3u;
#ifdef SOURCE_TEXTURE
#ifndef GALLERY_RGB_SCALAR
    return textureCodes(pixel);
#else
    return uvec3(packedCode(first), packedCode(first + 1u), packedCode(first + 2u));
#endif
#else
    if (p.bayer.w == 0u)
        return uvec3(packedCode(first), packedCode(first + 1u), packedCode(first + 2u));
    if (p.bayer.w == 3u) {
        vec3 raw = vec3(packedCode(first), packedCode(first + 1u), packedCode(first + 2u));
        vec3 normalized = (raw - p.textureBlack.xyz) * p.textureInverseRange.xyz;
        return uvec3(floor(clamp(normalized, vec3(0.0), vec3(1.0)) * 65535.0 + 0.5));
    }
    // Reconstruct RGB together so integer Bayer coordinate division is done
    // once per output pixel, rather than once for every color component.
    int x = int(pixel % p.bayer.x), y = int(pixel / p.bayer.x);
    vec3 raw;
#ifndef GALLERY_BAYER_LOOPS
    // Fixed gathers avoid divergent neighbor loops. Clamp missing neighbors
    // for safe loads, then mask them out to preserve the scalar edge average.
    int right = min(x + 1, int(p.bayer.x) - 1);
    int bottom = min(y + 1, int(p.bayer.y) - 1);
    int left = max(x - 1, 0), top = max(y - 1, 0);
    uint hasLeft = uint(x > 0), hasRight = uint(x + 1 < int(p.bayer.x));
    uint hasTop = uint(y > 0), hasBottom = uint(y + 1 < int(p.bayer.y));
    uint nx = hasLeft + hasRight, ny = hasTop + hasBottom;
    uint center = bayerCode(x, y);
    uint horizontal = bayerCode(left, y) * hasLeft + bayerCode(right, y) * hasRight;
    uint vertical = bayerCode(x, top) * hasTop + bayerCode(x, bottom) * hasBottom;
    uint h = (horizontal + nx / 2u) >> uint(nx == 2u);
    uint v = (vertical + ny / 2u) >> uint(ny == 2u);
    uint count = nx + ny;
    uint crossSum = horizontal + vertical;
    uint green = count == 4u ? (crossSum + 2u) >> 2u :
                 count == 2u ? (crossSum + 1u) >> 1u : (crossSum + 1u) / 3u;
    uint diagonal = bayerCode(left, top) * hasLeft * hasTop +
                    bayerCode(right, top) * hasRight * hasTop +
                    bayerCode(left, bottom) * hasLeft * hasBottom +
                    bayerCode(right, bottom) * hasRight * hasBottom;
    uint d = (diagonal + (nx * ny) / 2u) >> (uint(nx == 2u) + uint(ny == 2u));
    uint native = bayerColor(x, y);
    bool redHorizontal = bayerColor(x + 1, y) == 0u;
    raw = vec3(native == 0u ? center : native == 1u ? (redHorizontal ? h : v) : d,
               native == 1u ? center : green,
               native == 2u ? center : native == 1u ? (redHorizontal ? v : h) : d);
#else
    raw = vec3(bayerAverage(x, y, 0u), bayerAverage(x, y, 1u), bayerAverage(x, y, 2u));
#endif
    vec3 normalized = (raw - p.bayerBlack.xyz) * p.bayerInverseRange.xyz;
    return uvec3(floor(clamp(normalized, vec3(0.0), vec3(1.0)) * 65535.0 + 0.5));
#endif
}
uint encode(float value) {
    uint index = uint(floor(clamp(value, 0.0, 1.0) * 65535.0 + 0.5));
    return uint(profiles.values[profiles.offsets[0].w + index]);
}
float srgbEncode(float value) {
    value = max(0.0, value);
    return value <= 0.0031308 ? value * 12.92 :
        1.055 * pow(value, 1.0 / 2.4) - 0.055;
}
float srgbDecode(float value) {
    value = max(0.0, value);
    return value <= 0.04045 ? value / 12.92 :
        pow((value + 0.055) / 1.055, 2.4);
}
float wrappedHue(float hue) {
    hue = mod(hue, 360.0);
    return hue < 0.0 ? hue + 360.0 : hue;
}
float tableValue(uint table, uint v, uint h, uint s, uint channel) {
    uvec4 shape = profiles.shapes[table];
    uint index = (((v * shape.x + h) * shape.y + s) * 3u + channel);
    return profiles.values[profiles.offsets[table].x + index];
}
vec3 tableSample(uint table, float hue, float saturation, float value) {
    uvec4 shape = profiles.shapes[table];
    if (shape.x == 0u || shape.y == 0u || shape.z == 0u)
        return vec3(0.0, 1.0, 1.0);
    float h = wrappedHue(hue) * float(shape.x) / 360.0;
    float s = clamp(saturation, 0.0, 1.0) * float(shape.y - 1u);
    float v = shape.z > 1u ? clamp(value, 0.0, 1.0) * float(shape.z - 1u) : 0.0;
    uint h0 = uint(h) % shape.x, h1 = h0 + 1u;
    if (h1 == shape.x) h1 = 0u;
    uint s0 = uint(s), s1 = min(s0 + 1u, shape.y - 1u);
    uint v0 = uint(v), v1 = min(v0 + 1u, shape.z - 1u);
    float hf = h - floor(h), sf = s - float(s0), vf = v - float(v0);
#ifdef GALLERY_PROFILE_TEXTURES
    vec3 lower = mix(
        mix(profileTexel(table,v0,h0,s0), profileTexel(table,v0,h0,s1), sf),
        mix(profileTexel(table,v0,h1,s0), profileTexel(table,v0,h1,s1), sf), hf);
    if (v0 == v1) return lower;
    vec3 upper = mix(
        mix(profileTexel(table,v1,h0,s0), profileTexel(table,v1,h0,s1), sf),
        mix(profileTexel(table,v1,h1,s0), profileTexel(table,v1,h1,s1), sf), hf);
    return mix(lower, upper, vf);
#else
    vec3 result;
    for (uint channel = 0u; channel < 3u; ++channel) {
        float lower = mix(
            mix(tableValue(table, v0, h0, s0, channel),
                tableValue(table, v0, h0, s1, channel), sf),
            mix(tableValue(table, v0, h1, s0, channel),
                tableValue(table, v0, h1, s1, channel), sf), hf);
        if (v0 == v1) {
            result[channel] = lower;
            continue;
        }
        float upper = mix(
            mix(tableValue(table, v1, h0, s0, channel),
                tableValue(table, v1, h0, s1, channel), sf),
            mix(tableValue(table, v1, h1, s0, channel),
                tableValue(table, v1, h1, s1, channel), sf), hf);
        result[channel] = mix(lower, upper, vf);
    }
    return result;
#endif
}
vec3 rgbToHsv(vec3 rgb) {
    float maximum = max(max(rgb.r, rgb.g), max(rgb.b, 0.0));
    float minimum = max(0.0, min(rgb.r, min(rgb.g, rgb.b)));
    float delta = maximum - minimum;
    float hue = 0.0;
    if (delta > 1e-8) {
        if (maximum == rgb.r) hue = 60.0 * (rgb.g - rgb.b) / delta;
        else if (maximum == rgb.g) hue = 60.0 * (2.0 + (rgb.b - rgb.r) / delta);
        else hue = 60.0 * (4.0 + (rgb.r - rgb.g) / delta);
    }
    return vec3(wrappedHue(hue), maximum > 1e-8 ? delta / maximum : 0.0,
                maximum);
}
vec3 hsvToRgb(vec3 hsv) {
    float chroma = hsv.z * hsv.y;
    float sector = wrappedHue(hsv.x) / 60.0;
    float x = chroma * (1.0 - abs(sector - 2.0 * floor(sector * 0.5) - 1.0));
    vec3 rgb;
    if (sector < 1.0) rgb = vec3(chroma, x, 0.0);
    else if (sector < 2.0) rgb = vec3(x, chroma, 0.0);
    else if (sector < 3.0) rgb = vec3(0.0, chroma, x);
    else if (sector < 4.0) rgb = vec3(0.0, x, chroma);
    else if (sector < 5.0) rgb = vec3(x, 0.0, chroma);
    else rgb = vec3(chroma, 0.0, x);
    return rgb + vec3(hsv.z - chroma);
}
vec3 applyTable(vec3 rgb, uint table) {
    if (profiles.shapes[table].x == 0u) return rgb;
    vec3 hsv = rgbToHsv(rgb);
    bool encoded = profiles.shapes[table].z > 1u &&
        profiles.shapes[table].w == 1u;
    if (encoded) hsv.z = srgbEncode(hsv.z);
    vec3 adjustment = tableSample(table, hsv.x, hsv.y, hsv.z);
    hsv.x = wrappedHue(hsv.x + adjustment.x);
    hsv.y = clamp(hsv.y * adjustment.y, 0.0, 1.0);
    hsv.z = clamp(hsv.z * adjustment.z, 0.0, 1.0);
    vec3 result = hsvToRgb(hsv);
    if (encoded) result = vec3(srgbDecode(result.r), srgbDecode(result.g),
                               srgbDecode(result.b));
    return result;
}
vec3 applyProfiles(vec3 camera) {
    vec3 xyz = vec3(dot(p.xyzRows[0].xyz, camera),
                    dot(p.xyzRows[1].xyz, camera),
                    dot(p.xyzRows[2].xyz, camera));
    vec3 proPhoto = vec3(
        1.3459433 * xyz.x - 0.2556075 * xyz.y - 0.0511118 * xyz.z,
       -0.5445989 * xyz.x + 1.5081673 * xyz.y + 0.0205351 * xyz.z,
        1.2118128 * xyz.z);
    bool combined = profiles.shapes[0].x != 0u &&
        profiles.shapes[1].x != 0u &&
        profiles.shapes[0].z == 1u &&
        profiles.shapes[0].w == 0u && profiles.shapes[1].w == 0u &&
        p.gains.w > 0.0 && !isnan(p.gains.w) && !isinf(p.gains.w);
    if (combined) {
        vec3 hsv = rgbToHsv(proPhoto);
        vec3 hueAdjustment = tableSample(0u, hsv.x, hsv.y, hsv.z);
        hsv.x = wrappedHue(hsv.x + hueAdjustment.x);
        hsv.y = clamp(hsv.y * hueAdjustment.y, 0.0, 1.0);
        hsv.z = clamp(hsv.z * hueAdjustment.z, 0.0, 1.0) * p.gains.w;
        vec3 lookAdjustment = tableSample(1u, hsv.x, hsv.y, hsv.z);
        hsv.x = wrappedHue(hsv.x + lookAdjustment.x);
        hsv.y = clamp(hsv.y * lookAdjustment.y, 0.0, 1.0);
        hsv.z = clamp(hsv.z * lookAdjustment.z, 0.0, 1.0);
        proPhoto = hsvToRgb(hsv);
    } else {
        proPhoto = applyTable(proPhoto, 0u) * p.gains.w;
        proPhoto = applyTable(proPhoto, 1u);
    }
    xyz = vec3(
        0.7976749 * proPhoto.x + 0.1351917 * proPhoto.y +
            0.0313534 * proPhoto.z,
        0.2880402 * proPhoto.x + 0.7118741 * proPhoto.y +
            0.0000857 * proPhoto.z,
        0.82521 * proPhoto.z);
    return vec3(
         3.1338561 * xyz.x - 1.6168667 * xyz.y - 0.4906146 * xyz.z,
        -0.9787684 * xyz.x + 1.9161415 * xyz.y + 0.0334540 * xyz.z,
         0.0719453 * xyz.x - 0.2289914 * xyz.y + 1.4052427 * xyz.z);
}
uvec3 transform(uint pixel) {
    uvec3 codes = sourceCodes(pixel);
    vec3 camera = vec3(codes) / 65535.0;
    vec3 display;
    if (p.dimensions.z == 2u) display = applyProfiles(camera);
    else if (p.dimensions.z != 0u) {
        display = vec3(dot(p.matrixRows[0].xyz, camera),
                       dot(p.matrixRows[1].xyz, camera),
                       dot(p.matrixRows[2].xyz, camera));
    } else display = camera * p.gains.xyz;
    if (p.dimensions.w == 0u &&
        (codes.x == 65535u || codes.y == 65535u || codes.z == 65535u))
        display.y = max(display.y, min(display.x, display.z));
    float t = p.dimensions.w != 0u ? 0.0 :
        clamp((float(min(codes.x, min(codes.y, codes.z))) / 65535.0 - 0.90) /
              0.10, 0.0, 1.0);
    float weight = t * t * (3.0 - 2.0 * t);
    float neutral = display.x + display.y + display.z -
        min(display.x, min(display.y, display.z)) -
        max(display.x, max(display.y, display.z));
    return uvec3(encode(mix(display.x, neutral, weight)),
                 encode(mix(display.y, neutral, weight)),
                 encode(mix(display.z, neutral, weight)));
}
uint sourcePixel(uint outputPixel) {
    uint outputWidth = (p.geometry.z == 90u || p.geometry.z == 270u)
        ? p.geometry.y : p.geometry.x;
    uint x = outputPixel % outputWidth, y = outputPixel / outputWidth;
    if (p.thumbnail.z != 0u) {
        uint outputHeight = (p.geometry.z == 90u || p.geometry.z == 270u)
            ? p.geometry.x : p.geometry.y;
        x = min(outputWidth - 1u,
            ((2u * (outputPixel % p.thumbnail.x) + 1u) * outputWidth) /
            (2u * p.thumbnail.x));
        y = min(outputHeight - 1u,
            ((2u * (outputPixel / p.thumbnail.x) + 1u) * outputHeight) /
            (2u * p.thumbnail.y));
    }
    if (p.geometry.z == 90u)
        return (p.geometry.y - 1u - x) * p.geometry.x + y;
    if (p.geometry.z == 180u)
        return (p.geometry.y - 1u - y) * p.geometry.x +
            (p.geometry.x - 1u - x);
    if (p.geometry.z == 270u)
        return x * p.geometry.x + (p.geometry.x - 1u - y);
    return y * p.geometry.x + x;
}
uvec3 outputColorFromSource(uint source) {
    uvec3 color = transform(source);
#ifndef GALLERY_CLIPPING
    return color;
#else
    if ((p.geometry.w & 2u) == 0u) return color;
    uint indication;
#ifdef SOURCE_TEXTURE
    if ((p.geometry.w & 16u) != 0u) {
        ivec2 xy = ivec2(int(source % p.geometry.x), int(source / p.geometry.x));
        vec3 codes = floor(clamp(texelFetch(sourceImage, xy, 0).rgb,
                                0.0, 1.0) * 65535.0 + 0.5);
        vec3 white = floor(p.textureBlack.rgb +
            1.0 / p.textureInverseRange.rgb + 0.5);
        bvec3 high = greaterThanEqual(codes, white);
        uint bits = (high.r ? 1u : 0u) | (high.g ? 2u : 0u) | (high.b ? 4u : 0u);
        indication = any(lessThan(codes, p.textureBlack.rgb))
            ? 1u : bits != 0u ? 2u + bits : 0u;
    } else
#endif
    {
        uint word = clipping.words[source >> 2u];
        indication = (word >> ((source & 3u) * 8u)) & 255u;
    }
    if (indication == 0u) return color;
    if (indication == 1u) return uvec3(32768u, 16384u, 0u);
    uint bits = indication > 1u ? indication - 2u : 0u;
    if (bits == 7u) return uvec3(0u);
    return uvec3((bits & 1u) != 0u ? 65535u : 0u,
                 (bits & 2u) != 0u ? 65535u : 0u,
                 (bits & 4u) != 0u ? 65535u : 0u);
#endif
}
uvec3 outputColor(uint outputPixel) {
    return outputColorFromSource(sourcePixel(outputPixel));
}
void main() {
    uint group = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y *
        gl_NumWorkGroups.x * gl_WorkGroupSize.x;
#ifdef GALLERY_GAIN_FIELD
    if (group < p.dimensions.x) {
#ifdef SOURCE_TEXTURE
        vec3 gain = textureGainProduct(uvec2(group % p.geometry.x, group / p.geometry.x));
        dst.values[group*3u] = gain.r;
        dst.values[group*3u+1u] = gain.g;
        dst.values[group*3u+2u] = gain.b;
#else
        dst.values[group] = bayerGainProduct(int(group % p.bayer.x), int(group / p.bayer.x));
#endif
    }
#else
#if !defined(GALLERY_IMAGE_OUTPUT) && !defined(GALLERY_BAYER_PREPARE)
    uint rowWidth = (p.geometry.z == 90u || p.geometry.z == 270u)
        ? p.geometry.y : p.geometry.x;
    uint perInvocation = (p.geometry.w & 13u) != 0u ? 4u : 2u;
    if (p.thumbnail.z == 0u && (p.geometry.z == 90u || p.geometry.z == 270u) && rowWidth % perInvocation == 0u) {
        uint columns = rowWidth / perInvocation;
        uint gx = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x % 16u;
        uint gy = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.x / 16u;
        if (gx >= columns) return;
        group = gy * columns + gx;
    }
#endif
#ifdef GALLERY_BAYER_PREPARE
    uint first = group * 2u;
    if (first >= p.dimensions.x) return;
    uint lo = bayerCode(int(first % p.bayer.x), int(first / p.bayer.x));
    uint hi = first + 1u < p.dimensions.x
        ? bayerCode(int((first + 1u) % p.bayer.x), int((first + 1u) / p.bayer.x)) : 0u;
    dst.words[group] = lo | (hi << 16u);
    return;
#else
#ifdef GALLERY_IMAGE_OUTPUT
    // Traverse source tiles so portrait rotation cannot turn every Bayer
    // neighbour load into an uncached full-row stride.
    uvec2 sourceXY = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(sourceXY, p.geometry.xy))) return;
    uint outputWidth = (p.geometry.z == 90u || p.geometry.z == 270u)
        ? p.geometry.y : p.geometry.x;
    uvec2 outputXY = sourceXY;
    if (p.geometry.z == 90u) outputXY = uvec2(p.geometry.y - 1u - sourceXY.y, sourceXY.x);
    else if (p.geometry.z == 180u) outputXY = p.geometry.xy - uvec2(1u) - sourceXY;
    else if (p.geometry.z == 270u) outputXY = uvec2(sourceXY.y, p.geometry.x - 1u - sourceXY.x);
#ifdef GALLERY_REPEAT_COORDINATES
    uvec3 color = outputColor(outputXY.y * outputWidth + outputXY.x);
#else
    // This invocation already owns the source coordinate. Avoid converting
    // to a rotated output index and dividing/rotating it back for sampling.
    uvec3 color = outputColorFromSource(sourceXY.y * p.geometry.x + sourceXY.x);
#endif
    ivec2 xy = ivec2(outputXY);
#ifdef GALLERY_IMAGE_RGB10A2
    imageStore(dstImage, xy, vec4(vec3(color) / 65535.0, 1.0));
#else
    imageStore(dstImage, xy, vec4(vec3(color >> uvec3(8u)) / 255.0, 1.0));
#endif
    return;
#endif
#ifndef GALLERY_IMAGE_OUTPUT
    if ((p.geometry.w & 4u) != 0u) {
        uint first = group * 4u;
        for (uint i = 0u; i < 4u && first + i < p.dimensions.x; ++i) {
            uvec3 color = (outputColor(first + i) * 1023u + 32767u) / 65535u;
            dst.words[first + i] = color.x | (color.y << 10u) |
                (color.z << 20u) | 0xc0000000u;
        }
        return;
    }
    if ((p.geometry.w & 8u) != 0u) {
        uint first = group * 4u;
        for (uint i = 0u; i < 4u && first + i < p.dimensions.x; ++i) {
            uvec3 color = outputColor(first + i) >> uvec3(8u);
            dst.words[first + i] = color.x | (color.y << 8u) |
                (color.z << 16u) | 0xff000000u;
        }
        return;
    }
    if ((p.geometry.w & 1u) != 0u) {
        uint first = group * 4u;
        if (first >= p.dimensions.x) return;
        uint codes[12];
        for (uint i = 0u; i < 12u; ++i) codes[i] = 0u;
        for (uint i = 0u; i < 4u && first + i < p.dimensions.x; ++i) {
            uvec3 color = outputColor(first + i) >> uvec3(8u);
            codes[i * 3u] = color.x;
            codes[i * 3u + 1u] = color.y;
            codes[i * 3u + 2u] = color.z;
        }
        for (uint i = 0u; i < 3u; ++i) {
            uint word = group * 3u + i;
            if (word >= p.dimensions.y) break;
            uint j = i * 4u;
            dst.words[word] = codes[j] | (codes[j + 1u] << 8u) |
                (codes[j + 2u] << 16u) | (codes[j + 3u] << 24u);
        }
    } else {
        uint first = group * 2u;
        if (first >= p.dimensions.x) return;
        uvec3 a = outputColor(first);
        uint word = group * 3u;
        dst.words[word] = a.x | (a.y << 16u);
        if (first + 1u < p.dimensions.x) {
            uvec3 b = outputColor(first + 1u);
            dst.words[word + 1u] = a.z | (b.x << 16u);
            dst.words[word + 2u] = b.y | (b.z << 16u);
        } else dst.words[word + 1u] = a.z;
    }
#endif
#endif // GALLERY_BAYER_PREPARE
#endif
}
)glsl";

std::string galleryColorShader(bool textureSource, bool imageOutput = false,
                               bool imageRgb10a2 = false, bool bayerTexture = false,
                               bool clipping = true, bool prepareBayer = false, bool profileTextures = false,
                               bool gainField = false, bool cachedGain = false, bool captureGain = false) {
    const char* newline = std::strchr(colorShader, '\n');
    if (!newline) return {};
    std::string source = textureSource
        ? std::string(colorShader, newline + 1) +
            "#define SOURCE_TEXTURE 1\n" + (newline + 1)
        : std::string(colorShader);
    if (rawpreamp::debugEnvironmentIsSet("RAWPREAMP_GALLERY_RGB_SCALAR"))
        source.insert(source.find('\n') + 1, "#define GALLERY_RGB_SCALAR 1\n");
    if (bayerTexture) {
        source = std::string(colorShader);
        source.insert(source.find('\n') + 1, "#define SOURCE_BAYER_TEXTURE 1\n");
    }
    if (rawpreamp::debugEnvironmentIsSet("RAWPREAMP_GALLERY_BAYER_LOOPS"))
        source.insert(source.find('\n') + 1, "#define GALLERY_BAYER_LOOPS 1\n");
    if (imageOutput && rawpreamp::debugEnvironmentIsSet("RAWPREAMP_GALLERY_REPEAT_COORDINATES"))
        source.insert(source.find('\n') + 1, "#define GALLERY_REPEAT_COORDINATES 1\n");
    if (imageOutput) {
        const auto newlinePos = source.find('\n');
        source.insert(newlinePos + 1, "#define GALLERY_IMAGE_OUTPUT 1\n");
        if (imageRgb10a2)
            source.insert(newlinePos + 1, "#define GALLERY_IMAGE_RGB10A2 1\n");
    }
    if (clipping)
        source.insert(source.find('\n') + 1, "#define GALLERY_CLIPPING 1\n");
    if (profileTextures)
        source.insert(source.find('\n') + 1, "#define GALLERY_PROFILE_TEXTURES\n");
    if (prepareBayer)
        source.insert(source.find('\n') + 1, "#define GALLERY_BAYER_PREPARE 1\n");
    if (gainField)
        source.insert(source.find('\n') + 1, "#define GALLERY_GAIN_FIELD 1\n");
    if (cachedGain)
        source.insert(source.find('\n') + 1, "#define GALLERY_CACHED_GAIN 1\n");
    if (captureGain)
        source.insert(source.find('\n') + 1, "#define GALLERY_CAPTURE_GAIN 1\n");
    constexpr char marker[] = "// @DISPLAY_MATH@";
    const auto position = source.find(marker);
    if (position == std::string::npos) return {};
    source.replace(position, sizeof(marker) - 1, vulkanGainDisplayMathShader);
    return source;
}

struct Constants {
    uint32_t dimensions[4];
    float black[4];
    float inverseRange[4];
};
struct ColorConstants {
    uint32_t dimensions[4];
    float gains[4];
    float matrixRows[12];
    float xyzRows[12];
    uint32_t bayer[4];
    float bayerBlack[4];
    float bayerInverseRange[4];
    uint32_t geometry[4];
    float textureBlack[4];
    float textureInverseRange[4];
    uint32_t thumbnail[4];
};

struct GalleryShaderCache {
    std::mutex ioMutex;
    std::filesystem::path directory;
    pl_cache cache;
    static constexpr size_t maxObjectBytes = 8u * 1024u * 1024u;

    GalleryShaderCache() {
        pl_cache_params params{};
        params.max_object_size = maxObjectBytes;
        params.max_total_size = 32u * 1024u * 1024u;
        params.priv = this;
        params.get = [](void* priv, uint64_t key) {
            auto& self = *static_cast<GalleryShaderCache*>(priv);
            std::lock_guard lock(self.ioMutex);
            if (self.directory.empty()) return pl_cache_obj{};
            auto prefix = (self.directory / "").u8string();
            // libplacebo validates the header, key and checksum, and removes
            // incomplete or corrupt entries instead of passing them to Vulkan.
            return pl_cache_get_file(prefix.data(), key);
        };
        params.set = [](void* priv, pl_cache_obj object) {
            auto& self = *static_cast<GalleryShaderCache*>(priv);
            std::lock_guard lock(self.ioMutex);
            if (self.directory.empty()) return;
            char key[17];
            std::snprintf(key, sizeof(key), "%016llx", static_cast<unsigned long long>(object.key));
            const auto path = self.directory / key;
            std::error_code error;
            if (!object.size) {
                std::filesystem::remove(path, error);
                return;
            }
#ifdef _WIN32
            const auto pid = _getpid();
#else
            const auto pid = ::getpid();
#endif
            auto prefix = (self.directory / (std::to_string(pid) + ".tmp-")).u8string();
            const auto temporary = std::filesystem::u8path(prefix + key);
            // Each process writes its own temporary entry. Publish only after
            // the library has finished writing its checksummed cache format.
            std::filesystem::remove(temporary, error);
            pl_cache_set_file(prefix.data(), object);
            std::filesystem::rename(temporary, path, error);
#ifdef _WIN32
            // Windows rename does not replace an existing cache entry.
            if (error && std::filesystem::exists(temporary, error)) {
                std::filesystem::remove(path, error);
                std::filesystem::rename(temporary, path, error);
            }
#endif
            std::filesystem::remove(temporary, error);
        };
        cache = pl_cache_create(&params);
    }
    void configure(const std::string& path) {
        std::lock_guard lock(ioMutex);
        if (path.empty()) return;
        const auto candidate = std::filesystem::u8path(path);
        std::error_code error;
        std::filesystem::create_directories(candidate, error);
        if (!error) directory = candidate;
    }
};

GalleryShaderCache& galleryShaderCache() {
    // Shared across decoder devices and kept alive with the process-wide GPU
    // cache. libplacebo keys compiled objects for their shader/compiler/device.
    static auto* cache = new GalleryShaderCache;
    return *cache;
}

struct Device {
    struct HostFrame {
        std::shared_ptr<uint8_t> memory;
        pl_buf buffer = nullptr;
        size_t size = 0;
    };
    struct ThumbnailReadback {
        pl_buf output = nullptr;
        pl_buf download = nullptr;
        size_t capacity = 0;
        size_t byteCount = 0;
        uint32_t width = 0, height = 0;
        bool clipping = false;
        int sourceFrame = -1;
        uint64_t generation = 0;
        bool pending = false;
    };
    pl_log log = nullptr;
    pl_vulkan vk = nullptr;
    pl_gpu gpu = nullptr;
    bool textureSource = false;
    bool bufferSource = false;
    bool bayerTextureSource = false;
    bool unifiedMemory = false;
    bool directHostInput = false;
    bool externalBufferRejected = false;
    std::atomic_bool sharedTextureRejected{false};
    pl_pass pass = nullptr, colorPass = nullptr, unclippedColorPass = nullptr;
    pl_pass colorImagePass = nullptr, colorImage10Pass = nullptr;
    pl_pass bayerPreparePass = nullptr, bayerPrepareCachedPass = nullptr, gainFieldPass = nullptr;
    pl_pass bayerTexturePreparePass = nullptr, bayerTexturePrepareCachedPass = nullptr;
    pl_pass bayerPrepareCapturePass = nullptr, bayerTexturePrepareCapturePass = nullptr;
    VulkanSpatialGainCache spatialGains;
    pl_pass comparePreparedPass = nullptr;
    pl_buf comparePrepared = nullptr, compareResult = nullptr, compareStage = nullptr;
    pl_buf preparedBayer = nullptr;
    pl_timer prepareTimer = nullptr, colorTimer = nullptr;
    pl_buf source = nullptr, upload = nullptr, output = nullptr, download = nullptr;
    ThumbnailReadback thumbnailReadback;
    std::array<ThumbnailReadback,3> thumbnailReadbacks{};
    size_t nextThumbnailReadback = 0;
    pl_buf profiles = nullptr, profilesUpload = nullptr;
    pl_buf clipping = nullptr, clippingUpload = nullptr;
    size_t previousClippingBytes=0;
    pl_buf gain = nullptr, gainUpload = nullptr;
    VulkanUploadCache gainCache, profileCache;
    pl_buf emptyGain = nullptr, srgbLookup = nullptr;
    std::array<pl_pass,8> profileTexturePasses{};
    std::array<pl_pass,2> cachedColorPasses{}, cachedColorImagePasses{};
    bool profileTextureRejected = false;
    struct ProfileTexture {
        pl_tex image = nullptr;
        std::vector<float> values;
        uint32_t width = 0, height = 0, depth = 0;
        bool update(pl_gpu gpu, const DNGProfileTable* table) {
            const bool empty = !table || table->values.empty();
            const uint32_t w = empty ? 1u : table->saturationDivisions;
            const uint32_t h = empty ? 1u : table->hueDivisions;
            const uint32_t d = empty ? 1u : table->valueDivisions;
            if (std::max({w, h, d}) > gpu->limits.max_tex_3d_dim ||
                uint64_t(w)*h*d > gpu->limits.max_ssbo_size / (4u*sizeof(float))) return false;
            if (image && width == w && height == h && depth == d &&
                (empty ? values.empty() : values == table->values)) return true;
            const auto format = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
            if (!format) return false;
            std::vector<float> rgba(size_t(w) * h * d * 4u, 0.0f);
            if (empty) rgba = {0, 1, 1, 0};
            else for (size_t i = 0; i < table->values.size() / 3u; ++i)
                std::copy_n(table->values.data() + i*3u, 3u, rgba.data() + i*4u);
            width = height = depth = 0;
            pl_tex_params params{};
            params.w = w; params.h = h; params.d = d;
            params.format = format; params.sampleable = params.host_writable = true;
            if (!pl_tex_recreate(gpu, &image, &params)) return false;
            pl_tex_transfer_params upload{};
            upload.tex = image; upload.ptr = rgba.data();
            if (!pl_tex_upload(gpu, &upload)) return false;
            width = w;
            height = h;
            depth = d;
            values = empty ? std::vector<float>{} : table->values;
            return true;
        }
    };
    std::array<ProfileTexture,2> profileTextures;
    std::vector<HostFrame> hostFrames;
    ~Device() {
        if (gpu) {
            pl_gpu_finish(gpu);
            for (auto& frame : hostFrames)
                pl_buf_destroy(gpu, &frame.buffer);
            pl_buf_destroy(gpu, &source);
            pl_buf_destroy(gpu, &upload);
            pl_buf_destroy(gpu, &output);
            pl_buf_destroy(gpu, &download);
            pl_buf_destroy(gpu, &thumbnailReadback.output);
            pl_buf_destroy(gpu, &thumbnailReadback.download);
            for(auto& slot:thumbnailReadbacks){
                pl_buf_destroy(gpu,&slot.output);
                pl_buf_destroy(gpu,&slot.download);
            }
            pl_buf_destroy(gpu, &profiles);
            pl_buf_destroy(gpu, &profilesUpload);
            pl_buf_destroy(gpu, &clipping);
            pl_buf_destroy(gpu, &clippingUpload);
            pl_buf_destroy(gpu, &gain);
            pl_buf_destroy(gpu, &gainUpload);
            pl_buf_destroy(gpu, &emptyGain);
            pl_buf_destroy(gpu, &srgbLookup);
            for (auto& profile : profileTextures) pl_tex_destroy(gpu, &profile.image);
            for (auto& pass : profileTexturePasses) pl_pass_destroy(gpu, &pass);
            for (auto& pass : cachedColorPasses) pl_pass_destroy(gpu, &pass);
            for (auto& pass : cachedColorImagePasses) pl_pass_destroy(gpu, &pass);
            pl_buf_destroy(gpu, &preparedBayer);
            pl_pass_destroy(gpu, &bayerPreparePass);
            pl_pass_destroy(gpu, &bayerPrepareCachedPass);
            pl_pass_destroy(gpu, &bayerTexturePreparePass);
            pl_pass_destroy(gpu, &bayerTexturePrepareCachedPass);
            pl_pass_destroy(gpu, &bayerPrepareCapturePass);
            pl_pass_destroy(gpu, &bayerTexturePrepareCapturePass);
            pl_pass_destroy(gpu, &gainFieldPass);
            spatialGains.reset();
            pl_pass_destroy(gpu, &comparePreparedPass);
            pl_buf_destroy(gpu, &comparePrepared);
            pl_buf_destroy(gpu, &compareResult);
            pl_buf_destroy(gpu, &compareStage);
            pl_timer_destroy(gpu, &prepareTimer);
            pl_timer_destroy(gpu, &colorTimer);
            pl_pass_destroy(gpu, &pass);
            pl_pass_destroy(gpu, &colorPass);
            pl_pass_destroy(gpu, &unclippedColorPass);
            pl_pass_destroy(gpu, &colorImagePass);
            pl_pass_destroy(gpu, &colorImage10Pass);
        }
        pl_vulkan_destroy(&vk);
        pl_log_destroy(&log);
    }
    bool init() {
#ifdef __ANDROID__
        pl_log_params logParams = {};
        logParams.log_cb = galleryPlaceboLog;
        logParams.log_level = PL_LOG_ERR;
        log = pl_log_create(PL_API_VER, &logParams);
#else
        const auto logParams = galleryVulkanLogParams();
        log = pl_log_create(PL_API_VER, &logParams);
#endif
        if (!log) return false;
        vk = createGalleryVulkanContext(log);
        if (!vk || !vk->gpu->glsl.compute) return false;
        gpu = vk->gpu;
        pl_gpu_set_cache(gpu, galleryShaderCache().cache);
        return initResources();
    }
    bool ensureDemosaicPass() {
        if (pass) return true;
        pl_desc descriptors[2] = {};
        descriptors[0] = {"src", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY};
        descriptors[1] = {"dst", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_WRITEONLY};
        pl_pass_params params = {};
        params.type = PL_PASS_COMPUTE;
        params.descriptors = descriptors;
        params.num_descriptors = 2;
        params.push_constants_size = sizeof(Constants);
        params.glsl_shader = shader;
        pass = pl_pass_create(gpu, &params);
        return pass != nullptr;
    }
    bool initExternal(pl_gpu externalGpu, bool sampled = true, bool bayerTexture = false) {
        if (!externalGpu || !externalGpu->glsl.compute) return false;
        gpu = externalGpu;
        textureSource = sampled;
        bufferSource = !sampled;
        bayerTextureSource = bayerTexture;
        pl_gpu_set_cache(gpu, galleryShaderCache().cache);
        return initResources();
    }
    bool initResources() {
        if (const pl_vulkan vulkan = pl_vulkan_get(gpu))
            unifiedMemory = vulkanUsesUnifiedMemory(vulkan->instance,
                vulkan->phys_device, vulkan->get_proc_addr);
        // Every shader reads this binding, including plain RGB with no bake.
        const std::array<uint8_t, bayerGainHeaderBytes> zero{};
        pl_buf_params gainParams{};
        gainParams.size = zero.size();
        gainParams.storable = gainParams.host_writable = true;
        gainParams.initial_data = zero.data();
        emptyGain = pl_buf_create(gpu, &gainParams);
        if (!emptyGain) return false;
        // Generate on the same GPU with the original transfer function. Only
        // the quantized 16-bit input code indexes this immutable table.
        pl_buf_params lookupParams{};
        lookupParams.size = 65536u * sizeof(float);
        lookupParams.storable = true;
        lookupParams.memory_type = PL_BUF_MEM_DEVICE;
        srgbLookup = pl_buf_create(gpu, &lookupParams);
        if (!srgbLookup) return false;
        pl_desc descriptor{"dst", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_WRITEONLY};
        pl_pass_params params{};
        params.type = PL_PASS_COMPUTE;
        params.glsl_shader = srgbLookupShader;
        params.descriptors = &descriptor; params.num_descriptors = 1;
        pl_pass lookupPass = pl_pass_create(gpu, &params);
        if (!lookupPass) return false;
        pl_desc_binding binding{}; binding.object = srgbLookup;
        pl_pass_run_params run{};
        run.pass = lookupPass; run.desc_bindings = &binding;
        run.compute_groups[0] = 256;
        run.compute_groups[1] = run.compute_groups[2] = 1;
        pl_pass_run(gpu, &run);
        pl_pass_destroy(gpu, &lookupPass);
        return true;
    }
    pl_pass ensureColorPass(bool clipping, bool textured = false, bool cachedGain = false) {
        if (textured && !profileTextureRejected) {
            auto& pass = profileTexturePasses[(cachedGain ? 4 : 0) + (clipping ? 1 : 0)];
            if (!pass && !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_TEST_DCP_TEXTURE_FAIL"))
                pass = createColorPass(false, false, clipping, false, true, false, cachedGain);
            if (pass) return pass;
            profileTextureRejected = true;
        }
        auto& selected = cachedGain ? cachedColorPasses[clipping ? 1 : 0]
                                    : clipping ? colorPass : unclippedColorPass;
        if (!selected) selected = createColorPass(false, false, clipping, false, false, false, cachedGain);
        return selected;
    }
    pl_pass createColorPass(bool image = false, bool rgb10a2 = false, bool clipping = true,
                            bool prepareBayer = false, bool profileTextures = false,
                            bool field = false, bool cachedGain = false, bool textureBayerInput = false,
                            bool captureGain = false) {
        const bool sampled = textureSource || textureBayerInput;
        const bool bayerTexture = bayerTextureSource || textureBayerInput;
        pl_desc descriptors[8] = {};
        descriptors[0] = {sampled ? "sourceImage" : "src",
                          sampled ? PL_DESC_SAMPLED_TEX : PL_DESC_BUF_STORAGE,
                          0, PL_DESC_ACCESS_READONLY};
        descriptors[1] = {image ? "dstImage" : "dst",
                          image ? PL_DESC_STORAGE_IMG : PL_DESC_BUF_STORAGE,
                          1, PL_DESC_ACCESS_WRITEONLY};
        descriptors[2] = {"profiles", PL_DESC_BUF_STORAGE,
                          2, PL_DESC_ACCESS_READONLY};
        descriptors[3] = {"clipping", PL_DESC_BUF_STORAGE,
                          3, PL_DESC_ACCESS_READONLY};
        descriptors[4] = {sampled && !bayerTexture ? "galleryGain" : "bayerGain", PL_DESC_BUF_STORAGE,
                          4, PL_DESC_ACCESS_READONLY};
        descriptors[5] = {"hueTexture", PL_DESC_SAMPLED_TEX, 5, PL_DESC_ACCESS_READONLY};
        descriptors[6] = {"lookTexture", PL_DESC_SAMPLED_TEX, 6, PL_DESC_ACCESS_READONLY};
        // A cached pass is used only with a complete field, never a placeholder.
        if (cachedGain || captureGain)
            descriptors[profileTextures ? 7 : 5] = {"cachedGain", PL_DESC_BUF_STORAGE, 7,
                captureGain ? PL_DESC_ACCESS_WRITEONLY : PL_DESC_ACCESS_READONLY};
        const std::string source = galleryColorShader(sampled, image, rgb10a2, bayerTexture,
            clipping, prepareBayer, profileTextures, field, cachedGain, captureGain);
        if (source.empty()) return nullptr;
        pl_pass_params params = {};
        params.type = PL_PASS_COMPUTE;
        params.descriptors = descriptors;
        params.num_descriptors = (profileTextures ? 7 : 5) + ((cachedGain || captureGain) ? 1 : 0);
        params.push_constants_size = sizeof(ColorConstants);
        params.glsl_shader = source.c_str();
        return pl_pass_create(gpu, &params);
    }
    pl_pass ensureColorImagePass(bool rgb10a2, bool textured = false, bool cachedGain = false) {
        if (textured && !profileTextureRejected) {
            auto& pass = profileTexturePasses[(cachedGain ? 4 : 0) + (rgb10a2 ? 3 : 2)];
            if (!pass && !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_TEST_DCP_TEXTURE_FAIL"))
                pass = createColorPass(true, rgb10a2, true, false, true, false, cachedGain);
            if (pass) return pass;
            profileTextureRejected = true;
        }
        auto& imagePass = cachedGain ? cachedColorImagePasses[rgb10a2 ? 1 : 0]
                                     : rgb10a2 ? colorImage10Pass : colorImagePass;
        if (!imagePass)
            imagePass = createColorPass(true, rgb10a2, true, false, false, false, cachedGain);
        return imagePass;
    }
    bool reserveOutput(size_t bytes) {
        pl_buf_params params{};
        params.size = bytes;
        params.storable = true;
        params.memory_type = PL_BUF_MEM_DEVICE;
        return pl_buf_recreate(gpu, &output, &params);
    }
    bool reserve(size_t sourceBytes, size_t outputBytes, bool outputRequired = true) {
        if ((!textureSource && sourceBytes > gpu->limits.max_ssbo_size) ||
            outputBytes > gpu->limits.max_ssbo_size) return false;
        pl_buf_params src = {}, srcStage = {};
        src.size = sourceBytes;
        src.storable = true;
        src.memory_type = PL_BUF_MEM_DEVICE;
        srcStage.size = sourceBytes;
        srcStage.host_writable = true;
        srcStage.memory_type = PL_BUF_MEM_HOST;
        if (textureSource || bufferSource)
            return !outputRequired || reserveOutput(outputBytes);
        // On unified memory consume the host-visible SSBO directly. Keep the
        // dedicated-GPU staging path, also used if allocation is unsupported.
        directHostInput = false;
        if (unifiedMemory) {
            auto directParams = srcStage;
            directParams.storable = true;
            directHostInput = pl_buf_recreate(gpu, &upload, &directParams);
            if (directHostInput) {
                pl_buf_destroy(gpu, &source);
                return !outputRequired || reserveOutput(outputBytes);
            }
        }
        if (!pl_buf_recreate(gpu, &source, &src) ||
            !pl_buf_recreate(gpu, &upload, &srcStage)) {
            return false;
        }
        return !outputRequired || reserveOutput(outputBytes);
    }
    pl_tex createOutputTexture(uint32_t width, uint32_t height,
                               bool rgb10a2) {
        const pl_fmt format = rgb10a2 ? pl_find_named_fmt(gpu, "rgb10a2")
            : pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8,
                static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_STORABLE));
        if (!format || (format->caps & (PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_STORABLE)) !=
                (PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_STORABLE)) {
#ifdef __ANDROID__
            qWarning() << "Gallery Vulkan shared-image format unavailable" << rgb10a2;
#endif
            return nullptr;
        }
        pl_tex_params params{};
        params.w = static_cast<int>(width);
        params.h = static_cast<int>(height);
        params.format = format;
        params.sampleable = true;
        params.storable = true;
        pl_tex result = pl_tex_create(gpu, &params);
#ifdef __ANDROID__
        if (!result) qWarning() << "Gallery Vulkan shared-image allocation failed"
                                << format->name << width << height;
#endif
        return result;
    }
    bool reserveDownload(size_t bytes) {
        pl_buf_params params = {};
        params.size = bytes;
        params.host_readable = true;
        params.memory_type = PL_BUF_MEM_HOST;
        return pl_buf_recreate(gpu, &download, &params);
    }
    bool reserveThumbnail(ThumbnailReadback& slot,size_t bytes) {
        if(bytes>gpu->limits.max_ssbo_size)return false;
        if(slot.capacity==bytes&&slot.output&&slot.download)return true;
        pl_buf_params outputParams={},readParams={};
        outputParams.size=bytes;
        outputParams.storable=true;
        outputParams.memory_type=PL_BUF_MEM_DEVICE;
        readParams.size=bytes;
        readParams.host_readable=true;
        readParams.memory_type=PL_BUF_MEM_HOST;
        if(!pl_buf_recreate(gpu,&slot.output,&outputParams)||
           !pl_buf_recreate(gpu,&slot.download,&readParams))return false;
        slot.capacity=bytes;
        return true;
    }
    bool acquireHostFrame(size_t bytes, pl_buf& buffer,
                          std::shared_ptr<uint8_t>& memory) {
        if (!textureSource || !(gpu->import_caps.buf & PL_HANDLE_HOST_PTR))
            return false;
        for (auto& frame : hostFrames) {
            // Qt's QImage keeps another reference until it has finished with
            // the previous frame; only the pool's sole reference is reusable.
            if (frame.size >= bytes && frame.memory.use_count() == 1) {
                buffer = frame.buffer;
                memory = frame.memory;
                return true;
            }
        }
        for (auto it = hostFrames.begin(); it != hostFrames.end();) {
            if (it->memory.use_count() == 1 && it->size < bytes) {
                pl_buf_destroy(gpu, &it->buffer);
                it = hostFrames.erase(it);
            } else ++it;
        }
        if (hostFrames.size() >= 8u) return false;
        auto* pixels = new (std::nothrow) uint8_t[bytes];
        if (!pixels) return false;
        std::shared_ptr<uint8_t> owner(pixels, std::default_delete<uint8_t[]>());
        pl_buf_params params = {};
        params.size = bytes;
        params.host_readable = true;
        params.import_handle = PL_HANDLE_HOST_PTR;
        params.shared_mem.handle.ptr = pixels;
        params.shared_mem.size = bytes;
        pl_buf imported = pl_buf_create(gpu, &params);
        if (!imported) return false;
        hostFrames.push_back({std::move(owner), imported, bytes});
        buffer = imported;
        memory = hostFrames.back().memory;
        return true;
    }
    bool reserveProfiles(size_t bytes) {
        if (!profiles || profiles->params.size < bytes) profileCache.invalidate();
        if (bytes > gpu->limits.max_ssbo_size) return false;
        pl_buf_params params = {}, uploadParams = {};
        params.size = bytes;
        params.storable = true;
        params.memory_type = PL_BUF_MEM_DEVICE;
        uploadParams.size = bytes;
        uploadParams.host_writable = true;
        uploadParams.memory_type = PL_BUF_MEM_HOST;
        return ((profiles && profiles->params.size >= bytes) ||
                pl_buf_recreate(gpu, &profiles, &params)) &&
            pl_buf_recreate(gpu, &profilesUpload, &uploadParams);
    }
    bool reserveClipping(size_t bytes) {
        if (bytes > gpu->limits.max_ssbo_size) return false;
        pl_buf_params params = {}, uploadParams = {};
        params.size = bytes;
        params.storable = true;
        params.memory_type = PL_BUF_MEM_DEVICE;
        uploadParams.size = bytes;
        uploadParams.host_writable = true;
        uploadParams.memory_type = PL_BUF_MEM_HOST;
        return pl_buf_recreate(gpu, &clipping, &params) &&
            pl_buf_recreate(gpu, &clippingUpload, &uploadParams);
    }
    bool reserveGain(size_t bytes) {
        if (!gain || gain->params.size < bytes) gainCache.invalidate();
        if (bytes > gpu->limits.max_ssbo_size) return false;
        pl_buf_params params = {}, uploadParams = {};
        params.size = bytes;
        params.storable = true;
        params.memory_type = PL_BUF_MEM_DEVICE;
        uploadParams.size = bytes;
        uploadParams.host_writable = true;
        uploadParams.memory_type = PL_BUF_MEM_HOST;
        return ((gain && gain->params.size >= bytes) ||
                pl_buf_recreate(gpu, &gain, &params)) &&
            pl_buf_recreate(gpu, &gainUpload, &uploadParams);
    }
};

struct DeviceCache {
    std::mutex mutex;
    std::shared_ptr<Device> device;
    std::unordered_map<pl_gpu, std::shared_ptr<Device>> textureDevices;
    std::unordered_map<pl_gpu, std::shared_ptr<Device>> bufferDevices;
    std::unordered_map<pl_gpu, std::shared_ptr<Device>> bayerTextureDevices;
    bool unavailable = false;
};

DeviceCache& deviceCache() {
    // libplacebo's Android shaderc/glslang teardown is not safe when multiple
    // Vulkan devices are destroyed from C++ static destructors. Keep this
    // process-wide cache alive until the OS tears down the process instead.
    static auto* cache = new DeviceCache;
    return *cache;
}

std::mutex& deviceMutex = deviceCache().mutex;
std::shared_ptr<Device>& device = deviceCache().device;
std::unordered_map<pl_gpu, std::shared_ptr<Device>>& textureDevices =
    deviceCache().textureDevices;
std::unordered_map<pl_gpu, std::shared_ptr<Device>>& bufferDevices =
    deviceCache().bufferDevices;
std::unordered_map<pl_gpu, std::shared_ptr<Device>>& bayerTextureDevices =
    deviceCache().bayerTextureDevices;
bool& unavailable = deviceCache().unavailable;
std::once_flag warmOnce;
std::future<void> warmup;
std::atomic_bool warming{false};

uint64_t samplePackedFrame(pl_gpu gpu, pl_buf source,
                           uint32_t width, uint32_t height) {
    pl_buf_params params{};
    params.size = 64u * sizeof(uint32_t);
    params.host_readable = true;
    params.memory_type = PL_BUF_MEM_HOST;
    pl_buf samples = pl_buf_create(gpu, &params);
    if (!samples) return 0;
    for (uint32_t row = 0; row < 8; ++row)
        for (uint32_t col = 0; col < 8; ++col) {
            const size_t x = (size_t(col) * 2u + 1u) * width / 16u;
            const size_t y = (size_t(row) * 2u + 1u) * height / 16u;
            pl_buf_copy(gpu, samples, (row * 8u + col) * 4u,
                        source, (y * width + x) * 4u, 4u);
        }
    std::array<uint8_t, 64u * 4u> bytes{};
    const bool read = pl_buf_read(gpu, samples, 0, bytes.data(), bytes.size());
    pl_buf_destroy(gpu, &samples);
    if (!read) return 0;
    uint64_t hash = 1469598103934665603ull;
    for (uint8_t byte : bytes) { hash ^= byte; hash *= 1099511628211ull; }
    return hash;
}
} // namespace

void warmVulkanGalleryPreview(const std::string& cacheDirectory) {
    std::call_once(warmOnce, [&] {
        galleryShaderCache().configure(cacheDirectory);
        warming = true;
        warmup = std::async(std::launch::async, [] {
            std::lock_guard lock(deviceMutex);
            auto candidate = std::make_shared<Device>();
            if (candidate->init()) {
                device = std::move(candidate);
                spdlog::info("Gallery Vulkan preview initialized");
            } else {
                unavailable = true;
                spdlog::warn("Gallery Vulkan preview initialization failed; CPU preview processing will be used");
            }
            warming = false;
        });
    });
}

bool vulkanGalleryPreviewReady() {
    if (warming.load()) return false;
    std::lock_guard lock(deviceMutex);
    return device != nullptr;
}

bool waitVulkanGalleryPreviewReady() {
    warmVulkanGalleryPreview();
    if (warmup.valid()) warmup.wait();
    return vulkanGalleryPreviewReady();
}

void releaseVulkanGalleryTextureDevice(const void* gpu) {
    std::lock_guard lock(deviceMutex);
    textureDevices.erase(static_cast<pl_gpu>(gpu));
    bufferDevices.erase(static_cast<pl_gpu>(gpu));
    bayerTextureDevices.erase(static_cast<pl_gpu>(gpu));
}

bool demosaicGalleryBayerVulkan(
    const uint16_t* cfa, std::vector<uint8_t>& rgb,
    uint32_t width, uint32_t height,
    const std::array<uint8_t, 4>& phase,
    const std::array<double, 3>& black,
    const std::array<double, 3>& white) {
    if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_FORCE_CPU")) return false;
    const auto started = rawpreamp::diagnosticNow();
    if (!cfa || width < 2 || height < 2 ||
        width > std::numeric_limits<uint32_t>::max() / height / 3u ||
        (phase != std::array<uint8_t, 4>{0, 1, 1, 2} &&
         phase != std::array<uint8_t, 4>{2, 1, 1, 0} &&
         phase != std::array<uint8_t, 4>{1, 0, 2, 1} &&
         phase != std::array<uint8_t, 4>{1, 2, 0, 1})) return false;
    for (size_t channel = 0; channel < 3; ++channel)
        if (!(white[channel] > black[channel])) return false;
    const size_t sourceBytes = ((static_cast<size_t>(width) * height + 1) / 2) * 4;
    const size_t outputBytes = ((static_cast<size_t>(width) * height * 3 + 1) / 2) * 4;
    std::lock_guard lock(deviceMutex);
    if (unavailable) return false;
    if (!device) {
        auto candidate = std::make_shared<Device>();
        if (!candidate->init()) { unavailable = true; return false; }
        device = std::move(candidate);
    }
    if (!device->ensureDemosaicPass() ||
        !device->reserve(sourceBytes, outputBytes) ||
        !device->reserveDownload(outputBytes)) return false;
    const size_t completeSourceBytes = static_cast<size_t>(width) * height * 2;
    if (completeSourceBytes == sourceBytes)
        pl_buf_write(device->gpu, device->upload, 0, cfa, sourceBytes);
    else {
        pl_buf_write(device->gpu, device->upload, 0, cfa,
                     completeSourceBytes - 2);
        const uint32_t tail = cfa[static_cast<size_t>(width) * height - 1];
        pl_buf_write(device->gpu, device->upload,
                     completeSourceBytes - 2, &tail, sizeof(tail));
    }
    if (!device->directHostInput)
        pl_buf_copy(device->gpu, device->source, 0,
                    device->upload, 0, sourceBytes);
    Constants constants = {};
    constants.dimensions[0] = width;
    constants.dimensions[1] = height;
    constants.dimensions[2] = phase[0] | (phase[1] << 2u) |
        (phase[2] << 4u) | (phase[3] << 6u);
    constants.dimensions[3] = static_cast<uint32_t>(outputBytes / 4);
    for (size_t channel = 0; channel < 3; ++channel) {
        constants.black[channel] = static_cast<float>(black[channel]);
        constants.inverseRange[channel] = static_cast<float>(
            1.0 / (white[channel] - black[channel]));
    }
    pl_desc_binding bindings[2] = {};
    bindings[0].object = device->directHostInput
        ? device->upload : device->source;
    bindings[1].object = device->output;
    pl_pass_run_params run = {};
    run.pass = device->pass;
    run.desc_bindings = bindings;
    run.push_constants = &constants;
    run.compute_groups[0] = static_cast<int>((constants.dimensions[3] + 255u) / 256u);
    run.compute_groups[1] = 1;
    run.compute_groups[2] = 1;
    if (run.compute_groups[0] > device->gpu->limits.max_dispatch[0]) return false;
    pl_pass_run(device->gpu, &run);
    pl_buf_copy(device->gpu, device->download, 0,
                device->output, 0, outputBytes);
    std::vector<uint8_t> result(outputBytes);
    if (!pl_buf_read(device->gpu, device->download, 0,
                     result.data(), outputBytes) ||
        pl_gpu_is_failed(device->gpu)) {
        device.reset();
        unavailable = true;
        return false;
    }
    result.resize(static_cast<size_t>(width) * height * 6);
    rgb = std::move(result);
    if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_PROFILE"))
        std::fprintf(stderr,
            "GALLERY_PERF event=vulkan_gallery_demosaic width=%u height=%u elapsed_ms=%.3f\n",
            width, height, std::chrono::duration<double, std::milli>(
                rawpreamp::diagnosticNow() - started).count());
    return true;
}

bool colorGalleryRgbVulkan(
    std::vector<uint8_t>& rgb,
    const std::array<float, 3>& gains,
    const std::array<float, 9>& cameraToDisplay,
    const std::array<float, 9>& cameraToXyz,
    float exposure, const DNGProfileTable* hueSat,
    const DNGProfileTable* look,
    bool useMatrix, bool gainMapOnlyDebug,
    const VulkanGalleryBayerSource* bayer,
    const VulkanGalleryOutput* output,
    const VulkanGalleryTextureSource* texture,
    bool* warmingFallback) {
    if (warmingFallback) *warmingFallback = false;
    if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_FORCE_CPU")) return false;
    // Device warmup runs on another thread. Keep host previews responsive
    // through the CPU fallback instead of blocking on device creation.
    // Resident sources need their final shader variant before playback starts.
    // Falling back during warmup would compile it on a later playback frame.
    const bool residentSource = texture ||
        (bayer && bayer->gpu && (bayer->buffer || bayer->texture));
    if (residentSource && rawpreamp::debugEnvironmentIsSet("RAWPREAMP_TEST_FAIL_GALLERY_COLOR"))
        return false;
    if (warming.load() && !residentSource) {
        if (warmingFallback) *warmingFallback = true;
        return false;
    }
    if (texture) {
        if (bayer || !texture->gpu || (!texture->texture && !texture->buffer) ||
            texture->width == 0 || texture->height == 0 ||
            texture->width > std::numeric_limits<uint32_t>::max() /
                texture->height / 3u) return false;
        for (size_t channel = 0; channel < 3; ++channel)
            if (!(texture->white[channel] > texture->black[channel])) return false;
        if (texture->buffer && texture->gain) return false;
        if (texture->gain) {
            const auto& gain = *texture->gain;
            if (!gain.width || !gain.height || !gain.spacingH || !gain.spacingV ||
                gain.values.size() != static_cast<size_t>(gain.width) * gain.height * 3u ||
                gain.encodedWhite > 65535u)
                return false;
            if (gain.secondaryWidth &&
                (!gain.secondaryHeight || !gain.secondarySpacingH ||
                 !gain.secondarySpacingV ||
                 gain.secondaryValues.size() != static_cast<size_t>(
                     gain.secondaryWidth) * gain.secondaryHeight))
                return false;
        }
    } else if (bayer) {
        if ((!bayer->samples && !(bayer->gpu && (bayer->buffer || bayer->texture))) || bayer->width < 2 || bayer->height < 2 ||
            bayer->width > std::numeric_limits<uint32_t>::max() /
                bayer->height / 3u ||
            (bayer->phase != std::array<uint8_t, 4>{0, 1, 1, 2} &&
             bayer->phase != std::array<uint8_t, 4>{2, 1, 1, 0} &&
             bayer->phase != std::array<uint8_t, 4>{1, 0, 2, 1} &&
             bayer->phase != std::array<uint8_t, 4>{1, 2, 0, 1})) return false;
        for (size_t channel = 0; channel < 3; ++channel)
            if (!(bayer->white[channel] > bayer->black[channel])) return false;
    } else if (rgb.empty() || rgb.size() % 6u ||
               rgb.size() / 6u > std::numeric_limits<uint32_t>::max() / 3u)
        return false;
    const uint32_t pixels = texture ? texture->width * texture->height
        : bayer ? bayer->width * bayer->height
        : static_cast<uint32_t>(rgb.size() / 6u);
    const uint32_t sourceWidth = output ? output->width
        : texture ? texture->width : bayer ? bayer->width : pixels;
    const uint32_t sourceHeight = output ? output->height
        : texture ? texture->height : bayer ? bayer->height : 1u;
    if (!sourceWidth || !sourceHeight ||
        static_cast<size_t>(sourceWidth) * sourceHeight != pixels ||
        (bayer && (sourceWidth != bayer->width || sourceHeight != bayer->height)) ||
        (texture && (sourceWidth != texture->width || sourceHeight != texture->height)))
        return false;
    const int orientation = output ? output->orientation : 0;
    if (orientation != 0 && orientation != 90 && orientation != 180 &&
        orientation != 270) return false;
    const bool rgb24 = output && output->rgb24;
    const bool rgb10a2 = output && output->rgb10a2;
    if (rgb10a2 && !rgb24) return false;
    const auto* clip = output ? output->clipping : nullptr;
    const bool textureClipping = clip && clip->empty() && texture && texture->texture;
    if (clip && clip->size() != pixels && !textureClipping) return false;
    const void* externalGpu = texture ? texture->gpu
        : bayer && (bayer->buffer || bayer->texture) ? bayer->gpu : nullptr;
    const bool bayerTexture = bayer && bayer->texture;
    const bool prepareTextureBayer = bayerTexture && bayer->gain;
    const void* externalBuffer = texture ? texture->buffer
        : bayer ? bayer->buffer : nullptr;
    const size_t inputBytes = externalGpu ? 0u
        : bayer ? static_cast<size_t>(pixels) * 2u : rgb.size();
    std::lock_guard lock(deviceMutex);
    if (unavailable && !externalGpu) return false;
    auto& selectedDevice = bayerTexture && !prepareTextureBayer
        ? bayerTextureDevices[static_cast<pl_gpu>(externalGpu)]
        : externalBuffer || prepareTextureBayer
        ? bufferDevices[static_cast<pl_gpu>(externalGpu)]
        : texture ? textureDevices[static_cast<pl_gpu>(externalGpu)] : device;
    if (texture && rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_PROFILE")) {
        static std::atomic_bool loggedExternalCaps{false};
        if (!loggedExternalCaps.exchange(true))
            std::fprintf(stderr, "GALLERY_PERF event=gallery_external_caps export_buf=%d import_buf=%d export_tex=%d import_tex=%d\n",
                static_cast<int>(static_cast<pl_gpu>(texture->gpu)->export_caps.buf),
                static_cast<int>(static_cast<pl_gpu>(texture->gpu)->import_caps.buf),
                static_cast<int>(static_cast<pl_gpu>(texture->gpu)->export_caps.tex),
                static_cast<int>(static_cast<pl_gpu>(texture->gpu)->import_caps.tex));
    }
    if (!selectedDevice) {
        auto candidate = std::make_shared<Device>();
        if (!(externalGpu ? candidate->initExternal(static_cast<pl_gpu>(externalGpu), !externalBuffer && !prepareTextureBayer, bayerTexture && !prepareTextureBayer)
                      : candidate->init())) {
            if (!texture) unavailable = true;
            return false;
        }
        selectedDevice = std::move(candidate);
    }
    Device* selected = selectedDevice.get();
    const bool alignedRgb24 = selected->unifiedMemory && rgb24 && !rgb10a2;
    const size_t outputBytes = static_cast<size_t>(pixels) *
        (rgb10a2 || alignedRgb24 ? 4u : rgb24 ? 3u : 6u);
    const size_t paddedInput = texture ? 0u : (inputBytes + 3u) & ~size_t{3};
    const size_t paddedOutput = (outputBytes + 3u) & ~size_t{3};
    if (output && output->requireZeroCopy && !externalGpu) return false;
    // Shared-image/storage-buffer output does not need a second full-frame
    // scratch buffer. Allocate it below only for paths that actually use it.
    if (!selected->reserve(paddedInput, paddedOutput, false)) return false;
    const bool hasClipMask = clip && !clip->empty();
    const size_t clipBytes = hasClipMask
        ? (clip->size() + 3u) & ~size_t{3}
        : std::max(size_t(4), selected->previousClippingBytes);
    if (!selected->reserveClipping(clipBytes)) return false;
    // Clear the retired mask once when disabling indication. Deferred GPU
    // thumbnail work must not retain the previous frame's clipping colors.
    if (hasClipMask || selected->previousClippingBytes) {
        std::vector<uint8_t> padded(clipBytes);
        if (hasClipMask) std::memcpy(padded.data(), clip->data(), clip->size());
        pl_buf_write(selected->gpu, selected->clippingUpload, 0,
                     padded.data(), clipBytes);
        pl_buf_copy(selected->gpu, selected->clipping, 0,
                    selected->clippingUpload, 0, clipBytes);
    }
    selected->previousClippingBytes = hasClipMask ? clipBytes : 0;
    const size_t hueCount = hueSat ? hueSat->values.size() : 0u;
    const size_t lookCount = look ? look->values.size() : 0u;
    if (hueCount > std::numeric_limits<uint32_t>::max() ||
        lookCount > std::numeric_limits<uint32_t>::max() - hueCount) return false;
    auto validTable = [](const DNGProfileTable* table) {
        return !table || table->values.empty() ||
            (table->hueDivisions && table->saturationDivisions &&
             table->valueDivisions && table->values.size() ==
                 static_cast<size_t>(table->hueDivisions) *
                 table->saturationDivisions * table->valueDivisions * 3u);
    };
    if (!validTable(hueSat) || !validTable(look)) return false;
    const char* profileTextureOverride = rawpreamp::debugEnvironment("RAWPREAMP_DCP_TEXTURES");
    const bool profileTextures = (!profileTextureOverride || std::strcmp(profileTextureOverride, "0") != 0) &&
        !selected->profileTextureRejected && (hueCount || lookCount) && selected->profileTextures[0].update(selected->gpu, hueSat) &&
        selected->profileTextures[1].update(selected->gpu, look);
    const size_t profilePayloadBytes = 64u + (hueCount + lookCount) * sizeof(float);
    const size_t profileBytes = profilePayloadBytes + 65536u * sizeof(float);
    if (!selected->reserveProfiles(profileBytes)) return false;
    std::vector<uint8_t> profileData(profilePayloadBytes);
    auto writeTable = [&](const DNGProfileTable* table, size_t index,
                          uint32_t offset) {
        if (!table || table->values.empty()) return;
        const uint32_t shape[4] = {table->hueDivisions,
            table->saturationDivisions, table->valueDivisions, table->encoding};
        const uint32_t location[4] = {offset,
            static_cast<uint32_t>(table->values.size()), 0u, 0u};
        std::memcpy(profileData.data() + index * 16u, shape, sizeof(shape));
        std::memcpy(profileData.data() + 32u + index * 16u,
                    location, sizeof(location));
        std::memcpy(profileData.data() + 64u +
                        static_cast<size_t>(offset) * sizeof(float),
                    table->values.data(), table->values.size() * sizeof(float));
    };
    writeTable(hueSat, 0u, 0u);
    writeTable(look, 1u, static_cast<uint32_t>(hueCount));
    const uint32_t transferOffset = static_cast<uint32_t>(hueCount + lookCount);
    std::memcpy(profileData.data() + 44u, &transferOffset, sizeof(transferOffset));
    if (selected->profileCache.upload(selected->gpu, selected->profiles,
            selected->profilesUpload, profileData.data(), profilePayloadBytes))
        pl_buf_copy(selected->gpu, selected->profiles, profilePayloadBytes,
                    selected->srgbLookup, 0, 65536u * sizeof(float));
    std::vector<uint8_t> spatialGainKey;
    if (bayer && bayer->gain) {
        const auto* gain = bayer->gain;
        if (gain->maps.size() > 32) return false;
        size_t values = 0;
        for (const auto& map : gain->maps) {
            if (map.values.size() > UINT32_MAX - values) return false;
            values += map.values.size();
        }
        const size_t lookup = gain->encodedWhite
            ? gain->encodedWhite + 1u : 0u;
        const size_t encodeLookup = lookup ? 65536u : 0u;
        const size_t sourceLookup = gain->sourceLinearization.size();
        if (sourceLookup > 65536u) return false;
        constexpr size_t headerBytes = bayerGainHeaderBytes;
        if (values + lookup + sourceLookup + encodeLookup > (std::numeric_limits<size_t>::max() -
                               headerBytes) / sizeof(float)) return false;
        const size_t gainBytes = headerBytes + (values + lookup + sourceLookup + encodeLookup) * sizeof(float);
        if (!selected->reserveGain(gainBytes)) return false;
        std::vector<uint8_t> data(gainBytes);
        const uint32_t packedPhase = gain->phase[0] |
            (gain->phase[1] << 2u) | (gain->phase[2] << 4u) |
            (gain->phase[3] << 6u);
        const uint32_t header[4] = {
            static_cast<uint32_t>(gain->maps.size()), gain->encodedWhite,
            gain->ditherSeed,
            (gain->highQualityDither ? 1u : 0u) |
            (gain->debugGain ? 2u : 0u) | (sourceLookup ? 4u : 0u) | (packedPhase << 8u)};
        const uint32_t lookupOffset[4] = {
            static_cast<uint32_t>(values), static_cast<uint32_t>(values + lookup),
            static_cast<uint32_t>(sourceLookup),
            static_cast<uint32_t>(values + lookup + sourceLookup)};
        std::memcpy(data.data(), header, sizeof(header));
        std::memcpy(data.data() + 16u, gain->black.data(), 16u);
        std::memcpy(data.data() + 32u, gain->scale.data(), 16u);
        std::memcpy(data.data() + 48u, lookupOffset, sizeof(lookupOffset));
        const uint32_t sourceGeometry[4] = {
            gain->sourceWidth ? gain->sourceWidth : bayer->width,
            gain->sourceHeight ? gain->sourceHeight : bayer->height,
            std::max(1u, gain->sourceScale), gain->sourceParityOffset};
        std::memcpy(data.data() + 64u, sourceGeometry, sizeof(sourceGeometry));
        uint32_t offset = 0;
        for (size_t index = 0; index < gain->maps.size(); ++index) {
            const auto& map = gain->maps[index];
            const uint32_t bounds[4] = {
                map.left, map.top, map.right, map.bottom};
            const float axes[4] = {
                map.originH, map.originV, map.spacingH, map.spacingV};
            const uint32_t shape[4] = {
                map.width, map.height, map.channels, offset};
            const uint32_t pitch[4] = {
                map.rowPitch, map.colPitch, 0u, 0u};
            std::memcpy(data.data() + 80u + index * 16u, bounds, 16u);
            std::memcpy(data.data() + 592u + index * 16u, axes, 16u);
            std::memcpy(data.data() + 1104u + index * 16u, shape, 16u);
            std::memcpy(data.data() + 1616u + index * 16u, pitch, 16u);
            std::memcpy(data.data() + headerBytes + offset * sizeof(float),
                map.values.data(), map.values.size() * sizeof(float));
            offset += static_cast<uint32_t>(map.values.size());
        }
        if (lookup) {
            const auto& table = cachedLog60LinearizationTable(
                static_cast<uint16_t>(gain->encodedWhite));
            for (size_t index = 0; index < lookup; ++index) {
                const float code = static_cast<float>(table[index]);
                std::memcpy(data.data() + headerBytes +
                    (values + index) * sizeof(float), &code, sizeof(code));
            }
        }
        for (size_t index = 0; index < sourceLookup; ++index) {
            const float code = static_cast<float>(gain->sourceLinearization[index]);
            std::memcpy(data.data() + headerBytes +
                (values + lookup + index) * sizeof(float), &code, sizeof(code));
        }
        if (encodeLookup) {
            const auto& table = cachedLog60EncodeLookup(0.0, 65535.0,
                static_cast<uint16_t>(gain->encodedWhite));
            std::memcpy(data.data() + headerBytes +
                (values + lookup + sourceLookup) * sizeof(float),
                table.data(), table.size() * sizeof(float));
        }
        selected->gainCache.upload(selected->gpu, selected->gain,
            selected->gainUpload, data.data(), gainBytes);
        if (
            !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_GAIN_FIELD_DISABLE") &&
            size_t(pixels) * sizeof(float) <= std::min<size_t>(64u * 1024u * 1024u,
                selected->gpu->limits.max_ssbo_size) &&
            std::any_of(gain->maps.begin(), gain->maps.end(), [](const auto& map) {
                return map.width > 1 || map.height > 1;
            })) {
            // Only spatial inputs belong in this key. Levels, LOG and dither
            // still run every frame and must not invalidate the gain product.
            const uint32_t shape[] = {bayer->width, bayer->height, packedPhase,
                                     static_cast<uint32_t>(gain->maps.size())};
            spatialGainKey.resize(sizeof(shape) + sizeof(sourceGeometry) +
                                  headerBytes - 80u + values * sizeof(float));
            auto* out = spatialGainKey.data();
            std::memcpy(out, shape, sizeof(shape)); out += sizeof(shape);
            std::memcpy(out, sourceGeometry, sizeof(sourceGeometry)); out += sizeof(sourceGeometry);
            std::memcpy(out, data.data() + 80u, headerBytes - 80u + values * sizeof(float));
        }
    }
    if (texture && !externalBuffer) {
        const auto* gain = texture->gain;
        const size_t primaryValues = gain ? gain->values.size() : 0u;
        const size_t values = primaryValues +
            (gain ? gain->secondaryValues.size() : 0u);
        const size_t lookup = gain && gain->encodedWhite
            ? gain->encodedWhite + 1u : 0u;
        const size_t gainBytes = 112u + (values + lookup) * sizeof(uint32_t);
        if (!selected->reserveGain(gainBytes)) return false;
        std::vector<uint8_t> data(gainBytes);
        if (gain) {
            const uint32_t shape[4] = {gain->width, gain->height,
                gain->encodedWhite, gain->ditherSeed};
            const uint32_t bounds[4] = {gain->left, gain->top,
                gain->right, gain->bottom};
            const float axes[4] = {gain->originH, gain->originV,
                gain->spacingH, gain->spacingV};
            const uint32_t flags[4] = {gain->highQualityDither ? 1u : 0u,
                static_cast<uint32_t>(values), gain->secondaryWidth ? 1u : 0u,
                gain->debugGainMap ? 1u : 0u};
            const uint32_t secondaryShape[4] = {gain->secondaryWidth,
                gain->secondaryHeight, 1u, static_cast<uint32_t>(primaryValues)};
            const uint32_t secondaryBounds[4] = {gain->secondaryLeft,
                gain->secondaryTop, gain->secondaryRight, gain->secondaryBottom};
            const float secondaryAxes[4] = {gain->secondaryOriginH,
                gain->secondaryOriginV, gain->secondarySpacingH,
                gain->secondarySpacingV};
            std::memcpy(data.data(), shape, sizeof(shape));
            std::memcpy(data.data() + 16, bounds, sizeof(bounds));
            std::memcpy(data.data() + 32, axes, sizeof(axes));
            std::memcpy(data.data() + 48, flags, sizeof(flags));
            std::memcpy(data.data() + 64, secondaryShape, sizeof(secondaryShape));
            std::memcpy(data.data() + 80, secondaryBounds, sizeof(secondaryBounds));
            std::memcpy(data.data() + 96, secondaryAxes, sizeof(secondaryAxes));
            std::memcpy(data.data() + 112, gain->values.data(),
                primaryValues * sizeof(float));
            if (gain->secondaryWidth)
                std::memcpy(data.data() + 112 + primaryValues * sizeof(float),
                    gain->secondaryValues.data(),
                    gain->secondaryValues.size() * sizeof(float));
            if (lookup) {
                const auto& linearization = cachedLog60LinearizationTable(gain->encodedWhite);
                for (size_t index = 0; index < lookup; ++index) {
                    const float code = static_cast<float>(linearization[index]);
                    std::memcpy(data.data() + 112 + (values + index) * sizeof(float),
                        &code, sizeof(code));
                }
            }
        }
        selected->gainCache.upload(selected->gpu, selected->gain,
            selected->gainUpload, data.data(), gainBytes);
        if (gain && (gain->width > 1 || gain->height > 1 ||
                     gain->secondaryWidth > 1 || gain->secondaryHeight > 1)) {
            spatialGainKey.assign(data.begin(), data.begin() + 112u + values * sizeof(float));
            std::fill(spatialGainKey.begin() + 8, spatialGainKey.begin() + 16, 0);
            std::fill(spatialGainKey.begin() + 48, spatialGainKey.begin() + 64, 0);
            const uint32_t geometry[] = {sourceWidth, sourceHeight};
            const auto bytes = reinterpret_cast<const uint8_t*>(geometry);
            spatialGainKey.insert(spatialGainKey.end(), bytes, bytes + sizeof(geometry));
        }
    }
    if (spatialGainKey.empty()) selected->spatialGains.reset();
    if (!externalGpu) {
        const auto* input = bayer
            ? reinterpret_cast<const uint8_t*>(bayer->samples) : rgb.data();
        if (paddedInput == inputBytes)
            pl_buf_write(selected->gpu, selected->upload, 0,
                         input, inputBytes);
        else {
            pl_buf_write(selected->gpu, selected->upload, 0,
                         input, inputBytes - 2u);
            uint32_t tail = 0;
            std::memcpy(&tail, input + inputBytes - 2u, 2u);
            pl_buf_write(selected->gpu, selected->upload,
                         inputBytes - 2u, &tail, sizeof(tail));
        }
        if (!selected->directHostInput)
            pl_buf_copy(selected->gpu, selected->source, 0,
                        selected->upload, 0, paddedInput);
    }
    ColorConstants constants = {};
    constants.dimensions[0] = pixels;
    constants.dimensions[1] = static_cast<uint32_t>(paddedOutput / 4u);
    constants.dimensions[2] = hueCount || lookCount ? 2u : useMatrix ? 1u : 0u;
    constants.dimensions[3] = gainMapOnlyDebug ? 1u : 0u;
    for (size_t channel = 0; channel < 3; ++channel) {
        constants.gains[channel] = gains[channel];
        for (size_t column = 0; column < 3; ++column)
            constants.matrixRows[channel * 4u + column] =
                cameraToDisplay[channel * 3u + column];
        for (size_t column = 0; column < 3; ++column)
            constants.xyzRows[channel * 4u + column] =
                cameraToXyz[channel * 3u + column];
    }
    constants.gains[3] = exposure;
    constants.geometry[0] = sourceWidth;
    constants.geometry[1] = sourceHeight;
    constants.geometry[2] = static_cast<uint32_t>(orientation);
    constants.geometry[3] = (rgb24 ? 1u : 0u) | (clip ? 2u : 0u) |
        (rgb10a2 ? 4u : 0u) | (alignedRgb24 ? 8u : 0u) |
        (textureClipping ? 16u : 0u);
    if (bayer) {
        constants.bayer[0] = bayer->width;
        constants.bayer[1] = bayer->height;
        constants.bayer[2] = bayer->phase[0] | (bayer->phase[1] << 2u) |
            (bayer->phase[2] << 4u) | (bayer->phase[3] << 6u);
        constants.bayer[3] = bayer->gain ? 2u : 1u;
        for (size_t channel = 0; channel < 3; ++channel) {
            constants.bayerBlack[channel] =
                static_cast<float>(bayer->black[channel]);
            constants.bayerInverseRange[channel] = static_cast<float>(
                1.0 / (bayer->white[channel] - bayer->black[channel]));
        }
    }
    if (texture && texture->buffer) constants.bayer[3] = 3u;
    if (texture) for (size_t channel = 0; channel < 3; ++channel) {
        constants.textureBlack[channel] =
            static_cast<float>(texture->black[channel]);
        constants.textureInverseRange[channel] = static_cast<float>(
            1.0 / (texture->white[channel] - texture->black[channel]));
    }
    pl_desc_binding bindings[8] = {};
    pl_buf exported = nullptr;
    bool sharedTexture = false;
    bool sharedImage10 = rgb10a2;
    pl_tex frameOutputTexture = nullptr;
    bool exportNeedsCopy = false;
    // Sharing depends on a common VkDevice, regardless of memory topology.
    const uint32_t outputWidth = orientation == 90 || orientation == 270
        ? sourceHeight : sourceWidth;
    const uint32_t outputHeight = orientation == 90 || orientation == 270
        ? sourceWidth : sourceHeight;
    if (output && output->gpuBuffer && rgb24 && !selected->sharedTextureRejected &&
        pl_vulkan_get(selected->gpu) && sharedVulkanDevice() &&
        pl_vulkan_get(selected->gpu)->device == sharedVulkanDevice()->device) {
        if (selected->ensureColorImagePass(sharedImage10, profileTextures))
            frameOutputTexture = selected->createOutputTexture(outputWidth, outputHeight, sharedImage10);
        // A driver may sample packed 10-bit images without supporting storage
        // writes. Keep the shared transport and use RGBA8 in that case.
        if (!frameOutputTexture && sharedImage10) {
            sharedImage10 = false;
            if (selected->ensureColorImagePass(false, profileTextures))
                frameOutputTexture = selected->createOutputTexture(outputWidth, outputHeight, false);
        }
        sharedTexture = frameOutputTexture != nullptr;
    }
    if (output && output->requireZeroCopy && !sharedTexture) return false;
    const auto galleryHandle = galleryExternalBufferHandle(selected->gpu);
    const bool exportGpu = !sharedTexture && !selected->externalBufferRejected &&
        output && output->gpuBuffer && rgb24 &&
        (selected->gpu->export_caps.buf & galleryHandle);
    if (rgb10a2 && !sharedTexture && !exportGpu) selected->externalBufferRejected = true;
    if (exportGpu) {
        pl_buf_params params = {};
        params.size = paddedOutput;
        // Use the exported allocation as the compute output when supported,
        // avoiding a second GPU buffer copy before the presenter imports it.
        params.storable = true;
        params.export_handle = galleryHandle;
        exported = pl_buf_create(selected->gpu, &params);
        if (!exported) {
            // Some drivers can export transfer buffers but not storage
            // buffers. Use the extra copy only on those drivers.
            params.storable = false;
            exported = pl_buf_create(selected->gpu, &params);
        }
        exportNeedsCopy = exported && !params.storable;
        if (!exported) selected->externalBufferRejected = true;
    }
    if (!sharedTexture && (!exported || exportNeedsCopy) &&
        !selected->reserveOutput(paddedOutput)) {
        if (exported) pl_buf_destroy(selected->gpu, &exported);
        return false;
    }
    if (!sharedTexture && !selected->ensureColorPass(bool(clip), profileTextures)) {
        if (exported) pl_buf_destroy(selected->gpu, &exported);
        return false;
    }
    bindings[0].object = externalBuffer ? externalBuffer
        : bayerTexture ? bayer->texture
        : texture ? texture->texture
        : static_cast<const void*>(selected->directHostInput
            ? selected->upload : selected->source);
    bindings[0].sample_mode = PL_TEX_SAMPLE_NEAREST;
    bindings[1].object = sharedTexture
        ? static_cast<const void*>(frameOutputTexture)
        : exported && !exportNeedsCopy
        ? exported : selected->output;
    bindings[2].object = selected->profiles;
    bindings[3].object = selected->clipping;
    bindings[4].object = (texture && texture->gain) || (bayer && bayer->gain)
        ? selected->gain : selected->emptyGain;
    if (profileTextures) for (size_t i=0;i<2;++i) {
        bindings[5+i].object = selected->profileTextures[i].image;
        bindings[5+i].sample_mode = PL_TEX_SAMPLE_NEAREST;
    }
    pl_pass_run_params run = {};
    run.pass = sharedTexture
        ? selected->ensureColorImagePass(sharedImage10, profileTextures)
        : selected->ensureColorPass(bool(clip), profileTextures);
    run.desc_bindings = bindings;
    run.push_constants = &constants;
    const uint32_t pixelsPerInvocation = sharedTexture ? 1u : rgb24 ? 4u : 2u;
    const uint32_t groups =
        (((pixels + pixelsPerInvocation - 1u) / pixelsPerInvocation) + 255u) / 256u;
    run.compute_groups[0] = std::min(groups, selected->gpu->limits.max_dispatch[0]);
    run.compute_groups[1] = (groups + run.compute_groups[0] - 1u) / run.compute_groups[0];
    run.compute_groups[2] = 1;
    if (sharedTexture) {
        run.compute_groups[0] = (sourceWidth + 15u) / 16u;
        run.compute_groups[1] = (sourceHeight + 15u) / 16u;
    } else if ((orientation == 90 || orientation == 270) && outputWidth % pixelsPerInvocation == 0u) {
        run.compute_groups[0] = (outputWidth / pixelsPerInvocation + 15u) / 16u;
        run.compute_groups[1] = (outputHeight + 15u) / 16u;
    }
    if (run.compute_groups[1] > selected->gpu->limits.max_dispatch[1]) {
        pl_tex_destroy(selected->gpu, &frameOutputTexture);
        if (exported) pl_buf_destroy(selected->gpu, &exported);
        return false;
    }
    const bool profileStages = rawpreamp::debugEnvironment("RAWPREAMP_RAW_GPU_TIMERS");
    if (profileStages) {
        if (!selected->prepareTimer) selected->prepareTimer = pl_timer_create(selected->gpu);
        if (!selected->colorTimer) selected->colorTimer = pl_timer_create(selected->gpu);
        uint64_t ns;
        while ((ns = pl_timer_query(selected->gpu, selected->prepareTimer)))
            spdlog::info("GALLERY_PERF event=gallery_gpu_stage stage=bayer_prepare elapsed_ms={:.3f}", ns / 1e6);
        while ((ns = pl_timer_query(selected->gpu, selected->colorTimer)))
            spdlog::info("GALLERY_PERF event=gallery_gpu_stage stage=color elapsed_ms={:.3f}", ns / 1e6);
        run.timer = selected->colorTimer;
    }
    // Bake once per CFA sample, rather than repeating gain-map interpolation,
    // dithering and LOG lookup at every demosaic neighbour. The intermediate
    // stays on this device and is consumed directly by the color pass.
    if (bayer && bayer->gain) {
        auto& uncachedPrepare = prepareTextureBayer ? selected->bayerTexturePreparePass : selected->bayerPreparePass;
        auto& cachedPrepare = prepareTextureBayer ? selected->bayerTexturePrepareCachedPass : selected->bayerPrepareCachedPass;
        if (!uncachedPrepare)
            uncachedPrepare = selected->createColorPass(false, false, false, true, false, false, false, prepareTextureBayer);
        pl_buf_params preparedParams{};
        preparedParams.size = ((size_t(pixels) + 1u) / 2u) * sizeof(uint32_t);
        preparedParams.storable = true;
        preparedParams.memory_type = PL_BUF_MEM_DEVICE;
        if (!uncachedPrepare ||
            !pl_buf_recreate(selected->gpu, &selected->preparedBayer, &preparedParams)) {
            pl_tex_destroy(selected->gpu, &frameOutputTexture);
            if (exported) pl_buf_destroy(selected->gpu, &exported);
            return false;
        }
        std::array<pl_desc_binding, 8> prepareBindings;
        std::copy(std::begin(bindings), std::end(bindings), prepareBindings.begin());
        prepareBindings[1].object = selected->preparedBayer;
        bool capturedThisFrame = false;
        const bool captureGains = !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_GAIN_FIELD_CAPTURE_DISABLE");
        auto gainField = selected->spatialGains.acquire(selected->gpu, spatialGainKey,
            size_t(pixels) * sizeof(float), 1u, [&](pl_buf target) {
                if (captureGains) {
                    auto& capturePass = prepareTextureBayer ? selected->bayerTexturePrepareCapturePass
                                                            : selected->bayerPrepareCapturePass;
                    if (!capturePass)
                        capturePass = selected->createColorPass(false, false, false, true, false,
                            false, false, prepareTextureBayer, true);
                    if (!cachedPrepare)
                        cachedPrepare = selected->createColorPass(false, false, false, true, false,
                            false, true, prepareTextureBayer);
                    if (!capturePass || !cachedPrepare) return false;
                    auto captureBindings = prepareBindings;
                    captureBindings[5].object = target;
                    auto captureRun = run;
                    captureRun.pass = capturePass;
                    captureRun.timer = profileStages ? selected->prepareTimer : nullptr;
                    captureRun.desc_bindings = captureBindings.data();
                    const uint32_t groups = ((pixels + 1u) / 2u + 255u) / 256u;
                    captureRun.compute_groups[0] = std::min(groups, selected->gpu->limits.max_dispatch[0]);
                    captureRun.compute_groups[1] = (groups + captureRun.compute_groups[0] - 1u) / captureRun.compute_groups[0];
                    captureRun.compute_groups[2] = 1;
                    if (captureRun.compute_groups[1] > selected->gpu->limits.max_dispatch[1]) return false;
                    pl_pass_run(selected->gpu, &captureRun);
                    capturedThisFrame = true;
                    return true;
                }
                if (!selected->gainFieldPass)
                    selected->gainFieldPass = selected->createColorPass(false, false, false, false, false, true);
                if (!cachedPrepare)
                    cachedPrepare = selected->createColorPass(false, false, false, true, false, false, true, prepareTextureBayer);
                if (!selected->gainFieldPass || !cachedPrepare) return false;
                auto fieldBindings = prepareBindings;
                fieldBindings[0].object = selected->preparedBayer;
                fieldBindings[1].object = target;
                auto fieldRun = run;
                fieldRun.pass = selected->gainFieldPass;
                fieldRun.timer = nullptr;
                fieldRun.desc_bindings = fieldBindings.data();
                const uint32_t fieldGroups = (pixels + 255u) / 256u;
                fieldRun.compute_groups[0] = std::min(fieldGroups, selected->gpu->limits.max_dispatch[0]);
                fieldRun.compute_groups[1] = (fieldGroups + fieldRun.compute_groups[0] - 1u) / fieldRun.compute_groups[0];
                fieldRun.compute_groups[2] = 1;
                if (fieldRun.compute_groups[1] > selected->gpu->limits.max_dispatch[1]) return false;
                pl_pass_run(selected->gpu, &fieldRun);
                return true;
            }, captureGains);
        // A buffer and a texture can share a field on this device. Ensure the
        // consuming variant even when this source type did not build the field.
        if (gainField && !cachedPrepare)
            cachedPrepare = selected->createColorPass(false, false, false, true, false,
                false, true, prepareTextureBayer);
        const bool cachedSpatialGain = gainField && cachedPrepare;
        if (cachedSpatialGain) prepareBindings[5].object = gainField;
        if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_PROFILE"))
            spdlog::info("GALLERY_PERF event=gallery_gain_field cached={} bytes={}",
                         cachedSpatialGain, cachedSpatialGain ? size_t(pixels) * sizeof(float) : 0u);
        auto prepareRun = run;
        prepareRun.pass = cachedSpatialGain ? cachedPrepare : uncachedPrepare;
        prepareRun.timer = profileStages ? selected->prepareTimer : nullptr;
        prepareRun.desc_bindings = prepareBindings.data();
        const uint32_t prepareGroups = ((pixels + 1u) / 2u + 255u) / 256u;
        prepareRun.compute_groups[0] = std::min(prepareGroups, selected->gpu->limits.max_dispatch[0]);
        prepareRun.compute_groups[1] = (prepareGroups + prepareRun.compute_groups[0] - 1u) /
            prepareRun.compute_groups[0];
        if (prepareRun.compute_groups[1] > selected->gpu->limits.max_dispatch[1]) {
            pl_tex_destroy(selected->gpu, &frameOutputTexture);
            if (exported) pl_buf_destroy(selected->gpu, &exported);
            return false;
        }
        if (!capturedThisFrame) pl_pass_run(selected->gpu, &prepareRun);
        if (cachedSpatialGain && rawpreamp::debugEnvironmentIsSet("RAWPREAMP_GAIN_FIELD_VERIFY")) {
            if (!selected->comparePreparedPass) {
                pl_desc descriptors[] = {
                    {"a", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                    {"b", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
                    {"result", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_READWRITE}};
                pl_pass_params params{};
                params.type = PL_PASS_COMPUTE;
                params.glsl_shader = comparePreparedShader;
                params.descriptors = descriptors; params.num_descriptors = 3;
                params.push_constants_size = sizeof(uint32_t);
                selected->comparePreparedPass = pl_pass_create(selected->gpu, &params);
            }
            pl_buf_params resultParams{};
            resultParams.size = 2u * sizeof(uint32_t); resultParams.storable = true;
            pl_buf_params stageParams{};
            stageParams.size = resultParams.size;
            stageParams.host_writable = stageParams.host_readable = true;
            if (!selected->comparePreparedPass ||
                !pl_buf_recreate(selected->gpu, &selected->comparePrepared, &preparedParams) ||
                !pl_buf_recreate(selected->gpu, &selected->compareResult, &resultParams) ||
                !pl_buf_recreate(selected->gpu, &selected->compareStage, &stageParams)) {
                pl_tex_destroy(selected->gpu, &frameOutputTexture);
                if (exported) pl_buf_destroy(selected->gpu, &exported);
                return false;
            }
            const uint32_t zero[2] = {0,0};
            pl_buf_write(selected->gpu, selected->compareStage, 0, zero, sizeof(zero));
            pl_buf_copy(selected->gpu, selected->compareResult, 0, selected->compareStage, 0, sizeof(zero));
            auto referenceBindings = prepareBindings;
            referenceBindings[1].object = selected->comparePrepared;
            auto referenceRun = prepareRun;
            referenceRun.pass = uncachedPrepare;
            referenceRun.timer = nullptr;
            referenceRun.desc_bindings = referenceBindings.data();
            pl_pass_run(selected->gpu, &referenceRun);
            pl_desc_binding compareBindings[3]{};
            compareBindings[0].object = selected->preparedBayer;
            compareBindings[1].object = selected->comparePrepared;
            compareBindings[2].object = selected->compareResult;
            auto compareRun = prepareRun;
            compareRun.pass = selected->comparePreparedPass;
            compareRun.timer = nullptr;
            compareRun.desc_bindings = compareBindings;
            uint32_t compareSamples = pixels;
            compareRun.push_constants = &compareSamples;
            pl_pass_run(selected->gpu, &compareRun);
            pl_buf_copy(selected->gpu, selected->compareStage, 0, selected->compareResult, 0, sizeof(zero));
            uint32_t result[2]{};
            const bool read = pl_buf_read(selected->gpu, selected->compareStage, 0, result, sizeof(result));
            spdlog::info("GALLERY_PERF event=gain_field_verify samples={} max_difference={} different={}",
                         pixels, result[0], result[1]);
            if (!read || result[0] != 0) {
                pl_tex_destroy(selected->gpu, &frameOutputTexture);
                if (exported) pl_buf_destroy(selected->gpu, &exported);
                return false;
            }
        }
        bindings[0].object = selected->preparedBayer;
        bindings[4].object = selected->emptyGain;
        constants.bayer[3] = 1u;
    }
    pl_buf rgbGainField = nullptr;
    if (texture && !externalBuffer && texture->gain) {
        auto field = selected->spatialGains.acquire(selected->gpu, spatialGainKey,
            size_t(pixels) * 3u * sizeof(float), 3u, [&](pl_buf target) {
                if (!selected->gainFieldPass)
                    selected->gainFieldPass = selected->createColorPass(false, false, false, false, false, true);
                if (!selected->gainFieldPass) return false;
                auto fieldBindings = std::array<pl_desc_binding,8>{};
                std::copy(std::begin(bindings), std::end(bindings), fieldBindings.begin());
                fieldBindings[1].object = target;
                auto fieldRun = run;
                fieldRun.pass = selected->gainFieldPass; fieldRun.timer = nullptr;
                fieldRun.desc_bindings = fieldBindings.data();
                const uint32_t groups = (pixels + 255u) / 256u;
                fieldRun.compute_groups[0] = std::min(groups, selected->gpu->limits.max_dispatch[0]);
                fieldRun.compute_groups[1] = (groups + fieldRun.compute_groups[0] - 1u) / fieldRun.compute_groups[0];
                fieldRun.compute_groups[2] = 1;
                if (fieldRun.compute_groups[1] > selected->gpu->limits.max_dispatch[1]) return false;
                pl_pass_run(selected->gpu, &fieldRun);
                return true;
            });
        if (field) {
            const auto cachedPass = sharedTexture
                ? selected->ensureColorImagePass(sharedImage10, profileTextures, true)
                : selected->ensureColorPass(bool(clip), profileTextures, true);
            if (cachedPass) {
                const size_t fieldBinding = profileTextures && !selected->profileTextureRejected ? 7u : 5u;
                bindings[fieldBinding].object = field;
                run.pass = cachedPass;
                rgbGainField = field;
            }
        }
    }
    Device::ThumbnailReadback* scheduledThumbnail=nullptr;
    size_t scheduledThumbnailBytes=0;
    int scheduledThumbnailFrame=-1;
    ColorConstants scheduledThumbnailConstants{};
    std::array<pl_desc_binding,8> scheduledThumbnailBindings{};
    pl_pass_run_params scheduledThumbnailRun{};
    if ((hueCount || lookCount) && rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_PROFILE"))
        spdlog::info("GALLERY_PERF event=gallery_dcp_sampling backend={} hue_points={} look_points={}",
            profileTextures && !selected->profileTextureRejected ? "texture" : "buffer",
            hueCount/3u, lookCount/3u);
    pl_pass_run(selected->gpu, &run);
    if (exported && exportNeedsCopy)
        pl_buf_copy(selected->gpu, exported, 0, selected->output, 0, paddedOutput);
    if (output && output->completedThumbnailSourceFrame)
        *output->completedThumbnailSourceFrame = -1;
    if (output && output->thumbnailPixels && output->thumbnailWidth &&
        output->thumbnailHeight) {
        auto& thumbnail = *output->thumbnailPixels;
        thumbnail.clear();
        const size_t count = static_cast<size_t>(output->thumbnailWidth) *
            output->thumbnailHeight;
        if (count <= pixels && count <= std::numeric_limits<uint32_t>::max() / 3u) {
            const size_t bytes = count * 3u;
            const size_t padded = (bytes + 3u) & ~size_t{3};
            const bool deferred = selected->unifiedMemory &&
                output->completedThumbnailSourceFrame;
            auto* slot = &selected->thumbnailReadback;
            if (deferred) {
                // Reclaim completed slots, discarding results from a previous
                // seek/clip before they can enter the current thumbnail cache.
                Device::ThumbnailReadback* ready = nullptr;
                for (auto& candidate : selected->thumbnailReadbacks) {
                    if (!candidate.pending || pl_buf_poll(selected->gpu, candidate.download, 0))
                        continue;
                    if (candidate.generation != output->thumbnailGeneration ||
                        candidate.width != output->thumbnailWidth ||
                        candidate.height != output->thumbnailHeight ||
                        candidate.clipping != bool(output->clipping)) {
                        candidate.pending = false;
                        continue;
                    }
                    if (!ready || candidate.sourceFrame < ready->sourceFrame)
                        ready = &candidate;
                }
                if (ready) {
                    thumbnail.resize(ready->byteCount);
                    if (pl_buf_read(selected->gpu, ready->download, 0,
                                    thumbnail.data(), ready->byteCount))
                        *output->completedThumbnailSourceFrame = ready->sourceFrame;
                    else thumbnail.clear();
                    ready->pending = false;
                }
                slot = nullptr;
                for (size_t offset = 0; offset < selected->thumbnailReadbacks.size(); ++offset) {
                    const size_t index = (selected->nextThumbnailReadback + offset) %
                        selected->thumbnailReadbacks.size();
                    if (!selected->thumbnailReadbacks[index].pending) {
                        slot = &selected->thumbnailReadbacks[index];
                        selected->nextThumbnailReadback = (index + 1) %
                            selected->thumbnailReadbacks.size();
                        break;
                    }
                }
            }
            const auto thumbnailPass = slot
                ? selected->ensureColorPass(bool(clip), profileTextures, rgbGainField != nullptr)
                : nullptr;
            if (thumbnailPass && selected->reserveThumbnail(*slot, padded)) {
                scheduledThumbnailConstants = constants;
                auto& c = scheduledThumbnailConstants;
                c.dimensions[0] = static_cast<uint32_t>(count);
                c.dimensions[1] = static_cast<uint32_t>(padded / 4u);
                c.geometry[3] = (c.geometry[3] | 1u) & ~(4u | 8u);
                c.geometry[2] = static_cast<uint32_t>(output->thumbnailOrientation);
                c.thumbnail[0] = output->thumbnailWidth;
                c.thumbnail[1] = output->thumbnailHeight;
                c.thumbnail[2] = 1u;
                std::copy(std::begin(bindings), std::end(bindings),
                          scheduledThumbnailBindings.begin());
                scheduledThumbnailBindings[1].object = slot->output;
                if (rgbGainField) {
                    // Thumbnail shader creation can reject DCP textures after
                    // the image shader succeeded. Bind for the selected pass.
                    const size_t fieldBinding = profileTextures && !selected->profileTextureRejected ? 7u : 5u;
                    scheduledThumbnailBindings[fieldBinding].object = rgbGainField;
                }
                scheduledThumbnailRun = run;
                scheduledThumbnailRun.timer = nullptr;
                scheduledThumbnailRun.pass = thumbnailPass;
                scheduledThumbnailRun.desc_bindings = scheduledThumbnailBindings.data();
                scheduledThumbnailRun.push_constants = &scheduledThumbnailConstants;
                scheduledThumbnailRun.compute_groups[0] = static_cast<int>(
                    (((count + 3u) / 4u) + 255u) / 256u);
                scheduledThumbnailRun.compute_groups[1] = 1;
                if (deferred) {
                    scheduledThumbnail = slot;
                    scheduledThumbnailBytes = bytes;
                    scheduledThumbnailFrame = output->thumbnailSourceFrame;
                } else {
                    thumbnail.resize(padded);
                    pl_pass_run(selected->gpu, &scheduledThumbnailRun);
                    pl_buf_copy(selected->gpu, slot->download, 0, slot->output, 0, padded);
                    if (pl_buf_read(selected->gpu, slot->download, 0, thumbnail.data(), padded)) {
                        thumbnail.resize(bytes);
                        if (output->completedThumbnailSourceFrame)
                            *output->completedThumbnailSourceFrame = output->thumbnailSourceFrame;
                    } else thumbnail.clear();
                }
            }
        }
    }
    auto enqueueThumbnailReadback=[&]{
        if(!scheduledThumbnail)return;
        pl_pass_run(selected->gpu,&scheduledThumbnailRun);
        pl_buf_copy(selected->gpu,scheduledThumbnail->download,0,
            scheduledThumbnail->output,0,
            (scheduledThumbnailBytes+3u)&~size_t{3});
        scheduledThumbnail->byteCount=scheduledThumbnailBytes;
        scheduledThumbnail->width=output->thumbnailWidth;
        scheduledThumbnail->height=output->thumbnailHeight;
        scheduledThumbnail->clipping=bool(output->clipping);
        scheduledThumbnail->sourceFrame=scheduledThumbnailFrame;
        scheduledThumbnail->generation=output->thumbnailGeneration;
        scheduledThumbnail->pending=true;
        scheduledThumbnail=nullptr;
    };
    if (sharedTexture) {
        const auto producer = pl_vulkan_get(selected->gpu);
        const auto getDeviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            producer->get_proc_addr(producer->instance, "vkGetDeviceProcAddr"));
        const auto createSemaphore = getDeviceProc ? reinterpret_cast<PFN_vkCreateSemaphore>(
            getDeviceProc(producer->device, "vkCreateSemaphore")) : nullptr;
        const auto destroySemaphore = getDeviceProc ? reinterpret_cast<PFN_vkDestroySemaphore>(
            getDeviceProc(producer->device, "vkDestroySemaphore")) : nullptr;
        auto waitSemaphores = getDeviceProc ? reinterpret_cast<PFN_vkWaitSemaphores>(
            getDeviceProc(producer->device, "vkWaitSemaphores")) : nullptr;
        if (!waitSemaphores && getDeviceProc)
            waitSemaphores = reinterpret_cast<PFN_vkWaitSemaphores>(
                getDeviceProc(producer->device, "vkWaitSemaphoresKHR"));
        if (!waitSemaphores || !createSemaphore || !destroySemaphore) {
            pl_tex_destroy(selected->gpu, &frameOutputTexture);
            return false;
        }
        VkSemaphoreTypeCreateInfo type{};
        type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        create.pNext = &type;
        VkSemaphore ready = VK_NULL_HANDLE;
        if (createSemaphore(producer->device, &create, nullptr, &ready) != VK_SUCCESS) {
            pl_tex_destroy(selected->gpu, &frameOutputTexture);
            return false;
        }
        pl_vulkan_hold_params hold{};
        hold.tex = frameOutputTexture;
        hold.layout = VK_IMAGE_LAYOUT_GENERAL;
        hold.qf = VK_QUEUE_FAMILY_IGNORED;
        hold.semaphore = {ready, 1};
        if (!pl_vulkan_hold_ex(selected->gpu, &hold)) {
            pl_tex_destroy(selected->gpu, &frameOutputTexture);
            destroySemaphore(producer->device, ready, nullptr);
            return false;
        }
        pl_gpu_flush(selected->gpu);
        enqueueThumbnailReadback();
        auto gpuFrame = std::make_shared<VulkanGalleryGpuBuffer>();
        gpuFrame->gpu = selected->gpu;
        gpuFrame->texture = frameOutputTexture;
        gpuFrame->rejectSharedImage = [weak = std::weak_ptr<Device>(selectedDevice)] {
            const auto owner = weak.lock();
            return owner && !owner->sharedTextureRejected.exchange(true);
        };
        static_assert(sizeof(ready) == sizeof(gpuFrame->readySemaphore));
        std::memcpy(&gpuFrame->readySemaphore, &ready, sizeof(ready));
        gpuFrame->rgb10a2 = sharedImage10;
        gpuFrame->width = outputWidth;
        gpuFrame->height = outputHeight;
        gpuFrame->sourceFrameIndex = output->thumbnailSourceFrame;
        const bool fenceCompletion = vulkanNeedsFenceCompletion(producer);
        gpuFrame->resourceLease = std::shared_ptr<void>(
            new pl_tex(frameOutputTexture), [owner = selectedDevice, producer, ready, waitSemaphores, destroySemaphore, fenceCompletion](void* ptr) {
                const uint64_t value = 1;
                VkSemaphoreWaitInfo wait{};
                wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                wait.semaphoreCount = 1;
                wait.pSemaphores = &ready;
                wait.pValues = &value;
                if (fenceCompletion)
                    pl_gpu_finish(owner->gpu);
                else
                    waitSemaphores(producer->device, &wait, UINT64_MAX);
                auto* texture = static_cast<pl_tex*>(ptr);
                pl_tex_destroy(owner->gpu, texture);
                destroySemaphore(producer->device, ready, nullptr);
                delete texture;
            });
        if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_PROFILE"))
            spdlog::info("GALLERY_PERF event=gallery_shared_image width={} height={} direct=true resident_input={}",
                outputWidth, outputHeight, externalGpu != nullptr);
        *output->gpuBuffer = std::move(gpuFrame);
        rgb.clear();
        return true;
    }
#ifndef _WIN32
    if (exported) {
        const uint64_t debugSampleHash =
            rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_GPU_HASH")
            ? samplePackedFrame(selected->gpu, exported,
                orientation == 90 || orientation == 270 ? sourceHeight : sourceWidth,
                orientation == 90 || orientation == 270 ? sourceWidth : sourceHeight)
            : 0;
        const bool ready = pl_buf_export(selected->gpu, exported);
        while (ready && pl_buf_poll(selected->gpu, exported, UINT64_MAX)) {}
        const int fd = ready ? ::dup(exported->shared_mem.handle.fd) : -1;
        if (fd < 0) {
            pl_buf_destroy(selected->gpu, &exported);
            return false;
        }
        // Queue the thumbnail pass after the displayed frame's export fence.
        // The thumbnail copy can finish later without extending the frame's
        // producer/presenter handoff wait.
        enqueueThumbnailReadback();
        auto gpuFrame = std::make_shared<VulkanGalleryGpuBuffer>();
        gpuFrame->fd = fd;
        gpuFrame->bytes = paddedOutput;
        gpuFrame->allocationBytes = exported->shared_mem.size;
        gpuFrame->allocationOffset = exported->shared_mem.offset;
        gpuFrame->drmFormatModifier = exported->shared_mem.drm_format_mod;
        gpuFrame->width = orientation == 90 || orientation == 270
            ? sourceHeight : sourceWidth;
        gpuFrame->height = orientation == 90 || orientation == 270
            ? sourceWidth : sourceHeight;
        gpuFrame->rgb10a2 = rgb10a2;
        gpuFrame->debugSampleHash = debugSampleHash;
        gpuFrame->release = [owner = selectedDevice, buffer = exported] {
            std::lock_guard lock(deviceMutex);
            pl_buf releaseBuffer = buffer;
            pl_buf_destroy(owner->gpu, &releaseBuffer);
        };
        if (rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_PERF_PROFILE"))
            spdlog::info("GALLERY_PERF event=gallery_dma_buf width={} height={} direct={} resident_input={}",
                gpuFrame->width, gpuFrame->height, !exportNeedsCopy, externalGpu != nullptr);
        *output->gpuBuffer = std::move(gpuFrame);
        rgb.clear();
        return true;
    }
#endif
    if (output && output->requireZeroCopy) return false;
    const bool hostImport = texture && !rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_STAGED_READBACK") &&
        (selected->gpu->import_caps.buf & PL_HANDLE_HOST_PTR);
    const bool directView = hostImport && output && output->hostPixels && rgb24 &&
        !alignedRgb24 && !rgb10a2 &&
        !rawpreamp::debugEnvironment("RAWPREAMP_GALLERY_VECTOR_READBACK");
    std::shared_ptr<uint8_t> owned;
    pl_buf imported = nullptr;
    const bool pooled = directView &&
        selected->acquireHostFrame(paddedOutput, imported, owned);
    std::vector<uint8_t> result;
    if (!pooled) result.resize(paddedOutput);
    bool readOk = false;
    // Keep imported display buffers until Qt releases their host memory.
    // Other paths can still import the vector or use staging.
    if (hostImport && !pooled) {
        pl_buf_params params = {};
        params.size = paddedOutput;
        params.host_readable = true;
        params.import_handle = PL_HANDLE_HOST_PTR;
        params.shared_mem.handle.ptr = result.data();
        params.shared_mem.size = paddedOutput;
        imported = pl_buf_create(selected->gpu, &params);
    }
    if (!imported && !selected->reserveDownload(paddedOutput)) return false;
    pl_buf_copy(selected->gpu, imported ? imported : selected->download, 0,
                selected->output, 0, paddedOutput);
    if (imported) {
        while (pl_buf_poll(selected->gpu, imported, UINT64_MAX)) {}
        if (!pooled) pl_buf_destroy(selected->gpu, &imported);
        readOk = true;
    } else {
        readOk = pl_buf_read(selected->gpu, selected->download, 0,
                             result.data(), paddedOutput);
    }
    if (!readOk || pl_gpu_is_failed(selected->gpu)) {
        selectedDevice.reset();
        if (!texture) unavailable = true;
        return false;
    }
    enqueueThumbnailReadback();
    if (pooled) {
        *output->hostPixels = std::move(owned);
        rgb.clear();
        return true;
    }
    result.resize(outputBytes);
    if (alignedRgb24) {
        const size_t count = static_cast<size_t>(pixels);
        for (size_t i = 0; i < count; ++i) {
            result[i * 3u] = result[i * 4u];
            result[i * 3u + 1u] = result[i * 4u + 1u];
            result[i * 3u + 2u] = result[i * 4u + 2u];
        }
        result.resize(count * 3u);
    } else if (rgb10a2) {
        const size_t count = static_cast<size_t>(pixels);
        for (size_t i = 0; i < count; ++i) {
            uint32_t packed = 0;
            std::memcpy(&packed, result.data() + i * 4u, sizeof(packed));
            for (unsigned channel = 0; channel < 3; ++channel) {
                const uint32_t code = (packed >> (channel * 10u)) & 1023u;
                result[i * 3u + channel] = static_cast<uint8_t>(
                    (code * 255u + 511u) / 1023u);
            }
        }
        result.resize(count * 3u);
    }
    rgb = std::move(result);
    return true;
}
} // namespace rawpreamp
#endif

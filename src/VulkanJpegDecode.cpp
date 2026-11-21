#include "DebugInstrumentation.h"
#include "SharedVulkanDevice.h"
#include "VulkanJpegDecode.h"

#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include "VulkanJpegContext.h"
#include "VulkanGalleryPreview.h"
#include <libplacebo/gpu.h>
#include <libplacebo/vulkan.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace rawpreamp {
namespace {

constexpr uint32_t kMetaWords = 77;
constexpr uint32_t kLookupBits = 7;
constexpr uint32_t kLookupSize = 2u << kLookupBits;
// One invocation entropy-decodes each tile serially. Large, single-scan DNGs
// are substantially slower than CPU decode and can monopolize the GPU.
constexpr uint64_t kMaxSerialTileSamples = 65536;
constexpr char residentMetaShader[] = R"glsl(#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Lengths { uint words[]; } lengths;
layout(std430, binding = 1) readonly buffer Table { uint words[]; } table;
layout(std430, binding = 2) writeonly buffer Metadata { uint words[]; } meta;
layout(std430, binding = 3) writeonly buffer Lookups { uint words[]; } lookup;
layout(push_constant) uniform Constants {
    uint tiles; uint width; uint height; uint channels;
    uint capacityBytes; uint maxBodyBytes; uint slotWords; uint bits;
} p;
void main() {
    uint tile = gl_GlobalInvocationID.x;
    if (tile >= p.tiles) return;
    uint m = tile * 77u;
    uint begin = tile * p.slotWords * 4u;
    uint headerBytes = 41u + table.words[1] + 5u * p.channels;
    uint bodyBytes = lengths.words[tile];
    meta.words[m] = begin + headerBytes;
    meta.words[m + 1u] = bodyBytes;
    meta.words[m + 2u] = p.width;
    meta.words[m + 3u] = p.height;
    meta.words[m + 4u] = p.bits;
    meta.words[m + 5u] = table.words[0];
    meta.words[m + 6u] = tile * p.width * p.height * p.channels;
    meta.words[m + 7u] = 0u;
    for (uint i = 0u; i < 16u; ++i)
        meta.words[m + 8u + i] = table.words[2u + i];
    for (uint i = 0u; i < 17u; ++i)
        meta.words[m + 24u + i] = table.words[18u + i];
    meta.words[m + 41u] = p.channels;
    meta.words[m + 76u] = 0u;
    bool good = bodyBytes > 0u && bodyBytes <= p.maxBodyBytes &&
                begin <= p.capacityBytes &&
                headerBytes + bodyBytes + 2u <= p.capacityBytes - begin &&
                table.words[0] >= 1u && table.words[0] <= 7u &&
                table.words[1] > 0u && table.words[1] <= 17u;
    if (tile != 0u) {
        meta.words[m + 42u] = good ? 1u : 0u;
        return;
    }
    uint base = 0u;
    for (uint i = 0u; i < 128u; ++i) lookup.words[base + i] = 0u;
    uint code = 0u, symbol = 0u;
    for (uint length = 1u; length <= 16u; ++length) {
        uint count = table.words[1u + length];
        if (length <= 7u) {
            for (uint i = 0u; i < count; ++i) {
                uint start = (code + i) << (7u - length);
                uint repeat = 1u << (7u - length);
                if (start + repeat > 128u) { good = false; break; }
                for (uint j = 0u; j < repeat; ++j)
                    lookup.words[base + start + j] =
                        (table.words[18u + symbol + i] << 8u) | length;
            }
        }
        symbol += count;
        code = (code + count) << 1u;
    }
    meta.words[m + 42u] = good ? 1u : 0u;
}
)glsl";
constexpr char parseShader[] = R"glsl(#version 450
#ifndef JPEG_LOOKUP_BITS
#define JPEG_LOOKUP_BITS 7u
#endif
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Stream { uint words[]; } stream;
layout(std430, binding = 1) buffer Metadata { uint words[]; } meta;
layout(std430, binding = 2) writeonly buffer Lookups { uint words[]; } lookup;
layout(std430, binding = 3) readonly buffer Seeds { uint words[]; } seed;
layout(push_constant) uniform Constants { uint tiles;
#ifdef JPEG_SHARED_LOOKUP
    uint lookupTables;
#endif
} p;
uint byteAt(uint index) {
    return (stream.words[index >> 2u] >> ((index & 3u) * 8u)) & 255u;
}
uint big16(uint index) { return (byteAt(index) << 8u) | byteAt(index + 1u); }
void main() {
    uint tile = gl_GlobalInvocationID.x;
    if (tile >= p.tiles) return;
    uint m = tile * 77u;
#ifdef JPEG_SHARED_LOOKUP
    uint s = tile * 8u;
#else
    uint s = tile * 7u;
#endif
    uint begin = seed.words[s], size = seed.words[s + 1u], end = begin + size;
    uint width = seed.words[s + 2u], height = seed.words[s + 3u];
    uint bits = seed.words[s + 4u], channels = seed.words[s + 6u];
    meta.words[m] = begin;
    meta.words[m + 1u] = size;
    meta.words[m + 2u] = width;
    meta.words[m + 3u] = height;
    meta.words[m + 4u] = bits;
    meta.words[m + 6u] = seed.words[s + 5u];
    meta.words[m + 41u] = channels;
    bool good = size >= 16u && end >= begin &&
                byteAt(begin) == 255u && byteAt(begin + 1u) == 216u;
    bool frame = false, table = false, scan = false;
    uint tables = 0u;
    meta.words[m + 76u] = 0u;
    uint ids[3] = uint[3](0u, 0u, 0u);
    uint pos = begin + 2u;
    for (uint section = 0u; section < 32u && good && !scan; ++section) {
        if (pos + 4u > end || byteAt(pos++) != 255u) { good = false; break; }
        while (pos < end && byteAt(pos) == 255u) ++pos;
        if (pos + 2u > end) { good = false; break; }
        uint marker = byteAt(pos++);
        uint length = big16(pos);
        if (length < 2u || length > end - pos) { good = false; break; }
        uint payload = pos + 2u, n = length - 2u;
        pos += length;
        if (marker == 195u) {
            if (channels == 1u && n >= 6u && byteAt(payload + 5u) == 2u &&
                big16(payload + 3u) * 2u == width) {
                channels = 2u; width /= 2u;
                meta.words[m + 2u] = width; meta.words[m + 41u] = channels;
            }
            if (frame || n != 6u + 3u * channels ||
                byteAt(payload) != bits || big16(payload + 1u) != height ||
                big16(payload + 3u) != width ||
                byteAt(payload + 5u) != channels) { good = false; break; }
            for (uint c = 0u; c < channels; ++c) {
                uint at = payload + 6u + 3u * c;
                ids[c] = byteAt(at);
                if (byteAt(at + 1u) != 17u || byteAt(at + 2u) != 0u)
                    good = false;
                for (uint earlier = 0u; earlier < c; ++earlier)
                    if (ids[earlier] == ids[c]) good = false;
            }
            frame = true;
        } else if (marker == 196u) {
            uint at = payload;
            while (at < payload + n && good) {
                if (payload + n - at < 17u) { good = false; break; }
                uint id = byteAt(at++);
                if (id > 1u || (tables & (1u << id)) != 0u) { good = false; break; }
                uint counts = id == 0u ? 8u : 43u, symbols = id == 0u ? 24u : 59u;
                uint total = 0u;
                for (uint i = 0u; i < 16u; ++i) {
                    uint count = byteAt(at + i);
                    meta.words[m + counts + i] = count; total += count;
                }
                at += 16u;
                if (total == 0u || total > 17u || total > payload + n - at) {
                    good = false; break;
                }
                for (uint i = 0u; i < total; ++i) {
                    uint category = byteAt(at + i);
                    if (category > 16u) good = false;
                    meta.words[m + symbols + i] = category;
                }
                at += total;
                tables |= 1u << id;
            }
            table = tables != 0u;
        } else if (marker == 218u) {
            if (!frame || !table || n != 4u + 2u * channels ||
                byteAt(payload) != channels) { good = false; break; }
            for (uint c = 0u; c < channels; ++c) {
                uint id = byteAt(payload + 2u + 2u * c) >> 4u;
                if (byteAt(payload + 1u + 2u * c) != ids[c] || id > 1u ||
                    (byteAt(payload + 2u + 2u * c) & 15u) != 0u ||
                    (tables & (1u << id)) == 0u) good = false;
                meta.words[m + 76u] |= id << (c * 4u);
            }
            uint selector = payload + 1u + 2u * channels;
            uint mode = byteAt(selector);
            if (mode < 1u || mode > 7u || byteAt(selector + 1u) != 0u ||
                byteAt(selector + 2u) != 0u ||
                pos + 2u > end || byteAt(end - 2u) != 255u ||
                byteAt(end - 1u) != 217u) { good = false; break; }
#ifdef JPEG_SERIAL_VALIDATE
            if (width * height * channels <= 4096u) {
                for (uint at = pos; at < end - 2u && good;) {
                    if ((at & 3u) == 0u && at + 4u <= end - 2u) {
                        uint packed = stream.words[at >> 2u];
                        if ((packed & 255u) != 255u &&
                            ((packed >> 8u) & 255u) != 255u &&
                            ((packed >> 16u) & 255u) != 255u &&
                            (packed >> 24u) != 255u) {
                            at += 4u;
                            continue;
                        }
                    }
                    if (byteAt(at++) != 255u) continue;
                    if (at >= end - 2u || byteAt(at++) != 0u) good = false;
                }
            }
#endif
            meta.words[m] = pos;
            meta.words[m + 1u] = end - 2u - pos;
            meta.words[m + 5u] = mode;
            scan = true;
        } else if (marker < 224u || marker > 239u) {
            good = false;
        }
    }
    good = good && frame && table && scan && meta.words[m + 1u] > 0u;
    // Separate banks keep single-table tiles densely packed in cache even
    // when paired-component streams need a second independent table.
    uint lookupId=tile,lookupTables=p.tiles;
#ifdef JPEG_SHARED_LOOKUP
    lookupId=seed.words[s+7u]&0x7fffffffu;
    lookupTables=p.lookupTables;
    meta.words[m+7u]=lookupId;meta.words[m+42u]=good?1u:0u;
    if((seed.words[s+7u]&0x80000000u)==0u) return;
#endif
    uint lookupWidth=1u<<JPEG_LOOKUP_BITS;
    uint lookupBase=lookupId*lookupWidth;
    for (uint i = 0u; i < (lookupWidth * 2u); ++i) {
        uint tableId = i / lookupWidth;
        if ((tables & (1u << tableId)) != 0u)
            lookup.words[lookupBase + tableId * lookupTables * lookupWidth + i % lookupWidth] = 0u;
    }
    for (uint id = 0u; id < 2u && good; ++id) {
        if ((tables & (1u << id)) == 0u) continue;
        uint counts = id == 0u ? 8u : 43u, symbols = id == 0u ? 24u : 59u;
        uint code = 0u, symbol = 0u;
        for (uint length = 1u; length <= 16u; ++length) {
            uint count = meta.words[m + counts + length - 1u];
            if (length <= JPEG_LOOKUP_BITS) {
                for (uint i = 0u; i < count; ++i) {
                    uint start = (code + i) << (JPEG_LOOKUP_BITS - length);
                    uint repeat = 1u << (JPEG_LOOKUP_BITS - length);
                    if (start + repeat > lookupWidth) { good = false; break; }
                    for (uint j = 0u; j < repeat; ++j)
                        lookup.words[lookupBase + id * lookupTables * lookupWidth + start + j] =
                            (meta.words[m + symbols + symbol + i] << 8u) | length;
                }
            }
            symbol += count; code = (code + count) << 1u;
        }
    }
    meta.words[m + 7u] = lookupId;
    meta.words[m + 42u] = good ? 1u : 0u;
}
)glsl";
// The entropy byte-stuffing check has no serial dependencies. Validate it
// across the workgroup instead of scanning a large tile in one parser lane.
constexpr char validateStreamShader[] = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(std430,binding=0) readonly buffer Stream { uint words[]; } stream;
layout(std430,binding=1) buffer Metadata { uint words[]; } meta;
layout(push_constant) uniform Constants { uint tiles; } p;
uint byteAt(uint at) { return (stream.words[at/4u] >> ((at%4u)*8u)) & 255u; }
shared uint valid;
void main() {
    uint tile=gl_WorkGroupID.x+gl_WorkGroupID.y*gl_NumWorkGroups.x;
    if(tile>=p.tiles) return;
    uint m=tile*77u, begin=meta.words[m], size=meta.words[m+1u];
    if(gl_LocalInvocationID.x==0u) valid=meta.words[m+42u];
    barrier();
    if (valid == 0u) return;
#ifdef JPEG_SERIAL_VALIDATE
    if (meta.words[m+2u]*meta.words[m+3u]*meta.words[m+41u]<=4096u) return;
#endif
    for(uint i=gl_LocalInvocationID.x;i<size;i+=256u) {
        if(byteAt(begin+i)==255u && (i+1u>=size || byteAt(begin+i+1u)!=0u))
            atomicAnd(meta.words[m+42u],0u);
    }
}
)glsl";
constexpr char decodeShader[] = R"glsl(#version 450
#ifndef JPEG_LOOKUP_BITS
#define JPEG_LOOKUP_BITS 7u
#endif
#ifndef JPEG_PREFILL_BITS
#define JPEG_PREFILL_BITS 7u
#endif
#ifndef JPEG_GROUP_SIZE
#define JPEG_GROUP_SIZE 8
#endif
layout(local_size_x = JPEG_GROUP_SIZE) in;
layout(std430, binding = 0) readonly restrict buffer Stream { uint words[]; } stream;
layout(std430, binding = 1) readonly restrict buffer Metadata { uint words[]; } meta;
layout(std430, binding = 2) restrict buffer Image { uint words[]; } image;
layout(std430, binding = 3) writeonly restrict buffer Status { uint words[]; } status;
layout(std430, binding = 4) readonly restrict buffer Lookups { uint words[]; } lookup;
#ifdef JPEG_RESIDENT_DIRECT
layout(push_constant) uniform Constants {
    uint tiles; uint fullWidth; uint tileWidth; uint tileHeight; uint tilesAcross; uint lookupStride;
} p;
#else
layout(push_constant) uniform Constants { uint tiles; uint lookupStride; } p;
#endif
uint getByte(uint index) {
    return (stream.words[index >> 2u] >> ((index & 3u) * 8u)) & 255u;
}
#ifdef JPEG_BIT_RESERVOIR
bool refill(uint needed, uint offset, uint size, inout uint pos,
            inout uint cache, inout uint available) {
    while (available < needed && pos < size) {
        uint value = getByte(offset + pos);
        cache |= value << (24u - available);
        available += 8u; pos += value == 255u ? 2u : 1u;
    }
    return available >= needed && pos <= size;
}
#endif
uint pixel(uint index) {
    uint word = image.words[index >> 1u];
    return ((index & 1u) == 0u) ? (word & 65535u) : (word >> 16u);
}
int predictor(uint mode, int a, int b, int c) {
    if (mode == 1u) return a;
    if (mode == 2u) return b;
    if (mode == 3u) return c;
    if (mode == 4u) return a + b - c;
    if (mode == 5u) return a + ((b - c) >> 1);
    if (mode == 6u) return b + ((a - c) >> 1);
    return (a + b) >> 1;
}
void main() {
    uint tile = gl_GlobalInvocationID.x;
    if (tile >= p.tiles) return;
    uint m = tile * 77u;
    if (meta.words[m + 42u] != 1u) { status.words[tile] = 0u; return; }
    uint streamOffset = meta.words[m], streamBytes = meta.words[m + 1u];
    uint width = meta.words[m + 2u], height = meta.words[m + 3u];
    uint bits = meta.words[m + 4u];
#ifdef JPEG_PREDICTOR
    const uint mode = JPEG_PREDICTOR;
#else
    uint mode = meta.words[m + 5u];
#endif
    uint outputBase = meta.words[m + 6u];
#ifdef JPEG_COMPONENTS
    const uint channels = JPEG_COMPONENTS;
#else
    uint channels = meta.words[m + 41u];
#endif
    uint lookupBase = meta.words[m + 7u] * (1u << JPEG_LOOKUP_BITS);
    uint tableSelectors = meta.words[m + 76u];
    uint rowStride = width * channels;
#ifdef JPEG_RESIDENT_DIRECT
    rowStride = p.fullWidth * (channels == 2u ? 1u : channels);
    uint tileX = (tile % p.tilesAcross) * p.tileWidth;
    uint tileY = (tile / p.tilesAcross) * p.tileHeight;
#endif
    uint bytePos = 0u, pendingValue = 0u;
#ifdef JPEG_BIT_RESERVOIR
    uint cache = 0u, available = 0u;
#else
    uint bit = 0u;
#endif
    uint sampleIndex = 0u;
    uint first = 0u, index = 0u, mask = (1u << bits) - 1u;
    bool good = true;
    for (uint y = 0u; y < height && good; ++y) {
        int left[3] = int[3](0, 0, 0);
        int aboveLeft[3] = int[3](0, 0, 0);
        for (uint x = 0u; x < width && good; ++x) {
          for (uint component = 0u; component < channels; ++component) {
            uint tableId = (tableSelectors >> (component * 4u)) & 15u;
            uint counts = tableId == 0u ? 8u : 43u, symbols = tableId == 0u ? 24u : 59u;
            uint category = 0xffffffffu;
            int diff = 0;
#ifdef JPEG_BIT_RESERVOIR
            // Fill ahead for both the Huffman prefix and residual. A 24-bit
            // target keeps every byte-append shift valid in the 32-bit cache;
            // short tails still use the existing available-bit checks.
            refill(JPEG_PREFILL_BITS, streamOffset, streamBytes, bytePos, cache, available);
            uint look = lookup.words[lookupBase + tableId * p.lookupStride + (cache >> (32u - JPEG_LOOKUP_BITS))];
            uint length = look & 255u;
#ifdef JPEG_COMBINED_RESIDUAL
            // The prefilled reservoir usually holds both fields. Consume them
            // together, leaving short tails and long codes on the checked path.
            uint fastCategory = look >> 8u;
            uint extra = fastCategory == 16u ? 0u : fastCategory;
            uint consumed = length + extra;
            if (length != 0u && fastCategory <= 16u && consumed <= available) {
                category = fastCategory;
                uint value = extra == 0u ? 0u : (cache << length) >> (32u - extra);
                cache <<= consumed; available -= consumed;
                diff = category == 16u ? 32768 : int(value);
                if (extra != 0u && value < (1u << (extra - 1u)))
                    diff += 1 - int(1u << extra);
            } else {
#endif
            if (length != 0u && length <= available) {
                category = look >> 8u;
                cache <<= length; available -= length;
            } else {
                first = 0u; index = 0u;
                uint code = 0u;
                for (uint len = 1u; len <= 16u; ++len) {
                    if (!refill(1u, streamOffset, streamBytes, bytePos, cache, available)) {
                        good = false; break;
                    }
                    code = (code << 1u) | (cache >> 31u);
                    cache <<= 1u; --available;
                    uint count = meta.words[m + counts + len - 1u];
                    if (code >= first && code - first < count) {
                        category = meta.words[m + symbols + index + code - first]; break;
                    }
                    index += count; first = (first + count) << 1u;
                }
            }
            if (!good || category > 16u) { good = false; break; }
            if (category == 16u) diff = 32768;
            else if (category != 0u) {
                if (!refill(category, streamOffset, streamBytes, bytePos, cache, available)) {
                    good = false; break;
                }
                uint value = cache >> (32u - category);
                cache <<= category; available -= category;
                diff = int(value);
                if (value < (1u << (category - 1u))) diff += 1 - int(1u << category);
            }
#ifdef JPEG_COMBINED_RESIDUAL
            }
#endif
#else
            if (bytePos >= streamBytes) { good = false; break; }
            uint current = getByte(streamOffset + bytePos);
            uint nextByte = bytePos + (current == 255u ? 2u : 1u);
            uint window = current << 8u;
            if (nextByte < streamBytes)
                window |= getByte(streamOffset + nextByte);
            uint look = lookup.words[lookupBase + tableId * p.lookupStride +
                                     ((window << bit) >> 9u & 127u)];
            if (look != 0u) {
                category = look >> 8u;
                uint consumed = bit + (look & 255u);
                if (consumed >= 8u)
                    bytePos = nextByte;
                bit = consumed & 7u;
                if (bytePos > streamBytes ||
                    (bytePos == streamBytes && bit != 0u)) good = false;
            } else {
                first = 0u; index = 0u;
                uint code = 0u;
                for (uint len = 1u; len <= 16u; ++len) {
                    if (bytePos >= streamBytes) { good = false; break; }
                    current = getByte(streamOffset + bytePos);
                    code = (code << 1u) | ((current >> (7u - bit)) & 1u);
                    if (++bit == 8u) {
                        bit = 0u;
                        bytePos += current == 255u ? 2u : 1u;
                    }
                    uint count = meta.words[m + counts + len - 1u];
                    if (code >= first && code - first < count) {
                        category = meta.words[m + symbols + index + code - first];
                        break;
                    }
                    index += count;
                    first = (first + count) << 1u;
                }
            }
            if (!good || category > 16u) { good = false; break; }
            if (category == 16u) {
                diff = 32768;
            } else if (category != 0u) {
                if (bytePos >= streamBytes) { good = false; break; }
                uint b0 = getByte(streamOffset + bytePos);
                uint p1 = bytePos + (b0 == 255u ? 2u : 1u);
                uint needed = bit + category;
                uint b1 = 0u, b2 = 0u, p2 = p1;
                if (needed > 8u) {
                    if (p1 >= streamBytes) { good = false; break; }
                    b1 = getByte(streamOffset + p1);
                    p2 = p1 + (b1 == 255u ? 2u : 1u);
                }
                if (needed > 16u) {
                    if (p2 >= streamBytes) { good = false; break; }
                    b2 = getByte(streamOffset + p2);
                }
                uint value = ((b0 << 16u) | (b1 << 8u) | b2) >>
                             (24u - needed);
                value &= (1u << category) - 1u;
                if (needed >= 16u) bytePos = p2;
                else if (needed >= 8u) bytePos = p1;
                bit = needed & 7u;
                if (bytePos > streamBytes ||
                    (bytePos == streamBytes && bit != 0u)) { good = false; break; }
                diff = int(value);
                if (value < (1u << (category - 1u))) diff += 1 - int(1u << category);
            }
#endif
            uint i = sampleIndex++;
            uint outputIndex = outputBase + i;
#ifdef JPEG_RESIDENT_DIRECT
            outputIndex = ((tileY + y) * p.fullWidth + tileX) *
                          (channels == 2u ? 1u : channels) + x * channels + component;
#endif
            int pred;
            if (y == 0u && x == 0u) pred = int(1u << (bits - 1u));
            else if (y == 0u) pred = left[component];
            else {
                const bool needsAbove = x == 0u || mode != 1u;
                int b = needsAbove ? int(pixel(outputIndex - rowStride)) : 0;
                if (x == 0u) pred = b;
                else {
                    int c = mode >= 3u && mode <= 6u
                        ? aboveLeft[component] : 0;
                    pred = predictor(mode, left[component], b, c);
                }
                if (mode >= 3u && mode <= 6u)
                    aboveLeft[component] = b;
            }
            uint value = uint(pred + diff) & mask;
            left[component] = int(value);
            if ((i & 1u) == 0u) { pendingValue = value; }
            else {
                image.words[outputIndex >> 1u] = pendingValue | (value << 16u);
            }
          }
        }
    }
    if (good && (sampleIndex & 1u) != 0u)
        image.words[(outputBase + sampleIndex - 1u) >> 1u] = pendingValue;
    status.words[tile] = good ? 1u : 0u;
}
)glsl";

constexpr char scatterShader[] = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Tiles { uint words[]; } tiles;
layout(std430, binding = 1) readonly buffer Metadata { uint words[]; } meta;
layout(std430, binding = 2) writeonly buffer Image { uint words[]; } image;
layout(push_constant) uniform Constants {
    uint width; uint height; uint tileWidth; uint tileHeight; uint tilesAcross;
    uint channels;
} p;
uint sampleAt(uint rowSample, uint y) {
    uint x = rowSample / p.channels;
    uint component = rowSample % p.channels;
    uint tile = (y / p.tileHeight) * p.tilesAcross + x / p.tileWidth;
    uint local = ((y % p.tileHeight) * p.tileWidth + x % p.tileWidth) *
                 p.channels + component;
    uint base = meta.words[tile * 77u + 6u];
    uint word = tiles.words[(base + local) >> 1u];
    return (local & 1u) == 0u ? word & 65535u : word >> 16u;
}
void main() {
    uint pair = gl_GlobalInvocationID.x;
    uint wordsPerRow = (p.width * p.channels) >> 1u;
    if (pair >= wordsPerRow * p.height) return;
    uint y = pair / wordsPerRow, rowSample = (pair % wordsPerRow) * 2u;
    image.words[pair] = sampleAt(rowSample, y) |
                        (sampleAt(rowSample + 1u, y) << 16u);
}
)glsl";

constexpr char validateShader[] = R"glsl(#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer Status { uint words[]; } status;
layout(std430, binding = 1) writeonly buffer Result { uint words[]; } result;
layout(push_constant) uniform Constants { uint tiles; } p;
void main() {
    uint valid = 1u;
    for (uint i = 0u; i < p.tiles; ++i)
        valid &= status.words[i] == 1u ? 1u : 0u;
    result.words[0] = valid;
}
)glsl";

struct ParsedTile {
    std::array<uint32_t, kMetaWords> meta{};
    std::array<uint32_t, kLookupSize> lookup{};
    size_t scanOffset = 0, scanLength = 0;
};

uint16_t be16(const uint8_t* p) { return (uint16_t(p[0]) << 8u) | p[1]; }

bool parseTile(const VulkanJpegTile& tile, ParsedTile& result) {
    if (!tile.bytes || tile.size < 16 || tile.size > UINT32_MAX ||
        tile.bytes[0] != 0xff || tile.bytes[1] != 0xd8 ||
        !tile.width || !tile.height || tile.width > 65535 ||
        tile.height > 65535 || tile.bits < 8 || tile.bits > 16 ||
        (tile.channels != 1 && tile.channels != 3))
        return false;
    bool frame = false, table = false, scan = false;
    uint32_t tables = 0, inputWidth = tile.width, components = tile.channels;
    std::array<uint8_t, 3> componentIds{};
    size_t pos = 2;
    while (pos + 4 <= tile.size) {
        if (tile.bytes[pos++] != 0xff) return false;
        while (pos < tile.size && tile.bytes[pos] == 0xff) ++pos;
        if (pos >= tile.size) return false;
        const uint8_t marker = tile.bytes[pos++];
        if (marker == 0xd9) break;
        if (marker == 0xd8 || marker == 0x01 ||
            (marker >= 0xd0 && marker <= 0xd7)) return false;
        if (pos + 2 > tile.size) return false;
        const uint16_t length = be16(tile.bytes + pos);
        if (length < 2 || length > tile.size - pos) return false;
        const uint8_t* payload = tile.bytes + pos + 2;
        const size_t n = length - 2;
        pos += length;
        if (marker == 0xc3) {
            if (n >= 6u && tile.channels == 1u && payload[5] == 2u &&
                uint32_t(be16(payload + 3)) * 2u == tile.width) {
                components = 2u; inputWidth = tile.width / 2u;
            }
            if (frame || n != 6u + 3u * components ||
                payload[0] != tile.bits ||
                be16(payload + 1) != tile.height ||
                be16(payload + 3) != inputWidth ||
                payload[5] != components)
                return false;
            for (uint32_t i = 0; i < components; ++i) {
                const size_t at = 6u + i * 3u;
                if (payload[at + 1u] != 0x11 || payload[at + 2u] != 0)
                    return false;
                componentIds[i] = payload[at];
                for (uint32_t earlier = 0; earlier < i; ++earlier)
                    if (componentIds[earlier] == componentIds[i]) return false;
            }
            frame = true;
        } else if (marker == 0xc4) {
            size_t at = 0;
            while (at < n) {
                if (n - at < 17) return false;
                const uint32_t id = payload[at++];
                if (id > 1u || (tables & (1u << id))) return false;
                const uint32_t counts = id == 0 ? 8 : 43, symbols = id == 0 ? 24 : 59;
                uint32_t total = 0;
                for (size_t i = 0; i < 16; ++i)
                    total += result.meta[counts + i] = payload[at + i];
                at += 16;
                if (!total || total > 17 || total > n - at) return false;
                for (size_t i = 0; i < total; ++i) {
                    if (payload[at + i] > 16) return false;
                    result.meta[symbols + i] = payload[at + i];
                }
                at += total; tables |= 1u << id;
            }
            table = tables != 0;
        } else if (marker == 0xda) {
            if (!frame || !table || scan ||
                n != 4u + 2u * components || payload[0] != components)
                return false;
            for (uint32_t i = 0; i < components; ++i) {
                const uint32_t selector = payload[2u + 2u * i];
                const uint32_t id = selector >> 4u;
                if (payload[1u + 2u * i] != componentIds[i] || id > 1u ||
                    (selector & 15u) || !(tables & (1u << id))) return false;
                result.meta[76] |= id << (4u * i);
            }
            const size_t selector = 1u + 2u * components;
            if (payload[selector] < 1 || payload[selector] > 7 ||
                payload[selector + 1u] != 0 || payload[selector + 2u] != 0)
                return false;
            result.meta[2] = inputWidth;
            result.meta[3] = tile.height;
            result.meta[4] = tile.bits;
            result.meta[5] = payload[selector];
            result.meta[41] = components;
            const size_t scanStart = pos;
            while (pos < tile.size) {
                const uint8_t byte = tile.bytes[pos++];
                if (byte != 0xff) continue;
                if (pos >= tile.size) return false;
                const uint8_t next = tile.bytes[pos++];
                if (next == 0) continue;
                if (next == 0xd9 && pos == tile.size) {
                    result.scanOffset = scanStart;
                    result.scanLength = pos - scanStart - 2u;
                    scan = true;
                    break;
                }
                return false;
            }
            break;
        } else if (marker == 0xdd) {
            // Restart intervals need a separate decoder.
            return false;
        } else if (marker < 0xe0 || marker > 0xef) {
            // Exclude unsupported coding and extension markers.
            return false;
        }
    }
    if (!frame || !table || !scan || !result.scanLength) return false;
    for (uint32_t id = 0; id < 2; ++id) {
        if (!(tables & (1u << id))) continue;
        const uint32_t counts = id == 0 ? 8 : 43, symbols = id == 0 ? 24 : 59;
        uint32_t code = 0, symbol = 0;
        for (uint32_t length = 1; length <= 16; ++length) {
            const uint32_t count = result.meta[counts + length - 1];
            if (length <= kLookupBits) {
                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t start = (code + i) << (kLookupBits - length);
                    const uint32_t repeat = 1u << (kLookupBits - length);
                    if (start + repeat > 128u) return false;
                    std::fill_n(result.lookup.begin() + id * 128u + start, repeat,
                        (result.meta[symbols + symbol + i] << 8u) | length);
                }
            }
            symbol += count; code = (code + count) << 1u;
        }
    }
    result.meta[42] = 1u;
    return true;
}

// Inspect only the small SOF/SOS headers to select a shader variant. Full
// marker, entropy-range and output validation remains in the normal parser.
uint32_t headerVariant(const VulkanJpegTile& tile) {
    size_t at = 2; uint32_t components = 0;
    while (at + 4 <= tile.size) {
        if (tile.bytes[at++] != 255) return 0;
        while (at < tile.size && tile.bytes[at] == 255) ++at;
        if (at + 3 > tile.size) return 0;
        const uint32_t marker = tile.bytes[at++], length = be16(tile.bytes + at);
        if (length < 2 || length > tile.size - at) return 0;
        const auto* payload = tile.bytes + at + 2;
        if (marker == 195 && length >= 8) components = payload[5];
        if (marker == 218) {
            if (components < 1 || components > 3 || length != 6 + 2 * components) return 0;
            const uint32_t predictor = payload[1 + 2 * components];
            return predictor >= 1 && predictor <= 7 ? components | (predictor << 8) : 0;
        }
        at += length;
    }
    return 0;
}

// Generate constant-component bodies from the same checked decoder. This
// scalarizes predictor state without maintaining a second entropy algorithm.
std::string unrollComponents(std::string body, uint32_t channels) {
    const auto loop=body.find("for (uint component = 0u; component < channels; ++component)");
    if (loop==std::string::npos) return body;
    const auto begin=body.find('{',loop);
    size_t end=begin+1;unsigned depth=1;
    for (;end<body.size() && depth;++end) {
        if(body[end]=='{') ++depth;
        else if(body[end]=='}') --depth;
    }
    if (depth) return body;
    const auto inner=body.substr(begin+1,end-begin-2);
    std::string expanded;
    for(uint32_t c=0;c<channels;++c)
        expanded+="{ const uint component = "+std::to_string(c)+"u;\n"+inner+"\n}\n";
    body.replace(loop,end-loop,expanded);
    return body;
}
std::string specializeComponents(std::string source, uint32_t channels) {
    if (channels) return unrollComponents(std::move(source), channels);
    const auto begin = source.find("void main()");
    if (begin == std::string::npos) return source;
    std::string generic = source.substr(begin), specialized;
    const auto first = generic.find("#ifdef JPEG_COMPONENTS");
    const auto last = generic.find("#endif", first);
    if (first == std::string::npos || last == std::string::npos) return source;
    for (uint32_t count : {1u, 2u}) {
        std::string body = generic;
        body.replace(first, last + 6 - first, "const uint channels = " + std::to_string(count) + "u;");
        body = unrollComponents(std::move(body), count);
        body.replace(body.find("void main()"), 11, "void decode" + std::to_string(count) + "()");
        specialized += body;
    }
    generic.replace(generic.find("void main()"), 11, "void decodeGeneric()");
    source.resize(begin);
    return source + generic + specialized + R"glsl(
void main() {
    uint tile = gl_GlobalInvocationID.x;
    if (tile >= p.tiles) return;
    uint channels = meta.words[tile * 77u + 41u];
    if (channels == 1u) decode1();
    else if (channels == 2u) decode2();
    else decodeGeneric();
}
)glsl";
}

// Predictor 3 is common in tiled CFA DNGs. Select it once per tile so the
// compiler removes the per-sample predictor switch and unused predictor state.
std::string specializeDiagonal(std::string source, uint32_t channels) {
    if (channels > 1u || source.find("#define JPEG_PREDICTOR ") != std::string::npos)
        return source;
    const auto begin = source.find(channels ? "void main()" : "void decode1()");
    const auto end = channels ? source.size() : source.find("void decode2()", begin);
    if (begin == std::string::npos || end == std::string::npos) return source;
    std::string body = source.substr(begin, end - begin);
    const auto predictor = body.find("#ifdef JPEG_PREDICTOR");
    const auto last = body.find("#endif", predictor);
    if (predictor == std::string::npos || last == std::string::npos) return source;
    body.replace(predictor, last + 6u - predictor, "const uint mode = 3u;");
    body.replace(0, body.find('('), "void decodeDiagonal");
    if (channels) {
        source.replace(begin, 11u, "void decodeDynamic()");
        return source + body + R"glsl(
void main() {
    uint tile = gl_GlobalInvocationID.x;
    if (tile >= p.tiles) return;
    if (meta.words[tile * 77u + 5u] == 3u) decodeDiagonal();
    else decodeDynamic();
}
)glsl";
    }
    source.insert(source.rfind("void main()"), body);
    const std::string branch = "if (channels == 1u) decode1();";
    const auto dispatch = source.rfind(branch);
    if (dispatch != std::string::npos)
        source.replace(dispatch, branch.size(),
            "if (channels == 1u && meta.words[tile * 77u + 5u] == 3u) decodeDiagonal();\n"
            "    else if (channels == 1u) decode1();");
    return source;
}

// Keys cover DHT payload bytes only. Full headers and entropy are validated on
// GPU as before; identical keys share one designated GPU table builder.
bool jpegTableKey(const VulkanJpegTile &tile, std::string &key) {
    key.clear();
    key.reserve(68u);
    size_t at = 2u;
    for (unsigned section = 0; section < 32u; ++section) {
        if (at + 4u > tile.size || tile.bytes[at++] != 255u)
            return false;
        while (at < tile.size && tile.bytes[at] == 255u)
            ++at;
        if (at + 3u > tile.size)
            return false;
        const uint32_t marker = tile.bytes[at++];
        const size_t length = (size_t(tile.bytes[at]) << 8u) | tile.bytes[at + 1u];
        if (length < 2u || length > tile.size - at)
            return false;
        if (marker == 218u)
            return !key.empty();
        if (marker == 196u) {
            if (key.size() + length - 2u > 68u)
                return false;
            key.append(reinterpret_cast<const char *>(tile.bytes + at + 2u), length - 2u);
        }
        at += length;
    }
    return false;
}

uint32_t jpegLookupBits(uint32_t fallback = 7u) {
    const char *reservoir = rawpreamp::debugEnvironment("RAWPREAMP_JPEG_RESERVOIR");
    if (reservoir && std::strcmp(reservoir, "0") == 0)
        return 7u;
    const char *bits = rawpreamp::debugEnvironment("RAWPREAMP_JPEG_LOOKUP_BITS");
    if (bits && std::strcmp(bits, "7") == 0)
        return 7u;
    if (bits && std::strcmp(bits, "auto9") == 0) return 9u;
    if (bits && std::strcmp(bits, "9") == 0)
        return 9u;
    if (bits && std::strcmp(bits, "10") == 0)
        return 10u;
    return fallback;
}

// Debug comparison only: header, bounds and decoder status checks stay active.
bool jpegNeedsPrescan() {
    return !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_SKIP_SCAN");
}

struct DecodeState {
    pl_gpu gpu = nullptr;
    uint32_t sourceGroupSize = 8u, encodedGroupSize = 8u;
    std::atomic<uint32_t> preferredLookupBits{7u};
    pl_pass parser = nullptr, streamValidator = nullptr;
    pl_pass residentMeta = nullptr;
    std::unordered_map<uint32_t,pl_pass> wideMeta,wideParsers;
    pl_pass pass = nullptr;
    pl_pass residentDirect = nullptr;
    std::unordered_map<uint32_t, pl_pass> variants;
    pl_pass scatter = nullptr;
    pl_pass validate = nullptr;
    pl_timer timer = nullptr;
    pl_timer scatterTimer = nullptr;
    pl_timer parserTimer = nullptr;
    pl_buf stream = nullptr, metadata = nullptr, image = nullptr;
    pl_buf streamUpload = nullptr, metadataUpload = nullptr;
    pl_buf seed = nullptr, seedUpload = nullptr;
    pl_buf status = nullptr, readback = nullptr, statusReadback = nullptr;
    pl_buf lookup = nullptr, lookupUpload = nullptr;
    pl_buf assembled = nullptr;
    std::shared_ptr<void> assembledLease;
    pl_buf valid = nullptr;
    pl_buf validReadback = nullptr;
    ~DecodeState() {
        if (!gpu) return;
        for (auto& [key, variant] : variants) pl_pass_destroy(gpu, &variant);
        pl_pass_destroy(gpu, &streamValidator);
        releaseVulkanGalleryTextureDevice(gpu);
        pl_gpu_finish(gpu);
        if (!assembledLease) pl_buf_destroy(gpu, &assembled);
        for (auto* buffer : {&stream, &metadata, &image, &status,
                             &streamUpload, &metadataUpload, &seed, &seedUpload,
                             &readback, &statusReadback, &lookup, &lookupUpload,
                             &valid, &validReadback})
            pl_buf_destroy(gpu, buffer);
        pl_pass_destroy(gpu, &pass);
        pl_pass_destroy(gpu, &residentDirect);
        pl_pass_destroy(gpu, &parser);
        pl_pass_destroy(gpu, &residentMeta);
        for(auto& [bits,pass]:wideMeta) pl_pass_destroy(gpu,&pass);
        for(auto& [bits,pass]:wideParsers) pl_pass_destroy(gpu,&pass);
        pl_pass_destroy(gpu, &scatter);
        pl_pass_destroy(gpu, &validate);
        pl_timer_destroy(gpu, &timer);
        pl_timer_destroy(gpu, &scatterTimer);
        pl_timer_destroy(gpu, &parserTimer);
    }
    bool reserveAssembled(size_t bytes) {
        if (assembled && assembled->params.size == bytes) return true;
        assembledLease.reset(); assembled = nullptr;
        assembledLease = jpegContext().acquireImage(bytes, assembled);
        return bool(assembledLease);
    }
    bool init(pl_gpu device, bool resident = false) {
        if (gpu && gpu != device) return false;
        const auto ready = [&] {
            return pass && residentDirect && scatter && validate &&
                (resident ? bool(residentMeta) : bool(parser && streamValidator));
        };
        if (ready()) return true;
        const bool firstInitialization = !gpu;
        gpu = device;
#ifdef __ANDROID__
        // The source shader handles mixed JPEG layouts; resident encoder output
        // uses a simpler known-component shader and benefits from a wider group.
        if (const auto vk = firstInitialization ? pl_vulkan_get(gpu) : nullptr;
            vk && vk->get_proc_addr) {
            const auto getProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceProperties"));
            if (getProperties) {
                VkPhysicalDeviceProperties properties{};
                getProperties(vk->phys_device, &properties);
                if (std::strstr(properties.deviceName, "Adreno")) {
                    sourceGroupSize = 16u;
                    encodedGroupSize = 32u;
                    preferredLookupBits.store(9u,std::memory_order_relaxed);
                }
            }
        }
#else
        (void)firstInitialization;
#endif
        pl_desc descriptors[] = {
            {"stream", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
            {"meta", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
            {"image", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_READWRITE},
            {"status", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_WRITEONLY},
            {"lookup", PL_DESC_BUF_STORAGE, 4, PL_DESC_ACCESS_READONLY},
        };
        pl_pass_params params{};
        params.type = PL_PASS_COMPUTE;
        std::string defaultSource(decodeShader);
        defaultSource.insert(defaultSource.find('\n') + 1, "#define JPEG_BIT_RESERVOIR\n");
        params.glsl_shader = defaultSource.c_str();
        params.descriptors = descriptors;
        params.num_descriptors = 5;
        params.push_constants_size = 2 * sizeof(uint32_t);
        if (!pass) pass = pl_pass_create(gpu, &params);
        std::string directSource = defaultSource;
        directSource.insert(directSource.find('\n') + 1u,
                            "#define JPEG_RESIDENT_DIRECT\n");
        params.glsl_shader = directSource.c_str();
        params.push_constants_size = 6 * sizeof(uint32_t);
        if (!residentDirect) residentDirect = pl_pass_create(gpu, &params);
        if (!resident) {
            pl_desc parserDescriptors[] = {
                {"stream", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                {"meta", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READWRITE},
                {"lookup", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_WRITEONLY},
                {"seed", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_READONLY},
            };
            const bool serialValidation = jpegNeedsPrescan() &&
                rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_SERIAL_VALIDATE");
            std::string parserSource(parseShader), validatorSource(validateStreamShader);
            if (serialValidation) {
                parserSource.insert(parserSource.find('\n') + 1u, "#define JPEG_SERIAL_VALIDATE\n");
                validatorSource.insert(validatorSource.find('\n') + 1u, "#define JPEG_SERIAL_VALIDATE\n");
            }
            params.glsl_shader = parserSource.c_str();
            params.descriptors = parserDescriptors;
            params.num_descriptors = 4;
            params.push_constants_size = sizeof(uint32_t);
            if (!parser) parser = pl_pass_create(gpu, &params);
            params.glsl_shader = validatorSource.c_str();
            params.num_descriptors = 2;
            if (!streamValidator) streamValidator = pl_pass_create(gpu, &params);
        }
        if (resident) {
            pl_desc residentMetaDescriptors[] = {
                {"lengths", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                {"table", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
                {"meta", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_WRITEONLY},
                {"lookup", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_WRITEONLY},
            };
            params.glsl_shader = residentMetaShader;
            params.descriptors = residentMetaDescriptors;
            params.num_descriptors = 4;
            params.push_constants_size = 8 * sizeof(uint32_t);
            if (!residentMeta) residentMeta = pl_pass_create(gpu, &params);
        }
        pl_desc scatterDescriptors[] = {
            {"tiles", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
            {"meta", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
            {"image", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_WRITEONLY},
        };
        params.glsl_shader = scatterShader;
        params.descriptors = scatterDescriptors;
        params.num_descriptors = 3;
        params.push_constants_size = 6 * sizeof(uint32_t);
        if (!scatter) scatter = pl_pass_create(gpu, &params);
        pl_desc validateDescriptors[] = {
            {"status", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
            {"result", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_WRITEONLY},
        };
        params.glsl_shader = validateShader;
        params.descriptors = validateDescriptors;
        params.num_descriptors = 2;
        params.push_constants_size = sizeof(uint32_t);
        if (!validate) validate = pl_pass_create(gpu, &params);
#if RAWPREAMP_DEBUG_INSTRUMENTATION
        if (!timer) timer = pl_timer_create(gpu);
        if (!scatterTimer) scatterTimer = pl_timer_create(gpu);
        if (!parserTimer) parserTimer = pl_timer_create(gpu);
#endif
        return ready();
    }
    pl_pass sourceMetadataPass(uint32_t bits) {
        auto &pass = wideParsers[bits];
        if (pass)
            return pass;
        std::string shader(parseShader);
        std::string defines =
            "#define JPEG_LOOKUP_BITS " + std::to_string(bits) + "u\n#define JPEG_SHARED_LOOKUP\n";
        if (jpegNeedsPrescan() && rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_SERIAL_VALIDATE"))
            defines += "#define JPEG_SERIAL_VALIDATE\n";
        shader.insert(shader.find('\n') + 1u, defines);
        pl_desc desc[] = {{"stream", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                          {"meta", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READWRITE},
                          {"lookup", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_WRITEONLY},
                          {"seed", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_READONLY}};
        pl_pass_params params{};
        params.type = PL_PASS_COMPUTE;
        params.glsl_shader = shader.c_str();
        params.descriptors = desc;
        params.num_descriptors = 4;
        params.push_constants_size = 8;
        pass = pl_pass_create(gpu, &params);
        return pass;
    }
    pl_pass metadataPass(uint32_t bits) {
        if (bits == 7u)
            return residentMeta;
        auto &pass = wideMeta[bits];
        if (pass)
            return pass;
        std::string shader(residentMetaShader);
        auto replace = [&](const std::string &from, const std::string &to) {
            size_t at = 0;
            while ((at = shader.find(from, at)) != std::string::npos) {
                shader.replace(at, from.size(), to);
                at += to.size();
            }
        };
        replace("128u", std::to_string(1u << bits) + "u");
        replace("length <= 7u", "length <= " + std::to_string(bits) + "u");
        replace("7u - length", std::to_string(bits) + "u - length");
        pl_desc desc[] = {{"lengths", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                          {"table", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
                          {"meta", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_WRITEONLY},
                          {"lookup", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_WRITEONLY}};
        pl_pass_params params{};
        params.type = PL_PASS_COMPUTE;
        params.glsl_shader = shader.c_str();
        params.descriptors = desc;
        params.num_descriptors = 4;
        params.push_constants_size = 32;
        pass = pl_pass_create(gpu, &params);
        return pass;
    }
    bool decodeImage(pl_buf input, uint32_t tileCount, bool directOutput,
                     uint32_t width, uint32_t tileWidth, uint32_t tileHeight, bool profile,
                     uint32_t variant = 0, uint32_t lookupStride = 0, uint32_t lookupBits = 7u) {
        const uint32_t defaultGroupSize = lookupStride ? sourceGroupSize : encodedGroupSize;
        if (!lookupStride) lookupStride = tileCount * (1u << lookupBits);
        pl_pass selected = directOutput ? residentDirect : pass;
        const char* reservoirOverride = rawpreamp::debugEnvironment("RAWPREAMP_JPEG_RESERVOIR");
        const bool reservoir = !reservoirOverride || std::strcmp(reservoirOverride, "0") != 0;
        const char* prefillSetting = rawpreamp::debugEnvironment("RAWPREAMP_JPEG_PREFILL_BITS");
        const bool prefill = reservoir && (!prefillSetting || std::strcmp(prefillSetting, "7") != 0);
        bool usingPrefill = false;
        const bool unroll = !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_COMPONENT_LOOPS");
        const bool diagonal = unroll && !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_DYNAMIC_PREDICTOR");
        const bool combined = !rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_SEPARATE_RESIDUAL");
        uint32_t groupSize = defaultGroupSize;
        if (const char* setting = rawpreamp::debugEnvironment("RAWPREAMP_JPEG_GROUP_SIZE")) {
            if (std::strcmp(setting, "8") == 0) groupSize = 8u;
            else if (std::strcmp(setting, "16") == 0) groupSize = 16u;
            else if (std::strcmp(setting, "32") == 0) groupSize = 32u;
        }
        uint32_t selectedGroupSize = 8u;
        { // Cache every explicit reader configuration, including 7-bit reservoir mode.
            const uint32_t key = variant | (directOutput ? 65536u : 0u) |
                (reservoir ? 131072u : 0u) | (prefill ? 262144u : 0u) |
                (unroll ? 524288u : 0u) | (diagonal ? 1048576u : 0u) |
                (combined ? 2097152u : 0u) | (groupSize << 22u) | ((lookupBits - 7u) << 28u);
            auto& cached = variants[key];
            if (!cached) {
                std::string source(decodeShader);
                source.insert(source.find('\n') + 1u, "#define JPEG_GROUP_SIZE " + std::to_string(groupSize) + "\n");
                source.insert(source.find('\n') + 1u, "#define JPEG_LOOKUP_BITS "+std::to_string(lookupBits)+"u\n");
                if (combined) source.insert(source.find('\n') + 1, "#define JPEG_COMBINED_RESIDUAL\n");
                if (prefill) source.insert(source.find('\n') + 1, "#define JPEG_PREFILL_BITS 24u\n");
                if (variant) source.insert(source.find('\n') + 1, "#define JPEG_COMPONENTS " +
                    std::to_string(variant & 255u) + "u\n");
                if (variant >> 8u) source.insert(source.find('\n') + 1,
                    "#define JPEG_PREDICTOR " + std::to_string(variant >> 8u) + "u\n");
                if (reservoir) source.insert(source.find('\n') + 1, "#define JPEG_BIT_RESERVOIR\n");
                if (directOutput) source.insert(source.find('\n') + 1, "#define JPEG_RESIDENT_DIRECT\n");
                if (unroll) source = specializeComponents(std::move(source), variant & 255u);
                if (diagonal) source = specializeDiagonal(std::move(source), variant & 255u);
                pl_desc descriptors[] = {
                    {"stream", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                    {"meta", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
                    {"image", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_READWRITE},
                    {"status", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_WRITEONLY},
                    {"lookup", PL_DESC_BUF_STORAGE, 4, PL_DESC_ACCESS_READONLY}};
                pl_pass_params params{};
                params.type = PL_PASS_COMPUTE;
                params.glsl_shader = source.c_str();
                params.descriptors = descriptors; params.num_descriptors = 5;
                params.push_constants_size = (directOutput ? 6u : 2u) * sizeof(uint32_t);
                cached = pl_pass_create(gpu, &params);
            }
            if (!cached && lookupBits != 7u) return false;
            if (cached) { selected = cached; usingPrefill = prefill; selectedGroupSize = groupSize; }
        }
        pl_desc_binding bindings[5]{};
        bindings[0].object = input;
        bindings[1].object = metadata;
        bindings[2].object = directOutput ? assembled : image;
        bindings[3].object = status;
        bindings[4].object = lookup;
        uint32_t constants[6] = {
            tileCount, width, tileWidth, tileHeight, tileWidth ? width / tileWidth : 0u, lookupStride};
        uint32_t linearConstants[2] = {tileCount, lookupStride};
        pl_pass_run_params run{};
        run.pass = selected;
        run.desc_bindings = bindings;
        run.push_constants = directOutput ? constants : linearConstants;
        if (profile) {
            run.timer = timer;
            std::fprintf(stderr, "GALLERY_PERF event=jpeg_gpu_dispatch prefill_bits=%u tiles=%u group_size=%u\n",
                         usingPrefill ? 24u : 7u, tileCount, selectedGroupSize);
        }
        run.compute_groups[0] = int((tileCount + selectedGroupSize - 1u) / selectedGroupSize);
        run.compute_groups[1] = run.compute_groups[2] = 1;
        pl_pass_run(gpu, &run);
        return true;
    }
    void scatterImage(uint32_t width, uint32_t height, uint32_t tileWidth,
                      uint32_t tileHeight, uint32_t channels, bool profile) {
        uint32_t constants[6] = {width, height, tileWidth, tileHeight, width / tileWidth, channels};
        pl_desc_binding bindings[3]{};
        bindings[0].object = image;
        bindings[1].object = metadata;
        bindings[2].object = assembled;
        pl_pass_run_params run{};
        run.pass = scatter;
        run.desc_bindings = bindings;
        run.push_constants = constants;
        if (profile) run.timer = scatterTimer;
        run.compute_groups[0] = int((uint64_t(width) * height * channels / 2u + 255u) / 256u);
        run.compute_groups[1] = run.compute_groups[2] = 1;
        pl_pass_run(gpu, &run);
    }
    void validateTiles(uint32_t tileCount) {
        pl_desc_binding bindings[2]{};
        bindings[0].object = status;
        bindings[1].object = valid;
        pl_pass_run_params run{};
        run.pass = validate;
        run.desc_bindings = bindings;
        run.push_constants = &tileCount;
        run.compute_groups[0] = run.compute_groups[1] = run.compute_groups[2] = 1;
        pl_pass_run(gpu, &run);
    }
    bool buffer(pl_buf& target, size_t bytes, bool readable = false,
                bool writable = false) {
        pl_buf_params params{};
        params.size = bytes;
        params.storable = !readable && !writable;
        params.host_readable = readable;
        params.host_writable = writable;
        params.memory_type = readable || writable ? PL_BUF_MEM_HOST : PL_BUF_MEM_DEVICE;
        return bytes && bytes <= gpu->limits.max_ssbo_size &&
               pl_buf_recreate(gpu, &target, &params);
    }
};

DecodeState& state() {
    // Destroy cached GPU resources before releasing their device owner, even
    // when lookup selection constructs this state before the first decode.
    jpegContext();
    static VulkanProcessState<DecodeState> value;
    return value.get();
}
struct ResidentStates {
    std::mutex mutex;
    std::unordered_map<pl_gpu, std::unique_ptr<DecodeState>> states;
};
ResidentStates& residentStates() {
    static VulkanProcessState<ResidentStates> value;
    return value.get();
}
} // namespace

bool jpegDecodeResidentTiledImageVulkan(
    const void* gpuHandle, const void* jpegHandle,
    const void* lengthsHandle, const void* tableHandle,
    uint32_t width, uint32_t height, uint32_t channels,
    uint32_t tileWidth, uint32_t tileHeight, uint32_t tileCount,
    uint32_t slotWords,
    const std::function<bool(const VulkanDecodedJpegImage&)>& consume,
    uint32_t bits) {
    auto gpu = static_cast<pl_gpu>(const_cast<void*>(gpuHandle));
    auto stream = static_cast<pl_buf>(const_cast<void*>(jpegHandle));
    auto lengths = static_cast<pl_buf>(const_cast<void*>(lengthsHandle));
    auto table = static_cast<pl_buf>(const_cast<void*>(tableHandle));
    const uint64_t tileSamples = uint64_t(tileWidth) * tileHeight * channels;
    const uint64_t samples = uint64_t(width) * height * channels;
    if (!gpu || !stream || !lengths || !table || !consume || bits < 8u || bits > 16u ||
        (channels != 1u && channels != 3u) || !tileWidth || !tileHeight ||
        width % tileWidth || height % tileHeight ||
        uint64_t(width / tileWidth) * (height / tileHeight) != tileCount ||
        !tileCount || !slotWords ||
        uint64_t(slotWords) * tileCount * 4u > stream->params.size ||
        stream->params.size > UINT32_MAX ||
        tileSamples > kMaxSerialTileSamples ||
        (tileSamples & 1u) || (uint64_t(tileWidth) * channels & 1u) ||
        samples > UINT32_MAX ||
        tileCount > uint32_t(gpu->limits.max_dispatch[0]) * 8ull ||
        (samples / 2u + 255u) / 256u >
            uint32_t(gpu->limits.max_dispatch[0])) return false;
    auto& owner = residentStates();
    std::lock_guard lock(owner.mutex);
    auto& slot = owner.states[gpu];
    if (!slot) slot = std::make_unique<DecodeState>();
    DecodeState& s = *slot;
    const bool profile = rawpreamp::debugEnvironment("RAWPREAMP_JPEG_DECODE_PROFILE") != nullptr;
    const bool directOutput =
        rawpreamp::debugEnvironment("RAWPREAMP_JPEG_RESIDENT_SCATTER") == nullptr;
    const size_t metadataBytes = size_t(tileCount) * kMetaWords * 4u;
    if(!s.init(gpu, true)) return false;
    const uint32_t lookupBits=jpegLookupBits(s.preferredLookupBits.load(std::memory_order_relaxed));
    const size_t lookupBytes = (2u << lookupBits) * 4u;
    if (!s.buffer(s.metadata, metadataBytes) ||
        !s.buffer(s.lookup, lookupBytes) ||
        (!directOutput && !s.buffer(s.image, size_t(samples) * 2u)) ||
        !s.buffer(s.status, size_t(tileCount) * 4u) ||
        !s.buffer(s.assembled, size_t(samples) * 2u) ||
        !s.buffer(s.valid, 4u)) return false;
    uint32_t metaConstants[8] = {
        tileCount, tileWidth, tileHeight, channels,
        uint32_t(stream->params.size), uint32_t(tileSamples * 3u), slotWords, bits};
    pl_desc_binding metaBindings[4]{};
    metaBindings[0].object = lengths;
    metaBindings[1].object = table;
    metaBindings[2].object = s.metadata;
    metaBindings[3].object = s.lookup;
    pl_pass_run_params metaRun{};
    metaRun.pass = s.metadataPass(lookupBits);
    if(!metaRun.pass) return false;
    metaRun.desc_bindings = metaBindings;
    metaRun.push_constants = metaConstants;
    if (profile) metaRun.timer = s.parserTimer;
    metaRun.compute_groups[0] = int((tileCount + 63u) / 64u);
    metaRun.compute_groups[1] = metaRun.compute_groups[2] = 1;
    pl_pass_run(gpu, &metaRun);
    const uint32_t variant = rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_COMPONENT_LOOPS") ? 0u : channels;
    if(!s.decodeImage(stream, tileCount, directOutput, width, tileWidth, tileHeight, profile, variant, 0u, lookupBits)) return false;
    if (!directOutput) s.scatterImage(width, height, tileWidth, tileHeight, channels, profile);
    s.validateTiles(tileCount);
    const bool consumed = consume({gpu, s.assembled, s.valid, width, height,
                                   channels, bits, width * channels / 2u});
    if (profile) {
        pl_gpu_finish(gpu);
        std::fprintf(stderr,
            "GPU resident JPEG decode: channels=%u metadata=%.3f entropy=%.3f scatter=%.3f ms\n",
            channels, double(pl_timer_query(gpu, s.parserTimer)) / 1000000.0,
            double(pl_timer_query(gpu, s.timer)) / 1000000.0,
            directOutput ? 0.0 :
                double(pl_timer_query(gpu, s.scatterTimer)) / 1000000.0);
    }
    return consumed;
}

void releaseVulkanJpegDecodeDevice(const void* gpuHandle) {
    auto& owner = residentStates();
    std::lock_guard lock(owner.mutex);
    owner.states.erase(static_cast<pl_gpu>(const_cast<void*>(gpuHandle)));
}

static bool decodeTiles(const std::vector<VulkanJpegTile>& tiles,
                        std::vector<std::vector<uint16_t>>* pixels,
                        uint32_t width, uint32_t height,
                        uint32_t tileWidth, uint32_t tileHeight,
                        const std::function<bool(const VulkanDecodedJpegImage&)>* consume,
                        const VulkanJpegContiguousSource* contiguous,
                        bool readValidation = false, uint32_t lookupOverride = 0u,
                        bool preferThroughput = false) {
    const bool profile = rawpreamp::debugEnvironment("RAWPREAMP_JPEG_DECODE_PROFILE") != nullptr;
    // Retained previews prioritize throughput; streaming DNG consumers avoid
    // the extra host table-key setup on their synchronous request path.
    const uint32_t preferred=preferThroughput && tiles.size()>=128u
        ?state().preferredLookupBits.load(std::memory_order_relaxed):7u;
    const uint32_t lookupBits=lookupOverride?lookupOverride:jpegLookupBits(preferred),lookupWidth=1u<<lookupBits;
    const bool wide=lookupBits!=7u;
    const bool gpuParse = wide || contiguous || rawpreamp::debugEnvironment("RAWPREAMP_GPU_JPEG_GPU_PARSE") != nullptr;
    const auto started = rawpreamp::diagnosticNow();
    if (tiles.empty() || tiles.size() > UINT32_MAX ||
        rawpreamp::debugEnvironment("RAWPREAMP_GPU_JPEG_DECODE_DISABLE")) return false;
    if (contiguous && (!contiguous->bytes || !contiguous->size ||
                       contiguous->size > UINT32_MAX)) return false;
    std::vector<ParsedTile> parsed;
    if (!gpuParse) parsed.resize(tiles.size());
    std::vector<uint32_t> metadata;
    std::vector<uint32_t> seeds;
    const uint32_t seedStride=wide?8u:7u;
    std::unordered_map<std::string,uint32_t> sharedKeys;
    if(wide && gpuParse) sharedKeys.reserve(std::min<size_t>(tiles.size(),4096u));
    if (gpuParse) seeds.resize(tiles.size() * seedStride);
    else metadata.resize(tiles.size() * kMetaWords);
    std::string sharedKey;
    if(wide && gpuParse) sharedKey.reserve(68u);
    uint64_t scanBytes = 0, sampleCount = 0;
    uint32_t variant = 0;
    for (size_t i = 0; i < tiles.size(); ++i) {
        if (consume && (tiles[i].width != tileWidth || tiles[i].height != tileHeight ||
            tiles[i].bits != tiles.front().bits || tiles[i].channels != tiles.front().channels))
            return false;
        if (gpuParse) {
            if (!tiles[i].bytes || tiles[i].size < 16u ||
                tiles[i].size > UINT32_MAX || !tiles[i].width || !tiles[i].height ||
                tiles[i].width > 65535u || tiles[i].height > 65535u ||
                tiles[i].bits < 8u || tiles[i].bits > 16u ||
                (tiles[i].channels != 1u && tiles[i].channels != 3u)) return false;
            uint32_t* seed = seeds.data() + i * seedStride;
            seed[2] = tiles[i].width;
            seed[3] = tiles[i].height;
            seed[4] = tiles[i].bits;
            seed[6] = tiles[i].channels;
            if(wide) {
                if(!jpegTableKey(tiles[i],sharedKey)) return false;
                const auto [entry,owner]=sharedKeys.try_emplace(sharedKey,uint32_t(sharedKeys.size()));
                seed[7]=entry->second|(owner?0x80000000u:0u);
            }
        } else if (!parseTile(tiles[i], parsed[i])) return false;
        const uint64_t n = uint64_t(tiles[i].width) * tiles[i].height *
                           tiles[i].channels;
        const size_t bytes = gpuParse ? tiles[i].size : parsed[i].scanLength;
        if ((n > kMaxSerialTileSamples &&
             !rawpreamp::debugEnvironment("RAWPREAMP_GPU_JPEG_ALLOW_LARGE_SCAN")) ||
            scanBytes + bytes > UINT32_MAX ||
            sampleCount + ((n + 1u) & ~1ull) > UINT32_MAX) return false;
        if (contiguous) {
            const uintptr_t base = reinterpret_cast<uintptr_t>(contiguous->bytes);
            const uintptr_t pointer = reinterpret_cast<uintptr_t>(tiles[i].bytes);
            if (pointer < base || pointer - base > contiguous->size ||
                tiles[i].size > contiguous->size - (pointer - base)) return false;
            if (gpuParse) seeds[i * seedStride] = uint32_t(pointer - base);
            else parsed[i].meta[0] = uint32_t(pointer - base + parsed[i].scanOffset);
        } else {
            if (gpuParse) seeds[i * seedStride] = uint32_t(scanBytes);
            else parsed[i].meta[0] = uint32_t(scanBytes);
        }
        if (gpuParse) {
            seeds[i * seedStride + 1u] = uint32_t(bytes);
            seeds[i * seedStride + 5u] = uint32_t(sampleCount);
        } else {
            parsed[i].meta[1] = uint32_t(bytes);
            parsed[i].meta[6] = uint32_t(sampleCount);
        }
        // Small tiles already provide ample parallel work; avoid inspecting
        // thousands of JPEG headers on the CPU merely to specialize them.
        const uint32_t currentVariant = n > 4096u ? headerVariant(tiles[i]) : 0u;
        if (i == 0) variant = currentVariant;
        else if (variant != currentVariant) variant = 0;
        scanBytes += bytes;
        sampleCount += (n + 1u) & ~1ull;
    }
    if (contiguous) scanBytes = contiguous->size;
    std::unordered_map<uint64_t, std::vector<uint32_t>> lookupBuckets;
    std::vector<std::array<uint32_t, kLookupSize>> uniqueLookups;
    if (!gpuParse) for (auto& tile : parsed) {
        uint64_t hash = 1469598103934665603ull;
        for (uint32_t value : tile.lookup) {
            hash ^= value;
            hash *= 1099511628211ull;
        }
        auto& bucket = lookupBuckets[hash];
        auto match = std::find_if(bucket.begin(), bucket.end(), [&](uint32_t index) {
            return uniqueLookups[index] == tile.lookup;
        });
        if (match == bucket.end()) {
            tile.meta[7] = uint32_t(uniqueLookups.size());
            bucket.push_back(tile.meta[7]);
            uniqueLookups.push_back(tile.lookup);
        } else {
            tile.meta[7] = *match;
        }
    }
    const size_t uniqueCount=wide?sharedKeys.size():uniqueLookups.size();
    const auto parsedAt = rawpreamp::diagnosticNow();
    auto& context = jpegContext();
    std::lock_guard lock(context.mutex);
    if (!context.init()) return false;
    auto gpu = context.vulkan->gpu;
    auto& s = state();
    if (!s.init(gpu) || tiles.size() > uint32_t(gpu->limits.max_dispatch[0]) * 8ull)
        return false;
    const size_t streamSize = size_t((scanBytes + 3u) & ~3ull);
    const size_t imageSize = size_t(sampleCount) * 2u;
    // Reuse the encoder round-trip decoder's direct CFA output layout.
    const bool directOutput = consume && tiles.front().channels == 1u &&
        !rawpreamp::debugEnvironment("RAWPREAMP_JPEG_RESIDENT_SCATTER");
    const size_t statusSize = tiles.size() * 4u;
    const size_t metaSize = tiles.size() * kMetaWords * 4u;
    const size_t seedSize = seeds.size() * 4u;
    const size_t lookupSize = (gpuParse && !wide ? tiles.size() : uniqueCount) *
                              (lookupWidth * 2u) * 4u;
    const char* lookupSetting=rawpreamp::debugEnvironment("RAWPREAMP_JPEG_LOOKUP_BITS");
    const bool automaticLookup = !lookupOverride &&
        (!lookupSetting || std::strcmp(lookupSetting,"auto9")==0);
    if(wide && (lookupSize>std::min<size_t>(gpu->limits.max_ssbo_size,64u*1024u*1024u) ||
        (automaticLookup && sharedKeys.size()*2u>tiles.size()))) {
        if(profile) spdlog::info("GALLERY_PERF event=jpeg_lookup_fallback bits={} tables={} tiles={} bytes={}",
            lookupBits,sharedKeys.size(),tiles.size(),lookupSize);
        return decodeTiles(tiles,pixels,width,height,tileWidth,tileHeight,consume,contiguous,readValidation,7u,preferThroughput);
    }
    if (!s.buffer(s.stream, streamSize) || !s.buffer(s.metadata, metaSize) ||
        !s.buffer(s.streamUpload, streamSize, false, true) ||
        (gpuParse ? (!s.buffer(s.seed, seedSize) ||
                     !s.buffer(s.seedUpload, seedSize, false, true))
                  : !s.buffer(s.metadataUpload, metaSize, false, true)) ||
        !s.buffer(s.lookup, lookupSize) ||
        (!gpuParse && !s.buffer(s.lookupUpload, lookupSize, false, true)) ||
        (!directOutput && !s.buffer(s.image, imageSize)) || !s.buffer(s.status, statusSize) ||
        (!consume && !s.buffer(s.readback, imageSize, true)) ||
        (consume ? (!s.buffer(s.valid, 4u) ||
                    (readValidation && !s.buffer(s.validReadback, 4u, true)))
                 : !s.buffer(s.statusReadback, statusSize, true))) return false;
    if (consume) {
        if (!width || !height || !tileWidth || !tileHeight ||
            (uint64_t(width) * tiles.front().channels & 1u) ||
            (uint64_t(tileWidth) * tiles.front().channels & 1u) ||
            width % tileWidth || height % tileHeight ||
            uint64_t(width / tileWidth) * (height / tileHeight) != tiles.size() ||
            uint64_t(width) * height * tiles.front().channels > UINT32_MAX ||
            (uint64_t(width) * height * tiles.front().channels / 2u + 255u) / 256u >
                uint32_t(gpu->limits.max_dispatch[0]) ||
            !s.reserveAssembled(uint64_t(width) * height * tiles.front().channels * 2u))
            return false;
    }
    std::vector<uint8_t> stream;
    if (!contiguous) stream.resize(streamSize, 0);
    std::vector<uint32_t> lookup;
    if (!gpuParse) lookup.resize(uniqueCount * lookupWidth * 2u);
    // Contiguous GPU-parsed input needs neither stream packing nor metadata copies.
    if (!contiguous || !gpuParse) for (size_t i = 0; i < tiles.size(); ++i) {
        if (!contiguous) {
            const size_t offset = gpuParse ? seeds[i * seedStride]
                                           : parsed[i].meta[0];
            std::memcpy(stream.data() + offset,
                        tiles[i].bytes + (gpuParse ? 0u : parsed[i].scanOffset),
                        gpuParse ? tiles[i].size : parsed[i].scanLength);
        }
        if (!gpuParse)
            std::copy(parsed[i].meta.begin(), parsed[i].meta.end(),
                      metadata.begin() + i * kMetaWords);
    }
    if (!gpuParse) for (size_t i=0;i<uniqueCount;++i) {
        const auto* values=uniqueLookups[i].data();
        std::copy_n(values,lookupWidth,lookup.begin()+i*lookupWidth);
        std::copy_n(values+lookupWidth,lookupWidth,lookup.begin()+(uniqueCount+i)*lookupWidth);
    }
    if (contiguous) {
        const size_t complete = contiguous->size & ~size_t{3};
        if (complete)
            pl_buf_write(gpu, s.streamUpload, 0, contiguous->bytes, complete);
        if (streamSize > complete) {
            const uint32_t zero = 0;
            uint32_t last = zero;
            std::memcpy(&last, contiguous->bytes + complete,
                        contiguous->size - complete);
            pl_buf_write(gpu, s.streamUpload, complete, &last, 4u);
        }
    } else {
        pl_buf_write(gpu, s.streamUpload, 0, stream.data(), stream.size());
    }
    if (gpuParse)
        pl_buf_write(gpu, s.seedUpload, 0, seeds.data(), seedSize);
    else
        pl_buf_write(gpu, s.metadataUpload, 0, metadata.data(), metaSize);
    if (!gpuParse)
        pl_buf_write(gpu, s.lookupUpload, 0, lookup.data(), lookupSize);
    pl_buf_copy(gpu, s.stream, 0, s.streamUpload, 0, streamSize);
    if (gpuParse)
        pl_buf_copy(gpu, s.seed, 0, s.seedUpload, 0, seedSize);
    else
        pl_buf_copy(gpu, s.metadata, 0, s.metadataUpload, 0, metaSize);
    if (gpuParse) {
        uint32_t tileCount = uint32_t(tiles.size());
        pl_desc_binding parserBindings[4]{};
        parserBindings[0].object = s.stream;
        parserBindings[1].object = s.metadata;
        parserBindings[2].object = s.lookup;
        parserBindings[3].object = s.seed;
        pl_pass_run_params parserRun{};
        parserRun.pass = wide?s.sourceMetadataPass(lookupBits):s.parser;
        if(!parserRun.pass) {
            if(wide) return decodeTiles(tiles,pixels,width,height,tileWidth,tileHeight,consume,contiguous,readValidation,7u,preferThroughput);
            return false;
        }
        parserRun.desc_bindings = parserBindings;
        uint32_t parserConstants[2]={tileCount,uint32_t(uniqueCount)};
        parserRun.push_constants = parserConstants;
        if (profile) parserRun.timer = s.parserTimer;
        parserRun.compute_groups[0] = int((tileCount + 63u) / 64u);
        parserRun.compute_groups[1] = parserRun.compute_groups[2] = 1;
        pl_pass_run(gpu, &parserRun);
        if (jpegNeedsPrescan() &&
            (!rawpreamp::debugEnvironmentIsSet("RAWPREAMP_JPEG_SERIAL_VALIDATE") ||
             std::any_of(tiles.begin(), tiles.end(), [](const auto& tile) {
                 return uint64_t(tile.width) * tile.height * tile.channels > 4096u;
             }))) {
            parserRun.pass = s.streamValidator;
            parserRun.timer = nullptr;
            parserRun.compute_groups[0] = std::min(tileCount, uint32_t(gpu->limits.max_dispatch[0]));
            parserRun.compute_groups[1] = (tileCount + parserRun.compute_groups[0] - 1u) / parserRun.compute_groups[0];
            pl_pass_run(gpu, &parserRun);
        }
    } else {
        pl_buf_copy(gpu, s.lookup, 0, s.lookupUpload, 0, lookupSize);
    }
    const auto uploadedAt = rawpreamp::diagnosticNow();
    uint32_t tileCount = uint32_t(tiles.size());
    if(!s.decodeImage(s.stream, tileCount, directOutput, width, tileWidth, tileHeight, profile, variant,
        uint32_t(gpuParse && !wide ? tiles.size() : uniqueCount) * lookupWidth, lookupBits)) {
        if(wide) return decodeTiles(tiles,pixels,width,height,tileWidth,tileHeight,consume,contiguous,readValidation,7u,preferThroughput);
        return false;
    }
    if (!consume)
        pl_buf_copy(gpu, s.readback, 0, s.image, 0, imageSize);
    if (!consume) {
        pl_buf_copy(gpu, s.statusReadback, 0, s.status, 0, statusSize);
    }
    if (consume) s.validateTiles(tileCount);
    const auto dispatchedAt = rawpreamp::diagnosticNow();
    if (consume && !directOutput)
        s.scatterImage(width, height, tileWidth, tileHeight, tiles.front().channels, profile);
    if (consume) {
        if (readValidation) {
            pl_buf_copy(gpu, s.validReadback, 0, s.valid, 0, 4u);
            uint32_t valid = 0;
            if (!pl_buf_read(gpu, s.validReadback, 0, &valid, 4u) || valid != 1u)
                return false;
        }
        const VulkanDecodedJpegImage image{
            gpu, s.assembled, s.valid, width, height, tiles.front().channels,
            tiles.front().bits, width * tiles.front().channels / 2u};
        const auto preparedForConsumer = rawpreamp::diagnosticNow();
        const bool consumed = (*consume)(image);
        if (profile) {
            std::array<uint32_t, 8> predictorCounts{};
            const size_t stride = std::max<size_t>(1u, tiles.size() / 64u);
            for (size_t i = 0; i < tiles.size(); i += stride) {
                ParsedTile sample;
                if (parseTile(tiles[i], sample) && sample.meta[5] <= 7u)
                    ++predictorCounts[sample.meta[5]];
            }
            const auto ms = [](auto from, auto to) {
                return std::chrono::duration<double, std::milli>(to - from).count();
            };
            spdlog::info(
                "GALLERY_PERF event=jpeg_gpu_stages channels={} tiles={} unique_tables={} parse_ms={:.3f} upload_ms={:.3f} wait_ms={:.3f} consume_ms={:.3f} parser_ms={:.3f} decode_ms={:.3f} scatter_ms={:.3f}",
                tiles.front().channels, tiles.size(), uniqueCount,
                ms(started, parsedAt), ms(parsedAt, uploadedAt),
                ms(uploadedAt, preparedForConsumer),
                ms(preparedForConsumer, rawpreamp::diagnosticNow()),
                double(pl_timer_query(gpu, s.parserTimer)) / 1000000.0,
                double(pl_timer_query(gpu, s.timer)) / 1000000.0,
                double(pl_timer_query(gpu, s.scatterTimer)) / 1000000.0);
            std::fprintf(stderr,
                "GPU JPEG sampled predictor modes: p1=%u p2=%u p3=%u p4=%u p5=%u p6=%u p7=%u\n",
                predictorCounts[1], predictorCounts[2], predictorCounts[3],
                predictorCounts[4], predictorCounts[5], predictorCounts[6],
                predictorCounts[7]);
        }
        return consumed;
    }
    std::vector<uint32_t> statuses(tiles.size());
    if (!pl_buf_read(gpu, s.statusReadback, 0, statuses.data(), statusSize) ||
        !std::all_of(statuses.begin(), statuses.end(),
                     [](uint32_t value) { return value == 1u; })) return false;
    std::vector<uint16_t> decoded(sampleCount);
    if (!pl_buf_read(gpu, s.readback, 0, decoded.data(), imageSize)) return false;
    const auto readAt = rawpreamp::diagnosticNow();
    std::vector<std::vector<uint16_t>> result(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        const size_t first = gpuParse ? seeds[i * seedStride + 5u]
                                      : metadata[i * kMetaWords + 6u];
        const size_t count = size_t(tiles[i].width) * tiles[i].height *
                             tiles[i].channels;
        result[i].assign(decoded.begin() + first, decoded.begin() + first + count);
    }
    *pixels = std::move(result);
    if (profile) {
        const double kernelMs = double(pl_timer_query(gpu, s.timer)) / 1000000.0;
        const auto ms = [](auto from, auto to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        std::fprintf(stderr, "GPU JPEG decode stages: parse=%.3f prepare=%.3f submit=%.3f wait/read=%.3f copy=%.3f ms\n",
                     ms(started, parsedAt), ms(parsedAt, uploadedAt),
                     ms(uploadedAt, dispatchedAt), ms(dispatchedAt, readAt),
                     ms(readAt, rawpreamp::diagnosticNow()));
        if (gpuParse)
            std::fprintf(stderr,
                         "GPU JPEG first tile header parsed on GPU; stream_bytes=%u\n",
                         seeds[1]);
        else
            std::fprintf(stderr,
                         "GPU JPEG first tile predictor=%u scan_bytes=%u\n",
                         metadata[5], metadata[1]);
        std::fprintf(stderr, "GPU JPEG unique Huffman lookups=%zu/%zu\n",
                     uniqueCount, tiles.size());
        std::fprintf(stderr, "GPU JPEG decode kernel %.3f ms\n", kernelMs);
    }
    return true;
}

bool jpegDecodeTilesVulkan(const std::vector<VulkanJpegTile>& tiles,
                           std::vector<std::vector<uint16_t>>& pixels,
                           const VulkanJpegContiguousSource* contiguous) {
    return decodeTiles(tiles, &pixels, 0, 0, 0, 0, nullptr, contiguous);
}

bool jpegDecodeTiledImageVulkan(
    const std::vector<VulkanJpegTile>& tiles, uint32_t width, uint32_t height,
    uint32_t tileWidth, uint32_t tileHeight,
    const std::function<bool(const VulkanDecodedJpegImage&)>& consume,
    const VulkanJpegContiguousSource* contiguous) {
    return consume && decodeTiles(tiles, nullptr, width, height,
                                  tileWidth, tileHeight, &consume, contiguous);
}

bool jpegDecodeTiledImageVulkanRetained(
    const std::vector<VulkanJpegTile>& tiles, uint32_t width, uint32_t height,
    uint32_t tileWidth, uint32_t tileHeight, VulkanDecodedJpegImage& image,
    const VulkanJpegContiguousSource* contiguous) {
    VulkanDecodedJpegImage retained;
    const std::function<bool(const VulkanDecodedJpegImage&)> consume =
        [&](const VulkanDecodedJpegImage& decoded) {
            auto& s = state();
            retained = decoded;
            retained.lease = std::move(s.assembledLease);
            s.assembled = nullptr;
            // This status allocation is reused by the next decode. The output
            // was validated before the callback, so consumers do not need it.
            retained.validationBuffer = nullptr;
            return true;
        };
    if (!decodeTiles(tiles, nullptr, width, height, tileWidth, tileHeight,
                     &consume, contiguous, true, 0u, true)) return false;
    image = std::move(retained);
    return true;
}

bool downscaleVulkanDecodedJpegImage(const VulkanDecodedJpegImage& source,
                                     uint32_t scale, bool cfa, VulkanDecodedJpegImage& result) {
    return downscaleVulkanResidentImage(source, scale, cfa, result);
}
bool downloadVulkanDecodedJpegImage(const VulkanDecodedJpegImage& source,
                                    uint16_t* pixels, size_t samples) {
    return downloadVulkanResidentImage(source, pixels, samples);
}

bool jpegDecodeTiledImageVulkanValidated(
    const std::vector<VulkanJpegTile>& tiles, uint32_t width, uint32_t height,
    uint32_t tileWidth, uint32_t tileHeight,
    const VulkanJpegContiguousSource* contiguous) {
    const std::function<bool(const VulkanDecodedJpegImage&)> consume =
        [](const VulkanDecodedJpegImage&) { return true; };
    return decodeTiles(tiles, nullptr, width, height, tileWidth, tileHeight,
                       &consume, contiguous, true);
}

} // namespace rawpreamp
#else
namespace rawpreamp {
bool jpegDecodeTilesVulkan(const std::vector<VulkanJpegTile>&,
                           std::vector<std::vector<uint16_t>>&,
                           const VulkanJpegContiguousSource*) { return false; }
bool jpegDecodeTiledImageVulkan(
    const std::vector<VulkanJpegTile>&, uint32_t, uint32_t, uint32_t, uint32_t,
    const std::function<bool(const VulkanDecodedJpegImage&)>&,
    const VulkanJpegContiguousSource*) { return false; }
bool jpegDecodeTiledImageVulkanRetained(
    const std::vector<VulkanJpegTile>&, uint32_t, uint32_t, uint32_t, uint32_t,
    VulkanDecodedJpegImage&, const VulkanJpegContiguousSource*) { return false; }
bool downscaleVulkanDecodedJpegImage(const VulkanDecodedJpegImage&, uint32_t, bool,
                                     VulkanDecodedJpegImage&) { return false; }
bool downloadVulkanDecodedJpegImage(const VulkanDecodedJpegImage&, uint16_t*, size_t) {
    return false;
}
bool jpegDecodeTiledImageVulkanValidated(
    const std::vector<VulkanJpegTile>&, uint32_t, uint32_t, uint32_t, uint32_t,
    const VulkanJpegContiguousSource*) { return false; }
bool jpegDecodeResidentTiledImageVulkan(
    const void*, const void*, const void*, const void*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    const std::function<bool(const VulkanDecodedJpegImage&)>&, uint32_t) { return false; }
void releaseVulkanJpegDecodeDevice(const void*) {}
}
#endif

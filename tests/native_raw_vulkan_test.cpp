#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include "VulkanJpegContext.h"
#include "VulkanUploadCache.h"
#endif
#include "VulkanRawDecode.h"
#include "DNGDecoder.h"
#include "CameraMetadata.h"
#include "CameraFrameMetadata.h"
#include "Utils.h"
#include "VulkanJpegDecode.h"
#include "VulkanGainBake.h"
#include "VulkanGalleryPreview.h"
#include "GalleryColor.h"
#include <motioncam/Mc24.hpp>
#include <motioncam/Decoder.hpp>
#include "../deps/motioncam-decoder/tests/DecoderMc24Vectors.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <future>
#include <thread>

void testResidentRawTable(const std::vector<uint16_t>&, const rawpreamp::VulkanDecodedJpegImage&,
                          const std::vector<uint16_t>&);
namespace {
void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
void u16(std::vector<uint8_t>& b, size_t at, uint32_t v) {
    b[at] = uint8_t(v);
    b[at + 1] = uint8_t(v >> 8);
}
void u32(std::vector<uint8_t>& b, size_t at, uint32_t v) {
    u16(b, at, v);
    u16(b, at + 2, v >> 16);
}
std::vector<uint8_t> strip(uint32_t width, uint32_t rows, uint32_t packing, uint32_t seed) {
    const uint32_t count = ((width + 31) / 32) * rows;
    std::vector<uint8_t> data((count + 1) / 2);
    for (uint32_t block = 0; block < count; ++block) {
        const uint32_t bits = (block + seed) % 16;
        data[block / 2] |= uint8_t(bits << ((block % 2) * 4));
        const size_t offset = data.size();
        data.resize(offset + bits * 4);
        std::array<uint64_t, 8> lanes{};
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t value = (block * 103 + i * 31 + seed * 13) & ((1u << bits) - 1u);
            if (packing)
                lanes[i % 8] |= uint64_t(value) << ((i / 8) * bits);
            else
                for (uint32_t bit = 0; bit < bits; ++bit)
                    data[offset + (i * bits + bit) / 8] |=
                        uint8_t(((value >> bit) & 1u) << ((i * bits + bit) % 8));
        }
        if (!packing)
            continue;
        for (uint32_t chunk = 0; chunk < bits / 4; ++chunk)
            for (uint32_t lane = 0; lane < 8; ++lane)
                u16(data, offset + chunk * 16 + lane * 2, uint32_t(lanes[lane] >> (chunk * 16)));
        uint32_t base = (bits / 4) * 16, shift = base;
        if (bits & 2) {
            for (uint32_t lane = 0; lane < 8; ++lane)
                data[offset + base + lane] = uint8_t(lanes[lane] >> shift);
            base += 8;
            shift += 8;
        }
        if (bits & 1)
            for (uint32_t lane = 0; lane < 8; ++lane)
                data[offset + base + lane / 2] |=
                    uint8_t(((lanes[lane] >> shift) & 15u) << ((lane % 2) * 4));
    }
    return data;
}
void testMc24() {
    constexpr uint32_t width = 70;
    for (uint32_t rows : {1u, 3u, 4u, 7u, 15u, 16u, 17u}) {
        const uint32_t height = rows * 2u + 2u, strips = (height + rows - 1u) / rows;
        const uint32_t head = 96 + strips * 8;
        for (uint32_t predictor : {0u, 1u})
            for (uint32_t packing : {0u, 1u, 3u})
                for (uint32_t threshold : {0u, 65u, 65535u}) {
                    std::vector<uint8_t> record(head);
                    std::memcpy(record.data(), "MCRF", 4);
                    u32(record, 8, head);
                    u16(record, 30, 96);
                    u16(record, 44, width);
                    u16(record, 46, height);
                    record[56] = uint8_t(predictor);
                    record[57] = uint8_t(packing);
                    u16(record, 58, threshold);
                    u16(record, 60, strips);
                    u16(record, 62, rows);
                    record[92] = 1;
                    record[93] = 0x10;
                    u16(record, 94, 1);
                    std::vector<uint16_t> expected(width * height);
                    for (uint32_t i = 0; i < strips; ++i) {
                        const uint32_t effective = packing == 3 ? (i == 0 ? 0 : 1) : packing;
                        const uint32_t count = std::min(rows, height - i * rows);
                        const auto data = strip(width, count, effective, i * 12);
                        u32(record, 96 + i * 4, uint32_t(data.size()));
                        record.insert(record.end(), data.begin(), data.end());
                        motioncam::mc24::decodeStrip(data.data(), data.size(),
                                                     expected.data() + i * rows * width, width, count,
                                                     0, effective, predictor, threshold);
                    }
                    u32(record, 4, uint32_t(record.size()));
                    rawpreamp::VulkanDecodedJpegImage image;
                    check(
                        rawpreamp::mc24DecodeVulkan(record.data(), record.size(), width, height, image),
                        "MC24 compute fixture decode failed");
                    std::vector<uint16_t> actual(expected.size());
                    check(rawpreamp::validateVulkanRawFrame(image), "MC24 valid fixture rejected");
                    check(rawpreamp::downloadVulkanDecodedJpegImage(image, actual.data(),
                                                                    actual.size()) &&
                              actual == expected,
                          "MC24 packing/predictor differs from CPU");
                    rawpreamp::VulkanDecodedJpegImage invalid;
                    check(!rawpreamp::mc24DecodeVulkan(record.data(), record.size() - 1, width, height,
                                                       invalid),
                          "Truncated MC24 accepted");
                    if (predictor == 1 && packing == 3 && threshold == 65) {
                        check(rawpreamp::mc24DecodeVulkan(record.data(), record.size(), width, height,
                                                          invalid, 0),
                              "MC24 range-validation dispatch failed");
                        check(!rawpreamp::validateVulkanRawFrame(invalid),
                              "Invalid MC24 quantized codes accepted");
                        const rawpreamp::VulkanGainBakeBufferSource source{
                            invalid.gpu, invalid.buffer, invalid.deviceLease, invalid.validationBuffer};
                        const std::vector<float> identity{1.0f};
                        rawpreamp::VulkanGainBakeParams params;
                        params.width = width;
                        params.height = height;
                        params.channels = 1;
                        params.scale.fill(1.0f);
                        params.bufferSource = &source;
                        params.cfaMapCount = 1;
                        params.cfaMaps[0] = {1, 1, 1, 0, 0, width, height, 1, 1, 0, 0, 1, 1, &identity};
                        std::vector<uint8_t> output(expected.size() * 2, 0xde);
                        check(!rawpreamp::bakeCfaGainVulkan(nullptr, params, output.data(),
                                                            output.size()) &&
                                  std::all_of(output.begin(), output.end(),
                                              [](uint8_t b) { return b == 0xde; }),
                              "Invalid MC24 image reached output readback");
                    }
                }
    }
}
void testQuantizedMc24(bool processing) {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("rawpreamp-mc24-compute-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".mcpro");
    {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(mc24_test::container.data()),
                     mc24_test::container.size());
        check(bool(output), "MC24 fixture write failed");
    }
    {
        motioncam::Decoder decoder(path.string());
        std::vector<uint8_t> bytes, cpu;
        nlohmann::json metadata;
        decoder.loadEncodedFrame(decoder.getFrames()[1], bytes, metadata);
        decoder.loadFrame(decoder.getFrames()[1], cpu, metadata);
        const auto table = metadata.at("linearizationTable").get<std::vector<uint16_t>>();
        rawpreamp::VulkanDecodedJpegImage image;
        check(rawpreamp::mc24DecodeVulkan(bytes.data(), bytes.size(), 4, 4, image,
                                          uint32_t(table.size() - 1)),
              "Quantized MC24 compute decode failed");
        check(rawpreamp::validateVulkanRawFrame(image), "Quantized MC24 codes rejected");
        std::vector<uint16_t> codes(16);
        check(rawpreamp::downloadVulkanDecodedJpegImage(image, codes.data(), codes.size()) &&
                  !std::memcmp(codes.data(), cpu.data(), cpu.size()),
              "Quantized MC24 raw codes differ");
        if (processing) testResidentRawTable(codes, image, table);
        // A proxy must retain the frame's decode status, including after the
        // caller releases the original image. Reduction cannot make invalid
        // quantized codes valid or lose the associated control-word storage.
        for (bool valid : {true, false}) {
            rawpreamp::VulkanResidentImage source, proxy;
            check(rawpreamp::mc24DecodeVulkan(bytes.data(), bytes.size(), 4, 4, source,
                      valid ? uint32_t(table.size() - 1) : 0u),
                  "MC24 proxy validation fixture decode failed");
            check(rawpreamp::validateVulkanRawFrame(source) == valid,
                  "Unexpected MC24 source validity");
            const void* status = source.validationBuffer;
            check(status && rawpreamp::downscaleVulkanResidentImage(source, 2, true, proxy),
                  "MC24 proxy validation fixture reduction failed");
            source = {};
            check(proxy.validationBuffer == status &&
                      rawpreamp::validateVulkanRawFrame(proxy) == valid,
                  "Proxy lost source decode validity or its lifetime");
        }
        // Deferred frames must keep distinct control words while other frames
        // are decoded. Exercise reuse again after all leases are retired.
        for (unsigned cycle = 0; cycle < 2; ++cycle) {
            std::array<rawpreamp::VulkanResidentImage, 8> held;
            for (size_t i = 0; i < held.size(); ++i) {
                check(rawpreamp::mc24DecodeVulkan(bytes.data(), bytes.size(), 4, 4, held[i],
                          (i & 1u) ? 0u : uint32_t(table.size() - 1)),
                      "Overlapping MC24 decode failed");
                for (size_t j = 0; j < i; ++j)
                    check(held[j].validationBuffer != held[i].validationBuffer,
                          "Live frames share a decode control word");
            }
            for (size_t i = 0; i < held.size(); ++i)
                check(rawpreamp::validateVulkanRawFrame(held[i]) == !(i & 1u),
                      "Deferred MC24 validity was overwritten");
        }
    }
    std::filesystem::remove(path);
}
void testGalleryRotation(const rawpreamp::VulkanDecodedJpegImage& image,
                         const std::vector<uint16_t>& samples) {
    const std::array<float, 3> gains{1, 1, 1};
    const std::array<float, 9> identity{1,0,0, 0,1,0, 0,0,1};
    rawpreamp::VulkanGalleryBayerSource source;
    source.width = image.width; source.height = image.height;
    source.white.fill(65535);
    source.samples = samples.data();
    for (const auto& phase : {std::array<uint8_t, 4>{0,1,1,2},
                             std::array<uint8_t, 4>{2,1,1,0},
                             std::array<uint8_t, 4>{1,0,2,1},
                             std::array<uint8_t, 4>{1,2,0,1}}) {
        source.phase = phase;
        std::vector<uint8_t> reference;
        // The standalone demosaic shader retains the scalar neighbor algorithm.
        // Compare the fused display shader against that independent path.
        check(rawpreamp::demosaicGalleryBayerVulkan(samples.data(), reference,
            image.width, image.height, source.phase, source.black, source.white),
            "Reference scalar demosaic failed");
        std::vector<uint8_t> cpuDemosaic;
        check(rawpreamp::utils::demosaicBilinearBayerToRgb16Bytes(samples.data(), cpuDemosaic,
            image.width, image.height, source.phase, source.black, source.white),
            "CPU rotation reference demosaic failed");
        if (reference != cpuDemosaic) {
            size_t first = 0;
            while (first < reference.size() && reference[first] == cpuDemosaic[first]) ++first;
            throw std::runtime_error("Scalar GPU demosaic differs from CPU: first_byte=" + std::to_string(first));
        }
        check(rawpreamp::colorGalleryRgbVulkan(reference, gains, identity, identity,
            1, nullptr, nullptr, false, false), "Reference gallery render failed");
        source.samples = nullptr; source.gpu = image.gpu; source.buffer = image.buffer;
        for (int angle : {0, 90, 180, 270}) for (bool rgb24 : {false, true}) {
            rawpreamp::VulkanGalleryOutput output;
            output.width = image.width; output.height = image.height;
            output.orientation = angle; output.rgb24 = rgb24;
            std::vector<uint8_t> actual;
            check(rawpreamp::colorGalleryRgbVulkan(actual, gains, identity, identity,
                1, nullptr, nullptr, false, false, &source, &output), "Rotated gallery render failed");
            const uint32_t width = angle == 90 || angle == 270 ? image.height : image.width;
            const uint32_t height = angle == 90 || angle == 270 ? image.width : image.height;
            std::vector<uint8_t> expected(size_t(width) * height * (rgb24 ? 3u : 6u));
            for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
                uint32_t sx = x, sy = y;
                if (angle == 90) { sx = y; sy = image.height - 1u - x; }
                if (angle == 180) { sx = image.width - 1u - x; sy = image.height - 1u - y; }
                if (angle == 270) { sx = image.width - 1u - y; sy = x; }
                for (uint32_t c = 0; c < 3; ++c) {
                    const size_t input = (size_t(sy) * image.width + sx) * 6u + c * 2u;
                    const size_t index = (size_t(y) * width + x) * 3u + c;
                    if (rgb24) expected[index] = reference[input + 1u];
                    else std::memcpy(expected.data() + index * 2u, reference.data() + input, 2u);
                }
            }
            if (actual != expected) {
                size_t first = 0;
                while (first < std::min(actual.size(), expected.size()) && actual[first] == expected[first]) ++first;
                unsigned maximum = 0;
                for (size_t i = 0; i < std::min(actual.size(), expected.size()); i += rgb24 ? 1u : 2u) {
                    uint16_t a = 0, b = 0;
                    if (rgb24) { a = actual[i]; b = expected[i]; }
                    else { std::memcpy(&a, actual.data()+i, 2); std::memcpy(&b, expected.data()+i, 2); }
                    maximum = std::max(maximum, unsigned(std::abs(int(a)-int(b))));
                }
                uint16_t av = 0, ev = 0, raw = 0;
                if (!rgb24 && first + 1 < actual.size()) {
                    const size_t at = first & ~size_t(1);
                    std::memcpy(&av, actual.data()+at, 2); std::memcpy(&ev, expected.data()+at, 2);
                    std::memcpy(&raw, cpuDemosaic.data()+at, 2);
                }
                throw std::runtime_error("Tiled gallery rotation differs: angle=" + std::to_string(angle) +
                    " rgb24=" + std::to_string(rgb24) + " phase=" + std::to_string(phase[0]) +
                    std::to_string(phase[1]) + std::to_string(phase[2]) + std::to_string(phase[3]) +
                    " first_byte=" + std::to_string(first) + " max_code_error=" + std::to_string(maximum) +
                    " actual=" + std::to_string(av) + " expected=" + std::to_string(ev) + " raw=" + std::to_string(raw));
            }
        }
    }
}
void testGalleryTransfer() {
    std::vector<uint16_t> input(65536u * 3u);
    for (uint32_t code = 0; code < 65536u; ++code)
        for (size_t c = 0; c < 3; ++c) input[size_t(code) * 3u + c] = uint16_t(code);
    std::vector<uint8_t> output(input.size() * 2u);
    std::memcpy(output.data(), input.data(), output.size());
    const std::array<float, 3> gains{1, 1, 1};
    const auto identity = rawpreamp::gallery::identityMatrix;
    check(rawpreamp::colorGalleryRgbVulkan(output, gains, identity, identity,
        1, nullptr, nullptr, false, false), "sRGB transfer render failed");
    unsigned maximum = 0;
    for (uint32_t code = 0; code < 65536u; ++code) {
        const auto expected = std::lround(std::clamp(
            rawpreamp::gallery::srgbEncode(code / 65535.0f), 0.0f, 1.0f) * 65535.0f);
        for (size_t c = 0; c < 3; ++c) {
            uint16_t actual;
            std::memcpy(&actual, output.data() + (size_t(code) * 3u + c) * 2u, 2u);
            maximum = std::max(maximum, unsigned(std::abs(int(actual) - int(expected))));
        }
    }
    check(maximum <= 1, "GPU sRGB lookup differs from CPU transfer curve");
}
void testGalleryProfiles() {
    auto table = [](uint32_t h, uint32_t s, uint32_t v, uint32_t encoding) {
        rawpreamp::DNGProfileTable t;
        t.hueDivisions = h; t.saturationDivisions = s; t.valueDivisions = v; t.encoding = encoding;
        for (uint32_t z = 0; z < v; ++z) for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < s; ++x) {
                t.values.push_back(float(y) * 0.5f - 1.0f);
                t.values.push_back(0.8f + float(x) * 0.03f + float(z) * 0.02f);
                t.values.push_back(0.8f + float(y) * 0.02f + float(z) * 0.03f);
            }
        return t;
    };
    const std::array<float, 3> gains{1, 1, 1};
    const auto identity = rawpreamp::gallery::identityMatrix;
    constexpr float exposure = 1.1f;
    std::vector<uint16_t> input(1536 * 3);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t((i * 7919u + i / 7u) % 16000u);
    for (unsigned maps : {1u, 2u, 3u}) for (uint32_t divisions : {1u, 3u}) for (uint32_t encoding : {0u, 1u}) {
        auto hue = table(7, 5, divisions, encoding), look = table(5, 4, 3, encoding);
        if (encoding) for (size_t i=0; i<hue.values.size(); i+=3) hue.values[i] += 4.0f;
        if (encoding) for (size_t i=2; i<look.values.size(); i+=3) look.values[i] *= 0.93f;
        std::vector<uint8_t> output(input.size() * 2u);
        std::memcpy(output.data(), input.data(), output.size());
        check(rawpreamp::colorGalleryRgbVulkan(output, gains, identity, identity,
            exposure, maps & 1u ? &hue : nullptr, maps & 2u ? &look : nullptr, true, false), "Profile gallery render failed");
        unsigned maximum = 0;
        for (size_t i = 0; i < input.size(); i += 3) {
            const float x = input[i] / 65535.0f, y = input[i + 1] / 65535.0f, z = input[i + 2] / 65535.0f;
            std::array<float, 3> rgb{1.3459433f*x - 0.2556075f*y - 0.0511118f*z,
                -0.5445989f*x + 1.5081673f*y + 0.0205351f*z, 1.2118128f*z};
            if (maps == 3 && divisions == 1 && encoding == 0)
                rawpreamp::gallery::applyHueSatAndLook(rgb, hue, look, exposure);
            else {
                if (maps & 1u) rawpreamp::gallery::applyProfileTable(rgb, hue);
                for (auto& v : rgb) v *= exposure;
                if (maps & 2u) rawpreamp::gallery::applyProfileTable(rgb, look);
            }
            const std::array<float, 3> xyz{0.7976749f*rgb[0]+0.1351917f*rgb[1]+0.0313534f*rgb[2],
                0.2880402f*rgb[0]+0.7118741f*rgb[1]+0.0000857f*rgb[2], 0.82521f*rgb[2]};
            const std::array<float, 3> display{
                3.1338561f*xyz[0]-1.6168667f*xyz[1]-0.4906146f*xyz[2],
                -0.9787684f*xyz[0]+1.9161415f*xyz[1]+0.0334540f*xyz[2],
                0.0719453f*xyz[0]-0.2289914f*xyz[1]+1.4052427f*xyz[2]};
            for (size_t c = 0; c < 3; ++c) {
                const auto linear = std::lround(std::clamp(display[c], 0.0f, 1.0f) * 65535.0f);
                const auto expected = std::lround(std::clamp(rawpreamp::gallery::srgbEncode(linear / 65535.0f), 0.0f, 1.0f) * 65535.0f);
                uint16_t actual;
                std::memcpy(&actual, output.data() + (i + c) * 2u, 2u);
                maximum = std::max(maximum, unsigned(std::abs(int(actual) - int(expected))));
            }
        }
        std::fprintf(stderr, "Gallery profile CPU comparison: maps=%u value_divisions=%u encoding=%u max_error=%u\n", maps, divisions, encoding, maximum);
        // The previous buffer sampler also differs by five codes for the
        // look-only, unencoded near-black fixture. Preserve its bound.
        const unsigned tolerance = maps == 2 && encoding == 0 ? 5u : 4u;
        check(maximum <= tolerance, "Profile interpolation differs from CPU color math");
    }
}
void testUnspektra() {
    constexpr uint32_t width = 34, height = 18;
    std::vector<uint8_t> bytes(16);
    std::memcpy(bytes.data(), "UFR1", 4);
    u32(bytes, 4, width);
    u32(bytes, 8, height);
    u32(bytes, 12, 2);
    std::vector<uint16_t> expected(width * height);
    uint32_t block = 0;
    for (uint32_t y = 0; y < height; y += 16)
        for (uint32_t x = 0; x < width; x += 16)
            for (uint32_t group = 0; group < 4; ++group, ++block) {
                const uint32_t bits = block % 17, reference = bits == 16 ? 0 : 123;
                const size_t at = bytes.size();
                bytes.resize(at + 3 + bits * 8);
                bytes[at] = uint8_t(bits);
                u16(bytes, at + 1, reference);
                for (uint32_t i = 0; i < 64; ++i) {
                    const uint32_t delta = (i * 113 + block * 37) & ((1u << bits) - 1u);
                    for (uint32_t bit = 0; bit < bits; ++bit)
                        bytes[at + 3 + (i * bits + bit) / 8] |=
                            uint8_t(((delta >> bit) & 1u) << ((i * bits + bit) % 8));
                    const uint32_t px = x + 2 * (i % 8) + (group & 1),
                                   py = y + 2 * (i / 8) + (group >> 1);
                    if (px < width && py < height)
                        expected[py * width + px] = uint16_t(reference + delta);
                }
            }
    for (uint32_t cfa : {2u, 4u}) {
        u32(bytes, 12, cfa);
        rawpreamp::VulkanDecodedJpegImage image;
        check(rawpreamp::unspektraDecodeVulkan(bytes.data(), bytes.size(), width, height, image),
              "UFR1 compute fixture decode failed");
        std::vector<uint16_t> actual(expected.size());
        check(rawpreamp::downloadVulkanDecodedJpegImage(image, actual.data(), actual.size()) &&
                  actual == expected,
              "UFR1 block widths or partial tiles differ");
        if (cfa == 2) testGalleryRotation(image, expected);
        check(
            !rawpreamp::unspektraDecodeVulkan(bytes.data(), bytes.size() - 1, width, height, image),
            "Truncated UFR1 accepted");
        auto malformed = bytes;
        u32(malformed, 12, 3);
        check(!rawpreamp::unspektraDecodeVulkan(malformed.data(), malformed.size(), width, height,
                                               image), "Unsupported UFR1 CFA size accepted");
        malformed = bytes;
        malformed[16] = 17;
        check(!rawpreamp::unspektraDecodeVulkan(malformed.data(), malformed.size(), width, height,
                                                image),
              "Invalid UFR1 block width accepted");
    }
}
} // namespace
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
void testSharedVulkanResources() {
    auto& context = rawpreamp::jpegContext();
    std::unique_lock lock(context.mutex);
    check(context.init(), "Shared Vulkan context unavailable");
    const auto gpu = context.vulkan->gpu;
    pl_buf first = nullptr, second = nullptr, recycled = nullptr;
    auto a = context.acquireImage(1156, first);
    auto b = context.acquireImage(1156, second);
    check(a && b && first != second, "Leased GPU images must not alias");
    pl_gpu_finish(gpu);
    a.reset();
    auto c = context.acquireImage(1156, recycled);
    check(c && recycled == first, "Retired GPU image was not reused");
    c.reset(); b.reset();
    // Changing clip/proxy sizes must not leave the pool occupied exclusively
    // by old resolutions. Keep its retained byte count bounded throughout.
    for (size_t bytes = 1200; bytes <= 1224; bytes += 4) {
        pl_buf current = nullptr;
        auto lease = context.acquireImage(bytes, current);
        check(bool(lease), "Pool size transition allocation failed");
        pl_gpu_finish(gpu);
        lease.reset();
        pl_buf reused = nullptr;
        auto again = context.acquireImage(bytes, reused);
        check(again && reused == current, "Old resolutions prevented pool reuse");
        check(context.retiredImageBytes <= 96u * 1024u * 1024u &&
              context.retiredImages.size() <= 4, "GPU pool exceeded its budget");
    }
    pl_buf_params params{};
    params.size = 32; params.storable = true;
    pl_buf target = pl_buf_create(gpu, &params);
    params.storable = false; params.host_writable = true;
    pl_buf staging = pl_buf_create(gpu, &params);
    params.host_writable = false; params.host_readable = true;
    pl_buf readback = pl_buf_create(gpu, &params);
    check(target && staging && readback, "Resource test allocation failed");
    rawpreamp::VulkanUploadCache cache;
    std::array<uint32_t, 8> values{1,2,3,4,5,6,7,8}, actual{};
    check(cache.upload(gpu, target, staging, values.data(), 32) == 32,
          "Initial resource upload missing");
    check(cache.upload(gpu, target, staging, values.data(), 32) == 0,
          "Unchanged resource was uploaded again");
    values[2] = 99;
    check(cache.upload(gpu, target, staging, values.data(), 32) == 4,
          "Frame seed update uploaded unchanged tables");
    pl_buf_copy(gpu, readback, 0, target, 0, 32);
    check(pl_buf_read(gpu, readback, 0, actual.data(), 32) && actual == values,
          "Partial resource update corrupted adjacent words");
    pl_buf_params replacementParams{};
    replacementParams.size = 32;
    replacementParams.storable = true;
    pl_buf replacement = pl_buf_create(gpu, &replacementParams);
    check(replacement && cache.upload(gpu, replacement, staging, values.data(), 32) == 32,
          "Unchanged data was not uploaded to a different target");
    pl_buf_copy(gpu, readback, 0, replacement, 0, 32);
    check(pl_buf_read(gpu, readback, 0, actual.data(), 32) && actual == values,
          "Replacement target contains stale data");
    pl_buf_destroy(gpu, &replacement);
    cache.invalidate();
    check(cache.upload(gpu, target, staging, values.data(), 32) == 32,
          "Invalidated resource was not restored");
    rawpreamp::VulkanResidentImage malformed;
    malformed.gpu = gpu; malformed.buffer = target;
    malformed.width = 32; malformed.height = 2; malformed.channels = 3;
    malformed.bits = 16; malformed.wordsPerRow = 48;
    malformed.lease = std::make_shared<int>(0);
    rawpreamp::VulkanResidentImage reduced;
    std::vector<uint16_t> oversized(32u * 2u * 3u);
    check(!rawpreamp::downscaleVulkanResidentImage(malformed, 2, false, reduced),
          "Proxy accepted a resident buffer smaller than its declared image");
    check(!rawpreamp::downloadVulkanResidentImage(malformed, oversized.data(), oversized.size()),
          "Download accepted a resident buffer smaller than its declared image");
    pl_buf_destroy(gpu, &readback);
    pl_buf_destroy(gpu, &staging);
    pl_buf_destroy(gpu, &target);

    pl_buf threadedBuffer = nullptr;
    auto threadedLease = context.acquireImage(1156, threadedBuffer);
    check(bool(threadedLease), "Threaded lease allocation failed");
    std::promise<void> started, finished;
    auto finishedFuture = finished.get_future();
    auto startedFuture = started.get_future();
    std::thread consumer([lease = std::move(threadedLease), &started, &finished]() mutable {
        started.set_value();
        lease.reset();
        finished.set_value();
    });
    startedFuture.wait();
    const bool serialized = finishedFuture.wait_for(std::chrono::milliseconds(20)) ==
                            std::future_status::timeout;
    lock.unlock();
    consumer.join();
    check(serialized, "Consumer retirement bypassed the GPU submission lock");
}
#endif
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
void testResidentCfaBinning() {
    auto& context = rawpreamp::jpegContext();
    std::lock_guard lock(context.mutex);
    check(context.init(), "CFA binning Vulkan initialization failed");
    const auto gpu = context.vulkan->gpu;
    constexpr uint32_t width = 24, height = 24;
    std::vector<uint16_t> pixels(width * height);
    for (size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = uint16_t(i * 1049u + (i / width) * 997u);
    pixels[0] = pixels[1] = pixels[width] = pixels[width + 1] = 65535;
    pl_buf_params params{};
    params.size = pixels.size() * 2; params.storable = true;
    params.initial_data = pixels.data();
    pl_buf buffer = pl_buf_create(gpu, &params);
    check(bool(buffer), "CFA binning source allocation failed");
    rawpreamp::VulkanResidentImage source;
    source.gpu = gpu; source.buffer = buffer;
    source.width = width; source.height = height; source.channels = 1;
    source.bits = 16; source.wordsPerRow = width / 2;
    source.deviceLease = rawpreamp::jpegContextLease();
    source.lease = std::shared_ptr<void>(new pl_buf(buffer),
        [gpu, owner = source.deviceLease, context = &context](void* pointer) {
            std::lock_guard lock(context->mutex);
            auto* owned = static_cast<pl_buf*>(pointer);
            pl_buf_destroy(gpu, owned); delete owned;
        });
    for (uint32_t factor : {2u, 3u, 4u}) {
        rawpreamp::VulkanResidentImage reduced;
        check(rawpreamp::binVulkanResidentCfa(source, factor, reduced), "GPU CFA binning failed");
        uint32_t w = 0, h = 0;
        std::vector<uint16_t> expected;
        rawpreamp::utils::binHigherCFA(pixels, expected, width, height, factor, w, h, 0);
        std::vector<uint16_t> actual(size_t(w) * h);
        check(reduced.width == w && reduced.height == h &&
            rawpreamp::downloadVulkanResidentImage(reduced, actual.data(), actual.size()) &&
            actual == expected, "GPU CFA binning differs from CPU rounding");
    }
#ifdef __ANDROID__
    // App self-tests link the metadata writer; the standalone Linux decoder
    // harness intentionally uses smaller utility implementations.
    {
        rawpreamp::CameraFrameMetadata frame{};
        frame.width = frame.originalWidth = width;
        frame.height = frame.originalHeight = height;
        frame.cfaSize = 2; frame.rowStride = width * 2;
        frame.pixelFormat = "raw16"; frame.iso = 100; frame.exposureTime = 10000000;
        frame.dynamicWhiteLevel = 65535; frame.asShotNeutral = {1, 1, 1};
        rawpreamp::CameraConfiguration camera{};
        camera.whiteLevel = 65535; camera.sensorArrangement = "rggb";
        camera.colorMatrix1 = camera.colorMatrix2 = {1,0,0,0,1,0,0,0,1};
        rawpreamp::CalibrationData calibration{};
        calibration.hasCfaSize = true; calibration.cfaSize = 4;
        rawpreamp::RenderSettings staging;
        staging.options = rawpreamp::RENDER_OPT_NONE;
        staging.cameraNativeStaging = false; staging.streamingPreview = false;
        staging.quadBayerOption = rawpreamp::QuadBayerMode::CorrectQBCFAMetadata;
        std::vector<uint8_t> noPixels;
        auto header = rawpreamp::utils::generateDng(noPixels, frame, camera, 24, 0, 1,
            staging, calibration, false, std::nullopt, std::nullopt,
            nullptr, false, nullptr, true, true);
        rawpreamp::DNGImageLayout layout;
        check(header && rawpreamp::DNGDecoder::getImageLayout(*header, layout) &&
              layout.cfaRepeatSize == 4 && layout.width == width && layout.height == height,
              "Resident MCRAW metadata staging lost the calibrated 4x4 CFA");
    }
#endif
    for (bool explicitBin : {false, true}) {
        rawpreamp::DecodedDNGImage cpu;
        cpu.layout.width = width; cpu.layout.height = height;
        cpu.layout.pixels = rawpreamp::DNGPixelLayout::CFA;
        cpu.layout.samplesPerPixel = 1; cpu.layout.bitsPerSample = 16;
        cpu.layout.cfaRepeatSize = 4; cpu.layout.cfaPhase = {0, 1, 1, 2};
        cpu.storedBitsPerSample = 16; cpu.metadata.inputBitDepth = 16;
        cpu.metadata.blackLevel.fill(0); cpu.metadata.blackLevelCount = 4;
        cpu.metadata.whiteLevel.fill(65535); cpu.metadata.whiteLevelCount = 4;
        cpu.samples = pixels;
        auto resident = cpu;
        resident.samples.clear();
        resident.gpuImage = std::make_shared<rawpreamp::DNGGpuImage>();
        resident.gpuImage->gpu = gpu; resident.gpuImage->buffer = buffer;
        resident.gpuImage->width = width; resident.gpuImage->height = height;
        resident.gpuImage->channels = 1; resident.gpuImage->lease = source.lease;
        resident.gpuImage->deviceLease = source.deviceLease;
        auto recoveryCalls = std::make_shared<unsigned>(0);
        resident.gpuImage->download = [pixels, recoveryCalls](uint16_t* out, size_t count) {
            ++*recoveryCalls;
            if (count != pixels.size()) return false;
            std::copy(pixels.begin(), pixels.end(), out); return true;
        };
        rawpreamp::RenderSettings settings;
        settings.streamingPreview = true; settings.draftScale = 2;
        settings.options = rawpreamp::RENDER_OPT_DRAFT;
        settings.quadBayerOption = explicitBin ? rawpreamp::QuadBayerMode::Binning
                                              : rawpreamp::QuadBayerMode::Demosaic;
        rawpreamp::PreviewFrame expected, actual;
        check(rawpreamp::DNGDecoder::decodePreview(std::move(cpu), settings, expected, true, false) &&
              rawpreamp::DNGDecoder::decodePreview(std::move(resident), settings, actual, true, false),
              "Higher-CFA CPU/resident preview failed");
        check(actual.gallerySource.buffer && actual.width == expected.width && actual.height == expected.height,
              "Higher-CFA GPU preview fell back or applied proxy scale twice");
        check(*recoveryCalls == 0, "Resident binning unexpectedly decoded CPU samples");
        std::vector<uint16_t> recovered(size_t(actual.width) * actual.height);
        check(actual.gallerySource.download &&
              actual.gallerySource.download(recovered.data(), recovered.size()) &&
              recovered == expected.galleryCfa && *recoveryCalls == 1,
              "Reduced preview recovery did not use the original CPU source");
        rawpreamp::VulkanResidentImage image;
        image.gpu = actual.gallerySource.gpu; image.buffer = actual.gallerySource.buffer;
        image.width = actual.width; image.height = actual.height;
        image.channels = 1; image.wordsPerRow = actual.width / 2;
        image.lease = actual.gallerySource.lease;
        std::vector<uint16_t> output(size_t(image.width) * image.height);
        check(rawpreamp::downloadVulkanResidentImage(image, output.data(), output.size()) &&
              output == expected.galleryCfa, "Higher-CFA resident preview differs from CPU");
    }
    // A gain ROI covering sensor columns/rows 6..7 must follow the selected
    // raw block. In particular, bin-2 + proxy-2 maps output (1,1) to sensor
    // (7,7), not the flattened-scale coordinate (5,5).
    for (unsigned mode = 0; mode < 4; ++mode) {
        const uint32_t topology = mode == 3 ? 1u : 2u;
        const uint32_t proxy = mode == 2 ? 1u : 2u;
        const bool explicitBin = mode == 0;
        rawpreamp::DecodedDNGImage cpu;
        cpu.layout.width = width; cpu.layout.height = height;
        cpu.layout.pixels = rawpreamp::DNGPixelLayout::CFA;
        cpu.layout.samplesPerPixel = 1; cpu.layout.bitsPerSample = 16;
        cpu.layout.cfaRepeatSize = 2 * topology; cpu.layout.cfaPhase = {0,1,1,2};
        cpu.storedBitsPerSample = cpu.metadata.inputBitDepth = 16;
        cpu.metadata.blackLevel.fill(0); cpu.metadata.blackLevelCount = 4;
        cpu.metadata.whiteLevel.fill(65535); cpu.metadata.whiteLevelCount = 4;
        cpu.samples = pixels;
        rawpreamp::GainMap map{};
        map.left = map.top = 6; map.right = map.bottom = 8;
        map.width = map.height = 1; map.channels = 4; map.planes = 1;
        map.rowPitch = map.colPitch = 1; map.spacingH = map.spacingV = 1;
        map.data.assign(4, 2.0f);
        cpu.opcodeList2.push_back(map);
        auto resident = cpu;
        resident.samples.clear();
        resident.gpuImage = std::make_shared<rawpreamp::DNGGpuImage>();
        resident.gpuImage->gpu = gpu; resident.gpuImage->buffer = buffer;
        resident.gpuImage->width = width; resident.gpuImage->height = height;
        resident.gpuImage->channels = 1; resident.gpuImage->lease = source.lease;
        resident.gpuImage->deviceLease = source.deviceLease;
        resident.gpuImage->download = [pixels](uint16_t* out, size_t count) {
            if (count != pixels.size()) return false;
            std::copy(pixels.begin(), pixels.end(), out); return true;
        };
        rawpreamp::RenderSettings settings;
        settings.streamingPreview = true;
        settings.draftScale = explicitBin ? proxy : topology * proxy;
        settings.options = static_cast<rawpreamp::FileRenderOptions>(
            rawpreamp::RENDER_OPT_DRAFT | rawpreamp::RENDER_OPT_APPLY_VIGNETTE_CORRECTION);
        settings.quadBayerOption = explicitBin ? rawpreamp::QuadBayerMode::Binning
                                              : rawpreamp::QuadBayerMode::Demosaic;
        rawpreamp::PreviewFrame cpuFrame, gpuFrame;
        check(rawpreamp::DNGDecoder::decodePreview(std::move(cpu), settings, cpuFrame, true, false) &&
              rawpreamp::DNGDecoder::decodePreview(std::move(resident), settings, gpuFrame, true, false),
              "Composed gain-coordinate preview failed");
        std::vector<uint16_t> binned = pixels;
        uint32_t binWidth = width, binHeight = height;
        if (topology > 1)
            rawpreamp::utils::binHigherCFA(pixels, binned, width, height, topology,
                                           binWidth, binHeight, 0);
        const uint32_t outWidth = binWidth / proxy, outHeight = binHeight / proxy;
        std::vector<uint16_t> oracle(size_t(outWidth) * outHeight);
        for (uint32_t y = 0; y < outHeight; ++y) for (uint32_t x = 0; x < outWidth; ++x) {
            const uint32_t bx = x * proxy + (proxy > 1 ? x & 1u : 0u);
            const uint32_t by = y * proxy + (proxy > 1 ? y & 1u : 0u);
            const uint32_t sx = bx * topology + (topology > 1 ? bx & 1u : 0u);
            const uint32_t sy = by * topology + (topology > 1 ? by & 1u : 0u);
            uint32_t value = binned[size_t(by) * binWidth + bx];
            if (sx >= 6 && sx < 8 && sy >= 6 && sy < 8) value *= 2;
            oracle[size_t(y) * outWidth + x] = uint16_t(std::min(value, 65535u));
        }
        check(cpuFrame.galleryCfa == oracle, "CPU gain lookup lost composed Bayer parity");
        check(gpuFrame.gallerySource.buffer && gpuFrame.galleryCfaGain &&
              gpuFrame.width == outWidth && gpuFrame.height == outHeight,
              "Composed gain preview did not remain resident");
        std::vector<uint16_t> recovered(oracle.size());
        check(gpuFrame.gallerySource.download(recovered.data(), recovered.size()) &&
              gpuFrame.galleryCfaRecoverCpu && gpuFrame.galleryCfaRecoverCpu(recovered) &&
              recovered == oracle, "CPU recovery lost composed gain coordinates");
        rawpreamp::VulkanGalleryBayerSource render;
        render.width = outWidth; render.height = outHeight;
        render.phase = {0,1,1,2}; render.white.fill(65535);
        render.samples = oracle.data();
        rawpreamp::VulkanGalleryOutput output;
        output.width = outWidth; output.height = outHeight;
        const std::array<float,3> gains{1,1,1};
        const std::array<float,9> identity{1,0,0,0,1,0,0,0,1};
        std::vector<uint8_t> expectedDisplay, actualDisplay;
        check(rawpreamp::colorGalleryRgbVulkan(expectedDisplay, gains, identity, identity,
                  1, nullptr, nullptr, false, false, &render, &output),
              "Gain-coordinate reference render failed");
        render.samples = nullptr; render.gpu = gpu; render.buffer = gpuFrame.gallerySource.buffer;
        render.gain = gpuFrame.galleryCfaGain.get();
        check(rawpreamp::colorGalleryRgbVulkan(actualDisplay, gains, identity, identity,
                  1, nullptr, nullptr, false, false, &render, &output) && actualDisplay == expectedDisplay,
              "GPU gain lookup lost composed Bayer parity");
    }
    auto malformed = source;
    malformed.width = 22; malformed.wordsPerRow = 11;
    rawpreamp::VulkanResidentImage rejected;
    check(!rawpreamp::binVulkanResidentCfa(malformed, 2, rejected),
          "CFA binning accepted an incomplete color block");
    std::fprintf(stderr, "Resident CFA binning: factors 2/3/4 match CPU exactly\n");
}
#endif
void testNativeRawVulkan(bool processing) {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
    testSharedVulkanResources();
    testResidentCfaBinning();
#endif
    testMc24();
    testUnspektra();
    testQuantizedMc24(processing);
    if (processing) {
        testGalleryProfiles();
        testGalleryTransfer();
    }
}

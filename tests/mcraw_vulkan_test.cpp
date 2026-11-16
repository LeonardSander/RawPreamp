#include "VulkanRawDecode.h"
#include "VulkanJpegDecode.h"
#include "VulkanJpegFullEncode.h"
#include "VulkanGainBake.h"
#include "DNGDecoder.h"
#include "tinydng/tiny_dng_writer.h"
#include <sstream>
#include <motioncam/RawData.hpp>
#include <motioncam/Decoder.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
void put32(std::vector<uint8_t>& data, size_t at, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
        data[at + i] = uint8_t(value >> (8 * i));
}
void table(std::vector<uint8_t>& data, const std::vector<uint16_t>& values) {
    const size_t at = data.size();
    data.resize(at + 6 + values.size() * 2);
    put32(data, at, uint32_t(values.size()));
    data[at + 4] = 0xb0;
    data[at + 5] = 0; // 64 literal uint16 values
    for (size_t i = 0; i < values.size(); ++i) {
        data[at + 6 + i * 2] = uint8_t(values[i]);
        data[at + 7 + i * 2] = uint8_t(values[i] >> 8);
    }
}
std::vector<uint8_t> dng(const std::vector<uint16_t>& pixels, bool gain, uint32_t width = 256,
                         uint32_t height = 16, const std::vector<uint16_t>& table = {}) {
    tinydngwriter::DNGImage image;
    image.SetBigEndian(false);
    const unsigned short bits = 16, black[] = {64, 72, 80, 88};
    require(
        image.SetImageWidth(width) && image.SetImageLength(height) &&
            image.SetRowsPerStrip(height) && image.SetSamplesPerPixel(1) &&
            image.SetBitsPerSample(1, &bits) &&
            image.SetCompression(tinydngwriter::COMPRESSION_NONE) &&
            image.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA) &&
            image.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG) &&
            image.SetBlackLevelRepeatDim(2, 2) && image.SetBlackLevel(4, black) &&
            image.SetWhiteLevel(65535) &&
            image.SetImageData(reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * 2),
        "DNG template failed");
    if (!table.empty())
        require(image.SetLinearizationTable(uint32_t(table.size()), table.data()),
                "DNG linearization failed");
    if (gain) {
        tinydngwriter::GainMapParams map{};
        map.bottom = height;
        map.right = width;
        map.planes = 1;
        map.row_pitch = map.col_pitch = 1;
        map.map_points_v = map.map_points_h = 2;
        map.map_spacing_v = map.map_spacing_h = 0.5;
        map.map_planes = 1;
        map.gain_data = {1.0f, 1.1f, 1.2f, 1.3f};
        tinydngwriter::OpcodeList opcodes;
        opcodes.AddGainMap(map);
        require(image.SetOpcodeList2(opcodes), "DNG gain map failed");
    }
    tinydngwriter::DNGWriter writer(false);
    require(writer.AddImage(&image), "DNG add image failed");
    std::ostringstream output(std::ios::binary);
    std::string error;
    require(writer.WriteToFile(output, &error), "DNG serialization failed");
    const auto bytes = output.str();
    return {bytes.begin(), bytes.end()};
}
std::vector<uint8_t> fixture() {
    constexpr uint32_t lengths[] = {0,  8,  16,  24,  32,  40,  48,  64, 64,
                                    80, 80, 128, 128, 128, 128, 128, 128};
    std::vector<uint8_t> bytes(16);
    put32(bytes, 0, 256);
    put32(bytes, 4, 16);
    std::vector<uint16_t> bits(64), refs(64);
    for (unsigned i = 0; i < 64; ++i) {
        bits[i] = i % 17;
        refs[i] = uint16_t(i * 1037);
        for (unsigned j = 0; j < lengths[bits[i]]; ++j)
            bytes.push_back(uint8_t(i * 71 + j * 113 + j / 7));
    }
    put32(bytes, 8, uint32_t(bytes.size()));
    table(bytes, bits);
    put32(bytes, 12, uint32_t(bytes.size()));
    table(bytes, refs);
    return bytes;
}
} // namespace
void testResidentRawTable(const std::vector<uint16_t>& codes,
                          const rawpreamp::VulkanDecodedJpegImage& image,
                          const std::vector<uint16_t>& table) {
    const rawpreamp::VulkanGainBakeBufferSource source{image.gpu, image.buffer, image.deviceLease,
                                                       image.validationBuffer};
    for (bool gain : {false, true}) {
        for (auto mode :
             {rawpreamp::LogTransformMode::Disabled, rawpreamp::LogTransformMode::ReduceBy4Bit,
              rawpreamp::LogTransformMode::KeepInput}) {
            auto reference = dng(codes, gain, image.width, image.height, table);
            auto resident = dng(std::vector<uint16_t>(codes.size(), 0xdead), gain, image.width,
                                image.height, table);
            rawpreamp::DNGFrameMetadata metadata;
            require(rawpreamp::DNGDecoder::getColorMetadata(reference, metadata),
                    "Table metadata failed");
            const uint32_t white = (1u << metadata.inputBitDepth) - 1u;
            bool fused = false;
            if (gain) {
                require(rawpreamp::DNGDecoder::bakeGainMaps(reference, false, false, false, false,
                                                            2, std::nullopt, mode, white, false,
                                                            &fused, true, 7, false),
                        "CPU table gain failed");
                if (fused)
                    require(rawpreamp::DNGDecoder::applyLogTransform(reference, mode, white, false,
                                                                     true),
                            "CPU table LOG tags failed");
                fused = false;
                require(rawpreamp::DNGDecoder::bakeGainMaps(
                            resident, false, false, false, false, 2, std::nullopt, mode, white,
                            false, &fused, true, 7, true, nullptr, nullptr, nullptr,
                            rawpreamp::LogCurve::LOG60, &source),
                        "GPU resident table gain failed");
                if (fused)
                    require(rawpreamp::DNGDecoder::applyLogTransform(resident, mode, white, false,
                                                                     true),
                            "GPU table LOG tags failed");
            } else {
                if (mode != rawpreamp::LogTransformMode::Disabled &&
                    mode != rawpreamp::LogTransformMode::KeepInput)
                    require(rawpreamp::DNGDecoder::applyLogTransform(reference, mode, white, false,
                                                                     false, 7),
                            "CPU table LOG failed");
                require(rawpreamp::DNGDecoder::bakeLogStripVulkan(
                            resident, mode, white, false, 7, nullptr, nullptr,
                            rawpreamp::LogCurve::LOG60, &source),
                        "GPU resident table pack/LOG failed");
            }
            require(rawpreamp::DNGDecoder::packUncompressedToWhiteLevel(reference),
                    "CPU table output packing failed");
            rawpreamp::DecodedDNGImage expected, actual;
            require(
                rawpreamp::DNGDecoder::decodeImage(std::move(reference), expected, false, false) &&
                    rawpreamp::DNGDecoder::decodeImage(resident, actual, false, false),
                "Table output DNG decode failed");
            require(expected.samples.size() == actual.samples.size() &&
                        expected.layout.bitsPerSample == actual.layout.bitsPerSample,
                    "Table output storage differs");
            unsigned maximumError = 0;
            for (size_t i = 0; i < codes.size(); ++i)
                maximumError =
                    std::max(maximumError,
                             unsigned(std::abs(int(expected.samples[i]) - int(actual.samples[i]))));
            std::fprintf(stderr,
                         "Resident table processing entries=%zu gain=%d mode=%d max_error=%u\n",
                         table.size(), gain, int(mode), maximumError);
            require(maximumError <= 1, "Resident table gain/LOG differs from CPU");
            if (actual.storedBitsPerSample >= 8) {
                auto compressed = dng(std::vector<uint16_t>(codes.size(), 0xdead), gain,
                                      image.width, image.height, table);
                std::vector<std::vector<uint8_t>> tiles;
                fused = false;
                const bool encoded =
                    gain ? rawpreamp::DNGDecoder::bakeGainMaps(
                               compressed, false, false, false, false, 2, std::nullopt, mode, white,
                               false, &fused, true, 7, true, nullptr, nullptr, &tiles,
                               rawpreamp::LogCurve::LOG60, &source)
                         : rawpreamp::DNGDecoder::bakeLogStripVulkan(
                               compressed, mode, white, false, 7, nullptr, &tiles,
                               rawpreamp::LogCurve::LOG60, &source);
                require(encoded && !tiles.empty(), "Resident table GPU JPEG failed");
                if (fused)
                    require(rawpreamp::DNGDecoder::applyLogTransform(compressed, mode, white, false,
                                                                     true),
                            "Table JPEG LOG tags failed");
                const auto shape = rawpreamp::jpeg92TileShape(1);
                require(rawpreamp::DNGDecoder::attachLosslessJPEGTiles(compressed, tiles,
                                                                       shape.width, shape.height),
                        "Table JPEG DNG assembly failed");
                rawpreamp::DecodedDNGImage roundtrip;
                require(rawpreamp::DNGDecoder::decodeImage(std::move(compressed), roundtrip, false,
                                                           false) &&
                            roundtrip.samples == actual.samples,
                        "Table GPU JPEG changed stored codes");
            }
        }
    }
}
void testRawSession() {
    const char* path = std::getenv("RAWPREAMP_TEST_RAW_SESSION");
    if (!path)
        return;
    std::ifstream input(path);
    nlohmann::json session;
    input >> session;
    std::map<motioncam::EncodedFrameFormat, unsigned> tested;
    for (const auto& clip : session.at("clips")) {
        const std::filesystem::path file = clip.at("path").get<std::string>();
        const auto ext = file.extension().string();
        if ((ext != ".mcraw" && ext != ".mcpro" && ext != ".mcpro24fps" && ext != ".unspektra") ||
            !std::filesystem::is_regular_file(file))
            continue;
        motioncam::Decoder decoder(file.string());
        const auto format = decoder.encodedFrameFormat();
        if (decoder.getFrames().empty() || tested[format] >= 4)
            continue;
        if (tested[format] >= 3) {
            nlohmann::json metadata;
            decoder.loadFrameMetadata(decoder.getFrames().front(), metadata);
            if (metadata.value("linearizationTable", nlohmann::json::array()).empty())
                continue;
        }
        std::vector<uint8_t> encoded, cpu;
        nlohmann::json metadata, referenceMetadata;
        require(decoder.loadEncodedFrame(decoder.getFrames().front(), encoded, metadata) ==
                        format &&
                    !encoded.empty(),
                "Session encoded frame load failed");
        decoder.loadFrame(decoder.getFrames().front(), cpu, referenceMetadata);
        require(metadata == referenceMetadata, "Encoded/decoded source metadata differs");
        if (rawpreamp::supportsRawFrameDecodeVulkan(format) &&
            (format != motioncam::EncodedFrameFormat::MotionCam ||
             metadata.value("compressionType", 0) == 7)) {
            const uint32_t w = metadata.at("width"), h = metadata.at("height");
            rawpreamp::VulkanDecodedJpegImage image;
            if (rawpreamp::rawFrameDecodeVulkan(
                    format, metadata.value("compressionType", 0), encoded.data(), encoded.size(), w,
                    h, image,
                    metadata.value("linearizationTable", nlohmann::json::array()).empty()
                        ? 65535u
                        : uint32_t(metadata.at("linearizationTable").size() - 1u))) {
                std::vector<uint16_t> pixels(size_t(w) * h);
                require(rawpreamp::downloadVulkanDecodedJpegImage(image, pixels.data(),
                                                                  pixels.size()) &&
                            cpu.size() == pixels.size() * 2 &&
                            !std::memcmp(cpu.data(), pixels.data(), cpu.size()),
                        "Session GPU raw decode mismatch");
                std::fprintf(stderr, "Session GPU decode exact: %s (%ux%u)\n",
                             file.filename().string().c_str(), w, h);
                require(rawpreamp::validateVulkanRawFrame(image),
                        "Session quantized range validation failed");
                if (metadata.contains("linearizationTable"))
                    testResidentRawTable(
                        pixels, image,
                        metadata.at("linearizationTable").get<std::vector<uint16_t>>());
            } else
                std::fprintf(stderr, "Session source retains CPU layout fallback: %s\n",
                             file.filename().string().c_str());
        } else
            std::fprintf(stderr, "Session compressed interface verified: %s codec=%d\n",
                         file.filename().string().c_str(), int(format));
        ++tested[format];
    }
}
void testNativeRawVulkan(bool processing = true);

// Decoder-only entry point for devices: keep the independent CPU/GPU LOG
// oracle in the broader processing suite, without weakening its tolerance.
void testRawDecodeVulkan() {
    const auto encoded = fixture();
    for (uint32_t width : {256u,196u,4u}) {
        std::vector<uint16_t> expected(width*16u), actual(expected.size());
        require(motioncam::raw::Decode(expected.data(),width,16,encoded.data(),encoded.size())==expected.size(),
                "MCRAW CPU fixture decode failed");
        rawpreamp::VulkanDecodedJpegImage image;
        require(rawpreamp::mcrawDecodeVulkan(encoded.data(),encoded.size(),width,16,image) &&
                rawpreamp::downloadVulkanDecodedJpegImage(image,actual.data(),actual.size()) && actual==expected,
                "MCRAW decode differs from CPU");
    }
    testNativeRawVulkan(false);
}

void testMcrawVulkan() {
    testRawSession();
    auto encoded = fixture();
    rawpreamp::VulkanDecodedJpegImage first;
    if (!rawpreamp::mcrawDecodeVulkan(encoded.data(), encoded.size(), 256, 16, first)) {
        std::fprintf(stderr, "MCRAW Vulkan unavailable; resident decode test skipped\n");
        require(!std::getenv("RAWPREAMP_TEST_REQUIRE_VULKAN"), "MCRAW Vulkan required");
        return;
    }
    testNativeRawVulkan();
    for (uint32_t width : {256u, 196u, 4u}) {
        std::vector<uint16_t> cpu(width * 16), gpu(cpu.size());
        require(motioncam::raw::Decode(cpu.data(), width, 16, encoded.data(), encoded.size()) ==
                    cpu.size(),
                "CPU fixture decode failed");
        rawpreamp::VulkanDecodedJpegImage image;
        require(rawpreamp::mcrawDecodeVulkan(encoded.data(), encoded.size(), width, 16, image),
                "MCRAW GPU decode failed");
        require(rawpreamp::downloadVulkanDecodedJpegImage(image, gpu.data(), gpu.size()),
                "MCRAW download failed");
        require(cpu == gpu, "MCRAW GPU/CPU block decode differs");
        if (width != 256) {
            auto header = dng(std::vector<uint16_t>(cpu.size(), 0xdead), false, width, 16);
            const rawpreamp::VulkanGainBakeBufferSource input{image.gpu, image.buffer,
                                                              image.deviceLease};
            std::vector<std::vector<uint8_t>> tiles;
            require(rawpreamp::DNGDecoder::bakeLogStripVulkan(
                        header, rawpreamp::LogTransformMode::Disabled, 65535, false, 7, nullptr,
                        &tiles, rawpreamp::LogCurve::LOG60, &input) &&
                        !tiles.empty(),
                    "Right/bottom edge GPU JPEG missing");
            const auto shape = rawpreamp::jpeg92TileShape(1);
            require(rawpreamp::DNGDecoder::attachLosslessJPEGTiles(header, tiles, shape.width,
                                                                   shape.height),
                    "Partial JPEG tile attachment failed");
            rawpreamp::DecodedDNGImage roundtrip;
            require(
                rawpreamp::DNGDecoder::decodeImage(std::move(header), roundtrip, false, false) &&
                    roundtrip.samples == cpu,
                "Partial edge tile roundtrip differs");
        }
    }
    // An earlier output survives reuse of the compressed input/descriptor buffers.
    std::vector<uint16_t> cpu(256 * 16), gpu(cpu.size());
    motioncam::raw::Decode(cpu.data(), 256, 16, encoded.data(), encoded.size());
    require(rawpreamp::downloadVulkanDecodedJpegImage(first, gpu.data(), gpu.size()),
            "Retained MCRAW download failed");
    require(cpu == gpu, "Retained MCRAW output was overwritten");
    const rawpreamp::VulkanGainBakeBufferSource source{first.gpu, first.buffer, first.deviceLease};
    {
        // The single-scan encoder declines tiny frames. A resident source must
        // report failure rather than downloading its packed pixels to retry.
        auto header = dng(std::vector<uint16_t>(cpu.size(), 0xdead), false);
        std::vector<uint8_t> jpeg;
        require(!rawpreamp::DNGDecoder::bakeLogStripVulkan(
                    header, rawpreamp::LogTransformMode::Disabled, 65535, false, 7, &jpeg, nullptr,
                    rawpreamp::LogCurve::LOG60, &source),
                "Resident encode failure silently downloaded an intermediate image");
    }

    for (bool gain : {false, true}) {
        for (auto mode :
             {rawpreamp::LogTransformMode::Disabled, rawpreamp::LogTransformMode::ReduceBy4Bit}) {
            auto reference = dng(cpu, gain);
            auto resident = dng(std::vector<uint16_t>(cpu.size(), 0xdead), gain);
            bool fused = false;
            if (gain) {
                require(rawpreamp::DNGDecoder::bakeGainMaps(reference, false, false, false, false,
                                                            2, std::nullopt, mode, 65535, false,
                                                            &fused, true, 7, false),
                        "CPU gain bake failed");
                if (fused)
                    require(rawpreamp::DNGDecoder::applyLogTransform(reference, mode, 65535, false,
                                                                     true),
                            "CPU log metadata failed");
                fused = false;
                require(rawpreamp::DNGDecoder::bakeGainMaps(
                            resident, false, false, false, false, 2, std::nullopt, mode, 65535,
                            false, &fused, true, 7, true, nullptr, nullptr, nullptr,
                            rawpreamp::LogCurve::LOG60, &source),
                        "Resident gain bake failed");
                if (fused)
                    require(rawpreamp::DNGDecoder::applyLogTransform(resident, mode, 65535, false,
                                                                     true),
                            "GPU log metadata failed");
            } else {
                if (mode != rawpreamp::LogTransformMode::Disabled)
                    require(rawpreamp::DNGDecoder::applyLogTransform(reference, mode, 65535, false,
                                                                     false, 7),
                            "CPU LOG failed");
                require(rawpreamp::DNGDecoder::bakeLogStripVulkan(
                            resident, mode, 65535, false, 7, nullptr, nullptr,
                            rawpreamp::LogCurve::LOG60, &source),
                        "Resident pack/LOG failed");
            }
            rawpreamp::DecodedDNGImage expected, actual;
            require(
                rawpreamp::DNGDecoder::decodeImage(std::move(reference), expected, false, false) &&
                    rawpreamp::DNGDecoder::decodeImage(std::move(resident), actual, false, false),
                "Processed DNG decode failed");
            require(expected.samples.size() == actual.samples.size(), "Processed DNG size differs");
            unsigned maxError = 0;
            for (size_t i = 0; i < expected.samples.size(); ++i)
                maxError =
                    std::max(maxError,
                             unsigned(std::abs(int(expected.samples[i]) - int(actual.samples[i]))));
            std::fprintf(stderr, "MCRAW resident processing gain=%d log=%d max_error=%u\n", gain,
                         int(mode), maxError);
            require(maxError <= 1, "Resident gain/LOG pixels differ from CPU");
            for (bool tiled : {false, true}) {
                if (!tiled && cpu.size() < 1000000u)
                    continue; // full-frame encoder threshold
                auto compressed = dng(std::vector<uint16_t>(cpu.size(), 0xdead), gain);
                std::vector<uint8_t> jpeg;
                std::vector<std::vector<uint8_t>> tiles;
                bool fusedJpeg = false;
                const bool encoded =
                    gain ? rawpreamp::DNGDecoder::bakeGainMaps(
                               compressed, false, false, false, false, 2, std::nullopt, mode, 65535,
                               false, &fusedJpeg, true, 7, true, nullptr, tiled ? nullptr : &jpeg,
                               tiled ? &tiles : nullptr, rawpreamp::LogCurve::LOG60, &source)
                         : rawpreamp::DNGDecoder::bakeLogStripVulkan(
                               compressed, mode, 65535, false, 7, tiled ? nullptr : &jpeg,
                               tiled ? &tiles : nullptr, rawpreamp::LogCurve::LOG60, &source);
                require(encoded && (tiled ? !tiles.empty() : !jpeg.empty()),
                        "Resident GPU JPEG missing");
                if (fusedJpeg)
                    require(rawpreamp::DNGDecoder::applyLogTransform(compressed, mode, 65535, false,
                                                                     true),
                            "JPEG log metadata failed");
                const auto shape = rawpreamp::jpeg92TileShape(1);
                require(tiled ? rawpreamp::DNGDecoder::attachLosslessJPEGTiles(
                                    compressed, tiles, shape.width, shape.height)
                              : rawpreamp::DNGDecoder::attachLosslessJPEG(compressed, jpeg),
                        "GPU JPEG attachment failed");
                rawpreamp::DecodedDNGImage roundtrip;
                require(rawpreamp::DNGDecoder::decodeImage(std::move(compressed), roundtrip, false,
                                                           false),
                        "GPU JPEG DNG decode failed");
                if (roundtrip.samples != actual.samples) {
                    unsigned error = 0;
                    size_t differing = 0;
                    for (size_t i = 0;
                         i < std::min(roundtrip.samples.size(), actual.samples.size()); ++i) {
                        error = std::max(error, unsigned(std::abs(int(roundtrip.samples[i]) -
                                                                  int(actual.samples[i]))));
                        if (roundtrip.samples[i] != actual.samples[i] && differing++ < 4)
                            std::fprintf(stderr, "JPEG mismatch at %zu: %u != %u\n", i,
                                         roundtrip.samples[i], actual.samples[i]);
                    }
                    std::fprintf(stderr,
                                 "JPEG mismatch gain=%d mode=%d tiled=%d max_error=%u "
                                 "differing=%zu bits=%u\n",
                                 gain, int(mode), tiled, error, differing,
                                 roundtrip.storedBitsPerSample);
                }
                require(roundtrip.samples == actual.samples,
                        "Resident JPEG roundtrip changed pixels");
            }
        }
    }
    for (bool gain : {false, true}) {
        for (int scale : {1, 2}) {
            auto header = dng(std::vector<uint16_t>(cpu.size(), 0xdead), gain);
            rawpreamp::DecodedDNGImage image;
            require(rawpreamp::DNGDecoder::getImageLayout(header, image.layout) &&
                        rawpreamp::DNGDecoder::getColorMetadata(header, image.metadata),
                    "Preview metadata failed");
            rawpreamp::DNGDecoder::getGainMaps(header, 2, image.opcodeList2);
            image.storedBitsPerSample = 16;
            image.gpuImage = std::make_shared<rawpreamp::DNGGpuImage>();
            image.gpuImage->gpu = first.gpu;
            image.gpuImage->buffer = first.buffer;
            image.gpuImage->width = 256;
            image.gpuImage->height = 16;
            image.gpuImage->channels = 1;
            image.gpuImage->lease = first.lease;
            // No downloader: any accidental CPU fallback must fail this test.
            rawpreamp::RenderSettings settings;
            settings.streamingPreview = true;
            settings.options = static_cast<rawpreamp::FileRenderOptions>(
                rawpreamp::RENDER_OPT_DRAFT | rawpreamp::RENDER_OPT_APPLY_VIGNETTE_CORRECTION);
            settings.draftScale = scale;
            rawpreamp::PreviewFrame preview;
            require(rawpreamp::DNGDecoder::decodePreview(std::move(image), settings, preview, true,
                                                         false),
                    "Resident gallery preview failed");
            require(preview.gallerySource.buffer && preview.rgb.empty() &&
                        preview.galleryCfa.empty() && preview.galleryCfaBytes.empty() &&
                        preview.width == 256u / scale && preview.height == 16u / scale,
                    "Preview left GPU storage");
        }
    }
    rawpreamp::VulkanDecodedJpegImage invalid;
    require(!rawpreamp::mcrawDecodeVulkan(encoded.data(), 15, 256, 16, invalid),
            "Truncated MCRAW accepted");
    require(!rawpreamp::mcrawDecodeVulkan(encoded.data(), encoded.size(), 257, 16, invalid),
            "Odd MCRAW width accepted");
    require(!rawpreamp::mcrawDecodeVulkan(encoded.data(), encoded.size(), 256, 15, invalid),
            "Invalid MCRAW height accepted");
    auto corrupt = encoded;
    put32(corrupt, 8, 0xfffffff0);
    require(!rawpreamp::mcrawDecodeVulkan(corrupt.data(), corrupt.size(), 256, 16, invalid),
            "Invalid metadata offset accepted");
    // Optional real recording: compare several complete frames to the SIMD decoder.
    if (const char* path = std::getenv("RAWPREAMP_TEST_MCRAW")) {
        motioncam::Decoder decoder(path);
        for (size_t i = 0; i < std::min(size_t(3), decoder.getFrames().size()); ++i) {
            std::vector<uint8_t> bytes, reference;
            nlohmann::json metadata;
            decoder.loadEncodedFrame(decoder.getFrames()[i], bytes, metadata);
            const uint32_t w = metadata.at("width"), h = metadata.at("height");
            require(metadata.at("compressionType") == 7, "Real fixture needs compression 7");
            rawpreamp::VulkanDecodedJpegImage image;
            require(rawpreamp::mcrawDecodeVulkan(bytes.data(), bytes.size(), w, h, image),
                    "Real MCRAW GPU decode failed");
            decoder.loadFrame(decoder.getFrames()[i], reference, metadata);
            std::vector<uint16_t> output(size_t(w) * h);
            require(rawpreamp::downloadVulkanDecodedJpegImage(image, output.data(), output.size()),
                    "Real MCRAW download failed");
            require(reference.size() == output.size() * 2 &&
                        !std::memcmp(reference.data(), output.data(), reference.size()),
                    "Real MCRAW GPU/CPU mismatch");
            std::fprintf(stderr, "MCRAW frame %zu: %ux%u GPU matches CPU exactly\n", i, w, h);
            if (i == 0)
                for (bool tiled : {false, true}) {
                    auto header = dng(std::vector<uint16_t>(output.size(), 0xdead), false, w, h);
                    const rawpreamp::VulkanGainBakeBufferSource input{image.gpu, image.buffer,
                                                                      image.deviceLease};
                    std::vector<uint8_t> jpeg;
                    std::vector<std::vector<uint8_t>> tiles;
                    require(rawpreamp::DNGDecoder::bakeLogStripVulkan(
                                header, rawpreamp::LogTransformMode::Disabled, 65535, false, 7,
                                tiled ? nullptr : &jpeg, tiled ? &tiles : nullptr,
                                rawpreamp::LogCurve::LOG60, &input),
                            "Real GPU pack failed");
                    require(tiled ? !tiles.empty() : !jpeg.empty(), "Real GPU JPEG missing");
                    const auto shape = rawpreamp::jpeg92TileShape(1);
                    require(tiled ? rawpreamp::DNGDecoder::attachLosslessJPEGTiles(
                                        header, tiles, shape.width, shape.height)
                                  : rawpreamp::DNGDecoder::attachLosslessJPEG(header, jpeg),
                            "Real JPEG DNG failed");
                    rawpreamp::DecodedDNGImage roundtrip;
                    require(rawpreamp::DNGDecoder::decodeImage(std::move(header), roundtrip, false,
                                                               false) &&
                                roundtrip.samples == output,
                            "Real GPU JPEG roundtrip differs");
                    std::fprintf(stderr, "MCRAW real GPU JPEG %s roundtrip matches CPU exactly\n",
                                 tiled ? "tiled" : "strip");
                }
        }
    }
}

#include "VulkanJpegBenchmark.h"
#include "DebugInstrumentation.h"
#if RAWPREAMP_DEBUG_INSTRUMENTATION &&                                         \
    defined(RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO)
#include "VulkanGainBake.h"
#include "VulkanJpegContext.h"
#include "VulkanJpegDecode.h"
#include "VulkanJpegFullEncode.h"
#include "liblj92/lj92.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <vector>

namespace rawpreamp {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
void check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.size() % 2
             ? values[values.size() / 2]
             : (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
}
struct DiagnosticLog {
  pl_log log;
  pl_gpu gpu;
  pl_log_params previous;
  std::vector<std::string> &errors;
  std::mutex mutex;
  DiagnosticLog(pl_log logger, pl_gpu device,
                std::vector<std::string> &messages)
      : log(logger), gpu(device), errors(messages) {
    pl_log_params params{};
    params.log_level = PL_LOG_ERR;
    params.log_priv = this;
    params.log_cb = [](void *pointer, pl_log_level, const char *text) {
      auto &self = *static_cast<DiagnosticLog *>(pointer);
      std::lock_guard lock(self.mutex);
      if (self.errors.size() < 32)
        self.errors.emplace_back(text);
    };
    previous = pl_log_update(log, &params);
  }
  ~DiagnosticLog() {
    pl_gpu_finish(gpu);
    pl_log_update(log, &previous);
  }
};
struct TiledSource {
  std::vector<uint8_t> bytes;
  std::vector<VulkanJpegTile> tiles;
  std::vector<uint16_t> expected;
  uint32_t width = 0, height = 0, channels = 0, bits = 0, tw = 0, th = 0,
           jpegComponents = 0;
  explicit TiledSource(std::filesystem::path path) {
    if (std::filesystem::is_directory(path)) {
      std::vector<std::filesystem::path> files;
      for (const auto &entry : std::filesystem::directory_iterator(path))
        if (entry.is_regular_file() && entry.path().extension() == ".dng")
          files.push_back(entry.path());
      check(!files.empty(), "No DNG frames in benchmark directory");
      std::sort(files.begin(), files.end());
      path = files.front();
    }
    std::ifstream input(path, std::ios::binary);
    check(bool(input), "Could not open benchmark DNG");
    bytes.assign(std::istreambuf_iterator<char>(input), {});
    check(bytes.size() >= 8, "Truncated TIFF header");
    const bool little = bytes[0] == 'I' && bytes[1] == 'I';
    check(little || (bytes[0] == 'M' && bytes[1] == 'M'),
          "Invalid TIFF byte order");
    auto u16 = [&](size_t at) {
      check(at <= bytes.size() && bytes.size() - at >= 2,
            "Truncated TIFF short");
      return little ? uint32_t(bytes[at]) | (uint32_t(bytes[at + 1]) << 8)
                    : uint32_t(bytes[at + 1]) | (uint32_t(bytes[at]) << 8);
    };
    auto u32 = [&](size_t at) {
      return little ? u16(at) | (u16(at + 2) << 16)
                    : u16(at + 2) | (u16(at) << 16);
    };
    check(u16(2) == 42, "Benchmark requires classic TIFF");
    std::vector<uint32_t> queue{u32(4)};
    std::set<uint32_t> visited;
    std::vector<uint32_t> offsets, lengths;
    for (size_t n = 0; n < queue.size() && n < 32; ++n) {
      const uint32_t ifd = queue[n];
      if (!ifd || !visited.insert(ifd).second)
        continue;
      const uint32_t count = u16(ifd);
      check(ifd + 2ull + uint64_t(count) * 12 + 4 <= bytes.size(),
            "Truncated TIFF directory");
      auto values = [&](uint16_t tag) {
        std::vector<uint32_t> result;
        for (uint32_t i = 0; i < count; ++i) {
          size_t entry = ifd + 2ull + i * 12u;
          if (u16(entry) != tag)
            continue;
          uint32_t type = u16(entry + 2), size = u32(entry + 4),
                   unit = type == 3 ? 2u : 4u;
          check(type == 3 || type == 4 || type == 13,
                "Unsupported TIFF benchmark tag type");
          size_t at = uint64_t(size) * unit <= 4 ? entry + 8 : u32(entry + 8);
          check(at <= bytes.size() &&
                    uint64_t(size) * unit <= bytes.size() - at,
                "Invalid TIFF tag extent");
          for (uint32_t j = 0; j < size; ++j)
            result.push_back(unit == 2 ? u16(at + j * unit)
                                       : u32(at + j * unit));
          break;
        }
        return result;
      };
      auto scalar = [&](uint16_t tag) {
        auto result = values(tag);
        return result.empty() ? 0u : result.front();
      };
      auto children = values(330);
      queue.insert(queue.end(), children.begin(), children.end());
      queue.push_back(u32(ifd + 2ull + count * 12u));
      if (scalar(259) != 7 || !scalar(322))
        continue;
      width = scalar(256);
      height = scalar(257);
      channels = scalar(277);
      bits = scalar(258);
      tw = scalar(322);
      th = scalar(323);
      offsets = values(324);
      lengths = values(325);
      break;
    }
    check(width && height && tw && th && (channels == 1 || channels == 3) &&
              bits >= 8 && bits <= 16,
          "No supported tiled lossless JPEG image");
    check(uint64_t(width) * height <= uint64_t(64u * 1024u * 1024u) / channels,
          "Benchmark image exceeds resident sample limit");
    check(width % tw == 0 && height % th == 0,
          "Resident benchmark requires complete tiles");
    check(offsets.size() == lengths.size() &&
              offsets.size() == uint64_t(width / tw) * (height / th),
          "Invalid tile count");
    expected.resize(size_t(width) * height * channels);
    for (size_t i = 0; i < offsets.size(); ++i) {
      check(offsets[i] <= bytes.size() &&
                lengths[i] <= bytes.size() - offsets[i],
            "Invalid JPEG tile extent");
      auto *data = bytes.data() + offsets[i];
      lj92 decoder = nullptr;
      int w = 0, h = 0, b = 0, c = 0;
      check(lj92_open(&decoder, data, lengths[i], &w, &h, &b, &c) ==
                LJ92_ERROR_NONE,
            "CPU JPEG open failed");
      std::vector<uint16_t> pixels(size_t(tw) * th * channels);
      bool valid =
          w * c == int(tw * channels) && h == int(th) && b == int(bits) &&
          lj92_decode(decoder, pixels.data(), tw * channels, 0, nullptr, 0) ==
              LJ92_ERROR_NONE;
      lj92_close(decoder);
      check(valid, "CPU JPEG reference decode failed");
      if (i == 0)
        jpegComponents = uint32_t(c);
      uint32_t x = uint32_t(i % (width / tw)) * tw,
               y = uint32_t(i / (width / tw)) * th;
      for (uint32_t row = 0; row < th; ++row)
        std::copy_n(pixels.data() + size_t(row) * tw * channels, tw * channels,
                    expected.data() + (size_t(y + row) * width + x) * channels);
      tiles.push_back({data, lengths[i], tw, th, bits, channels});
    }
  }
};
struct Comparison {
  pl_gpu gpu;
  const std::vector<uint16_t> &cpuPixels;
  pl_buf expected = nullptr, result = nullptr, valid = nullptr;
  pl_pass pass = nullptr;
  uint32_t samples;
  Comparison(pl_gpu device, const std::vector<uint16_t> &pixels)
      : gpu(device), cpuPixels(pixels), samples(uint32_t(pixels.size())) {
    constexpr char shader[] = R"glsl(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer A { uint words[]; } a;
layout(std430,binding=1) readonly buffer B { uint words[]; } b;
layout(std430,binding=2) readonly buffer Valid { uint good; } status;
layout(std430,binding=3) buffer Result { uint different; uint maximum; uint invalid; } result;
layout(push_constant) uniform Size { uint words; } p;
void main() {
    uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*gl_NumWorkGroups.x*256u;
    if(i>=p.words) return;
    if(i==0u && status.good!=1u) atomicOr(result.invalid,1u);
    uvec2 x=uvec2(a.words[i]&65535u,a.words[i]>>16u),y=uvec2(b.words[i]&65535u,b.words[i]>>16u);
    uvec2 delta=max(x,y)-min(x,y);
    if(any(notEqual(x,y))) atomicAdd(result.different,uint(delta.x!=0u)+uint(delta.y!=0u));
    atomicMax(result.maximum,max(delta.x,delta.y));
}
)glsl";
    pl_buf_params p{};
    p.size = pixels.size() * 2u;
    p.storable = true;
    p.initial_data = pixels.data();
    expected = pl_buf_create(gpu, &p);
    uint32_t one = 1;
    p = {};
    p.size = 4;
    p.storable = true;
    p.initial_data = &one;
    valid = pl_buf_create(gpu, &p);
    p = {};
    p.size = 12;
    p.storable = p.host_readable = p.host_writable = true;
    result = pl_buf_create(gpu, &p);
    pl_desc desc[] = {
        {"a", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
        {"b", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
        {"status", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_READONLY},
        {"result", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_READWRITE}};
    pl_pass_params pp{};
    pp.type = PL_PASS_COMPUTE;
    pp.glsl_shader = shader;
    pp.descriptors = desc;
    pp.num_descriptors = 4;
    pp.push_constants_size = 4;
    pass = pl_pass_create(gpu, &pp);
  }
  ~Comparison() {
    pl_pass_destroy(gpu, &pass);
    pl_buf_destroy(gpu, &expected);
    pl_buf_destroy(gpu, &result);
    pl_buf_destroy(gpu, &valid);
  }
  void equal(const void *reference, const VulkanDecodedJpegImage &image) {
    check(expected && result && valid && pass,
          "GPU comparison allocation failed");
    uint32_t words = samples / 2u, zero[3]{};
    check(image.gpu == gpu &&
              size_t(image.width) * image.height * image.channels == samples,
          "Roundtrip image shape changed");
    pl_buf_write(gpu, result, 0, zero, sizeof(zero));
    pl_desc_binding bindings[4]{};
    bindings[0].object = reference;
    bindings[1].object = image.buffer;
    bindings[2].object =
        image.validationBuffer ? image.validationBuffer : valid;
    bindings[3].object = result;
    pl_pass_run_params run{};
    run.pass = pass;
    run.desc_bindings = bindings;
    run.push_constants = &words;
    uint32_t groups = (words + 255u) / 256u;
    run.compute_groups[0] = std::min(groups, gpu->limits.max_dispatch[0]);
    run.compute_groups[1] =
        (groups + run.compute_groups[0] - 1u) / run.compute_groups[0];
    run.compute_groups[2] = 1;
    pl_pass_run(gpu, &run);
    uint32_t status[3]{};
    check(pl_buf_read(gpu, result, 0, status, sizeof(status)),
          "Could not read JPEG validation status");
    if (status[0] || status[1] || status[2]) {
      std::string values;
      pl_buf_params params{};
      params.size = 64;
      params.host_readable = true;
      pl_buf host = pl_buf_create(gpu, &params);
      if (host) {
        uint16_t pixels[32]{};
        pl_buf_copy(gpu, host, 0, static_cast<pl_buf>(image.buffer), 0, 64);
        if (pl_buf_read(gpu, host, 0, pixels, 64))
          for (auto v : pixels)
            values += " " + std::to_string(v);
        pl_buf_destroy(gpu, &host);
      }
      throw std::runtime_error(
          "JPEG roundtrip differs=" + std::to_string(status[0]) + " max=" +
          std::to_string(status[1]) + " invalid=" + std::to_string(status[2]) +
          " bits=" + std::to_string(image.bits) +
          " channels=" + std::to_string(image.channels) + " first=" + values);
    }
  }
};
// Validate actual JPEG headers as well as the resident metadata shortcut.
// Only compressed bytes are downloaded, outside the timed benchmark iterations.
void validateEncodedTile(pl_gpu gpu, pl_buf jpeg, uint32_t slotWords,
                         uint32_t width, uint32_t height, uint32_t channels,
                         uint32_t bits, uint32_t fullWidth,
                         const std::vector<uint16_t> &reference) {
  pl_buf_params params{};
  params.size = size_t(slotWords) * 4u;
  params.host_readable = true;
  pl_buf host = pl_buf_create(gpu, &params);
  check(bool(host), "JPEG staging failed");
  std::vector<uint8_t> bytes(params.size);
  pl_buf_copy(gpu, host, 0, jpeg, 0, bytes.size());
  const bool read = pl_buf_read(gpu, host, 0, bytes.data(), bytes.size());
  pl_buf_destroy(gpu, &host);
  check(read, "Could not read encoded JPEG fixture");
  lj92 decoder = nullptr;
  int w = 0, h = 0, b = 0, c = 0;
  check(lj92_open(&decoder, bytes.data(), int(bytes.size()), &w, &h, &b, &c) ==
            LJ92_ERROR_NONE,
        "CPU could not parse GPU encoded JPEG");
  const bool shape = w == int(width) && h == int(height) && b == int(bits) &&
                     c == int(channels);
  std::vector<uint16_t> pixels(size_t(width) * height * channels);
  const bool decoded =
      shape && lj92_decode(decoder, pixels.data(), width * channels, 0, nullptr,
                           0) == LJ92_ERROR_NONE;
  lj92_close(decoder);
  check(decoded, "CPU could not decode GPU encoded JPEG fixture");
  for (uint32_t y = 0; y < height; ++y)
    check(std::equal(pixels.begin() + size_t(y) * width * channels,
                     pixels.begin() + size_t(y + 1u) * width * channels,
                     reference.begin() + size_t(y) * fullWidth * channels),
          "CPU decoded GPU JPEG differs from original stored codes");
}

struct RoundtripTimes {
  double encode = 0, decode = 0;
};
RoundtripTimes roundtrip(const VulkanDecodedJpegImage &source, uint32_t bits,
                         uint32_t tw, uint32_t th, Comparison &comparison,
                         std::vector<uint8_t> &placeholder,
                         bool checkHeader = false) {
  RoundtripTimes times;
  auto begin = Clock::now();
  bool compared = false;
  const char *stage = "pack";
  auto encode = [&](const VulkanPackedImage &packed) {
    stage = "encode";
    try {
      return jpegEncodeTiledVulkanPackedBufferResident(
          packed.gpu, packed.buffer, packed.width, packed.height,
          packed.channels, packed.bits, packed.wordsPerRow, tw, th,
          [&](const void *gpu, const void *bytes, const void *lengths,
              const void *table, uint32_t width, uint32_t height,
              uint32_t channels, uint32_t tileWidth, uint32_t tileHeight,
              uint32_t count, uint32_t slotWords) {
            if (checkHeader)
              validateEncodedTile(static_cast<pl_gpu>(gpu),
                                  static_cast<pl_buf>(bytes), slotWords,
                                  tileWidth, tileHeight, channels, bits,
                                  source.width, comparison.cpuPixels);
            stage = "resident decode";
            pl_gpu_finish(static_cast<pl_gpu>(gpu));
            auto encoded = Clock::now();
            times.encode = ms(begin, encoded);
            return jpegDecodeResidentTiledImageVulkan(
                gpu, bytes, lengths, table, width, height, channels, tileWidth,
                tileHeight, count, slotWords,
                [&](const VulkanDecodedJpegImage &decoded) {
                  stage = "compare";
                  pl_gpu_finish(static_cast<pl_gpu>(gpu));
                  times.decode = ms(encoded, Clock::now());
                  check(decoded.bits == bits,
                        "JPEG roundtrip changed sample precision");
                  comparison.equal(source.buffer, decoded);
                  compared = true;
                  return true;
                },
                bits);
          });
    } catch (const std::exception &e) {
      auto gpu = static_cast<pl_gpu>(packed.gpu);
      pl_buf_params params{};
      params.size = 32;
      params.host_readable = true;
      pl_buf host = pl_buf_create(gpu, &params);
      std::string words;
      if (host) {
        uint32_t data[8]{};
        pl_buf_copy(gpu, host, 0, static_cast<pl_buf>(packed.buffer), 0, 32);
        if (pl_buf_read(gpu, host, 0, data, 32))
          for (auto v : data)
            words += " " + std::to_string(v);
        pl_buf_destroy(gpu, &host);
      }
      throw std::runtime_error(std::string(e.what()) + " packed=" + words);
    }
  };
  bool ok = false;
  if (bits == 16)
    ok = encode({source.gpu, source.buffer, source.width, source.height,
                 source.channels, 16, source.wordsPerRow});
  else {
    const std::vector<float> identity{1};
    VulkanRgbGainMap map{1, 1, 1, 0, 0, source.width, source.height, 1,
                         1, 0, 0, 1, 1, &identity};
    VulkanGainBakeBufferSource input{
        source.gpu, source.buffer, source.deviceLease, source.validationBuffer};
    VulkanGainBakeParams p;
    p.width = source.width;
    p.height = source.height;
    p.channels = source.channels;
    p.outputBits = bits;
    p.scale.fill(1);
    p.destinationWhite = float((1u << bits) - 1u);
    p.bufferSource = &input;
    p.map = map;
    p.cfaMaps[0] = map;
    p.cfaMapCount = 1;
    p.consumePackedOutput = encode;
    p.requirePackedOutputConsumer = true;
    ok = source.channels == 1
             ? bakeCfaGainVulkan(nullptr, p, placeholder.data(),
                                 placeholder.size())
             : bakeRgbGainVulkan(nullptr, p, placeholder.data(),
                                 placeholder.size());
  }
  if (!ok || !compared)
    throw std::runtime_error(std::string("GPU JPEG roundtrip failed at ") +
                             stage +
                             " channels=" + std::to_string(source.channels) +
                             " bits=" + std::to_string(bits));
  return times;
}
void validatePrecisions(pl_gpu gpu) {
  for (uint32_t channels : {1u, 3u})
    for (uint32_t bits : {8u, 12u, 16u}) {
      std::vector<uint16_t> samples(64u * 32u * channels);
      for (size_t i = 0; i < samples.size(); ++i)
        samples[i] =
            uint16_t((i * 1037u + (i / 64u) * 17u) & ((1u << bits) - 1u));
      Comparison compare(gpu, samples);
      VulkanDecodedJpegImage source{
          gpu, compare.expected, nullptr, 64,
          32,  channels,         bits,    64 * channels / 2u};
      std::vector<uint8_t> placeholder(((64u * channels * bits + 7u) / 8u) *
                                       32u);
      roundtrip(source, bits, 16, 16, compare, placeholder, true);
    }
  std::vector<uint16_t> samples(512u * 16u);
  for (size_t i = 0; i < samples.size(); ++i)
    samples[i] = uint16_t(i * 1037u);
  Comparison compare(gpu, samples);
  VulkanDecodedJpegImage source{gpu, compare.expected, nullptr, 512, 16, 1, 16,
                                256};
  std::vector<uint8_t> placeholder(samples.size() * 2u);
  roundtrip(source, 16, 256, 16, compare, placeholder, true);
}
} // namespace
int runVulkanJpegBenchmark(const std::string &configuration) {
  Json result;
  std::vector<std::string> gpuErrors;
  std::string report = configuration + ".result.json";
  try {
    std::ifstream input(configuration);
    Json config;
    input >> config;
    report = config.value("report", report);
    TiledSource source(config.at("source").get<std::string>());
    unsigned iterations = config.value("iterations", 7u);
    check(iterations >= 1 && iterations <= 100, "Invalid benchmark iterations");
    auto &context = jpegContext();
    std::lock_guard lock(context.mutex);
    check(context.init(), "Vulkan unavailable");
    auto gpu = context.vulkan->gpu;
    DiagnosticLog log(context.log, gpu, gpuErrors);
    Comparison comparison(gpu, source.expected);
    if (config.value("validate_precisions", true))
      validatePrecisions(gpu);
    const uint32_t tw =
        config.value("encode_tile_width", std::min(source.tw, 128u));
    const uint32_t th =
        config.value("encode_tile_height", std::min(source.th, 128u));
    check(tw && th && source.width % tw == 0 && source.height % th == 0,
          "Invalid roundtrip tile shape");
    std::vector<uint8_t> placeholder(
        ((size_t(source.width) * source.channels * source.bits + 7u) / 8u) *
        source.height);
    std::vector<double> decode, encode, redecode, total;
    VulkanJpegContiguousSource contiguous{source.bytes.data(),
                                          source.bytes.size()};
    for (unsigned i = 0; i < iterations + 2u; ++i) {
      auto begin = Clock::now();
      VulkanDecodedJpegImage original;
      check(jpegDecodeTiledImageVulkanRetained(
                source.tiles, source.width, source.height, source.tw, source.th,
                original, &contiguous),
            "Original JPEG GPU decode failed");
      double decoded = ms(begin, Clock::now());
      auto times =
          roundtrip(original, source.bits, tw, th, comparison, placeholder);
      comparison.equal(comparison.expected, original);
      if (i >= 2) {
        decode.push_back(decoded);
        encode.push_back(times.encode);
        redecode.push_back(times.decode);
        total.push_back(decoded + times.encode + times.decode);
      }
    }
    result = {{"status", "PASS"},
              {"width", source.width},
              {"height", source.height},
              {"channels", source.channels},
              {"bits", source.bits},
              {"source_jpeg_components", source.jpegComponents},
              {"tiles", source.tiles.size()},
              {"tile_width", source.tw},
              {"tile_height", source.th},
              {"encode_tile_width", tw},
              {"encode_tile_height", th},
              {"iterations", iterations},
              {"warmup_iterations", 2},
              {"pixel_codes_exact", true},
              {"timing", "synchronized wall milliseconds; validation excluded; "
                         "encode includes GPU packing below 16 bits"},
              {"median_ms",
               {{"source_decode", median(decode)},
                {"pack_encode", median(encode)},
                {"resident_redecode", median(redecode)},
                {"roundtrip", median(total)}}},
              {"source_decode_ms", decode},
              {"pack_encode_ms", encode},
              {"resident_redecode_ms", redecode},
              {"roundtrip_ms", total}};
  } catch (const std::exception &error) {
    result = {{"status", "FAIL"}, {"error", error.what()}};
  }
  if (!gpuErrors.empty())
    result["gpu_errors"] = gpuErrors;
  std::ofstream output(report);
  output << result.dump(2) << '\n';
  output.flush();
  return output && result.value("status", "") == "PASS" ? 0 : 1;
}
} // namespace rawpreamp
#else
namespace rawpreamp {
int runVulkanJpegBenchmark(const std::string &) { return 2; }
} // namespace rawpreamp
#endif

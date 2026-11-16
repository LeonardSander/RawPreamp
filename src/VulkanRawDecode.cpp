#include "VulkanRawDecode.h"
#include "DebugInstrumentation.h"
#include <motioncam/Decoder.hpp>
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
#include "VulkanJpegContext.h"
#include <libplacebo/gpu.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <spdlog/spdlog.h>

namespace rawpreamp {
namespace {
constexpr uint32_t lengths[] = {0,  8,  16,  24,  32,  40,  48,  64, 64,
                                80, 80, 128, 128, 128, 128, 128, 128};
uint32_t read32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
// Scalar equivalent of MotionCam's SIMD block decoder, also used to unpack
// the small, independently compressed bit-width/reference tables on the CPU.
uint16_t sample(const uint8_t* p, uint32_t bits, uint32_t i) {
    const uint32_t lane = i % 8, group = i / 8;
    auto b = [&](uint32_t n) { return uint32_t(p[n * 8 + lane]); };
    switch (bits) {
    case 0:
        return 0;
    case 1:
        return (b(0) >> group) & 1;
    case 2:
        return (b(group / 4) >> ((group % 4) * 2)) & 3;
    case 3:
        switch (group) {
        case 0:
            return b(0) & 7;
        case 1:
            return (b(0) >> 3) & 7;
        case 2:
            return (b(0) >> 6) | (((b(2) >> 6) & 1) << 2);
        case 3:
            return b(1) & 7;
        case 4:
            return (b(1) >> 3) & 7;
        case 5:
            return (b(1) >> 6) | (((b(2) >> 7) & 1) << 2);
        case 6:
            return b(2) & 7;
        default:
            return (b(2) >> 3) & 7;
        }
    case 4:
        return (b(group / 2) >> ((group % 2) * 4)) & 15;
    case 5:
        if (group < 5)
            return b(group) & 31;
        if (group == 5)
            return (b(0) >> 5) | (((b(3) >> 5) & 3) << 3);
        if (group == 6)
            return (b(1) >> 5) | (((b(4) >> 5) & 3) << 3);
        return (b(2) >> 5) | ((b(3) >> 7) << 3) | ((b(4) >> 7) << 4);
    case 6:
        if (group < 6)
            return b(group) & 63;
        return (b((group - 6) * 3) >> 6) | (b((group - 6) * 3 + 1) >> 6) << 2 |
               (b((group - 6) * 3 + 2) >> 6) << 4;
    case 7:
    case 8:
        return b(group);
    case 9:
    case 10:
        return b(group + group / 4) | (((b((group / 4) * 5 + 4) >> ((group % 4) * 2)) & 3) << 8);
    default:
        return uint16_t(p[i * 2]) | uint16_t(p[i * 2 + 1]) << 8;
    }
}
bool metadata(const uint8_t* bytes, size_t size, uint32_t offset, size_t needed,
              std::vector<uint16_t>& values) {
    if (offset > size || size - offset < 4)
        return false;
    const uint32_t count = read32(bytes + offset);
    if (count < needed || count > needed + 63 || (count % 64))
        return false;
    size_t at = size_t(offset) + 4;
    values.resize(count);
    for (size_t i = 0; i < count; i += 64) {
        if (at > size || size - at < 2)
            return false;
        const uint32_t bits = bytes[at] >> 4;
        const uint32_t reference = (uint32_t(bytes[at] & 15) << 8) | bytes[at + 1];
        at += 2;
        if (lengths[bits] > size - at)
            return false;
        for (uint32_t j = 0; j < 64; ++j)
            values[i + j] = uint16_t(sample(bytes + at, bits, j) + reference);
        at += lengths[bits];
    }
    return true;
}
constexpr char shader[] = R"glsl(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Stream { uint words[]; } src;
layout(std430,binding=1) readonly buffer Blocks { uint words[]; } blocks;
layout(std430,binding=2) writeonly buffer Image { uint words[]; } dst;
layout(push_constant) uniform Constants { uint width; uint height; uint encodedWidth; uint codec; } p;
uint byteAt(uint pos) { return (src.words[pos/4u] >> ((pos%4u)*8u)) & 255u; }
// Fields are at most 16 bits. Read one word, and only read the next word
// when the validated field crosses it; the host pads the final stream word.
uint wordField(uint pos, uint bit, uint bits) {
    uint shift=(pos&3u)*8u+bit;
    uint value=src.words[pos>>2u]>>shift;
    if(shift+bits>32u) value|=src.words[(pos>>2u)+1u]<<(32u-shift);
    return value&((1u<<bits)-1u);
}
uint little16(uint pos) {
#ifdef RAW_BYTE_FIELDS
    return byteAt(pos)|(byteAt(pos+1u)<<8u);
#else
    return wordField(pos,0u,16u);
#endif
}
uint valueAt(uint off, uint bits, uint i) {
    uint l=i%8u, g=i/8u;
    if(bits==0u) return 0u;
    if(bits==1u) return (byteAt(off+l)>>g)&1u;
    if(bits==2u) return (byteAt(off+(g/4u)*8u+l)>>((g%4u)*2u))&3u;
    if(bits==3u) {
        uint a=byteAt(off+l), b=byteAt(off+8u+l), c=byteAt(off+16u+l);
        if(g==0u) return a&7u; if(g==1u) return (a>>3u)&7u;
        if(g==2u) return (a>>6u)|(((c>>6u)&1u)<<2u);
        if(g==3u) return b&7u; if(g==4u) return (b>>3u)&7u;
        if(g==5u) return (b>>6u)|(((c>>7u)&1u)<<2u);
        if(g==6u) return c&7u; return (c>>3u)&7u;
    }
    if(bits==4u) return (byteAt(off+(g/2u)*8u+l)>>((g%2u)*4u))&15u;
    if(bits==5u) {
        if(g<5u) return byteAt(off+g*8u+l)&31u;
        if(g==5u) return (byteAt(off+l)>>5u)|(((byteAt(off+24u+l)>>5u)&3u)<<3u);
        if(g==6u) return (byteAt(off+8u+l)>>5u)|(((byteAt(off+32u+l)>>5u)&3u)<<3u);
        return (byteAt(off+16u+l)>>5u)|((byteAt(off+24u+l)>>7u)<<3u)|((byteAt(off+32u+l)>>7u)<<4u);
    }
    if(bits==6u) {
        if(g<6u) return byteAt(off+g*8u+l)&63u;
        uint base=off+(g-6u)*24u+l;
        return (byteAt(base)>>6u)|((byteAt(base+8u)>>6u)<<2u)|((byteAt(base+16u)>>6u)<<4u);
    }
    if(bits<=8u) return byteAt(off+g*8u+l);
    if(bits<=10u) return byteAt(off+(g+g/4u)*8u+l)|(((byteAt(off+((g/4u)*5u+4u)*8u+l)>>((g%4u)*2u))&3u)<<8u);
    return little16(off+i*2u);
}
uint packedField(uint off,uint bits,uint i) {
    if(bits==0u) return 0u;
    uint bit=i*bits;
#ifdef RAW_BYTE_FIELDS
    uint start=bit/8u, length=(bits*(RAW_CODEC==1u?64u:32u)+7u)/8u;
    uint v=byteAt(off+start);
    if(start+1u<length) v|=byteAt(off+start+1u)<<8u;
    if(start+2u<length) v|=byteAt(off+start+2u)<<16u;
    return (v>>(bit%8u))&((1u<<bits)-1u);
#else
    return wordField(off+bit/8u,bit&7u,bits);
#endif
}
uint laneField(uint off,uint bits,uint i) {
    if(bits==0u) return 0u;
    uint lane=i%8u;
    uvec2 value=uvec2(0u);
    for(uint n=0u;n<bits/4u;++n) {
        uint part=little16(off+n*16u+lane*2u);
        if(n<2u) value.x|=part<<(n*16u); else value.y|=part<<((n-2u)*16u);
    }
    uint base=(bits/4u)*16u, shift=base;
    if((bits&2u)!=0u) {
        uint part=byteAt(off+base+lane);
        if(shift<32u) value.x|=part<<shift; else value.y|=part<<(shift-32u);
        base+=8u; shift+=8u;
    }
    if((bits&1u)!=0u) {
        uint part=(byteAt(off+base+lane/2u)>>((lane%2u)*4u))&15u;
        if(shift<32u) value.x|=part<<shift; else value.y|=part<<(shift-32u);
    }
    uint bit=(i/8u)*bits;
    uint result=bit>=32u?value.y>>(bit-32u):value.x>>bit;
    if(bit>0u && bit<32u) result|=value.y<<(32u-bit);
    return result&((1u<<bits)-1u);
}
uint pixel(uint x,uint y) {
    if(RAW_CODEC==1u) {
        uint block=((y/16u)*p.encodedWidth+x/16u)*4u+(y%2u)*2u;
        uint i=((y%16u)/2u)*8u+(x%16u)/2u;
        uint a=block*3u, b=a+3u;
        uint lo=packedField(blocks.words[a],blocks.words[a+1u],i)+blocks.words[a+2u];
        uint hi=packedField(blocks.words[b],blocks.words[b+1u],i)+blocks.words[b+2u];
        return lo|(hi<<16u);
    }
    if(RAW_CODEC==2u) {
        uint a=(y*p.encodedWidth+x/32u)*3u, i=x%32u;
        uint off=blocks.words[a], bits=blocks.words[a+1u];
        uint lo=blocks.words[a+2u]==0u?packedField(off,bits,i):laneField(off,bits,i);
        uint hi=blocks.words[a+2u]==0u?packedField(off,bits,i+1u):laneField(off,bits,i+1u);
        return lo|(hi<<16u);
    }
    uint block=((y/4u)*(p.encodedWidth/64u)+x/64u)*4u+(y%2u)*2u;
    uint i=(y%4u)/2u*32u+(x%64u)/2u;
    uint a=block*3u, b=a+3u;
    uint lo=(valueAt(blocks.words[a],blocks.words[a+1u],i)+blocks.words[a+2u])&65535u;
    uint hi=(valueAt(blocks.words[b],blocks.words[b+1u],i)+blocks.words[b+2u])&65535u;
    return lo|(hi<<16u);
}
void main() {
    uint word=gl_GlobalInvocationID.x;
    if(word>=p.width*p.height/2u) return;
    uint pixelIndex=word*2u;
    dst.words[word]=pixel(pixelIndex%p.width,pixelIndex/p.width);
}
)glsl";
// Each invocation reconstructs one row parity in an independent MC24 strip.
// Horizontal prediction is serial; both CFA components share a packed word.
constexpr char mc24RestoreShader[] = R"glsl(#version 450
layout(local_size_x=8) in;
layout(std430,binding=0) readonly buffer Residuals { uint words[]; } src;
layout(std430,binding=1) readonly buffer Unused { uint words[]; } unusedBlocks;
layout(std430,binding=2) coherent buffer Image { uint words[]; } dst;
layout(std430,binding=3) buffer Validation { uint valid; } status;
layout(push_constant) uniform Constants {
    uint width; uint height; uint rows; uint strips; uint predictor; uint threshold; uint maximumCode;
} p;
ivec2 unpack(uint v) { return ivec2(int(v&65535u),int(v>>16u)); }
ivec2 signedResidual(ivec2 r) { return (r>>1)^(-(r&1)); }
ivec2 predict(ivec2 left,ivec2 top,ivec2 corner,ivec2 right) {
    ivec2 low=min(left,top), high=max(left,top);
    ivec2 rangeLow=min(low,min(corner,right)), rangeHigh=max(high,max(corner,right));
    return mix(clamp(left+top-corner,low,high),
        (min(ivec2(65535),left+top+corner+right)+2)/4,
        lessThanEqual(rangeHigh-rangeLow,ivec2(p.threshold)));
}
void main() {
    uint task=gl_GlobalInvocationID.x;
    if(task>=p.strips*2u) return;
    uint start=(task/2u)*p.rows, count=min(p.rows,p.height-start), stride=p.width/2u;
    for(uint row=task%2u;row<count;row+=2u) {
        ivec2 left=ivec2(0), corner=ivec2(0);
        uint base=(start+row)*stride;
        for(uint x=0u;x<stride;++x) {
            ivec2 r=unpack(src.words[base+x]), value;
            if(p.predictor==0u || row<2u) value=x==0u?r:left+signedResidual(r);
            else {
                ivec2 top=unpack(dst.words[base-2u*stride+x]);
                ivec2 right=unpack(dst.words[base-2u*stride+min(x+1u,stride-1u)]);
                value=(x==0u?top:predict(left,top,corner,right))+signedResidual(r);
                corner=top;
            }
            value &= ivec2(65535);
            if(any(greaterThan(value,ivec2(p.maximumCode)))) atomicAnd(status.valid,0u);
            dst.words[base+x]=uint(value.x)|(uint(value.y)<<16u);
            left=value;
        }
        memoryBarrierBuffer();
    }
}
)glsl";
// Control words are retained independently for overlapping/deferred frames.
// Pool access uses the context submission lock, including lease retirement.
struct ValidationPool : std::enable_shared_from_this<ValidationPool> {
    std::shared_ptr<void> deviceLease = jpegContextLease();
    VulkanJpegContext& context = jpegContext();
    std::vector<pl_buf> retired;
    ~ValidationPool() {
        std::lock_guard lock(context.mutex);
        for (auto& buffer : retired) pl_buf_destroy(context.vulkan->gpu, &buffer);
    }
    std::shared_ptr<void> acquire(pl_buf& buffer) {
        const auto gpu = context.vulkan->gpu;
        buffer = nullptr;
        const bool disabled = rawpreamp::debugEnvironmentIsSet("RAWPREAMP_GPU_POOL_DISABLE");
        if (!disabled) for (auto it = retired.begin(); it != retired.end(); ++it) {
            if (!pl_buf_poll(gpu, *it, 0)) {
                buffer = *it;
                retired.erase(it);
                break;
            }
        }
        if (!buffer) {
            pl_buf_params params{};
            params.size = sizeof(uint32_t);
            params.storable = true;
            params.host_writable = true;
            buffer = pl_buf_create(gpu, &params);
        }
        if (!buffer) return {};
        return std::shared_ptr<void>(new pl_buf(buffer),
            [pool = shared_from_this(), disabled](void* pointer) {
                auto* owned = static_cast<pl_buf*>(pointer);
                std::lock_guard lock(pool->context.mutex);
                if (!disabled && pool->retired.size() < 8) pool->retired.push_back(*owned);
                else pl_buf_destroy(pool->context.vulkan->gpu, owned);
                delete owned;
            });
    }
};
struct State {
    std::shared_ptr<void> deviceLease;
    pl_gpu gpu = nullptr;
    std::array<pl_pass, 3> passes{};
    pl_pass restore = nullptr, restoreWavefront = nullptr;
    pl_timer decodeTimer = nullptr, restoreTimer = nullptr;
    pl_buf stream = nullptr, descriptors = nullptr, residuals = nullptr;
    pl_buf validationReadback = nullptr;
    std::shared_ptr<ValidationPool> validationPool;
    ~State() {
        pl_buf_destroy(gpu, &validationReadback);
        pl_buf_destroy(gpu, &stream);
        pl_buf_destroy(gpu, &descriptors);
        for (auto& pass : passes) pl_pass_destroy(gpu, &pass);
        pl_timer_destroy(gpu, &decodeTimer);
        pl_timer_destroy(gpu, &restoreTimer);
        pl_pass_destroy(gpu, &restore);
        pl_pass_destroy(gpu, &restoreWavefront);
        pl_buf_destroy(gpu, &residuals);
    }
    bool init(pl_gpu device, uint32_t codec) {
        gpu = device;
        auto& pass = passes[codec];
        if (pass)
            return true;
        pl_desc desc[] = {{"src", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                          {"blocks", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
                          {"dst", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_WRITEONLY}};
        pl_pass_params params{};
        params.type = PL_PASS_COMPUTE;
        std::string source(shader);
        source.insert(source.find('\n') + 1, "#define RAW_CODEC " + std::to_string(codec) + "u\n");
        const bool byteFields = rawpreamp::debugEnvironmentIsSet("RAWPREAMP_RAW_BYTE_FIELDS");
        if (byteFields) source.insert(source.find('\n') + 1, "#define RAW_BYTE_FIELDS 1\n");
        params.glsl_shader = source.c_str();
        params.descriptors = desc;
        params.num_descriptors = 3;
        params.push_constants_size = 4 * sizeof(uint32_t);
        pass = pl_pass_create(gpu, &params);
        if (!pass && !byteFields) {
            source.insert(source.find('\n') + 1, "#define RAW_BYTE_FIELDS 1\n");
            params.glsl_shader = source.c_str();
            pass = pl_pass_create(gpu, &params);
        }
        return pass != nullptr;
    }
    bool initRestore(bool wavefront) {
        auto& selected = wavefront ? restoreWavefront : restore;
        if (selected)
            return true;
        pl_desc desc[] = {{"src", PL_DESC_BUF_STORAGE, 0, PL_DESC_ACCESS_READONLY},
                          {"unusedBlocks", PL_DESC_BUF_STORAGE, 1, PL_DESC_ACCESS_READONLY},
                          {"dst", PL_DESC_BUF_STORAGE, 2, PL_DESC_ACCESS_READWRITE},
                          {"status", PL_DESC_BUF_STORAGE, 3, PL_DESC_ACCESS_READWRITE}};
        pl_pass_params params{};
        params.type = PL_PASS_COMPUTE;
        std::string source(mc24RestoreShader);
        if (wavefront) {
            source.replace(source.find("local_size_x=8"), sizeof("local_size_x=8") - 1, "local_size_x=16");
            source.erase(source.find("void main()"));
            source += R"glsl(
// A diagonal wavefront reconstructs all rows of a strip concurrently. The
// previous-row right neighbour precedes this row by one step, and the top
// neighbour by two. Keep those columns in workgroup memory, rather than
// serializing an entire strip through one invocation and global memory.
shared uint history[16 * 3];
void main() {
    uint strip=gl_WorkGroupID.x, row=gl_LocalInvocationID.x;
    uint start=strip*p.rows, count=min(p.rows,p.height-start), stride=p.width/2u;
    ivec2 left=ivec2(0);
    uint last=stride+2u*((count-1u)/2u);
    for(uint step=0u;step<last;++step) {
        uint delay=2u*(row/2u);
        if(row<count && step>=delay && step-delay<stride) {
            uint x=step-delay, base=(start+row)*stride;
            ivec2 r=unpack(src.words[base+x]), value;
            if(p.predictor==0u || row<2u) value=x==0u?r:left+signedResidual(r);
            else {
                ivec2 top=unpack(history[(row-2u)*3u+(step+1u)%3u]);
                ivec2 right=x+1u<stride
                    ? unpack(history[(row-2u)*3u+(step+2u)%3u]) : top;
                ivec2 corner=x==0u?ivec2(0):unpack(history[(row-2u)*3u+step%3u]);
                value=(x==0u?top:predict(left,top,corner,right))+signedResidual(r);
            }
            value &= ivec2(65535);
            if(any(greaterThan(value,ivec2(p.maximumCode)))) atomicAnd(status.valid,0u);
            left=value;
            uint packed=uint(value.x)|(uint(value.y)<<16u);
            dst.words[base+x]=packed;
            // Read the old ring slot before any row overwrites it below.
        }
        barrier();
        if(row<count && step>=delay && step-delay<stride)
            history[row*3u+step%3u]=uint(left.x)|(uint(left.y)<<16u);
        barrier();
    }
}
)glsl";
        }
        params.glsl_shader = source.c_str();
        params.descriptors = desc;
        params.num_descriptors = 4;
        params.push_constants_size = 7 * sizeof(uint32_t);
        selected = pl_pass_create(gpu, &params);
        return selected != nullptr;
    }
};
State& rawState() {
    static State value;
    return value;
}
bool dispatchBlocks(const uint8_t* bytes, size_t size, uint32_t width, uint32_t height,
                    uint32_t encodedWidth, uint32_t codec, const std::vector<uint32_t>& blocks,
                    VulkanResidentImage& image, uint32_t rows = 0, uint32_t strips = 0,
                    uint32_t predictor = 0, uint32_t threshold = 0, uint32_t maximumCode = 65535) {
    auto& context = jpegContext();
    std::lock_guard lock(context.mutex);
    if (!context.init())
        return false;
    auto& state = rawState();
    if (!state.deviceLease)
        state.deviceLease = jpegContextLease();
    auto gpu = context.vulkan->gpu;
    if (!state.init(gpu, codec))
        return false;
    const uint64_t groups = (uint64_t(width) * height / 2u + 255u) / 256u;
    if (groups > uint32_t(gpu->limits.max_dispatch[0]))
        return false;
    auto buffer = [&](pl_buf& target, size_t n, bool upload) {
        pl_buf_params params{};
        params.size = n;
        params.storable = true;
        params.host_writable = upload;
        return n <= gpu->limits.max_ssbo_size && pl_buf_recreate(gpu, &target, &params);
    };
    pl_buf output = nullptr, validation = nullptr;
    auto outputLease = context.acquireImage(size_t(width) * height * 2, output);

    if (!buffer(state.stream, (size + 3u) & ~size_t(3), true) ||
        !buffer(state.descriptors, blocks.size() * 4, true) ||
        !outputLease)
        return false;
    std::shared_ptr<void> validationLease;
    if (codec == 2) {
        if (!state.initRestore(rows <= 16) ||
            !buffer(state.residuals, size_t(width) * height * 2, false)) return false;
        if (!state.validationPool) state.validationPool = std::make_shared<ValidationPool>();
        validationLease = state.validationPool->acquire(validation);
        if (!validationLease) return false;
    }
    // Zero the final upload word so byte reads never touch uninitialized padding.
    uint32_t tail = 0;
    size_t complete = size & ~size_t(3);
    if (complete)
        pl_buf_write(gpu, state.stream, 0, bytes, complete);
    if (size != complete) {
        std::memcpy(&tail, bytes + complete, size - complete);
        pl_buf_write(gpu, state.stream, complete, &tail, 4);
    }
    pl_buf_write(gpu, state.descriptors, 0, blocks.data(), blocks.size() * 4);
    pl_desc_binding bindings[4]{};
    bindings[0].object = state.stream;
    bindings[1].object = state.descriptors;
    bindings[2].object = codec == 2 ? state.residuals : output;
    uint32_t constants[] = {width, height, encodedWidth, codec};
    pl_pass_run_params run{};
    run.pass = state.passes[codec];
    run.desc_bindings = bindings;
    run.push_constants = constants;
    run.compute_groups[0] = int(groups);
    run.compute_groups[1] = run.compute_groups[2] = 1;
    const bool profile = rawpreamp::debugEnvironment("RAWPREAMP_RAW_GPU_TIMERS");
    if (profile) {
        if (!state.decodeTimer) state.decodeTimer = pl_timer_create(gpu);
        if (!state.restoreTimer) state.restoreTimer = pl_timer_create(gpu);
        uint64_t ns;
        while ((ns = pl_timer_query(gpu, state.decodeTimer)))
            spdlog::info("GALLERY_PERF event=raw_gpu_stage codec={} stage=unpack elapsed_ms={:.3f}", codec, ns / 1e6);
        while ((ns = pl_timer_query(gpu, state.restoreTimer)))
            spdlog::info("GALLERY_PERF event=raw_gpu_stage codec=2 stage=predict elapsed_ms={:.3f}", ns / 1e6);
        run.timer = state.decodeTimer;
    }
    pl_pass_run(gpu, &run);
    if (codec == 2) {
        uint32_t restoreConstants[] = {width,     height,    rows,       strips,
                                       predictor, threshold, maximumCode};
        uint32_t valid = 1;
        pl_buf_write(gpu, validation, 0, &valid, sizeof(valid));
        bindings[3].object = validation;
        bindings[0].object = state.residuals;
        bindings[2].object = output;
        run.timer = profile ? state.restoreTimer : nullptr;
        run.pass = rows <= 16 ? state.restoreWavefront : state.restore;
        run.push_constants = restoreConstants;
        run.compute_groups[0] = int(rows <= 16 ? strips : (strips * 2u + 7u) / 8u);
        pl_pass_run(gpu, &run);
    }
    VulkanResidentImage result;
    result.gpu = gpu;
    result.buffer = output;
    result.width = width;
    result.height = height;
    result.channels = 1;
    result.bits = 16;
    result.wordsPerRow = width / 2;
    result.deviceLease = state.deviceLease;
    result.validationBuffer = validation;
    struct Leases { std::shared_ptr<void> output, validation; };
    result.lease = std::make_shared<Leases>(
        Leases{std::move(outputLease), std::move(validationLease)});
    image = std::move(result);
    return true;
}
} // namespace
bool mcrawDecodeVulkan(const uint8_t* bytes, size_t size, uint32_t width, uint32_t height,
                       VulkanResidentImage& image) {
    if (!bytes || size < 16 || size > UINT32_MAX - 3u || !width || !height || (width % 2) ||
        (height % 4) || uint64_t(width) * height > UINT32_MAX / 2u)
        return false;
    const uint32_t ew = read32(bytes), eh = read32(bytes + 4);
    const uint32_t bo = read32(bytes + 8), ro = read32(bytes + 12);
    if (!ew || ew % 64 || ew < width || eh != height || bo < 16 || ro < 16 ||
        uint64_t(ew) * eh > UINT32_MAX)
        return false;
    const size_t count = uint64_t(ew) * eh / 64u;
    std::vector<uint16_t> bits, refs;
    if (!metadata(bytes, size, bo, count, bits) || !metadata(bytes, size, ro, count, refs))
        return false;
    std::vector<uint32_t> blocks(count * 3);
    size_t at = 16, end = std::min(bo, ro);
    for (size_t i = 0; i < count; ++i) {
        if (bits[i] > 16 || at > end || lengths[bits[i]] > end - at)
            return false;
        blocks[i * 3] = uint32_t(at);
        blocks[i * 3 + 1] = bits[i];
        blocks[i * 3 + 2] = refs[i];
        at += lengths[bits[i]];
    }
    return dispatchBlocks(bytes, size, width, height, ew, 0, blocks, image);
}

bool unspektraDecodeVulkan(const uint8_t* bytes, size_t size, uint32_t width, uint32_t height,
                           VulkanResidentImage& image) {
    if (!bytes || size < 16 || size > UINT32_MAX - 3u || !width || !height || width % 2 ||
        height % 2 || uint64_t(width) * height > UINT32_MAX / 2u || std::memcmp(bytes, "UFR1", 4) ||
        read32(bytes + 4) != width || read32(bytes + 8) != height ||
        (read32(bytes + 12) != 2u && read32(bytes + 12) != 4u))
        return false;
    const uint32_t across = (width + 15u) / 16u, down = (height + 15u) / 16u;
    const uint64_t count = uint64_t(across) * down * 4u;
    if (count > (size - 16) / 3u)
        return false;
    std::vector<uint32_t> blocks(size_t(count) * 3u);
    size_t at = 16;
    for (size_t i = 0; i < count; ++i) {
        if (size - at < 3)
            return false;
        const uint32_t bits = bytes[at],
                       reference = uint32_t(bytes[at + 1]) | uint32_t(bytes[at + 2]) << 8;
        if (bits > 16 || size - at - 3 < size_t(bits) * 8u ||
            reference + ((1u << bits) - 1u) > 65535u)
            return false;
        blocks[i * 3] = uint32_t(at + 3);
        blocks[i * 3 + 1] = bits;
        blocks[i * 3 + 2] = reference;
        at += 3 + bits * 8u;
    }
    return at == size && dispatchBlocks(bytes, size, width, height, across, 1, blocks, image);
}
bool mc24DecodeVulkan(const uint8_t* bytes, size_t size, uint32_t width, uint32_t height,
                      VulkanResidentImage& image, uint32_t maximumCode) {
    auto u16 = [&](size_t p) { return uint32_t(bytes[p]) | uint32_t(bytes[p + 1]) << 8; };
    if (!bytes || size < 96 || size > UINT32_MAX - 3u || !width || !height || width % 2 ||
        width > 16384 || maximumCode > 65535 || std::memcmp(bytes, "MCRF", 4) ||
        uint64_t(width) * height > UINT32_MAX / 2u || u16(44) != width || u16(46) != height ||
        bytes[55] != 0 || bytes[56] > 1 || bytes[92] != 1)
        return false;
    const uint32_t rows = u16(62), strips = u16(60), fixed = u16(30), head = read32(bytes + 8);
    if (!rows || rows > 16384 || !strips || strips != (uint64_t(height) + rows - 1u) / rows ||
        uint64_t(fixed) + strips * 8ull != head || fixed < 96 || head > size ||
        read32(bytes + 4) != size || uint64_t(width / 2u) * ((rows + 1u) / 2u) > 65536u)
        return false;
    const uint32_t perRow = (width + 31u) / 32u;
    const size_t blockCount = size_t(perRow) * height;
    if (blockCount > uint64_t(size - head) * 2u)
        return false;
    std::vector<uint32_t> blocks(blockCount * 3u);
    size_t position = head;
    for (uint32_t strip = 0; strip < strips; ++strip) {
        const size_t length = read32(bytes + fixed + strip * 4u);
        if (length > size - position)
            return false;
        const uint32_t count = std::min(rows, height - strip * rows) * perRow;
        const size_t widths = (count + 1u) / 2u;
        if (widths > length)
            return false;
        uint32_t packing = bytes[57];
        if (packing == 3u) {
            const uint32_t split = u16(94);
            if (!split || split >= strips)
                return false;
            packing = (bytes[93] >> (strip < split ? 0u : 4u)) & 15u;
        }
        if (packing > 1u)
            return false;
        size_t at = position + widths;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t bits = (bytes[position + i / 2u] >> ((i % 2u) * 4u)) & 15u;
            if (size_t(bits) * 4u > position + length - at)
                return false;
            const size_t index = (size_t(strip) * rows * perRow + i) * 3u;
            blocks[index] = uint32_t(at);
            blocks[index + 1] = bits;
            blocks[index + 2] = packing;
            at += bits * 4u;
        }
        if (at != position + length)
            return false;
        position += length;
    }
    return size - position <= 7u &&
           dispatchBlocks(bytes, size, width, height, perRow, 2, blocks, image, rows, strips,
                          bytes[56], u16(58), maximumCode);
}

bool validateVulkanRawFrame(const VulkanResidentImage& image) {
    if (!image.validationBuffer)
        return true;
    auto& context = jpegContext();
    std::lock_guard lock(context.mutex);
    if (!context.init() || image.gpu != context.vulkan->gpu)
        return false;
    auto gpu = context.vulkan->gpu;
    auto& state = rawState();
    if (!state.deviceLease) state.deviceLease = jpegContextLease();
    if (!state.validationReadback) {
        pl_buf_params params{};
        params.size = sizeof(uint32_t);
        params.host_readable = true;
        state.validationReadback = pl_buf_create(gpu, &params);
    }
    if (!state.validationReadback) return false;
    pl_buf_copy(gpu, state.validationReadback, 0,
                static_cast<pl_buf>(image.validationBuffer), 0, sizeof(uint32_t));
    uint32_t valid = 0;
    const bool ok = pl_buf_read(gpu, state.validationReadback, 0, &valid, sizeof(valid)) && valid == 1;
    return ok;
}

} // namespace rawpreamp
#else
namespace rawpreamp {
bool mcrawDecodeVulkan(const uint8_t*, size_t, uint32_t, uint32_t, VulkanResidentImage&) {
    return false;
}
bool unspektraDecodeVulkan(const uint8_t*, size_t, uint32_t, uint32_t, VulkanResidentImage&) {
    return false;
}
bool mc24DecodeVulkan(const uint8_t*, size_t, uint32_t, uint32_t, VulkanResidentImage&,
                      uint32_t) {
    return false;
}
bool validateVulkanRawFrame(const VulkanResidentImage&) { return false; }
} // namespace rawpreamp
#endif

namespace rawpreamp {
bool supportsRawFrameDecodeVulkan(motioncam::EncodedFrameFormat format) {
#ifdef RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO
    return format == motioncam::EncodedFrameFormat::MotionCam ||
           format == motioncam::EncodedFrameFormat::Mc24 ||
           format == motioncam::EncodedFrameFormat::Unspektra;
#else
    return false;
#endif
}
bool rawFrameDecodeVulkan(motioncam::EncodedFrameFormat format, uint32_t compression,
                          const uint8_t* bytes, size_t size, uint32_t width, uint32_t height,
                          VulkanResidentImage& image, uint32_t maximumCode) {
    if (!supportsRawFrameDecodeVulkan(format) ||
        rawpreamp::debugEnvironmentIsSet("RAWPREAMP_TEST_FAIL_RAW_GPU_DECODE"))
        return false;
    switch (format) {
    case motioncam::EncodedFrameFormat::MotionCam:
        return compression == 7 && mcrawDecodeVulkan(bytes, size, width, height, image);
    case motioncam::EncodedFrameFormat::Mc24:
        return mc24DecodeVulkan(bytes, size, width, height, image, maximumCode);
    case motioncam::EncodedFrameFormat::Unspektra:
        return unspektraDecodeVulkan(bytes, size, width, height, image);
    }
    return false;
}
} // namespace rawpreamp

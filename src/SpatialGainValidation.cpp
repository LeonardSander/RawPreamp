#include "SpatialGainValidation.h"
#include "DebugInstrumentation.h"
#include <fstream>
#include <filesystem>
#if RAWPREAMP_DEBUG_INSTRUMENTATION && defined(RAWPREAMP_HAS_NATIVE_VIDEO_PLACEBO)
#include "VulkanGainBake.h"
#include "DNGImage.h"
#include "VulkanSpatialGainCache.h"
#include "VulkanGalleryPreview.h"
#include "VulkanJpegContext.h"
#include <QtCore/qglobal.h>
#include <QtCore/QByteArray>
#include <cstring>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <vector>

void testRawDecodeVulkan();

namespace rawpreamp {
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct EnvironmentOverride {
    const char* name;
    bool existed;
    QByteArray previous;
    explicit EnvironmentOverride(const char* variable)
        : name(variable), existed(qEnvironmentVariableIsSet(name)), previous(qgetenv(name)) {}
    void set(const char* value) {
        if (value) qputenv(name, value);
        else qunsetenv(name);
    }
    ~EnvironmentOverride() {
        if (existed) qputenv(name, previous);
        else qunsetenv(name);
    }
};
struct CacheOverride : EnvironmentOverride {
    CacheOverride() : EnvironmentOverride("RAWPREAMP_GAIN_FIELD_DISABLE") {}
    void disable(bool disabled) { set(disabled ? "1" : nullptr); }
};
struct SourceResources {
    pl_gpu gpu;
    pl_buf buffer = nullptr;
    pl_tex texture = nullptr;
    ~SourceResources() {
        pl_tex_destroy(gpu, &texture);
        pl_buf_destroy(gpu, &buffer);
    }
};
}
int runRawDecodeValidation(const std::string& reportPath) {
    std::ofstream report(reportPath);
    try {
        require(bool(report), "Could not open raw decode report");
        EnvironmentOverride required("RAWPREAMP_TEST_REQUIRE_VULKAN"), temporary("TMPDIR");
        required.set("1");
        const auto directory=std::filesystem::absolute(reportPath).parent_path().string();
        temporary.set(directory.c_str());
        // These fixtures include cross-thread resource retirement. Each owns
        // its submission lock; do not hold the context lock around the suite.
        testRawDecodeVulkan();
        report<<"PASS raw codec fixtures (MCRAW/UFR1/MC24; packing, predictors, partial strips, quantized validity and resource lifetime)\n";
        return 0;
    } catch(const std::exception& error) { report<<"FAIL "<<error.what()<<"\n";return 1; }
}

int runSpatialGainValidation(const std::string& reportPath) {
    unsigned cases = 0;
    std::ofstream report(reportPath);
    try {
        require(bool(report), "Could not open spatial gain report");
        CacheOverride override;
        auto& context = jpegContext();
        std::lock_guard lock(context.mutex);
        require(context.init(), "Vulkan initialization failed");
        const auto gpu = context.vulkan->gpu;
        // Exceed the placeholder gain buffer: small images hide invalid warmup reads.
        constexpr uint32_t width = 66, height = 34;
        for (bool cfa : {true,false}) {
            const uint32_t channels = cfa ? 1u : 3u;
            std::vector<uint16_t> samples(width*height*channels);
            for (size_t i=0;i<samples.size();++i) samples[i]=uint16_t(100+(i*17)%512);
            pl_buf_params sourceParams{};
            sourceParams.size=samples.size()*2u; sourceParams.storable=true;
            sourceParams.initial_data=samples.data();
            SourceResources resources{gpu};
            auto& buffer = resources.buffer;
            buffer=pl_buf_create(gpu,&sourceParams);
            require(buffer,"Source buffer allocation failed");
            std::vector<uint16_t> rgba(width*height*4,65535);
            if (!cfa) for(size_t i=0;i<width*height;++i)
                for(unsigned c=0;c<3;++c) rgba[i*4+c]=samples[i*3+c];
            pl_tex_params texParams{};
            texParams.w=width;texParams.h=height;
            texParams.format=pl_find_fmt(gpu,PL_FMT_UNORM,cfa?1:4,16,16,PL_FMT_CAP_SAMPLEABLE);
            texParams.sampleable=texParams.host_writable=true;
            auto& texture = resources.texture;
            texture=pl_tex_create(gpu,&texParams);
            require(texture,"Source texture allocation failed");
            pl_tex_transfer_params upload{}; upload.tex=texture;
            upload.ptr=cfa?static_cast<void*>(samples.data()):static_cast<void*>(rgba.data());
            require(pl_tex_upload(gpu,&upload),"Source texture upload failed");
            std::vector<float> primary(4u*(cfa?4u:3u)), secondary{1,1.25f,0.75f,1.5f};
            for(size_t i=0;i<primary.size();++i) primary[i]=1.0f+float(i%4)*0.25f;
            const auto original=primary;
            VulkanRgbGainMap map{2,2,cfa?4u:3u,0,0,width,height,1,1,0,0,0.5f,0.5f,&primary};
            VulkanRgbGainMap luma{2,2,1,0,0,width,height,1,1,0,0,0.5f,0.5f,&secondary};
            VulkanGainBakeTextureSource texSource{gpu,texture};
            VulkanGainBakeBufferSource bufSource{gpu,buffer,jpegContextLease()};
            for(unsigned kind=0;kind<3;++kind) for(bool multi:{false,true}) for(bool log:{false,true}) {
                VulkanGainBakeParams params;
                params.width=width;params.height=height;params.channels=channels;
                params.outputBits=log?12:16;params.log60=log;params.encodedWhite=log?4095:0;
                params.scale.fill(1);params.destinationWhite=65535;
                if(multi) params.maps={map,luma};
                else if(cfa) {params.cfaMaps[0]=map;params.cfaMapCount=1;}
                else params.map=map;
                if(kind==1) params.bufferSource=&bufSource;
                if(kind==2) params.textureSource=&texSource;
                const size_t bytes=((size_t(width)*channels*params.outputBits+7)/8)*height;
                std::array<std::vector<uint8_t>,5> expected;
                const auto buildsBefore=spatialGainCacheBuilds.load();
                const auto hitsBefore=spatialGainCacheHits.load();
                for(unsigned mode=0;mode<2;++mode) {
                    override.disable(mode==0);
                    for(unsigned frame=0;frame<5;++frame) {
                        primary=original; if(frame>=2) primary[0]+=0.25f;
                        params.black.fill(float(frame));params.ditherSeed=23+frame;
                        params.logHighQuality=frame%2;
                        std::vector<uint8_t> output(bytes);
                        const bool ok=cfa?bakeCfaGainVulkan(kind?nullptr:samples.data(),params,output.data(),bytes)
                            :bakeRgbGainVulkan(kind?nullptr:samples.data(),params,output.data(),bytes);
                        require(ok,"General GPU bake failed");
                        if(mode==0) expected[frame]=std::move(output);
                        else require(output==expected[frame],"Cached GPU bake differs from uncached output");
                    }
                }
                require(spatialGainCacheBuilds.load()>=buildsBefore+2 && spatialGainCacheHits.load()>hitsBefore,
                        "Spatial fields were not built and reused for a supported source");
                ++cases;
            }
            // The gallery must work for CPU and texture Bayer sources and RGB textures.
            const std::array<float,3> gains{1,1,1};
            const std::array<float,9> identity{1,0,0,0,1,0,0,0,1};
            for(unsigned kind=0;kind<3u;++kind) {
                VulkanGalleryBayerGain bayerGain;
                bayerGain.sourceWidth=width;bayerGain.sourceHeight=height;bayerGain.scale.fill(1);
                bayerGain.encodedWhite=4095;
                VulkanGalleryBayerGainMap bayerMap;
                bayerMap.width=bayerMap.height=2;bayerMap.channels=4;
                bayerMap.right=width;bayerMap.bottom=height;
                bayerMap.spacingH=bayerMap.spacingV=0.5f;
                bayerMap.values.resize(16);
                bayerGain.maps={bayerMap};
                VulkanGalleryBayerSource bayer;
                bayer.width=width;bayer.height=height;bayer.white.fill(65535);bayer.gain=&bayerGain;
                if(kind==0) bayer.samples=samples.data();else {bayer.gpu=gpu;bayer.texture=texture;}
                VulkanGalleryGain rgbGain;
                rgbGain.width=rgbGain.height=2;rgbGain.right=width;rgbGain.bottom=height;
                rgbGain.spacingH=rgbGain.spacingV=0.5f;rgbGain.encodedWhite=4095;
                rgbGain.values.resize(12);
                VulkanGalleryTextureSource rgb;
                rgb.gpu=gpu;rgb.texture=texture;rgb.width=width;rgb.height=height;
                rgb.white.fill(65535);rgb.gain=kind==2 ? nullptr : &rgbGain;
                if (!cfa && kind==1) {
                    rgbGain.width=rgbGain.height=1;
                    rgbGain.spacingH=rgbGain.spacingV=1;
                    rgbGain.values.resize(3);
                }
                std::array<std::vector<uint8_t>,5> expected;
                const auto buildsBefore=spatialGainCacheBuilds.load();
                const auto hitsBefore=spatialGainCacheHits.load();
                for(unsigned mode=0;mode<2;++mode) {
                    override.disable(mode==0);
                    for(unsigned frame=0;frame<5;++frame) {
                        if (cfa && kind==1) {
                            // Preserve the same field while switching resident
                            // source types; each needs its own consuming pass.
                            bayer.texture=frame==3 ? nullptr : texture;
                            bayer.buffer=frame==3 ? resources.buffer : nullptr;
                        }
                        auto& values=cfa?bayerGain.maps[0].values:rgbGain.values;
                        for(size_t i=0;i<values.size();++i) values[i]=1.0f+float(i%4)*0.25f;
                        if(cfa && kind==2) values[0]+=float(frame)*0.25f;
                        else if(frame>=2) values[0]+=0.25f;
                        bayerGain.ditherSeed=rgbGain.ditherSeed=41+frame;
                        bayerGain.black.fill(float(frame));rgb.black.fill(float(frame));
                        std::vector<uint8_t> output;
                        require(colorGalleryRgbVulkan(output,gains,identity,identity,1,nullptr,nullptr,false,false,
                            cfa?&bayer:nullptr,nullptr,cfa?nullptr:&rgb),"General gallery processing failed");
                        require(!pl_gpu_is_failed(gpu), "Gallery processing failed the GPU");
                        require(std::any_of(output.begin(),output.end(),[](uint8_t v){return v!=0;}),
                                "Gallery output is black");
                        if(mode==0) expected[frame]=std::move(output);
                        else require(output==expected[frame],"Cached gallery differs from uncached output");
                    }
                }
                if (cfa && kind==2)
                    require(spatialGainCacheBuilds.load()==buildsBefore+2 && spatialGainCacheHits.load()==hitsBefore,
                            "Changing maps did not stop unused field captures");
                else if (cfa || kind==0)
                    require(spatialGainCacheBuilds.load()>buildsBefore && spatialGainCacheHits.load()>hitsBefore,
                            "Spatial cache was not built and reused for a supported source");
                else require(spatialGainCacheBuilds.load()==buildsBefore && spatialGainCacheHits.load()==hitsBefore,
                             "Non-spatial gallery input used a cached-field shader");
                ++cases;
            }
            if (cfa) {
                // Compare shared-image reconstruction at partial workgroup
                // edges and all rotations against the unchanged buffer pass.
                constexpr char captureShader[] = R"glsl(#version 450
layout(local_size_x=256) in;
layout(binding=0) uniform sampler2D image;
layout(std430,binding=1) writeonly buffer Capture { uint words[]; } capture;
layout(push_constant) uniform Size { uint width; uint height; } p;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if(i>=p.width*p.height) return;
    uvec4 code=uvec4(floor(texelFetch(image,ivec2(int(i%p.width),int(i/p.width)),0)*255.0+0.5));
    capture.words[i]=code.r | (code.g<<8u) | (code.b<<16u);
}
)glsl";
                struct CaptureResources {
                    pl_gpu gpu;
                    pl_buf buffer=nullptr;
                    pl_pass pass=nullptr;
                    ~CaptureResources() { pl_pass_destroy(gpu,&pass);pl_buf_destroy(gpu,&buffer); }
                } capture{gpu};
                pl_desc descriptors[]={{"image",PL_DESC_SAMPLED_TEX,0,PL_DESC_ACCESS_READONLY},
                                       {"capture",PL_DESC_BUF_STORAGE,1,PL_DESC_ACCESS_WRITEONLY}};
                pl_pass_params pass{};pass.type=PL_PASS_COMPUTE;pass.glsl_shader=captureShader;
                pass.descriptors=descriptors;pass.num_descriptors=2;pass.push_constants_size=8;
                capture.pass=pl_pass_create(gpu,&pass);
                pl_buf_params buffer{};buffer.size=width*height*4u;buffer.storable=buffer.host_readable=true;
                capture.buffer=pl_buf_create(gpu,&buffer);
                require(capture.pass && capture.buffer,"Shared Bayer capture allocation failed");
                VulkanGalleryBayerSource bayer;
                bayer.gpu=gpu;bayer.buffer=resources.buffer;bayer.width=width;bayer.height=height;
                bayer.white.fill(65535);
                for (int orientation:{0,90,180,270}) {
                    VulkanGalleryOutput output;
                    output.width=width;output.height=height;output.rgb24=true;output.orientation=orientation;
                    std::vector<uint8_t> expected,pixels;
                    require(colorGalleryRgbVulkan(expected,gains,identity,identity,1,nullptr,nullptr,false,false,
                        &bayer,&output),"Bayer buffer reference failed");
                    std::shared_ptr<VulkanGalleryGpuBuffer> shared;
                    output.gpuBuffer=&shared;output.requireZeroCopy=true;
                    require(colorGalleryRgbVulkan(pixels,gains,identity,identity,1,nullptr,nullptr,false,false,
                        &bayer,&output),"Shared Bayer image failed");
                    require(shared && shared->texture && !shared->rgb10a2,"Shared Bayer fixture expected RGBA8");
                    pl_vulkan_release_params release{};
                    release.tex=static_cast<pl_tex>(shared->texture);release.layout=VK_IMAGE_LAYOUT_GENERAL;
                    release.qf=VK_QUEUE_FAMILY_IGNORED;
                    std::memcpy(&release.semaphore.sem,&shared->readySemaphore,sizeof(shared->readySemaphore));
                    release.semaphore.value=1;
                    pl_vulkan_release_ex(gpu,&release);
                    pl_desc_binding bindings[2]{};bindings[0].object=shared->texture;bindings[1].object=capture.buffer;
                    uint32_t shape[]={shared->width,shared->height};
                    pl_pass_run_params run{};run.pass=capture.pass;run.desc_bindings=bindings;run.push_constants=shape;
                    run.compute_groups[0]=(width*height+255u)/256u;run.compute_groups[1]=run.compute_groups[2]=1;
                    pl_pass_run(gpu,&run);
                    std::vector<uint32_t> actual(width*height);
                    require(pl_buf_read(gpu,capture.buffer,0,actual.data(),actual.size()*4u),"Shared Bayer capture failed");
                    for(size_t i=0;i<actual.size();++i) for(unsigned c=0;c<3;++c)
                        require(uint8_t(actual[i]>>(c*8u))==expected[i*3u+c],"Shared Bayer differs from buffer reference");
                    pl_gpu_finish(gpu); // Retire the imported semaphore before releasing its owner.
                    ++cases;
                }
            }
            if (!cfa) {
                // Warm the shared-image DCP shader, then fail only the new
                // thumbnail shader. Its SSBO fallback has a different field slot.
                EnvironmentOverride profileMode("RAWPREAMP_DCP_TEXTURES");
                EnvironmentOverride profileFailure("RAWPREAMP_TEST_DCP_TEXTURE_FAIL");
                profileFailure.set(nullptr);
                DNGProfileTable table;
                table.hueDivisions=table.saturationDivisions=table.valueDivisions=2;
                for (unsigned i=0;i<8;++i) table.values.insert(table.values.end(), {2.0f,0.9f,0.95f});
                VulkanGalleryGain gain;
                gain.width=gain.height=2;gain.right=width;gain.bottom=height;
                gain.spacingH=gain.spacingV=0.5f;gain.values=original;
                VulkanGalleryTextureSource rgb;
                rgb.gpu=gpu;rgb.texture=texture;rgb.width=width;rgb.height=height;
                rgb.white.fill(65535);rgb.gain=&gain;
                VulkanGalleryOutput output;
                output.width=width;output.height=height;output.rgb24=true;
                output.thumbnailWidth=16;output.thumbnailHeight=8;
                std::vector<uint8_t> pixels, expectedThumbnail, actualThumbnail;
                output.thumbnailPixels=&expectedThumbnail;
                override.disable(true);profileMode.set("0");
                require(colorGalleryRgbVulkan(pixels,gains,identity,identity,1,&table,nullptr,false,false,
                    nullptr,&output,&rgb),"DCP thumbnail reference failed");
                override.disable(false);profileMode.set("1");
                std::shared_ptr<VulkanGalleryGpuBuffer> shared;
                output.gpuBuffer=&shared;output.requireZeroCopy=true;output.thumbnailPixels=nullptr;
                for (unsigned frame=0;frame<2;++frame) {
                    require(colorGalleryRgbVulkan(pixels,gains,identity,identity,1,&table,nullptr,false,false,
                        nullptr,&output,&rgb),"DCP shared-image warmup failed");
                    require(shared && shared->texture,"DCP fixture did not use shared-image output");
                    shared.reset();
                }
                profileFailure.set("1");
                output.thumbnailPixels=&actualThumbnail;
                require(colorGalleryRgbVulkan(pixels,gains,identity,identity,1,&table,nullptr,false,false,
                    nullptr,&output,&rgb),"DCP thumbnail shader fallback failed");
                require(!pl_gpu_is_failed(gpu),"DCP thumbnail fallback failed the GPU");
                require(!expectedThumbnail.empty() && actualThumbnail==expectedThumbnail,
                        "DCP thumbnail fallback differs from uncached reference");
                ++cases;
            }
        }
        report<<"PASS cases="<<cases<<" (CPU/buffer/texture GPU baking and CPU/texture gallery; cold/build/reuse/invalidation)\n";
        return 0;
    } catch(const std::exception& error) {report<<"FAIL cases="<<cases<<" "<<error.what()<<"\n";return 1;}
}
}
#else
namespace rawpreamp {
int runSpatialGainValidation(const std::string&) { return 2; }
int runRawDecodeValidation(const std::string&) { return 2; }
}
#endif

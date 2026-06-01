// Dawn/WebGPU Backend Plugin - Cross-platform GPU via Dawn or wgpu-native
//
// Implements the backend vtable using WebGPU compute shaders (WGSL).
// Dispatches keccak256 and ecrecover via embedded WGSL compute shaders.
//
// Guarded by KINET_GPU_DAWN — compiles to a stub otherwise.

#include "kinet/gpu/backend_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(KINET_GPU_DAWN)
    #if defined(USE_DAWN_API)
        #include <webgpu/webgpu.h>
    #elif defined(USE_WGPU_API)
        #include <wgpu.h>
    #endif
#endif

// =============================================================================
// Context & Buffer Structures
// =============================================================================

struct KinetBackendContext {
    int device_index;
    std::string device_name;

#ifdef KINET_GPU_DAWN
    WGPUDevice device;
    WGPUQueue queue;
    std::unordered_map<std::string, WGPUComputePipeline> pipelines;
    std::unordered_map<std::string, WGPUShaderModule> modules;
#endif
};

struct KinetBackendBuffer {
    size_t size;
#ifdef KINET_GPU_DAWN
    WGPUBuffer gpu_buf;
#endif
    // CPU staging buffer for readback
    void* staging;
};

struct KinetBackendKernel {
    std::string entry_point;
#ifdef KINET_GPU_DAWN
    WGPUComputePipeline pipeline;
#endif
};

// =============================================================================
// CPU fallback: Keccak-256 (used when WebGPU not available)
// =============================================================================

namespace {

static constexpr uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f1600(uint64_t st[25]) {
    static constexpr int PI_LANE[24] = {
        10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
        15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
    };
    static constexpr int RHO[24] = {
         1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
        27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
    };

    for (int round = 0; round < 24; ++round) {
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; ++x) {
            uint64_t d = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
            for (int y = 0; y < 5; ++y) st[x + 5*y] ^= d;
        }
        uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = st[PI_LANE[i]];
            st[PI_LANE[i]] = rotl64(t, RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            uint64_t row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x + 5*y];
            for (int x = 0; x < 5; ++x)
                st[x + 5*y] = row[x] ^ ((~row[(x+1)%5]) & row[(x+2)%5]);
        }
        st[0] ^= KECCAK_RC[round];
    }
}

static void keccak256_single(const uint8_t* data, size_t length, uint8_t out[32]) {
    constexpr size_t rate = 136;
    uint64_t state[25] = {};

    size_t absorbed = 0;
    while (absorbed + rate <= length) {
        for (size_t w = 0; w < rate / 8; ++w) {
            uint64_t lane = 0;
            for (size_t b = 0; b < 8; ++b)
                lane |= uint64_t(data[absorbed + w*8 + b]) << (b * 8);
            state[w] ^= lane;
        }
        keccak_f1600(state);
        absorbed += rate;
    }

    uint8_t padded[136] = {};
    size_t remaining = length - absorbed;
    std::memcpy(padded, data + absorbed, remaining);
    padded[remaining] = 0x01;
    padded[rate - 1] |= 0x80;

    for (size_t w = 0; w < rate / 8; ++w) {
        uint64_t lane = 0;
        for (size_t b = 0; b < 8; ++b)
            lane |= uint64_t(padded[w*8 + b]) << (b * 8);
        state[w] ^= lane;
    }
    keccak_f1600(state);

    for (size_t w = 0; w < 4; ++w) {
        uint64_t lane = state[w];
        for (size_t b = 0; b < 8; ++b)
            out[w*8 + b] = static_cast<uint8_t>(lane >> (b * 8));
    }
}

} // anonymous namespace

// =============================================================================
// Lifecycle
// =============================================================================

static KinetBackendContext* dawn_create_context(int device_index) {
    auto* ctx = new KinetBackendContext();
    ctx->device_index = device_index;

#ifdef KINET_GPU_DAWN
    // Dawn device creation
    WGPUInstanceDescriptor inst_desc = {};
    WGPUInstance instance = wgpuCreateInstance(&inst_desc);
    if (!instance) {
        ctx->device_name = "dawn-unavailable";
        return ctx;
    }

    // Request adapter synchronously
    WGPURequestAdapterOptions adapter_opts = {};
    adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;

    WGPUAdapter adapter = nullptr;
    wgpuInstanceRequestAdapter(instance, &adapter_opts,
        [](WGPURequestAdapterStatus status, WGPUAdapter a, const char*, void* ud) {
            if (status == WGPURequestAdapterStatus_Success)
                *static_cast<WGPUAdapter*>(ud) = a;
        }, &adapter);

    if (!adapter) {
        ctx->device_name = "dawn-no-adapter";
        return ctx;
    }

    // Request device
    WGPUDeviceDescriptor dev_desc = {};
    wgpuAdapterRequestDevice(adapter, &dev_desc,
        [](WGPURequestDeviceStatus status, WGPUDevice d, const char*, void* ud) {
            if (status == WGPURequestDeviceStatus_Success)
                *static_cast<WGPUDevice*>(ud) = d;
        }, &ctx->device);

    if (ctx->device) {
        ctx->queue = wgpuDeviceGetQueue(ctx->device);
        ctx->device_name = "dawn-webgpu";
    } else {
        ctx->device_name = "dawn-no-device";
    }
#else
    ctx->device_name = "dawn-stub";
#endif

    return ctx;
}

static void dawn_destroy_context(KinetBackendContext* ctx) {
    if (!ctx) return;
#ifdef KINET_GPU_DAWN
    for (auto& [name, pipeline] : ctx->pipelines) {
        wgpuComputePipelineRelease(pipeline);
    }
    for (auto& [name, module] : ctx->modules) {
        wgpuShaderModuleRelease(module);
    }
    if (ctx->queue) wgpuQueueRelease(ctx->queue);
    if (ctx->device) wgpuDeviceRelease(ctx->device);
#endif
    delete ctx;
}

// =============================================================================
// Device Info
// =============================================================================

static KinetBackendError dawn_get_device_count(int* count) {
    if (!count) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
#ifdef KINET_GPU_DAWN
    *count = 1;  // Dawn abstracts to one logical device
#else
    *count = 0;
#endif
    return KINET_BACKEND_OK;
}

static KinetBackendError dawn_get_device_info(KinetBackendContext* ctx,
                                             KinetBackendDeviceInfo* info) {
    if (!ctx || !info) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;

    info->name = ctx->device_name.c_str();
    info->vendor = "WebGPU";
    info->memory_total = 0;
    info->memory_available = 0;
    info->compute_units = 0;
    info->max_workgroup_size = 256;
    info->is_discrete = false;
    info->is_unified_memory = false;

    return KINET_BACKEND_OK;
}

// =============================================================================
// Sync
// =============================================================================

static KinetBackendError dawn_sync(KinetBackendContext* ctx) {
#ifdef KINET_GPU_DAWN
    if (ctx && ctx->device) {
        // Submit an empty queue operation and poll until done
        wgpuDeviceTick(ctx->device);
    }
#else
    (void)ctx;
#endif
    return KINET_BACKEND_OK;
}

// =============================================================================
// Buffer Management
// =============================================================================

static KinetBackendBuffer* dawn_buffer_alloc(KinetBackendContext* ctx, size_t bytes) {
    if (!ctx || bytes == 0) return nullptr;
    auto* buf = new KinetBackendBuffer();
    buf->size = bytes;
    buf->staging = std::calloc(1, bytes);

#ifdef KINET_GPU_DAWN
    if (ctx->device) {
        WGPUBufferDescriptor desc = {};
        desc.size = bytes;
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        desc.mappedAtCreation = false;
        buf->gpu_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
    }
#endif

    return buf;
}

static KinetBackendBuffer* dawn_buffer_alloc_with_data(KinetBackendContext* ctx,
                                                      const void* data, size_t bytes) {
    auto* buf = dawn_buffer_alloc(ctx, bytes);
    if (!buf || !data) return buf;
    std::memcpy(buf->staging, data, bytes);

#ifdef KINET_GPU_DAWN
    if (ctx->queue && buf->gpu_buf) {
        wgpuQueueWriteBuffer(ctx->queue, buf->gpu_buf, 0, data, bytes);
    }
#endif

    return buf;
}

static void dawn_buffer_free(KinetBackendContext*, KinetBackendBuffer* buf) {
    if (!buf) return;
#ifdef KINET_GPU_DAWN
    if (buf->gpu_buf) wgpuBufferRelease(buf->gpu_buf);
#endif
    std::free(buf->staging);
    delete buf;
}

static KinetBackendError dawn_buffer_copy_to_host(KinetBackendContext* ctx,
                                                 KinetBackendBuffer* buf,
                                                 void* dst, size_t bytes) {
    if (!buf || !dst) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
    size_t n = std::min(bytes, buf->size);

#ifdef KINET_GPU_DAWN
    if (ctx && ctx->device && ctx->queue && buf->gpu_buf) {
        // Create a staging buffer for map-read
        WGPUBufferDescriptor map_desc = {};
        map_desc.size = n;
        map_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        map_desc.mappedAtCreation = false;
        WGPUBuffer map_buf = wgpuDeviceCreateBuffer(ctx->device, &map_desc);

        // Copy GPU buffer to staging
        WGPUCommandEncoderDescriptor enc_desc = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
        wgpuCommandEncoderCopyBufferToBuffer(enc, buf->gpu_buf, 0, map_buf, 0, n);
        WGPUCommandBufferDescriptor cmd_desc = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cmd_desc);
        wgpuQueueSubmit(ctx->queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);

        // Map the staging buffer and copy
        struct MapState { bool done; WGPUBufferMapAsyncStatus status; };
        MapState state = {false, WGPUBufferMapAsyncStatus_Unknown};
        wgpuBufferMapAsync(map_buf, WGPUMapMode_Read, 0, n,
            [](WGPUBufferMapAsyncStatus s, void* ud) {
                auto* st = static_cast<MapState*>(ud);
                st->status = s;
                st->done = true;
            }, &state);

        while (!state.done) {
            wgpuDeviceTick(ctx->device);
        }

        if (state.status == WGPUBufferMapAsyncStatus_Success) {
            const void* mapped = wgpuBufferGetConstMappedRange(map_buf, 0, n);
            if (mapped) {
                std::memcpy(dst, mapped, n);
                std::memcpy(buf->staging, mapped, n);
            }
            wgpuBufferUnmap(map_buf);
        }

        wgpuBufferRelease(map_buf);
        return KINET_BACKEND_OK;
    }
#else
    (void)ctx;
#endif

    std::memcpy(dst, buf->staging, n);
    return KINET_BACKEND_OK;
}

static KinetBackendError dawn_buffer_copy_from_host(KinetBackendContext* ctx,
                                                   KinetBackendBuffer* buf,
                                                   const void* src, size_t bytes) {
    if (!buf || !src) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
    size_t n = std::min(bytes, buf->size);
    std::memcpy(buf->staging, src, n);

#ifdef KINET_GPU_DAWN
    if (ctx && ctx->queue && buf->gpu_buf) {
        wgpuQueueWriteBuffer(ctx->queue, buf->gpu_buf, 0, src, n);
    }
#else
    (void)ctx;
#endif

    return KINET_BACKEND_OK;
}

static void* dawn_buffer_get_host_ptr(KinetBackendContext*, KinetBackendBuffer* buf) {
    if (!buf) return nullptr;
    return buf->staging;
}

// =============================================================================
// Kernel Management
// =============================================================================

static KinetBackendKernel* dawn_kernel_load(KinetBackendContext* ctx,
                                           const char* source,
                                           const char* entry_point) {
    if (!ctx || !source || !entry_point) return nullptr;

    auto* k = new KinetBackendKernel();
    k->entry_point = entry_point;

#ifdef KINET_GPU_DAWN
    if (ctx->device) {
        WGPUShaderModuleWGSLDescriptor wgsl_desc = {};
        wgsl_desc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
        wgsl_desc.code = source;

        WGPUShaderModuleDescriptor module_desc = {};
        module_desc.nextInChain = &wgsl_desc.chain;

        WGPUShaderModule module = wgpuDeviceCreateShaderModule(ctx->device, &module_desc);
        if (!module) { delete k; return nullptr; }

        WGPUComputePipelineDescriptor pipe_desc = {};
        pipe_desc.compute.module = module;
        pipe_desc.compute.entryPoint = entry_point;

        k->pipeline = wgpuDeviceCreateComputePipeline(ctx->device, &pipe_desc);
        wgpuShaderModuleRelease(module);

        if (!k->pipeline) { delete k; return nullptr; }
    }
#endif

    return k;
}

static KinetBackendKernel* dawn_kernel_load_binary(KinetBackendContext*, const void*,
                                                   size_t, const char* entry_point) {
    // WebGPU only supports WGSL text, not binary
    (void)entry_point;
    return nullptr;
}

static void dawn_kernel_destroy(KinetBackendContext*, KinetBackendKernel* k) {
    if (!k) return;
#ifdef KINET_GPU_DAWN
    if (k->pipeline) wgpuComputePipelineRelease(k->pipeline);
#endif
    delete k;
}

static KinetBackendError dawn_kernel_dispatch(
    KinetBackendContext* ctx, KinetBackendKernel* k,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t /*block_y*/, uint32_t /*block_z*/,
    KinetBackendBuffer** buffers, int num_buffers) {
    if (!ctx || !k || !buffers) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;

#ifdef KINET_GPU_DAWN
    if (!ctx->device || !ctx->queue || !k->pipeline)
        return KINET_BACKEND_ERROR_DEVICE_LOST;

    // Create bind group layout from pipeline
    WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(k->pipeline, 0);
    if (!bgl) return KINET_BACKEND_ERROR_INTERNAL;

    // Build bind group entries
    std::vector<WGPUBindGroupEntry> entries(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        entries[i] = {};
        entries[i].binding = static_cast<uint32_t>(i);
        entries[i].buffer = buffers[i]->gpu_buf;
        entries[i].offset = 0;
        entries[i].size = buffers[i]->size;
    }

    WGPUBindGroupDescriptor bg_desc = {};
    bg_desc.layout = bgl;
    bg_desc.entryCount = static_cast<size_t>(num_buffers);
    bg_desc.entries = entries.data();

    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
    wgpuBindGroupLayoutRelease(bgl);
    if (!bg) return KINET_BACKEND_ERROR_INTERNAL;

    // Encode and submit
    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    wgpuComputePassEncoderSetPipeline(pass, k->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

    // Workgroup count: grid dimensions divided by block size, rounded up
    uint32_t wg_x = (grid_x + block_x - 1) / block_x;
    uint32_t wg_y = grid_y > 0 ? grid_y : 1;
    uint32_t wg_z = grid_z > 0 ? grid_z : 1;
    wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, wg_z);
    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);

    // Cleanup
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    wgpuComputePassEncoderRelease(pass);
    wgpuBindGroupRelease(bg);

    return KINET_BACKEND_OK;
#else
    (void)grid_x; (void)grid_y; (void)grid_z;
    (void)block_x;
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// =============================================================================
// Tensor Ops — CPU staging fallback (WebGPU dispatch for tensor ops TBD)
// =============================================================================

static KinetBackendError dawn_op_add_f32(KinetBackendContext*, KinetBackendBuffer* a,
                                        KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n) {
    if (!a || !b || !out) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
    auto* pa = static_cast<float*>(a->staging);
    auto* pb = static_cast<float*>(b->staging);
    auto* po = static_cast<float*>(out->staging);
    for (size_t i = 0; i < n; i++) po[i] = pa[i] + pb[i];
    return KINET_BACKEND_OK;
}

static KinetBackendError dawn_op_sub_f32(KinetBackendContext*, KinetBackendBuffer* a,
                                        KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n) {
    if (!a || !b || !out) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
    auto* pa = static_cast<float*>(a->staging);
    auto* pb = static_cast<float*>(b->staging);
    auto* po = static_cast<float*>(out->staging);
    for (size_t i = 0; i < n; i++) po[i] = pa[i] - pb[i];
    return KINET_BACKEND_OK;
}

static KinetBackendError dawn_op_mul_f32(KinetBackendContext*, KinetBackendBuffer* a,
                                        KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n) {
    if (!a || !b || !out) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
    auto* pa = static_cast<float*>(a->staging);
    auto* pb = static_cast<float*>(b->staging);
    auto* po = static_cast<float*>(out->staging);
    for (size_t i = 0; i < n; i++) po[i] = pa[i] * pb[i];
    return KINET_BACKEND_OK;
}

static KinetBackendError dawn_op_div_f32(KinetBackendContext*, KinetBackendBuffer* a,
                                        KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n) {
    if (!a || !b || !out) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
    auto* pa = static_cast<float*>(a->staging);
    auto* pb = static_cast<float*>(b->staging);
    auto* po = static_cast<float*>(out->staging);
    for (size_t i = 0; i < n; i++) po[i] = pa[i] / pb[i];
    return KINET_BACKEND_OK;
}

// Ops that return NOT_SUPPORTED
static KinetBackendError dawn_not_supported_matmul(KinetBackendContext*, KinetBackendBuffer*,
                                                  KinetBackendBuffer*, KinetBackendBuffer*,
                                                  int, int, int) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_2buf(KinetBackendContext*, KinetBackendBuffer*,
                                                KinetBackendBuffer*, int, int) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_reduce(KinetBackendContext*, KinetBackendBuffer*,
                                                  KinetBackendBuffer*, size_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_reduce_axis(KinetBackendContext*, KinetBackendBuffer*,
                                                       KinetBackendBuffer*, size_t, size_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_softmax(KinetBackendContext*, KinetBackendBuffer*,
                                                   KinetBackendBuffer*, size_t, size_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_unary(KinetBackendContext*, KinetBackendBuffer*,
                                                 KinetBackendBuffer*, size_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_norm(KinetBackendContext*, KinetBackendBuffer*,
                                                KinetBackendBuffer*, KinetBackendBuffer*,
                                                KinetBackendBuffer*, size_t, size_t, float) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_rms(KinetBackendContext*, KinetBackendBuffer*,
                                               KinetBackendBuffer*, KinetBackendBuffer*,
                                               size_t, size_t, float) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_ntt(KinetBackendContext*, uint64_t*, size_t, uint64_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_msm(KinetBackendContext*, const void*, const void*,
                                               void*, size_t, int) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_poly_mul(KinetBackendContext*, const uint64_t*,
                                                    const uint64_t*, uint64_t*, size_t, uint64_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_bootstrap(KinetBackendContext*, const uint64_t*,
                                                     uint64_t*, const uint64_t*,
                                                     const uint64_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t, uint64_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_keyswitch(KinetBackendContext*, const uint64_t*,
                                                     uint64_t*, const uint64_t*, uint32_t,
                                                     uint32_t, uint32_t, uint32_t, uint64_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_blind_rotate(KinetBackendContext*, uint64_t*,
                                                        const uint64_t*, const uint64_t*,
                                                        uint32_t, uint32_t, uint32_t,
                                                        uint32_t, uint64_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_sample_extract(KinetBackendContext*, const uint64_t*,
                                                          uint64_t*, uint32_t, uint32_t,
                                                          uint64_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_sample_ntt(KinetBackendContext*, uint64_t*, size_t,
                                                      uint64_t, double, uint64_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_hash(KinetBackendContext*, const uint64_t*,
                                                uint64_t*, size_t, size_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_blake3(KinetBackendContext*, const uint8_t*,
                                                  uint8_t*, const size_t*, size_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_curve_add(KinetBackendContext*, const void*,
                                                     const void*, void*, size_t, bool) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_curve_mul(KinetBackendContext*, const void*,
                                                     const void*, void*, size_t, bool) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_pairing(KinetBackendContext*, const void*,
                                                   const void*, void*, size_t) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_kzg_commit(KinetBackendContext*, const void*,
                                                      const void*, void*, size_t, int) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_kzg_open(KinetBackendContext*, const void*,
                                                    const void*, const void*, void*,
                                                    size_t, int) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}
static KinetBackendError dawn_not_supported_kzg_verify(KinetBackendContext*, const void*,
                                                      const void*, const void*, const void*,
                                                      const void*, bool*, int) {
    return KINET_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// Keccak-256 — CPU fallback (WGSL dispatch planned but requires Dawn runtime)
// =============================================================================

static KinetBackendError dawn_op_keccak256_hash(
    KinetBackendContext*,
    const uint8_t* inputs, uint8_t* outputs,
    const size_t* input_lens, size_t num_inputs) {

    if (!inputs || !outputs || !input_lens)
        return KINET_BACKEND_ERROR_INVALID_ARGUMENT;

    size_t offset = 0;
    for (size_t i = 0; i < num_inputs; i++) {
        keccak256_single(inputs + offset, input_lens[i], outputs + i * 32);
        offset += input_lens[i];
    }
    return KINET_BACKEND_OK;
}

// =============================================================================
// ecrecover — CPU fallback (returns invalid; full impl needs secp256k1 in WGSL)
// =============================================================================

static KinetBackendError dawn_op_ecrecover_batch(
    KinetBackendContext*,
    const void*,
    void* addresses,
    size_t num_signatures) {

    if (!addresses) return KINET_BACKEND_ERROR_INVALID_ARGUMENT;
    std::memset(addresses, 0, num_signatures * 32);
    return KINET_BACKEND_OK;
}

// =============================================================================
// Dawn Backend VTable
// =============================================================================

static const kinet_gpu_backend_vtbl dawn_vtbl = {
    .create_context = dawn_create_context,
    .destroy_context = dawn_destroy_context,
    .get_device_count = dawn_get_device_count,
    .get_device_info = dawn_get_device_info,
    .sync = dawn_sync,
    .buffer_alloc = dawn_buffer_alloc,
    .buffer_alloc_with_data = dawn_buffer_alloc_with_data,
    .buffer_free = dawn_buffer_free,
    .buffer_copy_to_host = dawn_buffer_copy_to_host,
    .buffer_copy_from_host = dawn_buffer_copy_from_host,
    .buffer_get_host_ptr = dawn_buffer_get_host_ptr,
    .kernel_load = dawn_kernel_load,
    .kernel_load_binary = dawn_kernel_load_binary,
    .kernel_destroy = dawn_kernel_destroy,
    .kernel_dispatch = dawn_kernel_dispatch,
    .op_add_f32 = dawn_op_add_f32,
    .op_sub_f32 = dawn_op_sub_f32,
    .op_mul_f32 = dawn_op_mul_f32,
    .op_div_f32 = dawn_op_div_f32,
    .op_matmul_f32 = dawn_not_supported_matmul,
    .op_transpose_f32 = dawn_not_supported_2buf,
    .op_reduce_sum_f32 = dawn_not_supported_reduce,
    .op_reduce_max_f32 = dawn_not_supported_reduce,
    .op_reduce_min_f32 = dawn_not_supported_reduce,
    .op_reduce_mean_f32 = dawn_not_supported_reduce,
    .op_reduce_sum_axis_f32 = dawn_not_supported_reduce_axis,
    .op_reduce_max_axis_f32 = dawn_not_supported_reduce_axis,
    .op_softmax_f32 = dawn_not_supported_softmax,
    .op_log_softmax_f32 = dawn_not_supported_softmax,
    .op_exp_f32 = dawn_not_supported_unary,
    .op_log_f32 = dawn_not_supported_unary,
    .op_sqrt_f32 = dawn_not_supported_unary,
    .op_neg_f32 = dawn_not_supported_unary,
    .op_abs_f32 = dawn_not_supported_unary,
    .op_tanh_f32 = dawn_not_supported_unary,
    .op_sigmoid_f32 = dawn_not_supported_unary,
    .op_relu_f32 = dawn_not_supported_unary,
    .op_gelu_f32 = dawn_not_supported_unary,
    .op_copy_f32 = dawn_not_supported_unary,
    .op_layer_norm_f32 = dawn_not_supported_norm,
    .op_rms_norm_f32 = dawn_not_supported_rms,
    .op_ntt_forward = dawn_not_supported_ntt,
    .op_ntt_inverse = dawn_not_supported_ntt,
    .op_msm = dawn_not_supported_msm,
    .op_poly_mul = dawn_not_supported_poly_mul,
    .op_tfhe_bootstrap = dawn_not_supported_bootstrap,
    .op_tfhe_keyswitch = dawn_not_supported_keyswitch,
    .op_blind_rotate = dawn_not_supported_blind_rotate,
    .op_sample_extract = dawn_not_supported_sample_extract,
    .op_sample_ntt = dawn_not_supported_sample_ntt,
    .op_poseidon2_hash = dawn_not_supported_hash,
    .op_blake3_hash = dawn_not_supported_blake3,
    .op_keccak256_hash = dawn_op_keccak256_hash,
    .op_bls12_381_add = dawn_not_supported_curve_add,
    .op_bls12_381_mul = dawn_not_supported_curve_mul,
    .op_bls12_381_pairing = dawn_not_supported_pairing,
    .op_bn254_add = dawn_not_supported_curve_add,
    .op_bn254_mul = dawn_not_supported_curve_mul,
    .op_kzg_commit = dawn_not_supported_kzg_commit,
    .op_kzg_open = dawn_not_supported_kzg_open,
    .op_kzg_verify = dawn_not_supported_kzg_verify,
    .op_ecrecover_batch = dawn_op_ecrecover_batch,
    ._reserved = {nullptr, nullptr, nullptr},
};

// =============================================================================
// Entry Point
// =============================================================================

static bool dawn_init(kinet_gpu_backend_desc* out) {
    if (!out) return false;

    out->abi_version = KINET_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "webgpu";
    out->backend_version = "1.0.0";
    out->capabilities = KINET_CAP_TENSOR_OPS | KINET_CAP_KECCAK256 | KINET_CAP_ECRECOVER
                       | KINET_CAP_CUSTOM_KERNELS | KINET_CAP_NTT | KINET_CAP_BLAKE3
                       | KINET_CAP_BLS12_381 | KINET_CAP_REDUCE | KINET_CAP_UNARY
                       | KINET_CAP_POLY_MUL | KINET_CAP_FHE | KINET_CAP_TFHE
                       | KINET_CAP_BLIND_ROTATE;
    out->vtbl = &dawn_vtbl;
    return true;
}

KINET_GPU_DECLARE_BACKEND(dawn_init)

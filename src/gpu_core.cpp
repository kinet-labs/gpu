// Core GPU Library - Plugin-based backend management

#include "gpu_internal.h"
#include "plugin_loader.hpp"
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

// =============================================================================
// Built-in CPU backend declaration
// =============================================================================

extern "C" bool cpu_backend_init(kinet_gpu_backend_desc* out);

// =============================================================================
// KinetGPU Implementation (declared in gpu_internal.h)
// =============================================================================

KinetGPU::~KinetGPU() {
    if (ctx && vtbl && vtbl->destroy_context) {
        vtbl->destroy_context(ctx);
    }
}

void KinetGPU::set_error(const char* msg) {
    std::lock_guard<std::mutex> lock(mutex);
    last_error = msg ? msg : "";
}

// =============================================================================
// Tensor wrapper (bridges plugin buffers to public API)
// =============================================================================

struct KinetTensor {
    std::vector<int64_t> shape;
    KinetDtype dtype;
    std::vector<uint8_t> host_data;
    KinetBackendBuffer* device_buffer = nullptr;
    const kinet_gpu_backend_vtbl* vtbl = nullptr;
    KinetBackendContext* ctx = nullptr;

    // Returns total number of elements, or -1 on overflow
    int64_t size() const {
        int64_t s = 1;
        for (auto d : shape) {
            // Check for overflow before multiplication
            if (d <= 0) return 0;
            if (s > std::numeric_limits<int64_t>::max() / d) {
                return -1;  // Overflow
            }
            s *= d;
        }
        return s;
    }

    // Safe size check - returns false on overflow or invalid size
    bool size_valid() const {
        return size() > 0;
    }

    size_t element_size() const {
        switch (dtype) {
            case KINET_FLOAT32: case KINET_INT32: case KINET_UINT32: return 4;
            case KINET_FLOAT16: case KINET_BFLOAT16: return 2;
            case KINET_INT64: case KINET_UINT64: return 8;
            case KINET_BOOL: return 1;
            default: return 0;
        }
    }

    ~KinetTensor() {
        if (device_buffer && vtbl && vtbl->buffer_free && ctx) {
            vtbl->buffer_free(ctx, device_buffer);
        }
    }
};

// =============================================================================
// Global initialization
// =============================================================================

static std::once_flag g_init_flag;
static kinet_gpu_backend_desc g_cpu_backend = {};

static void global_init() {
    // Initialize built-in CPU backend
    cpu_backend_init(&g_cpu_backend);

    // Scan for plugins in all search paths
    auto& loader = kinet::gpu::PluginLoader::instance();

    // Try to load each backend (will search all paths)
    loader.load_backend("metal");
    loader.load_backend("cuda");
    loader.load_backend("webgpu");
}

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

KinetGPU* kinet_gpu_create(void) {
    return kinet_gpu_create_with_backend(KINET_BACKEND_AUTO);
}

KinetGPU* kinet_gpu_create_with_backend(KinetBackend backend) {
    return kinet_gpu_create_with_device(backend, 0);
}

KinetGPU* kinet_gpu_create_with_device(KinetBackend backend, int device_index) {
    std::call_once(g_init_flag, global_init);

    auto gpu = new KinetGPU();
    auto& loader = kinet::gpu::PluginLoader::instance();

    const kinet_gpu_backend_vtbl* vtbl = nullptr;
    std::string name;

    if (backend == KINET_BACKEND_AUTO) {
        // Try to find the best available backend
        if (auto* best = loader.get_best_backend()) {
            vtbl = best->desc.vtbl;
            name = best->name;
        }
        // Fall back to CPU
        if (!vtbl) {
            vtbl = g_cpu_backend.vtbl;
            name = "cpu";
        }
    } else {
        // Specific backend requested
        switch (backend) {
            case KINET_BACKEND_CPU:
                vtbl = g_cpu_backend.vtbl;
                name = "cpu";
                break;

            case KINET_BACKEND_METAL:
                if (!loader.is_available("metal")) {
                    loader.load_backend("metal");
                }
                if (auto* b = loader.get_backend("metal")) {
                    vtbl = b->desc.vtbl;
                    name = "metal";
                }
                break;

            case KINET_BACKEND_CUDA:
                if (!loader.is_available("cuda")) {
                    loader.load_backend("cuda");
                }
                if (auto* b = loader.get_backend("cuda")) {
                    vtbl = b->desc.vtbl;
                    name = "cuda";
                }
                break;

            case KINET_BACKEND_DAWN:
                if (!loader.is_available("webgpu")) {
                    loader.load_backend("webgpu");
                }
                if (auto* b = loader.get_backend("webgpu")) {
                    vtbl = b->desc.vtbl;
                    name = "webgpu";
                }
                break;

            default:
                break;
        }
    }

    if (!vtbl) {
        // Fall back to CPU
        vtbl = g_cpu_backend.vtbl;
        name = "cpu";
    }

    gpu->vtbl = vtbl;
    gpu->backend_name = name;

    // Create context
    if (vtbl && vtbl->create_context) {
        gpu->ctx = vtbl->create_context(device_index);
        if (!gpu->ctx) {
            gpu->set_error("Failed to create backend context");
            // Fall back to CPU
            gpu->vtbl = g_cpu_backend.vtbl;
            gpu->backend_name = "cpu";
            gpu->ctx = g_cpu_backend.vtbl->create_context(0);
        }
    }

    return gpu;
}

void kinet_gpu_destroy(KinetGPU* gpu) {
    delete gpu;
}

KinetBackend kinet_gpu_backend(KinetGPU* gpu) {
    if (!gpu) return KINET_BACKEND_CPU;
    if (gpu->backend_name == "cpu") return KINET_BACKEND_CPU;
    if (gpu->backend_name == "metal") return KINET_BACKEND_METAL;
    if (gpu->backend_name == "cuda") return KINET_BACKEND_CUDA;
    if (gpu->backend_name == "webgpu") return KINET_BACKEND_DAWN;
    return KINET_BACKEND_CPU;
}

const char* kinet_gpu_backend_name(KinetGPU* gpu) {
    return gpu ? gpu->backend_name.c_str() : "cpu";
}

KinetError kinet_gpu_set_backend(KinetGPU* gpu, KinetBackend backend) {
    if (!gpu) return KINET_ERROR_INVALID_ARGUMENT;

    // Create new context for requested backend
    auto* new_gpu = kinet_gpu_create_with_backend(backend);
    if (!new_gpu || (new_gpu->backend_name == "cpu" && backend != KINET_BACKEND_CPU && backend != KINET_BACKEND_AUTO)) {
        delete new_gpu;
        return KINET_ERROR_BACKEND_NOT_AVAILABLE;
    }

    // Swap internals
    std::swap(gpu->vtbl, new_gpu->vtbl);
    std::swap(gpu->ctx, new_gpu->ctx);
    std::swap(gpu->backend_name, new_gpu->backend_name);

    delete new_gpu;
    return KINET_OK;
}

KinetError kinet_gpu_device_info(KinetGPU* gpu, KinetDeviceInfo* info) {
    if (!gpu || !gpu->vtbl || !info) return KINET_ERROR_INVALID_ARGUMENT;

    KinetBackendDeviceInfo binfo = {};
    KinetBackendError err = gpu->vtbl->get_device_info(gpu->ctx, &binfo);
    if (err != KINET_BACKEND_OK) return static_cast<KinetError>(err);

    info->backend = kinet_gpu_backend(gpu);
    info->index = 0;
    info->name = binfo.name;
    info->vendor = binfo.vendor;
    info->memory_total = binfo.memory_total;
    info->memory_available = binfo.memory_available;
    info->compute_units = binfo.compute_units;
    info->max_workgroup_size = binfo.max_workgroup_size;
    info->is_discrete = binfo.is_discrete;
    info->is_unified_memory = binfo.is_unified_memory;

    return KINET_OK;
}

KinetError kinet_gpu_sync(KinetGPU* gpu) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    return static_cast<KinetError>(gpu->vtbl->sync(gpu->ctx));
}

const char* kinet_gpu_error(KinetGPU* gpu) {
    return gpu ? gpu->last_error.c_str() : "null gpu";
}

// Backend query
int kinet_backend_count(void) {
    std::call_once(g_init_flag, global_init);
    return static_cast<int>(kinet::gpu::PluginLoader::instance().available_backends().size());
}

bool kinet_backend_available(KinetBackend backend) {
    std::call_once(g_init_flag, global_init);
    auto& loader = kinet::gpu::PluginLoader::instance();

    switch (backend) {
        case KINET_BACKEND_CPU: return true;
        case KINET_BACKEND_METAL: return loader.is_available("metal") || loader.load_backend("metal");
        case KINET_BACKEND_CUDA: return loader.is_available("cuda") || loader.load_backend("cuda");
        case KINET_BACKEND_DAWN: return loader.is_available("webgpu") || loader.load_backend("webgpu");
        default: return false;
    }
}

const char* kinet_backend_name(KinetBackend backend) {
    switch (backend) {
        case KINET_BACKEND_AUTO: return "auto";
        case KINET_BACKEND_CPU: return "cpu";
        case KINET_BACKEND_METAL: return "metal";
        case KINET_BACKEND_CUDA: return "cuda";
        case KINET_BACKEND_DAWN: return "webgpu";
        default: return "unknown";
    }
}

int kinet_device_count(KinetBackend backend) {
    std::call_once(g_init_flag, global_init);
    auto& loader = kinet::gpu::PluginLoader::instance();

    if (backend == KINET_BACKEND_CPU) {
        return 1;  // CPU always has one "device"
    }

    const char* name = nullptr;
    switch (backend) {
        case KINET_BACKEND_METAL: name = "metal"; break;
        case KINET_BACKEND_CUDA: name = "cuda"; break;
        case KINET_BACKEND_DAWN: name = "webgpu"; break;
        default: return 0;
    }

    if (!loader.is_available(name)) {
        if (!loader.load_backend(name)) {
            return 0;
        }
    }

    const auto* b = loader.get_backend(name);
    if (!b || !b->desc.vtbl || !b->desc.vtbl->get_device_count) {
        return 0;
    }

    int count = 0;
    if (b->desc.vtbl->get_device_count(&count) == KINET_BACKEND_OK) {
        return count;
    }
    return 0;
}

KinetError kinet_device_info(KinetBackend backend, int index, KinetDeviceInfo* info) {
    if (!info) return KINET_ERROR_INVALID_ARGUMENT;

    std::call_once(g_init_flag, global_init);
    auto& loader = kinet::gpu::PluginLoader::instance();

    if (backend == KINET_BACKEND_CPU) {
        // CPU backend info
        info->backend = KINET_BACKEND_CPU;
        info->index = 0;
        info->name = "CPU";
        info->vendor = "Host";
        info->memory_total = 0;
        info->memory_available = 0;
        info->is_discrete = false;
        info->is_unified_memory = true;
        info->compute_units = 1;
        info->max_workgroup_size = 1;
        return KINET_OK;
    }

    const char* name = nullptr;
    switch (backend) {
        case KINET_BACKEND_METAL: name = "metal"; break;
        case KINET_BACKEND_CUDA: name = "cuda"; break;
        case KINET_BACKEND_DAWN: name = "webgpu"; break;
        default: return KINET_ERROR_BACKEND_NOT_AVAILABLE;
    }

    if (!loader.is_available(name)) {
        if (!loader.load_backend(name)) {
            return KINET_ERROR_BACKEND_NOT_AVAILABLE;
        }
    }

    const auto* b = loader.get_backend(name);
    if (!b || !b->desc.vtbl) {
        return KINET_ERROR_BACKEND_NOT_AVAILABLE;
    }

    // Create temporary context for device info query
    // NOTE: The returned name/vendor pointers must be static strings in the backend.
    // Backend implementations must not return pointers to context-owned memory.
    if (!b->desc.vtbl->create_context) {
        return KINET_ERROR_NOT_SUPPORTED;
    }

    KinetBackendContext* ctx = b->desc.vtbl->create_context(index);
    if (!ctx) {
        return KINET_ERROR_DEVICE_NOT_FOUND;
    }

    KinetBackendDeviceInfo binfo = {};
    KinetBackendError err = KINET_BACKEND_OK;
    if (b->desc.vtbl->get_device_info) {
        err = b->desc.vtbl->get_device_info(ctx, &binfo);
    }

    // Copy string data before destroying context (in case backend returns
    // context-owned strings - backends should return static strings per contract)
    const char* saved_name = binfo.name;
    const char* saved_vendor = binfo.vendor;

    if (b->desc.vtbl->destroy_context) {
        b->desc.vtbl->destroy_context(ctx);
    }

    if (err != KINET_BACKEND_OK) {
        return static_cast<KinetError>(err);
    }

    info->backend = backend;
    info->index = index;
    info->name = saved_name;
    info->vendor = saved_vendor;
    info->memory_total = binfo.memory_total;
    info->memory_available = binfo.memory_available;
    info->compute_units = binfo.compute_units;
    info->max_workgroup_size = binfo.max_workgroup_size;
    info->is_discrete = binfo.is_discrete;
    info->is_unified_memory = binfo.is_unified_memory;

    return KINET_OK;
}

// =============================================================================
// Tensor Operations
// =============================================================================

KinetTensor* kinet_tensor_zeros(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype) {
    if (!gpu || !gpu->vtbl || !shape || ndim <= 0) return nullptr;

    auto t = new KinetTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    // Check for size overflow
    int64_t total = t->size();
    if (total <= 0) {
        delete t;
        return nullptr;
    }

    size_t elem_size = t->element_size();
    // Check for byte count overflow
    if (static_cast<uint64_t>(total) > std::numeric_limits<size_t>::max() / elem_size) {
        delete t;
        return nullptr;
    }
    size_t bytes = static_cast<size_t>(total) * elem_size;
    t->host_data.resize(bytes, 0);

    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, t->host_data.data(), bytes);
    } else if (gpu->vtbl->buffer_alloc) {
        t->device_buffer = gpu->vtbl->buffer_alloc(gpu->ctx, bytes);
    }

    return t;
}

KinetTensor* kinet_tensor_ones(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype) {
    return kinet_tensor_full(gpu, shape, ndim, dtype, 1.0);
}

KinetTensor* kinet_tensor_full(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype, double value) {
    if (!gpu || !gpu->vtbl || !shape || ndim <= 0) return nullptr;

    auto t = new KinetTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    // Check for size overflow
    int64_t total = t->size();
    if (total <= 0) {
        delete t;
        return nullptr;
    }

    size_t elem_size = t->element_size();
    if (static_cast<uint64_t>(total) > std::numeric_limits<size_t>::max() / elem_size) {
        delete t;
        return nullptr;
    }
    size_t bytes = static_cast<size_t>(total) * elem_size;
    t->host_data.resize(bytes);

    // Fill host data
    if (dtype == KINET_FLOAT32) {
        float v = static_cast<float>(value);
        float* ptr = reinterpret_cast<float*>(t->host_data.data());
        for (int64_t i = 0; i < t->size(); i++) ptr[i] = v;
    }

    // Copy to device
    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, t->host_data.data(), bytes);
    }

    return t;
}

KinetTensor* kinet_tensor_from_data(KinetGPU* gpu, const void* data, const int64_t* shape, int ndim, KinetDtype dtype) {
    if (!gpu || !gpu->vtbl || !data || !shape || ndim <= 0) return nullptr;

    auto t = new KinetTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    // Check for size overflow
    int64_t total = t->size();
    if (total <= 0) {
        delete t;
        return nullptr;
    }

    size_t elem_size = t->element_size();
    if (static_cast<uint64_t>(total) > std::numeric_limits<size_t>::max() / elem_size) {
        delete t;
        return nullptr;
    }
    size_t bytes = static_cast<size_t>(total) * elem_size;
    t->host_data.resize(bytes);
    std::memcpy(t->host_data.data(), data, bytes);

    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, data, bytes);
    }

    return t;
}

void kinet_tensor_destroy(KinetTensor* tensor) {
    delete tensor;
}

int kinet_tensor_ndim(KinetTensor* tensor) {
    return tensor ? static_cast<int>(tensor->shape.size()) : 0;
}

int64_t kinet_tensor_shape(KinetTensor* tensor, int dim) {
    return (tensor && dim >= 0 && dim < static_cast<int>(tensor->shape.size()))
        ? tensor->shape[dim] : 0;
}

int64_t kinet_tensor_size(KinetTensor* tensor) {
    return tensor ? tensor->size() : 0;
}

KinetDtype kinet_tensor_dtype(KinetTensor* tensor) {
    return tensor ? tensor->dtype : KINET_FLOAT32;
}

KinetError kinet_tensor_to_host(KinetTensor* tensor, void* data, size_t size) {
    if (!tensor || !data) return KINET_ERROR_INVALID_ARGUMENT;

    size_t bytes = tensor->size() * tensor->element_size();
    if (size < bytes) return KINET_ERROR_INVALID_ARGUMENT;

    // If we have device buffer, sync from it
    if (tensor->device_buffer && tensor->vtbl && tensor->vtbl->buffer_copy_to_host) {
        KinetBackendError err = tensor->vtbl->buffer_copy_to_host(
            tensor->ctx, tensor->device_buffer, data, bytes
        );
        return static_cast<KinetError>(err);
    }

    // Otherwise copy from host data
    std::memcpy(data, tensor->host_data.data(), bytes);
    return KINET_OK;
}

// Binary operations helper
static KinetTensor* binary_op(KinetGPU* gpu, KinetTensor* a, KinetTensor* b,
                            KinetBackendError (*op)(KinetBackendContext*, KinetBackendBuffer*, KinetBackendBuffer*, KinetBackendBuffer*, size_t)) {
    if (!gpu || !a || !b || a->shape != b->shape) return nullptr;

    auto out = kinet_tensor_zeros(gpu, a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);
    if (!out) return nullptr;

    if (op && a->device_buffer && b->device_buffer && out->device_buffer) {
        KinetBackendError err = op(gpu->ctx, a->device_buffer, b->device_buffer, out->device_buffer, a->size());
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

KinetTensor* kinet_tensor_add(KinetGPU* gpu, KinetTensor* a, KinetTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_add_f32);
}

KinetTensor* kinet_tensor_sub(KinetGPU* gpu, KinetTensor* a, KinetTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_sub_f32);
}

KinetTensor* kinet_tensor_mul(KinetGPU* gpu, KinetTensor* a, KinetTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_mul_f32);
}

KinetTensor* kinet_tensor_div(KinetGPU* gpu, KinetTensor* a, KinetTensor* b) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_div_f32) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_div_f32);
}

KinetTensor* kinet_tensor_matmul(KinetGPU* gpu, KinetTensor* a, KinetTensor* b) {
    if (!gpu || !gpu->vtbl || !a || !b) return nullptr;
    if (a->shape.size() != 2 || b->shape.size() != 2) return nullptr;
    if (a->shape[1] != b->shape[0]) return nullptr;

    int M = static_cast<int>(a->shape[0]);
    int K = static_cast<int>(a->shape[1]);
    int N = static_cast<int>(b->shape[1]);

    int64_t out_shape[2] = {M, N};
    auto out = kinet_tensor_zeros(gpu, out_shape, 2, a->dtype);
    if (!out) return nullptr;

    if (gpu->vtbl->op_matmul_f32 && a->device_buffer && b->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_matmul_f32(
            gpu->ctx, a->device_buffer, b->device_buffer, out->device_buffer, M, K, N
        );
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Unary operations helper
// =============================================================================

static KinetTensor* unary_op(KinetGPU* gpu, KinetTensor* t,
                           KinetBackendError (*op)(KinetBackendContext*, KinetBackendBuffer*, KinetBackendBuffer*, size_t)) {
    if (!gpu || !gpu->vtbl || !t || !op) return nullptr;

    auto out = kinet_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        KinetBackendError err = op(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

KinetTensor* kinet_tensor_neg(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_neg_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_neg_f32);
}

KinetTensor* kinet_tensor_exp(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_exp_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_exp_f32);
}

KinetTensor* kinet_tensor_log(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_log_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_log_f32);
}

KinetTensor* kinet_tensor_sqrt(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_sqrt_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_sqrt_f32);
}

KinetTensor* kinet_tensor_abs(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_abs_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_abs_f32);
}

KinetTensor* kinet_tensor_tanh(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_tanh_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_tanh_f32);
}

KinetTensor* kinet_tensor_sigmoid(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_sigmoid_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_sigmoid_f32);
}

KinetTensor* kinet_tensor_relu(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_relu_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_relu_f32);
}

KinetTensor* kinet_tensor_gelu(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_gelu_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_gelu_f32);
}

// =============================================================================
// Scalar reductions
// =============================================================================

float kinet_tensor_reduce_sum(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_sum_f32) return 0.0f;

    int64_t one = 1;
    auto out = kinet_tensor_zeros(gpu, &one, 1, KINET_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_sum_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    kinet_tensor_to_host(out, &result, sizeof(float));
    kinet_tensor_destroy(out);
    return result;
}

float kinet_tensor_reduce_max(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_max_f32) return 0.0f;

    int64_t one = 1;
    auto out = kinet_tensor_zeros(gpu, &one, 1, KINET_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_max_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    kinet_tensor_to_host(out, &result, sizeof(float));
    kinet_tensor_destroy(out);
    return result;
}

float kinet_tensor_reduce_min(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_min_f32) return 0.0f;

    int64_t one = 1;
    auto out = kinet_tensor_zeros(gpu, &one, 1, KINET_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_min_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    kinet_tensor_to_host(out, &result, sizeof(float));
    kinet_tensor_destroy(out);
    return result;
}

float kinet_tensor_reduce_mean(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_mean_f32) return 0.0f;

    int64_t one = 1;
    auto out = kinet_tensor_zeros(gpu, &one, 1, KINET_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_mean_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    kinet_tensor_to_host(out, &result, sizeof(float));
    kinet_tensor_destroy(out);
    return result;
}

// =============================================================================
// Axis Reduction Helpers
// =============================================================================

// Compute output shape after reducing along specified axes
static std::vector<int64_t> compute_reduced_shape(const std::vector<int64_t>& shape, const int* axes, int naxes) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); i++) {
        if (!reduce_axis[i]) {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);  // Scalar result
    }
    return out_shape;
}

// Compute outer_size and inner_size for contiguous last-axis reduction
// Returns true if reduction can be expressed as (outer_size, inner_size) -> outer_size
static bool can_use_axis_reduction(const std::vector<int64_t>& shape, const int* axes, int naxes,
                                    size_t* outer_size, size_t* inner_size) {
    if (naxes != 1 || shape.empty()) return false;

    int ax = axes[0];
    if (ax < 0) ax += static_cast<int>(shape.size());
    if (ax < 0 || ax >= static_cast<int>(shape.size())) return false;

    // Only support reduction along last axis for backend dispatch
    if (ax != static_cast<int>(shape.size()) - 1) return false;

    *inner_size = static_cast<size_t>(shape.back());
    *outer_size = 1;
    for (size_t i = 0; i < shape.size() - 1; i++) {
        *outer_size *= static_cast<size_t>(shape[i]);
    }
    return true;
}

// CPU fallback for sum reduction along axes
static void cpu_reduce_sum_axes(const float* in, float* out, const std::vector<int64_t>& shape,
                                 const int* axes, int naxes, const std::vector<int64_t>& out_shape) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    // Compute strides for input
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= static_cast<size_t>(shape[i]);
    }

    // Compute output size
    size_t out_size = 1;
    for (auto d : out_shape) out_size *= static_cast<size_t>(d);
    std::memset(out, 0, out_size * sizeof(float));

    // Iterate over input and accumulate
    size_t in_size = stride;
    for (size_t idx = 0; idx < in_size; idx++) {
        // Decompose linear index into multi-index
        size_t tmp = idx;
        size_t out_idx = 0;
        size_t out_stride = 1;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; d--) {
            size_t coord = tmp % static_cast<size_t>(shape[d]);
            tmp /= static_cast<size_t>(shape[d]);
            if (!reduce_axis[d]) {
                // Find position in output
                int out_d = 0;
                for (int k = 0; k < d; k++) {
                    if (!reduce_axis[k]) out_d++;
                }
                // Compute contribution to out_idx
                size_t os = 1;
                for (size_t k = out_d + 1; k < out_shape.size(); k++) {
                    os *= static_cast<size_t>(out_shape[k]);
                }
                out_idx += coord * os;
            }
        }
        out[out_idx] += in[idx];
    }
}

// CPU fallback for max reduction along axes
static void cpu_reduce_max_axes(const float* in, float* out, const std::vector<int64_t>& shape,
                                 const int* axes, int naxes, const std::vector<int64_t>& out_shape) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= static_cast<size_t>(shape[i]);
    }

    size_t out_size = 1;
    for (auto d : out_shape) out_size *= static_cast<size_t>(d);

    // Initialize with -inf
    for (size_t i = 0; i < out_size; i++) {
        out[i] = -std::numeric_limits<float>::infinity();
    }

    size_t in_size = stride;
    for (size_t idx = 0; idx < in_size; idx++) {
        size_t tmp = idx;
        size_t out_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; d--) {
            size_t coord = tmp % static_cast<size_t>(shape[d]);
            tmp /= static_cast<size_t>(shape[d]);
            if (!reduce_axis[d]) {
                int out_d = 0;
                for (int k = 0; k < d; k++) {
                    if (!reduce_axis[k]) out_d++;
                }
                size_t os = 1;
                for (size_t k = out_d + 1; k < out_shape.size(); k++) {
                    os *= static_cast<size_t>(out_shape[k]);
                }
                out_idx += coord * os;
            }
        }
        if (in[idx] > out[out_idx]) {
            out[out_idx] = in[idx];
        }
    }
}

// CPU fallback for min reduction along axes
static void cpu_reduce_min_axes(const float* in, float* out, const std::vector<int64_t>& shape,
                                 const int* axes, int naxes, const std::vector<int64_t>& out_shape) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= static_cast<size_t>(shape[i]);
    }

    size_t out_size = 1;
    for (auto d : out_shape) out_size *= static_cast<size_t>(d);

    for (size_t i = 0; i < out_size; i++) {
        out[i] = std::numeric_limits<float>::infinity();
    }

    size_t in_size = stride;
    for (size_t idx = 0; idx < in_size; idx++) {
        size_t tmp = idx;
        size_t out_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; d--) {
            size_t coord = tmp % static_cast<size_t>(shape[d]);
            tmp /= static_cast<size_t>(shape[d]);
            if (!reduce_axis[d]) {
                int out_d = 0;
                for (int k = 0; k < d; k++) {
                    if (!reduce_axis[k]) out_d++;
                }
                size_t os = 1;
                for (size_t k = out_d + 1; k < out_shape.size(); k++) {
                    os *= static_cast<size_t>(out_shape[k]);
                }
                out_idx += coord * os;
            }
        }
        if (in[idx] < out[out_idx]) {
            out[out_idx] = in[idx];
        }
    }
}

// =============================================================================
// Axis Reduction API
// =============================================================================

KinetTensor* kinet_tensor_sum(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t) return nullptr;

    // Global reduction: axes=null && naxes=0
    if (axes == nullptr || naxes <= 0) {
        // Return scalar with global sum
        float sum = kinet_tensor_reduce_sum(gpu, t);
        int64_t one = 1;
        auto out = kinet_tensor_zeros(gpu, &one, 1, t->dtype);
        if (!out) return nullptr;

        // Set the scalar value
        float* data = &sum;
        out->host_data.resize(sizeof(float));
        std::memcpy(out->host_data.data(), data, sizeof(float));
        if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
            gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer, data, sizeof(float));
        }
        return out;
    }

    std::vector<int64_t> out_shape = compute_reduced_shape(t->shape, axes, naxes);
    auto out = kinet_tensor_zeros(gpu, out_shape.data(), static_cast<int>(out_shape.size()), t->dtype);
    if (!out) return nullptr;

    // Try backend dispatch for single last-axis reduction
    size_t outer_size, inner_size;
    if (gpu->vtbl->op_reduce_sum_axis_f32 &&
        can_use_axis_reduction(t->shape, axes, naxes, &outer_size, &inner_size) &&
        t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_reduce_sum_axis_f32(
            gpu->ctx, t->device_buffer, out->device_buffer, outer_size, inner_size);
        if (err == KINET_BACKEND_OK) return out;
    }

    // CPU fallback: sync input to host, compute, sync output to device
    std::vector<float> in_data(t->size());
    kinet_tensor_to_host(t, in_data.data(), in_data.size() * sizeof(float));

    std::vector<float> out_data(out->size());
    cpu_reduce_sum_axes(in_data.data(), out_data.data(), t->shape, axes, naxes, out_shape);

    // Copy result to output tensor host_data and device
    std::memcpy(out->host_data.data(), out_data.data(), out_data.size() * sizeof(float));
    if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer,
                                          out_data.data(), out_data.size() * sizeof(float));
    }

    return out;
}

KinetTensor* kinet_tensor_mean(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t) return nullptr;

    // Global reduction: axes=null && naxes=0
    if (axes == nullptr || naxes <= 0) {
        float mean = kinet_tensor_reduce_mean(gpu, t);
        int64_t one = 1;
        auto out = kinet_tensor_zeros(gpu, &one, 1, t->dtype);
        if (!out) return nullptr;

        float* data = &mean;
        out->host_data.resize(sizeof(float));
        std::memcpy(out->host_data.data(), data, sizeof(float));
        if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
            gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer, data, sizeof(float));
        }
        return out;
    }

    // Compute sum first
    KinetTensor* sum_tensor = kinet_tensor_sum(gpu, t, axes, naxes);
    if (!sum_tensor) return nullptr;

    // Compute reduction factor
    int64_t reduce_count = 1;
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(t->shape.size());
        if (ax >= 0 && ax < static_cast<int>(t->shape.size())) {
            reduce_count *= t->shape[ax];
        }
    }

    // Divide by reduction count (in-place on host data)
    std::vector<float> data(sum_tensor->size());
    kinet_tensor_to_host(sum_tensor, data.data(), data.size() * sizeof(float));

    float scale = 1.0f / static_cast<float>(reduce_count);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] *= scale;
    }

    std::memcpy(sum_tensor->host_data.data(), data.data(), data.size() * sizeof(float));
    if (sum_tensor->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, sum_tensor->device_buffer,
                                          data.data(), data.size() * sizeof(float));
    }

    return sum_tensor;
}

KinetTensor* kinet_tensor_max(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t || !axes || naxes <= 0) return nullptr;

    std::vector<int64_t> out_shape = compute_reduced_shape(t->shape, axes, naxes);
    auto out = kinet_tensor_zeros(gpu, out_shape.data(), static_cast<int>(out_shape.size()), t->dtype);
    if (!out) return nullptr;

    // Try backend dispatch for single last-axis reduction
    size_t outer_size, inner_size;
    if (gpu->vtbl->op_reduce_max_axis_f32 &&
        can_use_axis_reduction(t->shape, axes, naxes, &outer_size, &inner_size) &&
        t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_reduce_max_axis_f32(
            gpu->ctx, t->device_buffer, out->device_buffer, outer_size, inner_size);
        if (err == KINET_BACKEND_OK) return out;
    }

    // CPU fallback
    std::vector<float> in_data(t->size());
    kinet_tensor_to_host(t, in_data.data(), in_data.size() * sizeof(float));

    std::vector<float> out_data(out->size());
    cpu_reduce_max_axes(in_data.data(), out_data.data(), t->shape, axes, naxes, out_shape);

    std::memcpy(out->host_data.data(), out_data.data(), out_data.size() * sizeof(float));
    if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer,
                                          out_data.data(), out_data.size() * sizeof(float));
    }

    return out;
}

KinetTensor* kinet_tensor_min(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t || !axes || naxes <= 0) return nullptr;

    std::vector<int64_t> out_shape = compute_reduced_shape(t->shape, axes, naxes);
    auto out = kinet_tensor_zeros(gpu, out_shape.data(), static_cast<int>(out_shape.size()), t->dtype);
    if (!out) return nullptr;

    // CPU fallback (no backend op_reduce_min_axis_f32 in vtable)
    std::vector<float> in_data(t->size());
    kinet_tensor_to_host(t, in_data.data(), in_data.size() * sizeof(float));

    std::vector<float> out_data(out->size());
    cpu_reduce_min_axes(in_data.data(), out_data.data(), t->shape, axes, naxes, out_shape);

    std::memcpy(out->host_data.data(), out_data.data(), out_data.size() * sizeof(float));
    if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer,
                                          out_data.data(), out_data.size() * sizeof(float));
    }

    return out;
}

// =============================================================================
// Softmax and normalization
// =============================================================================

KinetTensor* kinet_tensor_softmax(KinetGPU* gpu, KinetTensor* t, int axis) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_softmax_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    // For now, only support last axis softmax
    (void)axis;
    size_t cols = static_cast<size_t>(t->shape.back());
    size_t rows = static_cast<size_t>(t->size() / cols);

    auto out = kinet_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_softmax_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

KinetTensor* kinet_tensor_log_softmax(KinetGPU* gpu, KinetTensor* t, int axis) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_log_softmax_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    (void)axis;
    size_t cols = static_cast<size_t>(t->shape.back());
    size_t rows = static_cast<size_t>(t->size() / cols);

    auto out = kinet_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_log_softmax_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

KinetTensor* kinet_tensor_layer_norm(KinetGPU* gpu, KinetTensor* t, KinetTensor* gamma, KinetTensor* beta, float eps) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_layer_norm_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    size_t dim = static_cast<size_t>(t->shape.back());
    size_t batch_size = static_cast<size_t>(t->size() / dim);

    auto out = kinet_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    KinetBackendBuffer* gamma_buf = gamma ? gamma->device_buffer : nullptr;
    KinetBackendBuffer* beta_buf = beta ? beta->device_buffer : nullptr;

    if (t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_layer_norm_f32(
            gpu->ctx, t->device_buffer, out->device_buffer,
            gamma_buf, beta_buf, batch_size, dim, eps
        );
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

KinetTensor* kinet_tensor_rms_norm(KinetGPU* gpu, KinetTensor* t, KinetTensor* weight, float eps) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_rms_norm_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    size_t dim = static_cast<size_t>(t->shape.back());
    size_t batch_size = static_cast<size_t>(t->size() / dim);

    auto out = kinet_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    KinetBackendBuffer* weight_buf = weight ? weight->device_buffer : nullptr;

    if (t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_rms_norm_f32(
            gpu->ctx, t->device_buffer, out->device_buffer,
            weight_buf, batch_size, dim, eps
        );
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Transpose and copy
// =============================================================================

KinetTensor* kinet_tensor_transpose(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_transpose_f32) return nullptr;
    if (t->shape.size() != 2) return nullptr;

    int rows = static_cast<int>(t->shape[0]);
    int cols = static_cast<int>(t->shape[1]);

    int64_t out_shape[2] = {cols, rows};
    auto out = kinet_tensor_zeros(gpu, out_shape, 2, t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_transpose_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

KinetTensor* kinet_tensor_copy(KinetGPU* gpu, KinetTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_copy_f32) return nullptr;

    auto out = kinet_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        KinetBackendError err = gpu->vtbl->op_copy_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
        if (err != KINET_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Stream/Event Management
// =============================================================================
//
// Streams provide ordered execution queues. For CPU backend, operations are
// synchronous so streams are lightweight handles that track parent context.
// Events mark points in stream execution for synchronization and timing.

struct KinetStream {
    KinetGPU* gpu;
    bool valid;

    explicit KinetStream(KinetGPU* g) : gpu(g), valid(true) {}
};

struct KinetEvent {
    KinetGPU* gpu;
    bool recorded;
    std::chrono::steady_clock::time_point timestamp;

    explicit KinetEvent(KinetGPU* g) : gpu(g), recorded(false), timestamp() {}
};

KinetStream* kinet_stream_create(KinetGPU* gpu) {
    if (!gpu) return nullptr;
    return new KinetStream(gpu);
}

void kinet_stream_destroy(KinetStream* stream) {
    delete stream;
}

KinetError kinet_stream_sync(KinetStream* stream) {
    if (!stream || !stream->valid) return KINET_ERROR_INVALID_ARGUMENT;
    // For CPU backend, all operations are synchronous - nothing to wait for
    // For GPU backends, this would dispatch to backend sync
    if (stream->gpu && stream->gpu->vtbl && stream->gpu->vtbl->sync) {
        return static_cast<KinetError>(stream->gpu->vtbl->sync(stream->gpu->ctx));
    }
    return KINET_OK;
}

KinetEvent* kinet_event_create(KinetGPU* gpu) {
    if (!gpu) return nullptr;
    return new KinetEvent(gpu);
}

void kinet_event_destroy(KinetEvent* event) {
    delete event;
}

KinetError kinet_event_record(KinetEvent* event, KinetStream* stream) {
    if (!event) return KINET_ERROR_INVALID_ARGUMENT;
    // Stream can be null (use default stream)
    // For CPU: record current time
    // For GPU: would insert marker into command queue
    event->timestamp = std::chrono::steady_clock::now();
    event->recorded = true;
    return KINET_OK;
}

KinetError kinet_event_wait(KinetEvent* event, KinetStream* stream) {
    if (!event || !event->recorded) return KINET_ERROR_INVALID_ARGUMENT;
    // Stream can be null (use default stream)
    // For CPU: event is already complete (synchronous execution)
    // For GPU: would wait until event is signaled
    return KINET_OK;
}

float kinet_event_elapsed(KinetEvent* start, KinetEvent* end) {
    if (!start || !end || !start->recorded || !end->recorded) return 0.0f;

    auto duration = end->timestamp - start->timestamp;
    // Return elapsed time in milliseconds
    return std::chrono::duration<float, std::milli>(duration).count();
}

// NTT Operations
KinetError kinet_ntt_forward(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !data) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_forward) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_ntt_forward(gpu->ctx, data, n, modulus));
}

KinetError kinet_ntt_inverse(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !data) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_inverse) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_ntt_inverse(gpu->ctx, data, n, modulus));
}

KinetError kinet_ntt_batch(KinetGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !polys) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_forward) return KINET_ERROR_NOT_SUPPORTED;

    // Process each polynomial in sequence
    for (size_t i = 0; i < count; i++) {
        KinetBackendError err = gpu->vtbl->op_ntt_forward(gpu->ctx, polys[i], n, modulus);
        if (err != KINET_BACKEND_OK) return static_cast<KinetError>(err);
    }
    return KINET_OK;
}

// =============================================================================
// Polynomial Arithmetic
// =============================================================================

KinetError kinet_poly_mul(KinetGPU* gpu, const uint64_t* a, const uint64_t* b, uint64_t* result, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !a || !b || !result) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_poly_mul) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_poly_mul(gpu->ctx, a, b, result, n, modulus));
}

// =============================================================================
// TFHE Operations
// =============================================================================

KinetError kinet_tfhe_bootstrap(KinetGPU* gpu,
                            const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* bsk, const uint64_t* test_poly,
                            uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!lwe_in || !lwe_out || !bsk || !test_poly) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_tfhe_bootstrap) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_tfhe_bootstrap(gpu->ctx, lwe_in, lwe_out, bsk, test_poly, n_lwe, N, k, l, q));
}

KinetError kinet_tfhe_keyswitch(KinetGPU* gpu,
                            const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* ksk,
                            uint32_t n_in, uint32_t n_out, uint32_t l, uint32_t base_log, uint64_t q) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!lwe_in || !lwe_out || !ksk) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_tfhe_keyswitch) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_tfhe_keyswitch(gpu->ctx, lwe_in, lwe_out, ksk, n_in, n_out, l, base_log, q));
}

KinetError kinet_blind_rotate(KinetGPU* gpu,
                          uint64_t* acc, const uint64_t* bsk, const uint64_t* lwe_a,
                          uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!acc || !bsk || !lwe_a) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_blind_rotate) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_blind_rotate(gpu->ctx, acc, bsk, lwe_a, n_lwe, N, k, l, q));
}

// =============================================================================
// Crypto: Hash Functions
// =============================================================================

KinetError kinet_poseidon2_hash(KinetGPU* gpu, const uint64_t* inputs, uint64_t* outputs, size_t rate, size_t num_hashes) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!inputs || !outputs) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_poseidon2_hash) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_poseidon2_hash(gpu->ctx, inputs, outputs, rate, num_hashes));
}

KinetError kinet_blake3_hash(KinetGPU* gpu, const uint8_t* inputs, uint8_t* outputs, const size_t* input_lens, size_t num_hashes) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!inputs || !outputs || !input_lens) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_blake3_hash) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_blake3_hash(gpu->ctx, inputs, outputs, input_lens, num_hashes));
}

KinetError kinet_gpu_keccak256_batch(KinetGPU* gpu, const uint8_t* inputs, uint8_t* outputs, const size_t* input_lens, size_t num_inputs) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!inputs || !outputs || !input_lens) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_keccak256_hash) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_keccak256_hash(gpu->ctx, inputs, outputs, input_lens, num_inputs));
}

// =============================================================================
// Crypto: secp256k1 ECDSA Recovery (Ethereum ecrecover)
// =============================================================================

KinetError kinet_gpu_ecrecover_batch(KinetGPU* gpu, const KinetEcrecoverInput* signatures, KinetEcrecoverOutput* addresses, size_t num_signatures) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!signatures || !addresses) return KINET_ERROR_INVALID_ARGUMENT;
    if (num_signatures == 0) return KINET_OK;
    if (!gpu->vtbl->op_ecrecover_batch) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_ecrecover_batch(gpu->ctx, signatures, addresses, num_signatures));
}

// =============================================================================
// Crypto: MSM
// =============================================================================

KinetError kinet_msm(KinetGPU* gpu, const void* scalars, const void* points, void* result, size_t count, KinetCurve curve) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!scalars || !points || !result) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_msm) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_msm(gpu->ctx, scalars, points, result, count, static_cast<int>(curve)));
}

// =============================================================================
// Crypto: BLS12-381 Curve
// =============================================================================

KinetError kinet_bls12_381_add(KinetGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!a || !b || !out) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_add) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_bls12_381_add(gpu->ctx, a, b, out, count, is_g2));
}

KinetError kinet_bls12_381_mul(KinetGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!points || !scalars || !out) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_mul) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_bls12_381_mul(gpu->ctx, points, scalars, out, count, is_g2));
}

KinetError kinet_bls12_381_pairing(KinetGPU* gpu, const void* g1_points, const void* g2_points, void* out, size_t count) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!g1_points || !g2_points || !out) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_pairing) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_bls12_381_pairing(gpu->ctx, g1_points, g2_points, out, count));
}

// =============================================================================
// Crypto: BN254 Curve
// =============================================================================

KinetError kinet_bn254_add(KinetGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!a || !b || !out) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bn254_add) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_bn254_add(gpu->ctx, a, b, out, count, is_g2));
}

KinetError kinet_bn254_mul(KinetGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!points || !scalars || !out) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bn254_mul) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_bn254_mul(gpu->ctx, points, scalars, out, count, is_g2));
}

// =============================================================================
// Crypto: KZG Polynomial Commitments
// =============================================================================

KinetError kinet_kzg_commit(KinetGPU* gpu, const void* coeffs, const void* srs, void* commitment, size_t degree, KinetCurve curve) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!coeffs || !srs || !commitment) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_commit) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_kzg_commit(gpu->ctx, coeffs, srs, commitment, degree, static_cast<int>(curve)));
}

KinetError kinet_kzg_open(KinetGPU* gpu, const void* coeffs, const void* srs, const void* point, void* proof, size_t degree, KinetCurve curve) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!coeffs || !srs || !point || !proof) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_open) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_kzg_open(gpu->ctx, coeffs, srs, point, proof, degree, static_cast<int>(curve)));
}

KinetError kinet_kzg_verify(KinetGPU* gpu, const void* commitment, const void* proof, const void* point, const void* value, const void* srs_g2, bool* result, KinetCurve curve) {
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!commitment || !proof || !point || !value || !srs_g2 || !result) return KINET_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_verify) return KINET_ERROR_NOT_SUPPORTED;
    return static_cast<KinetError>(gpu->vtbl->op_kzg_verify(gpu->ctx, commitment, proof, point, value, srs_g2, result, static_cast<int>(curve)));
}

// =============================================================================
// BLS Signature Operations
// =============================================================================
//
// BLS signatures use BLS12-381 curve with:
// - G1: 48-byte compressed points (public keys)
// - G2: 96-byte compressed points (signatures)
// - Verification: e(pubkey, H(msg)) == e(G1_generator, sig)
//
// These functions require the backend to support BLS12-381 pairing operations.
// Without backend support, they return KINET_ERROR_NOT_SUPPORTED.

// BLS12-381 constants
static constexpr size_t BLS_G1_COMPRESSED_SIZE = 48;
static constexpr size_t BLS_G2_COMPRESSED_SIZE = 96;
static constexpr size_t BLS_G1_UNCOMPRESSED_SIZE = 96;
static constexpr size_t BLS_G2_UNCOMPRESSED_SIZE = 192;

KinetError kinet_bls_verify(KinetGPU* gpu,
                        const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len,
                        bool* result) {
    // Validate inputs
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!sig || !msg || !pubkey || !result) return KINET_ERROR_INVALID_ARGUMENT;

    // Check backend supports required operations
    if (!gpu->vtbl->op_bls12_381_pairing) {
        gpu->set_error("BLS verify requires BLS12-381 pairing support");
        return KINET_ERROR_NOT_SUPPORTED;
    }

    // Validate point sizes
    // Accept both compressed (48/96) and uncompressed (96/192) formats
    bool pubkey_compressed = (pubkey_len == BLS_G1_COMPRESSED_SIZE);
    bool sig_compressed = (sig_len == BLS_G2_COMPRESSED_SIZE);

    if (!pubkey_compressed && pubkey_len != BLS_G1_UNCOMPRESSED_SIZE) {
        gpu->set_error("Invalid public key size: expected 48 or 96 bytes");
        return KINET_ERROR_INVALID_ARGUMENT;
    }
    if (!sig_compressed && sig_len != BLS_G2_UNCOMPRESSED_SIZE) {
        gpu->set_error("Invalid signature size: expected 96 or 192 bytes");
        return KINET_ERROR_INVALID_ARGUMENT;
    }

    // BLS verification requires hash-to-curve (H(msg) -> G2) and pairing check
    // Full implementation requires:
    // 1. Decompress pubkey to G1 point
    // 2. Hash message to G2 point (hash_to_curve per RFC 9380)
    // 3. Decompress sig to G2 point
    // 4. Verify: e(pubkey, H(msg)) == e(G1_generator, sig)
    //
    // The backend's pairing operation handles the pairing math.
    // Hash-to-curve requires field arithmetic not yet in the vtable.
    //
    // For now, we return NOT_SUPPORTED with clear error message.
    // A full implementation would integrate with a crypto library like blst.

    gpu->set_error("BLS verify requires hash-to-curve; integrate blst or similar");
    return KINET_ERROR_NOT_SUPPORTED;
}

KinetError kinet_bls_verify_batch(KinetGPU* gpu,
                              const uint8_t* const* sigs, const size_t* sig_lens,
                              const uint8_t* const* msgs, const size_t* msg_lens,
                              const uint8_t* const* pubkeys, const size_t* pubkey_lens,
                              int count, bool* results) {
    // Validate inputs
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!sigs || !sig_lens || !msgs || !msg_lens) return KINET_ERROR_INVALID_ARGUMENT;
    if (!pubkeys || !pubkey_lens || !results) return KINET_ERROR_INVALID_ARGUMENT;
    if (count <= 0) return KINET_ERROR_INVALID_ARGUMENT;

    // Check backend supports required operations
    if (!gpu->vtbl->op_bls12_381_pairing) {
        gpu->set_error("BLS batch verify requires BLS12-381 pairing support");
        return KINET_ERROR_NOT_SUPPORTED;
    }

    // Batch verification uses randomized linear combination for efficiency:
    // Instead of n independent pairing checks, verify:
    // e(sum(r_i * pubkey_i), H(msg_i)) == e(G1, sum(r_i * sig_i))
    // where r_i are random scalars.
    //
    // This requires the same primitives as single verify plus:
    // - Scalar multiplication on G1 and G2
    // - Point addition on G1 and G2
    //
    // For now, fall back to sequential verification.
    for (int i = 0; i < count; i++) {
        KinetError err = kinet_bls_verify(gpu, sigs[i], sig_lens[i],
                                       msgs[i], msg_lens[i],
                                       pubkeys[i], pubkey_lens[i],
                                       &results[i]);
        if (err != KINET_OK && err != KINET_ERROR_NOT_SUPPORTED) {
            return err;
        }
        // If single verify is not supported, all results are indeterminate
        if (err == KINET_ERROR_NOT_SUPPORTED) {
            return err;
        }
    }
    return KINET_OK;
}

KinetError kinet_bls_aggregate(KinetGPU* gpu,
                           const uint8_t* const* sigs, const size_t* sig_lens,
                           int count, uint8_t* out, size_t* out_len) {
    // Validate inputs
    if (!gpu || !gpu->vtbl) return KINET_ERROR_INVALID_ARGUMENT;
    if (!sigs || !sig_lens || !out || !out_len) return KINET_ERROR_INVALID_ARGUMENT;
    if (count <= 0) return KINET_ERROR_INVALID_ARGUMENT;

    // Check backend supports required operations
    if (!gpu->vtbl->op_bls12_381_add) {
        gpu->set_error("BLS aggregate requires BLS12-381 point addition");
        return KINET_ERROR_NOT_SUPPORTED;
    }

    // Validate all signatures have consistent size
    size_t first_len = sig_lens[0];
    bool compressed = (first_len == BLS_G2_COMPRESSED_SIZE);
    if (!compressed && first_len != BLS_G2_UNCOMPRESSED_SIZE) {
        gpu->set_error("Invalid signature size");
        return KINET_ERROR_INVALID_ARGUMENT;
    }

    for (int i = 1; i < count; i++) {
        if (sig_lens[i] != first_len) {
            gpu->set_error("All signatures must have same size for aggregation");
            return KINET_ERROR_INVALID_ARGUMENT;
        }
    }

    // Aggregation: agg_sig = sig_1 + sig_2 + ... + sig_n (G2 point addition)
    // For compressed points, we need to decompress, add, recompress.
    // The backend add operation works on uncompressed points.

    if (compressed) {
        // Compressed format requires decompression - not yet implemented
        gpu->set_error("Compressed signature aggregation requires point decompression");
        return KINET_ERROR_NOT_SUPPORTED;
    }

    // Uncompressed format: direct G2 addition
    // Allocate working buffer for accumulator
    std::vector<uint8_t> acc(BLS_G2_UNCOMPRESSED_SIZE);
    std::memcpy(acc.data(), sigs[0], BLS_G2_UNCOMPRESSED_SIZE);

    for (int i = 1; i < count; i++) {
        std::vector<uint8_t> result(BLS_G2_UNCOMPRESSED_SIZE);
        KinetBackendError err = gpu->vtbl->op_bls12_381_add(
            gpu->ctx,
            acc.data(),      // a
            sigs[i],         // b
            result.data(),   // out
            1,               // count
            true             // is_g2
        );
        if (err != KINET_BACKEND_OK) {
            gpu->set_error("G2 point addition failed during aggregation");
            return static_cast<KinetError>(err);
        }
        acc = std::move(result);
    }

    // Copy result
    if (*out_len < BLS_G2_UNCOMPRESSED_SIZE) {
        gpu->set_error("Output buffer too small");
        return KINET_ERROR_INVALID_ARGUMENT;
    }
    std::memcpy(out, acc.data(), BLS_G2_UNCOMPRESSED_SIZE);
    *out_len = BLS_G2_UNCOMPRESSED_SIZE;

    return KINET_OK;
}

} // extern "C"

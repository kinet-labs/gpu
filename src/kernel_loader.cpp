// Kernel Loader - Implementation of kernel caching and registry

#include "kinet/gpu/kernel_loader.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <cstring>

namespace kinet::gpu {

// =============================================================================
// Kernel Variant Hash
// =============================================================================

struct VariantHash {
    size_t operator()(const KinetKernelVariant& v) const {
        // FNV-1a hash
        size_t h = 14695981039346656037ULL;
        if (v.name) {
            for (const char* p = v.name; *p; ++p) {
                h ^= static_cast<size_t>(*p);
                h *= 1099511628211ULL;
            }
        }
        h ^= static_cast<size_t>(v.dtype);
        h *= 1099511628211ULL;
        h ^= static_cast<size_t>(v.size_hint);
        h *= 1099511628211ULL;
        h ^= static_cast<size_t>(v.flags);
        h *= 1099511628211ULL;
        return h;
    }
};

struct VariantEqual {
    bool operator()(const KinetKernelVariant& a, const KinetKernelVariant& b) const {
        if (a.dtype != b.dtype) return false;
        if (a.size_hint != b.size_hint) return false;
        if (a.flags != b.flags) return false;
        if (a.name == nullptr && b.name == nullptr) return true;
        if (a.name == nullptr || b.name == nullptr) return false;
        return std::strcmp(a.name, b.name) == 0;
    }
};

// =============================================================================
// Kernel Cache Implementation
// =============================================================================

} // namespace kinet::gpu

// =============================================================================
// KinetKernelCache Implementation (C-linkage compatible)
// =============================================================================

struct KinetKernelCache {
    std::mutex mutex;
    std::unordered_map<KinetKernelVariant, KinetKernel*, kinet::gpu::VariantHash, kinet::gpu::VariantEqual> cache;
    size_t total_memory = 0;

    // Store name strings to ensure lifetime
    std::unordered_map<std::string, std::string> name_storage;
};

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

KinetKernelCache* kinet_kernel_cache_create(void) {
    return new KinetKernelCache();
}

void kinet_kernel_cache_destroy(KinetKernelCache* cache) {
    if (!cache) return;

    // Note: We don't destroy kernels here - they may be in use.
    // Caller is responsible for clearing cache before destruction.
    delete cache;
}

KinetKernel* kinet_kernel_cache_get(
    KinetKernelCache* cache,
    const KinetKernelVariant* variant
) {
    if (!cache || !variant) return nullptr;

    std::lock_guard<std::mutex> lock(cache->mutex);
    auto it = cache->cache.find(*variant);
    return (it != cache->cache.end()) ? it->second : nullptr;
}

void kinet_kernel_cache_put(
    KinetKernelCache* cache,
    const KinetKernelVariant* variant,
    KinetKernel* kernel
) {
    if (!cache || !variant || !kernel) return;

    std::lock_guard<std::mutex> lock(cache->mutex);

    // Store name string to ensure lifetime
    std::string name_key;
    if (variant->name) {
        name_key = variant->name;
        cache->name_storage[name_key] = name_key;
    }

    // Create stable variant with stored name
    KinetKernelVariant stored = *variant;
    if (variant->name) {
        stored.name = cache->name_storage[name_key].c_str();
    }

    cache->cache[stored] = kernel;
}

void kinet_kernel_cache_clear(KinetKernelCache* cache) {
    if (!cache) return;

    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->cache.clear();
    cache->name_storage.clear();
    cache->total_memory = 0;
}

void kinet_kernel_cache_stats(
    KinetKernelCache* cache,
    size_t* count,
    size_t* memory_bytes
) {
    if (!cache) {
        if (count) *count = 0;
        if (memory_bytes) *memory_bytes = 0;
        return;
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    if (count) *count = cache->cache.size();
    if (memory_bytes) *memory_bytes = cache->total_memory;
}

// =============================================================================
// Kernel Registry
// =============================================================================

// Forward declarations of backend registries (defined in generated files)
extern const KinetKernelRegistry* kinet_kernel_registry_metal_get(void);
extern const KinetKernelRegistry* kinet_kernel_registry_cuda_get(void);
extern const KinetKernelRegistry* kinet_kernel_registry_webgpu_get(void);

const KinetKernelRegistry* kinet_kernel_registry_get(const char* backend) {
    if (!backend) return nullptr;

    if (std::strcmp(backend, "metal") == 0) {
        return kinet_kernel_registry_metal_get();
    } else if (std::strcmp(backend, "cuda") == 0) {
        return kinet_kernel_registry_cuda_get();
    } else if (std::strcmp(backend, "webgpu") == 0) {
        return kinet_kernel_registry_webgpu_get();
    }

    return nullptr;
}

const KinetEmbeddedKernel* kinet_kernel_registry_find(
    const KinetKernelRegistry* registry,
    const char* name
) {
    if (!registry || !name) return nullptr;

    for (size_t i = 0; i < registry->count; ++i) {
        if (std::strcmp(registry->kernels[i].name, name) == 0) {
            return &registry->kernels[i];
        }
    }

    return nullptr;
}

// =============================================================================
// Kernel Compilation API (backend-specific)
// These are placeholder stubs. Real implementations live in each backend plugin.
// =============================================================================

KinetKernel* kinet_kernel_compile(
    void* device_context,
    const KinetKernelSource* source
) {
    // Backend-specific implementation required
    // This stub returns nullptr as the core library cannot compile kernels
    // without knowing the target backend. Use backend-specific compilation
    // or load pre-compiled binaries.
    (void)device_context;
    (void)source;
    return nullptr;
}

KinetKernel* kinet_kernel_load_binary(
    void* device_context,
    const void* binary,
    size_t binary_len,
    const char* entry_point
) {
    // Backend-specific implementation required
    (void)device_context;
    (void)binary;
    (void)binary_len;
    (void)entry_point;
    return nullptr;
}

void kinet_kernel_destroy(KinetKernel* kernel) {
    // Kernels are backend-owned; this stub does nothing.
    // Each backend tracks and destroys its own kernels.
    (void)kernel;
}

const char* kinet_kernel_entry_point(KinetKernel* kernel) {
    // Backend-specific; stub returns nullptr
    (void)kernel;
    return nullptr;
}

} // extern "C"

// =============================================================================
// Weak symbols for backend registries (overridden by backends)
// =============================================================================

#if !defined(_WIN32)
__attribute__((weak))
#endif
const KinetKernelRegistry* kinet_kernel_registry_metal_get(void) {
    return nullptr;
}

#if !defined(_WIN32)
__attribute__((weak))
#endif
const KinetKernelRegistry* kinet_kernel_registry_cuda_get(void) {
    return nullptr;
}

#if !defined(_WIN32)
__attribute__((weak))
#endif
const KinetKernelRegistry* kinet_kernel_registry_webgpu_get(void) {
    return nullptr;
}

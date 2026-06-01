// Internal header for KinetGPU struct access
// Used by zk_ops.cpp and other internal files that need backend vtable access

#ifndef KINET_GPU_INTERNAL_H
#define KINET_GPU_INTERNAL_H

#include "kinet/gpu.h"
#include "kinet/gpu/backend_plugin.h"
#include <string>
#include <mutex>

// KinetGPU struct must match gpu_core.cpp definition exactly
struct KinetGPU {
    std::string backend_name;
    const kinet_gpu_backend_vtbl* vtbl = nullptr;
    KinetBackendContext* ctx = nullptr;
    std::string last_error;
    std::mutex mutex;

    ~KinetGPU();

    void set_error(const char* msg);
};

#endif // KINET_GPU_INTERNAL_H

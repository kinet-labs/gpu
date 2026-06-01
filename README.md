# Kinet GPU Core

Lightweight plugin-based GPU acceleration library for blockchain and ML workloads.

## Architecture

This is the **core library only**. It provides:
- **Stable ABI** (`backend_plugin.h`) - Plugin contract
- **Plugin Loader** - Dynamic loading of backend plugins
- **CPU Fallback** - Builtin CPU backend for any platform
- **Tests** - Backend-agnostic test harness

Backend plugins are built and distributed separately:

| Plugin | Repo | Platform | Dependencies |
|--------|------|----------|--------------|
| Metal | `kinetcpp/metal` | macOS arm64 | MLX, Metal.framework |
| CUDA | `kinetcpp/cuda` | Linux, Windows | CUDA Toolkit, CCCL |
| WebGPU | `kinetcpp/webgpu` | All | Dawn/wgpu, gpu.cpp |

## Building

```bash
# Core only (CPU backend)
cmake -B build
cmake --build build

# Run tests
ctest --test-dir build
```

## Usage

```c
#include <kinet/gpu.h>

int main() {
    // Initialize (loads best available backend)
    kinet_gpu_init();

    // Or specify backend explicitly
    // kinet_gpu_set_backend(KINET_BACKEND_CUDA);

    KinetContext* ctx = kinet_gpu_create_context(-1);

    // Allocate and compute...
    KinetBuffer* buf = kinet_gpu_alloc(ctx, 1024 * sizeof(float));

    kinet_gpu_free(ctx, buf);
    kinet_gpu_destroy_context(ctx);
    kinet_gpu_shutdown();
}
```

## Backend Selection

At runtime, backends are selected in priority order:
1. **CUDA** - If NVIDIA GPU detected
2. **Metal** - If macOS arm64
3. **WebGPU** - Cross-platform fallback
4. **CPU** - Final fallback (always available)

Override via environment or API:
```bash
export KINET_BACKEND=cuda  # or metal, webgpu, cpu
```

## Plugin Loading

Backends are loaded from:
1. `KINET_GPU_BACKEND_PATH` environment variable
2. System library paths (`/usr/lib/kinet-gpu`, etc.)
3. Relative to executable

Plugin naming: `libkinetgpu_backend_<name>.{so,dylib,dll}`

## ABI Stability

The plugin ABI is versioned. Plugins must match the core ABI version:
```c
// backend_plugin.h
#define KINET_GPU_ABI_VERSION 1
```

## License

BSD-3-Clause-Eco

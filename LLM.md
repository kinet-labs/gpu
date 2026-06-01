# Kinet GPU - C++ Backend Library

Plugin-based GPU acceleration for blockchain and ML workloads.

## Structure

```
kinetcpp/gpu/
├── include/kinet/gpu.h      # C API (stable)
├── src/
│   ├── gpu_core.cpp       # Core dispatch logic
│   ├── cpu_backend.cpp    # CPU SIMD backend
│   ├── bn254_field.hpp    # BN254 field arithmetic
│   └── zk_ops.cpp         # ZK primitives
├── webgpu/                # Dawn WebGPU backend
│   ├── gpu.hpp            # gpu.cpp header (Dawn wrapper)
│   └── kernels/           # WGSL kernel sources
├── kernels/cpu/           # CPU kernel implementations
├── benchmarks/            # Performance tests
└── test/                  # Unit tests
```

## Build Requirements

- Cuda compilation tools, release 13.0, V13.0.88

- libblas-dev, liblapack-dev, and liblapacke-dev (Linux)

```bash
sudo apt update && sudo apt install -y libblas-dev liblapack-dev liblapacke-dev
```

- A C++ compiler with C++20 support (e.g. Clang >= 15.0)

```bash
sudo apt update && sudo apt install -y clang libstdc++-dev
```

- cmake – version 3.25 or later, and make

```bash
sudo apt-get update && sudo apt-get install -y cmake ninja-build
```

- Xcode >= 15.0 and macOS SDK >= 14.0

## Backends

| Backend | Location | Notes |
|---------|----------|-------|
| CPU | `src/cpu_backend.cpp` | Always available, SIMD optimized |
| Metal | `kinetcpp/metal` (separate) | Apple Silicon, uses MLX |
| CUDA | `kinetcpp/cuda` (separate) | NVIDIA, uses CCCL |
| WebGPU | `webgpu/` | Dawn-based, WGSL kernels |

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Key C API Functions

### Context
```c
KinetGPU* kinet_gpu_create(void);
void kinet_gpu_destroy(KinetGPU* gpu);
KinetError kinet_gpu_sync(KinetGPU* gpu);
KinetError kinet_gpu_set_backend(KinetGPU* gpu, KinetBackend backend);
```

### Tensors
```c
KinetTensor* kinet_tensor_zeros(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype);
KinetTensor* kinet_tensor_add(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
KinetTensor* kinet_tensor_matmul(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
void kinet_tensor_destroy(KinetTensor* tensor);
```

### Crypto Operations
```c
// BLS12-381
KinetError kinet_bls12_381_add(KinetGPU* gpu, ...);
KinetError kinet_bls12_381_mul(KinetGPU* gpu, ...);
KinetError kinet_bls12_381_pairing(KinetGPU* gpu, ...);
KinetError kinet_bls_verify(KinetGPU* gpu, ...);

// BN254
KinetError kinet_bn254_add(KinetGPU* gpu, ...);
KinetError kinet_bn254_mul(KinetGPU* gpu, ...);

// KZG Commitments
KinetError kinet_kzg_commit(KinetGPU* gpu, ...);
KinetError kinet_kzg_open(KinetGPU* gpu, ...);
KinetError kinet_kzg_verify(KinetGPU* gpu, ...);

// Poseidon2 Hash
KinetError kinet_poseidon2_hash(KinetGPU* gpu, ...);
KinetError kinet_gpu_poseidon2(KinetGPU* gpu, ...);
KinetError kinet_gpu_merkle_root(KinetGPU* gpu, ...);
```

### FHE Operations
```c
KinetError kinet_ntt_forward(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
KinetError kinet_ntt_inverse(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
KinetError kinet_tfhe_bootstrap(KinetGPU* gpu, ...);
KinetError kinet_tfhe_keyswitch(KinetGPU* gpu, ...);
KinetError kinet_blind_rotate(KinetGPU* gpu, ...);
```

## Environment Variables

- `KINET_GPU_BACKEND` - Force backend: `metal`, `cuda`, `webgpu`, `cpu`
- `KINET_GPU_BACKEND_PATH` - Custom plugin search path

## Go Bindings

See `kinet/gpu` for Go bindings that wrap this C API via CGO.

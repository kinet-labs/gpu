# API Reference {#api}

Complete reference for the Kinet GPU Core C API.

## Header Files

| Header | Description |
|--------|-------------|
| `<kinet/gpu.h>` | Main API - GPU context, tensors, operations |
| `<kinet/gpu/backend_plugin.h>` | Backend plugin interface (for implementers) |
| `<kinet/gpu/kernel_loader.h>` | Kernel loading and caching |

---

## GPU Context

### kinet_gpu_create

```c
KinetGPU* kinet_gpu_create(void);
```

Create a GPU context with automatic backend selection.

Backend priority: CUDA > Metal > WebGPU > CPU

**Returns**: GPU context handle, or NULL on failure.

### kinet_gpu_create_with_backend

```c
KinetGPU* kinet_gpu_create_with_backend(KinetBackend backend);
```

Create a GPU context with a specific backend.

**Parameters**:
- `backend`: Backend type (KINET_BACKEND_CUDA, KINET_BACKEND_METAL, etc.)

**Returns**: GPU context handle, or NULL if backend unavailable.

### kinet_gpu_create_with_device

```c
KinetGPU* kinet_gpu_create_with_device(KinetBackend backend, int device_index);
```

Create a GPU context with a specific backend and device.

**Parameters**:
- `backend`: Backend type
- `device_index`: Device index (0 for first device)

**Returns**: GPU context handle, or NULL on failure.

### kinet_gpu_destroy

```c
void kinet_gpu_destroy(KinetGPU* gpu);
```

Destroy a GPU context and release all resources.

### kinet_gpu_sync

```c
KinetError kinet_gpu_sync(KinetGPU* gpu);
```

Synchronize all pending GPU operations.

**Returns**: KINET_OK on success, error code otherwise.

### kinet_gpu_backend

```c
KinetBackend kinet_gpu_backend(KinetGPU* gpu);
```

Get the current backend type.

### kinet_gpu_backend_name

```c
const char* kinet_gpu_backend_name(KinetGPU* gpu);
```

Get the current backend name as a string.

### kinet_gpu_set_backend

```c
KinetError kinet_gpu_set_backend(KinetGPU* gpu, KinetBackend backend);
```

Switch to a different backend at runtime.

**Note**: This invalidates all existing tensors and buffers.

### kinet_gpu_device_info

```c
KinetError kinet_gpu_device_info(KinetGPU* gpu, KinetDeviceInfo* info);
```

Get information about the current device.

### kinet_gpu_error

```c
const char* kinet_gpu_error(KinetGPU* gpu);
```

Get the last error message.

---

## Backend Query

### kinet_backend_count

```c
int kinet_backend_count(void);
```

Get the number of available backends.

### kinet_backend_available

```c
bool kinet_backend_available(KinetBackend backend);
```

Check if a backend is available on this system.

### kinet_backend_name

```c
const char* kinet_backend_name(KinetBackend backend);
```

Get the name of a backend as a string.

### kinet_device_count

```c
int kinet_device_count(KinetBackend backend);
```

Get the number of devices for a backend.

### kinet_device_info

```c
KinetError kinet_device_info(KinetBackend backend, int index, KinetDeviceInfo* info);
```

Get device info for a specific backend and device index.

---

## Tensor Creation

### kinet_tensor_zeros

```c
KinetTensor* kinet_tensor_zeros(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype);
```

Create a tensor filled with zeros.

**Parameters**:
- `gpu`: GPU context
- `shape`: Array of dimension sizes
- `ndim`: Number of dimensions
- `dtype`: Data type (KINET_FLOAT32, KINET_FLOAT16, etc.)

**Returns**: Tensor handle, or NULL on failure.

### kinet_tensor_ones

```c
KinetTensor* kinet_tensor_ones(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype);
```

Create a tensor filled with ones.

### kinet_tensor_full

```c
KinetTensor* kinet_tensor_full(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype, double value);
```

Create a tensor filled with a specific value.

### kinet_tensor_from_data

```c
KinetTensor* kinet_tensor_from_data(KinetGPU* gpu, const void* data, const int64_t* shape, int ndim, KinetDtype dtype);
```

Create a tensor from host data.

**Parameters**:
- `data`: Pointer to host data (copied to device)
- `shape`: Array of dimension sizes
- `ndim`: Number of dimensions
- `dtype`: Data type

### kinet_tensor_destroy

```c
void kinet_tensor_destroy(KinetTensor* tensor);
```

Destroy a tensor and free device memory.

---

## Tensor Properties

### kinet_tensor_ndim

```c
int kinet_tensor_ndim(KinetTensor* tensor);
```

Get the number of dimensions.

### kinet_tensor_shape

```c
int64_t kinet_tensor_shape(KinetTensor* tensor, int dim);
```

Get the size of a specific dimension.

### kinet_tensor_size

```c
int64_t kinet_tensor_size(KinetTensor* tensor);
```

Get the total number of elements.

### kinet_tensor_dtype

```c
KinetDtype kinet_tensor_dtype(KinetTensor* tensor);
```

Get the data type.

### kinet_tensor_to_host

```c
KinetError kinet_tensor_to_host(KinetTensor* tensor, void* data, size_t size);
```

Copy tensor data to host memory.

**Parameters**:
- `tensor`: Source tensor
- `data`: Destination host buffer
- `size`: Size of destination buffer in bytes

---

## Arithmetic Operations

All arithmetic operations return a new tensor with the result.

### kinet_tensor_add

```c
KinetTensor* kinet_tensor_add(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
```

Element-wise addition: `result = a + b`

### kinet_tensor_sub

```c
KinetTensor* kinet_tensor_sub(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
```

Element-wise subtraction: `result = a - b`

### kinet_tensor_mul

```c
KinetTensor* kinet_tensor_mul(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
```

Element-wise multiplication: `result = a * b`

### kinet_tensor_div

```c
KinetTensor* kinet_tensor_div(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
```

Element-wise division: `result = a / b`

### kinet_tensor_matmul

```c
KinetTensor* kinet_tensor_matmul(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
```

Matrix multiplication: `result = a @ b`

**Parameters**:
- `a`: Left matrix [M, K]
- `b`: Right matrix [K, N]

**Returns**: Result matrix [M, N]

---

## Unary Operations

### kinet_tensor_neg

```c
KinetTensor* kinet_tensor_neg(KinetGPU* gpu, KinetTensor* t);
```

Element-wise negation: `result = -t`

### kinet_tensor_exp

```c
KinetTensor* kinet_tensor_exp(KinetGPU* gpu, KinetTensor* t);
```

Element-wise exponential: `result = exp(t)`

### kinet_tensor_log

```c
KinetTensor* kinet_tensor_log(KinetGPU* gpu, KinetTensor* t);
```

Element-wise natural logarithm: `result = log(t)`

### kinet_tensor_sqrt

```c
KinetTensor* kinet_tensor_sqrt(KinetGPU* gpu, KinetTensor* t);
```

Element-wise square root: `result = sqrt(t)`

### kinet_tensor_abs

```c
KinetTensor* kinet_tensor_abs(KinetGPU* gpu, KinetTensor* t);
```

Element-wise absolute value: `result = |t|`

### kinet_tensor_tanh

```c
KinetTensor* kinet_tensor_tanh(KinetGPU* gpu, KinetTensor* t);
```

Element-wise hyperbolic tangent.

### kinet_tensor_sigmoid

```c
KinetTensor* kinet_tensor_sigmoid(KinetGPU* gpu, KinetTensor* t);
```

Element-wise sigmoid: `result = 1 / (1 + exp(-t))`

### kinet_tensor_relu

```c
KinetTensor* kinet_tensor_relu(KinetGPU* gpu, KinetTensor* t);
```

Element-wise ReLU: `result = max(0, t)`

### kinet_tensor_gelu

```c
KinetTensor* kinet_tensor_gelu(KinetGPU* gpu, KinetTensor* t);
```

Element-wise GELU activation.

---

## Reduction Operations

### Full Reductions (to scalar)

```c
float kinet_tensor_reduce_sum(KinetGPU* gpu, KinetTensor* t);
float kinet_tensor_reduce_max(KinetGPU* gpu, KinetTensor* t);
float kinet_tensor_reduce_min(KinetGPU* gpu, KinetTensor* t);
float kinet_tensor_reduce_mean(KinetGPU* gpu, KinetTensor* t);
```

### Axis Reductions

```c
KinetTensor* kinet_tensor_sum(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);
KinetTensor* kinet_tensor_mean(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);
KinetTensor* kinet_tensor_max(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);
KinetTensor* kinet_tensor_min(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);
```

**Parameters**:
- `axes`: Array of axes to reduce
- `naxes`: Number of axes

---

## Normalization Operations

### kinet_tensor_softmax

```c
KinetTensor* kinet_tensor_softmax(KinetGPU* gpu, KinetTensor* t, int axis);
```

Softmax along specified axis.

### kinet_tensor_log_softmax

```c
KinetTensor* kinet_tensor_log_softmax(KinetGPU* gpu, KinetTensor* t, int axis);
```

Log-softmax along specified axis (numerically stable).

### kinet_tensor_layer_norm

```c
KinetTensor* kinet_tensor_layer_norm(KinetGPU* gpu, KinetTensor* t, KinetTensor* gamma, KinetTensor* beta, float eps);
```

Layer normalization with learnable parameters.

### kinet_tensor_rms_norm

```c
KinetTensor* kinet_tensor_rms_norm(KinetGPU* gpu, KinetTensor* t, KinetTensor* weight, float eps);
```

RMS normalization (used in LLaMA, etc.).

---

## Cryptographic Operations

### Hash Functions

#### kinet_poseidon2_hash

```c
KinetError kinet_poseidon2_hash(KinetGPU* gpu,
                            const uint64_t* inputs,
                            uint64_t* outputs,
                            size_t rate,
                            size_t num_hashes);
```

Poseidon2 algebraic hash for ZK circuits.

#### kinet_blake3_hash

```c
KinetError kinet_blake3_hash(KinetGPU* gpu,
                         const uint8_t* inputs,
                         uint8_t* outputs,
                         const size_t* input_lens,
                         size_t num_hashes);
```

BLAKE3 cryptographic hash.

### Multi-Scalar Multiplication

#### kinet_msm

```c
KinetError kinet_msm(KinetGPU* gpu,
                 const void* scalars,
                 const void* points,
                 void* result,
                 size_t count,
                 KinetCurve curve);
```

Multi-scalar multiplication for elliptic curves.

**Supported Curves**:
- `KINET_CURVE_BLS12_381`
- `KINET_CURVE_BN254`
- `KINET_CURVE_SECP256K1`
- `KINET_CURVE_ED25519`

### BLS12-381 Operations

```c
KinetError kinet_bls12_381_add(KinetGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2);
KinetError kinet_bls12_381_mul(KinetGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2);
KinetError kinet_bls12_381_pairing(KinetGPU* gpu, const void* g1_points, const void* g2_points, void* out, size_t count);
```

### BLS Signature Operations

```c
KinetError kinet_bls_verify(KinetGPU* gpu, const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len,
                        bool* result);

KinetError kinet_bls_verify_batch(KinetGPU* gpu, ...);  // Batch verification
KinetError kinet_bls_aggregate(KinetGPU* gpu, ...);     // Signature aggregation
```

### BN254 Operations

```c
KinetError kinet_bn254_add(KinetGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2);
KinetError kinet_bn254_mul(KinetGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2);
```

### KZG Polynomial Commitments

```c
KinetError kinet_kzg_commit(KinetGPU* gpu, const void* coeffs, const void* srs, void* commitment, size_t degree, KinetCurve curve);
KinetError kinet_kzg_open(KinetGPU* gpu, const void* coeffs, const void* srs, const void* point, void* proof, size_t degree, KinetCurve curve);
KinetError kinet_kzg_verify(KinetGPU* gpu, const void* commitment, const void* proof, const void* point, const void* value, const void* srs_g2, bool* result, KinetCurve curve);
```

---

## FHE Operations

### NTT (Number Theoretic Transform)

```c
KinetError kinet_ntt_forward(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
KinetError kinet_ntt_inverse(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
KinetError kinet_ntt_batch(KinetGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus);
```

### Polynomial Multiplication

```c
KinetError kinet_poly_mul(KinetGPU* gpu, const uint64_t* a, const uint64_t* b, uint64_t* result, size_t n, uint64_t modulus);
```

Polynomial multiplication modulo (X^n + 1).

### TFHE Operations

```c
KinetError kinet_tfhe_bootstrap(KinetGPU* gpu, const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* bsk, const uint64_t* test_poly,
                            uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q);

KinetError kinet_tfhe_keyswitch(KinetGPU* gpu, const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* ksk, uint32_t n_in, uint32_t n_out,
                            uint32_t l, uint32_t base_log, uint64_t q);

KinetError kinet_blind_rotate(KinetGPU* gpu, uint64_t* acc, const uint64_t* bsk,
                          const uint64_t* lwe_a, uint32_t n_lwe, uint32_t N,
                          uint32_t k, uint32_t l, uint64_t q);
```

---

## ZK Primitives

### KinetFr256 Type

```c
typedef struct {
    uint64_t limbs[4];
} KinetFr256;
```

BN254 scalar field element (256-bit).

### kinet_gpu_poseidon2

```c
KinetError kinet_gpu_poseidon2(KinetGPU* gpu, KinetFr256* out, const KinetFr256* left, const KinetFr256* right, size_t n);
```

Poseidon2 2-to-1 compression.

### kinet_gpu_merkle_root

```c
KinetError kinet_gpu_merkle_root(KinetGPU* gpu, KinetFr256* out, const KinetFr256* leaves, size_t n);
```

Compute Merkle tree root from leaves.

### kinet_gpu_commitment

```c
KinetError kinet_gpu_commitment(KinetGPU* gpu, KinetFr256* out, const KinetFr256* values,
                            const KinetFr256* blindings, const KinetFr256* salts, size_t n);
```

Pedersen-style commitments.

### kinet_gpu_nullifier

```c
KinetError kinet_gpu_nullifier(KinetGPU* gpu, KinetFr256* out, const KinetFr256* keys,
                           const KinetFr256* commitments, const KinetFr256* indices, size_t n);
```

Derive nullifiers for double-spend prevention.

---

## Stream and Event Management

### Streams

```c
KinetStream* kinet_stream_create(KinetGPU* gpu);
void kinet_stream_destroy(KinetStream* stream);
KinetError kinet_stream_sync(KinetStream* stream);
```

### Events

```c
KinetEvent* kinet_event_create(KinetGPU* gpu);
void kinet_event_destroy(KinetEvent* event);
KinetError kinet_event_record(KinetEvent* event, KinetStream* stream);
KinetError kinet_event_wait(KinetEvent* event, KinetStream* stream);
float kinet_event_elapsed(KinetEvent* start, KinetEvent* end);
```

---

## Enumerations

### KinetBackend

```c
typedef enum {
    KINET_BACKEND_AUTO  = 0,  // Auto-detect best backend
    KINET_BACKEND_CPU   = 1,  // CPU with SIMD
    KINET_BACKEND_METAL = 2,  // Apple Metal
    KINET_BACKEND_CUDA  = 3,  // NVIDIA CUDA
    KINET_BACKEND_DAWN  = 4,  // WebGPU via Dawn
} KinetBackend;
```

### KinetDtype

```c
typedef enum {
    KINET_FLOAT32  = 0,
    KINET_FLOAT16  = 1,
    KINET_BFLOAT16 = 2,
    KINET_INT32    = 3,
    KINET_INT64    = 4,
    KINET_UINT32   = 5,
    KINET_UINT64   = 6,
    KINET_BOOL     = 7,
} KinetDtype;
```

### KinetError

```c
typedef enum {
    KINET_OK                         = 0,
    KINET_ERROR_INVALID_ARGUMENT     = 1,
    KINET_ERROR_OUT_OF_MEMORY        = 2,
    KINET_ERROR_BACKEND_NOT_AVAILABLE = 3,
    KINET_ERROR_DEVICE_NOT_FOUND     = 4,
    KINET_ERROR_KERNEL_FAILED        = 5,
    KINET_ERROR_NOT_SUPPORTED        = 6,
} KinetError;
```

### KinetCurve

```c
typedef enum {
    KINET_CURVE_BLS12_381 = 0,
    KINET_CURVE_BN254     = 1,
    KINET_CURVE_SECP256K1 = 2,
    KINET_CURVE_ED25519   = 3,
} KinetCurve;
```

---

## Structures

### KinetDeviceInfo

```c
typedef struct {
    KinetBackend backend;
    int index;
    const char* name;
    const char* vendor;
    uint64_t memory_total;
    uint64_t memory_available;
    bool is_discrete;
    bool is_unified_memory;
    int compute_units;
    int max_workgroup_size;
} KinetDeviceInfo;
```

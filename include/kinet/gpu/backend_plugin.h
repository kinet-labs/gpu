// Backend Plugin ABI - Stable C interface for runtime-loaded GPU backends
//
// Each backend shared library exports one symbol: kinet_gpu_backend_init
// The core library dlopen()s backends and calls this to get the vtable.

#ifndef KINET_GPU_BACKEND_PLUGIN_H
#define KINET_GPU_BACKEND_PLUGIN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// ABI Version - bump on breaking changes
// =============================================================================

#define KINET_GPU_BACKEND_ABI_VERSION 2

// =============================================================================
// Forward declarations (opaque handles)
// =============================================================================

typedef struct KinetBackendContext KinetBackendContext;
typedef struct KinetBackendBuffer KinetBackendBuffer;
typedef struct KinetBackendKernel KinetBackendKernel;

// =============================================================================
// Data types (must match kinet/gpu.h)
// =============================================================================

typedef enum {
    KINET_DTYPE_FLOAT32 = 0,
    KINET_DTYPE_FLOAT16 = 1,
    KINET_DTYPE_BFLOAT16 = 2,
    KINET_DTYPE_INT32 = 3,
    KINET_DTYPE_INT64 = 4,
    KINET_DTYPE_UINT32 = 5,
    KINET_DTYPE_UINT64 = 6,
    KINET_DTYPE_BOOL = 7,
} KinetBackendDtype;

typedef enum {
    KINET_BACKEND_OK = 0,
    KINET_BACKEND_ERROR_INVALID_ARGUMENT = 1,
    KINET_BACKEND_ERROR_OUT_OF_MEMORY = 2,
    KINET_BACKEND_ERROR_NOT_SUPPORTED = 3,
    KINET_BACKEND_ERROR_DEVICE_LOST = 4,
    KINET_BACKEND_ERROR_INTERNAL = 5,
} KinetBackendError;

typedef struct {
    const char* name;
    const char* vendor;
    uint64_t memory_total;
    uint64_t memory_available;
    int compute_units;
    int max_workgroup_size;
    bool is_discrete;
    bool is_unified_memory;
} KinetBackendDeviceInfo;

// =============================================================================
// Curve type identifiers for elliptic curve operations
// Use KinetCurve from kinet/gpu.h - values must match:
//   KINET_CURVE_BLS12_381 = 0
//   KINET_CURVE_BN254     = 1
//   KINET_CURVE_SECP256K1 = 2
//   KINET_CURVE_ED25519   = 3
// =============================================================================

// =============================================================================
// Backend Virtual Table
// =============================================================================

typedef struct kinet_gpu_backend_vtbl {
    // Lifecycle
    KinetBackendContext* (*create_context)(int device_index);
    void (*destroy_context)(KinetBackendContext* ctx);

    // Device info
    KinetBackendError (*get_device_count)(int* count);
    KinetBackendError (*get_device_info)(KinetBackendContext* ctx, KinetBackendDeviceInfo* info);

    // Synchronization
    KinetBackendError (*sync)(KinetBackendContext* ctx);

    // Buffer management
    KinetBackendBuffer* (*buffer_alloc)(KinetBackendContext* ctx, size_t bytes);
    KinetBackendBuffer* (*buffer_alloc_with_data)(KinetBackendContext* ctx, const void* data, size_t bytes);
    void (*buffer_free)(KinetBackendContext* ctx, KinetBackendBuffer* buf);
    KinetBackendError (*buffer_copy_to_host)(KinetBackendContext* ctx, KinetBackendBuffer* buf, void* dst, size_t bytes);
    KinetBackendError (*buffer_copy_from_host)(KinetBackendContext* ctx, KinetBackendBuffer* buf, const void* src, size_t bytes);
    void* (*buffer_get_host_ptr)(KinetBackendContext* ctx, KinetBackendBuffer* buf);  // For unified memory

    // Kernel management (for custom kernels)
    KinetBackendKernel* (*kernel_load)(KinetBackendContext* ctx, const char* source, const char* entry_point);
    KinetBackendKernel* (*kernel_load_binary)(KinetBackendContext* ctx, const void* binary, size_t size, const char* entry_point);
    void (*kernel_destroy)(KinetBackendContext* ctx, KinetBackendKernel* kernel);

    // Kernel dispatch
    KinetBackendError (*kernel_dispatch)(
        KinetBackendContext* ctx,
        KinetBackendKernel* kernel,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z,
        KinetBackendBuffer** buffers, int num_buffers
    );

    // ==========================================================================
    // Elementwise Operations
    // ==========================================================================

    KinetBackendError (*op_add_f32)(KinetBackendContext* ctx, KinetBackendBuffer* a, KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_sub_f32)(KinetBackendContext* ctx, KinetBackendBuffer* a, KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_mul_f32)(KinetBackendContext* ctx, KinetBackendBuffer* a, KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_div_f32)(KinetBackendContext* ctx, KinetBackendBuffer* a, KinetBackendBuffer* b, KinetBackendBuffer* out, size_t n);

    // ==========================================================================
    // Matrix Operations
    // ==========================================================================

    KinetBackendError (*op_matmul_f32)(KinetBackendContext* ctx, KinetBackendBuffer* a, KinetBackendBuffer* b, KinetBackendBuffer* out, int M, int K, int N);
    KinetBackendError (*op_transpose_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, int rows, int cols);

    // ==========================================================================
    // Reduction Operations
    // ==========================================================================

    // Full array reductions (n elements -> 1 scalar)
    KinetBackendError (*op_reduce_sum_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_reduce_max_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_reduce_min_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_reduce_mean_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);

    // Axis reductions (outer_size x inner_size -> outer_size)
    KinetBackendError (*op_reduce_sum_axis_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t outer_size, size_t inner_size);
    KinetBackendError (*op_reduce_max_axis_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t outer_size, size_t inner_size);

    // ==========================================================================
    // Softmax Operations
    // ==========================================================================

    KinetBackendError (*op_softmax_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t batch_size, size_t dim);
    KinetBackendError (*op_log_softmax_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t batch_size, size_t dim);

    // ==========================================================================
    // Unary Operations
    // ==========================================================================

    KinetBackendError (*op_exp_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_log_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_sqrt_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_neg_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_abs_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_tanh_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_sigmoid_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_relu_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);
    KinetBackendError (*op_gelu_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, size_t n);

    // ==========================================================================
    // Copy Operations
    // ==========================================================================

    KinetBackendError (*op_copy_f32)(KinetBackendContext* ctx, KinetBackendBuffer* src, KinetBackendBuffer* dst, size_t n);

    // ==========================================================================
    // Normalization Operations
    // ==========================================================================

    KinetBackendError (*op_layer_norm_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, KinetBackendBuffer* gamma, KinetBackendBuffer* beta, size_t batch_size, size_t dim, float eps);
    KinetBackendError (*op_rms_norm_f32)(KinetBackendContext* ctx, KinetBackendBuffer* in, KinetBackendBuffer* out, KinetBackendBuffer* weight, size_t batch_size, size_t dim, float eps);

    // ==========================================================================
    // NTT Operations (for FHE/ZK)
    // ==========================================================================
    KinetBackendError (*op_ntt_forward)(KinetBackendContext* ctx, uint64_t* data, size_t n, uint64_t modulus);
    KinetBackendError (*op_ntt_inverse)(KinetBackendContext* ctx, uint64_t* data, size_t n, uint64_t modulus);

    // MSM operations (for ZK)
    KinetBackendError (*op_msm)(KinetBackendContext* ctx, const void* scalars, const void* points, void* result, size_t n, int curve_type);

    // ==========================================================================
    // FHE Operations (Fully Homomorphic Encryption)
    // ==========================================================================

    // Polynomial multiplication via NTT: result = a * b mod (X^n + 1) mod q
    KinetBackendError (*op_poly_mul)(
        KinetBackendContext* ctx,
        const uint64_t* a,
        const uint64_t* b,
        uint64_t* result,
        size_t n,
        uint64_t modulus
    );

    // TFHE programmable bootstrap: evaluates LUT on encrypted input
    KinetBackendError (*op_tfhe_bootstrap)(
        KinetBackendContext* ctx,
        const uint64_t* lwe_in,       // Input LWE [n_lwe + 1]
        uint64_t* lwe_out,            // Output LWE [N + 1]
        const uint64_t* bsk,          // Bootstrapping key
        const uint64_t* test_poly,    // Test polynomial (LUT)
        uint32_t n_lwe,               // Input LWE dimension
        uint32_t N,                   // GLWE polynomial degree
        uint32_t k,                   // GLWE dimension
        uint32_t l,                   // Decomposition levels
        uint64_t q                    // Modulus
    );

    // TFHE key switching: changes LWE key
    KinetBackendError (*op_tfhe_keyswitch)(
        KinetBackendContext* ctx,
        const uint64_t* lwe_in,       // Input LWE [n_in + 1]
        uint64_t* lwe_out,            // Output LWE [n_out + 1]
        const uint64_t* ksk,          // Key switching key
        uint32_t n_in,                // Input dimension
        uint32_t n_out,               // Output dimension
        uint32_t l,                   // Decomposition levels
        uint32_t base_log,            // Base log
        uint64_t q                    // Modulus
    );

    // Blind rotation: rotates polynomial accumulator by encrypted amount
    KinetBackendError (*op_blind_rotate)(
        KinetBackendContext* ctx,
        uint64_t* acc,                // Accumulator GLWE [(k+1) * N]
        const uint64_t* bsk,          // Bootstrapping key
        const uint64_t* lwe_a,        // LWE 'a' coefficients [n_lwe]
        uint32_t n_lwe,               // LWE dimension
        uint32_t N,                   // GLWE polynomial degree
        uint32_t k,                   // GLWE dimension
        uint32_t l,                   // Decomposition levels
        uint64_t q                    // Modulus
    );

    // Sample extraction: extracts LWE from GLWE at position 0
    KinetBackendError (*op_sample_extract)(
        KinetBackendContext* ctx,
        const uint64_t* glwe,         // Input GLWE [(k+1) * N]
        uint64_t* lwe,                // Output LWE [N + 1]
        uint32_t N,                   // Polynomial degree
        uint32_t k,                   // GLWE dimension
        uint64_t q                    // Modulus
    );

    // Sample polynomial in NTT domain with discrete Gaussian noise
    KinetBackendError (*op_sample_ntt)(
        KinetBackendContext* ctx,
        uint64_t* output,             // Output polynomial in NTT domain
        size_t n,                     // Polynomial degree
        uint64_t modulus,             // Prime modulus
        double sigma,                 // Standard deviation for Gaussian
        uint64_t seed                 // RNG seed
    );

    // ==========================================================================
    // Crypto: Hash Operations
    // ==========================================================================

    // Poseidon2 hash (algebraic hash for ZK)
    KinetBackendError (*op_poseidon2_hash)(
        KinetBackendContext* ctx,
        const uint64_t* inputs,       // Input field elements [num_hashes * rate]
        uint64_t* outputs,            // Output hashes [num_hashes]
        size_t rate,                  // Poseidon rate parameter
        size_t num_hashes             // Number of parallel hashes
    );

    // BLAKE3 hash
    KinetBackendError (*op_blake3_hash)(
        KinetBackendContext* ctx,
        const uint8_t* inputs,        // Input data (concatenated)
        uint8_t* outputs,             // Output 32-byte hashes [num_hashes * 32]
        const size_t* input_lens,     // Length of each input
        size_t num_hashes             // Number of parallel hashes
    );

    // Keccak-256 hash (Ethereum variant)
    KinetBackendError (*op_keccak256_hash)(
        KinetBackendContext* ctx,
        const uint8_t* inputs,        // Input data (concatenated)
        uint8_t* outputs,             // Output 32-byte hashes [num_inputs * 32]
        const size_t* input_lens,     // Length of each input
        size_t num_inputs             // Number of parallel hashes
    );

    // ==========================================================================
    // Crypto: BLS12-381 Curve Operations
    // ==========================================================================

    // Point addition (G1 or G2)
    KinetBackendError (*op_bls12_381_add)(
        KinetBackendContext* ctx,
        const void* a,                // Points (G1 or G2)
        const void* b,                // Points (G1 or G2)
        void* out,                    // Sum points
        size_t n,                     // Number of point pairs
        bool is_g2                    // true for G2, false for G1
    );

    // Scalar multiplication (G1 or G2)
    KinetBackendError (*op_bls12_381_mul)(
        KinetBackendContext* ctx,
        const void* points,           // Points (G1 or G2)
        const void* scalars,          // Scalar field elements
        void* out,                    // Product points
        size_t n,                     // Number of operations
        bool is_g2                    // true for G2, false for G1
    );

    // Pairing computation (multi-pairing supported)
    KinetBackendError (*op_bls12_381_pairing)(
        KinetBackendContext* ctx,
        const void* g1_points,        // G1 points
        const void* g2_points,        // G2 points
        void* out,                    // Pairing result (Gt element)
        size_t n                      // Number of pairings (multi-pairing)
    );

    // ==========================================================================
    // Crypto: BN254 Curve Operations
    // ==========================================================================

    // Point addition (G1 or G2)
    KinetBackendError (*op_bn254_add)(
        KinetBackendContext* ctx,
        const void* a,                // Points
        const void* b,                // Points
        void* out,                    // Sum points
        size_t n,                     // Number of point pairs
        bool is_g2                    // true for G2, false for G1
    );

    // Scalar multiplication (G1 or G2)
    KinetBackendError (*op_bn254_mul)(
        KinetBackendContext* ctx,
        const void* points,           // Points
        const void* scalars,          // Scalar field elements
        void* out,                    // Product points
        size_t n,                     // Number of operations
        bool is_g2                    // true for G2, false for G1
    );

    // ==========================================================================
    // Crypto: KZG Polynomial Commitments
    // ==========================================================================

    // Commit to polynomial using SRS
    KinetBackendError (*op_kzg_commit)(
        KinetBackendContext* ctx,
        const void* coeffs,           // Polynomial coefficients (field elements)
        const void* srs,              // SRS G1 points (powers of tau)
        void* commitment,             // Output commitment point
        size_t degree,                // Polynomial degree
        int curve_type                // KinetCurveType
    );

    // Open commitment at evaluation point
    KinetBackendError (*op_kzg_open)(
        KinetBackendContext* ctx,
        const void* coeffs,           // Polynomial coefficients
        const void* srs,              // SRS G1 points
        const void* point,            // Evaluation point (field element)
        void* proof,                  // Output proof (point)
        size_t degree,                // Polynomial degree
        int curve_type                // KinetCurveType
    );

    // Verify KZG opening proof
    KinetBackendError (*op_kzg_verify)(
        KinetBackendContext* ctx,
        const void* commitment,       // Commitment point
        const void* proof,            // Proof point
        const void* point,            // Evaluation point
        const void* value,            // Claimed evaluation
        const void* srs_g2,           // G2 element from SRS
        bool* result,                 // Verification result
        int curve_type                // KinetCurveType
    );

    // ==========================================================================
    // Crypto: secp256k1 ECDSA Recovery (Ethereum ecrecover)
    // ==========================================================================

    // Batch ecrecover: recover Ethereum addresses from ECDSA signatures.
    // signatures: array of packed (r, s, v, msg_hash) tuples (128 bytes each)
    // addresses: output array of 20-byte Ethereum addresses (32 bytes each, padded)
    // num_signatures: number of signatures to process
    KinetBackendError (*op_ecrecover_batch)(
        KinetBackendContext* ctx,
        const void* signatures,       // KinetEcrecoverInput[num_signatures]
        void* addresses,              // KinetEcrecoverOutput[num_signatures]
        size_t num_signatures
    );

    // ==========================================================================
    // Crypto: Post-Quantum Signatures
    // ==========================================================================

    // ML-DSA-65 batch verification
    KinetBackendError (*op_mldsa_verify_batch)(
        KinetBackendContext* ctx,
        const void* pubkeys,          // MLDSAPublicKey[count]
        const void* messages,         // MLDSAMessage[count]
        const void* signatures,       // MLDSASignature[count]
        uint32_t* results,            // 1=valid, 0=invalid
        size_t count
    );

    // ML-KEM-768 batch decapsulation
    KinetBackendError (*op_mlkem_decapsulate_batch)(
        KinetBackendContext* ctx,
        const void* secret_keys,      // MLKEMSecretKey[count]
        const void* ciphertexts,      // MLKEMCiphertext[count]
        void* shared_secrets,         // 32 bytes per output
        size_t count
    );

    // SLH-DSA batch verification
    KinetBackendError (*op_slhdsa_verify_batch)(
        KinetBackendContext* ctx,
        const void* pubkeys,
        const void* messages,
        const void* signatures,
        uint32_t* results,
        size_t count
    );

    // ==========================================================================
    // Crypto: Threshold Signatures
    // ==========================================================================

    // Ringtail partial signing
    KinetBackendError (*op_ringtail_partial_sign_batch)(
        KinetBackendContext* ctx,
        const void* shares,
        const void* messages,
        void* partial_sigs,
        size_t count
    );

    // Ringtail combine partial signatures
    KinetBackendError (*op_ringtail_combine_batch)(
        KinetBackendContext* ctx,
        const void* partial_sigs,
        const int32_t* lagrange_coeffs,
        void* combined_sigs,
        size_t threshold,
        size_t count
    );

    // FROST partial verification
    KinetBackendError (*op_frost_partial_verify_batch)(
        KinetBackendContext* ctx,
        const void* commitments,
        const void* signatures,
        const void* pubkeys,
        const void* challenges,
        uint32_t* results,
        size_t count
    );

    // CGGMP21 partial signing
    KinetBackendError (*op_cggmp21_partial_sign_batch)(
        KinetBackendContext* ctx,
        const void* inputs,
        const void* r_x,
        void* partial_sigs,
        size_t count
    );

    // ==========================================================================
    // Crypto: Ed25519 / sr25519
    // ==========================================================================

    // Ed25519 batch verification
    KinetBackendError (*op_ed25519_verify_batch)(
        KinetBackendContext* ctx,
        const void* pubkeys,
        const void* messages,
        const void* signatures,
        uint32_t* results,
        size_t count
    );

    // sr25519 batch verification
    KinetBackendError (*op_sr25519_verify_batch)(
        KinetBackendContext* ctx,
        const void* pubkeys,
        const void* messages,
        const void* signatures,
        uint32_t* results,
        size_t count
    );

} kinet_gpu_backend_vtbl;

// =============================================================================
// Backend Descriptor (returned by plugin init)
// =============================================================================

typedef struct {
    uint32_t abi_version;           // Must be KINET_GPU_BACKEND_ABI_VERSION
    const char* backend_name;       // "cpu" | "metal" | "cuda" | "webgpu"
    const char* backend_version;    // e.g., "0.1.0"
    uint32_t capabilities;          // Bitmask of supported features
    const kinet_gpu_backend_vtbl* vtbl;
} kinet_gpu_backend_desc;

// Capability flags
#define KINET_CAP_TENSOR_OPS      (1 << 0)   // Basic tensor ops (add, sub, mul, div)
#define KINET_CAP_MATMUL          (1 << 1)   // Matrix multiplication
#define KINET_CAP_NTT             (1 << 2)   // NTT operations
#define KINET_CAP_MSM             (1 << 3)   // Multi-scalar multiplication
#define KINET_CAP_CUSTOM_KERNELS  (1 << 4)   // Custom kernel loading
#define KINET_CAP_UNIFIED_MEMORY  (1 << 5)   // Unified memory support
#define KINET_CAP_FHE             (1 << 6)   // Fully homomorphic encryption
#define KINET_CAP_TFHE            (1 << 7)   // TFHE bootstrap/keyswitch
#define KINET_CAP_REDUCE          (1 << 8)   // Reduction ops (sum, max, min, mean)
#define KINET_CAP_SOFTMAX         (1 << 9)   // Softmax and log-softmax
#define KINET_CAP_UNARY           (1 << 10)  // Unary ops (exp, log, sqrt, tanh, etc.)
#define KINET_CAP_NORMALIZATION   (1 << 11)  // Layer norm, RMS norm
#define KINET_CAP_BLS12_381       (1 << 12)  // BLS12-381 curve operations
#define KINET_CAP_BN254           (1 << 13)  // BN254 curve operations
#define KINET_CAP_KZG             (1 << 14)  // KZG polynomial commitments
#define KINET_CAP_POSEIDON2       (1 << 15)  // Poseidon2 hash
#define KINET_CAP_BLAKE3          (1 << 16)  // BLAKE3 hash
#define KINET_CAP_BLIND_ROTATE    (1 << 17)  // Blind rotation
#define KINET_CAP_POLY_MUL        (1 << 18)  // Polynomial multiplication
#define KINET_CAP_KECCAK256       (1 << 19)  // Keccak-256 hash (Ethereum)
#define KINET_CAP_ECRECOVER       (1 << 20)  // secp256k1 ECDSA recovery (Ethereum ecrecover)
#define KINET_CAP_MLDSA           (1 << 21)  // ML-DSA-65 (FIPS 204) signature verification
#define KINET_CAP_MLKEM           (1 << 22)  // ML-KEM-768 (FIPS 203) key encapsulation
#define KINET_CAP_SLHDSA          (1 << 23)  // SLH-DSA (FIPS 205) hash-based signatures
#define KINET_CAP_RINGTAIL        (1 << 24)  // Ringtail lattice-based threshold signatures
#define KINET_CAP_FROST           (1 << 25)  // FROST threshold Schnorr signatures
#define KINET_CAP_CGGMP21         (1 << 26)  // CGGMP21 threshold ECDSA
#define KINET_CAP_ED25519         (1 << 27)  // Ed25519 EdDSA
#define KINET_CAP_SR25519         (1 << 28)  // sr25519 Schnorrkel/Ristretto255

// =============================================================================
// Plugin Entry Point
// =============================================================================

// Every backend shared library must export this symbol
// Returns true on success, false if backend unavailable on this system
typedef bool (*kinet_gpu_backend_init_fn)(kinet_gpu_backend_desc* out);

// Symbol name to dlopen
#define KINET_GPU_BACKEND_INIT_SYMBOL "kinet_gpu_backend_init"

// Macro to declare the entry point
#ifdef _WIN32
#define KINET_GPU_BACKEND_EXPORT __declspec(dllexport)
#else
#define KINET_GPU_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define KINET_GPU_DECLARE_BACKEND(init_func) \
    extern "C" KINET_GPU_BACKEND_EXPORT bool kinet_gpu_backend_init(kinet_gpu_backend_desc* out) { \
        return init_func(out); \
    }
#else
#define KINET_GPU_DECLARE_BACKEND(init_func) \
    KINET_GPU_BACKEND_EXPORT bool kinet_gpu_backend_init(kinet_gpu_backend_desc* out) { \
        return init_func(out); \
    }
#endif

#ifdef __cplusplus
}
#endif

#endif // KINET_GPU_BACKEND_PLUGIN_H

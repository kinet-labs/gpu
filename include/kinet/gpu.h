// Kinet GPU - Unified GPU acceleration with switchable backends
//
// Backends:
//   - Metal: Apple Silicon (macOS/iOS)
//   - CUDA: NVIDIA GPUs
//   - Dawn: WebGPU via Dawn (cross-platform)
//   - CPU: SIMD-optimized fallback
//
// Usage:
//   #include <kinet/gpu.h>
//
//   KinetGPU* gpu = kinet_gpu_create();
//   kinet_gpu_set_backend(gpu, KINET_BACKEND_METAL);
//
//   KinetTensor* a = kinet_tensor_zeros(gpu, shape, 2, KINET_FLOAT32);
//   KinetTensor* b = kinet_tensor_ones(gpu, shape, 2, KINET_FLOAT32);
//   KinetTensor* c = kinet_tensor_add(gpu, a, b);
//
//   kinet_gpu_sync(gpu);
//   kinet_gpu_destroy(gpu);

#ifndef KINET_GPU_H
#define KINET_GPU_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Version
// =============================================================================

#define KINET_GPU_VERSION_MAJOR 0
#define KINET_GPU_VERSION_MINOR 2
#define KINET_GPU_VERSION_PATCH 0

// =============================================================================
// Backend Types
// =============================================================================

typedef enum {
    KINET_BACKEND_AUTO = 0,  // Auto-detect best backend
    KINET_BACKEND_CPU  = 1,  // CPU with SIMD
    KINET_BACKEND_METAL = 2, // Apple Metal
    KINET_BACKEND_CUDA = 3,  // NVIDIA CUDA
    KINET_BACKEND_DAWN = 4,  // WebGPU via Dawn
} KinetBackend;

typedef enum {
    KINET_FLOAT32 = 0,
    KINET_FLOAT16 = 1,
    KINET_BFLOAT16 = 2,
    KINET_INT32 = 3,
    KINET_INT64 = 4,
    KINET_UINT32 = 5,
    KINET_UINT64 = 6,
    KINET_BOOL = 7,
} KinetDtype;

typedef enum {
    KINET_OK = 0,
    KINET_ERROR_INVALID_ARGUMENT = 1,
    KINET_ERROR_OUT_OF_MEMORY = 2,
    KINET_ERROR_BACKEND_NOT_AVAILABLE = 3,
    KINET_ERROR_DEVICE_NOT_FOUND = 4,
    KINET_ERROR_KERNEL_FAILED = 5,
    KINET_ERROR_NOT_SUPPORTED = 6,
} KinetError;

// =============================================================================
// Curve Types (for crypto operations)
// =============================================================================

typedef enum {
    KINET_CURVE_BLS12_381 = 0,
    KINET_CURVE_BN254 = 1,
    KINET_CURVE_SECP256K1 = 2,
    KINET_CURVE_ED25519 = 3,
} KinetCurve;

// =============================================================================
// Opaque Types
// =============================================================================

typedef struct KinetGPU KinetGPU;
typedef struct KinetTensor KinetTensor;
typedef struct KinetStream KinetStream;
typedef struct KinetEvent KinetEvent;

// =============================================================================
// Device Info
// =============================================================================

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

// =============================================================================
// GPU Context
// =============================================================================

// Create GPU context (auto-detects best backend)
KinetGPU* kinet_gpu_create(void);

// Create GPU context with specific backend
KinetGPU* kinet_gpu_create_with_backend(KinetBackend backend);

// Create GPU context with specific device
KinetGPU* kinet_gpu_create_with_device(KinetBackend backend, int device_index);

// Destroy GPU context
void kinet_gpu_destroy(KinetGPU* gpu);

// Get current backend
KinetBackend kinet_gpu_backend(KinetGPU* gpu);

// Get backend name string
const char* kinet_gpu_backend_name(KinetGPU* gpu);

// Switch backend at runtime
KinetError kinet_gpu_set_backend(KinetGPU* gpu, KinetBackend backend);

// Get device info
KinetError kinet_gpu_device_info(KinetGPU* gpu, KinetDeviceInfo* info);

// Synchronize all operations
KinetError kinet_gpu_sync(KinetGPU* gpu);

// Get last error message
const char* kinet_gpu_error(KinetGPU* gpu);

// =============================================================================
// Backend Query
// =============================================================================

// Get number of available backends
int kinet_backend_count(void);

// Check if backend is available
bool kinet_backend_available(KinetBackend backend);

// Get backend name
const char* kinet_backend_name(KinetBackend backend);

// Get number of devices for backend
int kinet_device_count(KinetBackend backend);

// Get device info for backend/index
KinetError kinet_device_info(KinetBackend backend, int index, KinetDeviceInfo* info);

// =============================================================================
// Tensor Operations
// =============================================================================

// Create tensor filled with zeros
KinetTensor* kinet_tensor_zeros(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype);

// Create tensor filled with ones
KinetTensor* kinet_tensor_ones(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype);

// Create tensor filled with value
KinetTensor* kinet_tensor_full(KinetGPU* gpu, const int64_t* shape, int ndim, KinetDtype dtype, double value);

// Create tensor from data
KinetTensor* kinet_tensor_from_data(KinetGPU* gpu, const void* data, const int64_t* shape, int ndim, KinetDtype dtype);

// Destroy tensor
void kinet_tensor_destroy(KinetTensor* tensor);

// Get tensor shape
int kinet_tensor_ndim(KinetTensor* tensor);
int64_t kinet_tensor_shape(KinetTensor* tensor, int dim);
int64_t kinet_tensor_size(KinetTensor* tensor);
KinetDtype kinet_tensor_dtype(KinetTensor* tensor);

// Copy tensor data to host
KinetError kinet_tensor_to_host(KinetTensor* tensor, void* data, size_t size);

// Arithmetic operations
KinetTensor* kinet_tensor_add(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
KinetTensor* kinet_tensor_sub(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
KinetTensor* kinet_tensor_mul(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
KinetTensor* kinet_tensor_div(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);
KinetTensor* kinet_tensor_matmul(KinetGPU* gpu, KinetTensor* a, KinetTensor* b);

// Unary operations
KinetTensor* kinet_tensor_neg(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_exp(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_log(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_sqrt(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_abs(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_tanh(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_sigmoid(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_relu(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_gelu(KinetGPU* gpu, KinetTensor* t);

// Reductions (full tensor -> scalar)
float kinet_tensor_reduce_sum(KinetGPU* gpu, KinetTensor* t);
float kinet_tensor_reduce_max(KinetGPU* gpu, KinetTensor* t);
float kinet_tensor_reduce_min(KinetGPU* gpu, KinetTensor* t);
float kinet_tensor_reduce_mean(KinetGPU* gpu, KinetTensor* t);

// Reductions along axes
KinetTensor* kinet_tensor_sum(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);
KinetTensor* kinet_tensor_mean(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);
KinetTensor* kinet_tensor_max(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);
KinetTensor* kinet_tensor_min(KinetGPU* gpu, KinetTensor* t, const int* axes, int naxes);

// Softmax and normalization
KinetTensor* kinet_tensor_softmax(KinetGPU* gpu, KinetTensor* t, int axis);
KinetTensor* kinet_tensor_log_softmax(KinetGPU* gpu, KinetTensor* t, int axis);
KinetTensor* kinet_tensor_layer_norm(KinetGPU* gpu, KinetTensor* t, KinetTensor* gamma, KinetTensor* beta, float eps);
KinetTensor* kinet_tensor_rms_norm(KinetGPU* gpu, KinetTensor* t, KinetTensor* weight, float eps);

// Transpose and copy
KinetTensor* kinet_tensor_transpose(KinetGPU* gpu, KinetTensor* t);
KinetTensor* kinet_tensor_copy(KinetGPU* gpu, KinetTensor* t);

// =============================================================================
// Crypto Operations: Hash Functions
// =============================================================================

// Poseidon2 hash (algebraic hash for ZK circuits)
KinetError kinet_poseidon2_hash(KinetGPU* gpu,
                            const uint64_t* inputs,      // [num_hashes * rate]
                            uint64_t* outputs,           // [num_hashes]
                            size_t rate,                 // Poseidon rate parameter
                            size_t num_hashes);

// BLAKE3 hash (high-performance cryptographic hash)
KinetError kinet_blake3_hash(KinetGPU* gpu,
                         const uint8_t* inputs,         // Concatenated inputs
                         uint8_t* outputs,              // [num_hashes * 32]
                         const size_t* input_lens,      // Length of each input
                         size_t num_hashes);

// Keccak-256 hash (Ethereum variant, NOT NIST SHA-3)
//   - Padding: 0x01 || 0x00...0x00 || 0x80 (Keccak, not SHA-3's 0x06)
//   - Output: 32 bytes per input
//   - Primary use: EVM state trie hashing, address derivation
KinetError kinet_gpu_keccak256_batch(KinetGPU* gpu,
                                 const uint8_t* inputs,         // Concatenated inputs
                                 uint8_t* outputs,              // [num_inputs * 32]
                                 const size_t* input_lens,      // Length of each input
                                 size_t num_inputs);

// =============================================================================
// Crypto Operations: secp256k1 ECDSA Recovery (Ethereum ecrecover)
// =============================================================================

// Packed signature for ecrecover batch operations.
// Each entry: r[32] || s[32] || v[1] || pad[3] || msg_hash[32] || pad[28] = 128 bytes
typedef struct {
    uint8_t r[32];        // ECDSA r value (big-endian)
    uint8_t s[32];        // ECDSA s value (big-endian)
    uint8_t v;            // Recovery id (0 or 1)
    uint8_t _pad[3];      // Alignment padding
    uint8_t msg_hash[32]; // Message hash (big-endian)
    uint8_t _pad2[28];    // Pad to 128 bytes
} KinetEcrecoverInput;

// Output of ecrecover: recovered Ethereum address.
typedef struct {
    uint8_t address[20]; // Recovered address (or zeros on failure)
    uint8_t valid;       // 1 if recovery succeeded, 0 otherwise
    uint8_t _pad[11];    // Pad to 32 bytes
} KinetEcrecoverOutput;

// Batch secp256k1 ECDSA public key recovery → Ethereum address.
//
// For each signature (r, s, v, msg_hash):
//   1. Recover public key Q from the ECDSA signature
//   2. Compute address = keccak256(Q.x || Q.y)[12:]
//
// This is the EVM ecrecover precompile, batched for GPU parallelism.
// Each GPU thread processes one signature independently.
//
// Returns KINET_OK on success (individual failures are indicated by valid=0
// in the output; the batch call itself only fails on argument errors).
KinetError kinet_gpu_ecrecover_batch(KinetGPU* gpu,
                                 const KinetEcrecoverInput* signatures,
                                 KinetEcrecoverOutput* addresses,
                                 size_t num_signatures);

// =============================================================================
// Crypto Operations: MSM (Multi-Scalar Multiplication)
// =============================================================================

KinetError kinet_msm(KinetGPU* gpu,
                 const void* scalars,           // Scalar field elements
                 const void* points,            // Curve points (affine)
                 void* result,                  // Single output point
                 size_t count,                  // Number of scalar-point pairs
                 KinetCurve curve);               // Which curve to use

// =============================================================================
// Crypto Operations: BLS12-381 Curve
// =============================================================================

// Point addition (G1 or G2)
KinetError kinet_bls12_381_add(KinetGPU* gpu,
                           const void* a, const void* b, void* out,
                           size_t count, bool is_g2);

// Scalar multiplication (G1 or G2)
KinetError kinet_bls12_381_mul(KinetGPU* gpu,
                           const void* points, const void* scalars, void* out,
                           size_t count, bool is_g2);

// Pairing computation (multi-pairing for efficiency)
KinetError kinet_bls12_381_pairing(KinetGPU* gpu,
                               const void* g1_points, const void* g2_points,
                               void* out, size_t count);

// High-level BLS signature verification
KinetError kinet_bls_verify(KinetGPU* gpu,
                        const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len,
                        bool* result);

KinetError kinet_bls_verify_batch(KinetGPU* gpu,
                              const uint8_t* const* sigs, const size_t* sig_lens,
                              const uint8_t* const* msgs, const size_t* msg_lens,
                              const uint8_t* const* pubkeys, const size_t* pubkey_lens,
                              int count, bool* results);

KinetError kinet_bls_aggregate(KinetGPU* gpu,
                           const uint8_t* const* sigs, const size_t* sig_lens,
                           int count, uint8_t* out, size_t* out_len);

// =============================================================================
// Crypto Operations: BN254 Curve
// =============================================================================

// Point addition (G1 or G2)
KinetError kinet_bn254_add(KinetGPU* gpu,
                       const void* a, const void* b, void* out,
                       size_t count, bool is_g2);

// Scalar multiplication (G1 or G2)
KinetError kinet_bn254_mul(KinetGPU* gpu,
                       const void* points, const void* scalars, void* out,
                       size_t count, bool is_g2);

// =============================================================================
// Crypto Operations: KZG Polynomial Commitments
// =============================================================================

// Commit to polynomial using SRS
KinetError kinet_kzg_commit(KinetGPU* gpu,
                        const void* coeffs,        // Polynomial coefficients
                        const void* srs,           // SRS G1 points
                        void* commitment,          // Output commitment
                        size_t degree,             // Polynomial degree
                        KinetCurve curve);

// Open commitment at evaluation point
KinetError kinet_kzg_open(KinetGPU* gpu,
                      const void* coeffs,          // Polynomial coefficients
                      const void* srs,             // SRS G1 points
                      const void* point,           // Evaluation point
                      void* proof,                 // Output proof
                      size_t degree,               // Polynomial degree
                      KinetCurve curve);

// Verify KZG opening proof
KinetError kinet_kzg_verify(KinetGPU* gpu,
                        const void* commitment,    // Commitment point
                        const void* proof,         // Proof point
                        const void* point,         // Evaluation point
                        const void* value,         // Claimed evaluation
                        const void* srs_g2,        // G2 element from SRS
                        bool* result,              // Verification result
                        KinetCurve curve);

// =============================================================================
// FHE Operations: NTT (Number Theoretic Transform)
// =============================================================================

KinetError kinet_ntt_forward(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
KinetError kinet_ntt_inverse(KinetGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
KinetError kinet_ntt_batch(KinetGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus);

// =============================================================================
// FHE Operations: Polynomial Arithmetic
// =============================================================================

// Polynomial multiplication: result = a * b mod (X^n + 1) mod modulus
KinetError kinet_poly_mul(KinetGPU* gpu,
                      const uint64_t* a, const uint64_t* b,
                      uint64_t* result, size_t n, uint64_t modulus);

// =============================================================================
// FHE Operations: TFHE
// =============================================================================

// TFHE programmable bootstrap: evaluates LUT on encrypted input
KinetError kinet_tfhe_bootstrap(KinetGPU* gpu,
                            const uint64_t* lwe_in,       // Input LWE [n_lwe + 1]
                            uint64_t* lwe_out,            // Output LWE [N + 1]
                            const uint64_t* bsk,          // Bootstrapping key
                            const uint64_t* test_poly,    // Test polynomial (LUT)
                            uint32_t n_lwe,               // Input LWE dimension
                            uint32_t N,                   // GLWE polynomial degree
                            uint32_t k,                   // GLWE dimension
                            uint32_t l,                   // Decomposition levels
                            uint64_t q);                  // Modulus

// TFHE key switching: changes LWE key
KinetError kinet_tfhe_keyswitch(KinetGPU* gpu,
                            const uint64_t* lwe_in,       // Input LWE [n_in + 1]
                            uint64_t* lwe_out,            // Output LWE [n_out + 1]
                            const uint64_t* ksk,          // Key switching key
                            uint32_t n_in,                // Input dimension
                            uint32_t n_out,               // Output dimension
                            uint32_t l,                   // Decomposition levels
                            uint32_t base_log,            // Base log
                            uint64_t q);                  // Modulus

// Blind rotation: rotates polynomial accumulator by encrypted amount
KinetError kinet_blind_rotate(KinetGPU* gpu,
                          uint64_t* acc,                  // Accumulator GLWE [(k+1) * N]
                          const uint64_t* bsk,            // Bootstrapping key
                          const uint64_t* lwe_a,          // LWE 'a' coefficients [n_lwe]
                          uint32_t n_lwe,                 // LWE dimension
                          uint32_t N,                     // GLWE polynomial degree
                          uint32_t k,                     // GLWE dimension
                          uint32_t l,                     // Decomposition levels
                          uint64_t q);                    // Modulus

// =============================================================================
// ZK Primitives: Field Elements and High-Level Operations
// =============================================================================

// BN254 scalar field element (Fr) - 256-bit integer in 4 x 64-bit limbs
// Represents elements of the scalar field of BN254 curve
typedef struct {
    uint64_t limbs[4];
} KinetFr256;

// Poseidon2 compression: out[i] = Poseidon2(left[i], right[i])
// Poseidon2 is an algebraic hash function optimized for ZK circuits.
KinetError kinet_gpu_poseidon2(KinetGPU* gpu,
                           KinetFr256* out,
                           const KinetFr256* left,
                           const KinetFr256* right,
                           size_t n);

// Merkle tree root computation using Poseidon2 hash
// Computes root from n leaves (pads to next power of 2 internally)
KinetError kinet_gpu_merkle_root(KinetGPU* gpu,
                             KinetFr256* out,
                             const KinetFr256* leaves,
                             size_t n);

// Pedersen-style commitment: out[i] = Poseidon2(Poseidon2(value, blinding), salt)
// Suitable for hiding commitments in ZK protocols
KinetError kinet_gpu_commitment(KinetGPU* gpu,
                            KinetFr256* out,
                            const KinetFr256* values,
                            const KinetFr256* blindings,
                            const KinetFr256* salts,
                            size_t n);

// Nullifier derivation: out[i] = Poseidon2(Poseidon2(key, commitment), index)
// Used to prevent double-spending in ZK protocols
KinetError kinet_gpu_nullifier(KinetGPU* gpu,
                           KinetFr256* out,
                           const KinetFr256* keys,
                           const KinetFr256* commitments,
                           const KinetFr256* indices,
                           size_t n);

// =============================================================================
// Crypto Operations: Post-Quantum Signatures
// =============================================================================

// ML-DSA-65 (FIPS 204, CRYSTALS-Dilithium) batch signature verification
// pubkeys: array of public keys (1952 bytes each)
// messages: array of message hashes (64 bytes each)
// signatures: array of signatures (3360 bytes each, padded)
// results: output boolean array (1=valid, 0=invalid)
KinetError kinet_gpu_mldsa_verify_batch(KinetGPU* gpu,
                                    const uint8_t* const* pubkeys,
                                    const uint8_t* const* messages,
                                    const uint8_t* const* signatures,
                                    bool* results,
                                    size_t count);

// ML-KEM-768 (FIPS 203, CRYSTALS-Kyber) batch decapsulation
// secret_keys: array of decapsulation keys (2400 bytes each)
// ciphertexts: array of ciphertexts (1088 bytes each)
// shared_secrets: output array of shared secrets (32 bytes each)
KinetError kinet_gpu_mlkem_decapsulate_batch(KinetGPU* gpu,
                                         const uint8_t* const* secret_keys,
                                         const uint8_t* const* ciphertexts,
                                         uint8_t** shared_secrets,
                                         size_t count);

// SLH-DSA (FIPS 205, SPHINCS+) batch signature verification
// pubkeys: array of public keys (32 bytes each for SHAKE-128f)
// messages: array of message hashes (32 bytes each)
// signatures: array of signatures (up to 17088 bytes each)
KinetError kinet_gpu_slhdsa_verify_batch(KinetGPU* gpu,
                                     const uint8_t* const* pubkeys,
                                     const uint8_t* const* messages,
                                     const uint8_t* const* signatures,
                                     bool* results,
                                     size_t count);

// =============================================================================
// Crypto Operations: Threshold Signatures
// =============================================================================

// Ringtail lattice-based threshold partial signing
// shares: array of secret shares (1024 bytes each, 256 int32 coefficients)
// messages: array of message hashes (32 bytes each)
// partial_sigs: output partial signatures (1024 bytes each)
KinetError kinet_gpu_ringtail_partial_sign_batch(KinetGPU* gpu,
                                             const uint8_t* const* shares,
                                             const uint8_t* const* messages,
                                             uint8_t** partial_sigs,
                                             size_t count);

// Ringtail threshold combine: merge k partial sigs into one
// partial_sigs: array of partial signatures [count * threshold]
// lagrange_coeffs: Lagrange interpolation coefficients [count * threshold]
// combined_sigs: output combined signatures [count]
KinetError kinet_gpu_ringtail_combine_batch(KinetGPU* gpu,
                                        const uint8_t* const* partial_sigs,
                                        const int32_t* lagrange_coeffs,
                                        uint8_t** combined_sigs,
                                        size_t threshold,
                                        size_t count);

// FROST threshold Schnorr partial signature verification
// commitments: participant commitments (66 bytes each)
// signatures: partial signature scalars (32 bytes each)
// pubkeys: public key shares (33 bytes each)
// challenges: pre-computed c*lambda_i scalars (32 bytes each)
KinetError kinet_gpu_frost_partial_verify_batch(KinetGPU* gpu,
                                            const uint8_t* const* commitments,
                                            const uint8_t* const* signatures,
                                            const uint8_t* const* pubkeys,
                                            const uint8_t* const* challenges,
                                            bool* results,
                                            size_t count);

// CGGMP21 threshold ECDSA partial signing
// inputs: k_share[32] || chi_share[32] || msg_hash[32] || gamma_share[32] per entry
// r_x: x-coordinate of combined nonce R (32 bytes)
// partial_sigs: output sigma_i values (32 bytes each)
KinetError kinet_gpu_cggmp21_partial_sign_batch(KinetGPU* gpu,
                                            const uint8_t* const* inputs,
                                            const uint8_t* r_x,
                                            uint8_t** partial_sigs,
                                            size_t count);

// =============================================================================
// Crypto Operations: Ed25519 / sr25519
// =============================================================================

// Ed25519 batch signature verification
// pubkeys: 32-byte compressed points
// messages: 64-byte pre-computed H(R||A||M), reduced mod L by host
// signatures: 64-byte signatures (R[32] || S[32])
KinetError kinet_gpu_ed25519_verify_batch(KinetGPU* gpu,
                                      const uint8_t* const* pubkeys,
                                      const uint8_t* const* messages,
                                      const uint8_t* const* signatures,
                                      bool* results,
                                      size_t count);

// sr25519 (Schnorrkel/Ristretto255) batch signature verification
// pubkeys: 32-byte Ristretto255 compressed points
// messages: 64-byte pre-computed transcript hashes
// signatures: 64-byte signatures (R[32] || s[32])
KinetError kinet_gpu_sr25519_verify_batch(KinetGPU* gpu,
                                      const uint8_t* const* pubkeys,
                                      const uint8_t* const* messages,
                                      const uint8_t* const* signatures,
                                      bool* results,
                                      size_t count);

// =============================================================================
// Stream/Event Management
// =============================================================================

KinetStream* kinet_stream_create(KinetGPU* gpu);
void kinet_stream_destroy(KinetStream* stream);
KinetError kinet_stream_sync(KinetStream* stream);

KinetEvent* kinet_event_create(KinetGPU* gpu);
void kinet_event_destroy(KinetEvent* event);
KinetError kinet_event_record(KinetEvent* event, KinetStream* stream);
KinetError kinet_event_wait(KinetEvent* event, KinetStream* stream);
float kinet_event_elapsed(KinetEvent* start, KinetEvent* end);

#ifdef __cplusplus
}
#endif

#endif // KINET_GPU_H

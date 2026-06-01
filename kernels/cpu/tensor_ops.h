// CPU Tensor Operations - SIMD-optimized implementations
// Portable across x86-64 (AVX/AVX2/AVX-512) and ARM64 (NEON)

#ifndef KINET_CPU_TENSOR_OPS_H
#define KINET_CPU_TENSOR_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Elementwise Binary Operations
// =============================================================================

void kinet_cpu_add_f32(float* out, const float* a, const float* b, size_t n);
void kinet_cpu_sub_f32(float* out, const float* a, const float* b, size_t n);
void kinet_cpu_mul_f32(float* out, const float* a, const float* b, size_t n);
void kinet_cpu_div_f32(float* out, const float* a, const float* b, size_t n);

// =============================================================================
// Unary Operations
// =============================================================================

void kinet_cpu_exp_f32(float* out, const float* in, size_t n);
void kinet_cpu_log_f32(float* out, const float* in, size_t n);
void kinet_cpu_sqrt_f32(float* out, const float* in, size_t n);
void kinet_cpu_neg_f32(float* out, const float* in, size_t n);
void kinet_cpu_abs_f32(float* out, const float* in, size_t n);
void kinet_cpu_tanh_f32(float* out, const float* in, size_t n);
void kinet_cpu_sigmoid_f32(float* out, const float* in, size_t n);
void kinet_cpu_relu_f32(float* out, const float* in, size_t n);
void kinet_cpu_gelu_f32(float* out, const float* in, size_t n);

// =============================================================================
// Matrix Operations
// =============================================================================

void kinet_cpu_matmul_f32(float* c, const float* a, const float* b, int M, int K, int N);
void kinet_cpu_transpose_f32(float* out, const float* in, int rows, int cols);

// =============================================================================
// Reduction Operations
// =============================================================================

float kinet_cpu_reduce_sum_f32(const float* in, size_t n);
float kinet_cpu_reduce_max_f32(const float* in, size_t n);
float kinet_cpu_reduce_min_f32(const float* in, size_t n);
float kinet_cpu_reduce_mean_f32(const float* in, size_t n);

void kinet_cpu_reduce_sum_axis_f32(float* out, const float* in, size_t outer_size, size_t inner_size);
void kinet_cpu_reduce_max_axis_f32(float* out, const float* in, size_t outer_size, size_t inner_size);
void kinet_cpu_reduce_mean_axis_f32(float* out, const float* in, size_t outer_size, size_t inner_size);

// =============================================================================
// Softmax Operations
// =============================================================================

void kinet_cpu_softmax_f32(float* out, const float* in, size_t batch_size, size_t dim);
void kinet_cpu_log_softmax_f32(float* out, const float* in, size_t batch_size, size_t dim);

// =============================================================================
// Normalization Operations
// =============================================================================

void kinet_cpu_layer_norm_f32(float* out, const float* in, const float* gamma, const float* beta,
                            size_t batch_size, size_t dim, float eps);
void kinet_cpu_rms_norm_f32(float* out, const float* in, const float* weight,
                          size_t batch_size, size_t dim, float eps);

// =============================================================================
// Copy
// =============================================================================

void kinet_cpu_copy_f32(float* dst, const float* src, size_t n);

#ifdef __cplusplus
}
#endif

#endif // KINET_CPU_TENSOR_OPS_H

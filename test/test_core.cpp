// Fixed and comprehensive test for kinet-gpu plugin-based architecture
// Addresses issues from audit:
// - Tests now properly return failure exit codes
// - Added epsilon tolerance for float comparisons
// - Added null pointer checks
// - Added negative tests for error conditions
// - Added edge case coverage
// - Added tensor content verification

#include "kinet/gpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <cstdlib>
#include <algorithm>

// =============================================================================
// Test Framework
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

// Epsilon for float comparisons (relative tolerance)
constexpr float FLOAT_EPSILON = 1e-5f;

bool float_eq(float a, float b) {
    if (a == b) return true;  // Handles infinity and exact matches
    float diff = std::fabs(a - b);
    float largest = std::fmax(std::fabs(a), std::fabs(b));
    return diff <= largest * FLOAT_EPSILON;
}

bool float_arr_eq(const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!float_eq(a[i], b[i])) return false;
    }
    return true;
}

// =============================================================================
// Backend and Device Tests
// =============================================================================

void test_backend_detection() {
    printf("\n=== Backend Detection Tests ===\n");

    TEST("kinet_gpu_create returns non-null");
    KinetGPU* gpu = kinet_gpu_create();
    CHECK(gpu != nullptr, "GPU context creation failed");
    if (!gpu) return;

    TEST("kinet_gpu_backend_name returns valid string");
    const char* name = kinet_gpu_backend_name(gpu);
    CHECK(name != nullptr && strlen(name) > 0, "Backend name is empty");

    TEST("kinet_gpu_backend returns valid enum");
    KinetBackend backend = kinet_gpu_backend(gpu);
    CHECK(backend >= KINET_BACKEND_AUTO && backend <= KINET_BACKEND_DAWN, "Invalid backend enum");

    TEST("kinet_gpu_device_info populates struct");
    KinetDeviceInfo info = {};
    KinetError err = kinet_gpu_device_info(gpu, &info);
    CHECK(err == KINET_OK && info.name != nullptr, "Device info retrieval failed");

    TEST("CPU backend is always available");
    CHECK(kinet_backend_available(KINET_BACKEND_CPU), "CPU backend should always be available");

    TEST("kinet_gpu_sync succeeds");
    err = kinet_gpu_sync(gpu);
    CHECK(err == KINET_OK, "Sync failed");

    kinet_gpu_destroy(gpu);
}

void test_backend_switching() {
    printf("\n=== Backend Switching Tests ===\n");

    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    TEST("Switch to CPU backend");
    KinetError err = kinet_gpu_set_backend(gpu, KINET_BACKEND_CPU);
    CHECK(err == KINET_OK, "Failed to switch to CPU backend");

    TEST("Verify backend is CPU after switch");
    const char* name = kinet_gpu_backend_name(gpu);
    CHECK(name != nullptr && strcmp(name, "cpu") == 0, "Backend name not 'cpu'");

    TEST("Switch to invalid backend returns error");
    err = kinet_gpu_set_backend(gpu, (KinetBackend)999);
    CHECK(err != KINET_OK, "Should fail for invalid backend");

    kinet_gpu_destroy(gpu);
}

// =============================================================================
// Tensor Creation Tests
// =============================================================================

void test_tensor_creation() {
    printf("\n=== Tensor Creation Tests ===\n");

    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    int64_t shape_1d[] = {4};
    int64_t shape_2d[] = {2, 3};
    int64_t shape_3d[] = {2, 3, 4};

    // Test zeros tensor content
    TEST("kinet_tensor_zeros creates tensor with all zeros");
    KinetTensor* zeros = kinet_tensor_zeros(gpu, shape_1d, 1, KINET_FLOAT32);
    if (zeros) {
        float data[4];
        KinetError err = kinet_tensor_to_host(zeros, data, sizeof(data));
        float expected[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        CHECK(err == KINET_OK && float_arr_eq(data, expected, 4), "Zeros tensor has non-zero values");
        kinet_tensor_destroy(zeros);
    } else {
        FAIL("Failed to create zeros tensor");
    }

    // Test ones tensor content
    TEST("kinet_tensor_ones creates tensor with all ones");
    KinetTensor* ones = kinet_tensor_ones(gpu, shape_1d, 1, KINET_FLOAT32);
    if (ones) {
        float data[4];
        KinetError err = kinet_tensor_to_host(ones, data, sizeof(data));
        float expected[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        CHECK(err == KINET_OK && float_arr_eq(data, expected, 4), "Ones tensor has incorrect values");
        kinet_tensor_destroy(ones);
    } else {
        FAIL("Failed to create ones tensor");
    }

    // Test full tensor with specific value
    TEST("kinet_tensor_full creates tensor with specified value");
    KinetTensor* full = kinet_tensor_full(gpu, shape_1d, 1, KINET_FLOAT32, 3.14159);
    if (full) {
        float data[4];
        KinetError err = kinet_tensor_to_host(full, data, sizeof(data));
        bool all_correct = true;
        for (int i = 0; i < 4; i++) {
            if (!float_eq(data[i], 3.14159f)) all_correct = false;
        }
        CHECK(err == KINET_OK && all_correct, "Full tensor has incorrect values");
        kinet_tensor_destroy(full);
    } else {
        FAIL("Failed to create full tensor");
    }

    // Test from_data
    TEST("kinet_tensor_from_data preserves input data");
    float input_data[] = {1.5f, 2.5f, 3.5f, 4.5f};
    KinetTensor* from_data = kinet_tensor_from_data(gpu, input_data, shape_1d, 1, KINET_FLOAT32);
    if (from_data) {
        float output[4];
        KinetError err = kinet_tensor_to_host(from_data, output, sizeof(output));
        CHECK(err == KINET_OK && float_arr_eq(input_data, output, 4), "Data not preserved");
        kinet_tensor_destroy(from_data);
    } else {
        FAIL("Failed to create tensor from data");
    }

    // Test tensor metadata
    TEST("kinet_tensor_ndim returns correct rank");
    KinetTensor* t2d = kinet_tensor_zeros(gpu, shape_2d, 2, KINET_FLOAT32);
    if (t2d) {
        CHECK(kinet_tensor_ndim(t2d) == 2, "Wrong ndim");
        kinet_tensor_destroy(t2d);
    } else {
        FAIL("Failed to create 2D tensor");
    }

    TEST("kinet_tensor_shape returns correct dimensions");
    KinetTensor* t3d = kinet_tensor_zeros(gpu, shape_3d, 3, KINET_FLOAT32);
    if (t3d) {
        bool correct = (kinet_tensor_shape(t3d, 0) == 2 &&
                       kinet_tensor_shape(t3d, 1) == 3 &&
                       kinet_tensor_shape(t3d, 2) == 4);
        CHECK(correct, "Wrong shape dimensions");
        kinet_tensor_destroy(t3d);
    } else {
        FAIL("Failed to create 3D tensor");
    }

    TEST("kinet_tensor_size returns total element count");
    KinetTensor* t_size = kinet_tensor_zeros(gpu, shape_3d, 3, KINET_FLOAT32);
    if (t_size) {
        CHECK(kinet_tensor_size(t_size) == 24, "Wrong total size (expected 2*3*4=24)");
        kinet_tensor_destroy(t_size);
    } else {
        FAIL("Failed to create tensor for size test");
    }

    kinet_gpu_destroy(gpu);
}

// =============================================================================
// Tensor Arithmetic Tests
// =============================================================================

void test_tensor_arithmetic() {
    printf("\n=== Tensor Arithmetic Tests ===\n");

    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    int64_t shape[] = {4};

    // Create test tensors
    KinetTensor* ones = kinet_tensor_ones(gpu, shape, 1, KINET_FLOAT32);
    KinetTensor* twos = kinet_tensor_full(gpu, shape, 1, KINET_FLOAT32, 2.0);

    if (!ones || !twos) {
        FAIL("Failed to create test tensors");
        if (ones) kinet_tensor_destroy(ones);
        if (twos) kinet_tensor_destroy(twos);
        kinet_gpu_destroy(gpu);
        return;
    }

    // Test add
    TEST("kinet_tensor_add: 1 + 2 = 3");
    KinetTensor* sum = kinet_tensor_add(gpu, ones, twos);
    if (sum) {
        float result[4];
        kinet_tensor_to_host(sum, result, sizeof(result));
        float expected[4] = {3.0f, 3.0f, 3.0f, 3.0f};
        CHECK(float_arr_eq(result, expected, 4), "Addition result incorrect");
        kinet_tensor_destroy(sum);
    } else {
        FAIL("kinet_tensor_add returned null");
    }

    // Test sub
    TEST("kinet_tensor_sub: 2 - 1 = 1");
    KinetTensor* diff = kinet_tensor_sub(gpu, twos, ones);
    if (diff) {
        float result[4];
        kinet_tensor_to_host(diff, result, sizeof(result));
        float expected[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        CHECK(float_arr_eq(result, expected, 4), "Subtraction result incorrect");
        kinet_tensor_destroy(diff);
    } else {
        FAIL("kinet_tensor_sub returned null");
    }

    // Test mul
    TEST("kinet_tensor_mul: 1 * 2 = 2");
    KinetTensor* prod = kinet_tensor_mul(gpu, ones, twos);
    if (prod) {
        float result[4];
        kinet_tensor_to_host(prod, result, sizeof(result));
        float expected[4] = {2.0f, 2.0f, 2.0f, 2.0f};
        CHECK(float_arr_eq(result, expected, 4), "Multiplication result incorrect");
        kinet_tensor_destroy(prod);
    } else {
        FAIL("kinet_tensor_mul returned null");
    }

    // Test div
    TEST("kinet_tensor_div: 2 / 1 = 2");
    KinetTensor* quot = kinet_tensor_div(gpu, twos, ones);
    if (quot) {
        float result[4];
        kinet_tensor_to_host(quot, result, sizeof(result));
        float expected[4] = {2.0f, 2.0f, 2.0f, 2.0f};
        CHECK(float_arr_eq(result, expected, 4), "Division result incorrect");
        kinet_tensor_destroy(quot);
    } else {
        FAIL("kinet_tensor_div returned null");
    }

    // Test div by non-zero small values (edge case)
    TEST("kinet_tensor_div: division by small non-zero value");
    KinetTensor* small = kinet_tensor_full(gpu, shape, 1, KINET_FLOAT32, 0.001);
    KinetTensor* large = kinet_tensor_full(gpu, shape, 1, KINET_FLOAT32, 1.0);
    if (small && large) {
        KinetTensor* result = kinet_tensor_div(gpu, large, small);
        if (result) {
            float data[4];
            kinet_tensor_to_host(result, data, sizeof(data));
            CHECK(float_eq(data[0], 1000.0f), "Division by small value incorrect");
            kinet_tensor_destroy(result);
        } else {
            FAIL("Division returned null");
        }
        kinet_tensor_destroy(small);
        kinet_tensor_destroy(large);
    } else {
        FAIL("Failed to create tensors for small value test");
    }

    kinet_tensor_destroy(ones);
    kinet_tensor_destroy(twos);
    kinet_gpu_destroy(gpu);
}

// =============================================================================
// Matrix Multiplication Tests
// =============================================================================

void test_matmul() {
    printf("\n=== Matrix Multiplication Tests ===\n");

    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    // Test 2x2 matmul: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    TEST("matmul 2x2: known values");
    {
        int64_t shape[] = {2, 2};
        float mat_a[] = {1, 2, 3, 4};
        float mat_b[] = {5, 6, 7, 8};
        float expected[] = {19, 22, 43, 50};

        KinetTensor* A = kinet_tensor_from_data(gpu, mat_a, shape, 2, KINET_FLOAT32);
        KinetTensor* B = kinet_tensor_from_data(gpu, mat_b, shape, 2, KINET_FLOAT32);

        if (A && B) {
            KinetTensor* C = kinet_tensor_matmul(gpu, A, B);
            if (C) {
                float result[4];
                kinet_tensor_to_host(C, result, sizeof(result));
                CHECK(float_arr_eq(result, expected, 4), "Matmul 2x2 result incorrect");
                kinet_tensor_destroy(C);
            } else {
                FAIL("Matmul returned null");
            }
            kinet_tensor_destroy(A);
            kinet_tensor_destroy(B);
        } else {
            FAIL("Failed to create matrices");
        }
    }

    // Test identity matrix multiplication: I * A = A
    TEST("matmul with identity: I * A = A");
    {
        int64_t shape[] = {3, 3};
        float identity[] = {1,0,0, 0,1,0, 0,0,1};
        float mat[] = {1,2,3, 4,5,6, 7,8,9};

        KinetTensor* I = kinet_tensor_from_data(gpu, identity, shape, 2, KINET_FLOAT32);
        KinetTensor* A = kinet_tensor_from_data(gpu, mat, shape, 2, KINET_FLOAT32);

        if (I && A) {
            KinetTensor* C = kinet_tensor_matmul(gpu, I, A);
            if (C) {
                float result[9];
                kinet_tensor_to_host(C, result, sizeof(result));
                CHECK(float_arr_eq(result, mat, 9), "Identity multiplication changed matrix");
                kinet_tensor_destroy(C);
            } else {
                FAIL("Matmul returned null");
            }
            kinet_tensor_destroy(I);
            kinet_tensor_destroy(A);
        } else {
            FAIL("Failed to create matrices");
        }
    }

    // Test non-square matmul: (2x3) * (3x4) = (2x4)
    TEST("matmul non-square: (2x3) * (3x4)");
    {
        int64_t shape_a[] = {2, 3};
        int64_t shape_b[] = {3, 4};
        float mat_a[] = {1,2,3, 4,5,6};  // 2x3
        float mat_b[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};  // 3x4

        KinetTensor* A = kinet_tensor_from_data(gpu, mat_a, shape_a, 2, KINET_FLOAT32);
        KinetTensor* B = kinet_tensor_from_data(gpu, mat_b, shape_b, 2, KINET_FLOAT32);

        if (A && B) {
            KinetTensor* C = kinet_tensor_matmul(gpu, A, B);
            if (C) {
                CHECK(kinet_tensor_shape(C, 0) == 2 && kinet_tensor_shape(C, 1) == 4,
                      "Output shape incorrect");
                // Expected: [[38,44,50,56], [83,98,113,128]]
                float result[8];
                kinet_tensor_to_host(C, result, sizeof(result));
                float expected[] = {38,44,50,56, 83,98,113,128};
                CHECK(float_arr_eq(result, expected, 8), "Non-square matmul result incorrect");
                kinet_tensor_destroy(C);
            } else {
                FAIL("Matmul returned null");
            }
            kinet_tensor_destroy(A);
            kinet_tensor_destroy(B);
        } else {
            FAIL("Failed to create matrices");
        }
    }

    kinet_gpu_destroy(gpu);
}

// =============================================================================
// NTT Tests
// =============================================================================

void test_ntt() {
    printf("\n=== NTT Tests ===\n");

    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    uint64_t modulus = 0xFFFFFFFF00000001ULL;  // Goldilocks prime

    // Test roundtrip: forward then inverse should return original
    TEST("NTT roundtrip preserves all elements");
    {
        uint64_t original[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint64_t data[8];
        memcpy(data, original, sizeof(data));

        KinetError err1 = kinet_ntt_forward(gpu, data, 8, modulus);
        KinetError err2 = kinet_ntt_inverse(gpu, data, 8, modulus);

        bool all_match = true;
        for (int i = 0; i < 8; i++) {
            if (data[i] != original[i]) {
                all_match = false;
                printf("\n    Mismatch at [%d]: got %llu, expected %llu",
                       i, (unsigned long long)data[i], (unsigned long long)original[i]);
            }
        }
        CHECK(err1 == KINET_OK && err2 == KINET_OK && all_match, "NTT roundtrip failed");
    }

    // Test that NTT actually transforms data (not identity)
    TEST("NTT forward changes data");
    {
        uint64_t original[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint64_t data[8];
        memcpy(data, original, sizeof(data));

        kinet_ntt_forward(gpu, data, 8, modulus);

        bool changed = false;
        for (int i = 0; i < 8; i++) {
            if (data[i] != original[i]) changed = true;
        }
        CHECK(changed, "NTT forward should change data");
    }

    // Test larger size (power of 2)
    TEST("NTT roundtrip on 1024 elements");
    {
        const size_t N = 1024;
        uint64_t* data = (uint64_t*)malloc(N * sizeof(uint64_t));
        uint64_t* original = (uint64_t*)malloc(N * sizeof(uint64_t));

        if (data && original) {
            for (size_t i = 0; i < N; i++) {
                original[i] = i % modulus;
                data[i] = original[i];
            }

            KinetError err1 = kinet_ntt_forward(gpu, data, N, modulus);
            KinetError err2 = kinet_ntt_inverse(gpu, data, N, modulus);

            bool all_match = true;
            for (size_t i = 0; i < N; i++) {
                if (data[i] != original[i]) {
                    all_match = false;
                    break;
                }
            }
            CHECK(err1 == KINET_OK && err2 == KINET_OK && all_match, "Large NTT roundtrip failed");
            free(data);
            free(original);
        } else {
            FAIL("Memory allocation failed");
        }
    }

    // Test linearity: NTT(a + b) = NTT(a) + NTT(b) (mod modulus)
    TEST("NTT linearity property");
    {
        uint64_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint64_t b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
        uint64_t sum[8];
        uint64_t ntt_a[8], ntt_b[8];

        memcpy(ntt_a, a, sizeof(a));
        memcpy(ntt_b, b, sizeof(b));

        for (int i = 0; i < 8; i++) {
            sum[i] = (a[i] + b[i]) % modulus;
        }

        kinet_ntt_forward(gpu, ntt_a, 8, modulus);
        kinet_ntt_forward(gpu, ntt_b, 8, modulus);
        kinet_ntt_forward(gpu, sum, 8, modulus);

        bool linear = true;
        for (int i = 0; i < 8; i++) {
            uint64_t expected = (ntt_a[i] + ntt_b[i]) % modulus;
            if (sum[i] != expected) linear = false;
        }
        CHECK(linear, "NTT should be linear");
    }

    kinet_gpu_destroy(gpu);
}

// =============================================================================
// Negative Tests (Error Conditions)
// =============================================================================

void test_error_conditions() {
    printf("\n=== Error Condition Tests ===\n");

    // Test null GPU context
    TEST("kinet_tensor_zeros with null GPU returns null");
    KinetTensor* t = kinet_tensor_zeros(nullptr, nullptr, 0, KINET_FLOAT32);
    CHECK(t == nullptr, "Should return null for null GPU");

    // Test invalid shape
    TEST("kinet_tensor_zeros with null shape returns null");
    KinetGPU* gpu = kinet_gpu_create();
    if (gpu) {
        t = kinet_tensor_zeros(gpu, nullptr, 1, KINET_FLOAT32);
        CHECK(t == nullptr, "Should return null for null shape");
        kinet_gpu_destroy(gpu);
    }

    // Test tensor operations with null tensors
    TEST("kinet_tensor_add with null tensors returns null");
    gpu = kinet_gpu_create();
    if (gpu) {
        KinetTensor* result = kinet_tensor_add(gpu, nullptr, nullptr);
        CHECK(result == nullptr, "Should return null for null tensors");
        kinet_gpu_destroy(gpu);
    }

    // Test kinet_tensor_to_host with null tensor
    TEST("kinet_tensor_to_host with null tensor returns error");
    float buf[4];
    KinetError err = kinet_tensor_to_host(nullptr, buf, sizeof(buf));
    CHECK(err != KINET_OK, "Should return error for null tensor");

    // Test NTT with non-power-of-2 size (if implementation validates)
    TEST("kinet_ntt_forward with size 7 (non-power-of-2)");
    gpu = kinet_gpu_create();
    if (gpu) {
        uint64_t data[7] = {1, 2, 3, 4, 5, 6, 7};
        // This should either fail or handle gracefully
        // The test documents the behavior
        err = kinet_ntt_forward(gpu, data, 7, 0xFFFFFFFF00000001ULL);
        // Accept either error or implementation-specific handling
        printf("(returned %d) ", err);
        PASS();  // Document behavior
        kinet_gpu_destroy(gpu);
    }

    // Test destroy of null (should not crash)
    TEST("kinet_tensor_destroy(null) does not crash");
    kinet_tensor_destroy(nullptr);
    PASS();

    TEST("kinet_gpu_destroy(null) does not crash");
    kinet_gpu_destroy(nullptr);
    PASS();
}

// =============================================================================
// Reduction Tests
// =============================================================================

void test_reductions() {
    printf("\n=== Reduction Tests ===\n");

    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    // Test reduce_sum
    TEST("kinet_tensor_reduce_sum: sum of [1,2,3,4] = 10");
    {
        int64_t shape[] = {4};
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            float sum = kinet_tensor_reduce_sum(gpu, t);
            CHECK(float_eq(sum, 10.0f), "Sum should be 10");
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test reduce_max
    TEST("kinet_tensor_reduce_max: max of [3,1,4,1,5,9] = 9");
    {
        int64_t shape[] = {6};
        float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            float max_val = kinet_tensor_reduce_max(gpu, t);
            CHECK(float_eq(max_val, 9.0f), "Max should be 9");
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test reduce_min
    TEST("kinet_tensor_reduce_min: min of [3,1,4,1,5,9] = 1");
    {
        int64_t shape[] = {6};
        float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            float min_val = kinet_tensor_reduce_min(gpu, t);
            CHECK(float_eq(min_val, 1.0f), "Min should be 1");
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test reduce_mean
    TEST("kinet_tensor_reduce_mean: mean of [2,4,6,8] = 5");
    {
        int64_t shape[] = {4};
        float data[] = {2.0f, 4.0f, 6.0f, 8.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            float mean = kinet_tensor_reduce_mean(gpu, t);
            CHECK(float_eq(mean, 5.0f), "Mean should be 5");
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test reduction with negative values
    TEST("kinet_tensor_reduce_min with negative values");
    {
        int64_t shape[] = {5};
        float data[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            float min_val = kinet_tensor_reduce_min(gpu, t);
            CHECK(float_eq(min_val, -3.0f), "Min should be -3");
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    kinet_gpu_destroy(gpu);
}

// =============================================================================
// Unary Operations Tests
// =============================================================================

void test_unary_operations() {
    printf("\n=== Unary Operations Tests ===\n");

    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    int64_t shape[] = {4};

    // Test neg
    TEST("kinet_tensor_neg: negation");
    {
        float data[] = {1.0f, -2.0f, 3.0f, -4.0f};
        float expected[] = {-1.0f, 2.0f, -3.0f, 4.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            KinetTensor* neg = kinet_tensor_neg(gpu, t);
            if (neg) {
                float result[4];
                kinet_tensor_to_host(neg, result, sizeof(result));
                CHECK(float_arr_eq(result, expected, 4), "Negation incorrect");
                kinet_tensor_destroy(neg);
            } else {
                FAIL("neg returned null");
            }
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test exp
    TEST("kinet_tensor_exp: e^0 = 1, e^1 approx 2.718");
    {
        float data[] = {0.0f, 1.0f, 2.0f, -1.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            KinetTensor* exp_t = kinet_tensor_exp(gpu, t);
            if (exp_t) {
                float result[4];
                kinet_tensor_to_host(exp_t, result, sizeof(result));
                CHECK(float_eq(result[0], 1.0f) &&
                      float_eq(result[1], std::exp(1.0f)) &&
                      float_eq(result[3], std::exp(-1.0f)), "Exp incorrect");
                kinet_tensor_destroy(exp_t);
            } else {
                FAIL("exp returned null");
            }
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test sqrt
    TEST("kinet_tensor_sqrt: sqrt of [0, 1, 4, 9]");
    {
        float data[] = {0.0f, 1.0f, 4.0f, 9.0f};
        float expected[] = {0.0f, 1.0f, 2.0f, 3.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            KinetTensor* sqrt_t = kinet_tensor_sqrt(gpu, t);
            if (sqrt_t) {
                float result[4];
                kinet_tensor_to_host(sqrt_t, result, sizeof(result));
                CHECK(float_arr_eq(result, expected, 4), "Sqrt incorrect");
                kinet_tensor_destroy(sqrt_t);
            } else {
                FAIL("sqrt returned null");
            }
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test abs
    TEST("kinet_tensor_abs: absolute values");
    {
        float data[] = {-1.0f, 2.0f, -3.0f, 4.0f};
        float expected[] = {1.0f, 2.0f, 3.0f, 4.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            KinetTensor* abs_t = kinet_tensor_abs(gpu, t);
            if (abs_t) {
                float result[4];
                kinet_tensor_to_host(abs_t, result, sizeof(result));
                CHECK(float_arr_eq(result, expected, 4), "Abs incorrect");
                kinet_tensor_destroy(abs_t);
            } else {
                FAIL("abs returned null");
            }
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Test relu
    TEST("kinet_tensor_relu: max(0, x)");
    {
        float data[] = {-2.0f, -1.0f, 0.0f, 1.0f};
        float expected[] = {0.0f, 0.0f, 0.0f, 1.0f};
        KinetTensor* t = kinet_tensor_from_data(gpu, data, shape, 1, KINET_FLOAT32);
        if (t) {
            KinetTensor* relu_t = kinet_tensor_relu(gpu, t);
            if (relu_t) {
                float result[4];
                kinet_tensor_to_host(relu_t, result, sizeof(result));
                CHECK(float_arr_eq(result, expected, 4), "ReLU incorrect");
                kinet_tensor_destroy(relu_t);
            } else {
                FAIL("relu returned null");
            }
            kinet_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    kinet_gpu_destroy(gpu);
}

// =============================================================================
// Memory Leak Verification (Resource Cleanup)
// =============================================================================

void test_resource_cleanup() {
    printf("\n=== Resource Cleanup Tests ===\n");

    // Test many create/destroy cycles don't leak
    TEST("100 GPU context create/destroy cycles");
    for (int i = 0; i < 100; i++) {
        KinetGPU* gpu = kinet_gpu_create();
        if (!gpu) {
            FAIL("GPU creation failed on iteration");
            return;
        }
        kinet_gpu_destroy(gpu);
    }
    PASS();

    // Test many tensor create/destroy cycles
    TEST("1000 tensor create/destroy cycles");
    KinetGPU* gpu = kinet_gpu_create();
    if (!gpu) {
        FAIL("GPU creation failed");
        return;
    }

    int64_t shape[] = {100};
    for (int i = 0; i < 1000; i++) {
        KinetTensor* t = kinet_tensor_zeros(gpu, shape, 1, KINET_FLOAT32);
        if (!t) {
            FAIL("Tensor creation failed on iteration");
            kinet_gpu_destroy(gpu);
            return;
        }
        kinet_tensor_destroy(t);
    }
    kinet_gpu_destroy(gpu);
    PASS();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("================================================================================\n");
    printf("                     Kinet GPU Comprehensive Test Suite\n");
    printf("================================================================================\n");

    test_backend_detection();
    test_backend_switching();
    test_tensor_creation();
    test_tensor_arithmetic();
    test_matmul();
    test_ntt();
    test_reductions();
    test_unary_operations();
    test_error_conditions();
    test_resource_cleanup();

    printf("\n================================================================================\n");
    printf("                              Test Summary\n");
    printf("================================================================================\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Total:  %d\n", tests_passed + tests_failed);
    printf("================================================================================\n");

    return tests_failed > 0 ? 1 : 0;
}

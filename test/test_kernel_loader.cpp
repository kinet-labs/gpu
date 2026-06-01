// Kernel Loader Tests

#include "kinet/gpu/kernel_loader.h"
#include <cstdio>
#include <cstring>
#include <cassert>

// Test helpers
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { \
        printf("  %-50s ", name); \
    } while(0)

#define PASS() \
    do { \
        printf("[PASS]\n"); \
        tests_passed++; \
    } while(0)

#define FAIL(msg) \
    do { \
        printf("[FAIL] %s\n", msg); \
        tests_failed++; \
    } while(0)

#define ASSERT(cond, msg) \
    if (!(cond)) { FAIL(msg); return; }

// =============================================================================
// Kernel Cache Tests
// =============================================================================

void test_cache_create_destroy() {
    TEST("cache create/destroy");

    KinetKernelCache* cache = kinet_kernel_cache_create();
    ASSERT(cache != nullptr, "cache creation failed");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

void test_cache_stats_empty() {
    TEST("cache stats (empty)");

    KinetKernelCache* cache = kinet_kernel_cache_create();

    size_t count = 999;
    size_t memory = 999;
    kinet_kernel_cache_stats(cache, &count, &memory);

    ASSERT(count == 0, "count should be 0");
    ASSERT(memory == 0, "memory should be 0");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

void test_cache_get_missing() {
    TEST("cache get (missing key)");

    KinetKernelCache* cache = kinet_kernel_cache_create();

    KinetKernelVariant variant = { "nonexistent", 0, 0, 0 };
    KinetKernel* kernel = kinet_kernel_cache_get(cache, &variant);

    ASSERT(kernel == nullptr, "should return null for missing key");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

void test_cache_put_get() {
    TEST("cache put/get");

    KinetKernelCache* cache = kinet_kernel_cache_create();

    // Use a dummy pointer as kernel (we're testing cache, not actual kernels)
    KinetKernel* dummy_kernel = reinterpret_cast<KinetKernel*>(0xDEADBEEF);

    KinetKernelVariant variant = { "test_kernel", 0, 0, 0 };
    kinet_kernel_cache_put(cache, &variant, dummy_kernel);

    KinetKernel* retrieved = kinet_kernel_cache_get(cache, &variant);
    ASSERT(retrieved == dummy_kernel, "retrieved kernel should match");

    // Check stats
    size_t count = 0;
    kinet_kernel_cache_stats(cache, &count, nullptr);
    ASSERT(count == 1, "count should be 1");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

void test_cache_variant_dtype() {
    TEST("cache variant by dtype");

    KinetKernelCache* cache = kinet_kernel_cache_create();

    KinetKernel* kernel_f32 = reinterpret_cast<KinetKernel*>(0x1111);
    KinetKernel* kernel_f16 = reinterpret_cast<KinetKernel*>(0x2222);

    KinetKernelVariant v_f32 = { "add", 0, 0, 0 };  // dtype 0 = float32
    KinetKernelVariant v_f16 = { "add", 1, 0, 0 };  // dtype 1 = float16

    kinet_kernel_cache_put(cache, &v_f32, kernel_f32);
    kinet_kernel_cache_put(cache, &v_f16, kernel_f16);

    ASSERT(kinet_kernel_cache_get(cache, &v_f32) == kernel_f32, "f32 mismatch");
    ASSERT(kinet_kernel_cache_get(cache, &v_f16) == kernel_f16, "f16 mismatch");

    size_t count = 0;
    kinet_kernel_cache_stats(cache, &count, nullptr);
    ASSERT(count == 2, "count should be 2");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

void test_cache_variant_size() {
    TEST("cache variant by size_hint");

    KinetKernelCache* cache = kinet_kernel_cache_create();

    KinetKernel* kernel_256 = reinterpret_cast<KinetKernel*>(0x100);
    KinetKernel* kernel_1024 = reinterpret_cast<KinetKernel*>(0x400);

    KinetKernelVariant v_256 = { "ntt", 0, 256, 0 };
    KinetKernelVariant v_1024 = { "ntt", 0, 1024, 0 };

    kinet_kernel_cache_put(cache, &v_256, kernel_256);
    kinet_kernel_cache_put(cache, &v_1024, kernel_1024);

    ASSERT(kinet_kernel_cache_get(cache, &v_256) == kernel_256, "256 mismatch");
    ASSERT(kinet_kernel_cache_get(cache, &v_1024) == kernel_1024, "1024 mismatch");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

void test_cache_clear() {
    TEST("cache clear");

    KinetKernelCache* cache = kinet_kernel_cache_create();

    KinetKernel* dummy = reinterpret_cast<KinetKernel*>(0xCAFE);
    KinetKernelVariant v1 = { "kernel1", 0, 0, 0 };
    KinetKernelVariant v2 = { "kernel2", 0, 0, 0 };

    kinet_kernel_cache_put(cache, &v1, dummy);
    kinet_kernel_cache_put(cache, &v2, dummy);

    size_t count = 0;
    kinet_kernel_cache_stats(cache, &count, nullptr);
    ASSERT(count == 2, "should have 2 entries");

    kinet_kernel_cache_clear(cache);

    kinet_kernel_cache_stats(cache, &count, nullptr);
    ASSERT(count == 0, "should be empty after clear");

    ASSERT(kinet_kernel_cache_get(cache, &v1) == nullptr, "v1 should be gone");
    ASSERT(kinet_kernel_cache_get(cache, &v2) == nullptr, "v2 should be gone");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

void test_cache_null_name() {
    TEST("cache with null name");

    KinetKernelCache* cache = kinet_kernel_cache_create();

    KinetKernel* dummy = reinterpret_cast<KinetKernel*>(0xBEEF);
    KinetKernelVariant v1 = { nullptr, 0, 0, 0 };
    KinetKernelVariant v2 = { nullptr, 1, 0, 0 };

    kinet_kernel_cache_put(cache, &v1, dummy);
    kinet_kernel_cache_put(cache, &v2, dummy);

    // Both should be retrievable by their unique dtype
    ASSERT(kinet_kernel_cache_get(cache, &v1) == dummy, "v1 retrieval failed");
    ASSERT(kinet_kernel_cache_get(cache, &v2) == dummy, "v2 retrieval failed");

    kinet_kernel_cache_destroy(cache);
    PASS();
}

// =============================================================================
// Registry Tests
// =============================================================================

void test_registry_get_unknown() {
    TEST("registry get unknown backend");

    const KinetKernelRegistry* reg = kinet_kernel_registry_get("unknown_backend");
    ASSERT(reg == nullptr, "should return null for unknown backend");

    PASS();
}

void test_registry_get_null() {
    TEST("registry get null");

    const KinetKernelRegistry* reg = kinet_kernel_registry_get(nullptr);
    ASSERT(reg == nullptr, "should return null for null input");

    PASS();
}

void test_registry_find_null() {
    TEST("registry find with null args");

    // Find with null registry
    const KinetEmbeddedKernel* k = kinet_kernel_registry_find(nullptr, "test");
    ASSERT(k == nullptr, "should return null with null registry");

    // Can't test with null name on real registry without embedded kernels
    PASS();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("\n=== Kernel Loader Tests ===\n\n");

    printf("Kernel Cache:\n");
    test_cache_create_destroy();
    test_cache_stats_empty();
    test_cache_get_missing();
    test_cache_put_get();
    test_cache_variant_dtype();
    test_cache_variant_size();
    test_cache_clear();
    test_cache_null_name();

    printf("\nKernel Registry:\n");
    test_registry_get_unknown();
    test_registry_get_null();
    test_registry_find_null();

    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("\n");

    return tests_failed > 0 ? 1 : 0;
}

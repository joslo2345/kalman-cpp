// Counts heap allocations on macOS by interposing the C allocation functions
// (operator new and cv::fastMalloc both end up here). Linked into bench_alloc;
// heaptrack, which the guide uses, is Linux-only.
#include <atomic>
#include <cstdlib>
#include <malloc/malloc.h>

#define DYLD_INTERPOSE(replacement, replacee)                                                       \
    __attribute__((used)) static struct {                                                           \
        const void* replacement;                                                                    \
        const void* replacee;                                                                       \
    } _interpose_##replacee __attribute__((section("__DATA,__interpose"))) = {                      \
        reinterpret_cast<const void*>(&replacement), reinterpret_cast<const void*>(&replacee)};

namespace {
std::atomic<unsigned long> g_count{0};
}

extern "C" {

__attribute__((visibility("default"))) void alloc_counter_reset() { g_count.store(0); }
__attribute__((visibility("default"))) unsigned long alloc_counter_get() { return g_count.load(); }

static void* counting_malloc(size_t n) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    return malloc(n);
}
static void* counting_calloc(size_t n, size_t s) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    return calloc(n, s);
}
static void* counting_realloc(void* p, size_t n) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    return realloc(p, n);
}
static int counting_posix_memalign(void** p, size_t a, size_t n) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    return posix_memalign(p, a, n);
}
static void* counting_aligned_alloc(size_t a, size_t n) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    return aligned_alloc(a, n);
}
static void* counting_valloc(size_t n) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    return valloc(n);
}

}  // extern "C"

DYLD_INTERPOSE(counting_malloc, malloc)
DYLD_INTERPOSE(counting_calloc, calloc)
DYLD_INTERPOSE(counting_realloc, realloc)
DYLD_INTERPOSE(counting_posix_memalign, posix_memalign)
DYLD_INTERPOSE(counting_aligned_alloc, aligned_alloc)
DYLD_INTERPOSE(counting_valloc, valloc)

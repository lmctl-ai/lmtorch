// TEMPORARY diagnostic: stand-ins for the two mimalloc entry points c10 uses, so the cached libtorch links without
// mimalloc-static.lib (whose TLS-callback init faults before the static CRT heap exists).
#include <cstddef>
#include <malloc.h>
extern "C" void* mi_malloc_aligned(std::size_t size, std::size_t alignment) { return _aligned_malloc(size, alignment); }
extern "C" void mi_free(void* p) { _aligned_free(p); }

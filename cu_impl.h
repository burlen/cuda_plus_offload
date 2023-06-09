#include "cump_config.h"
#include <cstddef>

template <typename T>
T *alloc_cu(size_t n) CUMP_EXPORT;

template <typename T>
void init_cu(T *ptr, size_t n, const T &val) CUMP_EXPORT;

template <typename T>
void fetch_cu(T *dest, T *src, size_t n) CUMP_EXPORT;

#if defined(DEFINE_CU)
#include "cu_impl.hxx"

#define INSTANTIATE_CU(cpp_t)                                           \
template cpp_t *alloc_cu<cpp_t>(size_t n);                              \
template void init_cu<cpp_t>(cpp_t *ptr, size_t n, const cpp_t &val);   \
template void fetch_cu<cpp_t>(cpp_t *dest, cpp_t *src, size_t n);

#endif

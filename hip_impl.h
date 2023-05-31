#include <cstddef>

template <typename T>
T *alloc_hip(size_t n);

template <typename T>
void init_hip(T *ptr, size_t n, const T &val);

template <typename T>
void fetch_hip(T *dest, T *src, size_t n);

#if defined(DEFINE_HIP)
#include "hip_impl.hxx"

#define INSTANTIATE_HIP(cpp_t)                                          \
template cpp_t *alloc_hip<cpp_t>(size_t n);                              \
template void init_hip<cpp_t>(cpp_t *ptr, size_t n, const cpp_t &val);   \
template void fetch_hip<cpp_t>(cpp_t *dest, cpp_t *src, size_t n);

#endif

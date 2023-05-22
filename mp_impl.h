#include <cstddef>

template <typename T>
T *alloc_omp(size_t n);

template <typename T>
void init_omp(T *ptr, size_t n, const T &val);

template <typename T>
void fetch_omp(T *dest, T *src, size_t n);

template <typename T>
void free_omp(T *ptr);

#if defined(DEFINE_MP)

#include "mp_impl.hxx"

#define INSTANTIATE_MP(cpp_t) \
template cpp_t *alloc_omp<cpp_t>(size_t n); \
template void init_omp<cpp_t>(cpp_t *ptr, size_t n, const cpp_t &val); \
template void fetch_omp<cpp_t>(cpp_t *dest, cpp_t *src, size_t n); \
template void free_omp<cpp_t>(cpp_t *ptr);

#endif

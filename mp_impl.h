#include <omp.h>

template <typename T>
T *alloc_omp(size_t n)
{
    int dev = omp_get_default_device();
    T *ptr = (T*)omp_target_alloc(n*sizeof(T), dev);
    return ptr;
}

template <typename T>
void init_omp(T *ptr, size_t n, const T &val)
{
    #pragma omp target teams distribute parallel for is_device_ptr(ptr) map(to: val)
    for (size_t i = 0; i < n; ++i)
        ptr[i] = val;
}

template <typename T>
void fetch_omp(T *dest, T *src, size_t n)
{
    int hid = omp_get_initial_device();
    int tid = omp_get_default_device();
    omp_target_memcpy(dest, src, n*sizeof(T), 0, 0, hid, tid);
}

template <typename T>
void free_omp(T *ptr)
{
    int dev = omp_get_default_device();
    omp_target_free(ptr, dev);
}

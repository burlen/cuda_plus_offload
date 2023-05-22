#include <iostream>
#include <vector>
#if defined(CUMP_USE_CUDA)
#include "cu_impl.h"
#endif
#if defined(CUMP_USE_HIP)
#include "hip_impl.h"
#endif
#if defined(CUMP_USE_OPENMP)
#include "mp_impl.h"
#endif

template <typename T>
void print(T *ptr, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        std::cout << ptr[i] << ", ";
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    size_t n = 32;
    std::vector<float> vec(n);
    float *devp = nullptr;

#if defined(CUMP_USE_CUDA)
    std::cout << "running w/ cuda" << std::endl;
    devp = alloc_cu<float>(n);
    init_cu(devp, n, 3.14f);
    fetch_cu(vec.data(), devp, 32);
    print(vec.data(), 32);
#endif

#if defined(CUMP_USE_HIP)
    std::cout << "running w/ hip" << std::endl;
    devp = alloc_hip<float>(n);
    init_hip(devp, n, 3.14f);
    fetch_hip(vec.data(), devp, 32);
    print(vec.data(), 32);
#endif

#if defined(CUMP_USE_OPENMP)
    std::cout << "running w/ openmp offload" << std::endl;
    devp = alloc_omp<float>(n);
    init_omp(devp, n, 3.14f);
    fetch_omp(vec.data(), devp, 32);
    print(vec.data(), 32);
#endif

    return 0;
}

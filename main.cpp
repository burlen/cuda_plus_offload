#include <iostream>
#include <vector>
#if defined(CUMP_USE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(CUMP_USE_OPENMP)
#include <omp.h>
#endif

template <typename T>
void print(T *ptr, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        std::cout << ptr[i] << ", ";
    std::cout << std::endl;
}




#if defined(CUMP_USE_OPENMP)
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
    #pragma omp target teams loop is_device_ptr(ptr) map(to: val)
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
#endif




#if defined(CUMP_USE_CUDA)
template <typename T>
T *alloc_cu(size_t n)
{
    T *ptr = nullptr;
    cudaError_t ierr = cudaMalloc(&ptr, n*sizeof(T));
    if (ierr != cudaSuccess)
        std::cerr << "cudaMalloc failed " << cudaGetErrorString(ierr)  << std::endl;
    return ptr;
}

template <typename T>
__global__
void init_cu(T *ptr, size_t n, T val)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptr[i] = val;
    //printf("ptr[%ld] = %g\n", i, ptr[i]);
}

template <typename T>
void fetch_cu(T *dest, T *src, size_t n)
{
    cudaError_t ierr = cudaMemcpy(dest, src, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (ierr != cudaSuccess)
        std::cerr << "cudaMemcpy failed " << cudaGetErrorString(ierr) << std::endl;
}
#endif







int main(int argc, char **argv)
{
    size_t n = 32;
    std::vector<float> vec(n);
    float *devp = nullptr;

#if defined(CUMP_USE_CUDA)
    std::cout << "running w/ cuda" << std::endl;
    devp = alloc_cu<float>(n);
    init_cu<<<n/128+1,128>>>(devp, n, 3.14f);
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
        std::cerr << "kernel launch failed " << cudaGetErrorString(ierr) << std::endl;
    cudaDeviceSynchronize();
    fetch_cu(vec.data(), devp, 32);
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

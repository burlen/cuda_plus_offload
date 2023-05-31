#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

template <typename T>
T *alloc_hip(size_t n)
{
    T *ptr = nullptr;
    hipError_t ierr = hipMalloc(&ptr, n*sizeof(T));
    if (ierr != hipSuccess)
        std::cerr << "hipMalloc failed " << hipGetErrorString(ierr)  << std::endl;
    return ptr;
}

template <typename T>
__global__
void init_hip_impl(T *ptr, size_t n, T val)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptr[i] = val;
    //printf("ptr[%ld] = %g\n", i, ptr[i]);
}

template <typename T>
void init_hip(T *ptr, size_t n, const T &val)
{
    init_hip_impl<<<n/128+1,128>>>(ptr, n, val);
    hipError_t ierr = hipGetLastError();
    if (ierr != hipSuccess)
        std::cerr << "kernel launch failed " << hipGetErrorString(ierr) << std::endl;
    hipDeviceSynchronize();
}

template <typename T>
void fetch_hip(T *dest, T *src, size_t n)
{
    hipError_t ierr = hipMemcpy(dest, src, n*sizeof(T), hipMemcpyDeviceToHost);
    if (ierr != hipSuccess)
        std::cerr << "hipMemcpy failed " << hipGetErrorString(ierr) << std::endl;
}  


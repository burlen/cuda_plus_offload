#include <cuda.h>
#include <cuda_runtime.h>

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
void init_cu_impl(T *ptr, size_t n, T val)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptr[i] = val;
    //printf("ptr[%ld] = %g\n", i, ptr[i]);
}

template <typename T>
void init_cu(T *ptr, size_t n, const T &val)
{
    init_cu_impl<<<n/128+1,128>>>(ptr, n, val);
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
        std::cerr << "kernel launch failed " << cudaGetErrorString(ierr) << std::endl;
    cudaDeviceSynchronize();
}

template <typename T>
void fetch_cu(T *dest, T *src, size_t n)
{
    cudaError_t ierr = cudaMemcpy(dest, src, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (ierr != cudaSuccess)
        std::cerr << "cudaMemcpy failed " << cudaGetErrorString(ierr) << std::endl;
}  

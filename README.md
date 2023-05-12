### CUDA (or HIP) and OpenMP device offloading compiled together
Test code to see if it possible to compile both CUDA(or HIP) kernels and OpenMP
device offload in the same program.

### Compiling
Note: the makefile may need to be tweaked according to your system/GPU.

#### AMD
The main file has two sets of templated functions, one using HIP one using
OpenMP offload, to allocate initialize (on the gpu) and then fetch and print
(on the cpu) an array. If -DCUMP_USE_OPENMP is defined the OpenMP code is
enabled. If -DCUMP_USE_HIP is defined the HIP code is enabled. If both are
defined both OpenMP and HIP are enabled.

The makefile will build 3 executables, one for just HIP, one for just OpenMP,
one for both HIP and OpenMP. When both HIP and OpenMP are enabled the compiler
errors out.
```
make -f Makefile.hip
```

#### NVIDIA
The main file has two sets of templated functions, one using CUDA one using
OpenMP offload, to allocate initialize (on the gpu) and then fetch and print
(on the cpu) an array. If -DCUMP_USE_OPENMP is defined the OpenMP code is
enabled. If -DCUMP_USE_CUDA is defined the CUDA code is enabled. If both are
defined both OpenMP and CUDA are enabled.

The makefile will build 3 executables, one for just CUDA, one for just OpenMP,
one for both CUDA and OpenMP. Works if one uses NVIDIA HPC compiler.
```
make -f Makefile.nvhpc
```



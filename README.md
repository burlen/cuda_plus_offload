### CUDA (or HIP) and OpenMP device offloading compiled together
Test code to see if it possible to compile both CUDA(or HIP) kernels and OpenMP
device offload in the same program.

The main file has three sets of templated functions, one using CUDA, one using HIP, one using
OpenMP offload. These 3 functions respectively: allocate space, initialize (on the gpu) to a known value, and then fetch and print
(on the cpu) an array. If -DCUMP_USE_OPENMP is defined the OpenMP code is
enabled. If -DCUMP_USE_HIP or -DCUMP_USE_CUDA is defined the HIP or CUDA code is enabled. If both -DCUMP_USE_OPENMP and -DCUMP_USE_CUDA (exclusive or -DCUMP_USE_HIP) are defined both OpenMP and CUDA (exclusive or HIP) are enabled.

The makefile will build 3 executables, one for just HIP(or CUDA), one for just OpenMP,
one for both HIP(or CUDA) and OpenMP.

#### Status
Mainline clang 17 : unknown
AMD clang : confirmed not to work
NVIDIA : works with NVHPC compilers


### Compiling
Note: the makefile may need to be tweaked according to your system/GPU.

#### main line Clang on NVIDIA device
```
make -f Makefile.clang17.nv
```
Compiler doesn't generate offload code when -x cuda is passed.

#### AMD Clang Compiler
```
make -f Makefile.hip
```
When both HIP and OpenMP are enabled the compiler errors out.

#### NVIDIA HPC compiler
```
make -f Makefile.nvhpc
```
Works if one uses NVIDIA HPC compiler to compile everythin.
Various runtime problems ensue when mixing nvcc and other offload capable compilers.



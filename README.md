### CUDA (or HIP) and OpenMP device offloading compiled together
Test code to see if it possible to compile both CUDA(or HIP) kernels and OpenMP
device offload in the same translation unti.

The main.cpp file has three sets of templated functions, one using CUDA(or HIP), one using
OpenMP offload. Each set defines 3 functions which respectively: allocate space for an array, initialize (on the gpu) to a known value, and then fetch and print
(on the cpu). Pre-processor flags are used to control the build. 
The makefile builds 3 executables, one for just OpenMP, one for just HIP(or CUDA),
one for both HIP(or CUDA) and OpenMP.

| pre-processor flag combo | result |
| ------------------------ | ------ |
| -DCUMP_USE_OPENMP  | OpenMP offload only. |
| -DCUMP_USE_CUDA | CUDA only |
| -DCUMP_USE_HIP | HIP only |
| -DCUMP_USE_OPENMP -DCUMP_USE_CUDA | OpenMP offload and CUDA together |
| -DCUMP_USE_OPENMP -DCUMP_USE_HIP | OpenMP offload and HIP together. |

### Compiling
Note: the makefile may need to be tweaked according to your system/GPU as it's currently hard wired for my systems/GPUs.

#### main line Clang on NVIDIA device
```
make -f Makefile.clang17.nv
```
Compiler doesn't generate offload code when -x cuda is passed.

#### AMD Clang 15 Compiler
```
make -f Makefile.hip
```
When both HIP and OpenMP are enabled the compiler errors out. Confirmed not to work by AMD.

#### NVIDIA HPC compiler
```
make -f Makefile.nvhpc
```
Works if one uses NVIDIA HPC compiler to compile everything.
Various runtime problems ensue when mixing nvcc and other offload capable compilers.

### Status
Mainline clang 17 : unknown<br>
AMD clang : confirmed not to work by AMD<br>
NVIDIA : works with NVHPC compiler is used to compile both CUDA and OpenMP offload.<br>

### CUDA (or HIP) and OpenMP device offloading compiled together
Test code to see if it possible to compile both CUDA(or HIP) kernels and OpenMP
device offload in the same translation unit.

The main.cpp file has three sets of templated functions, one using CUDA(or HIP), one using
OpenMP offload. Each set defines 3 functions which respectively: allocate space for an array, initialize (on the gpu) to a known value, and then fetch and print
(on the cpu). Pre-processor flags are used to control the build. 
The makefile builds 3 executables, one for just OpenMP, one for just HIP(or CUDA),
one for both HIP(or CUDA) and OpenMP.

| pre-processor flag combo | result |
| ------------------------ | ------ |
| `-DCUMP_USE_OPENMP`  | OpenMP offload only. |
| `-DCUMP_USE_CUDA` | CUDA only |
| `-DCUMP_USE_HIP` | HIP only |
| `-DCUMP_USE_OPENMP` `-DCUMP_USE_CUDA` | OpenMP offload and CUDA together |
| `-DCUMP_USE_OPENMP` `-DCUMP_USE_HIP` | OpenMP offload and HIP together. |

### Compiling
Note: the makefile may need to be tweaked according to your system/GPU as it's currently hard wired for my systems/GPUs.

#### main line Clang on NVIDIA device
```
make -f Makefile.clang17.nv
```


#### AMD Clang 15 Compiler
```
make -f Makefile.hip
```

#### NVIDIA HPC compiler
```
make -f Makefile.nvhpc
```

### Status
Mainline clang 17 : works if the OpenMP code and HIP/CUDA code are factored into separate translation units<br>
AMD clang 15 :  works if the OpenMP code and HIP/CUDA code are factored into separate translation units<br>
NVIDIA : works with NVHPC compiler is used to compile both CUDA and OpenMP offload.<br>

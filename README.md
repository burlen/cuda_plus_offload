### Using CUDA and OpenMP device offloading
Test code to see if it possible to compile both CUDA kernells and OpenMP device offload in the same program.

The main file has two sets of tyemplated functions, one using CUDA one using OpenMP offload, to allocate initialize (on the gpu) and then fetch and print (on the cpu) an array. If -DCUMP_USE_OPENMP is defined the OpenMP code is enabled. If -DCUMP_USE_CUDA is defined the CUDA code is enabled. If both are defined both OpenMP and CUDA are enabled.

### Compiling
The makefile will build 3 executables, one for just CUDA, one for just OpenMP, one for both CUDA and OpenMP. When both CUDA and OpeMP are enabled the link step fails.
```
make -f Makefile.nvhpc
```



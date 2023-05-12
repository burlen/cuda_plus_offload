
all: cump_mp cump_hip cump_both_hipcc


cump_mp: main.cpp
	/opt/rocm/llvm/bin/clang++ -Wall -Wextra -g -DCUMP_USE_OPENMP -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030 main.cpp -o cump_mp

cump_hip: main.cpp
	hipcc -Wall -Wextra -g -DCUMP_USE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -isystem /opt/rocm-5.4.3/hip/include -x hip --offload-arch=gfx1030 main.cpp -o cump_hip

cump_both_hipcc: main.cpp
	hipcc -Wall -Wextra -g -DCUMP_USE_HIP -DCUMP_USE_OPENMP -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -isystem /opt/rocm-5.4.3/hip/include -x hip --offload-arch=gfx1030 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030 main.cpp -o cump_both_hipcc


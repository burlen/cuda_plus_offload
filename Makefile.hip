
all: cump_mp cump_hip cump_both

clean:
	rm -f cump_mp cump_hip cump_both *.o *.a


cump_mp: main.cpp
	/opt/rocm/llvm/bin/clang++ -Wall -Wextra -g -DCUMP_USE_OPENMP -DDEFINE_MP -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030 main.cpp -o cump_mp

cump_hip: main.cpp
	hipcc -Wall -Wextra -g -DCUMP_USE_HIP -DDEFINE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -isystem /opt/rocm-5.4.3/hip/include -x hip --offload-arch=gfx1030 main.cpp -o cump_hip

hip_impl.o: hip_impl.cpp
	/opt/rocm/llvm/bin/clang++ -x hip --cuda-gpu-arch=gfx1030 -c hip_impl.cpp -o hip_impl.o
	ar -rcs libhip_impl.a hip_impl.o

mp_impl.o: mp_impl.cpp
	/opt/rocm/llvm/bin/clang++ -fopenmp --offload-arch=gfx1030 -fopenmp-offload-mandatory -c mp_impl.cpp -o mp_impl.o
	ar -rcs libmp_impl.a mp_impl.o

cump_both: main.cpp hip_impl.o mp_impl.o
	/opt/rocm/llvm/bin/clang++  -Wall -Wextra -g -DCUMP_USE_HIP -DCUMP_USE_OPENMP  main.cpp -o cump_both -L`pwd` -lmp_impl -lhip_impl -fopenmp --offload-arch=gfx1030 -L/opt/rocm-5.4.3/hip/lib/ -lamdhip64 -L /opt/rocm-5.4.3/llvm/lib -lomptarget

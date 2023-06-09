
# OpenMP only exec
add_executable(cump_mp ../main.cpp)
target_link_libraries(cump_mp omp omptarget)
target_compile_definitions(cump_mp PUBLIC -DCUMP_USE_OPENMP -DDEFINE_MP)
target_compile_options(cump_mp PUBLIC -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)
target_link_options(cump_mp PUBLIC -L/home/bloring/work/llvm/llvm-install/lib/ -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)

# OpenMP only lib
add_library(mp_impl ../mp_impl.cpp)
target_compile_options(mp_impl PUBLIC -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)
target_link_options(mp_impl PUBLIC -L/home/bloring/work/llvm/llvm-install/lib/ -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)

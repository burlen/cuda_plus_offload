
# library with both OpenMP and CUDA
set_source_files_properties(../cu_impl.cpp PROPERTIES LANGUAGE CUDA)
add_library(cump_impl ../mp_impl.cpp ../cu_impl.cpp)
target_compile_options(cump_impl PUBLIC -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)
set_target_properties(cump_impl PROPERTIES CUDA_ARCHITECTURES "75")


add_executable(cump_both ../main.cpp)
target_link_libraries(cump_both mp_impl cu_impl omp omptarget)
target_compile_definitions(cump_both PUBLIC -DCUMP_USE_OPENMP -DCUMP_USE_CUDA)
target_compile_options(cump_both PUBLIC -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)
target_link_options(cump_both  PUBLIC -L/home/bloring/work/llvm/llvm-install/lib/ -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)

add_executable(cump_both_2 ../main.cpp)
target_link_libraries(cump_both_2 cump_impl omp omptarget)
target_compile_definitions(cump_both_2 PUBLIC -DCUMP_USE_OPENMP -DCUMP_USE_CUDA)
target_compile_options(cump_both_2 PUBLIC -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)
target_link_options(cump_both_2  PUBLIC -L/home/bloring/work/llvm/llvm-install/lib/ -fopenmp --offload-arch=sm_75 -fopenmp-offload-mandatory --offload-new-driver)


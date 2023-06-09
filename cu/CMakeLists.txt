# cuda only exec
set_source_files_properties(../main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cump_cu ../main.cpp)
target_compile_definitions(cump_cu PUBLIC -DCUMP_USE_CUDA -DDEFINE_CU)
set_target_properties(cump_cu PROPERTIES CUDA_ARCHITECTURES "75")

# cuda only library
set_source_files_properties(../cu_impl.cpp PROPERTIES LANGUAGE CUDA)
add_library(cu_impl ../cu_impl.cpp)
set_target_properties(cu_impl PROPERTIES CUDA_ARCHITECTURES "75")

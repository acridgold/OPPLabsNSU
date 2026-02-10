cmake_minimum_required(VERSION 3.10)
project(MPI_Project)

find_package(MPI)
add_executable(mpi_app src/main.cpp)
target_link_libraries(mpi_app
        PRIVATE
        MPI::MPI_C
        MPI::MPI_CXX
)
target_compile_options(mpi_app PRIVATE ${MPI_C_COMPILE_FLAGS})
target_link_libraries(mpi_app PRIVATE ${MPI_C_LINK_FLAGS})

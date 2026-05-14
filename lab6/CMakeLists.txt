cmake_minimum_required(VERSION 3.10)
project(lab6)
set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

add_executable(lb src/mpi.cpp)

target_include_directories(lb PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(lb PRIVATE ${MPI_CXX_LIBRARIES} Threads::Threads m)
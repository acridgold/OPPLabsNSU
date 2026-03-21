#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

constexpr int N = 1000;
constexpr double EPS = 1e-5;
constexpr int MAX_ITER = 10000;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_start = MPI_Wtime();

    int local_N = N / size;
    int local_start = rank * local_N;

    // Чтение локальной части матрицы
    std::vector<double> A_local(local_N * N);
    MPI_File fh_a;
    MPI_File_open(MPI_COMM_WORLD, "matrix_a.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_a);
    MPI_File_read_at(fh_a, local_start * N * sizeof(double), A_local.data(), local_N * N, MPI_DOUBLE,
                     MPI_STATUS_IGNORE);
    MPI_File_close(&fh_a);

    // Чтение всего вектора B (он нужен всем для расчета r_local)
    std::vector<double> b(N);
    MPI_File fh_b;
    MPI_File_open(MPI_COMM_WORLD, "vector_b.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_b);
    MPI_File_read_all(fh_b, 0, b.data(), N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh_b);

    std::vector<double> x(N, 0.0);
    std::vector<double> r_local(local_N), Ar_local(local_N), r_global(N);

    double b_norm2 = 0.0;
    for (double val : b) b_norm2 += val * val;
    double r0_norm = std::sqrt(b_norm2);

    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        // r_local = b_local - (Ax)_local
        for (int i = 0; i < local_N; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) sum += A_local[i * N + j] * x[j];
            r_local[i] = b[local_start + i] - sum;
        }

        // Собираем полную невязку для следующего шага
        MPI_Allgather(r_local.data(), local_N, MPI_DOUBLE, r_global.data(), local_N, MPI_DOUBLE, MPI_COMM_WORLD);

        double loc_r2 = 0.0;
        for (double val : r_local) loc_r2 += val * val;
        double glob_r2;
        MPI_Allreduce(&loc_r2, &glob_r2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // Собираем для проверки

        if (std::sqrt(glob_r2) / r0_norm < EPS) break; // Критерий остоновки

        // Считаем tau
        for (int i = 0; i < local_N; ++i)
        {
            Ar_local[i] = 0.0;
            for (int j = 0; j < N; ++j) Ar_local[i] += A_local[i * N + j] * r_global[j];
        }

        double loc_rAr = 0.0, loc_ArAr = 0.0;
        for (int i = 0; i < local_N; ++i)
        {
            loc_rAr += r_local[i] * Ar_local[i];
            loc_ArAr += Ar_local[i] * Ar_local[i];
        }

        double g_rAr, g_ArAr;
        MPI_Allreduce(&loc_rAr, &g_rAr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&loc_ArAr, &g_ArAr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double tau = g_rAr / g_ArAr;
        for (int j = 0; j < N; ++j) x[j] += tau * r_global[j];
    }

    if (rank == 0)
    {
        std::cout << "MPI Time: " << MPI_Wtime() - t_start << "s\n"
            << "x[0]: " << x[0] << "\n";
    }

    MPI_Finalize();
    return 0;
}

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>

constexpr int N = 6960;
constexpr double EPS = 1e-12;
constexpr int MAX_ITER = 10000;

int main()
{
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<double> A(N * N);
    std::vector<double> b(N);

    // Загрузка матрицы
    std::ifstream fa("matrix_a.bin", std::ios::binary);
    if (!fa)
    {
        std::cerr << "Error opening matrix_a.bin\n";
        return 1;
    }
    fa.read(reinterpret_cast<char*>(A.data()), N * N * sizeof(double));
    fa.close();

    // Загрузка вектора
    std::ifstream fb("vector_b.bin", std::ios::binary);
    if (!fb)
    {
        std::cerr << "Error opening vector_b.bin\n";
        return 1;
    }
    fb.read(reinterpret_cast<char*>(b.data()), N * sizeof(double));
    fb.close();

    std::vector<double> x(N, 0.0);
    std::vector<double> r(N), Ar(N);

    double r0_norm = 0.0;
    for (double val : b) r0_norm += val * val;
    r0_norm = std::sqrt(r0_norm);

    int converged_iter = -1;
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        // r = b - Ax
        for (int i = 0; i < N; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) sum += A[i * N + j] * x[j];
            r[i] = b[i] - sum;
        }

        double r_norm2 = 0.0;
        for (double val : r) r_norm2 += val * val;
        double r_norm = std::sqrt(r_norm2);

        if (r_norm / r0_norm < EPS)
        {
            converged_iter = iter;
            break;
        }

        // tau = (r, Ar) / (Ar, Ar)
        for (int i = 0; i < N; ++i)
        {
            Ar[i] = 0.0;
            for (int j = 0; j < N; ++j) Ar[i] += A[i * N + j] * r[j];
        }

        double rAr = 0.0, ArAr = 0.0;
        for (int i = 0; i < N; ++i)
        {
            rAr += r[i] * Ar[i];
            ArAr += Ar[i] * Ar[i];
        }

        double tau = rAr / ArAr;
        for (int i = 0; i < N; ++i) x[i] += tau * r[i];
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
    std::cout << "Converged: " << converged_iter << ", x[0]: " << x[0] << "\n";

    return 0;
}

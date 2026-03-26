#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <omp.h>

constexpr int N = 2400;
constexpr double EPS = 1e-12;
constexpr int MAX_ITER = 15000;

int main() {
    double t_start = omp_get_wtime();

    std::vector<double> A(N * N);
    std::vector<double> b(N);
    std::vector<double> x(N, 0.0);
    std::vector<double> r(N), Ar(N);

    // Загрузка данных
    std::ifstream fa("matrix.bin", std::ios::binary);
    if (!fa) return 1;
    fa.read((char*)A.data(), N * N * sizeof(double));
    std::ifstream fb("vector_b.bin", std::ios::binary);
    if (!fb) return 1;
    fb.read((char*)b.data(), N * sizeof(double));

    double b_norm2 = 0.0;
    for (double val : b) b_norm2 += val * val;
    double r0_norm = std::sqrt(b_norm2);

    int converged_iter = -1;
    double r_norm2_shared = 0.0;
    double ArAr_shared = 0.0;
    double rAr_shared = 0.0;
    double tau = 0.0;
    bool stop_flag = false;

#pragma omp parallel
    {

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            // 1. r = b - Ax
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                double Ax_i = 0.0;
                for (int j = 0; j < N; ++j) {
                    Ax_i += A[i * N + j] * x[j];
                }
                r[i] = b[i] - Ax_i;
            }

            // 2. Норма невязки
            #pragma omp single
            r_norm2_shared = 0.0;

            #pragma omp for reduction(+:r_norm2_shared) schedule(static)
            for (int i = 0; i < N; ++i) {
                r_norm2_shared += r[i] * r[i];
            }

            #pragma omp single
            {
                if (std::sqrt(r_norm2_shared) / r0_norm < EPS) {
                    converged_iter = iter;
                    stop_flag = true;
                }
            }
            #pragma omp barrier
            if (stop_flag) break;

            // 3. Ar = A * r
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                double Ar_i = 0.0;
                for (int j = 0; j < N; ++j) {
                    Ar_i += A[i * N + j] * r[j];
                }
                Ar[i] = Ar_i;
            }

            // 4. Вычисление tau
            #pragma omp single
            { rAr_shared = 0.0; ArAr_shared = 0.0; }

            #pragma omp for reduction(+:rAr_shared, ArAr_shared) schedule(static)
            for (int i = 0; i < N; ++i) {
                rAr_shared += r[i] * Ar[i];
                ArAr_shared += Ar[i] * Ar[i];
            }

            #pragma omp single
            tau = rAr_shared / ArAr_shared;

            #pragma omp barrier

            // 5. x = x + tau * r
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                x[i] += tau * r[i];
            }
        }
    }

    double t_end = omp_get_wtime();
    double final_residual = std::sqrt(r_norm2_shared) / r0_norm;

    std::cout << "========================================" << std::endl;
    std::cout << "OpenMP Statistics (No Manual AVX):" << std::endl;
    std::cout << "  Time taken:     " << std::fixed << std::setprecision(4) << (t_end - t_start) << " s" << std::endl;
    std::cout << "  Threads:        " << omp_get_max_threads() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Algorithm Results:" << std::endl;
    if (converged_iter != -1) {
        std::cout << "  Status:         CONVERGED" << std::endl;
        std::cout << "  Iterations:     " << converged_iter << std::endl;
    } else {
        std::cout << "  Status:         DID NOT CONVERGE" << std::endl;
        std::cout << "  Iterations:     " << MAX_ITER << std::endl;
    }
    std::cout << "  Final Residual: " << std::scientific << final_residual << std::endl;
    std::cout << "  x[0]:           " << std::fixed << std::setprecision(6) << x[0] << std::endl;
    std::cout << "  x[N-1]:         " << std::fixed << std::setprecision(6) << x[N-1] << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
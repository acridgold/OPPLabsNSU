#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

int main() {
    int n = 2400;
    string mat_file = "matrix.bin";
    string vec_file = "vector_b.bin";

    mt19937 gen(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    uniform_real_distribution<double> dist_rhs(-1.0, 1.0);

    vector<double> R(n * n);
    vector<double> A(n * n, 0.0);
    vector<double> b(n);

    cout << "Генерация системы " << n << "x" << n << " (улучшенная версия)..." << endl;

    // 1. Генерируем вспомогательную матрицу R
    for (auto &v : R) v = dist(gen);

    // 2. A = R^T * R + 0.5 * I
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                A[i * n + j] += R[k * n + i] * R[k * n + j];
            }
        }
        A[i * n + i] += 10.0;          // eps = 0.5
    }

    // 3. b
    for (auto &v : b) v = dist_rhs(gen);

    ofstream f_mat(mat_file, ios::binary);
    f_mat.write(reinterpret_cast<char*>(A.data()), A.size() * sizeof(double));
    f_mat.close();

    ofstream f_vec(vec_file, ios::binary);
    f_vec.write(reinterpret_cast<char*>(b.data()), b.size() * sizeof(double));
    f_vec.close();

    cout << "Файлы готовы: " << mat_file << ", " << vec_file << endl;
    return 0;
}
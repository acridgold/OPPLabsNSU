#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;

// Функция вычисления частичной суммы
int calculate_s_partial(int N, const vector<int>& local_a, const vector<int>& b)
{
    int s = 0;
    for (size_t i = 0; i < local_a.size(); i++)
    {
        for (int j = 0; j < N; j++)
        {
            s += local_a[i] * b[j];
        }
    }
    return s;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    vector<int> b;
    vector<int> local_a;

    // Процесс 0 читает данные
    if (rank == 0)
    {
        ifstream fin("../../input.txt");
        if (!fin.is_open())
        {
            cerr << "Error opening input.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        fin >> N;
        if (fin.fail() || N <= 0)
        {
            cerr << "Error reading N or invalid N" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        vector<int> full_a(N);
        b.resize(N);

        for (int i = 0; i < N; i++)
        {
            fin >> full_a[i];
        }

        for (int i = 0; i < N; i++)
        {
            fin >> b[i];
        }

        fin.close();

        cout << "N = " << N << endl;
        cout << "Number of MPI processes: " << size << endl;

        // Делим вектор
        int base_chunk = N / size;
        int remainder = N % size;

        int start_0 = 0 * base_chunk + min(0, remainder);
        int end_0 = start_0 + base_chunk + (0 < remainder ? 1 : 0);
        int local_size_0 = end_0 - start_0;

        local_a.resize(local_size_0);
        copy(full_a.begin() + start_0, full_a.begin() + end_0, local_a.begin());

        // Рассылаем N всем процессам
        for (int dest = 1; dest < size; dest++)
        {
            MPI_Send(&N, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }

        // Рассылаем вектор b ЦЕЛИКОМ всем процессам
        for (int dest = 1; dest < size; dest++)
        {
            MPI_Send(b.data(), N, MPI_INT, dest, 1, MPI_COMM_WORLD);
        }

        // Рассылаем ЧАСТИ вектора a каждому процессу
        for (int dest = 1; dest < size; dest++)
        {
            int dest_start = dest * base_chunk + min(dest, remainder);
            int dest_end = dest_start + base_chunk + (dest < remainder ? 1 : 0);
            int dest_size = dest_end - dest_start;

            // Отправляем размер части
            MPI_Send(&dest_size, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);

            // Отправляем часть вектора
            MPI_Send(&full_a[dest_start], dest_size, MPI_INT, dest, 3, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Получаем N
        MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Получаем вектор b
        b.resize(N);
        MPI_Recv(b.data(), N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Получаем размер своей части вектора a
        int local_size;
        MPI_Recv(&local_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_a.resize(local_size);

        // Получаем свою часть вектора a
        MPI_Recv(local_a.data(), local_size, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Синхронизируем
    MPI_Barrier(MPI_COMM_WORLD);

    // Начало замера времени
    chrono::high_resolution_clock::time_point start_time, end_time;
    if (rank == 0)
    {
        start_time = chrono::high_resolution_clock::now();
    }

    int local_s = calculate_s_partial(N, local_a, b);

    int global_s = 0;

    if (rank == 0)
    {
        global_s = local_s;

        for (int i = 1; i < size; i++)
        {
            int received_s;
            MPI_Recv(&received_s, 1, MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_s += received_s;
        }
    }
    else
    {
        MPI_Send(&local_s, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> total_duration = end_time - start_time;

        cout << "\n=== RESULTS ===" << endl;
        cout << "Total sum s = " << global_s << endl;
        cout << "Total execution time: " << total_duration.count() << " seconds" << endl;
        cout << "=================" << endl;
    }

    MPI_Finalize();
    return 0;
}

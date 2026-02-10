#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;

int calculate_partial_sum(int N, const vector<int>& local_a, const vector<int>& b)
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
    vector<int> full_a;
    vector<int> b;
    vector<int> local_a;

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

        cout << "N = " << N << endl;
        cout << "Number of processes: " << size << endl;

        full_a.resize(N);
        b.resize(N);

        for (int i = 0; i < N; i++) fin >> full_a[i];
        for (int i = 0; i < N; i++) fin >> b[i];

        fin.close();
    }

    // Распределяем по процессам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        b.resize(N);
    }
    MPI_Bcast(b.data(), N, MPI_INT, 0, MPI_COMM_WORLD); // b-вектор целиком (требование)

    // MPI_Scatter требует одинаковое количество элементов для каждого процесса
    // Если N не делится нацело на size, то MPI_Scatterv

    int base_chunk = N / size;
    int remainder = N % size;

    int local_size = base_chunk + (rank < remainder ? 1 : 0);
    local_a.resize(local_size);

    // MPI_Scatterv требует counts и displs
    vector<int> counts(size); // массив количеств элементов, посылаемых процессам
    vector<int> displs(size);
    // массив смещений, i-ое значение определяет смещение i-го блока данных относительно начала sendbuf

    int current_displ = 0;
    for (int i = 0; i < size; i++)
    {
        counts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i] = current_displ;
        current_displ += counts[i];
    }

    if (rank != 0)
    {
        full_a.resize(0);
    }

    MPI_Scatterv(
        rank == 0 ? full_a.data() : nullptr, // Отправной буфер
        counts.data(), // Сколько элементов каждому процессу
        displs.data(), // Смещения для каждого процесса
        MPI_INT,
        local_a.data(), // Приемный буфер
        local_size, // Сколько элементов получаем
        MPI_INT,
        0, // root
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    chrono::high_resolution_clock::time_point start_time, end_time;

    if (rank == 0)
    {
        start_time = chrono::high_resolution_clock::now();
    }

    int local_s = calculate_partial_sum(N, local_a, b);

    // Собираем по потокам
    int global_s = 0;
    MPI_Reduce(&local_s, &global_s, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

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

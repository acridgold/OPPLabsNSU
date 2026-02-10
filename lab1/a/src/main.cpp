//#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

int calculate_s(int N, vector<int> a, vector<int> b)
{
    int i = 0, j = 0, s = 0;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            s += a[i] * b[j];
        }
    }
    return s;
}

int main(int argc, char** argv)
{
    //MPI_Init(&argc, &argv);
    // input file: сперва N, затем N строк вектора a, затем N строк вектора b

    int N;
    ifstream fin("../../input.txt");
    fin >> N;

    vector<int> a(N), b(N);
    cout << "a = ";
    for (int i = 0; i < N; i++)
    {
        fin >> a[i];
        cout << a[i] << " ";
    }
    cout << endl;

    cout << "b = ";
    for (int i = 0; i < N; i++)
    {
        fin >> b[i];
        cout << b[i] << " ";
    }
    cout << endl;

    fin.close();

    {
        int s = 0;
        using namespace chrono;
        auto start = high_resolution_clock::now();

        s = calculate_s(N, a, b);

        auto end = high_resolution_clock::now();

        duration<double> duration = end - start;

        cout << "Result s = " << s << endl;
        cout << "Duration: " << duration.count() << " seconds" << endl;
    }

    //MPI_Finalize();
    return 0;
}

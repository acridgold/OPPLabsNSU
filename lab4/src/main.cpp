#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>

static double* alloc_zero(int n) { return new double[n](); }

// C += A*B,  A:[ra×k], B:[k×cb], C:[ra×cb]
static void matmul(const double* A, const double* B, double* C,
                   int ra, int k, int cb)
{
    for (int i = 0; i < ra; ++i)
        for (int l = 0; l < k; ++l)
        {
            double a = A[i * k + l];
            for (int j = 0; j < cb; ++j)
                C[i * cb + j] += a * B[l * cb + j];
        }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t0 = MPI_Wtime();
    int p1, p2;
    if (argc >= 3)
    {
        p1 = atoi(argv[1]);
        p2 = atoi(argv[2]);
    }
    else
    {
        p1 = (int)sqrt(size);
        while (size % p1) --p1;
        p2 = size / p1;
    }
    if (p1 * p2 != size)
    {
        if (!rank) fprintf(stderr, "Ошибка: p1*p2 != size (%d*%d != %d)\n", p1, p2, size);
        MPI_Finalize();
        return 1;
    }

    /* ── размеры матриц ──────────────────────────────────────────────── */
    int n1, n2, n3;
    if (argc >= 6)
    {
        n1 = atoi(argv[3]);
        n2 = atoi(argv[4]);
        n3 = atoi(argv[5]);
    }
    else
    {
        n1 = n2 = n3 = 2400;
    }

    if (n1 % p1 != 0 || n3 % p2 != 0)
    {
        if (!rank)
            fprintf(stderr, "Ошибка: n1(%d) %% p1(%d) = %d, n3(%d) %% p2(%d) = %d\n",
                    n1, p1, n1 % p1, n3, p2, n3 % p2);
        MPI_Finalize();
        return 1;
    }

    const int rows_a = n1 / p1; // строк A у каждого процесса
    const int cols_b = n3 / p2; // столбцов B у каждого процесса

    if (!rank)
        printf("Grid %dx%d | A[%dx%d] B[%dx%d] C[%dx%d] | rows_a=%d cols_b=%d\n",
               p1, p2, n1, n2, n2, n3, n1, n3, rows_a, cols_b);

    /* ── декартова топология ─────────────────────────────────────────── */
    int dims[2] = {p1, p2};
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    const int my_row = coords[0], my_col = coords[1];

    int rem_row[2] = {0, 1}; // все процессы одной строки решетки
    MPI_Comm row_comm;
    MPI_Cart_sub(cart, rem_row, &row_comm);

    int rem_col[2] = {1, 0}; // все процессы одного столбца решетки
    MPI_Comm col_comm;
    MPI_Cart_sub(cart, rem_col, &col_comm);

    /* ── буферы ──────────────────────────────────────────────────────── */
    double *A_full = nullptr, *B_full = nullptr, *C_full = nullptr;
    double* la = alloc_zero(rows_a * n2); // локальная полоса A
    double* lb = alloc_zero(n2 * cols_b); // локальная полоса B
    double* lc = alloc_zero(rows_a * cols_b); // локальный минор C

    /* ── генерация матриц на процессе (0,0) ──────────────────────────── */
    if (!rank)
    {
        A_full = alloc_zero(n1 * n2);
        B_full = alloc_zero(n2 * n3);
        C_full = alloc_zero(n1 * n3);
        srand(42);
        for (int i = 0; i < n1 * n2; ++i) A_full[i] = (double)(rand() % 10);
        for (int i = 0; i < n2 * n3; ++i) B_full[i] = (double)(rand() % 10);
    }

    /* ====================
     * Scatter горизонтальных полос A по col_comm
     * ==================== */
    if (my_col == 0)
        MPI_Scatter(A_full, rows_a * n2, MPI_DOUBLE,
                    la, rows_a * n2, MPI_DOUBLE,
                    0, col_comm);

    /* ====================
     * Раздача вертикальных полос B по строке 0 (Точка-Точка)
     * ==================== */
    if (my_row == 0)
    {
        if (my_col == 0)
        {
            for (int r = 0; r < n2; ++r)
                memcpy(lb + r * cols_b, B_full + r * n3, cols_b * sizeof(double));

            for (int j = 1; j < p2; ++j)
            {
                int dest;
                int dc[2] = {0, j};
                MPI_Cart_rank(cart, dc, &dest);

                MPI_Datatype vtype; // вертикальная полоса j
                MPI_Type_vector(n2, cols_b, n3, MPI_DOUBLE, &vtype);
                MPI_Type_commit(&vtype);
                MPI_Send(B_full + j * cols_b, 1, vtype, dest, 100, cart);
                MPI_Type_free(&vtype);
            }
        }
        else
        {
            int src;
            int sc[2] = {0, 0};
            MPI_Cart_rank(cart, sc, &src);
            MPI_Recv(lb, n2 * cols_b, MPI_DOUBLE, src, 100, cart, MPI_STATUS_IGNORE);
        }
    }

    /* ====================
     * Рассылка полос A по строкам
     * ==================== */
    MPI_Bcast(la, rows_a * n2, MPI_DOUBLE, 0, row_comm);

    /* ====================
     * Рассылка полос B по столбцам
     * ==================== */
    MPI_Bcast(lb, n2 * cols_b, MPI_DOUBLE, 0, col_comm);

    /* ====================
     * Локально умножаем
     * ==================== */
    MPI_Barrier(cart);
    double t_1 = MPI_Wtime();
    matmul(la, lb, lc, rows_a, n2, cols_b);
    double t_compute = MPI_Wtime() - t_1;

    /* ====================
     * Сбор миноров C на (0,0)
     * ==================== */
    if (rank == 0)
    {
        for (int r = 0; r < rows_a; ++r)
            memcpy(C_full + r * n3, lc + r * cols_b, cols_b * sizeof(double));

        for (int i = 0; i < p1; ++i)
        {
            for (int j = 0; j < p2; ++j)
            {
                if (i == 0 && j == 0) continue;
                int src;
                int sc[2] = {i, j};
                MPI_Cart_rank(cart, sc, &src);

                int arr_sz[2] = {n1, n3};
                int sub_sz[2] = {rows_a, cols_b};
                int starts[2] = {i * rows_a, j * cols_b};
                MPI_Datatype sub_t;
                MPI_Type_create_subarray(2, arr_sz, sub_sz, starts,
                                         MPI_ORDER_C, MPI_DOUBLE, &sub_t);
                MPI_Type_commit(&sub_t);
                MPI_Recv(C_full, 1, sub_t, src, 200, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&sub_t);
            }
        }
    }
    else
    {
        int root;
        int rc[2] = {0, 0};
        MPI_Cart_rank(cart, rc, &root);
        MPI_Send(lc, rows_a * cols_b, MPI_DOUBLE, root, 200, cart);
    }

    MPI_Barrier(cart);
    double t_total = MPI_Wtime() - t0;

    /* ── Верификация ─────────────────── */
    if (rank == 0)
    {
        if (n1 <= 300 && n2 <= 300 && n3 <= 300)
        {
            double* C_ref = alloc_zero(n1 * n3);
            matmul(A_full, B_full, C_ref, n1, n2, n3);
            double mx = 0;
            for (int i = 0; i < n1 * n3; ++i) mx = fmax(mx, fabs(C_full[i] - C_ref[i]));
            printf("Верификация: max_err=%.2e  %s\n", mx, mx < 1e-6 ? "OK" : "FAIL");
            delete[] C_ref;
        }
        printf("t_compute = %.4f s\nt_total   = %.4f s\n", t_compute, t_total);
    }

    /* ── Освобождение ────────────────────────────────────────────────── */
    delete[] la;
    delete[] lb;
    delete[] lc;
    if (rank == 0)
    {
        delete[] A_full;
        delete[] B_full;
        delete[] C_full;
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CELL(buf, r, c, Y)  ((buf)[(r)*(Y)+(c)])

using cell_t = unsigned char;

static int rows_for_rank(int total, int np, int r)
{
    return total / np + (r < total % np ? 1 : 0);
}

static int first_row_for_rank(int total, int np, int r)
{
    int base = total / np, extra = total % np;
    return r * base + (r < extra ? r : extra);
}

/* FNV-1a хэш локальной полосы */
static uint64_t hash_strip(const cell_t* buf, int local_rows, int Y)
{
    uint64_t h = 14695981039346656037ULL;
    const cell_t* p = buf;
    int n = local_rows * Y;
    for (int i = 0; i < n; i++)
    {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    double t_start = MPI_Wtime();

    if (argc < 2)
    {
        if (rank == 0) fprintf(stderr, "Usage: %s <input_file> [max_iter]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int X = 0, Y = 0;
    int max_iter = (argc >= 3) ? atoi(argv[2]) : 100000;
    cell_t* global_data = NULL;

    if (rank == 0)
    {
        FILE* f = fopen(argv[1], "r");
        if (!f) { X = -1; }
        else
        {
            char line[100000];
            if (fgets(line, sizeof(line), f))
            {
                Y = (int)strlen(line);
                while (Y > 0 && (line[Y - 1] == '\n' || line[Y - 1] == '\r')) Y--;
                X = 1;
                while (fgets(line, sizeof(line), f)) if (strlen(line) > 1) X++;
            }
            rewind(f);
            global_data = (cell_t*)malloc(X * Y * sizeof(cell_t));
            for (int i = 0; i < X; i++)
                if (fgets(line, sizeof(line), f))
                    for (int j = 0; j < Y; j++)
                        global_data[i * Y + j] = (line[j] == '#') ? 1 : 0;
            fclose(f);
        }
    }

    MPI_Bcast(&X, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (X < 0)
    {
        if (rank == 0) perror("File error");
        if (global_data) free(global_data);
        MPI_Finalize();
        return 1;
    }
    MPI_Bcast(&Y, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int local_rows = rows_for_rank(X, np, rank);
    const int prev_rank = (rank - 1 + np) % np;
    const int next_rank_p = (rank + 1) % np;

    cell_t* const buf0 = (cell_t*)calloc((local_rows + 2) * Y, sizeof(cell_t));
    cell_t* const buf1 = (cell_t*)calloc((local_rows + 2) * Y, sizeof(cell_t));

    /* Раздача данных */
    if (rank == 0)
    {
        for (int r = 0; r < np; r++)
        {
            int r_start = first_row_for_rank(X, np, r);
            int r_count = rows_for_rank(X, np, r);
            if (r == 0)
                memcpy(&CELL(buf0, 1, 0, Y), &global_data[0], r_count * Y);
            else
                MPI_Send(&global_data[r_start * Y], r_count * Y,
                         MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
        }
        free(global_data);
    }
    else
    {
        MPI_Recv(&CELL(buf0, 1, 0, Y), local_rows * Y,
                 MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int hash_cap = 1024;
    int hash_size = 0;
    uint64_t* hash_hist = NULL;
    if (rank == 0)
    {
        hash_hist = (uint64_t*)malloc(hash_cap * sizeof(uint64_t));
    }

    uint64_t local_h = hash_strip(&CELL(buf0, 1, 0, Y), local_rows, Y);
    uint64_t global_h = 0;
    MPI_Reduce(&local_h, &global_h, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        hash_hist[hash_size++] = global_h;
    }

    int flag_send[1];
    int stopped_iter = -1;

    for (int iter = 1; iter <= max_iter; iter++)
    {
        cell_t* cur = (iter % 2 == 1) ? buf0 : buf1;
        cell_t* dst = (iter % 2 == 1) ? buf1 : buf0;

        /* Обмен граничными строками */
        MPI_Request req[4];
        MPI_Isend(&CELL(cur, 1, 0, Y), Y, MPI_UNSIGNED_CHAR,
                  prev_rank, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&CELL(cur, local_rows, 0, Y), Y, MPI_UNSIGNED_CHAR,
                  next_rank_p, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&CELL(cur, 0, 0, Y), Y, MPI_UNSIGNED_CHAR,
                  prev_rank, 1, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&CELL(cur, local_rows+1, 0, Y), Y, MPI_UNSIGNED_CHAR,
                  next_rank_p, 0, MPI_COMM_WORLD, &req[3]);

        /* Внутренние строки */
        for (int r = 2; r <= local_rows - 1; r++)
            for (int c = 0; c < Y; c++)
            {
                int lc = (c - 1 + Y) % Y, rc = (c + 1) % Y;
                int alive =
                    CELL(cur, r-1, lc, Y) + CELL(cur, r-1, c, Y) + CELL(cur, r-1, rc, Y) +
                    CELL(cur, r, lc, Y) + CELL(cur, r, rc, Y) +
                    CELL(cur, r+1, lc, Y) + CELL(cur, r+1, c, Y) + CELL(cur, r+1, rc, Y);
                CELL(dst, r, c, Y) = CELL(cur, r, c, Y) ? (alive == 2 || alive == 3) : (alive == 3);
            }

        /* Верхний сосед -> первая строка */

        for (int c = 0; c < Y; c++)
        {
            int lc = (c - 1 + Y) % Y, rc = (c + 1) % Y;
            int alive =
                CELL(cur, 0, lc, Y) + CELL(cur, 0, c, Y) + CELL(cur, 0, rc, Y) +
                CELL(cur, 1, lc, Y) + CELL(cur, 1, rc, Y) +
                CELL(cur, 2, lc, Y) + CELL(cur, 2, c, Y) + CELL(cur, 2, rc, Y);
            CELL(dst, 1, c, Y) = CELL(cur, 1, c, Y) ? (alive == 2 || alive == 3) : (alive == 3);
        }

        /* Нижний сосед -> последняя строка */
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[2], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);
        {
            int r = local_rows;
            for (int c = 0; c < Y; c++)
            {
                int lc = (c - 1 + Y) % Y, rc = (c + 1) % Y;
                int alive =
                    CELL(cur, r-1, lc, Y) + CELL(cur, r-1, c, Y) + CELL(cur, r-1, rc, Y) +
                    CELL(cur, r, lc, Y) + CELL(cur, r, rc, Y) +
                    CELL(cur, r+1, lc, Y) + CELL(cur, r+1, c, Y) + CELL(cur, r+1, rc, Y);
                CELL(dst, r, c, Y) = CELL(cur, r, c, Y) ? (alive == 2 || alive == 3) : (alive == 3);
            }
        }

        /* Считаем локальный хэш dst, собираем глобальный через Iallreduce */
        local_h = hash_strip(&CELL(dst, 1, 0, Y), local_rows, Y);
        uint64_t global_h_new = 0;
        MPI_Request req_stop;
        MPI_Iallreduce(&local_h, &global_h_new, 1, MPI_UINT64_T,
                       MPI_BXOR, MPI_COMM_WORLD, &req_stop);

        MPI_Wait(&req_stop, MPI_STATUS_IGNORE);

        /* Ранг 0 проверяет совпадение с полной историей */
        flag_send[0] = 0;
        if (rank == 0)
        {
            for (int k = 0; k < hash_size; k++)
            {
                if (hash_hist[k] == global_h_new)
                {
                    flag_send[0] = 1;
                    if (rank == 0)
                        printf("  [совпадение с итерацией %d, период=%d]\n",
                               k, iter - k);
                    break;
                }
            }
            /* Сохранить хэш */
            if (hash_size == hash_cap)
            {
                hash_cap *= 2;
                hash_hist = (uint64_t*)realloc(hash_hist, hash_cap * sizeof(uint64_t));
            }
            hash_hist[hash_size++] = global_h_new;
        }

        /* Раздача флага останова от ранга 0 всем процессам*/
        MPI_Bcast(flag_send, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (flag_send[0])
        {
            stopped_iter = iter;
            break;
        }
    }

    double t_local = MPI_Wtime() - t_start;
    double t_total;
    MPI_Reduce(&t_local, &t_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int last_iter = (stopped_iter >= 0) ? stopped_iter : max_iter;
    cell_t* final_buf = (last_iter % 2 == 1) ? buf1 : buf0;

    int* recvcounts = NULL;
    int* displs = NULL;
    cell_t* gathered = NULL;
    if (rank == 0)
    {
        recvcounts = (int*)malloc(np * sizeof(int));
        displs = (int*)malloc(np * sizeof(int));
        for (int r = 0; r < np; r++)
        {
            recvcounts[r] = rows_for_rank(X, np, r) * Y;
            displs[r] = first_row_for_rank(X, np, r) * Y;
        }
        gathered = (cell_t*)malloc(X * Y * sizeof(cell_t));
    }

    MPI_Gatherv(&CELL(final_buf, 1, 0, Y), local_rows * Y, MPI_UNSIGNED_CHAR,
                gathered, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("t_total    = %.6f\n", t_total);
        printf("X=%d Y=%d\n", X, Y);
        if (stopped_iter >= 0)
            printf("stopped_at = %d\n", stopped_iter);
        else
            printf("stopped_at = max_iter (%d)\n", max_iter);

        FILE* out = fopen("output.txt", "w");
        if (out)
        {
            for (int i = 0; i < X; i++)
            {
                for (int j = 0; j < Y; j++)
                    fputc(gathered[i * Y + j] ? '#' : '.', out);
                fputc('\n', out);
            }
            fclose(out);
            printf("Финальное состояние записано в output.txt\n");
        }
        free(recvcounts);
        free(displs);
        free(gathered);
        free(hash_hist);
    }

    free(buf0);
    free(buf1);
    MPI_Finalize();
    return 0;
}

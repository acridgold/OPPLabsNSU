#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

using cell_t = unsigned char;
#define CELL(buf, r, c, Y) ((buf)[(r)*(Y)+(c)])

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static uint64_t hash_strip(const cell_t* buf, int rows, int Y) {
    uint64_t h = 14695981039346656037ULL;
    int n = rows * Y;
    for (int i = 0; i < n; i++) {
        h ^= (uint64_t)buf[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [max_iter]\n", argv[0]);
        return 1;
    }
    int max_iter = (argc >= 3) ? atoi(argv[2]) : 100000;
    int X = 0, Y = 0;

    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("File error"); return 1; }
    char line[100000];
    if (fgets(line, sizeof(line), f)) {
        Y = (int)strlen(line);
        while (Y > 0 && (line[Y-1] == '\n' || line[Y-1] == '\r')) Y--;
        X = 1;
        while (fgets(line, sizeof(line), f)) if (strlen(line) > 1) X++;
    }
    rewind(f);

    /* ghost rows: строки 1..X — данные, 0 и X+1 — призрачные */
    cell_t* const buf0 = (cell_t*)calloc((X + 2) * Y, sizeof(cell_t));
    cell_t* const buf1 = (cell_t*)calloc((X + 2) * Y, sizeof(cell_t));

    for (int i = 0; i < X; i++)
        if (fgets(line, sizeof(line), f))
            for (int j = 0; j < Y; j++)
                CELL(buf0, i+1, j, Y) = (line[j] == '#') ? 1 : 0;
    fclose(f);

    int hash_cap = 1024, hash_size = 0;
    uint64_t* hash_hist = (uint64_t*)malloc(hash_cap * sizeof(uint64_t));

    uint64_t h = hash_strip(&CELL(buf0, 1, 0, Y), X, Y);
    hash_hist[hash_size++] = h;

    double t_start = get_time();
    int stopped_iter = -1;

    for (int iter = 1; iter <= max_iter; iter++) {
        cell_t* cur = (iter % 2 == 1) ? buf0 : buf1;
        cell_t* dst = (iter % 2 == 1) ? buf1 : buf0;

        /* обновить призрачные строки (тороидальная топология) */
        memcpy(&CELL(cur, 0,   0, Y), &CELL(cur, X, 0, Y), Y);
        memcpy(&CELL(cur, X+1, 0, Y), &CELL(cur, 1, 0, Y), Y);

        for (int r = 1; r <= X; r++) {
            for (int c = 0; c < Y; c++) {
                int lc = (c - 1 + Y) % Y, rc = (c + 1) % Y;
                int alive =
                    CELL(cur,r-1,lc,Y)+CELL(cur,r-1,c,Y)+CELL(cur,r-1,rc,Y)+
                    CELL(cur,r,  lc,Y)+                   CELL(cur,r,  rc,Y)+
                    CELL(cur,r+1,lc,Y)+CELL(cur,r+1,c,Y)+CELL(cur,r+1,rc,Y);
                CELL(dst, r, c, Y) = CELL(cur, r, c, Y) ? (alive==2||alive==3) : (alive==3);
            }
        }

        uint64_t new_h = hash_strip(&CELL(dst, 1, 0, Y), X, Y);

        for (int k = 0; k < hash_size; k++) {
            if (hash_hist[k] == new_h) {
                printf("  [совпадение с итерацией %d, период=%d]\n", k, iter - k);
                stopped_iter = iter;
                break;
            }
        }

        if (hash_size == hash_cap) {
            hash_cap *= 2;
            hash_hist = (uint64_t*)realloc(hash_hist, hash_cap * sizeof(uint64_t));
        }
        hash_hist[hash_size++] = new_h;

        if (stopped_iter >= 0) break;
    }

    double t_total = get_time() - t_start;
    printf("t_sequential = %.6f\n", t_total);
    printf("X=%d Y=%d\n", X, Y);
    if (stopped_iter >= 0)
        printf("stopped_at = %d\n", stopped_iter);
    else
        printf("stopped_at = max_iter (%d)\n", max_iter);

    int last_iter = (stopped_iter >= 0) ? stopped_iter : max_iter;
    cell_t* final_buf = (last_iter % 2 == 1) ? buf1 : buf0;

    FILE* out = fopen("output.txt", "w");
    if (out) {
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++)
                fputc(CELL(final_buf, i+1, j, Y) ? '#' : '.', out);
            fputc('\n', out);
        }
        fclose(out);
        printf("Результат записан в output.txt\n");
    }

    free(hash_hist);
    free(buf0);
    free(buf1);
    return 0;
}
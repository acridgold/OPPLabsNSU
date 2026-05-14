#include <mpi.h>
#include <pthread.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static constexpr int TASKS_PER_PROC = 200;
static constexpr int L               = 10000;
static constexpr int NUM_ITER        = 10;

enum Tags {
    TAG_REQUEST = 1, TAG_REPLY, TAG_DATA,
    TAG_STOP_REPORT, TAG_BUSY_REPORT, TAG_FINAL_STOP
};

typedef struct { int id; int repeatNum; } Task;

typedef struct {
    int rank, size;
    Task* tasks;
    int   taskCount;
    int   front;
    pthread_mutex_t mu;
    double globalRes;
    int    doneTasks;
    int workerIdle;
    int iterDone;
    int reported_idle;
    int idle_count;
    pthread_cond_t cvWorker;
} State;

typedef struct { State* st; int iter; } Args;

static int task_weight(int i, int iter, int size) {
    int a = abs(50 - i % 100);
    int b = abs((i / 100) - (iter % size));
    return (a + 1) * (b + 1) * L;
}

static void* worker_thread(void* arg) {
    State* st = ((Args*)arg)->st;
    while (true) {
        pthread_mutex_lock(&st->mu);
        while (st->front >= st->taskCount && !st->iterDone)
            pthread_cond_wait(&st->cvWorker, &st->mu);
        if (st->iterDone && st->front >= st->taskCount) {
            pthread_mutex_unlock(&st->mu);
            break;
        }
        Task t = st->tasks[st->front++];
        pthread_mutex_unlock(&st->mu);

        double res = 0.0;
        for (int k = 0; k < t.repeatNum; k++) res += sin(t.id + k);

        pthread_mutex_lock(&st->mu);
        st->globalRes += res;
        st->doneTasks++;
        if (st->front >= st->taskCount) st->workerIdle = 1;
        pthread_mutex_unlock(&st->mu);
    }
    return NULL;
}

static void handle_requests(State* st) {
    int flag; MPI_Status status;
    while (MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &flag, &status), flag) {
        int src = status.MPI_SOURCE;
        int dummy; MPI_Recv(&dummy, 1, MPI_INT, src, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pthread_mutex_lock(&st->mu);
        int give = (st->taskCount - st->front) / 2;
        if (give < 1) give = 0;
        MPI_Send(&give, 1, MPI_INT, src, TAG_REPLY, MPI_COMM_WORLD);
        if (give > 0) {
            MPI_Send(st->tasks + (st->taskCount - give), give * (int)sizeof(Task), MPI_BYTE, src, TAG_DATA, MPI_COMM_WORLD);
            st->taskCount -= give;
        }
        pthread_mutex_unlock(&st->mu);
    }
    if (st->rank == 0) {
        while (MPI_Iprobe(MPI_ANY_SOURCE, TAG_STOP_REPORT, MPI_COMM_WORLD, &flag, &status), flag) {
            int d; MPI_Recv(&d, 1, MPI_INT, status.MPI_SOURCE, TAG_STOP_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            st->idle_count++;
        }
        while (MPI_Iprobe(MPI_ANY_SOURCE, TAG_BUSY_REPORT, MPI_COMM_WORLD, &flag, &status), flag) {
            int d; MPI_Recv(&d, 1, MPI_INT, status.MPI_SOURCE, TAG_BUSY_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            st->idle_count--;
        }
    } else if (MPI_Iprobe(0, TAG_FINAL_STOP, MPI_COMM_WORLD, &flag, &status), flag) {
        int d; MPI_Recv(&d, 1, MPI_INT, 0, TAG_FINAL_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pthread_mutex_lock(&st->mu);
        st->iterDone = 1;
        pthread_cond_signal(&st->cvWorker);
        pthread_mutex_unlock(&st->mu);
    }
}

static void* listener_thread(void* arg) {
    State* st = ((Args*)arg)->st;
    while (1) {
        handle_requests(st);
        pthread_mutex_lock(&st->mu);
        if (st->iterDone) { pthread_mutex_unlock(&st->mu); break; }
        int idle = st->workerIdle;
        pthread_mutex_unlock(&st->mu);

        if (idle) {
            int success = 0;
            for (int i = 1; i < st->size && !success; i++) {
                int target = (st->rank + i) % st->size;
                int d = 0; MPI_Send(&d, 1, MPI_INT, target, TAG_REQUEST, MPI_COMM_WORLD);
                int count = -1;
                while (1) {
                    handle_requests(st);
                    int f; MPI_Status s;
                    MPI_Iprobe(target, TAG_REPLY, MPI_COMM_WORLD, &f, &s);
                    if (f) {
                        MPI_Recv(&count, 1, MPI_INT, target, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        break;
                    }
                    if (st->iterDone) break;
                    usleep(500);
                }
                if (count > 0) {
                    Task* buf = (Task*)malloc((size_t)count * sizeof(Task));
                    MPI_Recv(buf, count * (int)sizeof(Task), MPI_BYTE, target, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pthread_mutex_lock(&st->mu);
                    if (st->reported_idle && st->rank != 0) {
                        int b = 1; MPI_Send(&b, 1, MPI_INT, 0, TAG_BUSY_REPORT, MPI_COMM_WORLD);
                        st->reported_idle = 0;
                    }
                    st->tasks = (Task*)realloc(st->tasks, (size_t)(st->taskCount + count) * sizeof(Task));
                    memcpy(st->tasks + st->taskCount, buf, (size_t)count * sizeof(Task));
                    st->taskCount += count; st->workerIdle = 0;
                    pthread_cond_signal(&st->cvWorker);
                    pthread_mutex_unlock(&st->mu);
                    free(buf); success = 1;
                }
                if (st->iterDone) break;
            }
            if (!success && !st->iterDone) {
                if (st->rank != 0) {
                    if (!st->reported_idle) {
                        int s = 1; MPI_Send(&s, 1, MPI_INT, 0, TAG_STOP_REPORT, MPI_COMM_WORLD);
                        st->reported_idle = 1;
                    }
                } else {
                    pthread_mutex_lock(&st->mu);
                    if (st->idle_count == st->size - 1) {
                        st->iterDone = 1; pthread_cond_signal(&st->cvWorker);
                        for (int k = 1; k < st->size; k++) {
                            int stop = 1; MPI_Send(&stop, 1, MPI_INT, k, TAG_FINAL_STOP, MPI_COMM_WORLD);
                        }
                    }
                    pthread_mutex_unlock(&st->mu);
                }
            }
        }
        usleep(1000);
    }
    return NULL;
}

int main(int argc, char** argv) {
    int prov; MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
    State st; memset(&st, 0, sizeof(st));
    MPI_Comm_rank(MPI_COMM_WORLD, &st.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &st.size);
    pthread_mutex_init(&st.mu, NULL);
    pthread_cond_init(&st.cvWorker, NULL);

    if (st.rank == 0)
        printf("=== MPI+pthreads | procs=%d tasks/proc=%d L=%d iters=%d ===\n\n",
               st.size, TASKS_PER_PROC, L, NUM_ITER);

    double* allTimes = (st.rank == 0)
                       ? (double*)malloc((size_t)st.size * sizeof(double))
                       : NULL;

    for (int iter = 0; iter < NUM_ITER; iter++) {
        pthread_mutex_lock(&st.mu);
        if (st.tasks) free(st.tasks);
        st.tasks = (Task*)malloc((size_t)TASKS_PER_PROC * sizeof(Task));
        st.taskCount = TASKS_PER_PROC; st.front = 0; st.globalRes = 0.0; st.doneTasks = 0;
        st.workerIdle = 0; st.iterDone = 0; st.reported_idle = 0; st.idle_count = 0;
        for (int i = 0; i < TASKS_PER_PROC; i++) {
            int gid = st.rank * TASKS_PER_PROC + i;
            st.tasks[i].id = gid;
            st.tasks[i].repeatNum = task_weight(gid, iter, st.size);
        }
        pthread_mutex_unlock(&st.mu);

        Args args = {&st, iter};
        pthread_t wt, lt;
        pthread_create(&wt, NULL, worker_thread,   &args);
        pthread_create(&lt, NULL, listener_thread, &args);

        double t1 = MPI_Wtime();
        pthread_join(lt, NULL);
        pthread_join(wt, NULL);
        double t2 = MPI_Wtime();
        double localTime = t2 - t1;

        printf("  [rank %d] iter=%d  t=%.4f s  tasks=%d  res=%.6f\n",
               st.rank, iter, localTime, st.doneTasks, st.globalRes);
        fflush(stdout);

        MPI_Gather(&localTime, 1, MPI_DOUBLE,
                   allTimes,   1, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (st.rank == 0) {
            double tmax = allTimes[0], tmin = allTimes[0];
            for (int p = 1; p < st.size; p++) {
                if (allTimes[p] > tmax) tmax = allTimes[p];
                if (allTimes[p] < tmin) tmin = allTimes[p];
            }
            double imb = (tmax > 0.0) ? (tmax - tmin) / tmax * 100.0 : 0.0;
            printf("  [iter %d] Tmax=%.4f s  Tmin=%.4f s  delta=%.4f s  imbalance=%.2f%%\n\n",
                   iter, tmax, tmin, tmax - tmin, imb);
            fflush(stdout);
        }
    }

    if (st.rank == 0) free(allTimes);
    free(st.tasks);
    pthread_mutex_destroy(&st.mu);
    pthread_cond_destroy(&st.cvWorker);
    MPI_Finalize();
    return 0;
}

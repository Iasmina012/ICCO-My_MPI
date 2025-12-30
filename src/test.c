#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "my_mpi.h"

#define BCAST_ITERS     10000
#define REDUCE_ITERS    10000
#define ALLREDUCE_ITERS 10000
#define SCATTER_ITERS   2000
#define GATHER_ITERS    2000
#define BARRIER_ITERS   2000

void print_csv(const char *name, int size, double my_mpi_time, double mpi_time) {

    printf("%s,%d,%.9f,%.9f\n", name, size, my_mpi_time, mpi_time);

}

int verify_tests(int rank, int size) {

    int ok=1;

    //Bcast
    {
        int x = (rank == 0) ? 42 : 0;

        My_MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (x != 42) 
            ok = 0;
    }

    //Reduce
    {
        int send=rank+1, recv=0;

        My_MPI_Reduce(&send, &recv, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            int expected=size*(size+1)/2;
            if (recv != expected) 
                ok=0;
        }
    }

    //Allreduce
    {
        int send=rank+1, recv=0;

        My_MPI_Allreduce(&send, &recv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        int expected=size*(size+1)/2;
        if (recv != expected) 
            ok=0;
    }

    //Scatter
    {
        int *sendbuf=NULL, recv=-1;

        if (rank == 0) {
            sendbuf=malloc(size*sizeof(int));
            for (int i=0;i<size;i++)
                sendbuf[i]=100+i;
        }
        My_MPI_Scatter(sendbuf, 1, MPI_INT, &recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (recv != 100 + rank)
            ok=0;
        if (sendbuf) 
            free(sendbuf);
    }

    //Gather
    {
        int send=rank, *recv=NULL;

        if (rank == 0)
            recv=malloc(size*sizeof(int));
        My_MPI_Gather(&send, 1, MPI_INT, recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int i=0;i<size;i++)
                if (recv[i] != i)
                    ok=0;
            free(recv);
        }
    }

    //Barrier
    {
        My_MPI_Barrier(MPI_COMM_WORLD);
        //if blocks => wrong 
    }

    return ok;

}

void benchmark_bcast(int rank, int size) {

    int x = (rank == 0) ? 123 : 0;
    double t1, t2, my_mpi_time, mpi_time;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<BCAST_ITERS;i++)
        My_MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    my_mpi_time=(t2-t1)/BCAST_ITERS;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<BCAST_ITERS;i++)
        MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    mpi_time=(t2-t1)/BCAST_ITERS;

    if (rank == 0)
        print_csv("bcast", size, my_mpi_time, mpi_time);
}

void benchmark_reduce(int rank, int size) {
   
    int send=rank+1, r1=0, r2=0;
    double t1, t2, my_mpi_time, mpi_time;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<REDUCE_ITERS;i++)
        My_MPI_Reduce(&send, &r1, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    my_mpi_time=(t2-t1)/REDUCE_ITERS;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<REDUCE_ITERS;i++)
        MPI_Reduce(&send, &r2, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    mpi_time=(t2-t1)/REDUCE_ITERS;

    if (rank == 0)
        print_csv("reduce", size, my_mpi_time, mpi_time);
}

void benchmark_allreduce(int rank, int size) {

    int send=rank+1, r1=0, r2=0;
    double t1, t2, my_mpi_time, mpi_time;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < ALLREDUCE_ITERS; i++)
        My_MPI_Allreduce(&send, &r1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    my_mpi_time = (t2 - t1) / ALLREDUCE_ITERS;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < ALLREDUCE_ITERS; i++)
        MPI_Allreduce(&send, &r2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    mpi_time = (t2 - t1) / ALLREDUCE_ITERS;

    if (rank == 0)
        print_csv("allreduce", size, my_mpi_time, mpi_time);
}

void benchmark_scatter(int rank, int size) {
    
    int *sendbuf=NULL, recv=0;
    double t1, t2, my_mpi_time, mpi_time;

    if (rank == 0) {
        sendbuf=malloc(size * sizeof(int));
        for (int i=0;i<size;i++)
            sendbuf[i]=100+i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<SCATTER_ITERS;i++)
        My_MPI_Scatter(sendbuf, 1, MPI_INT, &recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    my_mpi_time=(t2-t1)/SCATTER_ITERS;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<SCATTER_ITERS;i++)
        MPI_Scatter(sendbuf, 1, MPI_INT, &recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    mpi_time=(t2-t1)/SCATTER_ITERS;

    if (rank == 0)
        print_csv("scatter", size, my_mpi_time, mpi_time);

    if (sendbuf) 
        free(sendbuf);

}

void benchmark_gather(int rank, int size) {

    int send=rank, *recv=NULL;
    double t1, t2, my_mpi_time, mpi_time;

    if (rank == 0)
        recv=malloc(size * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<GATHER_ITERS;i++)
        My_MPI_Gather(&send, 1, MPI_INT, recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    my_mpi_time=(t2-t1)/GATHER_ITERS;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<GATHER_ITERS;i++)
        MPI_Gather(&send, 1, MPI_INT, recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    t2=MPI_Wtime();
    mpi_time=(t2-t1)/GATHER_ITERS;

    if (rank == 0)
        print_csv("gather", size, my_mpi_time, mpi_time);

    if (recv)  
        free(recv);

}

void benchmark_barrier(int rank, int size) {
    
    double t1, t2, my_mpi_time, mpi_time;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<BARRIER_ITERS;i++)
        My_MPI_Barrier(MPI_COMM_WORLD);
    t2=MPI_Wtime();
    my_mpi_time=(t2-t1)/BARRIER_ITERS;

    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime();
    for (int i=0;i<BARRIER_ITERS;i++)
        MPI_Barrier(MPI_COMM_WORLD);
    t2=MPI_Wtime();
    mpi_time=(t2-t1)/BARRIER_ITERS;

    if (rank == 0)
        print_csv("barrier", size, my_mpi_time, mpi_time);

}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ok=verify_tests(rank, size);
    if (rank == 0) {
        if (ok)
            printf("verify_tests\n");
        else
            printf("ERROR: TESTS_FAILED\n");
    }

    benchmark_bcast(rank, size);
    benchmark_reduce(rank, size);
    benchmark_allreduce(rank, size);
    benchmark_scatter(rank, size);
    benchmark_gather(rank, size);
    benchmark_barrier(rank, size);

    MPI_Finalize();
    return 0;

}
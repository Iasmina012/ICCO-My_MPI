#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "my_mpi.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ok=1;

    //Barrier
    if (rank == 0)
        printf("Testing My_MPI_Barrier...\n");
    My_MPI_Barrier(MPI_COMM_WORLD);

    //Bcast
    int N=4;
    int *buf=malloc(N * sizeof(int));
    int *buf_ref=malloc(N * sizeof(int));
    
    if (rank == 0) {
        for (int i=0;i<N;i++) 
            buf[i]=100+i;
    }
    else {
        for (int i=0;i<N;i++) 
            buf[i]=0;
    }
    My_MPI_Bcast(buf, N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i=0;i<N;i++) 
            buf_ref[i]=100+i;
    }
    else {
        for (int i=0;i<N;i++) 
            buf_ref[i]=0;
    }
    MPI_Bcast(buf_ref, N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i=0;i<N;i++)
        if (buf[i] != buf_ref[i]) 
            ok=0;

    //Reduce
    int send=rank+1;
    int my_mpi_red=0, mpi_red=0;
    My_MPI_Reduce(&send, &my_mpi_red, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&send, &mpi_red, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0 && my_mpi_red != mpi_red)
        ok=0;

    //Allreduce
    int my_mpi_all=0, mpi_all=0;

    My_MPI_Allreduce(&send, &my_mpi_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&send, &mpi_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (my_mpi_all != mpi_all)
        ok=0;

    //Scatter
    int *scatter_buf=NULL;
    int recv_scatter=-1, ref_scatter=-1;

    if (rank == 0) {
        scatter_buf=malloc(size * sizeof(int));
        for (int i=0;i<size;i++)
            scatter_buf[i]=10+i;
    }

    My_MPI_Scatter(scatter_buf, 1, MPI_INT, &recv_scatter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(scatter_buf, 1, MPI_INT, &ref_scatter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (recv_scatter != ref_scatter)
        ok=0;

    //Gather
    int send_g=rank*2;
    int *my_mpi_gather=NULL;
    int *mpi_gather=NULL;

    if (rank == 0) {
        my_mpi_gather=malloc(size * sizeof(int));
        mpi_gather=malloc(size * sizeof(int));
    }

    My_MPI_Gather(&send_g, 1, MPI_INT, my_mpi_gather, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&send_g, 1, MPI_INT, mpi_gather, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i=0;i<size;i++)
            if (my_mpi_gather[i] != mpi_gather[i])
                ok=0;
    }

    if (rank == 0) {
        if (ok) 
            printf("ALL_TESTS_OK\n");
        else 
            printf("ALL_TESTS_FAILED\n");
    }

    //Allreduce Benchmark
    MPI_Barrier(MPI_COMM_WORLD);

    int x=rank+1;
    int r1=0, r2=0;
    const int ITER=10000;

    double t1=MPI_Wtime();
    for (int i=0;i<ITER;i++)
        My_MPI_Allreduce(&x, &r1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double t2=MPI_Wtime();

    double t3=MPI_Wtime();
    for (int i=0;i<ITER;i++)
        MPI_Allreduce(&x, &r2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double t4=MPI_Wtime();

    if (rank == 0)
        printf("%.9f,%.9f\n", (t2-t1)/ITER, (t4-t3)/ITER);

    free(buf);
    free(buf_ref);
    if (scatter_buf) 
        free(scatter_buf);
    if (my_mpi_gather) 
        free(my_mpi_gather);
    if (mpi_gather) 
        free(mpi_gather);

    MPI_Finalize();
    return 0;
}
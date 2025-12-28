#include "my_mpi.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

//Barrier - dissemination using Sendrecv
int My_MPI_Barrier(MPI_Comm comm) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int step=1;
    while (step < size) {

        int send_to=(rank+step)%size;
        int recv_from=(rank-step+size)%size;
        // send/recv 0-byte messages to synchronize
        MPI_Sendrecv(NULL, 0, MPI_CHAR, send_to, step, NULL, 0, MPI_CHAR, recv_from, step, comm, MPI_STATUS_IGNORE);
        step<<=1;
   
    }

    return MPI_SUCCESS;

}

//Bcast - binomial tree
int My_MPI_Bcast(void *buf, int count, MPI_Datatype dt, int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int vrank=(rank-root+size)%size;

    for (int mask=1;mask<size;mask<<=1) {

        if (vrank < mask) {
            
            int dst=vrank+mask;
            if (dst < size) {
                int real_dst=(dst+root)%size;
                MPI_Send(buf, count, dt, real_dst, 0, comm);
            }

        } else if (vrank < 2 * mask) {

            int src=vrank-mask;
            int real_src=(src+root)%size;
            MPI_Recv(buf, count, dt, real_src, 0, comm, MPI_STATUS_IGNORE);

        }

    }

    return MPI_SUCCESS;

}

//Reduce - binomial tree, generic via MPI_Reduce_local
int My_MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm) {
   
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int typesize;
    MPI_Type_size(dt, &typesize);

    //acc = local acumulator
    void *acc=malloc(count * typesize);
    if (!acc) 
        return MPI_ERR_NO_MEM;
    memcpy(acc, sendbuf, count * typesize);

    int vrank=(rank-root+size)%size;

    for (int mask = 1; mask < size; mask <<= 1) {

        if (vrank & mask) {

            // send to partner and exit loop
            int dst_v=vrank-mask;
            int dst=(dst_v+root)%size;
            MPI_Send(acc, count, dt, dst, 0, comm);
            break;

        } else {
           
            int src_v=vrank+mask;
            if (src_v < size) {

                int src=(src_v+root)%size;
                void *tmp=malloc(count * typesize);
               
                if (!tmp) { 
                    free(acc); 
                    return MPI_ERR_NO_MEM; 
                }

                MPI_Recv(tmp, count, dt, src, 0, comm, MPI_STATUS_IGNORE);
                // apply operation: tmp (in) combined into acc (inout)
                MPI_Reduce_local(tmp, acc, count, dt, op);
                free(tmp);

            }

        }

    }

    if (rank == root) {
        memcpy(recvbuf, acc, count * typesize);
    }

    free(acc);
    return MPI_SUCCESS;

}

//Allreduce = reduce to root then bcast result
int My_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
   
    int rank;
    MPI_Comm_rank(comm, &rank);

    int typesize;
    MPI_Type_size(dt, &typesize);
    void *temp=malloc(count * typesize);
    if (!temp) 
        return MPI_ERR_NO_MEM;

    My_MPI_Reduce(sendbuf, temp, count, dt, op, 0, comm);
    My_MPI_Bcast(temp, count, dt, 0, comm);
    memcpy(recvbuf, temp, count * typesize);
    free(temp);
    return MPI_SUCCESS;

}

//Scatter
int My_MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int ssz=0, rsz=0;
    MPI_Type_size(sendtype, &ssz);
    MPI_Type_size(recvtype, &rsz);

    if (rank == root) {

        const char *sbase=(const char *)sendbuf;
        for (int i=0;i<size;i++) {
            if (i == root) {
                memcpy(recvbuf, sbase + i * (size_t)sendcount * ssz, (size_t)recvcount * rsz);
            } else {
                MPI_Send(sbase + i * (size_t)sendcount * ssz, sendcount, sendtype, i, 0, comm);
            }
        }

    } else {
        MPI_Recv(recvbuf, recvcount, recvtype, root, 0, comm, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;

}

//Gather
int My_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,  MPI_Datatype recvtype, int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int ssz=0, rsz=0;
    MPI_Type_size(sendtype, &ssz);
    MPI_Type_size(recvtype, &rsz);

    if (rank == root) {

        //copy root's sendbuf into correct position 
        char *rbase = (char *)recvbuf;
        memcpy(rbase + (size_t)rank * recvcount * rsz, sendbuf, (size_t)sendcount * ssz);
        for (int i=0;i<size;i++) {
            if (i == rank) 
                continue;
            MPI_Recv(rbase + (size_t)i * recvcount * rsz, recvcount, recvtype, i, 0, comm, MPI_STATUS_IGNORE);
        }

    } else {
        MPI_Send(sendbuf, sendcount, sendtype, root, 0, comm);
    }

    return MPI_SUCCESS;

}
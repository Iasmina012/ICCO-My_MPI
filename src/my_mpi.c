#include "my_mpi.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

//Barrier - dissemination barrier using Sendrecv
int My_MPI_Barrier(MPI_Comm comm) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int step=1, rounds=0;
    while (step < size) {
        step<<=1;
        rounds++;
    }

    for (int k=0;k<rounds;++k) {
        int send_to=(rank+(1<<k))%size;
        int recv_from=(rank-(1<<k)+size)%size;
        // send/recv 0-byte messages to synchronize, unique tag per round
        MPI_Sendrecv(NULL, 0, MPI_CHAR, send_to, 100+k, NULL, 0, MPI_CHAR, recv_from, 100+k, comm, MPI_STATUS_IGNORE);
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
                MPI_Send(buf, count, dt, real_dst, 200 + mask, comm);
            }

        } else if (vrank < 2 * mask) {

            int src=vrank-mask;
            int real_src=(src+root)%size;
            MPI_Recv(buf, count, dt, real_src, 200 + mask, comm, MPI_STATUS_IGNORE);

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
    void *acc=malloc((size_t)count * typesize);
    if (!acc) 
        return MPI_ERR_NO_MEM;
    memcpy(acc, sendbuf, (size_t)count * typesize);

    int vrank=(rank-root+size)%size;

    for (int mask=1;mask<size;mask<<=1) {

        if (vrank & mask) {

            // send to partner and exit loop
            int dst_v=vrank-mask;
            int dst=(dst_v+root)%size;
            MPI_Send(acc, count, dt, dst, 300 + mask, comm);
            break;

        } else {
           
            int src_v=vrank+mask;
            if (src_v < size) {

                int src=(src_v+root)%size;
                void *aux=malloc((size_t)count * typesize);
               
                if (!aux) { 
                    free(acc); 
                    return MPI_ERR_NO_MEM; 
                }

                MPI_Recv(aux, count, dt, src, 300 + mask, comm, MPI_STATUS_IGNORE);
                // apply operation: aux (in) combined into acc (inout)
                MPI_Reduce_local(aux, acc, count, dt, op);
                free(aux);

            }

        }

    }

    if (rank == root) {
        memcpy(recvbuf, acc, (size_t)count * typesize);
    }

    free(acc);
    return MPI_SUCCESS;

}

//Allreduce = reduce to root then bcast result
int My_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
   
    int rank;
    MPI_Comm_rank(comm, &rank);

    int typesize=0, root=0;
    MPI_Type_size(dt, &typesize);
    void *aux=malloc((size_t)count * typesize);
    if (!aux) 
        return MPI_ERR_NO_MEM;

    My_MPI_Reduce(sendbuf, aux, count, dt, op, root, comm);
    My_MPI_Bcast(aux, count, dt, root, comm);
    memcpy(recvbuf, aux,(size_t)count * typesize);
    free(aux);
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
        memcpy(recvbuf, sbase + (size_t)i * sendcount * ssz, (size_t)sendcount * ssz);
            } else {
        MPI_Send(sbase + (size_t)i * sendcount * ssz, sendcount, sendtype, i, 400 + i, comm);
            }
        }

    } else {
        MPI_Recv(recvbuf, recvcount, recvtype, root, 400 + rank, comm, MPI_STATUS_IGNORE);
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
            MPI_Recv(rbase + (size_t)i * recvcount * rsz, recvcount, recvtype, i, 500 + i, comm, MPI_STATUS_IGNORE);
        }

    } else {
        MPI_Send(sendbuf, sendcount, sendtype, root, 500 + rank, comm);
    }

    return MPI_SUCCESS;

}
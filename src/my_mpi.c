#include "my_mpi.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

//Barrier: dissemination barrier using Sendrecv
int My_MPI_Barrier(MPI_Comm comm) {

    int rank, size; //process' ID, number of processes
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int base_tag=1000; //unique tag
    int step=1, rounds=0; //distance, rounds
    
    //calculates the number of rounds
    while (step < size) {
        step<<=1; //each round doubles the number of synchronized processes
        rounds++; //in each round each process comms with a partner at the 2^k distance
    }

    for (int k=0;k<rounds;++k) {
        //each process communicates with the process at the 2^k distance
        int send_to=(rank+(1<<k))%size; //partner we send to
        int recv_from=(rank-(1<<k)+size)%size; //partner we wait a message from
        //sendrecv (avoids deadlock) 0 byte messages to synchronize, unique tag per round
        MPI_Sendrecv(NULL, 0, MPI_CHAR, send_to, base_tag + k, NULL, 0, MPI_CHAR, recv_from, base_tag + k, comm, MPI_STATUS_IGNORE);
    }
    
    return MPI_SUCCESS;

}

//Bcast: binomial tree
int My_MPI_Bcast(void *buf, int count, MPI_Datatype dt, int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //no data
    if (count == 0) 
        return MPI_SUCCESS;

    int base_tag=2000;
    int mask, vrank=(rank-root+size)%size; //rotate namespace of ranks so the virtual rank moves root to position 0

    for (mask=1;mask<size;mask<<=1) {
        
        //if process needs to recv in current round (bit mask from vrank=1 => child => recvs)
         if (vrank & mask) { //non null => this process needs to recv in that round
            int src_v=vrank & ~mask; //logic parent (partner who recvs from on that level)
            int real_src=(src_v+root)%size; //vrank to real rank
            MPI_Recv(buf, count, dt, real_src, base_tag + mask, comm, MPI_STATUS_IGNORE);
            break; //a process recvs only once in binomial tree
        }

    }

    //continues from current mask
    for (;mask<size;mask>>=1) {
        int dst_v=vrank | mask; //logic child
        if (dst_v < size) { //logic child exists
            int real_dst=(dst_v+root)%size; //vrank to real rank
            MPI_Send(buf, count, dt, real_dst, base_tag + mask, comm); //to child
            //root starts sending bc nodes that didn't need to recv in the first part don't do MPI_Recv
        }
    }

    //all children recvs
    return MPI_SUCCESS;

}

//Reduce: binomial tree, generic using MPI_Reduce_local to apply op between local buff and recv buff
int My_MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm) {
   
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //no data
    if (count == 0) 
        return MPI_SUCCESS;

    //element size
    int typesize;
    MPI_Type_size(dt, &typesize);
     
    //acc = local acumulator has partial result
    void *acc=malloc((size_t)count * typesize);
    if (!acc) //memory alloc
        return MPI_ERR_NO_MEM;

    //special case, only root can use MPI_IN_PLACE
    if (sendbuf == MPI_IN_PLACE) {
        if (rank == root)
            memcpy(acc, recvbuf, (size_t)count * typesize);
        else {
            free(acc);
            return MPI_ERR_ARG; //other process used MPI_IN_PLACE
        }
    } else {
        memcpy(acc, sendbuf, (size_t)count * typesize);
    }

    int base_tag=3000;
    int vrank=(rank-root+size)%size; //virtual rank moves root to position 0

    //reduce loop
    for (int mask=1;mask<size;mask<<=1) {

        if (vrank & mask) { //process is sender this round (non null)

            //sends accumulated value to partner and exit
            int dst_v = vrank & ~mask; //logic parent
            int real_dst = (dst_v + root) % size; //vrank to rank
            MPI_Send(acc, count, dt, real_dst, base_tag + mask, comm);
            free(acc);
            return MPI_SUCCESS;

        } else {
           
            int src_v = vrank | mask; //logic child
            if (src_v < size) { //logic child exists

                int real_src = (src_v + root) % size; //vrank to real rank
                void *aux = malloc((size_t)count * typesize);
                if (!aux) { 
                    free(acc); 
                    return MPI_ERR_NO_MEM; 
                }
                MPI_Recv(aux, count, dt, real_src, base_tag + mask, comm, MPI_STATUS_IGNORE); //child data
                MPI_Reduce_local(aux, acc, count, dt, op); //applies operation op, acc = op(aux, acc)
                free(aux);

            }

        }

    }

    if (rank == root) {
        memcpy(recvbuf, acc, (size_t)count * typesize); //root copies final result
    }

    free(acc);

    return MPI_SUCCESS;

}

//Allreduce: reduce to root then broadcast result
int My_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
   
    //element size
    int typesize=0;
    MPI_Type_size(dt, &typesize);

    void *aux=malloc((size_t)count * typesize);
    if (!aux) 
        return MPI_ERR_NO_MEM;

    //intialize local data
    if (sendbuf == MPI_IN_PLACE)
            memcpy(aux, recvbuf, (size_t)count * typesize);
        else
            memcpy(aux, sendbuf, (size_t)count * typesize);
    
    My_MPI_Reduce(aux, aux, count, dt, op, 0, comm); //reduces root to 0
    My_MPI_Bcast(aux, count, dt, 0, comm); //broadcast result
    
    //everyone recvs result
    memcpy(recvbuf, aux, (size_t)count * typesize);
    free(aux);
    
    return MPI_SUCCESS;

}

//Scatter: naive implementation which is root -> each process
int My_MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //types sizes
    int ssz=0, rsz=0;
    MPI_Type_size(sendtype, &ssz);
    MPI_Type_size(recvtype, &rsz);
    
    //verifies compatibilty
    if (sendcount != recvcount || ssz != rsz) 
        return MPI_ERR_ARG;

    int base_tag=4000;

    //root sends
    if (rank == root) {

        const char *sbase=(const char *)sendbuf; //pointer byte wise for offsets
        for (int i=0;i<size;i++) {
            if (i == root) {
                memcpy(recvbuf, sbase + (size_t)i * sendcount * ssz, (size_t)sendcount * ssz); //root copies its own chunk
            } else {
                MPI_Send(sbase + (size_t)i * sendcount * ssz, sendcount, sendtype, i, base_tag + i, comm); //sends chunk to process i
            }
        }

    } else {
        MPI_Recv(recvbuf, recvcount, recvtype, root, base_tag + rank, comm, MPI_STATUS_IGNORE); //non root processes recv
    }

    return MPI_SUCCESS;

}

//Gather: naive implementation which is each process -> root
int My_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,  MPI_Datatype recvtype, int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //types sizes
    int ssz=0, rsz=0;
    MPI_Type_size(sendtype, &ssz);
    MPI_Type_size(recvtype, &rsz);

    //verifies compatibility
    if (sendcount != recvcount || ssz != rsz) 
        return MPI_ERR_ARG;

    int base_tag=5000;

    //root sends
    if (rank == root) {

        //copy root's sendbuf into correct position (root copies its data)

        char *rbase = (char *)recvbuf; //pointer byte wise for offsets
        if (sendbuf != MPI_IN_PLACE) {
            memcpy(rbase + (size_t)rank * recvcount * rsz, sendbuf, (size_t)sendcount * ssz);
        }

        for (int i=0;i<size;i++) {
            if (i == rank) 
                continue;
            MPI_Recv(rbase + (size_t)i * recvcount * rsz, recvcount, recvtype, i, base_tag + i, comm, MPI_STATUS_IGNORE);
            //recvs from all other processes
        }

    } else {
        if (sendbuf == MPI_IN_PLACE) 
            return MPI_ERR_ARG;
        MPI_Send(sendbuf, sendcount, sendtype, root, base_tag + rank, comm); //sends to root
    }

    return MPI_SUCCESS;

}
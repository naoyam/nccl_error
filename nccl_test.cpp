#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <limits>
#include <cuda_runtime.h>
#include <nccl.h>
#include "mpi.h"

#define CHECK_CUDA(cuda_call)                                           \
  do {                                                                  \
    const cudaError_t cuda_status = cuda_call;                          \
    if (cuda_status != cudaSuccess) {                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define CHECK_MPI(call)                                                 \
  do {                                                                  \
    int status = call;                                                  \
    if (status != MPI_SUCCESS) {                                        \
      std::cerr << "MPI error" << std::endl;                            \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, status);                                \
    }                                                                   \
  } while (0)

#define CHECK_NCCL(call)                                                \
  do {                                                                  \
    ncclResult_t status = call;                                         \
    if (status != ncclSuccess) {                                        \
      std::cerr << "NCCL error" << std::endl;                           \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, status);                                \
    }                                                                   \
  } while (0)

void set_device() {
  char *env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = atoi(env);
  env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  int local_size = atoi(env);
  int gpu = local_rank % local_size;
  CHECK_CUDA(cudaSetDevice(gpu));
}

int main(int argc, char *argv[]) {
  set_device();
  int pid;
  int np;
  MPI_Init(&argc, &argv);
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &np));

  if (pid == 0) {
    std::cerr << "MPI initialized" << std::endl;
  }

  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
  cudaStream_t st;
  CHECK_CUDA(cudaStreamCreate(&st));
  if (pid == 0) {
    CHECK_NCCL(ncclGetUniqueId(&nccl_id));
  }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  CHECK_NCCL(ncclCommInitRank(&nccl_comm, np, nccl_id, pid));

  if (pid == 0) {
    std::cerr << "NCCL initialized" << std::endl;
  }

  // Allreduce one int value
  int init_val = 1;
  void *p;
  CHECK_CUDA(cudaMalloc(&p, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(p, &init_val, sizeof(int), cudaMemcpyHostToDevice));
  void *q;
  CHECK_CUDA(cudaMalloc(&q, sizeof(int)));
  CHECK_NCCL(ncclAllReduce(p,
                           q,
                           1,
                           ncclInt,
                           ncclSum,
                           nccl_comm,
                           st));

  if (pid == 0) {
    std::cerr << "ncclAllreduce issued" << std::endl;
  }

  // Check async errors
  ncclResult_t async_err;
  CHECK_NCCL(ncclCommGetAsyncError(nccl_comm, &async_err));
  if (async_err != ncclSuccess) {
    std::cerr << "NCCL error\n";
  }

  if (pid == 0) {
    std::cerr << "Async error checked" << std::endl;
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  if (pid == 0) {
    std::cerr << "CUDA operation synchronized" << std::endl;
  }

  if (pid == 0) {
    std::cerr << "All done" << std::endl;
  }

  int result;
  CHECK_CUDA(cudaMemcpy(&result, q, sizeof(int), cudaMemcpyDeviceToHost));

  std::cerr << "Result: " << result
            << "; " << ((result == np) ? "Success!" : "Incorrect!")
            << std::endl;

  MPI_Finalize();

return 0;
}

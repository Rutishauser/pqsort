//Udacity HW 4

#include "utils.h"
#include <thrust/host_vector.h>
#include "stdio.h"

///////////////////////////////////////////////////////////////////////////////////

void your_sort(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               const size_t numElems)
{
    int SIZE = 128 * 2048;
    int DELTA = SIZE - numElems;
    int M = 2048;
    int K = SIZE/M;
    int LIMIT = 1044000000;
    unsigned int * d_aP;
    unsigned int * d_aV;
    int * d_Bl;
    int * d_Bh;

    float *d_x;
    float m_x[8];
    
    printf("* SIZE = %d, DELTA = %d \n", SIZE, DELTA);
    
    checkCudaErrors( cudaMalloc((void**) &d_aP, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**) &d_aV, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**) &d_Bl, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**) &d_Bh, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc(&d_x, 8*sizeof(float)) );
    
    move<<<1, M/2>>>( d_inputVals, d_inputPos, d_aV, d_aP, 2*K, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    check<<<K, M/2>>>( d_aV, d_Bl, d_Bh, LIMIT, numElems );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    scan_collect<<<K, M/2>>>( (unsigned int *) d_Bl, M );
    scan_collect<<<K, M/2>>>( (unsigned int *) d_Bh, M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    scan_second<<<1, K/2>>>( (unsigned int *) d_Bl, K, M );
    scan_second<<<1, K/2>>>( (unsigned int *) d_Bh, K, M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    scan_distr<<<K, M/2>>>( (unsigned int *) d_Bl, M );
    scan_distr<<<K, M/2>>>( (unsigned int *) d_Bh, M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    place<<<2*K, M/2>>>( d_aV, d_aP, d_outputVals, d_outputPos, d_Bl, d_Bh, LIMIT, numElems );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    test<<<1, 1>>>( (unsigned int *) d_outputVals, d_outputPos, 0, numElems-1200, d_x ); // K*M
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    test<<<1, 1>>>( (unsigned int *) d_outputVals, d_outputPos, numElems-1100, numElems, d_x ); // K*M
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //
}

///////////////////////////////////////////////////////////////////////////////////


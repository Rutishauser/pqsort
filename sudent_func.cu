//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include "stdio.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.
   Note: ascending order == smallest to largest
   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.
   Implementing Parallel Radix Sort with CUDA
   ==========================================
   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.
   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there
   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.
 */
 
__global__
void move(  unsigned int* const d_inputVals, unsigned int* const d_inputPos,
	    unsigned int* const d_outputVals, unsigned int* const d_outputPos,
            const size_t size, const size_t num)
{
    int F = threadIdx.x * size;

    for(int i=0; i != size; i++)
    {
        if(F+i < num)
        {
            d_outputPos[F+i] = d_inputPos[F+i];
            d_outputVals[F+i] = d_inputVals[F+i];
        }
        else
        {
            d_outputPos[F+i] = F+i ;
            d_outputVals[F+i] = 0;
        }
    }    
}

__global__
void comp( unsigned int* const val, unsigned int* const pos,
           const size_t size, float* d_x)
{
    int I = threadIdx.x;
    unsigned int t;

    int z = size - 1;
    while ( val[z] != 0 ) z = z - 1 ;
    
    for(int i = z-1; i != -1 ; i--)
    {
        if( val[i] != 0 )
        {
            // swap z and i
            t = val[i];  val[i] = val[z]; val[z] = t;
            t = pos[i];  pos[i] = pos[z]; pos[z] = t;
            z = z - 1;
        }
    }
    d_x[0] = size - z;
}

__global__
void test( unsigned int* const val, unsigned int* const pos,
           const size_t start, const size_t end, float* d_x)
{
    int I = threadIdx.x;
    
    float sum = val[start];
    unsigned int min = val[start];
    unsigned int max = val[start];
    const size_t size = end - start;
    int zero = 0;
    int n = 1;
    int lt = 0;
    int eq = 0;
    int gt = 0;
    int z = -1;    
    int zp = -1;    
    int b = 0;
    int e = 0;
    
    for(int i=start+1; i < end; i++)
    {
        n = n + 1;
        sum = sum + val[i];
        if( val[i] > max ) max = val[i];
        if( val[i] < min ) min = val[i];
        if( val[i] == 0 )
        {
            zero = zero + 1;
            z = i;
            zp = pos[i];
        }
        if( val[i-1] < val[i] ) lt = lt + 1;
        if( val[i-1] == val[i] ) eq = eq + 1;
        if( val[i-1] > val[i] ) gt = gt + 1;
        if( val[i] > 1044000000 ) b = b + 1;
        if( val[i] == 1 ) e = e + 1;
    }    
    if( val[0] == 0 ) zero = zero + 1;

    printf("    size = %d, \n", n );
    printf("    min = %d, \n", min );
    printf("    max = %d, \n", max );
    // printf("    avr = %f, \n", sum/n );
    // printf("    n = %d, \n", n );
    // printf("    z = %d, \n", z );
    // printf("    zp = %d, \n", zp );
    printf("    zeros = %d, \n", zero );
    printf("    big = %d, \n", b ); // 923
    // printf("    # 1 = %d, \n", e ); // 1165

    for(int i=start; i != start+16; i++)
    {
        if( val[i] < 100 ) printf(" ");
        if( val[i] < 10 ) printf(" ");
        printf("  %d", val[i]);
    }
    printf("\n...\n");
    for(int i=end-16; i != end; i++)
    {
        if( val[i] < 100 ) printf(" ");
        if( val[i] < 10 ) printf(" ");
        printf("  %d", val[i]);
    }
    printf("\n");
}

__global__
void mm( unsigned int* const val, unsigned int* const pos,
           const size_t size, float* d_x)
{
    int tid = threadIdx.x;
    int gid = (2 * blockDim.x * blockIdx.x) + tid;

    int tempV, tempP;
    

    for (unsigned int s=blockDim.x; s>0; s>>=1) 
    {
        if (tid < s && gid < size)
        {
            if( val[gid] > val[gid+s] )
            {
                tempV = val[gid];  val[gid] = val[gid+s];  val[gid+s] = tempV;
                tempP = pos[gid];  pos[gid] = pos[gid+s];  pos[gid+s] = tempP;
            }
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        // printf("* val[%d] = %d, pos[%d] = %d \n", gid, val[gid], gid, pos[gid] );
    }
}

__global__
void mmm( unsigned int* const val, unsigned int* const pos,
           int M, const size_t size, float* d_x)
{
    int tid = threadIdx.x;

    int tempV, tempP;
    
    for (unsigned int s=blockDim.x; s>0; s>>=1) 
    {
        if (tid < s )
        {
            int p = M * s;
            int d = M * tid;
            
            if( val[tid] > val[tid+p] )
            {
                tempV = val[d];  val[d] = val[d+p];  val[d+s] = tempV;
                tempP = pos[d];  pos[d] = pos[d+p];  pos[d+s] = tempP;
            }
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        printf("*** val[%d] = %d, pos[%d] = %d \n", tid, val[tid], tid, pos[tid] );
        printf("*** blockDim.x = %d \n", blockDim.x );
    }
}


__global__
void above( unsigned int* const val, unsigned int* const pos,
           const size_t size, float* d_x)
{
    int tid = threadIdx.x;
    int gid = (2 * blockDim.x * blockIdx.x) + tid;

    unsigned int a = 1044000000; 
    
    if( val[gid] > a ) val[gid] = 1; else val[gid] = 0;
    if( val[gid+blockDim.x] > a ) val[gid+blockDim.x] = 1; else val[gid+blockDim.x] = 0;

    for (unsigned int s=blockDim.x; s>0; s>>=1) 
    {
        if (tid < s && gid < size)
        {
            val[gid] = val[gid] + val[gid+s];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        if( val[gid] != 0 )
        printf("    * val[%d] = %d, pos[%d] = %d \n", gid, val[gid], gid, pos[gid] );
    }
}

__global__
void summ( unsigned int* const val, unsigned int* const pos,
           int M, const size_t size, float* d_x)
{
    int tid = threadIdx.x;

    int tempV, tempP;
    
    for (unsigned int s=blockDim.x; s>0; s>>=1) 
    {
        if (tid < s )
        {
            int p = M * s;
            int d = M * tid;
            
            if( val[d] + val[d+p] > 0 )
            printf("val[%d] = val[%d] + val[%d] = %d + %d = %d \n", d, d, d+p, val[d], val[d+p], val[d] + val[d+p]);
            
            val[d] =  val[d] + val[d+p] ;
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        printf("*** val[%d] = %d, pos[%d] = %d \n", tid, val[tid], tid, pos[tid] );
        printf("*** blockDim.x = %d \n", blockDim.x );
    }
}

__global__
void init( unsigned int* const val, int M )
{
    printf("init: \n");
    for(int i=0; i != M; i++)
    {
        val[i] = val[i] % M;
        if( val[i] < 100 ) printf(" ");
        if( val[i] < 10 ) printf(" ");
        printf("  %d", val[i]);
    }
    printf("\nend init \n");
}

__global__
void check( unsigned int* const val, int* lo, int* hi, unsigned int limit, int num )
{
    int tid = threadIdx.x;
    int gid = (2 * blockDim.x * blockIdx.x) + tid;

    if( gid < num )
    {
        if( val[gid] > limit )
        {
            hi[gid] = 1;  lo[gid] = 0; 
        }
        else
        {
            hi[gid] = 0;  lo[gid] = 1;
        }
    }
    else
    {
        hi[gid] = 0;  lo[gid] = 0;
    }
    
    if( gid+blockDim.x < num )
    {
        if( val[gid+blockDim.x] > limit )
        {
            hi[gid+blockDim.x] = 1;  lo[gid+blockDim.x] = 0; 
        }
        else
        {
            hi[gid+blockDim.x] = 0;  lo[gid+blockDim.x] = 1;
        }
    }
    else
    {
        hi[gid+blockDim.x] = 0;  lo[gid+blockDim.x] = 0;
    }
}    

__global__
void scan( unsigned int* const val, int M )
{
    int tid = threadIdx.x;
    int F = 2 * blockDim.x * blockIdx.x ;
    int a, b;
    int d;
    
    for(int d=1; d < M; d = 2*d)
    {
        b = M - 1 - 2*d*tid ;
        a = b - d;

        if( tid < M/2/d )
        val[F+b] = val[F+a] + val[F+b] ;
    
        __syncthreads();
    }

    if(tid == 0)
    {
        val[F+M-1] = 0;
        // __syncthreads();
    }
    
    for(int d=M; d > 0; d = d/2)
    {
        b = M - 1 - 2*d*tid ;
        a = b - d;

        if( tid < M/2/d )
        {
            // val[b] = val[b] - val[a] ;
            int t = val[F+b];
            val[F+b] = val[F+b] + val[F+a];
            val[F+a] = t;
            // val[a] = val[b] - val[a];
        }
    
        __syncthreads();
    }

}

__global__
void ttt( unsigned int* const val, int N, int M )
{
    int tid = threadIdx.x;

    printf("ttt: %d - %d \n", N, M);
    
    if(tid == 0)
    {
       for(int i=N; i != M; i++)
       {
         if( i % 10 == 0 ) printf(" %5d :   ", i);
         if( val[i] < 1000 ) printf(" ");
         if( val[i] < 100 ) printf(" ");
         if( val[i] < 10 ) printf(" ");
         printf("  %d", val[i] );
         if( i % 10 == 9 ) printf("\n");
       }
       printf("\n");
    }
}

//     place<<<2*K, M/2>>>( d_aV, d_aP, d_Bl, d_Bh, 1044000000 );
__global__
void place( unsigned int* const val,  unsigned int* const pos,
            unsigned int* d_Ov, unsigned int* d_Op,
            int* lo, int* hi, unsigned int limit, int size )
{
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;

    if( gid < size )
    if( val[gid] > limit )
    {
        // H
        int iDst = size-1 - hi[gid];
        d_Ov[iDst] = val[gid];
        d_Op[iDst] = pos[gid];
    }
    else
    {
        // L
        int i = lo[gid];
        d_Ov[i] = val[gid];
        d_Op[i] = pos[gid];
    }
    
}

////////

__global__
void scan_collect( unsigned int* const val, int M )
{
    int tid = threadIdx.x;
    int F = 2 * blockDim.x * blockIdx.x ;
    int a, b;
    int d;
    
    for(int d=1; d < M; d = 2*d)
    {
        b = M - 1 - 2*d*tid ;
        a = b - d;

        if( tid < M/2/d )
        val[F+b] = val[F+a] + val[F+b] ;
    
        __syncthreads();
    }

}

__global__
void scan_second( unsigned int* const val, int M, int L )
{
    int tid = threadIdx.x;
    // blockIdx = 0  =>  F = 0
    int a, b;
    int d;
    
    for(int d=1; d < M; d = 2*d)
    {
        b = M - 1 - 2*d*tid ;
        a = b - d;

        if( tid < M/2/d )
        val[b*L + L - 1] = val[a*L + L - 1] + val[b*L + L - 1] ;
    
        __syncthreads();
    }

    if(tid == 0)
    {
        // (M-1)*L + L - 1 = M*L -1
        val[M*L - 1] = 0;
        // __syncthreads();
    }
    
    for(int d=M; d > 0; d = d/2)
    {
        b = M - 1 - 2*d*tid ;
        a = b - d;

        if( tid < M/2/d )
        {
            // val[b] = val[b] - val[a] ;
            int t = val[b*L + L - 1];
            val[b*L + L - 1] = val[b*L + L - 1] + val[a*L + L - 1];
            val[a*L + L - 1] = t;
            // val[a] = val[b] - val[a];
        }
    
        __syncthreads();
    }

}

__global__
void scan_distr( unsigned int* const val, int M )
{
    int tid = threadIdx.x;
    int F = 2 * blockDim.x * blockIdx.x ;
    int a, b;
    int d;
    
    for(int d=M; d > 0; d = d/2)
    {
        b = M - 1 - 2*d*tid ;
        a = b - d;

        if( tid < M/2/d )
        {
            // val[b] = val[b] - val[a] ;
            int t = val[F+b];
            val[F+b] = val[F+b] + val[F+a];
            val[F+a] = t;
            // val[a] = val[b] - val[a];
        }
    
        __syncthreads();
    }

}

///////////////////////////////////////////////////////////////////////////////////

void your_sort_old(unsigned int* d_inputVals,
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
    
    printf("SIZE = %d, DELTA = %d \n", SIZE, DELTA);
    
    checkCudaErrors( cudaMalloc((void**) &d_aP, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**) &d_aV, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**) &d_Bl, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc((void**) &d_Bh, SIZE*sizeof(int)) );
    checkCudaErrors( cudaMalloc(&d_x, 8*sizeof(float)) );
    
    move<<<1, M/2>>>( d_inputVals, d_inputPos, d_aV, d_aP, 2*K, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    K = 4;
    M = 8;
    init<<<1, 1>>>( d_aV, K*M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ttt<<<1, 1>>>( (unsigned int *) d_aV, 0, K*M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    check<<<K, M/2>>>( d_aV, d_Bl, d_Bh, 20, 25 ); // numElems
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ttt<<<1, 1>>>( (unsigned int *) d_Bl, 0, K*M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ttt<<<1, 1>>>( (unsigned int *) d_Bh, 0, K*M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // scan<<<K, M/2>>>( (unsigned int *) d_Bh, M );
    // scan<<<K, M/2>>>( (unsigned int *) d_Bl, M );
    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    scan_collect<<<K, M/2>>>( (unsigned int *) d_Bl, M );
    scan_collect<<<K, M/2>>>( (unsigned int *) d_Bh, M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    scan_second<<<1, K/2>>>( (unsigned int *) d_Bl, K, M );
    scan_second<<<1, K/2>>>( (unsigned int *) d_Bh, K, M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    scan_distr<<<K, M/2>>>( (unsigned int *) d_Bl, M );
    scan_distr<<<K, M/2>>>( (unsigned int *) d_Bh, M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ttt<<<1, 1>>>( (unsigned int *) d_Bl, 0, K*M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ttt<<<1, 1>>>( (unsigned int *) d_Bh, 0, K*M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    place<<<2*K, M/2>>>( d_aV, d_aP, d_outputVals, d_outputPos, d_Bl, d_Bh, 20, 25 );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ttt<<<1, 1>>>( (unsigned int *) d_outputVals, 0, K*M );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

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

    // comp<<<1, 1>>>( d_outputVals, d_outputPos, numElems, d_x );
    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //
}


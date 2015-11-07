
////////

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


//////// reduce: min and max

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

//////// reduce: number of big elements

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

//////// reduce: sum

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

//////// scan within one block

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

//////// dump

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

////////

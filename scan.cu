
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

////////

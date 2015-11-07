
//////// to move 2 arrays and fill the tail

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

//////// to mark big and small elements for partitioning

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
    
//////// to partition elements by indices

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

//////// to show the result

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

////////


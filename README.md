## Two Level Parallel Partition of an Array
### Fundamental GPU Algorithms (Applications of Sort and Scan)
Udacity CS344 sets an interesting assignment in home work 4 on sorting an array of 200K integers.
The students are supposed to implement a parallel version of the Radix Sort Algorithms.
It is not quite clear though how the sorted array would be used for alleged Red Eye Reduction.
It seems that the array in question doesn't need to be completely sorted. It actually needs
to be partitioned into two sub-arrays with small and big elements.  Then the big elements can
be used for identifying the pixels for the red eye reduction.  This sets us another, more
simple and interesting goal of parallel partitioning arrays.
The technique used for partitioning is an Exclusive Scan. 

Assumed that the pivot value is given, the proposed procedure is the following:
1. Create two auxiliary arrays.  The first array contains ones for all small elements of
the input array, and zeros for all other elements.  The second array contains ones for
all big elements of the input array, and zeros for all other elements.
This step is a simple map procedure.
2. Count the partials sums of the auxiliary arrays. This step is an exclusive scan procedure.
The result gives the indices where the big and small element should be moved to.
3. Place the big and the small elements into the output array, according to the indices.

The most interesting part of the proposed procedure is a parallel implementation of the
exclusive scan. 
The implementation consists of the two phases, the collection phase, and the distribution phase.
The collection phase starts with calculation of the sums of the pairs of next elements.
The requires N/2 additions, which can be done in parallel, using N/2 threads.  Next steps are less
demanding of calculations, and some threads will be idle.  The distribution phase requires
the same amount of calculations in a similar pattern.
Since the threads need to be synchronized after each step, all the threads should belong to the one
and the same block.  Assuming the maximum number of threads per block is 1024, we can scan
an array with up to 2048 elements without any problems.  But, we have to scan a much
larger array.

Scan of a larger array requires splitting the algorithm between a number of blocks and
may require a additional synchronization between blocks. 
Fortunately, the first steps of the scan algorithm are rather independent.
E.g. if we logically divide a big array into a great number of pairs of element, the sums
of pairs can be calculated independently in different blocks.  The next step involves
calculation of sums of four elements, and, once again the four elements belong to
one and the same block, and thus the calculation of the sums does not require synchronization
between blocks. The same is true about sums up to 2048 elements.  They all can be
calculated within the same block.

But at some point of the collection phase we will need to calculate the sums of 4096 elements,
and this will require access to the values (sums of 2048 elements) calculated within
different blocks.  Such access shall require synchronization between blocks, and
the recommended way of such synchronization is to complete the current kernel and
to synchronize at device level. Calculation of sums of 8192 elements seems also to require
synchronization between blocks.

Fortunately, further synchronizations between blocks can be avoided.  
Now the scan algorithm enters the phase of little calculation.  The number of active threads
drops dramatically. On each step we will calculate just a few sums. And so on, till the point
where we will need to calculate just one, the final sum, to substitute it with zero, and
to start the distribution phase. Initially, the distribution phase will require few
calculations too.  And then the amount of calculation will increase again.

The proposed idea is that the phase of little calculation can be performed by a little number
of threads belonging to the same block.  Thus the synchronization between threads can be
done within a block.  By and large the implementation is the following:

1. Launch the required number of blocks (at most 2048, for the two level implementation.)
Each block will be responsible for calculating sums of up to 2048 elements.
This is implemented as the scan_collect kernel.

2. Synchronize the device. And launch one block of with (at most 1024) threads.
Each thread will first continue the collection phase for two previous blocks.
After completion, the threads will start the distribution phase of the algorithm.
Symmetrically, the second phase will end when the required calculation reaches 2048
active threads.
This is implemented as the scan_second kernel.

3. Synchronize the device again. And launch the original number of blocks.
Each block will be responsible for calculating scan for 2048 elements.
This is implemented as the scan_distrib kernel.

This implementation is capable of performing the scan over 4M (2048*2048) elements.
It is tested on the given array of 200k integers. Without any optimization (e.g.
no shared memory is used) the proposed partitions the array in 55 ms.
(It takes 2 scans with some debug output.)



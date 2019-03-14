#ifndef __CONV_CONF_HEADER_
#define __CONV_CONF_HEADER_

#define LARGE_DATA_SET  1
#define MEDIUM_DATA_SET 0
#define SMALL_DATA_SET  0

#if LARGE_DATA_SET
        #define BATCH_SIZE 100
#elif MEDIUM_DATA_SET
        #define BATCH_SIZE 32
#elif SMALL_DATA_SET
        #define BATCH_SIZE 8
#endif

// Size of one data dimension
// Data is NxNx16
#if LARGE_DATA_SET
        #define NN 512
#elif MEDIUM_DATA_SET
        #define NN 64
#elif SMALL_DATA_SET
        #define NN 32
#endif

// Number of features in the input
#define FIn 16
// Number of features in the output
#define FOut 16

// Size of convolution filter (KxK)
#define K 5

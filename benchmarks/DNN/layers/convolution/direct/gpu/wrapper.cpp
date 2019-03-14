#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "configuration.h"
#include "wrapper.h"
#include <tiramisu/utils.h>



int main(int, char **)
{
    Halide::Buffer<float> input(NN+K, NN+K, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K, K, FIn, FOut);
    Halide::Buffer<float> bias(FIn);
    Halide::Buffer<float> conv_tiramisu_buffer(NN, NN, FOut, BATCH_SIZE);
    Halide::Buffer<int> parameters(5);

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    for (int y = 0; y < NN+K; ++y)
        for (int x = 0; x < NN+K; ++x)
            for (int z = 0; z < FIn; ++z)
                for (int n = 0; n < BATCH_SIZE; ++n)
                    input(x, y, z, n) = 1;

    for (int z = 0; z < FIn; ++z)
        bias(z) = 1;

     for (int y = 0; y < K; ++y)
        for (int x = 0; x < K; ++x)
            for (int z = 0; z < FIn; ++z)
                for (int q = 0; q < FOut; ++q)
                    filter(x, y, z, q) = 1;

    std::cout << "\t\tBuffers initialized" << std::endl;
    
    // Initialize parameters[]
    parameters(0) = NN;
    parameters(1) = K;
    parameters(2) = FIn;
    parameters(3) = FOut;
    parameters(4) = BATCH_SIZE;
    conv_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), conv_tiramisu_buffer.raw_buffer());

    return 0;
}

    


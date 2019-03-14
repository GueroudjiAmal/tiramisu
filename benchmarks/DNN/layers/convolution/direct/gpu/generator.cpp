#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/utils.h>

#include <string.h>
#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{

    init("conv_tiramisu");

    var i("i", 0, 5);
    input parameters("parameters",{i}, p_int32);

    constant C_N("C_N", parameters(0)+ parameters(1));
    constant C_N1("C_N1",parameters(0));
    constant C_K("C_K", parameters(1));
    constant C_FIn("C_FIn", parameters(2));
    constant C_FOut("C_FOut", parameters(3));
    constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4));

    var x("x", 0, C_N ), y("y", 0, C_N),  z("z", 0, C_FOut), n("n", 0, C_BATCH_SIZE ); // input
    var k_x("k_x",0,C_K), k_y("k_y",0,C_K), k_z("k_z",0,C_FIn); // filter variables
    var x1("x1", 0, C_N1), y1("y1", 0, C_N1); // conv

    // Input computations
    input c_input("c_input",{n, k_z, y, x} , p_float32);
    input bias("bias", {z}, p_float32);
    input filter("filter", {z, k_z , k_y, k_x}, p_float32);

    // First conv computations
    computation conv_init("conv_init",{n, z, y1, x1}, bias(z));
    computation conv("conv",{n, z, y1, x1, k_z, k_y, k_x }, conv_init(n, z, y1, x1) + filter(z, k_z, k_y, k_x) * c_input(n, k_z, y1 + k_y, x1 + k_x));

    global::get_implicit_function()->add_context_constraints("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{:C_N>1 and C_K>1 and C_FOut>1 and C_FIn>0 and C_BATCH_SIZE>1 and C_K=5 and C_FIn%16=0 and C_N%16=0}");
    // Layer II
  
    conv.after(conv_init, 2);
            
    // Layer III
    buffer parameters_buf("parameters_buf", {expr(5)}, p_int32, a_input);
    buffer input_buf("input_buf", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_input);
    buffer conv_buf("conv_buf", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N1), expr(C_N1)}, p_float32, a_output);
    buffer filter_buf("filter_buf", {expr(C_FOut), expr(C_FIn), expr(C_K), expr(C_K)}, p_float32, a_input);
    buffer bias_buf("bias_buf", {expr(C_FOut)}, p_float32, a_input);

    buffer input_buf_gpu("input_buf_gpu", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_temporary, global::get_implicit_function(), "input_buf");
    buffer conv_buf_gpu("conv_buf_gpu", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N1), expr(C_N1)}, p_float32, a_temporary, global::get_implicit_function(),"conv_buf");
    buffer filter_buf_gpu("filter_buf_gpu", {expr(C_FOut), expr(C_FIn), expr(C_K), expr(C_K)}, p_float32, a_temporary, global::get_implicit_function(), "filter_buf");
    buffer bias_buf_gpu("bias_buf_gpu", {expr(C_FOut)}, p_float32, a_temporary, global::get_implicit_function(),"bias_buf");

    input_buf_gpu.tag_gpu_global();
    conv_buf_gpu.tag_gpu_global();
    filter_buf_gpu.tag_gpu_global();
    bias_buf_gpu.tag_gpu_global();

    parameters.store_in(&parameters_buf);
    c_input.store_in(&input_buf_gpu);
    bias.store_in(&bias_buf_gpu);
    filter.store_in(&filter_buf_gpu);
    conv_init.store_in(&conv_buf_gpu);
    conv.store_in(&conv_buf_gpu);

    tiramisu::codegen({&parameters_buf, &input_buf, &filter_buf, &bias_buf, &conv_buf},"fct.o", true);

    return 0;
}

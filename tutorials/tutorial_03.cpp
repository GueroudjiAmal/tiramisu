#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"

/* Halide code for matrix multiplication.
Func matmul(Input A, Input B, Output C) {
    Halide::Func A, B, C;
    Halide::Var x, y;

    A(x,y) = 1;
    B(x,y) = 1;
    C(x,y) = 0;

    Halide::RDom r(0, N);
    C(x,y) = C(x,y) + A(x,r) * B(r,y);

    C.realize(N, N);
}
*/

#define SIZE0 1000

using namespace coli;

int main(int argc, char **argv)
{
    // Set default coli options.
    global::set_default_coli_options();

    /*
     * Declare a function matmul.
     * Declare two arguments (coli buffers) for the function: b_A and b_B
     * Declare an invariant for the function.
     */
    function matmul("matmul");
    buffer b_A("b_A", 2, {coli::expr(SIZE0),coli::expr(SIZE0)}, p_uint8, NULL, a_input, &matmul);
    buffer b_B("b_B", 2, {coli::expr(SIZE0),coli::expr(SIZE0)}, p_uint8, NULL, a_input, &matmul);
    buffer b_C("b_C", 2, {coli::expr(SIZE0),coli::expr(SIZE0)}, p_uint8, NULL, a_output, &matmul);
    invariant p0("N", expr((int32_t) SIZE0), &matmul);

    // Declare a computation c_A that represents a binding to the buffer b_A
    computation c_A("[N]->{c_A[i,j]: 0<=i<N and 0<=j<N}", NULL, false, p_uint8, &matmul);
    // Declare a computation c_B that represents a binding to the buffer b_B
    computation c_B("[N]->{c_B[i,j]: 0<=i<N and 0<=j<N}", NULL, false, p_uint8, &matmul);
    // Declare a computation c_C

    expr e1 = c_A(idx("i"), idx("k")) * c_B(idx("k"), idx("j"));

    computation c_C("[N]->{c_C[i,j,k]: 0<=i<N and 0<=j<N and 0<=k<N}", &e1, true, p_uint8, &matmul);

    // Map the computations to a buffer.
    c_A.set_access("{c_A[i,k]->b_A[i,k]}");
    c_B.set_access("{c_B[k,j]->b_B[k,j]}");
    c_C.set_access("{c_C[i,j,k]->b_C[i,j]}");

    // Set the schedule of each computation.
    // The identity schedule means that the program order is not modified
    // (i.e. no optimization is applied).
    c_C.tile(0,1,32,32);
    c_C.tag_parallel_dimension(0);

    // Set the arguments to blurxy
    matmul.set_arguments({&b_A, &b_B, &b_C});
    // Generate code
    matmul.gen_isl_ast();
    matmul.gen_halide_stmt();
    matmul.gen_halide_obj("build/generated_fct_tutorial_03.o");

    // Some debugging
    matmul.dump_iteration_domain();
    matmul.dump_halide_stmt();

    // Dump all the fields of the blurxy class.
    matmul.dump(true);

    return 0;
}
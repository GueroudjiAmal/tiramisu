#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

//int matmul(halide_buffer_t *b1, halide_buffer_t *b2, halide_buffer_t *b3);
int conv_tiramisu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}  // extern "C"
#endif

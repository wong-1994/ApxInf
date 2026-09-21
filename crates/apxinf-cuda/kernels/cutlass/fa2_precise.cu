// ApxInf-owned forward instantiations of upstream FA2 (BSD-3-Clause).
// Compile without --use_fast_math to preserve the final FP32 division.
// PyTorch SDPA disables the score-scale/subtract FMA before exp2.
#define UNFUSE_FMA
// Distinct template symbols prevent the linker selecting fast-math instantiations.
#undef FLASH_NAMESPACE
#define FLASH_NAMESPACE apxinf_fa2_precise
#include <cuda_runtime.h>
#include <cstring>
#include <cutlass/numeric_types.h>
#include "flash_attn/flash.h"
#include "flash_attn/flash_fwd_launch_template.h"
extern "C" void apxinf_fa2_precise_run(const void* raw_params,cudaStream_t stream){
 FLASH_NAMESPACE::Flash_fwd_params params;
 std::memcpy(&params,raw_params,sizeof(params));
 if(params.d==64) {
  FLASH_NAMESPACE::run_flash_fwd<Flash_fwd_kernel_traits<64,128,128,4,false,false,cutlass::bfloat16_t>,false,false>(params,stream);
 }else{
  FLASH_NAMESPACE::run_flash_fwd<Flash_fwd_kernel_traits<96,128,64,4,false,false,cutlass::bfloat16_t>,false,false>(params,stream);
 }
}

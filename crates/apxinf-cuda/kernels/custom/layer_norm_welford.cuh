// Adapted from PyTorch v2.9.1 aten/src/ATen/native/cuda/layer_norm_kernel.cu.
// Copyright (c) 2016- Facebook, Inc and contributors. BSD-3-Clause.
// Full upstream notice: ../cutlass/licenses/PyTorch-LICENSE.txt.
// BF16-only, forward-only extraction: no ATen dependency or output statistics.
struct WelfordBf16 { float mean, sigma2, count; };
__device__ WelfordBf16 norm_online(float val, WelfordBf16 current) {
  float delta = val - current.mean;
  float count = current.count + 1.f;
  float mean = current.mean + delta * (1.f / count);
  return {mean, current.sigma2 + delta * (val - mean), count};
}
__device__ WelfordBf16 norm_combine(WelfordBf16 dataB, WelfordBf16 dataA) {
  float delta = dataB.mean - dataA.mean;
  float count = dataA.count + dataB.count;
  if (count == 0.f) return {0.f,0.f,0.f};
  float coef = 1.f / count;
  float nA = dataA.count * coef, nB = dataB.count * coef;
  float mean = nA * dataA.mean + nB * dataB.mean;
  float sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
  return {mean,sigma2,count};
}
__global__ void layer_norm_welford_bf16_kernel(
    const __nv_bfloat16* input,const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,__nv_bfloat16* output,int rows,int cols,float eps) {
  __shared__ float shared[6];
  const int lane=threadIdx.x,warp=threadIdx.y,thread=lane+warp*32;
  const int64_t base=static_cast<int64_t>(blockIdx.x)*cols;
  WelfordBf16 wd{0.f,0.f,0.f};
  for(int group=thread;group<cols/4;group+=128) {
    #pragma unroll
    for(int j=0;j<4;++j) wd=norm_online(__bfloat162float(input[base+group*4+j]),wd);
  }
  for(int offset=16;offset>0;offset>>=1) {
    WelfordBf16 other{__shfl_down_sync(0xffffffff,wd.mean,offset),
      __shfl_down_sync(0xffffffff,wd.sigma2,offset),__shfl_down_sync(0xffffffff,wd.count,offset)};
    wd=norm_combine(wd,other);
  }
  for(int offset=2;offset>0;offset/=2) {
    if(lane==0&&warp>=offset&&warp<2*offset) {
      int w=warp-offset;shared[2*w]=wd.mean;shared[2*w+1]=wd.sigma2;shared[4+w]=wd.count;
    }
    __syncthreads();
    if(lane==0&&warp<offset)wd=norm_combine(wd,{shared[2*warp],shared[2*warp+1],shared[4+warp]});
    __syncthreads();
  }
  if(thread==0){shared[0]=wd.mean;shared[1]=wd.sigma2/float(cols);}
  __syncthreads();
  float inverse_std=rsqrtf(shared[1]+eps);
  for(int group=thread;group<cols/4;group+=128) {
    #pragma unroll
    for(int j=0;j<4;++j) {
      int col=group*4+j;
      float value=__bfloat162float(weight[col])*(inverse_std*(__bfloat162float(input[base+col])-shared[0]))+__bfloat162float(bias[col]);
      output[base+col]=__float2bfloat16(value);
    }
  }
}

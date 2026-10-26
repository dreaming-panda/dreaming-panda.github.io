// lane1_repro.cu: a ptxas miscompile of packed-half atomics on shared memory (sm_90).
//
// Each thread loads four halves as ONE 64-bit word and adds them, as two __half2 (or
// __nv_bfloat162), into zeroed shared memory with atomicAdd. The result should be the
// four halves unchanged. With nvcc 13.2 and the default ptxas optimisation level, the
// upper lane of each pair comes out wrong; with -Xptxas -O0, or on global memory, it is right.
//
//   nvcc -arch=sm_90a -o lane1 lane1_repro.cu && ./lane1
//   nvcc -arch=sm_90a -Xptxas -O0 -o lane1_O0 lane1_repro.cu && ./lane1_O0
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

template <class T2> __device__ __forceinline__ T2 as2(uint32_t u) { T2 h; memcpy(&h, &u, 4); return h; }
template <class T2> __device__ __forceinline__ uint32_t bits(T2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }

// Shared memory: the case ptxas gets wrong.
template <class T2>
__global__ void shared_add(const uint64_t* in, uint32_t* out) {
  __shared__ T2 s[2];
  if (threadIdx.x == 0) { s[0] = as2<T2>(0u); s[1] = as2<T2>(0u); }
  __syncthreads();
  uint64_t v = in[0];                                  // four 16-bit values in one 64-bit word
  atomicAdd(&s[0], as2<T2>((uint32_t)v));              // low pair
  atomicAdd(&s[1], as2<T2>((uint32_t)(v >> 32)));      // high pair
  __syncthreads();
  if (threadIdx.x == 0) { out[0] = bits(s[0]); out[1] = bits(s[1]); }
}

// Global memory: the same adds, as a control (native instruction, no emulation loop).
template <class T2>
__global__ void global_add(const uint64_t* in, T2* g, uint32_t* out) {
  uint64_t v = in[0];
  atomicAdd(&g[0], as2<T2>((uint32_t)v));
  atomicAdd(&g[1], as2<T2>((uint32_t)(v >> 32)));
}

template <class T2>
static bool run(const char* name, uint64_t value) {
  uint64_t* d_in; uint32_t* d_out; T2* d_g;
  cudaMalloc(&d_in, 8); cudaMalloc(&d_out, 8); cudaMalloc(&d_g, 8);
  cudaMemcpy(d_in, &value, 8, cudaMemcpyHostToDevice);
  cudaMemset(d_g, 0, 8);
  uint32_t sh[2], gl[2];
  shared_add<T2><<<1, 1>>>(d_in, d_out);
  cudaMemcpy(sh, d_out, 8, cudaMemcpyDeviceToHost);
  global_add<T2><<<1, 1>>>(d_in, d_g, d_out);
  cudaMemcpy(gl, d_g, 8, cudaMemcpyDeviceToHost);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return false; }
  uint32_t lo = (uint32_t)value, hi = (uint32_t)(value >> 32);
  bool ok_sh = sh[0] == lo && sh[1] == hi, ok_gl = gl[0] == lo && gl[1] == hi;
  printf("%-14s expected %08x %08x | shared %08x %08x %s | global %08x %08x %s\n", name, lo, hi,
         sh[0], sh[1], ok_sh ? "PASS" : "FAIL", gl[0], gl[1], ok_gl ? "PASS" : "FAIL");
  cudaFree(d_in); cudaFree(d_out); cudaFree(d_g);
  return ok_sh && ok_gl;
}

int main() {
  // f16:  1.0 = 0x3c00, 2.0 = 0x4000, 3.0 = 0x4200, 4.0 = 0x4400
  // bf16: 1.0 = 0x3f80, 2.0 = 0x4000, 3.0 = 0x4040, 4.0 = 0x4080
  bool ok = run<__half2>("__half2", 0x4400420040003c00ull);
  ok = run<__nv_bfloat162>("__nv_bfloat162", 0x4080404040003f80ull) && ok;
  printf("%s\n", ok ? "all PASS" : "MISCOMPILED: shared-memory result differs from the expected values");
  return ok ? 0 : 1;
}

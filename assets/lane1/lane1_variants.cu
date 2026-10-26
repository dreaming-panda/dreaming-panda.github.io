// lane1_variants.cu: which code patterns hit the shared-memory packed-half atomic miscompile (sm_90)?
//
// Every case adds four halves (1, 2, 3, 4) as two pairs into zeroed memory; the right answer is the four
// values back. Cases 1-4 hit the bug with nvcc 13.2 and the default ptxas optimisation. Cases 5-12 do not.
// Nothing here is misaligned: cases 2-4 vary where the data lives and how it is aligned, and still fail.
//
//   nvcc -arch=sm_90a -o variants lane1_variants.cu && ./variants
//   nvcc -arch=sm_90a -Xptxas -O0 -o variants_O0 lane1_variants.cu && ./variants_O0
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

__device__ __forceinline__ __half2 H2(uint32_t u) { __half2 h; memcpy(&h, &u, 4); return h; }
__device__ __forceinline__ uint32_t U(__half2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
__device__ __forceinline__ __half H1(uint16_t u) { __half h; memcpy(&h, &u, 2); return h; }
__device__ __forceinline__ uint16_t U1(__half h) { uint16_t u; memcpy(&u, &h, 2); return u; }

// The pattern: atomicAdd(__half2*) of the two 32-bit words of one 64-bit integer.
__device__ __forceinline__ void add_split(__half2* p0, __half2* p1, uint64_t v) {
  atomicAdd(p0, H2((uint32_t)v)); atomicAdd(p1, H2((uint32_t)(v >> 32)));
}
#define SHARED2 __shared__ __half2 s[2]; s[0] = H2(0); s[1] = H2(0); __syncthreads()
#define OUT2(a, b) __syncthreads(); out[0] = U(a); out[1] = U(b)

// 1. shared memory, the 64-bit value loaded from global memory
__global__ void k1(const uint64_t* in, uint32_t* out) { SHARED2; add_split(&s[0], &s[1], in[0]); OUT2(s[0], s[1]); }
// 2. shared memory, the 64-bit value is a kernel parameter: no memory load at all
__global__ void k2(uint64_t v, uint32_t* out) { SHARED2; add_split(&s[0], &s[1], v); OUT2(s[0], s[1]); }
// 3. shared targets at byte offsets 4 and 8 of the array
__global__ void k3(const uint64_t* in, uint32_t* out) {
  __shared__ __half2 s[3]; s[0] = s[1] = s[2] = H2(0); __syncthreads();
  add_split(&s[1], &s[2], in[0]); OUT2(s[1], s[2]);
}
// 4. shared targets in two separate 16-byte-aligned arrays
__global__ void k4(const uint64_t* in, uint32_t* out) {
  __shared__ __align__(16) __half2 a[4], b[4]; a[0] = H2(0); b[0] = H2(0); __syncthreads();
  add_split(&a[0], &b[0], in[0]); OUT2(a[0], b[0]);
}
// 5. workaround: load the same 8 bytes as a uint2 (two 32-bit words)
__global__ void k5(const uint2* in, uint32_t* out) {
  SHARED2; uint2 v = in[0]; atomicAdd(&s[0], H2(v.x)); atomicAdd(&s[1], H2(v.y)); OUT2(s[0], s[1]);
}
// 6. the 64-bit integer built from two 32-bit loads
__global__ void k6(const uint32_t* in32, uint32_t* out) {
  SHARED2; add_split(&s[0], &s[1], ((uint64_t)in32[1] << 32) | in32[0]); OUT2(s[0], s[1]);
}
// 7. workaround: a hand-written compare-and-swap loop doing the same __half2 add
__device__ void cas_add(__half2* p, __half2 x) {
  unsigned* a = (unsigned*)p; unsigned old = *a, assumed;
  do { assumed = old; old = atomicCAS(a, assumed, U(__hadd2(H2(assumed), x))); } while (old != assumed);
}
__global__ void k7(const uint64_t* in, uint32_t* out) {
  SHARED2; uint64_t v = in[0];
  cas_add(&s[0], H2((uint32_t)v)); cas_add(&s[1], H2((uint32_t)(v >> 32))); OUT2(s[0], s[1]);
}
// 8. atomicAdd(__half*), one half at a time, split from the 64-bit integer
__global__ void k8(const uint64_t* in, uint32_t* out) {
  __shared__ __half h[4]; for (int i = 0; i < 4; i++) h[i] = H1(0); __syncthreads();
  uint64_t v = in[0];
  for (int i = 0; i < 4; i++) atomicAdd(&h[i], H1((uint16_t)(v >> (16 * i))));
  __syncthreads(); out[0] = U1(h[0]) | ((uint32_t)U1(h[1]) << 16); out[1] = U1(h[2]) | ((uint32_t)U1(h[3]) << 16);
}
// 9. atomicAdd(float*): the same split, two floats (1.5f and 2.5f)
__global__ void k9(const uint64_t* in, uint32_t* out) {
  __shared__ float f[2]; f[0] = f[1] = 0.f; __syncthreads();
  uint64_t v = in[1]; uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32); float a, b;
  memcpy(&a, &lo, 4); memcpy(&b, &hi, 4); atomicAdd(&f[0], a); atomicAdd(&f[1], b);
  __syncthreads(); memcpy(&out[0], &f[0], 4); memcpy(&out[1], &f[1], 4);
}
// 10. no atomic: a plain __hadd2 read-modify-write on shared memory
__global__ void k10(const uint64_t* in, uint32_t* out) {
  SHARED2; uint64_t v = in[0];
  s[0] = __hadd2(s[0], H2((uint32_t)v)); s[1] = __hadd2(s[1], H2((uint32_t)(v >> 32))); OUT2(s[0], s[1]);
}
// 11. no memory: __hadd2 in registers (the zero comes from memory so nothing folds)
__global__ void k11(const uint64_t* in, const uint32_t* zero, uint32_t* out) {
  uint64_t v = in[0]; __half2 z = H2(zero[0]);
  out[0] = U(__hadd2(z, H2((uint32_t)v))); out[1] = U(__hadd2(z, H2((uint32_t)(v >> 32))));
}
// 12. the pattern of case 1 on global memory (a native instruction, no compare-and-swap loop)
__global__ void k12(const uint64_t* in, __half2* g) { add_split(&g[0], &g[1], in[0]); }

int main() {
  const uint64_t halves = 0x4400420040003c00ull;  // 1.0, 2.0, 3.0, 4.0 as halves
  const uint64_t floats = 0x402000003fc00000ull;  // 1.5f, 2.5f
  const uint64_t host[2] = {halves, floats};
  uint64_t* in; uint32_t *out, *zero; __half2* g;
  cudaMalloc(&in, 16); cudaMalloc(&out, 8); cudaMalloc(&zero, 4); cudaMalloc(&g, 8);
  cudaMemcpy(in, host, 16, cudaMemcpyHostToDevice); cudaMemset(zero, 0, 4); cudaMemset(g, 0, 8);
  int fails = 0;
  auto check = [&](const char* name, const void* src, uint64_t want) {
    uint32_t r[2]; cudaMemcpy(r, src, 8, cudaMemcpyDeviceToHost);
    bool ok = r[0] == (uint32_t)want && r[1] == (uint32_t)(want >> 32); fails += !ok;
    printf("%-58s %08x %08x  %s\n", name, r[0], r[1], ok ? "PASS" : "FAIL");
  };
  k1<<<1, 1>>>(in, out);                   check(" 1 shared, 64-bit value loaded from memory", out, halves);
  k2<<<1, 1>>>(halves, out);               check(" 2 shared, 64-bit value is a kernel parameter", out, halves);
  k3<<<1, 1>>>(in, out);                   check(" 3 shared, targets at byte offsets 4 and 8", out, halves);
  k4<<<1, 1>>>(in, out);                   check(" 4 shared, targets in separate 16B-aligned arrays", out, halves);
  k5<<<1, 1>>>((const uint2*)in, out);     check(" 5 shared, loaded as uint2 (workaround)", out, halves);
  k6<<<1, 1>>>((const uint32_t*)in, out);  check(" 6 shared, 64-bit value from two 32-bit loads", out, halves);
  k7<<<1, 1>>>(in, out);                   check(" 7 shared, hand-written atomicCAS loop (workaround)", out, halves);
  k8<<<1, 1>>>(in, out);                   check(" 8 shared, atomicAdd(__half*) one half at a time", out, halves);
  k9<<<1, 1>>>(in, out);                   check(" 9 shared, atomicAdd(float*)", out, floats);
  k10<<<1, 1>>>(in, out);                  check("10 shared, plain __hadd2, no atomic", out, halves);
  k11<<<1, 1>>>(in, zero, out);            check("11 registers, __hadd2", out, halves);
  k12<<<1, 1>>>(in, g);                    check("12 global, same pattern as case 1", g, halves);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
  printf("%d of 12 cases wrong\n", fails);
  return fails ? 1 : 0;
}

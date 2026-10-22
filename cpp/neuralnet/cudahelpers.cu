
#include "../neuralnet/cudahelpers.h"

#include <stdexcept>
#include <string>

#if __CUDA_ARCH__ >= 530
#define CUDA_SUPPORTS_FP16
#endif

//TODO maybe tune this number, it varies by GPU
static const int targetNumThreads = 512;

void splitThreadsAcrossDim01(int dim0Size, int dim1Size, int& threads0, int& blocks0, int& threads1, int& blocks1) {
  if(dim0Size > targetNumThreads) {
    threads0 = targetNumThreads/2;
    blocks0 = (dim0Size + threads0 - 1) / threads0;
    threads1 = 1;
    blocks1 = dim1Size;
  }
  else if(dim0Size > targetNumThreads/2) {
    threads0 = dim0Size;
    blocks0 = 1;
    threads1 = 1;
    blocks1 = dim1Size;
  }
  else {
    threads0 = dim0Size;
    blocks0 = 1;
    threads1 = targetNumThreads / dim0Size;
    blocks1 = (dim1Size + threads1 - 1) / threads1;
  }
}

__forceinline__ __device__ float mishf(float a) {
  return a * tanhf(a < 20.0f ? log1pf(expf(a)) : a);
}

#ifdef CUDA_SUPPORTS_FP16
__forceinline__ __device__ half mishh(half h) {
  float a = __half2float(h);
  return __float2half(a * tanhf(a < 20.0f ? log1pf(expf(a)) : a));
}
#endif

__forceinline__ __device__ float siluf(float a) {
  return a / (1.0f + expf(-a));
}

#ifdef CUDA_SUPPORTS_FP16
__forceinline__ __device__ half siluh(half h) {
  float a = __half2float(h);
  return __float2half(a / (1.0f + expf(-a)));
}
#endif

//--------------------------------------------------------------------------------------------------------------

template <typename T>
__global__
void channelConcatKernel(
  const T* inA,
  const T* inB,
  T* out,
  int chwA,
  int chwB,
  int numBlocksA,
  int numBlocksB,
  int n
) {
  if(blockIdx.x < numBlocksA) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < chwA) {
      int nchwA = n*chwA;
      int chwOut = (chwA+chwB);

      int aIdx = index;
      int outIdx = index;
      while(aIdx < nchwA) {
        out[outIdx] = inA[aIdx];
        aIdx += chwA;
        outIdx += chwOut;
      }
    }
  }
  else {
    int index = (blockIdx.x - numBlocksA) * blockDim.x + threadIdx.x;
    if(index < chwB) {
      int nchwB = n*chwB;
      int chwOut = (chwA+chwB);

      int bIdx = index;
      int outIdx = chwA+index;
      while(bIdx < nchwB) {
        out[outIdx] = inB[bIdx];
        bIdx += chwB;
        outIdx += chwOut;
      }
    }
  }
}

template <typename T>
void customCudaChannelConcatTemplate(const T* inA, const T* inB, T* out, int chwA, int chwB, int n) {
  int blockSize = targetNumThreads;
  int numBlocksA = (chwA + blockSize-1) / blockSize;
  int numBlocksB = (chwB + blockSize-1) / blockSize;
  int numBlocks = numBlocksA + numBlocksB;
  channelConcatKernel<<<numBlocks, blockSize>>>(inA,inB,out,chwA,chwB,numBlocksA,numBlocksB,n);
}
template void customCudaChannelConcatTemplate<float>(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
template void customCudaChannelConcatTemplate<half>(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n) {
  customCudaChannelConcatTemplate<float>(inA,inB,out,chwA,chwB,n);
}
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n) {
  customCudaChannelConcatTemplate<half>(inA,inB,out,chwA,chwB,n);
}

//--------------------------------------------------------------------------------------------------------------

template <typename T>
__global__
void extractChannel0KernelNHWC(const T *in, T* out, int nhwSize, int cSize)
{
  int nhwIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(nhwIdx < nhwSize) {
    out[nhwIdx] = in[nhwIdx*cSize];
  }
}
template <typename T>
void customCudaChannel0ExtractNHWCTemplate(const T *in, T* out, int n, int hw, int c) {
  int nhw = n*hw;
  int blockSize = targetNumThreads;
  int numBlocks = (nhw+blockSize-1)/blockSize;
  extractChannel0KernelNHWC<<<numBlocks,blockSize>>>(in,out,nhw,c);
}

template <typename T>
__global__
void extractChannel0KernelNCHW(const T *in, T* out, int nSize, int cSize, int hwSize)
{
  int hwIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(hwIdx < hwSize && nIdx < nSize) {
    out[nIdx * hwSize + hwIdx] = in[nIdx * cSize * hwSize + hwIdx];
  }
}
template <typename T>
void customCudaChannel0ExtractNCHWTemplate(const T *in, T* out, int nSize, int cSize, int hwSize) {
  int hwThreads;
  int hwBlocks;
  int nThreads;
  int nBlocks;
  splitThreadsAcrossDim01(hwSize, nSize, hwThreads, hwBlocks, nThreads, nBlocks);

  if(nBlocks > 65536)
    throw std::runtime_error("customCudaChannel0ExtractNCHW: nSize too large given hwSize");

  dim3 grid(hwBlocks,nBlocks,1);
  dim3 threads(hwThreads,nThreads,1);
  extractChannel0KernelNCHW<<<grid,threads>>>(in,out,nSize,cSize,hwSize);
}

void customCudaChannel0ExtractNCHW(const float* in, float* out, int n, int c, int hw) {
  customCudaChannel0ExtractNCHWTemplate<float>(in,out,n,c,hw);
}
void customCudaChannel0ExtractNCHW(const half* in, half* out, int n, int c, int hw) {
  customCudaChannel0ExtractNCHWTemplate<half>(in,out,n,c,hw);
}
void customCudaChannel0ExtractNHWC(const float* in, float* out, int n, int hw, int c) {
  customCudaChannel0ExtractNHWCTemplate<float>(in,out,n,hw,c);
}
void customCudaChannel0ExtractNHWC(const half* in, half* out, int n, int hw, int c) {
  customCudaChannel0ExtractNHWCTemplate<half>(in,out,n,hw,c);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void sumChannelsNCHWKernel(const float* in, float* out, int cSize, int xySize, float scaleSum)
{
  extern __shared__ float sumPoolNCHWShared[];
  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  float acc = 0.0f;
  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      acc += in[xyIdx + cIdx * xySize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
    sumPoolNCHWShared[sharedIdx] = acc;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNCHWShared[sharedIdx] += sumPoolNCHWShared[sharedIdx + s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize)
    out[cIdx + nIdx * cSize] = sumPoolNCHWShared[sharedIdx] * scaleSum;
}
__global__
void valueHeadPoolChannelsNCHWKernel(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum)
{
  extern __shared__ float sumPoolNCHWShared[];
  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  float acc = 0.0f;
  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      acc += in[xyIdx + cIdx * xySize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
    sumPoolNCHWShared[sharedIdx] = acc;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNCHWShared[sharedIdx] += sumPoolNCHWShared[sharedIdx + s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumPoolNCHWShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;
    out[cIdx + nIdx * cSize*3] = mean;
    out[cIdx + nIdx * cSize*3 + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * cSize*3 + cSize*2] = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
  }
}
__global__
void gPoolChannelsNCHWKernel(const float* in, float* out, int cSize, int xySize, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[xyIdx + cIdx * xySize + nIdx * xycSize];
      accSum += a;
      accMax = fmaxf(accMax, a);
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}
__global__
void gPoolChannelsNCHWMaskKernel(const float* in, float* out, int cSize, int xySize, const float* mask, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[xyIdx + cIdx * xySize + nIdx * xycSize];
      accSum += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      // We're relying on all padded space being equal to 0 because this gpool only ever follows a BN+Activate with a mask.
      accMax = fmaxf(accMax, a + (mask[xyIdx + nIdx * xySize] - 1.0f));
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}

void customCudaPoolRowsSumNCHW(const float* in, float* out, int nSize, int cSize, int xySize, float scaleSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  sumChannelsNCHWKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,scaleSum);
}
void customCudaValueHeadPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  valueHeadPoolChannelsNCHWKernel<<<grid,threads,sharedMemSize>>>(in,out,nSize,cSize,xySize,maskSum);
}
void customCudaPoolRowsGPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  if(mask != NULL)
    gPoolChannelsNCHWMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNCHWKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,maskSum,sharedMemElts);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void gPoolChannelsNCHWHalfKernel(const half* in, half* out, int cSize, int xySize, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[xyIdx + cIdx * xySize + nIdx * xycSize]);
      accSum += a;
      accMax = fmaxf(accMax, a);
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void gPoolChannelsNCHWHalfMaskKernel(const half* in, half* out, int cSize, int xySize, const half* mask, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[xyIdx + cIdx * xySize + nIdx * xycSize]);
      accSum += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      accMax = fmaxf(accMax, a + (__half2float(mask[xyIdx + nIdx * xySize]) - 1.0f));
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaPoolRowsGPoolNCHW(const half* in, half* out, int nSize, int cSize, int xySize, const half* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  if(mask != NULL)
    gPoolChannelsNCHWHalfMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNCHWHalfKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,maskSum,sharedMemElts);
}



//--------------------------------------------------------------------------------------------------------------

__global__
void sumChannelsNHWCKernel(const float* in, float* out, int xySize, int cSize, float scaleSum)
{
  extern __shared__ float sumPoolNHWCShared[];
  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumPoolNHWCShared[sharedIdx] = 0;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      sumPoolNHWCShared[sharedIdx] += in[cIdx + xyIdx * cSize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNHWCShared[sharedIdx] += sumPoolNHWCShared[sharedIdx + cBlockDim * s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize)
    out[cIdx + nIdx * cSize] = sumPoolNHWCShared[sharedIdx] * scaleSum;
}
__global__
void valueHeadPoolChannelsNHWCKernel(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum)
{
  extern __shared__ float sumPoolNHWCShared[];
  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumPoolNHWCShared[sharedIdx] = 0;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      sumPoolNHWCShared[sharedIdx] += in[cIdx + xyIdx * cSize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNHWCShared[sharedIdx] += sumPoolNHWCShared[sharedIdx + cBlockDim * s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumPoolNHWCShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;
    out[cIdx + nIdx * cSize*3] = mean;
    out[cIdx + nIdx * cSize*3 + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * cSize*3 + cSize*2] = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
  }
}
__global__
void gPoolChannelsNHWCKernel(const float* in, float* out, int xySize, int cSize, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[cIdx + xyIdx * cSize + nIdx * xycSize];
      sumShared[sharedIdx] += a;
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a);
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}
__global__
void gPoolChannelsNHWCMaskKernel(const float* in, float* out, int xySize, int cSize, const float* mask, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[cIdx + xyIdx * cSize + nIdx * xycSize];
      sumShared[sharedIdx] += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a + (mask[xyIdx + nIdx * xySize] - 1.0f));
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}


void customCudaPoolRowsSumNHWC(const float* in, float* out, int nSize, int xySize, int cSize, float scaleSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  sumChannelsNHWCKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,scaleSum);
}

void customCudaValueHeadPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  valueHeadPoolChannelsNHWCKernel<<<grid,threads,sharedMemSize>>>(in,out,nSize,xySize,cSize,maskSum);
}

void customCudaPoolRowsGPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  if(mask != NULL)
    gPoolChannelsNHWCMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNHWCKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,maskSum,sharedMemElts);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void gPoolChannelsNHWCHalfKernel(const half* in, half* out, int xySize, int cSize, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[cIdx + xyIdx * cSize + nIdx * xycSize]);
      sumShared[sharedIdx] += a;
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a);
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void gPoolChannelsNHWCHalfMaskKernel(const half* in, half* out, int xySize, int cSize, const half* mask, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[cIdx + xyIdx * cSize + nIdx * xycSize]);
      sumShared[sharedIdx] += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a + (__half2float(mask[xyIdx + nIdx * xySize]) - 1.0f));
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaPoolRowsGPoolNHWC(const half* in, half* out, int nSize, int xySize, int cSize, const half* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  if(mask != NULL)
    gPoolChannelsNHWCHalfMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNHWCHalfKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,maskSum,sharedMemElts);
}


//--------------------------------------------------------------------------------------------------------------

__global__
void copyToHalfKernel(const float *in, half* out, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    out[idx] = __float2half(in[idx]);
  }
}
__global__
void copyFromHalfKernel(const half *in, float* out, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    out[idx] = __half2float(in[idx]);
  }
}

void customCudaCopyToHalf(const float* in, half* out, int n) {
  int blockSize = targetNumThreads;
  int numBlocks = (n+blockSize-1)/blockSize;
  copyToHalfKernel<<<numBlocks, blockSize>>>(in,out,n);
}
void customCudaCopyFromHalf(const half* in, float* out, int n) {
  int blockSize = targetNumThreads;
  int numBlocks = (n+blockSize-1)/blockSize;
  copyFromHalfKernel<<<numBlocks, blockSize>>>(in,out,n);
}

//--------------------------------------------------------------------------------------------------------------


__global__
void addTensorInplaceHalfKernel(half *buf, const half* biases, int nSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < nSize) {
    buf[idx] = __hadd(buf[idx],biases[idx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
void customCudaAddTensorInplace(half* buf, const half* biases, int nSize) {
  int blockSize = targetNumThreads;
  int numBlocks = (nSize+blockSize-1)/blockSize;
  addTensorInplaceHalfKernel<<<numBlocks, blockSize>>>(buf,biases,nSize);
}

//--------------------------------------------------------------------------------------------------------------


__global__
void addCBiasInplaceNCKernel(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = buf[idx] + biases[cIdx];
  }
}
__global__
void addCBiasInplaceNCHalfKernel(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = __hadd(buf[idx],biases[cIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void addCBiasInplaceNCKernelRelu(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = fmaxf(buf[idx] + biases[cIdx],0.0f);
  }
}
__global__
void addCBiasInplaceNCHalfKernelRelu(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    const half halfzero = __float2half(0.0f);
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void addCBiasInplaceNCKernelMish(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = mishf(buf[idx] + biases[cIdx]);
  }
}
__global__
void addCBiasInplaceNCKernelSilu(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = siluf(buf[idx] + biases[cIdx]);
  }
}
__global__
void addCBiasInplaceNCHalfKernelMish(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void addCBiasInplaceNCHalfKernelSilu(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = siluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedAddCBiasInplaceNC(void* buf, const void* biases, int nSize, int cSize, bool isHalf, int activation) {
  int cThreads;
  int cBlocks;
  int nThreads;
  int nBlocks;
  splitThreadsAcrossDim01(cSize, nSize, cThreads, cBlocks, nThreads, nBlocks);

  if(nBlocks > 65536)
    throw std::runtime_error("customCudaAddCBiasInplaceNC: nSize too large given cSize");

  dim3 grid(cBlocks,nBlocks,1);
  dim3 threads(cThreads,nThreads,1);

  if(activation == ACTIVATION_IDENTITY) {
    if(isHalf)
      addCBiasInplaceNCHalfKernel<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernel<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else if(activation == ACTIVATION_RELU) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelRelu<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelRelu<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else if(activation == ACTIVATION_MISH) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelMish<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelMish<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else if(activation == ACTIVATION_SILU) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelSilu<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelSilu<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else {
    throw std::runtime_error("customCudaAddCBiasInplaceNC: unsupported activation");
  }
}

void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int nSize, int cSize, int activation) {
  sharedAddCBiasInplaceNC(buf,biases,nSize,cSize,false,activation);
}
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int nSize, int cSize, int activation) {
  sharedAddCBiasInplaceNC(buf,biases,nSize,cSize,true,activation);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void addNCBiasInplaceNCHWKernel(float *buf, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = ncIdx * sSize + sIdx;
    buf[idx] = buf[idx] + biases[ncIdx];
  }
}
__global__
void addNCBiasInplaceNCHWHalfKernel(half *buf, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = ncIdx * sSize + sIdx;
    buf[idx] = __hadd(buf[idx],biases[ncIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedAddNCBiasInplaceNCHW(void *buf, const void* biases, int nSize, int cSize, int xySize, bool isHalf) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNCHW: cSize too large");

  int sSize = xySize;
  int sThreads;
  int sBlocks;
  int cThreads;
  int cBlocks;
  splitThreadsAcrossDim01(sSize, cSize, sThreads, sBlocks, cThreads, cBlocks);

  dim3 grid(sBlocks,cBlocks,nSize);
  dim3 threads(sThreads,cThreads,1);
  if(isHalf)
    addNCBiasInplaceNCHWHalfKernel<<<grid,threads>>>((half*)buf,(const half*)biases,cSize,sSize);
  else
    addNCBiasInplaceNCHWKernel<<<grid,threads>>>((float*)buf,(const float*)biases,cSize,sSize);
}

void customCudaAddNCBiasInplaceNCHW(float *buf, const float* biases, int nSize, int cSize, int xySize) {
  sharedAddNCBiasInplaceNCHW(buf,biases,nSize,cSize,xySize,false);
}
void customCudaAddNCBiasInplaceNCHW(half *buf, const half* biases, int nSize, int cSize, int xySize) {
  sharedAddNCBiasInplaceNCHW(buf,biases,nSize,cSize,xySize,true);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void addNCBiasInplaceNHWCKernel(float *buf, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    buf[idx] = buf[idx] + biases[ncIdx];
  }
}
__global__
void addNCBiasInplaceNHWCHalfKernel(half *buf, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    buf[idx] = __hadd(buf[idx],biases[ncIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedAddNCBiasInplaceNHWC(void *buf, const void* biases, int nSize, int xySize, int cSize, bool isHalf) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNHWC: nSize too large");
  if(xySize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNHWC: xySize too large");

  int sSize = xySize;
  int cThreads;
  int cBlocks;
  int sThreads;
  int sBlocks;
  splitThreadsAcrossDim01(cSize, sSize, cThreads, cBlocks, sThreads, sBlocks);

  dim3 grid(cBlocks,sBlocks,nSize);
  dim3 threads(cThreads,sThreads,1);
  if(isHalf)
    addNCBiasInplaceNHWCHalfKernel<<<grid,threads>>>((half*)buf,(const half*)biases,sSize,cSize);
  else
    addNCBiasInplaceNHWCKernel<<<grid,threads>>>((float*)buf,(const float*)biases,sSize,cSize);
}

void customCudaAddNCBiasInplaceNHWC(float *buf, const float* biases, int nSize, int xySize, int cSize) {
  sharedAddNCBiasInplaceNHWC(buf,biases,nSize,xySize,cSize,false);
}
void customCudaAddNCBiasInplaceNHWC(half *buf, const half* biases, int nSize, int xySize, int cSize) {
  sharedAddNCBiasInplaceNHWC(buf,biases,nSize,xySize,cSize,true);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void applyCScaleBiasNCHWKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = in[idx] * scale[cIdx] + biases[cIdx];
  }
}
__global__
void applyCScaleBiasNCHWReluKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f);
  }
}
__global__
void applyCScaleBiasNCHWMishKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNCHWSiluKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNCHWMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = (in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWReluMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWMishMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWSiluMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = __hfma(in[idx],scale[cIdx],biases[cIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWReluHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWMishHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWSiluHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = siluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWReluMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWMishMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWSiluMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = siluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedApplyCScaleBiasNCHW(const void* in, void* out, const void* scale, const void* biases, const void* mask, int nSize, int cSize, int xySize, bool isHalf, int activation) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNCHW: cSize too large");

  int sSize = xySize;
  int sThreads;
  int sBlocks;
  int cThreads;
  int cBlocks;
  splitThreadsAcrossDim01(sSize, cSize, sThreads, sBlocks, cThreads, cBlocks);

  dim3 grid(sBlocks,cBlocks,nSize);
  dim3 threads(sThreads,cThreads,1);
  if(mask == NULL) {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNCHWHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNCHWReluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWReluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNCHWMishHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWMishKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNCHWSiluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWSiluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNCHW: unsupported activation");
    }
  }
  else {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNCHWMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNCHWReluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWReluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNCHWMishMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWMishMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNCHWSiluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWSiluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNCHW: unsupported activation");
    }
  }
}

void customCudaApplyCScaleBiasNCHW(const float* in, float* out, const float* scale, const float* biases, const float* mask, int nSize, int cSize, int xySize, int activation) {
  sharedApplyCScaleBiasNCHW(in,out,scale,biases,mask,nSize,cSize,xySize,false,activation);
}
void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, const half* mask, int nSize, int cSize, int xySize, int activation) {
  sharedApplyCScaleBiasNCHW(in,out,scale,biases,mask,nSize,cSize,xySize,true,activation);
}


//--------------------------------------------------------------------------------------------------------------

__global__
void applyCScaleBiasNHWCKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = in[idx] * scale[cIdx] + biases[cIdx];
  }
}
__global__
void applyCScaleBiasNHWCReluKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f);
  }
}
__global__
void applyCScaleBiasNHWCMishKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNHWCSiluKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNHWCMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = (in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCReluMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCMishMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCSiluMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = __hfma(in[idx],scale[cIdx],biases[cIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCReluHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCMishHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCSiluHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = siluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCReluMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCMishMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCSiluMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = siluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedApplyCScaleBiasNHWC(const void* in, void* out, const void* scale, const void* biases, const void* mask, int nSize, int xySize, int cSize, bool isHalf, int activation) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNHWC: nSize too large");
  if(xySize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNHWC: xySize too large");

  int sSize = xySize;
  int cThreads;
  int cBlocks;
  int sThreads;
  int sBlocks;
  splitThreadsAcrossDim01(cSize, sSize, cThreads, cBlocks, sThreads, sBlocks);

  dim3 grid(cBlocks,sBlocks,nSize);
  dim3 threads(cThreads,sThreads,1);
  if(mask == NULL) {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNHWCHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNHWCReluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCReluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNHWCMishHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCMishKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNHWCSiluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCSiluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNHWC: unsupported activation");
    }
  }
  else {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNHWCMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNHWCReluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCReluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNHWCMishMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCMishMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNHWCSiluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCSiluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNHWC: unsupported activation");
    }
  }
}

void customCudaApplyCScaleBiasNHWC(const float* in, float* out, const float* scale, const float* biases, const float* mask, int nSize, int xySize, int cSize, int activation) {
  sharedApplyCScaleBiasNHWC(in,out,scale,biases,mask,nSize,xySize,cSize,false,activation);
}
void customCudaApplyCScaleBiasNHWC(const half* in, half* out, const half* scale, const half* biases, const half* mask, int nSize, int xySize, int cSize, int activation) {
  sharedApplyCScaleBiasNHWC(in,out,scale,biases,mask,nSize,xySize,cSize,true,activation);
}

//--------------------------------------------------------------------------------------------------------------
// Transformer kernels, adapted from upstream KataGo v1.17.0. NHWC only: activations are [n, xy, c], so a
// [n, xy, heads*headDim] projection output is directly the BSHD layout of the attention kernel.
// Everything computes in fp32, half is only the storage type.

__device__ __forceinline__ float toFloatT(float x) { return x; }
__device__ __forceinline__ float toFloatT(half x) { return __half2float(x); }
__device__ __forceinline__ void storeFloatT(float* p, size_t idx, float v) { p[idx] = v; }
__device__ __forceinline__ void storeFloatT(half* p, size_t idx, float v) { p[idx] = __float2half(v); }
// The backend passes a null mask when the board fills the whole buffer (requireExactNNLen)
template<typename T>
__device__ __forceinline__ float maskAt(const T* mask, size_t idx) { return mask != NULL ? toFloatT(mask[idx]) : 1.0f; }

__device__ __forceinline__ float applyActivationf(float v, int activation) {
  if(activation == ACTIVATION_RELU)
    return fmaxf(v, 0.0f);
  if(activation == ACTIVATION_MISH)
    return mishf(v);
  if(activation == ACTIVATION_SILU)
    return siluf(v);
  return v;
}

// Spatial RMSNorm: one RMS per sample over all channels and on-board positions.
// Pass 1: one block per sample writes the masked sum of squares. Pass 2: one block per (n, xy) applies it.
template<typename T>
__global__
void spatialRMSNormSumSqNHWCKernel(const T* in, const T* mask, float* sumSqOut, int xySize, int cSize) {
  extern __shared__ float sumShared[];
  int n = blockIdx.x;
  int tid = threadIdx.x;
  int total = xySize * cSize;
  const T* inN = in + (size_t)n * total;
  float acc = 0.0f;
  for(int i = tid; i < total; i += blockDim.x) {
    float val = toFloatT(inN[i]) * maskAt(mask, n * xySize + i / cSize);
    acc += val * val;
  }
  sumShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s)
      sumShared[tid] += sumShared[tid + s];
    __syncthreads();
  }
  if(tid == 0)
    sumSqOut[n] = sumShared[0];
}

template<typename T>
__global__
void spatialRMSNormApplyNHWCKernel(
  const T* in, T* out, const T* gamma, const T* beta, const T* mask, const float* maskSum, const float* sumSq,
  int xySize, int cSize, float epsilon, int activation
) {
  int pos = blockIdx.x;
  int n = pos / xySize;
  float maskVal = maskAt(mask, pos);
  float rms = rsqrtf(sumSq[n] / fmaxf(1.0f, maskSum[n] * (float)cSize) + epsilon);
  for(int c = threadIdx.x; c < cSize; c += blockDim.x) {
    size_t idx = (size_t)pos * cSize + c;
    float val = applyActivationf(toFloatT(in[idx]) * rms * toFloatT(gamma[c]) + toFloatT(beta[c]), activation);
    storeFloatT(out, idx, val * maskVal);
  }
}

template<typename T>
static void launchSpatialRMSNormNHWC(
  const T* in, T* out, const T* gamma, const T* beta, const T* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf
) {
  if(nSize <= 0)
    return;
  int threads = targetNumThreads;
  spatialRMSNormSumSqNHWCKernel<T><<<nSize, threads, threads * sizeof(float)>>>(in, mask, sumSqBuf, xySize, cSize);
  int applyThreads = 32;
  while(applyThreads < cSize && applyThreads < targetNumThreads)
    applyThreads *= 2;
  spatialRMSNormApplyNHWCKernel<T><<<nSize * xySize, applyThreads>>>(in, out, gamma, beta, mask, maskSum, sumSqBuf, xySize, cSize, epsilon, activation);
}
void customCudaSpatialRMSNormNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf
) {
  launchSpatialRMSNormNHWC<float>(in, out, gamma, beta, mask, maskSum, nSize, xySize, cSize, epsilon, activation, sumSqBuf);
}
void customCudaSpatialRMSNormNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf
) {
  launchSpatialRMSNormNHWC<half>(in, out, gamma, beta, mask, maskSum, nSize, xySize, cSize, epsilon, activation, sumSqBuf);
}

// Learnable 2D RoPE in place on a BSHD buffer [n, xy, numBufHeads*headDim], recomputing the angles
// x*freqX + y*freqY from freqs (numKVHeads, numPairs, 2). Buffer head h uses kv head h*numKVHeads/numBufHeads.
// One block per (xy, n), one thread per (head, pair).
template<typename T>
__global__
void applyRoPENHWCKernel(
  T* buf, const float* freqs, int seqLen, int numBufHeads, int numKVHeads, int headDim, int numPairs, int nnXLen
) {
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  if(xy >= seqLen || hp >= numBufHeads * numPairs)
    return;
  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int totalDim = numBufHeads * headDim;
  size_t idx0 = (size_t)(n * seqLen + xy) * totalDim + h * headDim + 2 * pairIdx;
  size_t idx1 = idx0 + 1;

  int kvh = h * numKVHeads / numBufHeads;
  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float sinVal, cosVal;
  sincosf(angle, &sinVal, &cosVal);

  float x0 = toFloatT(buf[idx0]);
  float x1 = toFloatT(buf[idx1]);
  storeFloatT(buf, idx0, x0 * cosVal - x1 * sinVal);
  storeFloatT(buf, idx1, x0 * sinVal + x1 * cosVal);
}

template<typename T>
static void launchRoPENHWC(
  T* buf, const float* freqs, int batchSize, int seqLen, int numBufHeads, int numKVHeads, int headDim, int nnXLen
) {
  int numPairs = headDim / 2;
  int totalHP = numBufHeads * numPairs;
  if(totalHP > 1024)
    throw std::runtime_error("customCudaApplyRoPENHWC: numHeads*headDim/2 exceeds 1024 threads per block");
  if(batchSize > 65535)
    throw std::runtime_error("customCudaApplyRoPENHWC: batchSize too large");
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize, 1);
  applyRoPENHWCKernel<T><<<blocks, threads>>>(buf, freqs, seqLen, numBufHeads, numKVHeads, headDim, numPairs, nnXLen);
}
void customCudaApplyRoPENHWC(
  float* buf, const float* freqs, int batchSize, int seqLen, int numBufHeads, int numKVHeads, int headDim, int nnXLen
) {
  launchRoPENHWC<float>(buf, freqs, batchSize, seqLen, numBufHeads, numKVHeads, headDim, nnXLen);
}
void customCudaApplyRoPENHWC(
  half* buf, const float* freqs, int batchSize, int seqLen, int numBufHeads, int numKVHeads, int headDim, int nnXLen
) {
  launchRoPENHWC<half>(buf, freqs, batchSize, seqLen, numBufHeads, numKVHeads, headDim, nnXLen);
}

// Scaled dot product attention with online softmax, tiled over keys (upstream flashAttentionTiledImpl).
// Grid: (numQGroups, batchSize * numHeads), block: BLOCK_Q threads, each thread owns Q_PER_THREAD queries.
// Off-board keys are skipped and off-board queries give 0.
template<int qHeadDim, int vHeadDim, int BLOCK_Q, int BLOCK_KV, int Q_PER_THREAD, typename T>
__global__
void flashAttentionNHWCKernel(
  const T* Q, const T* K, const T* V, const T* mask, T* output,
  int seqLen, int numHeads, int numKVHeads, float scale
) {
  const int tid = threadIdx.x;
  const int qBlockStart = blockIdx.x * (BLOCK_Q * Q_PER_THREAD);
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h * numKVHeads / numHeads;

  const int qTotalDim = numHeads * qHeadDim;
  const int kTotalDim = numKVHeads * qHeadDim;
  const int vTotalDim = numKVHeads * vHeadDim;
  const int oTotalDim = numHeads * vHeadDim;

  __shared__ float kTile[BLOCK_KV * qHeadDim];
  __shared__ float vTile[BLOCK_KV * vHeadDim];
  __shared__ float kMaskTile[BLOCK_KV];

  float qReg[Q_PER_THREAD * qHeadDim];
  float qMask[Q_PER_THREAD];
  float runningMax[Q_PER_THREAD];
  float runningSum[Q_PER_THREAD];
  float acc[Q_PER_THREAD * vHeadDim];

  #pragma unroll
  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    qMask[qi] = 0.0f;
    if(qPos < seqLen) {
      qMask[qi] = maskAt(mask, n * seqLen + qPos);
      if(qMask[qi] != 0.0f) {
        const T* qPtr = Q + ((size_t)n * seqLen + qPos) * qTotalDim + h * qHeadDim;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d++)
          qReg[qi * qHeadDim + d] = toFloatT(qPtr[d]) * scale;
      }
    }
    runningMax[qi] = -1e30f;
    runningSum[qi] = 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++)
      acc[qi * vHeadDim + d] = 0.0f;
  }

  for(int kvStart = 0; kvStart < seqLen; kvStart += BLOCK_KV) {
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int tileKPos = t / qHeadDim;
      int tileD = t % qHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = toFloatT(K[((size_t)n * seqLen + globalKPos) * kTotalDim + kvh * qHeadDim + tileD]);
      kTile[tileKPos * qHeadDim + tileD] = v;
    }
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int tileKPos = t / vHeadDim;
      int tileD = t % vHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = toFloatT(V[((size_t)n * seqLen + globalKPos) * vTotalDim + kvh * vHeadDim + tileD]);
      vTile[tileKPos * vHeadDim + tileD] = v;
    }
    for(int t = tid; t < BLOCK_KV; t += BLOCK_Q) {
      int globalKPos = kvStart + t;
      kMaskTile[t] = globalKPos < seqLen ? maskAt(mask, n * seqLen + globalKPos) : 0.0f;
    }
    __syncthreads();

    int kvEnd = min(BLOCK_KV, seqLen - kvStart);
    #pragma unroll
    for(int qi = 0; qi < Q_PER_THREAD; qi++) {
      int qPos = qBlockStart + qi * BLOCK_Q + tid;
      if(qPos >= seqLen || qMask[qi] == 0.0f)
        continue;
      for(int tk = 0; tk < kvEnd; tk++) {
        if(kMaskTile[tk] == 0.0f)
          continue;
        float dot = 0.0f;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d++)
          dot += qReg[qi * qHeadDim + d] * kTile[tk * qHeadDim + d];
        float newMax = fmaxf(runningMax[qi], dot);
        float expOldMax = __expf(runningMax[qi] - newMax);
        float expCur = __expf(dot - newMax);
        #pragma unroll
        for(int d = 0; d < vHeadDim; d++)
          acc[qi * vHeadDim + d] = acc[qi * vHeadDim + d] * expOldMax + expCur * vTile[tk * vHeadDim + d];
        runningSum[qi] = runningSum[qi] * expOldMax + expCur;
        runningMax[qi] = newMax;
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    if(qPos >= seqLen)
      continue;
    T* outPtr = output + ((size_t)n * seqLen + qPos) * oTotalDim + h * vHeadDim;
    float invSum = (qMask[qi] != 0.0f && runningSum[qi] > 0.0f) ? (1.0f / runningSum[qi]) : 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++)
      storeFloatT(outPtr, d, acc[qi * vHeadDim + d] * invSum);
  }
}

template<typename T>
static void launchFlashAttentionNHWC(
  const T* Q, const T* K, const T* V, const T* mask, T* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim
) {
  if(batchSize * numHeads > 65535)
    throw std::runtime_error("customCudaAttentionNHWC: batchSize * numHeads too large");
  float scale = 1.0f / sqrtf((float)qHeadDim);
  constexpr int BQ = 128;
  constexpr int BKV = 32;
  dim3 grid((seqLen + BQ - 1) / BQ, batchSize * numHeads);
#define KATAGO_LAUNCH_ATTN(QD, VD) \
  flashAttentionNHWCKernel<(QD), (VD), BQ, BKV, 1, T><<<grid, BQ>>>(Q, K, V, mask, output, seqLen, numHeads, numKVHeads, scale)
  //Larger head dims would need the per-thread query and accumulator arrays restructured to avoid register spills
  if(qHeadDim == 32 && vHeadDim == 32) KATAGO_LAUNCH_ATTN(32, 32);
  else if(qHeadDim == 16 && vHeadDim == 16) KATAGO_LAUNCH_ATTN(16, 16);
  else if(qHeadDim == 64 && vHeadDim == 64) KATAGO_LAUNCH_ATTN(64, 64);
  else if(qHeadDim == 32 && vHeadDim == 16) KATAGO_LAUNCH_ATTN(32, 16);
  else if(qHeadDim == 32 && vHeadDim == 64) KATAGO_LAUNCH_ATTN(32, 64);
  else if(qHeadDim == 64 && vHeadDim == 32) KATAGO_LAUNCH_ATTN(64, 32);
  else
    throw std::runtime_error("customCudaAttentionNHWC: unsupported (qHeadDim, vHeadDim) = (" + std::to_string(qHeadDim) + ", " + std::to_string(vHeadDim) + ")");
#undef KATAGO_LAUNCH_ATTN
}
void customCudaAttentionNHWC(
  const float* Q, const float* K, const float* V, const float* mask, float* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim
) {
  launchFlashAttentionNHWC<float>(Q, K, V, mask, output, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim);
}
void customCudaAttentionNHWC(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim
) {
  launchFlashAttentionNHWC<half>(Q, K, V, mask, output, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim);
}

//--------------------------------------------------------------------------------------------------------------
// Macros the upstream kernels below are written against.
#ifdef CUDA_SUPPORTS_FP16
#define KATAGO_GPU_SUPPORTS_FP16
#endif
#define KATAGO_GPU_CUDA 1
// Upstream uses the fast __sincosf here. The accurate sincosf keeps fp32 results within ~1e-7 of python.
#define KATAGO_GPU_SINCOSF sincosf
#include "../neuralnet/cudaflashmma.cuh"
#define KATAGO_HAS_FLASH_MMA 1

//--------------------------------------------------------------------------------------------------------------
// Copied from upstream KataGo cpp/neuralnet/cudaandrocmhelpers.inc at 231e1c4b: RMSNorm, fused Q+K learnable RoPE,
// tensor-core flash attention wrappers and SwiGLU. Unchanged; callers pass stream 0 (this backend uses the default stream).

static void checkBufferIndexFitsInt(int64_t a, int64_t b, int64_t c, const char* whatKernel) {
  int64_t total = a * b * c;
  if(total >= (int64_t)2147483647)  // INT_MAX; indices range over [0,total) so total itself must fit
    throw std::runtime_error(
      std::string(whatKernel) + ": total element count " + std::to_string(total) +
      " exceeds the 32-bit index limit used by this kernel");
}


__global__
void applyRoPEQKLearnableRecomputeKernel(
  float* qBuf, float* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int qTotalDim, int kTotalDim, int numPairs, int nnXLen
) {
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int totalHP = numHeads * numPairs;
  if(xy >= seqLen || n >= batchSize || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int kvh = h * numKVHeads / numHeads;
  size_t col = (size_t)n * seqLen + xy;

  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal, sinVal;
  KATAGO_GPU_SINCOSF(angle, &sinVal, &cosVal);

  int c0 = h * qHeadDim + 2 * pairIdx;
  size_t qIdx0 = c0 + col * qTotalDim;
  float q0 = qBuf[qIdx0];
  float q1 = qBuf[qIdx0+1];
  qBuf[qIdx0] = q0 * cosVal - q1 * sinVal;
  qBuf[qIdx0+1] = q0 * sinVal + q1 * cosVal;

  if(h == kvh * numHeads / numKVHeads) {
    int kc0 = kvh * qHeadDim + 2 * pairIdx;
    size_t kIdx0 = kc0 + col * kTotalDim;
    float k0 = kBuf[kIdx0];
    float k1 = kBuf[kIdx0+1];
    kBuf[kIdx0] = k0 * cosVal - k1 * sinVal;
    kBuf[kIdx0+1] = k0 * sinVal + k1 * cosVal;
  }
}

__global__
void applyRoPEQKLearnableRecomputeHalfKernel(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int qTotalDim, int kTotalDim, int numPairs, int nnXLen
) {
#ifdef KATAGO_GPU_SUPPORTS_FP16
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int totalHP = numHeads * numPairs;
  if(xy >= seqLen || n >= batchSize || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int kvh = h * numKVHeads / numHeads;
  size_t col = (size_t)n * seqLen + xy;

  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal, sinVal;
  KATAGO_GPU_SINCOSF(angle, &sinVal, &cosVal);

  int c0 = h * qHeadDim + 2 * pairIdx;
  size_t qIdx0 = c0 + col * qTotalDim;
  float q0 = __half2float(qBuf[qIdx0]);
  float q1 = __half2float(qBuf[qIdx0+1]);
  qBuf[qIdx0] = __float2half(q0 * cosVal - q1 * sinVal);
  qBuf[qIdx0+1] = __float2half(q0 * sinVal + q1 * cosVal);

  if(h == kvh * numHeads / numKVHeads) {
    int kc0 = kvh * qHeadDim + 2 * pairIdx;
    size_t kIdx0 = kc0 + col * kTotalDim;
    float k0 = __half2float(kBuf[kIdx0]);
    float k1 = __half2float(kBuf[kIdx0+1]);
    kBuf[kIdx0] = __float2half(k0 * cosVal - k1 * sinVal);
    kBuf[kIdx0+1] = __float2half(k0 * sinVal + k1 * cosVal);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaApplyRoPEQKLearnableRecompute(
  float* qBuf, float* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int qStride, int kvStride, int numPairs, int nnXLen, cudaStream_t stream
) {
  // The fused Q+K kernels only rotate every K head when numHeads is a whole multiple of
  // numKVHeads. Other GQA ratios must use the separate single-buffer launchers.
  if(numHeads % numKVHeads != 0)
    throw std::runtime_error("fused RoPE QK launchers require numHeads divisible by numKVHeads");
  int qTotalDim = qStride;
  int kTotalDim = kvStride;
  int totalHP = numHeads * numPairs;
  if(totalHP > 1024)
    throw std::runtime_error("customCudaApplyRoPEQKLearnableRecompute: numHeads*numPairs (" + std::to_string(totalHP) + ") exceeds the 1024 threads/block limit");
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize, 1);
  applyRoPEQKLearnableRecomputeKernel<<<blocks, threads, 0, stream>>>(
    qBuf, kBuf, freqs, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, qTotalDim, kTotalDim, numPairs, nnXLen
  );
}
void customCudaApplyRoPEQKLearnableRecompute(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int qStride, int kvStride, int numPairs, int nnXLen, cudaStream_t stream
) {
  // The fused Q+K kernels only rotate every K head when numHeads is a whole multiple of
  // numKVHeads. Other GQA ratios must use the separate single-buffer launchers.
  if(numHeads % numKVHeads != 0)
    throw std::runtime_error("fused RoPE QK launchers require numHeads divisible by numKVHeads");
  int qTotalDim = qStride;
  int kTotalDim = kvStride;
  int totalHP = numHeads * numPairs;
  if(totalHP > 1024)
    throw std::runtime_error("customCudaApplyRoPEQKLearnableRecompute: numHeads*numPairs (" + std::to_string(totalHP) + ") exceeds the 1024 threads/block limit");
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize, 1);
  applyRoPEQKLearnableRecomputeHalfKernel<<<blocks, threads, 0, stream>>>(
    qBuf, kBuf, freqs, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, qTotalDim, kTotalDim, numPairs, nnXLen
  );
}

//Wrapper for the CUDA-only tensor-core flash attention (see cudaflashmma.cuh). On builds
//without it (ROCm), always reports unsupported so the caller falls back.
bool customCudaFlashAttentionMma(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  int qStride, int kvStride, cudaStream_t stream
) {
#ifdef KATAGO_HAS_FLASH_MMA
  return launchFlashAttentionMma(
    Q, K, V, mask, output, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim,
    qStride, kvStride, stream);
#else
  (void)Q; (void)K; (void)V; (void)mask; (void)output;
  (void)batchSize; (void)seqLen; (void)numHeads; (void)numKVHeads; (void)qHeadDim; (void)vHeadDim;
  (void)qStride; (void)kvStride; (void)stream;
  return false;
#endif
}

bool customCudaFlashAttentionMmaSupportsShape(int numHeads, int numKVHeads, int qHeadDim, int vHeadDim) {
#ifdef KATAGO_HAS_FLASH_MMA
  return flashAttentionMmaSupportsShape(numHeads, numKVHeads, qHeadDim, vHeadDim);
#else
  (void)numHeads; (void)numKVHeads; (void)qHeadDim; (void)vHeadDim;
  return false;
#endif
}

bool customCudaFlashAttentionMmaSupported() {
#ifdef KATAGO_HAS_FLASH_MMA
  return flashAttentionMmaSupportedOnCurrentDevice();
#else
  return false;
#endif
}

// RMSNorm with gamma/beta/activation (for trunk tip, non-spatial mode).
// NHWC: input/output [n, xy, c], gamma/beta [c], mask [n, xy]
// Each block handles one (n, xy) position.

__global__
void rmsNormGammaBetaNHWCKernel(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  extern __shared__ float rmsShared[];
  int pos = blockIdx.x; // n * xySize + xy
  int tid = threadIdx.x;
  int n = pos / xySize;
  int xy = pos % xySize;
  if(n >= nSize)
    return;

  float maskVal = (mask != NULL) ? mask[n * xySize + xy] : 1.0f;

  const float* inRow = in + (size_t)pos * cSize;

  float acc = 0.0f;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = inRow[c] * maskVal;
    acc += val * val;
  }
  rmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) rmsShared[tid] += rmsShared[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(rmsShared[0] / (float)cSize + epsilon);

  float* outRow = out + (size_t)pos * cSize;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = inRow[c] * maskVal * rms * gamma[c] + beta[c];
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[c] = val;
  }
}

// Lane count of a warp (CUDA) / wavefront (ROCm), for device code. On CUDA this is a literal so
// the reduction unrolls and the warp-id arithmetic stays a shift. On ROCm it has to be the runtime
// warpSize: wavefronts are 64 lanes on CDNA and either width on RDNA, and one binary can target
// both, so there is no compile-time answer.
#ifdef KATAGO_GPU_CUDA
#define KATAGO_WARP_LANES 32
#else
#define KATAGO_WARP_LANES warpSize
#endif

// Sum one value per lane across a warp/wavefront.
__device__ __forceinline__ float warpReduceSumFloat(float acc) {
#ifdef KATAGO_GPU_CUDA
  #pragma unroll
  for(int off = KATAGO_WARP_LANES / 2; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
#else
  for(int off = KATAGO_WARP_LANES / 2; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off);
#endif
  return acc;
}

// Thread-block sizes for the shuffle-reducing kernels below must be a whole number of warps. This
// is the host-side multiple to round to: 64 on ROCm regardless of the actual wavefront width,
// which is a multiple of 32 and so is warp-aligned on wave32 RDNA as well.
#ifdef KATAGO_GPU_CUDA
static constexpr int gpuWarpBlockMultiple = 32;
#else
static constexpr int gpuWarpBlockMultiple = 64;
#endif

// Vectorized half2 path: each thread loads ELTS_PER_THREAD half2 values (= 2*ELTS_PER_THREAD halfs).
// Block size = cSize / (2 * ELTS_PER_THREAD), which the launcher only picks when it is a whole
// number of warps. In-row values are kept in registers across the two passes so the kernel reads
// `in` only once. Per-warp shuffle reduction, then one inter-warp reduction in shared memory.
//
// Requires cSize % (2 * ELTS_PER_THREAD) == 0.

template<int ELTS_PER_THREAD>
__global__
void rmsNormGammaBetaNHWCHalfVecKernel(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
#ifdef KATAGO_GPU_SUPPORTS_FP16
  int pos = blockIdx.x;
  int tid = threadIdx.x;
  int n = pos / xySize;
  int xy = pos % xySize;
  if(n >= nSize)
    return;

  constexpr int VALS_PER_THREAD = 2 * ELTS_PER_THREAD;
  float maskVal = (mask != NULL) ? __half2float(mask[n * xySize + xy]) : 1.0f;

  const half2* inRow2 = reinterpret_cast<const half2*>(in + (size_t)pos * cSize);
  const half2* gamma2 = reinterpret_cast<const half2*>(gamma);
  const half2* beta2  = reinterpret_cast<const half2*>(beta);
  half2* outRow2 = reinterpret_cast<half2*>(out + (size_t)pos * cSize);

  // Stage 1: load all of this thread's values into registers, compute sum of squares.
  float vals[VALS_PER_THREAD];
  float acc = 0.0f;
  #pragma unroll
  for(int e = 0; e < ELTS_PER_THREAD; e++) {
    int idx2 = tid + e * blockDim.x;       // half2-pair index
    half2 v2 = inRow2[idx2];
    float v0 = __half2float(__low2half(v2)) * maskVal;
    float v1 = __half2float(__high2half(v2)) * maskVal;
    vals[2*e]     = v0;
    vals[2*e + 1] = v1;
    acc += v0 * v0 + v1 * v1;
  }

  // Stage 2: warp reduce, then inter-warp reduce via shared memory.
  acc = warpReduceSumFloat(acc);
  __shared__ float warpSums[32];  // blockDim is capped at 512, so at most 16 warps are used.
  int warpId = tid / KATAGO_WARP_LANES;
  int laneId = tid % KATAGO_WARP_LANES;
  if(laneId == 0) warpSums[warpId] = acc;
  __syncthreads();

  // First warp combines per-warp sums. blockDim is at most 512 and a warp is at least 32 lanes,
  // so numWarps is at most 16 and always fits in one warp.
  int numWarps = ((int)blockDim.x + KATAGO_WARP_LANES - 1) / KATAGO_WARP_LANES;
  float total = 0.0f;
  if(tid < numWarps) total = warpSums[tid];
  if(tid < KATAGO_WARP_LANES) {
    total = warpReduceSumFloat(total);
    if(tid == 0) warpSums[0] = total;
  }
  __syncthreads();
  float rms = rsqrtf(warpSums[0] / (float)cSize + epsilon);

  // Stage 3: compute output, reusing `vals` from registers.
  #pragma unroll
  for(int e = 0; e < ELTS_PER_THREAD; e++) {
    int idx2 = tid + e * blockDim.x;
    half2 g2 = gamma2[idx2];
    half2 b2 = beta2[idx2];
    float g0 = __half2float(__low2half(g2));
    float g1 = __half2float(__high2half(g2));
    float b0 = __half2float(__low2half(b2));
    float b1 = __half2float(__high2half(b2));
    float o0 = vals[2*e]     * rms * g0 + b0;
    float o1 = vals[2*e + 1] * rms * g1 + b1;
    if(activation == ACTIVATION_RELU) {
      o0 = fmaxf(o0, 0.0f);
      o1 = fmaxf(o1, 0.0f);
    } else if(activation == ACTIVATION_MISH) {
      o0 = mishf(o0);
      o1 = mishf(o1);
    } else if(activation == ACTIVATION_SILU) {
      o0 = siluf(o0);
      o1 = siluf(o1);
    }
    o0 *= maskVal;
    o1 *= maskVal;
    outRow2[idx2] = __halves2half2(__float2half(o0), __float2half(o1));
  }
#else
  (void)in; (void)out; (void)gamma; (void)beta; (void)mask;
  (void)nSize; (void)xySize; (void)cSize; (void)epsilon; (void)activation;
#endif
}

// Generic (scalar) fallback for shapes the vectorized path can't handle.
__global__
void rmsNormGammaBetaNHWCHalfKernel(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
#ifdef KATAGO_GPU_SUPPORTS_FP16
  extern __shared__ float rmsShared[];
  int pos = blockIdx.x;
  int tid = threadIdx.x;
  int n = pos / xySize;
  int xy = pos % xySize;
  if(n >= nSize)
    return;

  float maskVal = (mask != NULL) ? __half2float(mask[n * xySize + xy]) : 1.0f;

  const half* inRow = in + (size_t)pos * cSize;

  float acc = 0.0f;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = __half2float(inRow[c]) * maskVal;
    acc += val * val;
  }
  rmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) rmsShared[tid] += rmsShared[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(rmsShared[0] / (float)cSize + epsilon);

  half* outRow = out + (size_t)pos * cSize;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = __half2float(inRow[c]) * maskVal * rms * __half2float(gamma[c]) + __half2float(beta[c]);
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[c] = __float2half(val);
  }
#else
  //Do nothing, FP16 not supported
#endif
}


void customCudaRMSNormGammaBetaNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation, cudaStream_t stream
) {
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaRMSNormGammaBetaNHWC");
  int totalPositions = nSize * xySize;
  if(totalPositions <= 0)
    return;
  int threads = 1;
  while(threads < cSize && threads < targetNumThreads) threads *= 2;
  int sharedMem = threads * sizeof(float);
  rmsNormGammaBetaNHWCKernel<<<totalPositions, threads, sharedMem, stream>>>(
    in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
}
void customCudaRMSNormGammaBetaNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation, cudaStream_t stream
) {
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaRMSNormGammaBetaNHWC");
  int totalPositions = nSize * xySize;
  if(totalPositions <= 0)
    return;
  // Vectorized path: cSize must be even (use half2). Pick ELTS_PER_THREAD so we end up with a
  // thread count that is a whole number of warps and <= 512.
  // For cSize=384 -> 192 half2 pairs -> 192 threads, 1 elt/thread.
  // For cSize=768 -> 384 half2 pairs -> 384 threads, 1 elt/thread.
  // For cSize=1024 -> 512 half2 pairs -> 512 threads, 1 elt/thread.
  // Larger cSize -> 2 or more pairs per thread. Note the multi-pair branches engage only for
  // cSize > 1024, wider than any model in the standard test suites. To validate changes to
  // them, temporarily disable the earlier branches so an existing test model takes the
  // multi-pair path (any whole-warp block size is valid) and run testgpuerror.
  if(cSize % 2 == 0) {
    int halfPairs = cSize / 2;
    if(halfPairs <= 512 && halfPairs % gpuWarpBlockMultiple == 0) {
      rmsNormGammaBetaNHWCHalfVecKernel<1><<<totalPositions, halfPairs, 0, stream>>>(
        in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
      return;
    }
    if(halfPairs % (2 * gpuWarpBlockMultiple) == 0 && halfPairs / 2 <= 512) {
      rmsNormGammaBetaNHWCHalfVecKernel<2><<<totalPositions, halfPairs / 2, 0, stream>>>(
        in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
      return;
    }
    if(halfPairs % (4 * gpuWarpBlockMultiple) == 0 && halfPairs / 4 <= 512) {
      rmsNormGammaBetaNHWCHalfVecKernel<4><<<totalPositions, halfPairs / 4, 0, stream>>>(
        in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
      return;
    }
  }
  // Fallback to scalar kernel for shapes the vectorized path cannot cover.
  int threads = 1;
  while(threads < cSize && threads < targetNumThreads) threads *= 2;
  int sharedMem = threads * sizeof(float);
  rmsNormGammaBetaNHWCHalfKernel<<<totalPositions, threads, sharedMem, stream>>>(
    in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
}


//--------------------------------------------------------------------------------------------------------------
// SwiGLU: out[i] = SiLU(a[i]) * b[i]

__global__
void swiGLUKernel(const float* a, const float* b, float* out, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) {
    out[idx] = siluf(a[idx]) * b[idx];
  }
}

// Grid-stride pattern (mirrors OpenCL transformerSwiGLU): each thread handles ELTS_PER_THREAD
// half2 pairs separated by blockDim.x. swiGLU is purely memory-bound, so the win is issuing wide
// (32-bit half2) loads/stores instead of scalar 16-bit ones to better use memory bandwidth.
// Operates on pairCount = size/2 half2 elements; a scalar tail handles an odd final element.
template<int ELTS_PER_THREAD>
__global__
void swiGLUHalfStrideKernel(const half* a, const half* b, half* out, int size)
{
#ifdef KATAGO_GPU_SUPPORTS_FP16
  const half2* a2 = reinterpret_cast<const half2*>(a);
  const half2* b2 = reinterpret_cast<const half2*>(b);
  half2* out2 = reinterpret_cast<half2*>(out);
  int pairCount = size >> 1;
  int tileStart = blockIdx.x * blockDim.x * ELTS_PER_THREAD;
  int lid = threadIdx.x;
  #pragma unroll
  for(int d = 0; d < ELTS_PER_THREAD; d++) {
    int p = tileStart + d * blockDim.x + lid;
    if(p < pairCount) {
      half2 av = a2[p];
      half2 bv = b2[p];
      float a0 = __half2float(__low2half(av));
      float a1 = __half2float(__high2half(av));
      float b0 = __half2float(__low2half(bv));
      float b1 = __half2float(__high2half(bv));
      out2[p] = __halves2half2(__float2half(siluf(a0) * b0), __float2half(siluf(a1) * b1));
    }
  }
  // Tail: if size is odd, the last element isn't covered by any half2 pair. Handle it once.
  if((size & 1) != 0) {
    int last = size - 1;
    if(blockIdx.x == 0 && lid == 0) {
      float av = __half2float(a[last]);
      float bv = __half2float(b[last]);
      out[last] = __float2half(siluf(av) * bv);
    }
  }
#else
  (void)a; (void)b; (void)out; (void)size;
#endif
}

void customCudaSwiGLU(const float* a, const float* b, float* out, int size, cudaStream_t stream) {
  if(size <= 0)
    return;
  int threads = targetNumThreads;
  int blocks = (size + threads - 1) / threads;
  swiGLUKernel<<<blocks, threads, 0, stream>>>(a, b, out, size);
}
void customCudaSwiGLU(const half* a, const half* b, half* out, int size, cudaStream_t stream) {
  if(size <= 0)
    return;
  constexpr int ELTS_PER_THREAD = 4;  // half2 pairs per thread
  int threads = 256;
  int pairCount = size >> 1;
  int blocks = (pairCount + threads * ELTS_PER_THREAD - 1) / (threads * ELTS_PER_THREAD);
  if(blocks < 1) blocks = 1;  // ensure the odd-size tail still gets a block
  swiGLUHalfStrideKernel<ELTS_PER_THREAD><<<blocks, threads, 0, stream>>>(a, b, out, size);
}

//--------------------------------------------------------------------------------------------------------------

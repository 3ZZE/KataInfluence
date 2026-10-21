
#include "../neuralnet/cudahelpers.h"

#include <stdexcept>

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

// RMSNorm per position over channels: out = act(in * rms * gamma + beta) * mask. One block per (n, xy).
template<typename T>
__global__
void rmsNormNHWCKernel(
  const T* in, T* out, const float* gamma, const float* beta, const T* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  extern __shared__ float rmsShared[];
  int pos = blockIdx.x;
  int tid = threadIdx.x;
  if(pos >= nSize * xySize)
    return;
  float maskVal = maskAt(mask, pos);
  const T* inRow = in + (size_t)pos * cSize;

  float acc = 0.0f;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = toFloatT(inRow[c]);
    acc += val * val;
  }
  rmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s)
      rmsShared[tid] += rmsShared[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(rmsShared[0] / (float)cSize + epsilon);

  T* outRow = out + (size_t)pos * cSize;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = applyActivationf(toFloatT(inRow[c]) * rms * gamma[c] + beta[c], activation);
    storeFloatT(outRow, c, val * maskVal);
  }
}

template<typename T>
static void launchRMSNormNHWC(
  const T* in, T* out, const float* gamma, const float* beta, const T* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  int totalPositions = nSize * xySize;
  if(totalPositions <= 0)
    return;
  int threads = 32;
  while(threads < cSize && threads < targetNumThreads)
    threads *= 2;
  rmsNormNHWCKernel<T><<<totalPositions, threads, threads * sizeof(float)>>>(in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
}
void customCudaRMSNormNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  launchRMSNormNHWC<float>(in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
}
void customCudaRMSNormNHWC(
  const half* in, half* out, const float* gamma, const float* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  launchRMSNormNHWC<half>(in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
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
  const T* in, T* out, const float* gamma, const float* beta, const T* mask, const float* maskSum, const float* sumSq,
  int xySize, int cSize, float epsilon, int activation
) {
  int pos = blockIdx.x;
  int n = pos / xySize;
  float maskVal = maskAt(mask, pos);
  float rms = rsqrtf(sumSq[n] / fmaxf(1.0f, maskSum[n] * (float)cSize) + epsilon);
  for(int c = threadIdx.x; c < cSize; c += blockDim.x) {
    size_t idx = (size_t)pos * cSize + c;
    float val = applyActivationf(toFloatT(in[idx]) * rms * gamma[c] + beta[c], activation);
    storeFloatT(out, idx, val * maskVal);
  }
}

template<typename T>
static void launchSpatialRMSNormNHWC(
  const T* in, T* out, const float* gamma, const float* beta, const T* mask, const float* maskSum,
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
  const half* in, half* out, const float* gamma, const float* beta, const half* mask, const float* maskSum,
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

// SwiGLU in place: a = silu(a) * b
template<typename T>
__global__
void swiGLUKernel(T* a, const T* b, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
    storeFloatT(a, idx, siluf(toFloatT(a[idx])) * toFloatT(b[idx]));
}
void customCudaSwiGLU(float* a, const float* b, int size) {
  if(size <= 0)
    return;
  int threads = targetNumThreads;
  swiGLUKernel<float><<<(size + threads - 1) / threads, threads>>>(a, b, size);
}
void customCudaSwiGLU(half* a, const half* b, int size) {
  if(size <= 0)
    return;
  int threads = targetNumThreads;
  swiGLUKernel<half><<<(size + threads - 1) / threads, threads>>>(a, b, size);
}

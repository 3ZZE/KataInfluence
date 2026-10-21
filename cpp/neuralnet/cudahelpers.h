#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

//Given two tensors with shapes inA: [n,cA,h,w] and inB: [n,cB,h,w], that are on the GPU
//Copy them into a single tensor out: [n,cA+cB,h,w] that is also allocated on the gpu
void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

//Given a tensor [n,c,hw], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNCHW(const float* in, float* out, int n, int c, int hw);
void customCudaChannel0ExtractNCHW(const half* in, half* out, int n, int c, int hw);
//Given a tensor [n,hw,c], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNHWC(const float* in, float* out, int n, int hw, int c);
void customCudaChannel0ExtractNHWC(const half* in, half* out, int n, int hw, int c);

//Given an input tensor and an output buffer of shape [n,c], fill output buffer with sum or max over c.
void customCudaPoolRowsSumNCHW(const float* in, float* out, int nSize, int cSize, int xySize, float scaleSum);
void customCudaPoolRowsSumNHWC(const float* in, float* out, int nSize, int xySize, int cSize, float scaleSum);

//Specialized operations for value head and general global pooling. Same as the other pooling, but fusedly fills
//an output buffer of shape [n,c*3].
void customCudaValueHeadPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum);
void customCudaValueHeadPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const half* in, half* out, int nSize, int cSize, int xySize, const half* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const half* in, half* out, int nSize, int xySize, int cSize, const half* mask, const float* maskSum);

void customCudaCopyToHalf(const float* in, half* out, int n);
void customCudaCopyFromHalf(const half* in, float* out, int n);

//Given a tensor, add another tensor to it.
void customCudaAddTensorInplace(half* buf, const half* biases, int n);
//Given an input with shape [n,c] and biases of shape [c], add the biases in-place.
void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int n, int c, int activation);
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int n, int c, int activation);
//Given an input with shape [n,c,xy] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNCHW(float *buf, const float* biases, int nSize, int cSize, int xySize);
void customCudaAddNCBiasInplaceNCHW(half *buf, const half* biases, int nSize, int cSize, int xySize);
//Given an input with shape [n,xy,c] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNHWC(float *buf, const float* biases, int nSize, int xySize, int cSize);
void customCudaAddNCBiasInplaceNHWC(half *buf, const half* biases, int nSize, int xySize, int cSize);

//Given an input with shape [n,c,xy] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply an activation.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNCHW(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int c, int xy, int activation);
void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int c, int xy, int activation);
//Given an input with shape [n,xy,c] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply relu.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNHWC(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int xy, int c, int activation);
void customCudaApplyCScaleBiasNHWC(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int xy, int c, int activation);


//Transformer kernels, NHWC only ([n, xy, c] activations), computing in fp32 whatever the storage type.

//Spatial RMSNorm: one RMS per sample over all channels and on-board positions. sumSqBuf is float scratch [n].
void customCudaSpatialRMSNormNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf);
void customCudaSpatialRMSNormNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf);

//Learnable 2D RoPE in place on [n, xy, numBufHeads*headDim], with freqs (numKVHeads, headDim/2, 2) as float.
//Position xy is x = xy % nnXLen (column), y = xy / nnXLen (row). Fallback for the fused Q+K version below.
void customCudaApplyRoPENHWC(
  float* buf, const float* freqs, int batchSize, int seqLen, int numBufHeads, int numKVHeads, int headDim, int nnXLen);
void customCudaApplyRoPENHWC(
  half* buf, const float* freqs, int batchSize, int seqLen, int numBufHeads, int numKVHeads, int headDim, int nnXLen);

//Attention, Q [n, xy, numHeads*qHeadDim], K [n, xy, numKVHeads*qHeadDim], V [n, xy, numKVHeads*vHeadDim],
//output [n, xy, numHeads*vHeadDim], mask [n, xy]. Off-board keys are skipped, off-board queries give 0.
//Scalar fallback for customCudaFlashAttentionMma below.
void customCudaAttentionNHWC(
  const float* Q, const float* K, const float* V, const float* mask, float* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim);
void customCudaAttentionNHWC(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim);

//From upstream KataGo (cudaandrocmhelpers.inc), see cudahelpers.cu. Pass stream 0.

//RMSNorm per position over channels: out = act(in * rms * gamma + beta) * mask. gamma/beta [c], mask [n, xy] or NULL.
void customCudaRMSNormGammaBetaNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation, cudaStream_t stream);
void customCudaRMSNormGammaBetaNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation, cudaStream_t stream);

//Learnable 2D RoPE in place on Q and K in one launch. qStride/kvStride are the per-token row strides in elements.
//Requires numHeads % numKVHeads == 0.
void customCudaApplyRoPEQKLearnableRecompute(
  float* qBuf, float* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int qStride, int kvStride, int numPairs, int nnXLen, cudaStream_t stream);
void customCudaApplyRoPEQKLearnableRecompute(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int qStride, int kvStride, int numPairs, int nnXLen, cudaStream_t stream);

//Tensor-core flash attention (FP16, head dims 32/64, sm_75+), same layout and semantics as customCudaAttentionNHWC.
//Returns false if the shape, strides or alignment are unsupported, then the caller falls back.
//customCudaFlashAttentionMmaSupported() must be checked once per device first.
bool customCudaFlashAttentionMma(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  int qStride, int kvStride, cudaStream_t stream);
bool customCudaFlashAttentionMmaSupportsShape(int numHeads, int numKVHeads, int qHeadDim, int vHeadDim);
bool customCudaFlashAttentionMmaSupported();

//SwiGLU: out = silu(a) * b, out may alias a
void customCudaSwiGLU(const float* a, const float* b, float* out, int size, cudaStream_t stream);
void customCudaSwiGLU(const half* a, const half* b, half* out, int size, cudaStream_t stream);

#endif  // NEURALNET_CUDAHELPERS_H_

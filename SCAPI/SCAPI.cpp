#include "SCAPI.h"
#include "SJCUDARenderer.h"
SJCUDARenderer* SJCUDARenderer_New()
{
	return new SJCUDARenderer();
}
int SJCUDARenderer_Initialize(SJCUDARenderer* renderer, size_t* pDim, size_t* pMPIDim, float* pC2WCPU, float* pC2WCUDA, float* pW2CCPU, float* pW2CCUDA, float* pCIFCPU, float* pCIFCUDA)
{
	return renderer->Initialize(pDim, pMPIDim, pC2WCPU, pC2WCUDA, pW2CCPU, pW2CCUDA, pCIFCPU, pCIFCUDA);
}
int SJCUDARenderer_LoadMPI(SJCUDARenderer* renderer, unsigned char* pMPICPU, unsigned char* pMPICUDA)
{
	return renderer->LoadMPI(pMPICPU, pMPICUDA);
}
int SJCUDARenderer_LoadCompressedMPI(SJCUDARenderer* renderer, unsigned char* pCompressedMPICPU, unsigned char* pCompressedMPICUDA, unsigned char* pCompressedIdxCPU, unsigned char* pCompressedIdxCUDA)
{
	return renderer->LoadCompressedMPI(pCompressedMPICPU, pCompressedMPICUDA, pCompressedIdxCPU, pCompressedIdxCUDA);
}
int SJCUDARenderer_InitializeRendering(SJCUDARenderer* renderer, size_t outWidth, size_t outHeight, int numBlend)
{
	return renderer->InitializeRendering(outWidth, outHeight, numBlend);
}
int SJCUDARenderer_Rendering(SJCUDARenderer* renderer, float* pose_arr, unsigned char* pOutImageCUDA, unsigned char* pOutImageCPU)
{
	return renderer->Rendering(pose_arr, pOutImageCUDA, pOutImageCPU);
}
int SJCUDARenderer_Finalize(SJCUDARenderer* renderer)
{
	renderer->Finalize();
	delete renderer;
	return 0;
}